#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fusion_claim_proxy.py — claim 侧 raw+enh 分数融合：proxy α 搜索（#38 补牌）

口径与 train_tse_v5_asym.validate_dual 逐位同构（同 proxy 集 / 同错位冒认条件 /
同 verifier），仅新增两件事：① per-trial 分数落盘 ② raw 分支 + α 网格。
铁律：α 只在本 proxy 上搜；官方侧仅在 v6 判定后用冻结 α 一次性应用（Phase 0 模式）。

claim trial 组成（同 validate_dual）：
  target    = hard trials 的 target 对，enh 分支用 oracle 条件增强
  nontarget = 全部 proxy 测试句，enh 分支用错位说话人（排序后移一位）条件增强，
              对该条件者的注册嵌入打分
raw 分支 = 同 trial 组成，测试句不增强。

用法: python fusion_claim_proxy.py --ckpt CKPT --tag asym30 [--cpu] [--gpu 0]
输出: $PROXY/fusion_claim/{tag}.json（含 α* 与网格）
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import toml

_HERE = os.path.dirname(os.path.abspath(__file__))
_AI = os.path.dirname(_HERE)
for _p in (os.path.join(_AI, "8.27"), os.path.join(_AI, "8.18"),
           os.path.join(_AI, "8.20"), os.path.join(_AI, "8.21"),
           os.path.join(_AI, "8.26"), _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from train_tse import build_tse_fullsubnet                    # noqa: E402
from train_spk_preserve import _setup                         # noqa: E402
from trainer_spk_preserve import load_eres2net                # noqa: E402
from frontend_spk import DiffFBank                            # noqa: E402
from common_utils import setup_speakerlab                     # noqa: E402
from enhance_proxy import spk_embs_from_enrol                 # noqa: E402
from score_proxy import eer_of                                # noqa: E402
from train_tse_v5_asym import (                               # noqa: E402
    load_hard_trials, _embed_dir, _enhance_map, _cos, PROXY)


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--alpha-step", type=float, default=0.05)
    args = ap.parse_args()

    if not args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cpu" if args.cpu else "cuda:0"

    cfg = toml.load(os.path.join(_AI, "8.26", "train_tse.toml"))
    sv_ecapa = os.path.abspath(cfg["paths"]["sv_ecapa"])
    fullsubnet_root = os.path.abspath(cfg["paths"]["fullsubnet_root"])
    _setup(fullsubnet_root, sv_ecapa)
    setup_speakerlab(sv_ecapa, os.path.join(_AI, "8.27"))
    sr = cfg["acoustics"]["sr"]

    fullsubnet, stft_fn, istft_fn, decompress_cIRM, _ = build_tse_fullsubnet(
        fullsubnet_root, cfg["paths"]["fullsubnet_ckpt"], cfg["model"], device)
    ck = torch.load(args.ckpt, map_location=device)
    fullsubnet.load_state_dict(ck["model"])
    fullsubnet.eval()
    print(f"[ckpt] {args.ckpt} (epoch={ck.get('epoch', '?')})", flush=True)

    verifier = load_eres2net(sv_ecapa, cfg["paths"]["verifier_ckpt"], device)
    verifier.eval()
    diff_fbank = DiffFBank(n_mels=80, sample_rate=sr, n_fft=512, win_length=400,
                           hop_length=160, mean_nor=True).to(device)
    emb_by_spk = spk_embs_from_enrol(device)
    trials = load_hard_trials()
    spks = sorted(emb_by_spk)
    wrong = {s: spks[(i + 1) % len(spks)] for i, s in enumerate(spks)}
    n_fft, hop, win = (cfg["acoustics"][k]
                       for k in ("n_fft", "hop_length", "win_length"))

    out_root = os.path.join(PROXY, "fusion_claim")
    os.makedirs(out_root, exist_ok=True)
    report = {"tag": args.tag, "ckpt": args.ckpt, "cond": {}}

    for cond in ("noise_0db", "noise_m5db"):
        raw_dir = os.path.join(PROXY, cond)
        # ---- raw 分支嵌入（跨臂缓存，只算一次） ----
        cache = os.path.join(out_root, f"emb_raw_{cond}.npz")
        if os.path.exists(cache):
            e_raw = dict(np.load(cache, allow_pickle=True)["e"].item())
            print(f"  [cache] raw {cond}: {len(e_raw)} 句", flush=True)
        else:
            files = [os.path.join(raw_dir, f)
                     for f in sorted(os.listdir(raw_dir))]
            e_raw = _embed_dir(verifier, diff_fbank, files, device)
            np.savez(cache, e=np.array(e_raw, dtype=object))
        enrol_files = [os.path.join(PROXY, "enrol", f)
                       for f in sorted(os.listdir(os.path.join(PROXY, "enrol")))]
        e_enrol = _embed_dir(verifier, diff_fbank, enrol_files, device)

        # ---- enh 两路（oracle 条件 / 冒认条件） ----
        out_or = os.path.join(PROXY, f"fc_{args.tag}_or", cond)
        out_cl = os.path.join(PROXY, f"fc_{args.tag}_cl", cond)
        _enhance_map(fullsubnet, stft_fn, istft_fn, decompress_cIRM,
                     n_fft, hop, win, raw_dir, out_or,
                     lambda fn: emb_by_spk[fn.split("__")[0]], device)
        _enhance_map(fullsubnet, stft_fn, istft_fn, decompress_cIRM,
                     n_fft, hop, win, raw_dir, out_cl,
                     lambda fn: emb_by_spk[wrong[fn.split("__")[0]]], device)
        e_or = _embed_dir(verifier, diff_fbank,
                          [os.path.join(out_or, f)
                           for f in sorted(os.listdir(out_or))], device)
        e_cl = _embed_dir(verifier, diff_fbank,
                          [os.path.join(out_cl, f)
                           for f in sorted(os.listdir(out_cl))], device)

        # ---- 组 trial（同 validate_dual） ----
        tgt_pairs = [(a, b) for a, b, lab in trials[cond] if lab == "target"]
        sc_enh_t = np.asarray([_cos(e_enrol[a], e_or[b]) for a, b in tgt_pairs])
        sc_raw_t = np.asarray([_cos(e_enrol[a], e_raw[b]) for a, b in tgt_pairs])
        sc_enh_n = np.asarray(
            [_cos(emb_by_spk[wrong[b.split("__")[0]]], e_cl[b]) for b in e_cl])
        sc_raw_n = np.asarray(
            [_cos(emb_by_spk[wrong[b.split("__")[0]]], e_raw[b]) for b in e_cl])
        s_enh = np.concatenate([sc_enh_t, sc_enh_n])
        s_raw = np.concatenate([sc_raw_t, sc_raw_n])
        labs = np.concatenate([np.ones(len(sc_enh_t)),
                               np.zeros(len(sc_enh_n))])

        ze, zr = _z(s_enh), _z(s_raw)
        n_grid = int(round(1.0 / args.alpha_step))
        grid = [round(i / n_grid, 3) for i in range(n_grid + 1)]
        rows = [(a, float(eer_of(a * ze + (1 - a) * zr, labs))) for a in grid]
        a_star, e_star = min(rows, key=lambda r: r[1])

        entry = {
            "eer_enh": float(eer_of(s_enh, labs)),
            "eer_raw": float(eer_of(s_raw, labs)),
            "eer_fused_a03": float(eer_of(0.3 * ze + 0.7 * zr, labs)),
            "eer_fused_a05": float(eer_of(0.5 * ze + 0.5 * zr, labs)),
            "eer_fused_a07": float(eer_of(0.7 * ze + 0.3 * zr, labs)),
            "alpha_star": a_star, "eer_fused": e_star,
            "grid": rows,
            "n_target": int(len(sc_enh_t)),
            "n_nontarget": int(len(sc_enh_n)),
        }
        report["cond"][cond] = entry
        print(f"  [{cond}] enh={entry['eer_enh']:.3f} raw={entry['eer_raw']:.3f}"
              f" | fused α*= {a_star:.2f} → {e_star:.3f}"
              f" (a03/05/07 = {entry['eer_fused_a03']:.3f}/"
              f"{entry['eer_fused_a05']:.3f}/{entry['eer_fused_a07']:.3f})"
              f" n={entry['n_target']}T/{entry['n_nontarget']}N", flush=True)

    with open(os.path.join(out_root, f"{args.tag}.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[saved] {out_root}/{args.tag}.json", flush=True)


if __name__ == "__main__":
    main()
