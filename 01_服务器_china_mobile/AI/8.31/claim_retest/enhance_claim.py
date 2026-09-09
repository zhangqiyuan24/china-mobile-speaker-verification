#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""enhance_claim.py — #37 claim 口径增强（8.31）
对抽样的 nontarget 对，按注册侧（声称者）的 clean 质心条件化增强测试句：
    {out_root}/{cond}/{claim_spk}/{test_stem}.wav
target 侧不增强（claim=真身，与交付 oracle 增强逐位一致，直接复用既有嵌入）。
断点续跑：enhance_cond 对已存在 wav 自动跳过。末尾摊平硬链接供统一提嵌入。
"""
import argparse
import json
import os
import sys

import torch
import toml

_HERE = os.path.dirname(os.path.abspath(__file__))
_AI = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_AI, "8.18"), os.path.join(_AI, "8.20"),
           os.path.join(_AI, "8.21"), os.path.join(_AI, "8.26"),
           os.path.join(_AI, "8.27")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from train_tse import build_tse_fullsubnet          # noqa: E402
from train_spk_preserve import _setup               # noqa: E402
from trainer_spk_preserve import load_eres2net      # noqa: E402
from frontend_spk import DiffFBank                  # noqa: E402
from common_utils import setup_speakerlab           # noqa: E402
from evaluate_robust import get_wav_list            # noqa: E402
from enhance_tse import extract_enrol_embs, enhance_cond  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train_tse.toml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--groups", default=os.path.join(_HERE, "claim_groups.json"))
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--limit-groups", type=int, default=None)
    ap.add_argument("--limit-per", type=int, default=None)
    args = ap.parse_args()

    cfg = toml.load(os.path.join(_AI, "8.26", args.config))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    sv_ecapa = os.path.abspath(cfg["paths"]["sv_ecapa"])
    fullsubnet_root = os.path.abspath(cfg["paths"]["fullsubnet_root"])
    _setup(fullsubnet_root, sv_ecapa)
    setup_speakerlab(sv_ecapa, _HERE)
    test_root = os.path.join(sv_ecapa, "data", "data_for_test")
    clean_wavs = get_wav_list(os.path.join(test_root, "clean", "3s"), sv_ecapa)
    wav_by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in clean_wavs}

    groups = json.load(open(args.groups))
    keys = sorted(groups)
    if args.limit_groups:
        keys = keys[:args.limit_groups]
    if args.limit_per:
        groups = {k: v[:args.limit_per] for k, v in groups.items()}

    verifier = load_eres2net(sv_ecapa, cfg["paths"]["verifier_ckpt"], device)
    verifier.eval()
    fbank = DiffFBank(n_mels=80, sample_rate=16000, n_fft=512,
                      win_length=400, hop_length=160, mean_nor=True).to(device)
    emb_by_spk = extract_enrol_embs(verifier, fbank, wav_by_stem, sorted(groups),
                                    device)
    print(f"[cond] {len(emb_by_spk)} 个声称者注册质心(512, L2)", flush=True)

    model, stft_fn, istft_fn, decompress_cIRM, _b = build_tse_fullsubnet(
        fullsubnet_root, cfg["paths"]["fullsubnet_ckpt"], cfg["model"], device)
    ck = torch.load(args.checkpoint, map_location="cpu")
    sd = ck["model"] if "model" in ck else ck
    model.load_state_dict(sd)
    model.eval()
    print(f"[tse] 装载 {args.checkpoint}(epoch={ck.get('epoch', '?')})", flush=True)
    n_fft, hop, win = (cfg["acoustics"][k]
                       for k in ("n_fft", "hop_length", "win_length"))

    for cond in ("noise_0db", "noise_m5db"):
        wl = get_wav_list(os.path.join(test_root, cond, "3s"), sv_ecapa)
        path_by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in wl}
        for gi, spk in enumerate(keys):
            stems = [s for s in groups[spk] if s in path_by_stem]
            miss = [s for s in groups[spk] if s not in path_by_stem]
            if miss:
                print(f"[warn] {spk}: {len(miss)} 条不在 3s 测试目录", flush=True)
            if not stems:
                continue
            mapping = {s: spk for s in stems}          # 全部强制按声称者条件化
            wavs = [path_by_stem[s] for s in stems]
            enhance_cond(model, emb_by_spk, mapping, wavs,
                         os.path.join(args.out_root, cond, spk), device,
                         n_fft, hop, win, args.batch_size, stft_fn, istft_fn,
                         decompress_cIRM)
            print(f"[{cond}] {gi + 1}/{len(keys)} {spk} done", flush=True)

    # ---- 摊平硬链接 {claim}__{stem}.wav（统一提嵌入，键不冲突） ----
    for cond in ("noise_0db", "noise_m5db"):
        flat = os.path.join(args.out_root, "flat_" + cond)
        os.makedirs(flat, exist_ok=True)
        n = 0
        for spk in keys:
            d = os.path.join(args.out_root, cond, spk)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if fn.endswith(".wav"):
                    dst = os.path.join(flat, f"{spk}__{fn}")
                    if not os.path.isfile(dst):
                        os.link(os.path.join(d, fn), dst)
                        n += 1
        print(f"[flat] {cond}: {n} 新链", flush=True)
    print("[done] claim 增强 + 摊平完成", flush=True)


if __name__ == "__main__":
    main()
