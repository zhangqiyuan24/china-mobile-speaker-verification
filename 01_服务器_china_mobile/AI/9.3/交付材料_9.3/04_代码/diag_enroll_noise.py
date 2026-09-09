#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""diag_enroll_noise.py — 注册侧质量退化诊断（鲁棒性 2×2 的空白格）

问题：FiLM 条件与打分注册侧一直假设 clean（官方协议保证、训练分布亦然）。
若注册侧也带噪（部署真实风险），FE 相对 raw 的优势是否保持？

设计（与 build_proxy_val / validate_dual 严格同构）：
  - 固定 trials_hard_{cond}（一切 proxy 报告数字同一 trial 集），测试侧不动，
    只污染 enrol/
  - 注册 SNR ∈ {15, 10, 5, 0} dB，噪声 = held-out noise_test_freesound(168，
    与训练 615 条不相交)，snr_mix 与测试侧生成同实现，定种子可复现
  - 带噪注册嵌入一鱼两吃（部署真实耦合）：
      FiLM 条件  = 该说话人 2 条带噪 enrol 均值+L2（同 spk_embs_from_enrol）
      打分注册侧 = trials 引用的 _001 那条带噪嵌入
  - raw 行 = 带噪注册嵌入 × 带噪测试原始嵌入（无 FE 对照）
  - clean 行自校验：应复现 r5 oracle_hard proxy 值（1.807 / 3.739）

输出: exp/proxy/diag_enroll_noise/summary.json + 退化曲线表
用法: cd 8.27 && python diag_enroll_noise.py --gpu 1
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import toml

_HERE = os.path.dirname(os.path.abspath(__file__))
_AI = os.path.dirname(_HERE)
for _p in (os.path.join(_AI, "8.18"), os.path.join(_AI, "8.20"),
           os.path.join(_AI, "8.21"), os.path.join(_AI, "8.26"), _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from train_tse import build_tse_fullsubnet
from train_spk_preserve import _setup
from trainer_spk_preserve import load_eres2net, snr_mix
from frontend_spk import DiffFBank
from common_utils import setup_speakerlab
from score_proxy import eer_of
from train_tse_v5_asym import (_embed_dir, _enhance_map, _cos,
                               load_hard_trials, PROXY)

SR, TARGET_LEN = 16000, 3 * 16000
DEFAULT_CKPT = os.path.join(_HERE, "exp", "tse_v3_ns05duraug_r5",
                            "checkpoint_best.tar")
NOISE_TEST_LIST = os.path.join("data", "divide", "noise_test_freesound.txt")


def load_np(path):
    import torchaudio
    return torchaudio.load(path)[0][0].numpy()


def save_wav(path, arr):
    import torchaudio
    torchaudio.save(path, torch.from_numpy(np.asarray(arr, np.float32))
                    .unsqueeze(0), SR, encoding="PCM_S", bits_per_sample=16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--base-ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--snrs", default="15,10,5,0")
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()
    snrs = [int(s) for s in args.snrs.split(",") if s.strip()]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda:0"

    cfg = toml.load(os.path.join(_AI, "8.26", "train_tse.toml"))
    sv_ecapa = os.path.abspath(cfg["paths"]["sv_ecapa"])
    fullsubnet_root = os.path.abspath(cfg["paths"]["fullsubnet_root"])
    _setup(fullsubnet_root, sv_ecapa)
    setup_speakerlab(sv_ecapa, _HERE)

    fullsubnet, stft_fn, istft_fn, decompress_cIRM, _ = build_tse_fullsubnet(
        fullsubnet_root, args.base_ckpt, cfg["model"], device)
    fullsubnet.eval()
    verifier = load_eres2net(sv_ecapa, cfg["paths"]["verifier_ckpt"], device)
    verifier.eval()
    diff_fbank = DiffFBank(n_mels=80, sample_rate=16000, n_fft=512,
                           win_length=400, hop_length=160, mean_nor=True,
                           ).to(device)
    n_fft, hop, win = (cfg["acoustics"][k]
                       for k in ("n_fft", "hop_length", "win_length"))

    rng = random.Random(args.seed)
    with open(os.path.join(sv_ecapa, NOISE_TEST_LIST)) as f:
        noise_paths = [ln.strip() for ln in f if ln.strip()]
    noise_paths = [os.path.join(sv_ecapa, p) for p in noise_paths]
    trials = load_hard_trials()
    enrol_files = sorted(os.listdir(os.path.join(PROXY, "enrol")))
    out_root = os.path.join(PROXY, "diag_enroll_noise")
    os.makedirs(out_root, exist_ok=True)
    print(f"[diag] ckpt={os.path.basename(os.path.dirname(args.base_ckpt))}"
          f" noise={len(noise_paths)} enrol={len(enrol_files)} snrs={snrs}",
          flush=True)

    # ---- 带噪测试嵌入（缓存一次，全行共用） ----
    e_raw = {}
    for cond in trials:
        files = [os.path.join(PROXY, cond, f)
                 for f in sorted(os.listdir(os.path.join(PROXY, cond)))]
        e_raw[cond] = _embed_dir(verifier, diff_fbank, files, device)
        print(f"[raw-emb] {cond} {len(e_raw[cond])} 条", flush=True)

    summary = {}
    for row in ["clean"] + [f"snr{s}" for s in snrs]:
        # ---- 注册侧：clean 直接用；带噪行生成 corrupt enrol ----
        if row == "clean":
            enr_dir = os.path.join(PROXY, "enrol")
        else:
            enr_dir = os.path.join(out_root, f"enrol_{row}")
            os.makedirs(enr_dir, exist_ok=True)
            if len(os.listdir(enr_dir)) != len(enrol_files):
                snr_db = int(row[3:])
                for fn in enrol_files:
                    dst = os.path.join(enr_dir, fn)
                    if os.path.exists(dst):
                        continue
                    clean = load_np(os.path.join(PROXY, "enrol", fn))
                    noise = load_np(rng.choice(noise_paths))
                    noisy, _ = snr_mix(clean, noise, snr_db, TARGET_LEN)
                    save_wav(dst, noisy)
                print(f"[corrupt] {row} -> {len(os.listdir(enr_dir))} 条",
                      flush=True)
        files = [os.path.join(enr_dir, f) for f in enrol_files]
        e_enr = _embed_dir(verifier, diff_fbank, files, device)
        # FiLM 条件 = 每人 2 条带噪 enrol 均值 + L2（同 spk_embs_from_enrol）
        grp = {}
        for stem, v in e_enr.items():
            grp.setdefault(stem.split("__")[0], []).append(v)
        cond_map = {k: np.mean(v, axis=0) for k, v in grp.items()}
        cond_map = {k: v / (np.linalg.norm(v) + 1e-8) for k, v in cond_map.items()}

        row_res = {}
        for cond in trials:
            out_dir = os.path.join(out_root, f"val_{row}_{cond}")
            _enhance_map(fullsubnet, stft_fn, istft_fn, decompress_cIRM,
                         n_fft, hop, win, os.path.join(PROXY, cond), out_dir,
                         lambda fn: cond_map[fn.split("__")[0]], device)
            e_enh = _embed_dir(verifier, diff_fbank,
                               [os.path.join(out_dir, f) for f
                                in sorted(os.listdir(out_dir))], device)
            for system, bank in (("fe", e_enh), ("raw", e_raw[cond])):
                scs, labs = [], []
                for e_stem, t_stem, lab in trials[cond]:
                    sc = _cos(e_enr[e_stem], bank[t_stem])
                    scs.append(sc)
                    labs.append(1 if lab == "target" else 0)
                row_res[f"{cond}_{system}"] = eer_of(
                    np.asarray(scs), np.asarray(labs))
            print(f"  [{row}] {cond} fe={row_res[f'{cond}_fe']:.3f} "
                  f"raw={row_res[f'{cond}_raw']:.3f}", flush=True)
        summary[row] = row_res
        with open(os.path.join(out_root, "summary.json"), "w") as f:
            json.dump({"ckpt": args.base_ckpt, "seed": args.seed,
                       "rows": summary}, f, indent=2, ensure_ascii=False)

    print("\n==== 注册侧退化曲线（EER，越低越好） ====", flush=True)
    print(f"{'enroll':>8} | {'0dB_raw':>7} {'0dB_FE':>7} {'gap':>6} | "
          f"{'-5dB_raw':>8} {'-5dB_FE':>7} {'gap':>6}")
    for row, r in summary.items():
        print(f"{row:>8} | {r['noise_0db_raw']:7.3f} {r['noise_0db_fe']:7.3f} "
              f"{r['noise_0db_raw'] - r['noise_0db_fe']:+6.3f} | "
              f"{r['noise_m5db_raw']:8.3f} {r['noise_m5db_fe']:7.3f} "
              f"{r['noise_m5db_raw'] - r['noise_m5db_fe']:+6.3f}", flush=True)
    print("[自校验] clean 行 FE 应≈r5 oracle_hard 1.807/3.739（proxy 口径）",
          flush=True)


if __name__ == "__main__":
    main()
