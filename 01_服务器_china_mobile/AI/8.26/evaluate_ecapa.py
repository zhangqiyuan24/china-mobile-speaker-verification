#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""evaluate_ecapa.py — ECAPA-TDNN 官方协议评测（8.26 线②）

镜像 8.21 evaluate_combo 的双模式评测，模型换成微调后的 ECAPA(emb 192)：
  raw       : 原始带噪 wav → ECAPA
  frontend  : 增强后 wav(--enh-0db/--enh-m5db) → ECAPA
协议完全复用 evaluate_combo.run_mode（enrol=官方 clean/3s 同 ckpt、
trials_cross_device、原版 compute_score_metrics.py、ckpt 守卫、断点续跑），
缓存按 exp_dir 隔离，不会污染 ERes2Net 线的 combo 缓存。

基线（ECAPA 自己）：clean 9.31 / 0dB 11.150 / -5dB 14.360
达标线（-30%）：0dB ≤ 7.805 / -5dB ≤ 10.052

用法:
  python evaluate_ecapa.py --config config_ecapa_robust.toml --gpu 1 \
      --checkpoint exp/ecapa_robust_90k/checkpoints/checkpoint_best.tar \
      [--enh-0db .../enhanced/noise_0db --enh-m5db .../enhanced/noise_m5db]
"""
import argparse
import os
import sys

import torch
import toml

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(os.path.dirname(_HERE), "8.20"),
           os.path.join(os.path.dirname(_HERE), "8.21"),
           _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from common_utils import setup_speakerlab
from evaluate_robust import get_wav_list
from evaluate_combo import run_mode, extract_guarded, index_by_stem


def load_ecapa(ckpt_path, device):
    from speakerlab.models.ecapa_tdnn.ECAPA_TDNN import ECAPA_TDNN
    model = ECAPA_TDNN(input_size=80, lin_neurons=192, channels=[1024, 1024, 1024, 1024, 3072])
    ck = torch.load(ckpt_path, map_location="cpu")
    if "embedding_model" in ck:          # train_ecapa_robust 产物
        sd = ck["embedding_model"]
    elif "state_dict" in ck:             # speakerlab pretrained 原格式
        sd = ck["state_dict"]
    else:
        sd = ck
    model.load_state_dict(sd)
    model.to(device).eval()
    print(f"[ecapa] 装载 {ckpt_path}(epoch={ck.get('epoch', 'pretrained')})", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_ecapa_robust.toml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--enh-0db", default=None, help="noise_0db 增强后 wav 目录")
    ap.add_argument("--enh-m5db", default=None, help="noise_m5db 增强后 wav 目录")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cfg = toml.load(os.path.join(_HERE, args.config))
    base = _HERE
    p_cfg = cfg["paths"]
    sv_ecapa = p_cfg["sv_ecapa"]
    speakerlab_root = os.path.dirname(os.path.dirname(os.path.dirname(sv_ecapa)))
    setup_speakerlab(sv_ecapa, base)

    exp_dir = os.path.join(base, p_cfg["exp_dir"])
    ckpt = args.checkpoint or os.path.join(exp_dir, "checkpoints", "checkpoint_best.tar")
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(base, ckpt)
    assert os.path.isfile(ckpt), f"checkpoint 不存在: {ckpt}"

    model = load_ecapa(ckpt, device)

    test_root = os.path.join(sv_ecapa, "data", "data_for_test")
    trials_path = os.path.join(test_root, "clean", "3s", "trials_cross_device")
    out_root = os.path.join(exp_dir, "eval", "combo")
    os.makedirs(out_root, exist_ok=True)

    # ---- enrol 侧：官方 clean/3s，同一 ECAPA ckpt ----
    clean_wavs = get_wav_list(os.path.join(test_root, "clean", "3s"), sv_ecapa)
    enrol_dir = os.path.join(out_root, "common", "enrol_clean")
    extract_guarded(model, clean_wavs, enrol_dir, device, args, ckpt,
                    desc="enrol-clean(ecapa)")

    modes = ["raw"]
    enh_index = {}
    if args.enh_0db and args.enh_m5db:
        enh_index["noise_0db"] = index_by_stem(
            args.enh_0db if os.path.isabs(args.enh_0db) else os.path.join(base, args.enh_0db))
        enh_index["noise_m5db"] = index_by_stem(
            args.enh_m5db if os.path.isabs(args.enh_m5db) else os.path.join(base, args.enh_m5db))
        modes.append("frontend")
    else:
        print("[combo] 未提供 --enh-0db/--enh-m5db，只跑 raw 模式")

    all_results = {}
    for mode in modes:
        print(f"\n===== 模式 {mode.upper()}："
              + ("原始带噪 → ECAPA" if mode == "raw" else "增强 → ECAPA") + " =====")
        all_results[mode] = run_mode(mode, model, cfg, base, device, args, ckpt,
                                     sv_ecapa, speakerlab_root, enrol_dir,
                                     trials_path, enh_index, out_root)

    # ---- 汇总（vs ECAPA 自己的基线，-30% 达标判定） ----
    bl = cfg.get("baseline", {})
    print("\n===== ECAPA 组合评测汇总（trials_cross_device, 3s, 官方打分） =====")
    print(f"{'方案':<24}{'clean EER%':>12}{'0dB EER%':>10}{'-5dB EER%':>10}"
          f"{'0dB 降幅':>10}{'-5dB 降幅':>10}")
    t0, t5 = 0.7 * bl["noise_0db"], 0.7 * bl["noise_m5db"]

    def _eer(v):
        return v["eer"] if isinstance(v, dict) else v

    rows = [("ecapa baseline", bl)] + \
           [(f"ecapa-{m}", all_results[m]) for m in modes if m in all_results]
    for name, res in rows:
        rc = _eer(res.get("clean")) if res.get("clean") is not None else None
        r0, r5 = _eer(res.get("noise_0db")), _eer(res.get("noise_m5db"))
        d0 = f"{100.0*(bl['noise_0db']-r0)/bl['noise_0db']:>9.1f}%" if r0 else " " * 10
        d5 = f"{100.0*(bl['noise_m5db']-r5)/bl['noise_m5db']:>9.1f}%" if r5 else " " * 10
        cs = f"{rc:>12.4f}" if rc is not None else " " * 12
        print(f"{name:<24}{cs}{r0:>10.4f}{r5:>10.4f}{d0:>10}{d5:>10}")

    for mode in modes:
        if mode not in all_results:
            continue
        r0 = _eer(all_results[mode].get("noise_0db"))
        r5 = _eer(all_results[mode].get("noise_m5db"))
        ok0 = "✓" if r0 <= t0 else "✗"
        ok5 = "✓" if r5 <= t5 else "✗"
        print(f"  {mode}: 0dB {r0:.4f} {ok0} (目标≤{t0:.3f}) | "
              f"-5dB {r5:.4f} {ok5} (目标≤{t5:.3f})")


if __name__ == "__main__":
    main()
