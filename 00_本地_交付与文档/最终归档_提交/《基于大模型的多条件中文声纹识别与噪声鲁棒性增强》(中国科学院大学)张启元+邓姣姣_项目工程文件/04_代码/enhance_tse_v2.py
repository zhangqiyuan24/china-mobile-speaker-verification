#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""enhance_tse_v2.py — TSE 前端推理：支持 1s/3s/5s 测试时长（#19，8.28）

相对 8.26/enhance_tse.py 的唯一差异：
  - --dur {1,3,5}：测试 wav 目录 data_for_test/{cond}/{dur}s
  - 质心表保持 clean/3s 注册（说话人前缀为键，_1s/_5s 测试 stem 天然兼容；
    注册中心 3s 裁剪与训练条件分布一致，pad/裁剪不引入协议外信息）
其余逻辑（断点续跑、长度分组批推理、cIRM 解码）与原版逐行相同。
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torchaudio
import toml

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(os.path.dirname(_HERE), "8.18"),
           os.path.join(os.path.dirname(_HERE), "8.20"),
           os.path.join(os.path.dirname(_HERE), "8.21"),
           os.path.join(os.path.dirname(_HERE), "8.26"),
           _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from train_tse import build_tse_fullsubnet          # noqa: E402
from train_spk_preserve import _setup               # noqa: E402
from trainer_spk_preserve import load_eres2net      # noqa: E402
from frontend_spk import DiffFBank                  # noqa: E402
from common_utils import setup_speakerlab           # noqa: E402
from evaluate_robust import get_wav_list            # noqa: E402
from enhance_tse import (speaker_of, extract_enrol_embs,   # noqa: E402
                         enhance_cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train_tse.toml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dur", type=int, default=3, choices=(1, 3, 5),
                    help="测试 wav 时长目录（1s/3s/5s），质心表固定 clean/3s")
    ap.add_argument("--conds", default="noise_0db,noise_m5db",
                    help="逗号分隔条件目录名（#21 加 reverb/overlap* 场景）")
    ap.add_argument("--test-root", default=None,
                    help="测试数据根覆盖（#26 换域：AI/8.28/noise_domains；默认官方 data_for_test）")
    args = ap.parse_args()
    conds = tuple(args.conds.split(","))

    cfg = toml.load(os.path.join(os.path.dirname(_HERE), "8.26", args.config))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    sv_ecapa = os.path.abspath(cfg["paths"]["sv_ecapa"])
    fullsubnet_root = os.path.abspath(cfg["paths"]["fullsubnet_root"])
    _setup(fullsubnet_root, sv_ecapa)
    setup_speakerlab(sv_ecapa, _HERE)

    test_root = args.test_root or os.path.join(sv_ecapa, "data", "data_for_test")
    clean_root = os.path.join(sv_ecapa, "data", "data_for_test")  # 注册 clean 固定官方根（#26 修复）
    clean_wavs = get_wav_list(os.path.join(clean_root, "clean", "3s"), sv_ecapa)
    wav_by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in clean_wavs}
    print(f"[enrol] clean(3s) wav {len(wav_by_stem)}", flush=True)

    verifier = load_eres2net(sv_ecapa, cfg["paths"]["verifier_ckpt"], device)
    verifier.eval()
    fbank = DiffFBank(n_mels=80, sample_rate=16000, n_fft=512,
                      win_length=400, hop_length=160, mean_nor=True).to(device)

    all_test = []
    for cond in conds:
        wl = get_wav_list(os.path.join(test_root, cond, f"{args.dur}s"), sv_ecapa)
        if args.limit:
            wl = wl[:args.limit]
        all_test += [os.path.splitext(os.path.basename(p))[0] for p in wl]
    speakers = sorted({speaker_of(s) for s in all_test})
    emb_by_spk = extract_enrol_embs(verifier, fbank, wav_by_stem, speakers,
                                    device)
    print(f"[cond] {len(emb_by_spk)} 个说话人注册嵌入(512, L2)", flush=True)

    model, stft_fn, istft_fn, decompress_cIRM, _b = build_tse_fullsubnet(
        fullsubnet_root, cfg["paths"]["fullsubnet_ckpt"], cfg["model"], device)
    ck = torch.load(args.checkpoint, map_location="cpu")
    sd = ck["model"] if "model" in ck else ck
    model.load_state_dict(sd)
    model.eval()
    print(f"[tse] 装载 {args.checkpoint}(epoch={ck.get('epoch', '?')})", flush=True)

    n_fft, hop, win = (cfg["acoustics"][k] for k in ("n_fft", "hop_length", "win_length"))
    out_root = args.out_root or os.path.join(cfg["paths"]["exp_dir"], "enhanced")

    for cond in conds:
        wavs = get_wav_list(os.path.join(test_root, cond, f"{args.dur}s"), sv_ecapa)
        if args.limit:
            wavs = wavs[:args.limit]
        stems = [os.path.splitext(os.path.basename(p))[0] for p in wavs]
        test2spk = {s: speaker_of(s) for s in stems}
        miss = [s for s in stems if test2spk[s] not in emb_by_spk]
        assert not miss, f"{cond} 有 {len(miss)} 条测试的说话人无注册嵌入"
        print(f"[{cond}/{args.dur}s] {len(wavs)} 条带噪待增强", flush=True)
        enhance_cond(model, emb_by_spk, test2spk, wavs,
                     os.path.join(out_root, cond), device,
                     n_fft, hop, win, args.batch_size, stft_fn, istft_fn,
                     decompress_cIRM)
    print(f"[done] 增强产物 -> {out_root}/{' '.join(conds)}", flush=True)


if __name__ == "__main__":
    main()
