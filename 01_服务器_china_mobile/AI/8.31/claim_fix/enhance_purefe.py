#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""#43 无条件 FullSubNet+（spk_preserve e10，主线"通用前端/旧方法降噪"同源 ckpt）
场景集增强：overlap2/3 + reverb 五场景。

对照 r5 场景表（scenes_r5，注册条件化 TSE）——验证主张：
通用前端对重叠人声无判别信息 → EER 应≈raw → TSE 的不可替代性铁证。
增强逻辑逐行照抄 8.18/validate_spk_preserve.py::load_enhancer（无条件前向，
不喂声纹条件）。输出保留原文件名，供 infer_sv_batch + cross_device trials 复用。
"""
import os
import sys
import argparse
import torch
import torchaudio
import toml

AI = "/data/home/ligangrui/china_mobile/AI"
sys.path.insert(0, f"{AI}/8.18")
sys.path.insert(0, f"{AI}/8.17/code")  # FullSubNet+ 代码带 speech_enhance. 前缀 import
sys.path.insert(0, f"{AI}/8.17/code/speech_enhance")  # trainer 用无前缀 audio_zen import

from trainer_spk_preserve import load_fullsubnet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=f"{AI}/8.18/train_spk_preserve.toml")
    ap.add_argument("--ckpt",
                    default=f"{AI}/8.18/exp/spk_preserve/checkpoint_epoch10.tar")
    ap.add_argument("--wav-list", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    cfg = toml.load(args.config)
    device = torch.device(f"cuda:{args.gpu}")
    fs_root = cfg.get("paths", {}).get("fullsubnet_root", f"{AI}/8.17/code")
    model, stft_fn, istft_fn, decompress, _ = load_fullsubnet(
        fs_root, args.ckpt, cfg["model"], device)
    model.eval()
    n_fft = cfg["acoustics"]["n_fft"]
    hop = cfg["acoustics"]["hop_length"]
    win = cfg["acoustics"]["win_length"]

    with open(args.wav_list) as f:
        wavs = [l.strip().split()[-1] for l in f if l.strip()]
    print(f"[purefe] {len(wavs)} wavs, ckpt={os.path.basename(args.ckpt)} "
          f"-> {args.out_root}", flush=True)

    @torch.no_grad()
    def enhance(batch):  # (B, T)
        c = stft_fn(batch, n_fft, hop, win)
        cRM = model(c.abs().unsqueeze(1), c.real.unsqueeze(1), c.imag.unsqueeze(1))
        cRM = decompress(cRM.permute(0, 2, 3, 1))
        er = cRM[..., 0] * c.real - cRM[..., 1] * c.imag
        ei = cRM[..., 1] * c.real + cRM[..., 0] * c.imag
        return istft_fn(torch.stack([er, ei], -1), n_fft, hop, win,
                        length=batch.size(-1))

    n = 0
    for i in range(0, len(wavs), args.batch_size):
        chunk = wavs[i:i + args.batch_size]
        sigs, srs = [], []
        for p in chunk:
            x, sr = torchaudio.load(p)
            sigs.append(x[0])
            srs.append(sr)
        lens = [x.numel() for x in sigs]
        T = max(lens)
        batch = torch.zeros(len(chunk), T)
        for j, x in enumerate(sigs):
            batch[j, :x.numel()] = x
        out = enhance(batch.to(device)).cpu()
        for j, p in enumerate(chunk):
            dst = os.path.join(args.out_root, os.path.basename(p))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            torchaudio.save(dst, out[j:j + 1, :lens[j]], srs[j])
            n += 1
        if (i // args.batch_size) % 100 == 0:
            print(f"  [{n}/{len(wavs)}]", flush=True)
    print(f"[done] {n} wavs -> {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
