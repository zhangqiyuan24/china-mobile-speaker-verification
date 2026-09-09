#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立验证：对比 baseline（普通 FullSubNet+）与微调后（speaker-preserving）的 enhanced embedding。

重点：enhanced embedding 是否比普通 FullSubNet+ 更接近 clean。

第一版做 mean cosine 验证（0dB / -5dB）：
  - 对同批 clean + 同批 noise，分别混 0dB / -5dB
  - baseline checkpoint 和微调 checkpoint 分别降噪
  - 比较 cos(clean, enhanced) 的提升

完整的 EER 计算（clean enrollment vs enhanced test 打分）在后续接入 sv-ecapa 评测框架。
"""

import os
import sys
import random
import argparse
import numpy as np
import toml
import torch
import torchaudio

from trainer_spk_preserve import (
    load_fullsubnet, load_eres2net, snr_mix, _setup_paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train_spk_preserve.toml")
    parser.add_argument("--baseline_ckpt", default="")
    parser.add_argument("--finetuned_ckpt", default="")
    parser.add_argument("--n_samples", type=int, default=100)
    args = parser.parse_args()

    config = toml.load(args.config)
    base = os.path.dirname(os.path.abspath(args.config))
    sv_ecapa = os.path.abspath(os.path.join(base, config["paths"]["sv_ecapa"]))
    fullsubnet_root = os.path.abspath(os.path.join(base, config["paths"]["fullsubnet_root"]))
    _setup_paths(fullsubnet_root, sv_ecapa)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from frontend_spk import DiffFBank

    device = torch.device(config["training"]["device"])
    sr = config["acoustics"]["sr"]
    n_fft = config["acoustics"]["n_fft"]
    win_length = config["acoustics"]["win_length"]
    hop_length = config["acoustics"]["hop_length"]
    target_len = int(config["dataset"]["sub_sample_length"] * sr)

    baseline_ckpt = args.baseline_ckpt or config["paths"]["fullsubnet_ckpt"]
    finetuned_ckpt = args.finetuned_ckpt

    # 数据
    def load_paths(path, is_scp=False):
        out = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = line.split()[-1] if is_scp else line
                if not os.path.isabs(p):
                    p = os.path.join(sv_ecapa, p)
                out.append(p)
        return out

    clean_paths = load_paths(config["dataset"]["clean_scp"], is_scp=True)
    noise_paths = load_paths(config["dataset"]["noise_list"])
    random.shuffle(clean_paths)

    # 模型
    eres2net = load_eres2net(sv_ecapa, config["paths"]["eres2net_ckpt"], device)
    diff_fbank = DiffFBank(n_mels=80, sample_rate=sr, n_fft=512,
                           win_length=400, hop_length=160, mean_nor=True).to(device)

    def load_enhancer(ckpt):
        model, stft_fn, istft_fn, decompress, _ = load_fullsubnet(
            fullsubnet_root, ckpt, config["model"], device)
        model.eval()

        @torch.no_grad()
        def enhance(noisy):
            noisy_c = stft_fn(noisy, n_fft, hop_length, win_length)
            mag = noisy_c.abs().unsqueeze(1)
            cRM = model(mag, noisy_c.real.unsqueeze(1), noisy_c.imag.unsqueeze(1))
            cRM = decompress(cRM.permute(0, 2, 3, 1))
            er = cRM[..., 0] * noisy_c.real - cRM[..., 1] * noisy_c.imag
            ei = cRM[..., 1] * noisy_c.real + cRM[..., 0] * noisy_c.imag
            return istft_fn(torch.stack([er, ei], -1), n_fft, hop_length, win_length,
                            length=noisy.size(-1))
        return enhance

    baseline_enhance = load_enhancer(baseline_ckpt)
    finetuned_enhance = load_enhancer(finetuned_ckpt) if finetuned_ckpt else None

    def eval_cosine(enhance_fn, snr):
        cos_list = []
        for i in range(args.n_samples):
            clean, _ = torchaudio.load(clean_paths[i])
            clean = clean[0].numpy()
            noise, _ = torchaudio.load(random.choice(noise_paths))
            noise = noise[0].numpy()
            noisy, clean = snr_mix(clean, noise, snr, target_len)

            noisy_t = torch.from_numpy(noisy).unsqueeze(0).to(device)
            clean_t = torch.from_numpy(clean).unsqueeze(0).to(device)
            enhanced = enhance_fn(noisy_t)
            e_clean = eres2net(diff_fbank(clean_t))
            e_enh = eres2net(diff_fbank(enhanced))
            cos = torch.nn.functional.cosine_similarity(e_clean, e_enh, dim=1).item()
            cos_list.append(cos)
        return np.mean(cos_list)

    print("=" * 60)
    print("  enhanced embedding 与 clean 的 mean cosine 对比")
    print("=" * 60)
    for snr in [0, -5]:
        base_cos = eval_cosine(baseline_enhance, snr)
        line = f"  SNR {snr}dB: baseline={base_cos:.4f}"
        if finetuned_enhance:
            fine_cos = eval_cosine(finetuned_enhance, snr)
            line += f"  finetuned={fine_cos:.4f}  (提升 {fine_cos - base_cos:+.4f})"
        print(line)


if __name__ == "__main__":
    main()
