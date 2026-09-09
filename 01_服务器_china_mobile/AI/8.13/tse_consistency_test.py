#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WeSep 官方推理接口一致性测试 + target present/absent sanity

1. 一致性测试: 官方 Extractor.extract_speech_from_pcm vs common.tse_extract_batch
   - 同一 (enroll=A, mixture=A+B) 分别跑
   - 比较 waveform 长度 / cosine / max abs diff

2. target present sanity:
   Enroll=A, Mixture=A+B
   期望: cos(A, extracted) > cos(A, mixture)

3. target absent sanity:
   Enroll=A, Mixture=B+C
   期望: cos(A, extracted_from_BC) 不异常升高

用法:
  python tse_consistency_test.py \
      --enroll_wav <A_enroll.wav> \
      --target_wav <A_clean.wav> \
      --interferer_wav <B.wav> \
      --interferer2_wav <C.wav> \
      --out_dir ./out
"""

import os
import argparse
import numpy as np
import yaml
import torch
import soundfile as sf

import common


def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)


def make_mixture(target, interferers, tir_db=0.0):
    """TIR=0dB 混合多个干扰"""
    mixed = target.astype(np.float64)
    for int_wav in interferers:
        if len(int_wav) < len(target):
            reps = int(np.ceil(len(target) / len(int_wav)))
            int_wav = np.tile(int_wav, reps)[:len(target)]
        else:
            int_wav = int_wav[:len(target)]
        scale = rms(target) / (rms(int_wav) * 10 ** (tir_db / 20.0))
        mixed += int_wav * scale
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed / peak * 0.99
    return mixed.astype(np.float32)


def waveform_cosine(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--enroll_wav", required=True, help="A 的 enrollment")
    parser.add_argument("--target_wav", required=True, help="A 的另一条 clean")
    parser.add_argument("--interferer_wav", required=True, help="B")
    parser.add_argument("--interferer2_wav", default="", help="C (target absent 用)")
    parser.add_argument("--out_dir", default="./out")
    args = parser.parse_args()

    with open(args.config) as f:
        raw_cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = common.resolve_paths(raw_cfg)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 加载 ----
    print("[1] 加载 wav ...")
    enroll = common.load_wav_mono(args.enroll_wav)
    target = common.load_wav_mono(args.target_wav)
    interferer = common.load_wav_mono(args.interferer_wav)
    interferer2 = common.load_wav_mono(args.interferer2_wav) if args.interferer2_wav else None

    extractor = common.load_wesep_extractor(cfg)
    eres2net, fbank, device = common.load_eres2net(cfg)

    mixture_present = make_mixture(target, [interferer])

    # ============================================================
    # 1. 一致性测试
    # ============================================================
    print("\n[2] 一致性测试 (官方 vs batch wrapper) ...")
    # 官方 API
    extracted_official = extractor.extract_speech_from_pcm(
        torch.from_numpy(mixture_present).unsqueeze(0), 16000,
        torch.from_numpy(enroll).unsqueeze(0), 16000,
    )
    extracted_official = extracted_official.squeeze(0).cpu().numpy().astype(np.float32)

    # batch wrapper
    extracted_batch = common.tse_extract_batch(extractor, [mixture_present], [enroll])[0]

    len_match = (len(extracted_official) == len(extracted_batch))
    cos = waveform_cosine(extracted_official, extracted_batch)
    max_diff = float(np.abs(extracted_official - extracted_batch).max())
    print(f"  长度一致: {len_match} (official={len(extracted_official)}, batch={len(extracted_batch)})")
    print(f"  waveform cosine: {cos:.6f}")
    print(f"  max abs diff:    {max_diff:.6e}")

    if len_match and cos > 0.999 and max_diff < 1e-4:
        print("  [PASS] 官方 API 与 batch wrapper 输出一致 ✓")
    else:
        print("  [WARN] 输出不一致，需检查 tse_extract_batch 实现 ✗")

    # ============================================================
    # 2. target present sanity
    # ============================================================
    print("\n[3] target present sanity (Enroll=A, Mixture=A+B) ...")
    emb_enroll, emb_target, emb_mix, emb_ext = common.extract_embeddings_batch(
        eres2net, fbank, device, [enroll, target, mixture_present, extracted_batch])
    cos_clean = common.cosine(emb_enroll, emb_target)
    cos_mix = common.cosine(emb_enroll, emb_mix)
    cos_ext = common.cosine(emb_enroll, emb_ext)
    print(f"  cos(A, clean_A)   = {cos_clean:.4f}")
    print(f"  cos(A, mixture)   = {cos_mix:.4f}")
    print(f"  cos(A, extracted) = {cos_ext:.4f}")
    improved = cos_ext > cos_mix
    print(f"  TSE 改善: {'YES ✓' if improved else 'NO ✗'} ({cos_ext:.4f} vs {cos_mix:.4f})")

    # ============================================================
    # 3. target absent sanity
    # ============================================================
    if interferer2 is not None:
        print("\n[4] target absent sanity (Enroll=A, Mixture=B+C) ...")
        mixture_absent = make_mixture(interferer, [interferer2])
        extracted_absent = common.tse_extract_batch(extractor, [mixture_absent], [enroll])[0]
        sf.write(os.path.join(args.out_dir, "extracted_absent.wav"), extracted_absent, 16000)
        emb_absent = common.extract_embeddings_batch(eres2net, fbank, device, [extracted_absent])[0]
        cos_absent = common.cosine(emb_enroll, emb_absent)
        print(f"  cos(A, extracted_from_BC) = {cos_absent:.4f}")
        if cos_absent < cos_clean:
            print(f"  [PASS] target absent 未异常升高 ({cos_absent:.4f} < clean {cos_clean:.4f}) ✓")
        else:
            print(f"  [WARN] target absent 升高 ({cos_absent:.4f} >= clean {cos_clean:.4f}) ✗")
    else:
        print("\n[4] 跳过 target absent (未提供 --interferer2_wav)")

    # 保存
    sf.write(os.path.join(args.out_dir, "mixture.wav"), mixture_present, 16000)
    sf.write(os.path.join(args.out_dir, "extracted_official.wav"), extracted_official, 16000)
    sf.write(os.path.join(args.out_dir, "extracted_batch.wav"), extracted_batch, 16000)
    print(f"\n输出已保存到 {args.out_dir}/")


if __name__ == "__main__":
    main()
