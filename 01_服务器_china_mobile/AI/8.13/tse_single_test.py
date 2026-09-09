#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WeSep TSE 前端 — 单条 sanity test

验证: cos(enrollment, extracted_target) > cos(enrollment, mixture)

用法:
  python tse_single_test.py \
      --enroll_wav <A的enrollment.wav> \
      --target_wav <A的clean.wav> \
      --interferer_wav <B的clean.wav> \
      --out_dir ./out
"""

import os
import sys
import argparse
import numpy as np
import yaml
import torch
import soundfile as sf

import common


def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)


def make_mixture(target, interferer, tir_db=0.0):
    """TIR=0dB 混合, 对齐长度, 防削波"""
    if len(interferer) < len(target):
        reps = int(np.ceil(len(target) / len(interferer)))
        interferer = np.tile(interferer, reps)[:len(target)]
    else:
        interferer = interferer[:len(target)]
    scale = rms(target) / (rms(interferer) * 10 ** (tir_db / 20.0))
    mixed = target + interferer * scale
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed / peak * 0.99
    return mixed.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--enroll_wav", required=True)
    parser.add_argument("--target_wav", required=True)
    parser.add_argument("--interferer_wav", required=True)
    parser.add_argument("--tir_db", type=float, default=0.0)
    parser.add_argument("--out_dir", default="./out")
    args = parser.parse_args()

    with open(args.config) as f:
        raw_cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = common.resolve_paths(raw_cfg)

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 1. 加载 wav ----
    print("[1] 加载 wav ...")
    enroll = common.load_wav_mono(args.enroll_wav)
    target = common.load_wav_mono(args.target_wav)
    interferer = common.load_wav_mono(args.interferer_wav)

    # ---- 2. 生成 mixture ----
    print("[2] 生成 mixture (TIR={}dB) ...".format(args.tir_db))
    mixture = make_mixture(target, interferer, args.tir_db)
    sf.write(os.path.join(args.out_dir, "mixture.wav"), mixture, 16000)

    # ---- 3. WeSep 提取 ----
    print("[3] WeSep 提取目标说话人 ...")
    extractor = common.load_wesep_extractor(cfg)
    extracted = common.tse_extract_batch(extractor, [mixture], [enroll])[0]
    sf.write(os.path.join(args.out_dir, "extracted_target.wav"), extracted, 16000)

    # ---- 4. ERes2Net embedding ----
    print("[4] ERes2Net 提 embedding ...")
    model, fbank, device = common.load_eres2net(cfg)
    wavs = [enroll, target, mixture, extracted]
    embs = common.extract_embeddings_batch(model, fbank, device, wavs)
    emb_enroll, emb_target, emb_mixture, emb_extracted = embs

    # ---- 5. cosine ----
    print("\n[5] Cosine 结果:")
    cos_clean = common.cosine(emb_enroll, emb_target)
    cos_mix = common.cosine(emb_enroll, emb_mixture)
    cos_ext = common.cosine(emb_enroll, emb_extracted)
    print("  cos(enrollment, clean_target) = {:.4f}".format(cos_clean))
    print("  cos(enrollment, mixture)      = {:.4f}".format(cos_mix))
    print("  cos(enrollment, extracted)    = {:.4f}".format(cos_ext))

    improved = cos_ext > cos_mix
    print("\n  TSE 改善: {} ({:.4f} vs {:.4f})".format(
        "YES ✓" if improved else "NO ✗", cos_ext, cos_mix))


if __name__ == "__main__":
    main()
