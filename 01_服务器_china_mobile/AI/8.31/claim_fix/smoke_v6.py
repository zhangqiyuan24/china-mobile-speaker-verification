#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""smoke_v6.py — v6 数据集 CPU 冒烟：mix 路径形状/叠加生效/静音防护，不碰 GPU"""
import os
import sys

import numpy as np
import torch
import toml

HERE = "/data/home/ligangrui/china_mobile/AI/8.27"
for _p in (HERE,
           "/data/home/ligangrui/china_mobile/AI/8.18",
           "/data/home/ligangrui/china_mobile/AI/8.20",
           "/data/home/ligangrui/china_mobile/AI/8.21",
           "/data/home/ligangrui/china_mobile/AI/8.26"):
    sys.path.insert(0, _p)
from train_tse import build_paths_from_csv
from train_spk_preserve import load_noise_paths
from train_tse_v6_mix import TSEMixDatasetV6

cfg = toml.load(os.path.join(os.path.dirname(HERE), "8.26", "train_tse.toml"))
clean_paths, labels, spk2id = build_paths_from_csv(
    os.path.join(cfg["paths"]["data_dir_820"], "train_utts.csv"))
emb = np.load(cfg["paths"]["spk_embs"], allow_pickle=True)["embs"]
noise = load_noise_paths(cfg["dataset"]["noise_list"],
                         os.path.abspath(cfg["paths"]["sv_ecapa"]))
sr = cfg["acoustics"]["sr"]
tl = int(cfg["dataset"]["sub_sample_length"] * sr)
ds = TSEMixDatasetV6(clean_paths, labels, emb, noise,
                     cfg["dataset"]["snr_range"], tl, sr,
                     imp_prob=1.0, mix_prob=1.0)
mixn = 0
ratios = []
for i in range(12):
    noisy, clean, lb, e, impf, mixf = ds[i]
    assert noisy.shape == clean.shape == (tl,), noisy.shape
    assert float(impf) == 1.0, "imp 标志失效"
    p_a = float(clean.pow(2).mean())
    ratios.append(float(noisy.pow(2).mean()) / (p_a + 1e-12))
    mixn += int(mixf)
print(f"[smoke] 12 样本全过, mix命中 {mixn}/12, noisy/clean 功率比 "
      f"min={min(ratios):.2f} max={max(ratios):.2f}")
assert mixn >= 8, "mix 命中率异常（mix_prob=1.0 应≈全命中，允许静音句跳过）"
print("[smoke] OK")
