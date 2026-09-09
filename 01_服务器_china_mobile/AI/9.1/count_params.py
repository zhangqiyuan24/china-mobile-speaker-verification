#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""count_params.py — 数项目三个模型的真实参数量/文件大小（从 checkpoint 实测）"""
import glob
import os

import toml
import torch

AI = "/data/home/ligangrui/china_mobile/AI"
cfg = toml.load(os.path.join(AI, "8.26", "train_tse.toml"))


def load_sd(p):
    ck = torch.load(p, map_location="cpu")
    if isinstance(ck, dict):
        for k in ("model", "state_dict", "model_state_dict", "params"):
            v = ck.get(k)
            if hasattr(v, "values"):
                return v
        vals = list(ck.values())
        if vals and all(hasattr(v, "numel") for v in vals[:3]):
            return ck
    return ck


def report(name, p):
    if not p or not os.path.exists(p):
        print(f"{name}: NOT FOUND ({p})")
        return
    try:
        sd = load_sd(p)
        n = sum(int(v.numel()) for v in sd.values() if hasattr(v, "numel"))
        mb = os.path.getsize(p) / 1024 / 1024
        print(f"{name}: params={n / 1e6:.2f}M  file={mb:.1f}MB  ({p})")
    except Exception as ex:
        print(f"{name}: ERR {str(ex)[:90]}  ({p})")


print("== 前端 ==")
report("FullSubNet+ TSE 终版(r5)", os.path.join(
    AI, "8.27/exp/tse_v3_ns05duraug_r5/checkpoint_best.tar"))
report("FullSubNet+ 通用基座", cfg["paths"]["fullsubnet_ckpt"])

print("== 后端 ==")
report("ERes2Net-large (verifier/微调)", cfg["paths"]["verifier_ckpt"])
print("  8.26/exp 目录:", sorted(os.listdir(os.path.join(AI, "8.26", "exp")))[:12])
for pat in (AI + "/8.26/exp/**/*.pt", AI + "/8.26/exp/**/*.pth",
            AI + "/8.26/exp/**/*.bin", AI + "/8.26/exp/**/*.tar"):
    for p in sorted(glob.glob(pat, recursive=True))[:8]:
        report("  8.26exp ckpt", p)

print("== soup5 ==")
for pat in (AI + "/8.24/exp/soup5/**/*.pt",
            AI + "/8.24/exp/soup5/**/*.pth",
            AI + "/8.24/exp/soup5/**/*.bin",
            AI + "/8.24/exp/soup5*/*.pt"):
    for p in sorted(set(glob.glob(pat, recursive=True)))[:3]:
        report("soup5", p)
print("[done]")
