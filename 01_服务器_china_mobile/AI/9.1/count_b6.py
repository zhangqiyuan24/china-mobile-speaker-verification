#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""count_b6.py — 数 ReDimNet2-B6 官方权重参数量"""
import glob
import os

import torch

SV = ("/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/"
      "egs/3dspeaker/sv-ecapa")
cands = (glob.glob(SV + "/pretrained/redimnet2/*.pt")
         + glob.glob(SV + "/pretrained/redimnet2/**/*.pt", recursive=True)
         + glob.glob("/data/home/ligangrui/china_mobile/AI/**/b6*.pt",
                     recursive=True))
cands = sorted(set(cands))
print("候选:", cands or "无")


def tensor_count(d):
    return sum(1 for v in d.values() if hasattr(v, "numel"))


def deep_find_sd(obj):
    best = None
    if isinstance(obj, dict):
        if tensor_count(obj) > 0 and tensor_count(obj) >= len(obj) * 0.8:
            return obj
        for v in obj.values():
            r = deep_find_sd(v)
            if r is not None and (best is None
                                  or tensor_count(r) > tensor_count(best)):
                best = r
    return best


for p in cands[:3]:
    ck = torch.load(p, map_location="cpu")
    sd = deep_find_sd(ck)
    if sd is None:
        print(f"{p}: top keys={list(ck.keys())[:8] if isinstance(ck, dict) else type(ck)}")
        continue
    n = sum(int(v.numel()) for v in sd.values())
    print(f"ReDimNet2-B6: params={n / 1e6:.2f}M  layers={tensor_count(sd)}  "
          f"file={os.path.getsize(p) / 1024 / 1024:.1f}MB  ({os.path.basename(p)})")
print("[done]")
