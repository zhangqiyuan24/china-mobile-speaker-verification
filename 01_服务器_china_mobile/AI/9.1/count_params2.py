#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""count_params2.py — 递归解 speakerlab 嵌套 ckpt，数 ERes2Net/ECAPA 参数量"""
import os

import torch

AI = "/data/home/ligangrui/china_mobile/AI"


def tensor_count(d):
    return sum(1 for v in d.values() if hasattr(v, "numel"))


def deep_find_sd(obj, depth=0):
    """返回张量最多的 tensor-dict（处理 model/state_dict/嵌套包裹）"""
    best = None
    if isinstance(obj, dict):
        if tensor_count(obj) > 0 and tensor_count(obj) >= len(obj) * 0.8:
            return obj
        for v in obj.values():
            r = deep_find_sd(v, depth + 1)
            if r is not None:
                n = tensor_count(r)
                if best is None or n > tensor_count(best):
                    best = r
    return best


def report(name, p):
    ck = torch.load(p, map_location="cpu")
    sd = deep_find_sd(ck)
    if sd is None:
        print(f"{name}: 未找到 state_dict, top keys="
              f"{list(ck.keys())[:6] if isinstance(ck, dict) else type(ck)}")
        return
    n = sum(int(v.numel()) for v in sd.values())
    print(f"{name}: params={n / 1e6:.2f}M  layers={tensor_count(sd)}  "
          f"file={os.path.getsize(p) / 1024 / 1024:.1f}MB")


report("ERes2Net soup5(verifier)", f"{AI}/8.24/exp/soup5/checkpoints/checkpoint_soup5.tar")
report("ECAPA robust_90k", f"{AI}/8.26/exp/ecapa_robust_90k/checkpoints/checkpoint_best.tar")
print("[done]")
