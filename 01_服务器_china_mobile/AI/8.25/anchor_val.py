#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""anchor_val.py — 给指定 ckpt 算"训练内小验证"(800 trials)锚点。

用途: SAM/consist/distill 三线各自 init 不同, ep1 验证需要各自的起点锚,
否则无法判断"守住还是下探"。CPU 跑(不占卡), 与 train_* 的 val 完全同口径。
用法: python anchor_val.py --ckpts <ckpt1> <ckpt2> ...
"""
import argparse
import os
import sys

import torch

_BASE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_BASE)
for _p in (os.path.join(_PARENT, "8.18"), os.path.join(_PARENT, "8.20")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from common_utils import setup_speakerlab          # noqa: E402
from trainer_spk_preserve import load_eres2net     # noqa: E402
from validate_robust import run_validation         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--sv-ecapa",
                    default="/data/home/ligangrui/china_mobile/code/3D-speaker/"
                            "3D-speaker/egs/3dspeaker/sv-ecapa")
    ap.add_argument("--data",
                    default="/data/home/ligangrui/china_mobile/AI/8.22/exp/data_90k/data")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=24)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    setup_speakerlab(args.sv_ecapa, _BASE)

    for c in args.ckpts:
        model = load_eres2net(args.sv_ecapa, c, "cpu")
        m = run_validation(model, os.path.join(args.data, "val_bench"),
                           os.path.join(args.data, "val_trials.txt"),
                           "cpu", batch_size=args.batch_size)
        avg = (m["0db"]["eer"] + m["m5db"]["eer"]) / 2
        tag = "/".join(c.rstrip("/").split("/")[-3:])
        print(f"[anchor] {tag}: clean={m['clean']['eer']:.3f} "
              f"0db={m['0db']['eer']:.3f} m5db={m['m5db']['eer']:.3f} "
              f"avg01={avg:.3f}", flush=True)


if __name__ == "__main__":
    main()
