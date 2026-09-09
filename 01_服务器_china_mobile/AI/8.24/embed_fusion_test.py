#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""embed_fusion_test.py — 嵌入级融合验证（8.24 备胎①第一步）

对已落盘的官方 trial ark(evaluate_combo 产物)做嵌入平均融合:
每 utt 取 K 个模型 L2 归一化嵌入的均值再归一化, 余弦打分算 EER。
用途:
  1) 验证集成 teacher 质量(融合嵌入若显著优于单模型 -> born-again 蒸馏有料)
  2) 融合本身也是一种交付形态(口径同分数融合, 待老师确认)
纯 CPU/numpy, 只读 ark。
"""
import os
import sys
import itertools
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_V820 = os.path.join(os.path.dirname(_HERE), "8.20")
if _V820 not in sys.path:
    sys.path.insert(0, _V820)
from common_utils import load_ark_dir, load_trials, eer_minDCF, setup_speakerlab

setup_speakerlab("../../code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa", _HERE)

MODELS = {
    "proto_30k": os.path.join(_HERE, "../8.21/exp/proto_robust/eval/combo"),
    "matched_30k": os.path.join(_HERE, "../8.22/exp/matched_robust/eval/combo"),
    "matched_90k": os.path.join(_HERE, "../8.22/exp/matched_robust_90k/eval/combo"),
    "soup_50": os.path.join(_HERE, "exp/soup_821_90k/eval/combo"),
}
CONDS = ("clean", "noise_0db", "noise_m5db")
TRIALS = ("/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/"
          "egs/3dspeaker/sv-ecapa/data/data_for_test/clean/3s/trials_cross_device")
BASELINE = {"clean": 5.8248, "noise_0db": 8.3911, "noise_m5db": 11.0557}


def _norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def fuse(ar_list, keys):
    """ar_list: 每个模型 {utt: emb}; 返回 keys 顺序的融合归一化嵌入矩阵。"""
    out = []
    for ar in ar_list:
        out.append(_norm(np.stack([ar[k] for k in keys])))
    return _norm(np.mean(out, axis=0))


def eer_of(E, T, trials):
    e_row = {k: i for i, k in enumerate(EK)}
    t_row = {k: i for i, k in enumerate(TK)}
    sc, lb = [], []
    for e, t, lab in trials:
        sc.append(float(np.dot(E[e_row[e]], T[t_row[t]])))
        lb.append(1 if lab in (1, "1", "target") else 0)
    return eer_minDCF(np.array(sc), np.array(lb))[0]


if __name__ == "__main__":
    trials = load_trials(TRIALS)
    print(f"trials={len(trials)}")
    print(f"{'组合':<34}{'clean':>8}{'0dB':>8}{'-5dB':>8}   0dB/-5dB 相对基线")
    names = list(MODELS)
    # 单模型基准
    arks = {n: {c: load_ark_dir(os.path.join(MODELS[n], "raw", c))
                for c in CONDS} for n in names}
    enrols = {n: load_ark_dir(os.path.join(MODELS[n], "common", "enrol_clean"))
              for n in names}
    global EK, TK

    def report(label, ar_list, en_list):
        global EK, TK
        cond_lists = {c: [a[c] for a in ar_list] for c in CONDS}
        EK = sorted(en_list[0]); TK = sorted(cond_lists[CONDS[0]][0])
        E = fuse(en_list, EK)
        row = []
        for c in CONDS:
            T = fuse(cond_lists[c], TK)
            row.append(eer_of(E, T, trials))
        rel0 = 100 * (BASELINE["noise_0db"] - row[1]) / BASELINE["noise_0db"]
        rel5 = 100 * (BASELINE["noise_m5db"] - row[2]) / BASELINE["noise_m5db"]
        print(f"{label:<34}{row[0]:>8.4f}{row[1]:>8.4f}{row[2]:>8.4f}"
              f"   {rel0:+.1f}% / {rel5:+.1f}%")

    for n in names:
        report(n, [arks[n]], [enrols[n]])
    # 全部两两 + 三三 + 全四组合
    for r in range(2, len(names) + 1):
        for combo in itertools.combinations(names, r):
            report("+".join(combo), [arks[n] for n in combo],
                   [enrols[n] for n in combo])
