# -*- coding: utf-8 -*-
"""跨系统分数融合探索（8.24）：8.21 robust 与 8.22 matched-30k 的官方 score 文件
在 trial 级做 z-norm + 线性加权，探 ensemble 上限。纯 CPU，只读已有 .score。
注：30k raw_noise_0db 文件被截断（936k/1378k 行），0dB 档不含该路。
"""
import sys, itertools
import numpy as np


def eer_of(scores, labels):
    order = np.argsort(-scores)
    labs = labels[order]
    P = int(labels.sum()); N = int(len(labels) - P)
    tp = np.cumsum(labs); fp = np.cumsum(1 - labs)
    fpr = fp / float(N); miss = (P - tp) / float(P)
    idx = int(np.argmin(np.abs(fpr - miss)))
    return (fpr[idx] + miss[idx]) / 2.0 * 100.0

TRIALS = "/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa/data/data_for_test/clean/3s/trials_cross_device"
R821 = "/home/ligangrui/china_mobile/AI/8.21/exp/proto_robust/eval/combo/scores"
R90K = "/home/ligangrui/china_mobile/AI/8.22/exp/matched_robust_90k/eval/combo/scores"
RSOUP = "/home/ligangrui/china_mobile/AI/8.24/exp/soup_821_90k/eval/combo/scores"
R30K = "/home/ligangrui/china_mobile/AI/8.22/exp/matched_robust/eval/combo/scores"

def parse(path):
    keys, scores, labels = [], [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) != 4:
                continue
            keys.append((p[0], p[1]))
            labels.append(1 if p[2] == "target" else 0)
            scores.append(float(p[3]))
    return keys, np.asarray(scores), np.asarray(labels)

print("[load] trials ...")
tk, tl = [], []
with open(TRIALS) as f:
    for line in f:
        p = line.split()
        if len(p) != 3:
            continue
        tk.append((p[0], p[1]))
        tl.append(1 if p[2] == "target" else 0)
labels = np.asarray(tl)
print("trials:", len(tk))

def load_aligned(path):
    k, s, l = parse(path)
    assert len(k) == len(tk), path + ": 行数不一致 " + str(len(k))
    for i in (0, len(k) // 2, len(k) - 1):
        assert k[i] == tk[i], path + ": key 顺序不一致 @" + str(i)
    assert (l == labels).all(), path + ": 标签不一致"
    return s

def z(x):
    return (x - x.mean()) / (x.std() + 1e-8)

def grid_best(S_list):
    n = len(S_list)
    best = (1e9, None)
    steps = [i * 0.25 for i in range(5)]
    for w in itertools.product(steps, repeat=n):
        if abs(sum(w) - 1.0) > 1e-9 or all(x == 0 for x in w):
            continue
        f = sum(wi * si for wi, si in zip(w, S_list))
        e = eer_of(f, labels)
        if e < best[0]:
            best = (e, w)
    return best

BL = {"noise_0db": 8.3911, "noise_m5db": 11.0557}
for cond in ("noise_0db", "noise_m5db"):
    print("")
    print("===== " + cond + " =====")
    systems = [
        ("821_raw", R821 + "/raw_" + cond + "/trials_cross_device.score"),
        ("821_fe",  R821 + "/frontend_" + cond + "/trials_cross_device.score"),
        ("30k_fe",  R30K + "/frontend_" + cond + "/trials_cross_device.score"),
        ("30k_raw", R30K + "/raw_" + cond + "/trials_cross_device.score"),
        ("90k_raw", R90K + "/raw_" + cond + "/trials_cross_device.score"),
        ("90k_fe",  R90K + "/frontend_" + cond + "/trials_cross_device.score"),
        ("soup_raw", RSOUP + "/raw_" + cond + "/trials_cross_device.score"),
        ("soup_fe",  RSOUP + "/frontend_" + cond + "/trials_cross_device.score"),
    ]
    names, S = [], []
    for nm, p in systems:
        try:
            s = load_aligned(p)
        except AssertionError as ex:
            print("  跳过 %-8s（%s）" % (nm, str(ex)[:60]))
            continue
        names.append(nm)
        S.append(z(s))
        e = eer_of(s, labels)
        print("  单路 %-8s EER=%.4f%%" % (nm, e))
    uni = S[0].copy()
    for si in S[1:]:
        uni = uni + si
    uni = uni / len(S)
    e = eer_of(uni, labels)
    joined = "+".join(names)
    print("  uniform z 融合(%s) EER=%.4f%%" % (joined, e))
    be, bw = grid_best(S)
    wstr = "+".join("%.2f*%s" % (a, nm) for a, nm in zip(bw, names) if a > 0)
    rel = (BL[cond] - be) / BL[cond] * 100
    print("  grid 最优 EER=%.4f%%  (%s)  相对基线 -%.1f%%" % (be, wstr, rel))
