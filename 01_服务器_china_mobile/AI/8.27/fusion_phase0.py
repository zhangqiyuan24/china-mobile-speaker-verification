#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fusion_phase0.py — Phase 0：proxy 搜权重 → 冻结 → 官方一次性应用（纯 CPU）

铁律（用户指令）：融合权重只在 proxy validation 上搜索；official test 只允许用
冻结权重跑一次。本脚本单进程顺序执行两步，中间无人工干预、无第二轮搜索。

步骤：
  1) 读 11 成员 proxy .score（z-norm）→ 0.25 步长 grid → w*（分条件）
     同时打印旧官方权重 w_old=0.25×(821_raw+821_fe+30k_fe+90k_fe) 在 proxy 上的表现
  2) 读官方 .score（z-norm）→ 分别用 w_old（复现口径校验）与 w*（唯一一次）→ EER

输出: exp/phase0/result.json + 汇总表
"""
import itertools
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
PROXY_SCORES = os.path.join(_HERE, "exp", "proxy", "scores")
OUT_DIR = os.path.join(_HERE, "exp", "phase0")

H = "/data/home/ligangrui/china_mobile/AI"
OFFICIAL = {
    "821_raw": H + "/8.21/exp/proto_robust/eval/combo/scores",
    "821_fe": H + "/8.21/exp/proto_robust/eval/combo/scores",
    "30k_raw": H + "/8.22/exp/matched_robust/eval/combo/scores",
    "30k_fe": H + "/8.22/exp/matched_robust/eval/combo/scores",
    "90k_raw": H + "/8.22/exp/matched_robust_90k/eval/combo/scores",
    "90k_fe": H + "/8.22/exp/matched_robust_90k/eval/combo/scores",
    "soup_raw": H + "/8.24/exp/soup_821_90k/eval/combo/scores",
    "soup_fe": H + "/8.24/exp/soup_821_90k/eval/combo/scores",
    "p90k_raw": H + "/8.24/exp/proto_robust_90k/eval/combo/scores",
    "p90k_fe": H + "/8.24/exp/proto_robust_90k/eval/combo/scores",
    "tse_soup5": H + "/8.24/exp/soup5/eval/combo/scores",
}
# 官方侧 tse_soup5 用今天的 frontend 分数；其余成员 raw_*/frontend_* 前缀映射
OFFICIAL_SUBDIR = {m: ("raw" if m.endswith("_raw") else
                       ("frontend" if m != "tse_soup5" else "frontend"))
                   for m in OFFICIAL}

OLD_W = {"821_raw": .25, "821_fe": .25, "30k_fe": .25, "90k_fe": .25}
BASELINE = {"noise_0db": 8.3911, "noise_m5db": 11.0557}
OLD_FUSION_EER = {"noise_0db": 7.2049, "noise_m5db": 9.1533}  # 8.24 v3 官方记录


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


def eer_of(scores, labels):
    order = np.argsort(-scores)
    labs = labels[order]
    P = int(labels.sum())
    N = int(len(labels) - labels.sum())
    tp = np.cumsum(labs)
    fp = np.cumsum(1 - labs)
    fpr = fp / float(N)
    miss = (P - tp) / float(P)
    idx = int(np.argmin(np.abs(fpr - miss)))
    return (fpr[idx] + miss[idx]) / 2.0 * 100.0


def z(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def official_path(member, cond):
    sub = OFFICIAL_SUBDIR[member]
    return os.path.join(OFFICIAL[member], f"{sub}_{cond}", "trials_cross_device.score")


def apply_w(S, names, w):
    return sum(w[n] * s for n, s in zip(names, S) if w.get(n, 0) > 0)


TRIALS_OFFICIAL = ("/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker"
                   "/egs/3dspeaker/sv-ecapa/data/data_for_test/clean/3s"
                   "/trials_cross_device")


def load_official_aligned(path, tkeys, tlabels):
    """以官方 trials 文件为基准对齐（v3 load_aligned 同法）：长度/key 抽查/标签全检。"""
    k, s, l = parse(path)
    assert len(k) == len(tkeys), f"{path}: 行数 {len(k)} != {len(tkeys)}"
    for i in (0, len(k) // 2, len(k) - 1):
        assert k[i] == tkeys[i], f"{path}: key 顺序不一致 @{i}"
    assert (l == tlabels).all(), f"{path}: 标签不一致"
    return s


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {}
    for cond in ("noise_0db", "noise_m5db"):
        print(f"\n===== {cond} =====")
        # ---- 0) 官方可用成员预检（旧版 trials 的分数文件跳过, v3 同法） ----
        tkeys, tlabels = [], []
        with open(TRIALS_OFFICIAL) as f:
            for line in f:
                p = line.split()
                if len(p) == 3:
                    tkeys.append((p[0], p[1]))
                    tlabels.append(1 if p[2] == "target" else 0)
        tlabels = np.asarray(tlabels)
        names, Sn_off = [], {}
        for m in OFFICIAL:
            try:
                Sn_off[m] = load_official_aligned(official_path(m, cond),
                                                  tkeys, tlabels)
            except (AssertionError, IOError) as ex:
                print(f"  [官方侧跳过] {m}: {str(ex)[:70]}")
        names = [m for m in OFFICIAL if m in Sn_off]
        print(f"  [官方侧可用] {len(names)}/{len(OFFICIAL)}: {names}")

        # ---- 1) proxy 搜索（只在双侧可用成员上） ----
        S, labels = [], None
        for m in names:
            p = os.path.join(PROXY_SCORES, f"{m}_{cond}.score")
            k, s, l = parse(p)
            if labels is None:
                labels = l
            else:
                assert (l == labels).all(), f"{m} proxy 标签不一致"
            S.append(z(s))
            print(f"  proxy 单路 {m:12s} EER={eer_of(s, l):.3f}%")
        steps = [i * 0.25 for i in range(5)]
        best = (1e9, None)
        for w in itertools.product(steps, repeat=len(names)):
            if abs(sum(w) - 1.0) > 1e-9:
                continue
            wd = dict(zip(names, w))
            e = eer_of(apply_w(S, names, wd), labels)
            if e < best[0]:
                best = (e, wd)
        e_proxy_new, w_new = best
        e_proxy_old = eer_of(apply_w(S, names, OLD_W), labels)
        print(f"  [proxy] grid 最优 EER={e_proxy_new:.3f}%  w*="
              + "+".join(f"{v:.2f}×{k}" for k, v in w_new.items() if v > 0))
        print(f"  [proxy] 旧官方权重 在 proxy 上 EER={e_proxy_old:.3f}%")

        # ---- 2) 官方一次性应用（无搜索, 预检时已对齐） ----
        Sn = [z(Sn_off[m]) for m in names]
        e_off_old = eer_of(apply_w(Sn, names, OLD_W), tlabels)
        e_off_new = eer_of(apply_w(Sn, names, w_new), tlabels)
        rel_bl = (BASELINE[cond] - e_off_new) / BASELINE[cond] * 100
        rel_old = (OLD_FUSION_EER[cond] - e_off_new) / OLD_FUSION_EER[cond] * 100
        print(f"  [官方] 旧权重复现 EER={e_off_old:.4f}%（记录值 "
              f"{OLD_FUSION_EER[cond]})")
        print(f"  [官方] w* 一次应用 EER={e_off_new:.4f}%"
              f"  相对基线 -{rel_bl:.1f}%  相对旧融合 -{rel_old:.1f}%")
        report[cond] = {"w_new": w_new, "e_proxy_old": e_proxy_old,
                        "e_proxy_new": e_proxy_new,
                        "e_official_old_repro": e_off_old,
                        "e_official_new": e_off_new,
                        "rel_vs_baseline": rel_bl, "rel_vs_oldfusion": rel_old}

    with open(os.path.join(OUT_DIR, "result.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n===== Phase 0 汇总（官方, trials_cross_device） =====")
    print(f"{'方案':<28}{'0dB EER%':>10}{'-5dB EER%':>10}")
    print(f"{'旧融合(官方搜权, 记录值)':<28}{OLD_FUSION_EER['noise_0db']:>10.4f}"
          f"{OLD_FUSION_EER['noise_m5db']:>10.4f}")
    print(f"{'旧融合+TSE(proxy 权重 w*)':<28}"
          f"{report['noise_0db']['e_official_new']:>10.4f}"
          f"{report['noise_m5db']['e_official_new']:>10.4f}")
    print("[saved]", os.path.join(OUT_DIR, "result.json"))


if __name__ == "__main__":
    main()
