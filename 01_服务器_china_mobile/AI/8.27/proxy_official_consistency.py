#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""proxy_official_consistency.py — proxy EER 与官方 EER 的秩一致性检验（8.27）

背景：proxy 数值比官方乐观 ~5pp（说话人域/设备通道/trial 规模差异，README 已
声明），选点只依赖相对排序。用双侧都有分数的 10 成员（30k_raw 官方侧是旧版
trials 的文件，与 v3 同法跳过）算 Spearman 秩相关，回答「proxy 选点可信吗」。

只读、纯 CPU、不触任何实验。用法: python proxy_official_consistency.py
"""
import argparse
import os

import numpy as np

H = "/data/home/ligangrui/china_mobile/AI"
OFFICIAL = {
    "821_raw":  (H + "/8.21/exp/proto_robust/eval/combo/scores", "raw"),
    "821_fe":   (H + "/8.21/exp/proto_robust/eval/combo/scores", "frontend"),
    "30k_fe":   (H + "/8.22/exp/matched_robust/eval/combo/scores", "frontend"),
    "90k_raw":  (H + "/8.22/exp/matched_robust_90k/eval/combo/scores", "raw"),
    "90k_fe":   (H + "/8.22/exp/matched_robust_90k/eval/combo/scores", "frontend"),
    "soup_raw": (H + "/8.24/exp/soup_821_90k/eval/combo/scores", "raw"),
    "soup_fe":  (H + "/8.24/exp/soup_821_90k/eval/combo/scores", "frontend"),
    "p90k_raw": (H + "/8.24/exp/proto_robust_90k/eval/combo/scores", "raw"),
    "p90k_fe":  (H + "/8.24/exp/proto_robust_90k/eval/combo/scores", "frontend"),
    "tse_soup5": (H + "/8.24/exp/soup5/eval/combo/scores", "frontend"),
}
PROXY = H + "/8.27/exp/proxy/scores"


def eer_of(scores, labels):
    order = np.argsort(-scores)
    labs = labels[order]
    P = int(labels.sum())
    N = int(len(labels) - P)
    tp = np.cumsum(labs)
    fp = np.cumsum(1 - labs)
    fpr = fp / float(N)
    miss = (P - tp) / float(P)
    idx = int(np.argmin(np.abs(fpr - miss)))
    return (fpr[idx] + miss[idx]) / 2.0 * 100.0


def parse(path):
    s, l = [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 4:
                l.append(1 if p[2] == "target" else 0)
                s.append(float(p[3]))
    return np.asarray(s), np.asarray(l)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra = (ra - ra.mean()) / ra.std()
    rb = (rb - rb.mean()) / rb.std()
    return float((ra * rb).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="",
                    help="空=旧随机负对 proxy; hard=困难负对 proxy v2")
    args = ap.parse_args()
    proxy_dir = (PROXY if not args.suffix
                 else os.path.join(os.path.dirname(PROXY),
                                   f"scores_{args.suffix}"))
    print(f"[proxy 侧] {proxy_dir}")
    summary = {}
    for cond in ("noise_0db", "noise_m5db"):
        print(f"===== {cond} =====")
        rows = []
        for m, (d, sub) in OFFICIAL.items():
            try:
                so, lo = parse(os.path.join(d, f"{sub}_{cond}",
                                            "trials_cross_device.score"))
                sp, lp = parse(os.path.join(proxy_dir, f"{m}_{cond}.score"))
                rows.append((m, eer_of(sp, lp), eer_of(so, lo)))
            except IOError as ex:
                print(f"  [skip] {m}: {str(ex)[:60]}")
        print(f"{'member':<10}{'proxy%':>8}{'official%':>10}")
        for m, p, o in rows:
            print(f"{m:<10}{p:>8.3f}{o:>10.4f}")
        ps = np.array([r[1] for r in rows])
        os_ = np.array([r[2] for r in rows])
        rho = spearman(ps, os_)
        print(f"  Spearman rho = {rho:.3f} (n={len(rows)})")
        print(f"  proxy 均值 {ps.mean():.3f} vs 官方均值 {os_.mean():.3f}"
              f"  系统性偏差 {(os_ - ps).mean():+.3f}pp")
        # proxy 排序与官方排序的 top-1 是否一致（选点最关心的）
        best_p = rows[int(np.argmin(ps))][0]
        best_o = rows[int(np.argmin(os_))][0]
        print(f"  proxy 最优成员 = {best_p} / 官方最优成员 = {best_o}"
              f"  {'一致' if best_p == best_o else '不一致'}")
        summary[cond] = {"rho": rho, "bias_pp": float((os_ - ps).mean()),
                         "best_proxy": best_p, "best_official": best_o}
    print("\n结论: ", end="")
    r0, r5 = summary["noise_0db"]["rho"], summary["noise_m5db"]["rho"]
    print(f"rho(0dB)={r0:.2f}, rho(-5dB)={r5:.2f} —— "
          + ("排序可信, 选点有效" if min(r0, r5) >= 0.8
             else "排序弱相关, 选点需谨慎" if min(r0, r5) >= 0.5
             else "排序不可信, 选点仪器失效"))


if __name__ == "__main__":
    main()
