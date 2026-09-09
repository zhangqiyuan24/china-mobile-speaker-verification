#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""report_metrics.py — 任务书第4条分项报告（#21，纯 CPU，8.28）

输入官方管线的 *.score（四列: enrol test label score），每条件输出：
  metrics.json   EER / minDCF(p=0.01) 复算 + 固定阈值 FAR/FRR（阈值=clean raw EER 工作点）
  det.png        DET 曲线（正态变换，0.1%-50%）
  roc.csv        ROC 曲线数据（FAR/TPR 遍历阈值）
  failures.csv   FR top-N（target 分最低）+ FA top-N（nontarget 分最高），含元数据
  strata.txt     失败案例分层统计（Device/Distance/Dialect 富集 vs 全体）

校验锚：official_ns05ext2 raw_noise_0db EER=7.6139 minDCF=0.5942。
复用：report_all.sh 扫描目录批量跑；reverb 场景可 --metadata 关联 T60。

用法:
  python report_metrics.py --score path/to/x.score --name raw_noise_0db \
      --out-root exp/analysis_21 [--ref-thr 0.362] [--topn 20]
"""
import argparse
import json
import os
import re
from collections import Counter

import numpy as np

STE_RE = re.compile(
    r"(3D_SPK_\d+)_\d+_(Device\d+)_(Distance\d+)_(Dialect\d+)")


def parse_stem(stem):
    m = STE_RE.search(stem)
    if not m:
        return None
    return {"spk": m.group(1), "device": m.group(2),
            "distance": m.group(3), "dialect": m.group(4)}


def eer_minDCF(scores, labels, p_target=0.01, c_miss=1.0, c_fa=1.0):
    """ scores 升序扫描。返回 eer, eer_thr, mindcf, dcf_thr。
    与 speakerlab compute_score_metrics 同口径（p=0.01, c=1,1）。"""
    order = np.argsort(scores)
    s, y = np.asarray(scores)[order], np.asarray(labels)[order]
    n_tar = int((y == 1).sum())
    n_non = len(y) - n_tar
    # 阈值取 s 中的切点：t_k = s[k]，判 target 当 score >= t
    # FRR(k) = (k 中 target 数)/n_tar （< t 的 target）
    # FAR(k) = (n_non - k 中 nontarget 数)/n_non （>= t 的 nontarget）
    cum_tar = np.cumsum(y == 1)
    cum_non = np.cumsum(y == 0)
    k = np.arange(1, len(y) + 1)
    # 右侧（>= s[k-1]）的计数
    tar_ge = n_tar - cum_tar + (y == 1)          # 含并列
    non_ge = n_non - cum_non + (y == 0)
    frr = 1 - tar_ge / n_tar
    far = non_ge / n_non
    i = np.argmin(np.abs(far - frr))
    eer = (far[i] + frr[i]) / 2
    dcf = p_target * frr * c_miss + (1 - p_target) * far * c_fa
    j = np.argmin(dcf)
    dcfn = dcf[j] / min(p_target * c_miss, (1 - p_target) * c_fa)
    return float(eer * 100), float(s[i]), float(dcfn), float(s[j])


def rates_at(scores, labels, thr):
    y = np.asarray(labels)
    sc = np.asarray(scores)
    pred = sc >= thr
    far = float((pred & (y == 0)).sum() / (y == 0).sum())
    frr = float((~pred & (y == 1)).sum() / (y == 1).sum())
    return far * 100, frr * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ref-thr", type=float, default=None,
                    help="固定阈值（clean raw 的 EER 工作点）")
    ap.add_argument("--topn", type=int, default=20)
    args = ap.parse_args()

    enrol, test, lab, sc = [], [], [], []
    with open(args.score) as f:
        for line in f:
            p = line.split()
            if len(p) != 4:
                continue
            enrol.append(p[0]); test.append(p[1])
            lab.append(1 if p[2] == "target" else 0)
            sc.append(float(p[3]))
    sc = np.asarray(sc); lab = np.asarray(lab)
    out = os.path.join(args.out_root, args.name)
    os.makedirs(out, exist_ok=True)

    eer, eer_thr, mindcf, dcf_thr = eer_minDCF(sc, lab)
    res = {"name": args.name, "n_trials": len(lab),
           "n_target": int((lab == 1).sum()),
           "n_nontarget": int((lab == 0).sum()),
           "EER": round(eer, 4), "EER_thr": round(eer_thr, 5),
           "minDCF(p=0.01)": round(mindcf, 4), "minDCF_thr": round(dcf_thr, 5)}
    if args.ref_thr is not None:
        far, frr = rates_at(sc, lab, args.ref_thr)
        res["fixed_thr"] = args.ref_thr
        res["FAR@fixed"] = round(far, 3)
        res["FRR@fixed"] = round(frr, 3)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res), flush=True)

    # ---- ROC/DET 数据 ----
    order = np.argsort(sc)
    s, y = sc[order], lab[order]
    n_tar = int((y == 1).sum()); n_non = len(y) - n_tar
    cum_tar = np.cumsum(y == 1); cum_non = np.cumsum(y == 0)
    tar_ge = n_tar - cum_tar + (y == 1)
    non_ge = n_non - cum_non + (y == 0)
    frr = 1 - tar_ge / n_tar
    far = non_ge / n_non
    keep = slice(None, None, max(1, len(s) // 4000))   # 抽稀存 csv
    with open(os.path.join(out, "roc.csv"), "w") as f:
        f.write("thr,far,frr,tpr\n")
        for k in range(*keep.indices(len(s))):
            f.write(f"{s[k]:.5f},{far[k]:.6f},{frr[k]:.6f},{1-frr[k]:.6f}\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.stats import norm
        eps = 1e-5
        frr_p = np.clip(frr, eps, 1 - eps)
        far_p = np.clip(far, eps, 1 - eps)
        sel = (far_p > 1e-4) | (frr_p > 1e-4)
        plt.figure(figsize=(5, 5))
        plt.plot(norm.ppf(far_p[sel]), norm.ppf(frr_p[sel]),
                 lw=1.2, label=args.name)
        ticks = norm.ppf([0.001, 0.005, 0.02, 0.05, 0.2, 0.5])
        tlabs = ["0.1", "0.5", "2", "5", "20", "50"]
        plt.xticks(ticks, tlabs); plt.yticks(ticks, tlabs)
        plt.xlabel("FAR (%)"); plt.ylabel("FRR (%)")
        plt.title(f"DET {args.name} EER={eer:.2f}")
        plt.grid(alpha=.3); plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out, "det.png"), dpi=150)
        plt.close()
    except Exception as e:                                  # noqa: BLE001
        print(f"[warn] plot failed: {e}", flush=True)

    # ---- 失败案例 + 分层 ----
    ti = np.argsort(sc)
    fr_idx = [i for i in ti if lab[i] == 1][:args.topn]
    fa_idx = [i for i in ti[::-1] if lab[i] == 0][:args.topn]
    with open(os.path.join(out, "failures.csv"), "w") as f:
        f.write("type,score,enrol_spk,test_spk,enrol_stem,test_stem,"
                "device,distance,dialect\n")
        for i in fr_idx:
            meta = parse_stem(test[i]) or {}
            f.write(f"FR,{sc[i]:.4f},{parse_stem(enrol[i])['spk'] if parse_stem(enrol[i]) else '?'},"
                    f"{meta.get('spk','?')},{enrol[i]},{test[i]},"
                    f"{meta.get('device','?')},{meta.get('distance','?')},"
                    f"{meta.get('dialect','?')}\n")
        for i in fa_idx:
            meta = parse_stem(test[i]) or {}
            f.write(f"FA,{sc[i]:.4f},{parse_stem(enrol[i])['spk'] if parse_stem(enrol[i]) else '?'},"
                    f"{meta.get('spk','?')},{enrol[i]},{test[i]},"
                    f"{meta.get('device','?')},{meta.get('distance','?')},"
                    f"{meta.get('dialect','?')}\n")
    strata = []
    for tag, idxs in (("FR", fr_idx), ("FA", fa_idx)):
        for key in ("device", "distance", "dialect"):
            cnt = Counter()
            for i in idxs:
                m = parse_stem(test[i])
                if m:
                    cnt[m[key]] += 1
            top = ", ".join(f"{k}:{v}" for k, v in cnt.most_common(4))
            strata.append(f"{tag}-{key}: {top}")
    with open(os.path.join(out, "strata.txt"), "w") as f:
        f.write("\n".join(strata) + "\n")
    print(f"[done] {args.name} -> {out}", flush=True)


if __name__ == "__main__":
    main()
