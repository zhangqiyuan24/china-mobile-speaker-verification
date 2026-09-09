#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""analyze_claim_v2.py — #38 臂级官方 claim 评测分析（analyze_claim 参数化版）

对集与 #37 完全一致（全部 211,400 target + sampled_nontarget.txt，seed 2026），
保证与 r5 的 claimFE 10.722/12.850、raw 8.367/11.087 逐位可比。
三口径：
    raw     = cos(enrol, 官方基线 raw 嵌入)          ← 臂间共用锚
    oracleFE= cos(enrol, 该臂真身条件 FE 嵌入)        ← 交付口径
    claimFE = nontarget 侧换声称者条件 FE（target 侧 claim≡oracle）
自检 1/2（r5 官方 score 对齐）仅 r5 适用，本版略去；raw 口径即跨臂可比锚。
"""
import argparse
import glob
import json
import os

import numpy as np

SV = ("/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/"
      "egs/3dspeaker/sv-ecapa")
TRIALS = os.path.join(SV, "data/data_for_test/clean/3s/trials_cross_device")
ENROL_DIR = os.path.join(SV, "exp/clean_bench/eres2net/embeddings_3s")
BOOT = 200


def read_scp_dir(d):
    """目录内 kaldi scp → {key: vec(float32)}，官方同款 kaldiio。"""
    import kaldiio
    scp = sorted(glob.glob(os.path.join(d, "*.scp")))
    assert scp, f"无 scp: {d}"
    vecs = {}
    for s in scp:
        with kaldiio.ReadHelper("scp:" + s) as r:
            for key, vec in r:
                vecs[key] = np.asarray(vec, dtype=np.float32)
    return vecs


def eer_pct(scores, labels):
    """插值交点 EER(%)。scores>=t 判正。"""
    pos = np.sort(scores[labels == 1])
    neg = np.sort(scores[labels == 0])
    ts = np.unique(scores)
    far = 1.0 - np.searchsorted(neg, ts, side="left") / len(neg)
    frr = np.searchsorted(pos, ts, side="left") / len(pos)
    d = far - frr
    if (d >= 0).all() or (d <= 0).all():
        i = int(np.argmin(np.abs(d)))
        return float(min(far[i], frr[i]) * 100)
    idx = np.where(np.diff(np.sign(d)) != 0)[0]
    i = int(idx[np.argmin(np.abs(d[idx]))])
    t = d[i] / (d[i] - d[i + 1])
    return float((frr[i] + t * (frr[i + 1] - frr[i])) * 100)


def stack(d, ks):
    M = np.stack([d[k] for k in ks])
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fe-root", required=True,
                    help="全集 oracle 条件 FE 嵌入根（内含 {cond}/）")
    ap.add_argument("--claim-root", required=True,
                    help="冒认集声称者条件嵌入根（内含 {cond}/）")
    ap.add_argument("--sampled",
                    default="/data/home/ligangrui/china_mobile/AI/8.31/"
                            "claim_retest/sampled_nontarget.txt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tgt = []
    with open(TRIALS) as f:
        for line in f:
            e, t, lab = line.split()
            if lab == "target":
                tgt.append((e, t))
    non = []
    with open(args.sampled) as f:
        for line in f:
            e, t, _ = line.split()
            non.append((e, t))
    print(f"targets={len(tgt)} sampled_nontarget={len(non)}", flush=True)
    enrol = read_scp_dir(ENROL_DIR)

    out = {}
    for cond in ("noise_0db", "noise_m5db"):
        raw = read_scp_dir(os.path.join(SV, "exp/clean_bench", cond,
                                        "eres2net/embeddings_3s"))
        fe = read_scp_dir(os.path.join(args.fe_root, cond))
        claim = read_scp_dir(os.path.join(args.claim_root, cond))
        print(f"\n===== {cond} ===== raw={len(raw)} fe={len(fe)} "
              f"claim={len(claim)}", flush=True)

        e_all = [e for e, _ in tgt] + [e for e, _ in non]
        t_all = [t for _, t in tgt] + [t for _, t in non]
        ckeys = ["_".join(e.split("_")[:3]) + "__" + t for e, t in non]
        keep = [i for i, t in enumerate(t_all)
                if t in raw and t in fe
                and (i < len(tgt) or ckeys[i - len(tgt)] in claim)]
        n_miss = len(t_all) - len(keep)
        if n_miss:
            print(f"  [missing] 剔除 {n_miss} 对（缺嵌入）", flush=True)
        n_tgt = sum(1 for i in keep if i < len(tgt))
        e_all = [e_all[i] for i in keep]
        t_all = [t_all[i] for i in keep]
        ck = [ckeys[i - len(tgt)] for i in keep if i >= len(tgt)]
        E = stack(enrol, e_all)
        labels = np.array([1] * n_tgt + [0] * (len(e_all) - n_tgt))

        TR = stack(raw, t_all)
        TF = stack(fe, t_all)
        TC = stack(claim, ck)
        s_raw = np.einsum("ij,ij->i", E, TR)
        s_orc = np.einsum("ij,ij->i", E, TF)
        s_clm = s_orc.copy()
        s_clm[n_tgt:] = np.einsum("ij,ij->i", E[n_tgt:], TC)

        r = {k: eer_pct(s, labels) for k, s in
             (("raw", s_raw), ("oracleFE", s_orc), ("claimFE", s_clm))}
        rng = np.random.default_rng(2026)
        boots = []
        for _ in range(BOOT):
            idx = np.concatenate([
                rng.choice(n_tgt, n_tgt, replace=True),
                n_tgt + rng.choice(len(labels) - n_tgt,
                                   len(labels) - n_tgt, replace=True)])
            boots.append(eer_pct(s_clm[idx], labels[idx]))
        r["claimFE_ci95"] = [float(np.percentile(boots, 2.5)),
                             float(np.percentile(boots, 97.5))]
        r["nt_med"] = {"raw": float(np.median(s_raw[labels == 0])),
                       "oracle": float(np.median(s_orc[labels == 0])),
                       "claim": float(np.median(s_clm[labels == 0]))}
        print(f"  raw={r['raw']:.4f} oracleFE={r['oracleFE']:.4f} "
              f"claimFE={r['claimFE']:.4f} CI95=[{r['claimFE_ci95'][0]:.3f},"
              f"{r['claimFE_ci95'][1]:.3f}] nt_med={r['nt_med']}", flush=True)
        out[cond] = r

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[done] {args.out}", flush=True)


if __name__ == "__main__":
    main()
