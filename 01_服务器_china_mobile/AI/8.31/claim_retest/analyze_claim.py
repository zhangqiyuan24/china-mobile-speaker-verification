#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""analyze_claim.py — #37 claim 口径四对比：raw / oracleFE / claimFE（同对集 EER）

对集 = 全部 211,400 target 对 + 30,000 抽样 nontarget 对（seed 2026）。
- target 侧 claim ≡ oracle（声称者=真身），分数直接取交付 FE 嵌入；
- nontarget 侧三种口径分别计算：
    raw     = cos(enrol, raw带噪嵌入)
    oracle  = cos(enrol, FE嵌入[按真身条件化])   ← 交付口径
    claim   = cos(enrol, FE嵌入[按声称者条件化]) ← 部署口径（本次新增）
自检 1：由官方 .score 全文复算 EER，须命中官方值（0dB 4.8920 / −5dB 5.5634）。
自检 2：本脚本余弦对官方 .score 若干行逐行对齐（容差 1e-3）。
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SV = ("/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/"
      "egs/3dspeaker/sv-ecapa")
FIN = "/data/home/ligangrui/china_mobile/AI/8.28/final_eval_r5"
TRIALS = os.path.join(SV, "data/data_for_test/clean/3s/trials_cross_device")
ENROL_DIR = os.path.join(SV, "exp/clean_bench/eres2net/embeddings_3s")
BOOT = 200


def read_scp_dir(d):
    """目录内 kaldi scp → {key: vec(float32)}，用官方同款 kaldiio 库读取。"""
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
    idx = np.where(np.diff(np.sign(d)) != 0)[0]     # 相邻异号区间
    i = int(idx[np.argmin(np.abs(d[idx]))])
    t = d[i] / (d[i] - d[i + 1])
    return float((frr[i] + t * (frr[i + 1] - frr[i])) * 100)


def cosmat(vec_dict, keys):
    """按键序堆叠成 (n,d) 并 L2 归一。"""
    miss = [k for k in set(keys) if k not in vec_dict]
    return miss


def main():
    print("== 读 trials 与抽样 ==", flush=True)
    tgt = []
    with open(TRIALS) as f:
        for line in f:
            e, t, lab = line.split()
            if lab == "target":
                tgt.append((e, t))
    non = []
    with open(os.path.join(HERE, "sampled_nontarget.txt")) as f:
        for line in f:
            e, t, _ = line.split()
            non.append((e, t))
    print(f"targets={len(tgt)} sampled_nontarget={len(non)}", flush=True)

    enrol = read_scp_dir(ENROL_DIR)
    print(f"enrol 嵌入 {len(enrol)}", flush=True)

    out = {}
    for cond, official_eer in (("noise_0db", 4.8920), ("noise_m5db", 5.5634)):
        print(f"\n===== {cond} =====", flush=True)
        raw = read_scp_dir(os.path.join(SV, "exp/clean_bench", cond,
                                        "eres2net/embeddings_3s"))
        fe = read_scp_dir(os.path.join(FIN, f"r5_eres_{cond}_3s"))
        claim = read_scp_dir(os.path.join(HERE, "emb", cond))
        print(f"raw={len(raw)} fe={len(fe)} claim={len(claim)}", flush=True)

        def stack(d, ks):
            M = np.stack([d[k] for k in ks])
            M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
            return M

        # 自检 2：与官方 .score 头若干行余弦对齐
        scp_path = os.path.join(FIN, f"r5_eres_{cond}_3s/trials_cross_device.score")
        n_chk = ok = 0
        with open(scp_path) as f:
            for line in f:
                e, t, lab, s = line.split()
                if t in fe and e in enrol:
                    a = enrol[e] / (np.linalg.norm(enrol[e]) + 1e-12)
                    b = fe[t] / (np.linalg.norm(fe[t]) + 1e-12)
                    if abs(float(a @ b) - float(s)) < 1e-3:
                        ok += 1
                    n_chk += 1
                if n_chk >= 200:
                    break
        print(f"[自检2] FE 余弦 vs 官方 score: {ok}/{n_chk} 对齐", flush=True)

        # ---- 三口径分数（同对集） ----
        e_all = [e for e, _ in tgt] + [e for e, _ in non]
        t_all = [t for _, t in tgt] + [t for _, t in non]
        E = stack(enrol, e_all)
        miss_raw = cosmat(raw, t_all)
        miss_fe = cosmat(fe, t_all)
        ckeys = ["_".join(e.split("_")[:3]) + "__" + t for e, t in non]
        miss_cl = [k for k in set(ckeys) if k not in claim]
        if miss_raw or miss_fe or miss_cl:
            print(f"[missing] raw={len(miss_raw)} fe={len(miss_fe)} "
                  f"claim={len(miss_cl)}（缺者剔除该对）", flush=True)
            keep = [i for i, t in enumerate(t_all)
                    if t in raw and t in fe
                    and (i < len(tgt) or ckeys[i - len(tgt)] in claim)]
            e_all = [e_all[i] for i in keep]
            t_all = [t_all[i] for i in keep]
            ckeys = [ckeys[i - len(tgt)] for i in keep if i >= len(tgt)]
            n_tgt = sum(1 for i in keep if i < len(tgt))
            E = stack(enrol, e_all)
        else:
            n_tgt = len(tgt)
        labels = np.array([1] * n_tgt + [0] * (len(e_all) - n_tgt))

        TR = stack(raw, t_all)
        TF = stack(fe, t_all)
        s_raw = np.einsum("ij,ij->i", E, TR)
        s_orc = np.einsum("ij,ij->i", E, TF)
        s_clm = s_orc.copy()
        if len(ckeys):
            TC = stack(claim, ckeys)
            Ec = E[n_tgt:]
            s_clm[n_tgt:] = np.einsum("ij,ij->i", Ec, TC)

        # 自检 1：官方 .score 全文复算 EER
        sc, lb = [], []
        with open(scp_path) as f:
            for line in f:
                _, _, lab, s = line.split()
                lb.append(1 if lab == "target" else 0)
                sc.append(float(s))
        eer_full = eer_pct(np.array(sc), np.array(lb))
        print(f"[自检1] 官方全文复算 EER={eer_full:.4f} (官方 {official_eer})",
              flush=True)

        r = {}
        for name, s in (("raw", s_raw), ("oracleFE", s_orc), ("claimFE", s_clm)):
            r[name] = eer_pct(s, labels)
        # bootstrap CI（claimFE）
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
        print(f"raw={r['raw']:.4f}  oracleFE={r['oracleFE']:.4f}  "
              f"claimFE={r['claimFE']:.4f} CI95=[{r['claimFE_ci95'][0]:.3f},"
              f"{r['claimFE_ci95'][1]:.3f}]", flush=True)
        # 冒认侧分数位移（中位数）
        print(f"nontarget 分数中位: raw={np.median(s_raw[labels == 0]):.4f} "
              f"oracle={np.median(s_orc[labels == 0]):.4f} "
              f"claim={np.median(s_clm[labels == 0]):.4f}", flush=True)
        out[cond] = r

    with open(os.path.join(HERE, "claim_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n[done] claim_summary.json", flush=True)


if __name__ == "__main__":
    main()
