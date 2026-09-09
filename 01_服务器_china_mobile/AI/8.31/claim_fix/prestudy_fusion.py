#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""prestudy_fusion.py — #38 预研①：部署层分数融合回退（零训练，零 GPU）

问题：claim 口径 nontarget 分数被 L_spk 拉高（claim EER 10.72/12.85 反超 raw
8.37/11.09，冒认余弦中位 ×3）。部署时同一对可**同时**拿到 raw 分数与
FE·claim 分数——测分数层规则能修多少：
  R1  min(raw, claim)：冒认抬高被 raw 封顶（最稳、无参数）
  R2  s' = a·claim + b·raw：网格扫描（在本对集上调参 = 诊断上界，有过拟合
      警示；仅用于判断分数层信息量，不作交付值）
参照：raw / FE·oracle / FE·claim 三口径 EER（analyze_claim 已测）。

输出：fusion_summary.json + fusion_scores.npz（供后续复算）。
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLAIM = "/data/home/ligangrui/china_mobile/AI/8.31/claim_retest"
SV = ("/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/"
      "egs/3dspeaker/sv-ecapa")
FIN = "/data/home/ligangrui/china_mobile/AI/8.28/final_eval_r5"
TRIALS = os.path.join(SV, "data/data_for_test/clean/3s/trials_cross_device")
ENROL_DIR = os.path.join(SV, "exp/clean_bench/eres2net/embeddings_3s")


def read_scp_dir(d):
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


def stack(d, keys):
    M = np.stack([d[k] for k in keys])
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def main():
    os.chdir(SV)  # clean_bench 的 scp 内 ark 为相对路径，须以 SV 为 cwd 解析
    tgt, non = [], []
    with open(TRIALS) as f:
        for line in f:
            e, t, lab = line.split()
            if lab == "target":
                tgt.append((e, t))
    with open(os.path.join(CLAIM, "sampled_nontarget.txt")) as f:
        for line in f:
            e, t, _ = line.split()
            non.append((e, t))
    print(f"targets={len(tgt)} nontargets={len(non)}", flush=True)

    enrol = read_scp_dir(ENROL_DIR)
    out = {"n_target": len(tgt), "n_nontarget": len(non)}
    for cond in ("noise_0db", "noise_m5db"):
        raw = read_scp_dir(os.path.join(SV, "exp/clean_bench", cond,
                                        "eres2net/embeddings_3s"))
        fe = read_scp_dir(os.path.join(FIN, f"r5_eres_{cond}_3s"))
        claim = read_scp_dir(os.path.join(CLAIM, "emb", cond))

        e_all = [e for e, _ in tgt] + [e for e, _ in non]
        t_all = [t for _, t in tgt] + [t for _, t in non]
        ckeys = ["_".join(e.split("_")[:3]) + "__" + t for e, t in non]
        keep = [i for i, t in enumerate(t_all) if t in raw and t in fe]
        keep_non = [i for i, k in enumerate(ckeys) if k in claim]
        # 对齐：target 侧缺嵌入的剔除；nontarget 侧缺 claim 嵌入的剔除
        tgt_idx = [i for i in keep if i < len(tgt)]
        non_idx = [len(tgt) + j for j in keep_non
                   if (len(tgt) + j) in keep]
        n_tgt, n_non = len(tgt_idx), len(non_idx)
        labels = np.array([1] * n_tgt + [0] * n_non)

        E_t = stack(enrol, [e_all[i] for i in tgt_idx])
        E_n = stack(enrol, [e_all[i] for i in non_idx])
        TR_t = stack(raw, [t_all[i] for i in tgt_idx])
        TR_n = stack(raw, [t_all[i] for i in non_idx])
        TF_t = stack(fe, [t_all[i] for i in tgt_idx])
        CL_n = stack(claim, [ckeys[i - len(tgt)] for i in non_idx])

        s_raw = np.concatenate([np.einsum("ij,ij->i", E_t, TR_t),
                                np.einsum("ij,ij->i", E_n, TR_n)])
        s_orc = np.concatenate([np.einsum("ij,ij->i", E_t, TF_t),
                                np.full(n_non, np.nan)])  # nontarget 侧部署不可得
        s_clm = np.concatenate([np.einsum("ij,ij->i", E_t, TF_t),  # target: claim≡oracle
                                np.einsum("ij,ij->i", E_n, CL_n)])
        res = {
            "raw": eer_pct(s_raw, labels),
            "oracle": eer_pct(np.nan_to_num(s_orc, nan=-9), labels),  # 仅参考
            "claim": eer_pct(s_clm, labels),
        }
        # R1: min(raw, claim) —— 无参数规则
        res["min_raw_claim"] = eer_pct(np.minimum(s_raw, s_clm), labels)
        # R2: 线性网格（诊断上界，同集调参）
        best = (1e9, None, None)
        for a in np.arange(0.0, 1.51, 0.1):
            for b in np.arange(0.0, 1.51, 0.1):
                if a == 0 and b == 0:
                    continue
                e = eer_pct(a * s_clm + b * s_raw, labels)
                if e < best[0]:
                    best = (e, round(float(a), 2), round(float(b), 2))
        res["grid_best"], res["grid_a"], res["grid_b"] = best
        out[cond] = res
        print(f"[{cond}] " + " ".join(f"{k}={v:.3f}" if isinstance(v, float)
                                      else f"{k}={v}" for k, v in res.items()),
              flush=True)
        np.savez_compressed(
            os.path.join(HERE, f"fusion_scores_{cond}.npz"),
            labels=labels, s_raw=s_raw, s_clm=s_clm)

    with open(os.path.join(HERE, "fusion_summary.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("saved fusion_summary.json", flush=True)


if __name__ == "__main__":
    main()
