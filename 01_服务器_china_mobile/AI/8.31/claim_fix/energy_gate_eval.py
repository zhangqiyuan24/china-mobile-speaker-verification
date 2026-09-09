#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""#42 收官件：拒识臂能量门控 claim 评测（离线，零训练）。

机制：v8d 对"条件≠真身"（claim 形态）训练目标=输出静音，对"条件=真身"（oracle）
正常增强。嵌入比对口径已官方判死（静音嵌入=OOD 方向，claim 12.60/14.64），
但输出能量本身可能可分——本脚本验证：输出 RMS 能量作二级判据（能量<τ → 直接
判负/拒识）能否把 claim 压过纯前端 7.79/9.92。

trial 构造/嵌入读取/EER 与 analyze_claim_v2.py 逐位同源（跨臂可比）。
能量映射：wav_fe_noise_{cond}.txt / wav_claim_noise_{cond}.txt（官方链产物清单）。
用法: energy_gate_eval.py --arm-root .../official_v8d_sil --out .../energy_gate.json
"""
import os
import json
import argparse
import numpy as np
import torchaudio
from concurrent.futures import ThreadPoolExecutor

SV = ("/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/"
      "egs/3dspeaker/sv-ecapa")
TRIALS = os.path.join(SV, "data/data_for_test/clean/3s/trials_cross_device")
ENROL_DIR = os.path.join(SV, "exp/clean_bench/eres2net/embeddings_3s")


def read_scp_dir(d):
    """目录内 kaldi scp → {key: vec(float32)}，官方同款 kaldiio。"""
    import kaldiio
    import glob
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


def rms_db(path):
    try:
        x, _ = torchaudio.load(path)
        x = x[0].numpy().astype(np.float64)
        return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))
    except Exception:
        return float("nan")


def batch_energy(paths, workers=16):
    with ThreadPoolExecutor(workers) as ex:
        vals = list(ex.map(rms_db, paths))
    return np.array(vals, dtype=np.float64)


def pct(a):
    return {p: float(np.percentile(a, p)) for p in (1, 5, 50, 95)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-root", required=True,
                    help="official_* 臂根目录（enhanced_fe/enhanced_claim/wav_*.txt）")
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
        fe = read_scp_dir(os.path.join(args.arm_root, "fe_emb", cond))
        claim = read_scp_dir(os.path.join(args.arm_root, "claim_emb", cond))
        # raw 口径与门控无关，跳过（跨臂锚已在 summary.txt）

        e_all = [e for e, _ in tgt] + [e for e, _ in non]
        t_all = [t for _, t in tgt] + [t for _, t in non]
        ckeys = ["_".join(e.split("_")[:3]) + "__" + t for e, t in non]
        keep = [i for i, t in enumerate(t_all)
                if t in fe
                and (i < len(tgt) or ckeys[i - len(tgt)] in claim)]
        n_tgt = sum(1 for i in keep if i < len(tgt))
        e_all = [e_all[i] for i in keep]
        t_all = [t_all[i] for i in keep]
        ck = [ckeys[i - len(tgt)] for i in keep if i >= len(tgt)]
        E = stack(enrol, e_all)
        labels = np.array([1] * n_tgt + [0] * (len(e_all) - n_tgt))

        TF = stack(fe, t_all)
        TC = stack(claim, ck)
        s_orc = np.einsum("ij,ij->i", E, TF)
        s_clm = s_orc.copy()
        s_clm[n_tgt:] = np.einsum("ij,ij->i", E[n_tgt:], TC)

        # wav 清单 → basename 映射（fe 平铺；claim 在 flat_{cond}/ 下）
        fe_map = {}
        with open(os.path.join(args.arm_root, f"wav_fe_{cond}.txt")) as f:
            for line in f:
                p = line.strip()
                if p:
                    fe_map[os.path.basename(p)] = p
        clm_map = {}
        with open(os.path.join(args.arm_root, f"wav_claim_{cond}.txt")) as f:
            for line in f:
                p = line.strip()
                if p:
                    clm_map[os.path.basename(p)] = p

        e_fe = batch_energy([fe_map[t + ".wav"] for t in t_all])
        e_clm = batch_energy([clm_map[k + ".wav"] for k in ck])
        print(f"\n===== {cond} ===== keep={len(labels)} "
              f"(tgt={n_tgt}, non={len(ck)})", flush=True)
        print(f"  oracle 能量 dB: {pct(e_fe)}", flush=True)
        print(f"  claim  能量 dB: {pct(e_clm)}", flush=True)

        r = {"energy_oracle": pct(e_fe), "energy_claim": pct(e_clm), "gate": {}}
        for tau in range(-60, -8, 2):
            s_o = s_orc.copy()
            s_o[e_fe < tau] = -1.0
            s_c = s_clm.copy()
            s_c[e_fe < tau] = -1.0          # target 侧（claim≡oracle）
            s_c[n_tgt:][e_clm < tau] = -1.0  # nontarget 侧拒识
            r["gate"][str(tau)] = {
                "oracleFE": eer_pct(s_o, labels),
                "claimFE": eer_pct(s_c, labels),
                "oracle_mute%": float((e_fe < tau).mean() * 100),
                "claim_mute%": float((e_clm < tau).mean() * 100),
            }
            g = r["gate"][str(tau)]
            print(f"  τ={tau:>4}dB oracle={g['oracleFE']:.4f} "
                  f"claim={g['claimFE']:.4f} "
                  f"(哑: oracle {g['oracle_mute%']:.1f}% / "
                  f"claim {g['claim_mute%']:.1f}%)", flush=True)

        best = min(r["gate"].items(),
                   key=lambda kv: kv[1]["claimFE"] + kv[1]["oracleFE"])
        r["best_tau"] = {"db": int(best[0]), **best[1]}
        print(f"  [best] τ={best[0]}dB -> {best[1]}", flush=True)
        out[cond] = r

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[done] {args.out}", flush=True)


if __name__ == "__main__":
    main()
