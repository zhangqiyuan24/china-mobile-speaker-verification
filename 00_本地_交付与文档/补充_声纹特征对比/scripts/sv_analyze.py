#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare speaker embeddings: clean vs noisy vs FullSubNet+ vs TSE.

Uses the *exact* artifacts from the previous China Mobile internship runs:
  generic: AI/8.17/models/best_model.tar -> denoised/noise_0db/enhanced_0194
  TSE:     AI/8.27/exp/tse_v3_ns05duraug_r5/checkpoint_best.tar -> enhanced_official
Embeddings are the official ERes2Net-large embeddings stored by the old runs.
"""
import os, json, csv, re, math
import numpy as np
import kaldiio

BASE = "/home/ligangrui/china_mobile"
SV = os.path.join(BASE, "code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa")
OUT = os.path.join(BASE, "AI/10.17")
os.makedirs(os.path.join(OUT, "data"), exist_ok=True)

CONDS = ["clean", "noisy", "generic", "tse"]
EMB_DIRS = {
    "clean":   os.path.join(SV, "exp/clean_bench/eres2net/embeddings_3s"),
    "noisy":   os.path.join(SV, "exp/clean_bench/noise_0db/eres2net/embeddings_3s"),
    "generic": os.path.join(SV, "exp/clean_bench/denoised_noise_0db/eres2net/embeddings_3s"),
    "tse":     os.path.join(BASE, "AI/8.28/final_eval_r5/r5_eres_noise_0db_3s"),
}
METRIC_FILES = {
    "clean":   None,
    "noisy":   os.path.join(SV, "data/data_for_test/clean/result/noisy_0db/eres2net/3s/result.metrics"),
    "generic": os.path.join(SV, "data/data_for_test/clean/result/denoised_noise_0db/eres2net/3s/result.metrics"),
    "tse":     os.path.join(BASE, "AI/8.28/final_eval_r5/r5_eres_noise_0db_3s/result.metrics"),
}
TRIALS = os.path.join(SV, "data/data_for_test/clean/3s/trials_cross_device")

CHUNK = 10000


def load_embs(d):
    embs = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".ark"):
            continue
        for k, v in kaldiio.load_ark(os.path.join(d, fn)):
            v = np.asarray(v, dtype=np.float32)
            n = float(np.linalg.norm(v))
            if n > 1e-8:
                v = v / n
            embs[k] = v
    return embs


def read_metric(path):
    if path is None or not os.path.exists(path):
        return None
    txt = open(path, encoding="utf-8").read()
    out = {}
    m = re.search(r"EER\s*=\s*([0-9.]+)", txt)
    if m:
        out["eer_pct"] = float(m.group(1))
    m = re.search(r"minDCF[^=]*=\s*([0-9.]+)", txt)
    if m:
        out["minDCF"] = float(m.group(1))
    return out


def eer_from_scores(scores, labels):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(labels, scores, drop_intermediate=False)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2.0 * 100.0), float(thr[i]), float(fpr[i]), float(fnr[i])


def main():
    print("[load] embeddings ...", flush=True)
    EMB = {c: load_embs(d) for c, d in EMB_DIRS.items()}
    for c in CONDS:
        print(f"  {c:8s} {len(EMB[c])} embeddings", flush=True)
    sample_key = next(iter(EMB["clean"]))
    print("  dim =", EMB["clean"][sample_key].shape, flush=True)

    n_trials = 1378206
    scores = {c: np.empty(n_trials, dtype=np.float32) for c in CONDS}
    labels = np.empty(n_trials, dtype=bool)
    target_sum = {c: {} for c in CONDS}
    target_cnt = {c: {} for c in CONDS}
    target_enroll = {}

    print("[score] pass over trials ...", flush=True)
    idx = 0
    with open(TRIALS, encoding="utf-8") as f:
        lines = []
        for line in f:
            lines.append(line)
            if len(lines) >= CHUNK:
                en = []; te = []; lab = []
                for ln in lines:
                    p = ln.split()
                    if len(p) < 3:
                        continue
                    en.append(p[0]); te.append(p[1]); lab.append(p[2] == "target")
                lab_arr = np.asarray(lab, dtype=bool)
                n = len(lab_arr)
                # Official protocol: clean enrollment, condition-specific test side.
                E_clean = np.stack([EMB["clean"][x] for x in en])
                for c in CONDS:
                    T = np.stack([EMB[c][x] for x in te])
                    s = np.einsum("ij,ij->i", E_clean, T).astype(np.float32)
                    scores[c][idx:idx + n] = s
                labels[idx:idx + n] = lab_arr
                for j in np.nonzero(lab_arr)[0]:
                    t = te[j]
                    for c in CONDS:
                        target_sum[c][t] = target_sum[c].get(t, 0.0) + float(scores[c][idx + j])
                        target_cnt[c][t] = target_cnt[c].get(t, 0) + 1
                    target_enroll.setdefault(t, []).append(en[j])
                idx += n
                lines = []
                if idx % 200000 < CHUNK:
                    print(f"  {idx}/{n_trials}", flush=True)
        if lines:
            en = []; te = []; lab = []
            for ln in lines:
                p = ln.split()
                if len(p) < 3:
                    continue
                en.append(p[0]); te.append(p[1]); lab.append(p[2] == "target")
            lab_arr = np.asarray(lab, dtype=bool)
            n = len(lab_arr)
            E_clean = np.stack([EMB["clean"][x] for x in en])
            for c in CONDS:
                T = np.stack([EMB[c][x] for x in te])
                s = np.einsum("ij,ij->i", E_clean, T).astype(np.float32)
                scores[c][idx:idx + n] = s
            labels[idx:idx + n] = lab_arr
            for j in np.nonzero(lab_arr)[0]:
                t = te[j]
                for c in CONDS:
                    target_sum[c][t] = target_sum[c].get(t, 0.0) + float(scores[c][idx + j])
                    target_cnt[c][t] = target_cnt[c].get(t, 0) + 1
                target_enroll.setdefault(t, []).append(en[j])
            idx += n
    print(f"  scored {idx} trials", flush=True)

    # Keep only test IDs that have target trials (the scoring-side utterances)
    test_ids = sorted(target_cnt["clean"].keys())
    print(f"[target] {len(test_ids)} test utterances, {labels.sum()} target trials", flush=True)

    per_test = {c: np.array([target_sum[c].get(t, np.nan) / max(target_cnt[c].get(t, 1), 1)
                             for t in test_ids], dtype=np.float32) for c in CONDS}

    drift = {}
    for c in ["noisy", "generic", "tse"]:
        vals = []
        for t in test_ids:
            vals.append(float(np.dot(EMB["clean"][t], EMB[c][t])))
        drift[c] = np.asarray(vals, dtype=np.float32)
    drift["clean"] = np.ones(len(test_ids), dtype=np.float32)

    summary = {}
    for c in CONDS:
        eer, thr, fpr, fnr = eer_from_scores(scores[c], labels)
        tgt = scores[c][labels]
        non = scores[c][~labels]
        gap = float(tgt.mean() - non.mean())
        pooled = float(np.sqrt(0.5 * (tgt.var() + non.var())))
        summary[c] = {
            "eer_pct_computed": eer,
            "eer_pct_stored": (read_metric(METRIC_FILES[c]) or {}).get("eer_pct"),
            "minDCF_stored": (read_metric(METRIC_FILES[c]) or {}).get("minDCF"),
            "eer_threshold_cos": thr,
            "n_target": int(labels.sum()),
            "n_nontarget": int((~labels).sum()),
            "target_mean": float(tgt.mean()),
            "target_std": float(tgt.std()),
            "nontarget_mean": float(non.mean()),
            "nontarget_std": float(non.std()),
            "gap": gap,
            "dprime": gap / pooled if pooled > 0 else float("nan"),
            "per_utt_cos_to_enroll_mean": float(np.nanmean(per_test[c])),
            "per_utt_cos_to_enroll_median": float(np.nanmedian(per_test[c])),
            "cos_to_clean_mean": float(np.nanmean(drift[c])) if c != "clean" else 1.0,
            "cos_to_clean_median": float(np.nanmedian(drift[c])) if c != "clean" else 1.0,
        }
        print(f"[{c}] EER={eer:.4f}% (stored {summary[c]['eer_pct_stored']}) "
              f"target={tgt.mean():.4f} non={non.mean():.4f} gap={gap:.4f} "
              f"d'={summary[c]['dprime']:.3f}", flush=True)

    # Write per-utterance table
    csv_path = os.path.join(OUT, "data", "per_utt_scores.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["test_id", "speaker", "device", "clean", "noisy", "generic", "tse",
                    "drift_noisy", "drift_generic", "drift_tse", "enroll_ids"])
        for i, t in enumerate(test_ids):
            spk = t.split("_")[2] if t.startswith("3D_SPK_") else t.split("_")[0]
            m = re.search(r"Device(\d+)", t)
            dev = f"Device{m.group(1)}" if m else ""
            w.writerow([t, spk, dev] +
                       [f"{per_test[c][i]:.6f}" for c in CONDS] +
                       [f"{drift[c][i]:.6f}" for c in ["noisy", "generic", "tse"]] +
                       [";".join(sorted(set(target_enroll.get(t, []))))])

    # Selection: generic fails while raw passes, TSE recovers
    thr_n = summary["noisy"]["eer_threshold_cos"]
    thr_g = summary["generic"]["eer_threshold_cos"]
    thr_t = summary["tse"]["eer_threshold_cos"]
    cand = []
    for i, t in enumerate(test_ids):
        r, g, s, c = (float(per_test[x][i]) for x in ["noisy", "generic", "tse", "clean"])
        drop = r - g
        rec = s - g
        cand.append({
            "test_id": t, "speaker": t.split("_")[2] if t.startswith("3D_SPK_") else t.split("_")[0],
            "device": (re.search(r"Device\d+", t).group(0) if re.search(r"Device\d+", t) else ""),
            "clean": c, "noisy": r, "generic": g, "tse": s,
            "drop_by_generic": drop, "recovery_by_tse": rec,
            "raw_pass": r > thr_n, "generic_fail": g < thr_g, "tse_pass": s > thr_t,
        })
    # Prefer flips, then rank by drop + recovery
    preferred = [x for x in cand if x["raw_pass"] and x["generic_fail"] and x["tse_pass"]]
    if len(preferred) < 4:
        preferred = sorted(cand, key=lambda x: (x["drop_by_generic"] + x["recovery_by_tse"]), reverse=True)
    else:
        preferred = sorted(preferred, key=lambda x: (x["drop_by_generic"] + x["recovery_by_tse"]), reverse=True)
    selected = []
    used_spk = set(); dev_count = {}
    for x in preferred:
        if x["speaker"] in used_spk:
            continue
        if dev_count.get(x["device"], 0) >= 2:
            continue
        selected.append(x); used_spk.add(x["speaker"]); dev_count[x["device"]] = dev_count.get(x["device"], 0) + 1
        if len(selected) >= 5:
            break
    print("[select]")
    for x in selected:
        print("  ", x["test_id"], "clean=%.3f noisy=%.3f generic=%.3f tse=%.3f drop=%.3f rec=%.3f"
              % (x["clean"], x["noisy"], x["generic"], x["tse"], x["drop_by_generic"], x["recovery_by_tse"]))

    # Save selected embeddings for PCA (enrollment is clean by protocol)
    sel_ids = [x["test_id"] for x in selected]
    enroll_means = []
    for t in sel_ids:
        es = sorted(set(target_enroll[t]))
        enroll_means.append(np.stack([EMB["clean"][e] for e in es]).mean(axis=0))
    np.savez(os.path.join(OUT, "data", "selected_embeddings.npz"),
             ids=np.array(sel_ids),
             enroll_mean=np.stack(enroll_means),
             **{f"emb_{c}": np.stack([EMB[c][t] for t in sel_ids]) for c in CONDS})

    out = {"summary": summary, "selected": selected,
           "thresholds": {"noisy": thr_n, "generic": thr_g, "tse": thr_t},
           "n_test_utts": len(test_ids)}
    with open(os.path.join(OUT, "data", "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[done]", os.path.join(OUT, "data", "summary.json"), flush=True)


if __name__ == "__main__":
    main()
