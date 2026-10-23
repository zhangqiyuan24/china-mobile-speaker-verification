#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audio-level spectral features on a random subset, joined with embedding scores."""
import os, csv, json, random
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import correlate

BASE = "/home/ligangrui/china_mobile"
SV = os.path.join(BASE, "code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa")
OUT = os.path.join(BASE, "AI/10.17")
DATA = os.path.join(OUT, "data")
SR = 16000
CONDS = ["clean", "noisy", "generic", "tse"]


def map_wavs(root):
    m = {}
    for dp, _, fs in os.walk(root):
        for fn in fs:
            if fn.endswith(".wav"):
                m[os.path.splitext(fn)[0]] = os.path.join(dp, fn)
    return m


def load(p):
    y, _ = librosa.load(p, sr=SR, mono=True)
    return y.astype(np.float32)


def mel_db(y):
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=512, hop_length=160,
                                       win_length=400, n_mels=80, fmin=20, fmax=7600, power=2.0)
    return 10.0 * np.log10(np.maximum(S, 1e-12))


def si_sdr(est, ref):
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    c = correlate(est, ref, mode="full", method="fft")
    lag = int(np.argmax(np.abs(c)) - (n - 1))
    if lag > 0:
        est = est[lag:]
    elif lag < 0:
        est = np.concatenate([np.zeros(-lag, dtype=est.dtype), est])
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    est = est - est.mean(); ref = ref - ref.mean()
    a = float(np.dot(est, ref) / (np.dot(ref, ref) + 1e-10))
    e = est - a * ref
    return 10.0 * np.log10((np.sum((a * ref) ** 2) + 1e-10) / (np.sum(e ** 2) + 1e-10))


def main():
    print("[index] wavs ...", flush=True)
    CLEAN = map_wavs(os.path.join(SV, "data/data_for_test/clean/3s"))
    NOISY = map_wavs(os.path.join(SV, "data/data_for_test/noise_0db/3s"))
    GENERIC = map_wavs(os.path.join(SV, "data/data_for_test/denoised/noise_0db/enhanced_0194"))
    TSE = map_wavs(os.path.join(BASE, "AI/8.27/exp/tse_v3_ns05duraug_r5/enhanced_official/noise_0db"))
    paths = {"clean": CLEAN, "noisy": NOISY, "generic": GENERIC, "tse": TSE}

    scores = {}
    with open(os.path.join(DATA, "per_utt_scores.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            scores[r["test_id"]] = r
    ids = sorted(scores.keys())
    random.seed(2026)
    random.shuffle(ids)
    chosen = []
    for u in ids:
        if all(u in paths[c] for c in CONDS):
            chosen.append(u)
        if len(chosen) >= 120:
            break
    print("[sample]", len(chosen), "utterances", flush=True)

    rows = []
    for k, u in enumerate(chosen):
        y = {c: load(paths[c][u]) for c in CONDS}
        n = min(len(v) for v in y.values())
        y = {c: v[:n] for c, v in y.items()}
        D = {c: mel_db(v) for c, v in y.items()}
        T = min(v.shape[1] for v in D.values())
        D = {c: v[:, :T] for c, v in D.items()}
        active = D["clean"] > (D["clean"].max() - 35.0)

        def spec(c):
            S = np.abs(librosa.stft(y[c], n_fft=512, hop_length=160)) ** 2
            fr = librosa.fft_frequencies(sr=SR, n_fft=512)
            e = S.sum(1) + 1e-12
            band = lambda a, b: float(e[(fr >= a) & (fr < b)].sum())
            low = band(20, 500); mid = band(500, 2000); high = band(2000, 7600)
            tot = low + mid + high + 1e-12
            m = (fr >= 100) & (fr <= 4000)
            tilt = float(np.polyfit(np.log10(fr[m]), 10 * np.log10(e[m] + 1e-12), 1)[0])
            return low / tot, mid / tot, high / tot, tilt

        base_feat = {}
        for c in CONDS:
            low, mid, high, tilt = spec(c)
            base_feat[c] = (low, mid, high, tilt)
        rms = {c: float(np.sqrt(np.mean(y[c] ** 2)) + 1e-12) for c in CONDS}
        for c in ["generic", "tse", "noisy"]:
            d = D[c] - D["clean"]
            # offset removed per utterance: remove the mean over active bins
            dm = d[active]
            offset = float(np.mean(dm))
            shape = dm - offset
            row = {
                "test_id": u,
                "condition": c,
                "clean_target": float(scores[u]["clean"]),
                "noisy_target": float(scores[u]["noisy"]),
                "generic_target": float(scores[u]["generic"]),
                "tse_target": float(scores[u]["tse"]),
                "drift_clean": float(scores[u]["drift_" + c]),
                "rms_gain_db": 20 * np.log10(rms[c] / rms["clean"]),
                "low_ratio": base_feat[c][0], "mid_ratio": base_feat[c][1], "high_ratio": base_feat[c][2],
                "tilt": base_feat[c][3],
                "mean_db_diff": offset,
                "shape_rmse_db": float(np.sqrt(np.mean(shape ** 2))),
                "mean_abs_shape_db": float(np.mean(np.abs(shape))),
                "si_sdr_db": si_sdr(y[c], y["clean"]),
            }
            rows.append(row)
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(chosen)}", flush=True)

    out_csv = os.path.join(DATA, "audio_features.csv")
    fields = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print("[done]", out_csv, flush=True)

    # Aggregate and correlations
    def mean(a): return float(np.mean(a))

    def corr(a, b):
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        return float(np.corrcoef(a, b)[0, 1])

    summary = {}
    for c in ["noisy", "generic", "tse"]:
        subset = [r for r in rows if r["condition"] == c]
        summary[c] = {
            "n": len(subset),
            "rms_gain_db": mean([r["rms_gain_db"] for r in subset]),
            "low_ratio": mean([r["low_ratio"] for r in subset]),
            "mid_ratio": mean([r["mid_ratio"] for r in subset]),
            "high_ratio": mean([r["high_ratio"] for r in subset]),
            "tilt": mean([r["tilt"] for r in subset]),
            "mean_db_diff": mean([r["mean_db_diff"] for r in subset]),
            "shape_rmse_db": mean([r["shape_rmse_db"] for r in subset]),
            "si_sdr_db": mean([r["si_sdr_db"] for r in subset]),
            "corr_shape_vs_drift": corr([r["shape_rmse_db"] for r in subset],
                                        [r["drift_clean"] for r in subset]),
            "corr_tilt_vs_drift": corr([r["tilt"] for r in subset],
                                       [r["drift_clean"] for r in subset]),
            "corr_sisdr_vs_drift": corr([r["si_sdr_db"] for r in subset],
                                        [r["drift_clean"] for r in subset]),
        }
    # Relationship across all conditions: shape distortion vs identity drift
    summary["all"] = {
        "n": len(rows),
        "corr_shape_vs_drift": corr([r["shape_rmse_db"] for r in rows], [r["drift_clean"] for r in rows]),
        "corr_abs_shape_vs_drift": corr([r["mean_abs_shape_db"] for r in rows], [r["drift_clean"] for r in rows]),
        "corr_sisdr_vs_drift": corr([r["si_sdr_db"] for r in rows], [r["drift_clean"] for r in rows]),
    }
    with open(os.path.join(DATA, "audio_features_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
