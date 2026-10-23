#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Intrusive metrics on the same 120 random test utterances:
SI-SDR / SI-SDRi, VAD-segmental SNR / delta, PESQ-WB / delta, STOI / delta.
"""
import os, csv, json
import numpy as np
import soundfile as sf
from scipy.signal import correlate
from pesq import pesq
from pystoi import stoi

BASE = "/home/ligangrui/china_mobile"
SV = os.path.join(BASE, "code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa")
OUT = os.path.join(BASE, "AI/10.17")
DATA = os.path.join(OUT, "data")
if not os.path.isdir(DATA):
    # local fallback
    OUT = r"E:\china_mobile_review\声纹特征对比"
    DATA = os.path.join(OUT, "data")
SR = 16000
CONDS = ["noisy", "generic", "tse"]


def map_wavs(root):
    m = {}
    for dp, _, fs in os.walk(root):
        for fn in fs:
            if fn.endswith(".wav"):
                m[os.path.splitext(fn)[0]] = os.path.join(dp, fn)
    return m


def load(p):
    y, _ = sf.read(p, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(1)
    return y.astype(np.float32)


def align(est, ref, max_lag=800):
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    c = correlate(est, ref, mode="full", method="fft")
    lag = int(np.argmax(np.abs(c)) - (n - 1))
    if lag > 0:
        est = est[lag:]
    elif lag < 0:
        est = np.concatenate([np.zeros(-lag, dtype=est.dtype), est])
    n = min(len(est), len(ref))
    return est[:n], ref[:n], lag


def si_sdr(est, ref):
    est, ref, _ = align(est, ref)
    est = est - est.mean(); ref = ref - ref.mean()
    a = float(np.dot(est, ref) / (np.dot(ref, ref) + 1e-10))
    target = a * ref
    noise = est - target
    return 10.0 * np.log10((np.sum(target ** 2) + 1e-10) / (np.sum(noise ** 2) + 1e-10))


def frame_energy(x, win=512, hop=256):
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx]
    return np.mean(frames ** 2, axis=1) + 1e-12


def seg_snr(est, ref):
    """VAD-segmental SNR using clean reference energy to define noise/speech frames.
    Ratio of mean speech-frame power to mean noise-frame power within the same signal.
    """
    est, ref, _ = align(est, ref)
    ec = frame_energy(ref)
    ee = frame_energy(est)
    n = min(len(ec), len(ee)); ec, ee = ec[:n], ee[:n]
    lo = np.percentile(ec, 20)
    hi = np.percentile(ec, 60)
    noise_mask = ec <= lo
    speech_mask = ec >= hi
    if noise_mask.sum() < 2 or speech_mask.sum() < 2:
        return float("nan")
    speech = float(np.mean(ee[speech_mask]))
    noise = float(np.mean(ee[noise_mask]))
    return 10.0 * np.log10((speech + 1e-12) / (noise + 1e-12))


def main():
    # 120 random IDs used for spectral analysis
    ids = []
    seen = set()
    with open(os.path.join(DATA, "audio_features.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["test_id"] not in seen:
                seen.add(r["test_id"]); ids.append(r["test_id"])
    print("[ids]", len(ids), flush=True)

    CLEAN = map_wavs(os.path.join(SV, "data/data_for_test/clean/3s"))
    NOISY = map_wavs(os.path.join(SV, "data/data_for_test/noise_0db/3s"))
    GENERIC = map_wavs(os.path.join(SV, "data/data_for_test/denoised/noise_0db/enhanced_0194"))
    TSE = map_wavs(os.path.join(BASE, "AI/8.27/exp/tse_v3_ns05duraug_r5/enhanced_official/noise_0db"))
    paths = {"clean": CLEAN, "noisy": NOISY, "generic": GENERIC, "tse": TSE}

    rows = []
    for k, u in enumerate(ids):
        if not all(u in paths[c] for c in ["clean", "noisy", "generic", "tse"]):
            continue
        ref = load(paths["clean"][u])
        vals = {}
        for c in ["noisy", "generic", "tse"]:
            est = load(paths[c][u])
            n = min(len(ref), len(est))
            r, e = ref[:n], est[:n]
            vals[c] = {
                "si_sdr": si_sdr(e, r),
                "seg_snr": seg_snr(e, r),
                "pesq": float(pesq(SR, r, e, "wb")),
                "stoi": float(stoi(r, e, SR, extended=False)),
            }
        for c in CONDS:
            row = {"test_id": u, "condition": c,
                   "si_sdr_db": vals[c]["si_sdr"],
                   "seg_snr_db": vals[c]["seg_snr"],
                   "pesq_wb": vals[c]["pesq"],
                   "stoi": vals[c]["stoi"],
                   "si_sdri_db": vals[c]["si_sdr"] - vals["noisy"]["si_sdr"],
                   "snri_db": vals[c]["seg_snr"] - vals["noisy"]["seg_snr"],
                   "d_pesq": vals[c]["pesq"] - vals["noisy"]["pesq"],
                   "d_stoi": vals[c]["stoi"] - vals["noisy"]["stoi"]}
            rows.append(row)
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(ids)}", flush=True)

    out_csv = os.path.join(DATA, "intrusive_metrics.csv")
    fields = ["test_id", "condition", "si_sdr_db", "si_sdri_db", "seg_snr_db", "snri_db",
              "pesq_wb", "d_pesq", "stoi", "d_stoi"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print("[done]", out_csv, flush=True)

    summary = {}
    for c in CONDS:
        sub = [r for r in rows if r["condition"] == c]
        summary[c] = {}
        for key in ["si_sdr_db", "si_sdri_db", "seg_snr_db", "snri_db", "pesq_wb", "d_pesq", "stoi", "d_stoi"]:
            a = np.array([r[key] for r in sub], dtype=float)
            summary[c][key] = {"mean": float(np.nanmean(a)), "median": float(np.nanmedian(a)),
                               "std": float(np.nanstd(a))}
        summary[c]["n"] = len(sub)
        summary[c]["pct_improved_pesq"] = float(np.mean([r["d_pesq"] > 0 for r in sub]))
        summary[c]["pct_improved_stoi"] = float(np.mean([r["d_stoi"] > 0 for r in sub]))
        summary[c]["pct_improved_si_sdr"] = float(np.mean([r["si_sdri_db"] > 0 for r in sub]))
        summary[c]["pct_improved_snr"] = float(np.mean([r["snri_db"] > 0 for r in sub]))
    with open(os.path.join(DATA, "intrusive_metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"noisy": "#e69f00", "generic": "#d55e00", "tse": "#0072b2"}
    labels = {"noisy": "Noisy 0 dB", "generic": "FullSubNet+ (original)", "tse": "TSE (ours)"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key, title, ylim in [
            (axes[0], "si_sdr_db", "SI-SDR (dB)", None),
            (axes[1], "pesq_wb", "PESQ-WB", (0.5, 4.7)),
            (axes[2], "stoi", "STOI", (0.0, 1.0))]:
        vals = [summary[c][key]["mean"] for c in CONDS]
        errs = [summary[c][key]["std"] for c in CONDS]
        bars = ax.bar(range(3), vals, yerr=errs, color=[colors[c] for c in CONDS], capsize=4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(3)); ax.set_xticklabels(["Noisy", "FullSubNet+", "TSE"], rotation=20, ha="right")
        ax.set_title(title)
        if ylim: ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Intrusive metrics on 120 random test utterances (mean ± std)", y=1.02)
    fig.savefig(os.path.join(OUT, "figs", "intrusive_metrics.png"))
    plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
