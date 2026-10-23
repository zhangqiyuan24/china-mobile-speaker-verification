#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Make wav samples + spectrogram/embedding figures for the four-way comparison."""
import os, json, csv, shutil, re
import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.decomposition import PCA

BASE = "/home/ligangrui/china_mobile"
SV = os.path.join(BASE, "code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa")
OUT = os.path.join(BASE, "AI/10.17")
DATA = os.path.join(OUT, "data")
FIGS = os.path.join(OUT, "figs")
WAVOUT = os.path.join(OUT, "wav")
for d in (DATA, FIGS, WAVOUT):
    os.makedirs(d, exist_ok=True)

CONDS = ["clean", "noisy", "generic", "tse"]
TITLES = {"clean": "Clean", "noisy": "Noisy 0 dB", "generic": "FullSubNet+ (original)", "tse": "TSE (ours)"}
COLORS = {"clean": "#1b1b1b", "noisy": "#e69f00", "generic": "#d55e00", "tse": "#0072b2"}
MARKERS = {"clean": "o", "noisy": "s", "generic": "v", "tse": "*"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 180, "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10, "savefig.bbox": "tight",
})

# ---------------- load summary / selected ----------------
with open(os.path.join(DATA, "summary.json"), encoding="utf-8") as f:
    summary = json.load(f)
selected = summary["selected"]
sel_ids = [x["test_id"] for x in selected]
sel_set = set(sel_ids)

# ---------------- wav path maps ----------------
def map_wavs(root):
    m = {}
    for dp, _, fs in os.walk(root):
        for fn in fs:
            if fn.endswith(".wav"):
                m[os.path.splitext(fn)[0]] = os.path.join(dp, fn)
    return m

print("[paths] indexing wavs ...", flush=True)
CLEAN = map_wavs(os.path.join(SV, "data/data_for_test/clean/3s"))
NOISY = map_wavs(os.path.join(SV, "data/data_for_test/noise_0db/3s"))
GENERIC = map_wavs(os.path.join(SV, "data/data_for_test/denoised/noise_0db/enhanced_0194"))
TSE = map_wavs(os.path.join(BASE, "AI/8.27/exp/tse_v3_ns05duraug_r5/enhanced_official/noise_0db"))
print("  counts", len(CLEAN), len(NOISY), len(GENERIC), len(TSE), flush=True)

# ---------------- enrollment ids for selected ----------------
enroll = {}
with open(os.path.join(SV, "data/data_for_test/clean/3s/trials_cross_device"), encoding="utf-8") as f:
    for line in f:
        p = line.split()
        if len(p) >= 3 and p[1] in sel_set and p[2] == "target":
            enroll.setdefault(p[1], set()).add(p[0])
for u in sel_ids:
    print("  enroll", u, len(enroll.get(u, [])), flush=True)

for u in sel_ids:
    for name, m in [("clean", CLEAN), ("noisy", NOISY), ("generic", GENERIC), ("tse", TSE)]:
        if u not in m:
            print(f"[WARN] missing {name} wav for {u}")
    d = os.path.join(WAVOUT, u)
    os.makedirs(d, exist_ok=True)
    for name, m in [("clean", CLEAN), ("noisy_0db", NOISY), ("fullsubnetplus", GENERIC), ("tse", TSE)]:
        if u in m:
            shutil.copyfile(m[u], os.path.join(d, name + ".wav"))
    for i, e in enumerate(sorted(enroll.get(u, []))[:2], 1):
        if e in CLEAN:
            shutil.copyfile(CLEAN[e], os.path.join(d, f"enroll_{i}.wav"))

# ---------------- audio helpers ----------------
SR = 16000

def load(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)

def mel_db(y):
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=512, hop_length=160, win_length=400,
        n_mels=80, fmin=20, fmax=7600, power=2.0)
    return 10.0 * np.log10(np.maximum(S, 1e-12))

def align_est(est, ref, max_lag=800):
    """Align est to ref using FFT cross-correlation; returns aligned est (same length as ref)."""
    from scipy.signal import correlate
    n = min(len(est), len(ref))
    est = est[:n]; ref = ref[:n]
    c = correlate(est, ref, mode="full", method="fft")
    lag = int(np.argmax(np.abs(c)) - (n - 1))
    if lag > 0:
        est = est[lag:]
    elif lag < 0:
        est = np.concatenate([np.zeros(-lag, dtype=est.dtype), est])
    m = min(len(est), len(ref))
    return est[:m], ref[:m], lag

def si_sdr(est, ref):
    est = est - est.mean(); ref = ref - ref.mean()
    a = float(np.dot(est, ref) / (np.dot(ref, ref) + 1e-10))
    t = a * ref
    e = est - t
    return 10.0 * np.log10((np.sum(t ** 2) + 1e-10) / (np.sum(e ** 2) + 1e-10))

def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(np.sum(a * b) / (np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-12))

# ---------------- spectrograms + mel metrics ----------------
print("[fig] spectrograms ...", flush=True)
mel_metrics = []
mels_for_bands = {c: [] for c in CONDS}
active_masks = []
showcase = selected[0]["test_id"]

for x in selected:
    u = x["test_id"]
    wavs = {
        "clean": load(os.path.join(WAVOUT, u, "clean.wav")),
        "noisy": load(os.path.join(WAVOUT, u, "noisy_0db.wav")),
        "generic": load(os.path.join(WAVOUT, u, "fullsubnetplus.wav")),
        "tse": load(os.path.join(WAVOUT, u, "tse.wav")),
    }
    n = min(len(v) for v in wavs.values())
    wavs = {k: v[:n] for k, v in wavs.items()}
    D = {k: mel_db(v) for k, v in wavs.items()}
    T = min(D[k].shape[1] for k in D)
    D = {k: v[:, :T] for k, v in D.items()}

    # active speech mask from clean: top-energy TF bins
    thr = D["clean"].max() - 35.0
    active = D["clean"] > thr
    active |= (D["clean"] > (D["clean"].max() - 45.0)) & (D["clean"] > np.median(D["clean"]) + 6.0)
    active_masks.append(active)

    for c in CONDS:
        d = D[c] - D["clean"]
        mel_metrics.append({
            "test_id": u, "condition": c,
            "mse_db": float(np.mean(d[active] ** 2)),
            "corr": pearson(D[c][active], D["clean"][active]),
            "mean_db_diff": float(np.mean(d[active])),
            "over_suppression_frac_20db": float(np.mean(d[active] < -20.0)),
            "amplification_frac_20db": float(np.mean(d[active] > 20.0)),
        })
        if c != "clean":
            est, ref, lag = align_est(wavs[c], wavs["clean"])
            mel_metrics[-1]["si_sdr_db"] = si_sdr(est, ref)
            mel_metrics[-1]["align_lag_samples"] = lag

    for c in CONDS:
        # per-band mean difference over active bins, then remove global offset
        band = np.array([D[c][b][active[b]].mean() if active[b].any() else np.nan for b in range(80)])
        band = band - np.nanmean(band)
        mels_for_bands[c].append(band)

    # 4-panel spectrogram
    vmax = float(D["clean"].max()); vmin = vmax - 70.0
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4), sharey=True)
    for ax, c in zip(axes, CONDS):
        im = ax.imshow(D[c], origin="lower", aspect="auto", cmap="magma",
                       vmin=vmin, vmax=vmax, extent=[0, T * 160 / SR, 0, 80])
        ax.set_title(TITLES[c])
        ax.set_xlabel("time (s)")
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.set_yticklabels(["0", "1k", "2.3k", "4.2k", "7.6k"])
    axes[0].set_ylabel("mel frequency (Hz)")
    fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01, label="dB")
    fig.suptitle(f"{u}   target-score: clean {x['clean']:.3f} | noisy {x['noisy']:.3f} | "
                 f"FullSubNet+ {x['generic']:.3f} | TSE {x['tse']:.3f}", y=1.04, fontsize=10)
    fig.savefig(os.path.join(FIGS, f"spectrogram_{u}.png"))
    plt.close(fig)

    if u == showcase:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
        for ax, c in zip(axes, ["noisy", "generic", "tse"]):
            d = D[c] - D["clean"]
            im = ax.imshow(np.clip(d, -25, 25), origin="lower", aspect="auto", cmap="coolwarm",
                           vmin=-25, vmax=25, extent=[0, T * 160 / SR, 0, 80])
            ax.set_title(f"{TITLES[c]} − Clean")
            ax.set_xlabel("time (s)")
            ax.set_yticks([0, 20, 40, 60, 80])
            ax.set_yticklabels(["0", "1k", "2.3k", "4.2k", "7.6k"])
        axes[0].set_ylabel("mel frequency (Hz)")
        fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01, label="dB difference")
        fig.savefig(os.path.join(FIGS, f"spectrogram_diff_{u}.png"))
        plt.close(fig)

with open(os.path.join(DATA, "mel_metrics.csv"), "w", newline="", encoding="utf-8") as f:
    fields = ["test_id", "condition", "mse_db", "corr", "mean_db_diff",
              "over_suppression_frac_20db", "amplification_frac_20db", "si_sdr_db", "align_lag_samples"]
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in mel_metrics:
        w.writerow(r)

# band-difference figure
freqs = librosa.mel_frequencies(n_mels=80, fmin=20, fmax=7600)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, subtract_offset, title in [
        (axes[0], False, "raw spectral difference"),
        (axes[1], True, "spectral shape (offset removed)")]:
    for c in ["noisy", "generic", "tse"]:
        arr = np.stack(mels_for_bands[c])
        if subtract_offset:
            arr = arr - np.nanmean(arr, axis=1, keepdims=True)
        mu = np.nanmean(arr, axis=0); sd = np.nanstd(arr, axis=0)
        ax.plot(freqs, mu, label=TITLES[c], color=COLORS[c], lw=1.8)
        ax.fill_between(freqs, mu - sd, mu + sd, color=COLORS[c], alpha=0.15)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(20, 7600)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("mean dB difference vs clean")
    ax.set_title(title)
    ax.legend(fontsize=8)
fig.suptitle("Average mel-band difference over selected utterances", y=1.02)
fig.savefig(os.path.join(FIGS, "mel_band_diff.png"))
plt.close(fig)

# ---------------- score figures ----------------
print("[fig] score figures ...", flush=True)
# selected bar chart
fig, ax = plt.subplots(figsize=(10.5, 4.2))
xx = np.arange(len(selected)); width = 0.2
for j, c in enumerate(CONDS):
    vals = [x[c] for x in selected]
    bars = ax.bar(xx + (j - 1.5) * width, vals, width, label=TITLES[c], color=COLORS[c])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(xx)
ax.set_xticklabels([f"#{i+1}\n{x['speaker'][-5:]}\n{x['device']}" for i, x in enumerate(selected)], fontsize=8)
ax.set_ylabel("cosine to clean enrollment")
ax.set_ylim(0, 0.85)
ax.set_title("Target scores of selected utterances (same enrollment, same trial protocol)")
ax.legend(ncol=4, fontsize=8, loc="upper center")
ax.grid(axis="y", alpha=0.25)
fig.savefig(os.path.join(FIGS, "target_scores_selected.png"))
plt.close(fig)

# per-utt distributions
rows = list(csv.DictReader(open(os.path.join(DATA, "per_utt_scores.csv"), encoding="utf-8")))
for r in rows:
    for c in CONDS + ["drift_noisy", "drift_generic", "drift_tse"]:
        r[c] = float(r[c])

fig, ax = plt.subplots(figsize=(9, 4.2))
bins = np.linspace(-0.1, 0.95, 70)
for c in CONDS:
    vals = np.array([r[c] for r in rows])
    ax.hist(vals, bins=bins, density=True, histtype="stepfilled", alpha=0.22, color=COLORS[c])
    ax.hist(vals, bins=bins, density=True, histtype="step", color=COLORS[c], lw=1.6,
            label=f"{TITLES[c]}  mean={vals.mean():.3f}")
    ax.axvline(vals.mean(), color=COLORS[c], ls="--", lw=1.2)
ax.set_xlabel("per-utterance mean target cosine (clean enrollment)")
ax.set_ylabel("density")
ax.set_title("Distribution of per-utterance target scores (8507 test utterances)")
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
fig.savefig(os.path.join(FIGS, "target_score_distributions.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.2))
bins = np.linspace(0.0, 1.02, 80)
for c, key in [("noisy", "drift_noisy"), ("generic", "drift_generic"), ("tse", "drift_tse")]:
    vals = np.array([r[key] for r in rows])
    ax.hist(vals, bins=bins, density=True, histtype="stepfilled", alpha=0.22, color=COLORS[c])
    ax.hist(vals, bins=bins, density=True, histtype="step", color=COLORS[c], lw=1.6,
            label=f"{TITLES[c]}  mean={vals.mean():.3f}")
    ax.axvline(vals.mean(), color=COLORS[c], ls="--", lw=1.2)
ax.axvline(1.0, color="black", ls=":", lw=1.0, label="Clean = 1.0")
ax.set_xlabel("cosine(clean embedding, condition embedding)  — same utterance")
ax.set_ylabel("density")
ax.set_title("Identity drift: how far each condition moves the ERes2Net embedding from clean")
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
fig.savefig(os.path.join(FIGS, "cos_to_clean_distributions.png"))
plt.close(fig)

# scatter noisy versus generic, colored by TSE
fig, ax = plt.subplots(figsize=(6.4, 5.6))
xv = np.array([r["noisy"] for r in rows]); yv = np.array([r["generic"] for r in rows])
cv = np.array([r["tse"] for r in rows])
sc = ax.scatter(xv, yv, c=cv, cmap="viridis", s=9, alpha=0.55, vmin=0.1, vmax=0.75)
lo = min(xv.min(), yv.min()); hi = max(xv.max(), yv.max())
ax.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1)
sel_x = [x["noisy"] for x in selected]; sel_y = [x["generic"] for x in selected]
ax.scatter(sel_x, sel_y, s=70, facecolors="none", edgecolors="red", linewidths=1.5, label="selected examples")
ax.set_xlabel("target cosine — noisy")
ax.set_ylabel("target cosine — FullSubNet+")
ax.set_title("FullSubNet+ moves target scores below noisy (below diagonal)")
ax.legend(fontsize=8, loc="upper left")
fig.colorbar(sc, ax=ax, label="target cosine — TSE")
fig.savefig(os.path.join(FIGS, "noisy_vs_generic_scatter.png"))
plt.close(fig)

# metrics summary
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
metrics = [
    ("eer_pct_computed", "EER (%)", 0),
    ("target_mean", "mean target cosine", 1),
    ("gap", "target − nontarget gap", 2),
    ("dprime", "d-prime", 3),
]
for ax, (key, title, _) in zip(axes, metrics):
    vals = [summary["summary"][c][key] for c in CONDS]
    bars = ax.bar(range(4), vals, color=[COLORS[c] for c in CONDS])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["clean", "noisy", "FullSubNet+", "TSE"], rotation=25, ha="right", fontsize=8)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
fig.suptitle("Global metrics over 1,378,206 trials", y=1.02)
fig.savefig(os.path.join(FIGS, "metrics_summary.png"))
plt.close(fig)

# ---------------- PCA trajectories ----------------
print("[fig] PCA ...", flush=True)
z = np.load(os.path.join(DATA, "selected_embeddings.npz"))
ids = list(z["ids"])
M = np.concatenate([z[f"emb_{c}"] for c in CONDS] + [z["enroll_mean"]], axis=0)
pca = PCA(n_components=2).fit(M)
proj = {c: pca.transform(z[f"emb_{c}"]) for c in CONDS}
enroll_proj = pca.transform(z["enroll_mean"])
fig, ax = plt.subplots(figsize=(7.2, 6.0))
cmap = plt.get_cmap("tab10")
order = ["clean", "noisy", "generic", "tse"]
for i, u in enumerate(ids):
    pts = np.stack([proj[c][i] for c in order])
    ax.plot(pts[:, 0], pts[:, 1], color=cmap(i % 10), lw=1.2, alpha=0.7)
    for c in order:
        ax.scatter(proj[c][i, 0], proj[c][i, 1], marker=MARKERS[c], s=90 if c == "tse" else 55,
                   color=COLORS[c], edgecolor="white", linewidth=0.5, zorder=3)
    ax.scatter(enroll_proj[i, 0], enroll_proj[i, 1], marker="D", s=80,
               facecolors="none", edgecolors=cmap(i % 10), linewidths=1.5, zorder=2)
    ax.annotate(f"#{i+1}", proj["clean"][i], xytext=(4, 4), textcoords="offset points", fontsize=8)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("ERes2Net embedding trajectories (clean → noisy → FullSubNet+ → TSE)\ndiamonds = clean enrollment mean")
handles = [plt.Line2D([0], [0], marker=MARKERS[c], color="w", markerfacecolor=COLORS[c],
                      markersize=9, label=TITLES[c]) for c in order]
ax.legend(handles=handles, fontsize=8, loc="best")
ax.grid(alpha=0.2)
fig.savefig(os.path.join(FIGS, "embedding_pca.png"))
plt.close(fig)

# selected meta json
with open(os.path.join(DATA, "selected_meta.json"), "w", encoding="utf-8") as f:
    json.dump({"showcase": showcase, "selected": selected,
               "enroll_ids": {u: sorted(enroll.get(u, [])) for u in ids},
               "note": "wavs: clean.wav, noisy_0db.wav, fullsubnetplus.wav, tse.wav; enroll_1/2.wav are clean enrollment refs"},
              f, ensure_ascii=False, indent=2)
print("[done] figures ->", FIGS, flush=True)
