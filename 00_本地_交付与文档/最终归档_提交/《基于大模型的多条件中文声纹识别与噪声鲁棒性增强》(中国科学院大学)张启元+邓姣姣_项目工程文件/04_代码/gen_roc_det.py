#!/usr/bin/env python
# gen_roc_det.py — #21 ROC/DET + FAR/FRR 后处理（8.31，纯 CPU）
# 输入：各评测格 trials_cross_device.score（enrol test label score）
# 输出：每格曲线 CSV + EER 阈值 FAR/FRR + 固定工作点；raw/FE 成对 ROC+DET 图；汇总 JSON
# 校验：自算 EER 与官方 result.metrics 对账（>0.5pp 报警）
import os, json, glob

AI = "/data/home/ligangrui/china_mobile/AI"
OUT = f"{AI}/8.30/roc_det"
os.makedirs(OUT, exist_ok=True)

import numpy as np
try:
    from scipy.stats import norm as ndist
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_scores(path):
    lab, sc = [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 4:
                continue
            lab.append(1 if p[2] == "target" else 0)
            sc.append(float(p[3]))
    y = np.array(lab); s = np.array(sc)
    if y.mean() == 0 or y.mean() == 1:
        raise RuntimeError("单标签")
    # 方向校准：分数越高应越像 target
    if s[y == 1].mean() < s[y == 0].mean():
        s = -s
    return y, s

def sweep(y, s, npts=2001):
    qs = np.unique(np.quantile(s, np.linspace(0.0001, 0.9999, npts)))
    s_tar = s[y == 1]; s_non = s[y == 0]
    far = np.array([(s_non >= t).mean() for t in qs])
    frr = np.array([(s_tar < t).mean() for t in qs])
    return qs, far, frr

def at_x(x, y, target):
    j = int(np.argmin(np.abs(x - target)))
    return float(y[j])

def analyze(name, score_path, official_eer=None):
    y, s = load_scores(score_path)
    qs, far, frr = sweep(y, s)
    i = int(np.argmin(np.abs(far - frr)))
    eer = float((far[i] + frr[i]) / 2) * 100
    flag = ""
    if official_eer is not None and abs(eer - official_eer) > 0.5:
        flag = f" [WARN 与官方 {official_eer} 偏差>0.5pp]"
    idx = np.linspace(0, len(qs) - 1, 201).astype(int)
    with open(f"{OUT}/{name}_curve.csv", "w") as f:
        f.write("threshold,far,frr\n")
        for j in idx:
            f.write(f"{qs[j]:.6f},{far[j]:.6f},{frr[j]:.6f}\n")
    res = dict(name=name, n=len(y), tar=int(y.sum()), non=int(len(y) - y.sum()),
               eer=round(eer, 4), eer_thr=round(float(qs[i]), 6),
               far_at_frr1=round(at_x(frr, far, 0.01) * 100, 3),
               far_at_frr5=round(at_x(frr, far, 0.05) * 100, 3),
               frr_at_far1=round(at_x(far, frr, 0.01) * 100, 3),
               frr_at_far5=round(at_x(far, frr, 0.05) * 100, 3),
               warn=flag)
    np.save(f"{OUT}/{name}_curve.npy", np.vstack([qs, far, frr]))
    return res

def plot_pair(title, fname, curves):
    # curves: [(label, qs, far, frr, color)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, qs, far, frr, c in curves:
        axes[0].plot(far * 100, (1 - frr) * 100, c=c, label=label)
    axes[0].set_xlabel("FAR (%)"); axes[0].set_ylabel("1-FRR = TPR (%)")
    axes[0].set_xscale("log"); axes[0].set_title(f"ROC · {title}"); axes[0].legend(); axes[0].grid(alpha=.3)
    for label, qs, far, frr, c in curves:
        if HAVE_SCIPY:
            xx = np.clip(far, 1e-5, 1); yy = np.clip(frr, 1e-5, 1)
            axes[1].plot(ndist.ppf(xx), ndist.ppf(yy), c=c, label=label)
            axes[1].set_xlabel("FAR (normal deviate)"); axes[1].set_ylabel("FRR (normal deviate)")
        else:
            axes[1].plot(far * 100, frr * 100, c=c, label=label)
            axes[1].set_xlabel("FAR (%)"); axes[1].set_ylabel("FRR (%)")
            axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_title(f"DET · {title}"); axes[1].legend(); axes[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}.png", dpi=150); plt.close(fig)

def eer_of(metrics_path):
    try:
        for ln in open(metrics_path):
            if "EER" in ln:
                return float(ln.split("=")[1])
    except Exception:
        return None
    return None

PB = f"{AI}/8.27/exp/official_pipeline_3backends"
FE = f"{AI}/8.28/final_eval_r5"
SC = f"{AI}/8.28/scenes_r5/eres2net"

CELLS = {
    "main_eres_0db":  (f"{PB}/eres2net/raw_noise_0db",  f"{FE}/r5_eres_noise_0db_3s"),
    "main_eres_m5db": (f"{PB}/eres2net/raw_noise_m5db", f"{FE}/r5_eres_noise_m5db_3s"),
    "main_ecapa_0db": (f"{PB}/ecapa/raw_noise_0db",     f"{FE}/r5_ecapa_noise_0db_3s"),
    "main_ecapa_m5db":(f"{PB}/ecapa/raw_noise_m5db",    f"{FE}/r5_ecapa_noise_m5db_3s"),
    "scene_reverb":   (f"{SC}/raw_reverb_3s",           f"{SC}/fe_reverb_3s"),
    "scene_ov2cn":    (f"{SC}/raw_overlap2_cn_3s",      f"{SC}/fe_overlap2_cn_3s"),
    "scene_ov2en":    (f"{SC}/raw_overlap2_en_3s",      f"{SC}/fe_overlap2_en_3s"),
    "scene_ov3cn":    (f"{SC}/raw_overlap3_cn_3s",      f"{SC}/fe_overlap3_cn_3s"),
}

summary = []
for key, (rawd, fed) in CELLS.items():
    pair = []
    for mode, d in (("raw", rawd), ("fe", fed)):
        sp = glob.glob(f"{d}/*.score")
        if not sp:
            print(f"[miss] {key}/{mode}: 无 score 文件"); continue
        try:
            r = analyze(f"{key}_{mode}", sp[0], eer_of(f"{d}/result.metrics"))
            summary.append(r)
            arr = np.load(f"{OUT}/{key}_{mode}_curve.npy")
            pair.append((f"{mode} EER={r['eer']:.2f}", arr[0], arr[1], arr[2], ("#888" if mode == "raw" else "#c00")))
            print(f"[ok] {key}/{mode}: EER={r['eer']}{r['warn']}")
        except Exception as e:
            print(f"[fail] {key}/{mode}: {e}")
    if len(pair) == 2:
        plot_pair(key, key, pair)

with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)
print(f"[gen-roc-det done] {OUT} （DET 轴: scipy={'yes' if HAVE_SCIPY else 'no(log-log替代)'}）")
