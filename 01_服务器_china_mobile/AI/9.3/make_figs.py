#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""9.3 汇报图表生成 (CPU only, matplotlib Agg, English labels — server has no CJK fonts).
输出: AI/9.3/figs/*.png  (150dpi, 演示文稿配色)"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "/data/home/ligangrui/china_mobile/AI/9.3/figs"
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GRAY, DARK, GREEN, RED = "#4472C4", "#ED7D31", "#A5A5A5", "#44546A", "#70AD47", "#C00000"
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": "#888888", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# ---------- 1. 增量消融链 ----------
stages = ["Raw\n(no ENH)", "+ Speaker\npreserve", "+ Non-target\nsuppress",
          "+ Hard\nround 1", "+ Hard\nround 2", "+ Hard\nround 3\n(delivered\non this BE)", "+ Hard\nround 4\n(stopped)"]
e0 = [7.614, 7.293, 6.519, 5.766, 5.467, 5.235, 6.165]
e5 = [9.715, 9.202, 8.173, 6.972, 6.462, 6.098, 7.601]
fig, ax = plt.subplots(figsize=(10.5, 4.4))
x = np.arange(len(stages))
ax.plot(x, e0, "o-", color=BLUE, lw=2.2, ms=7, label="0 dB")
ax.plot(x, e5, "s-", color=ORANGE, lw=2.2, ms=7, label="-5 dB")
for i, (a, b) in enumerate(zip(e0, e5)):
    ax.annotate(f"{a:.2f}", (i, a), textcoords="offset points", xytext=(0, -16), ha="center", color=BLUE, fontsize=9)
    ax.annotate(f"{b:.2f}", (i, b), textcoords="offset points", xytext=(0, 9), ha="center", color=ORANGE, fontsize=9)
ax.axhline(7.614 * 0.7, color=BLUE, ls="--", lw=1, alpha=0.55)
ax.axhline(9.715 * 0.7, color=ORANGE, ls="--", lw=1, alpha=0.55)
ax.text(6.45, 7.614 * 0.7, "30% line (0 dB)", va="center", fontsize=8.5, color=BLUE)
ax.text(6.45, 9.715 * 0.7, "30% line (-5 dB)", va="center", fontsize=8.5, color=ORANGE)
ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=8.8)
ax.set_ylabel("EER (%)"); ax.set_ylim(4.5, 10.6)
ax.set_title("Incremental ablation: one variable added per stage (official chain, frozen-averaged backend)", fontsize=11.5)
ax.legend(loc="lower left", framealpha=0.9); style_ax(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_staircase.png", dpi=150); plt.close(fig)

# ---------- 2. 三后端交付对比 ----------
backs = ["ERes2Net-large\n(main, 29.1M)", "ECAPA-TDNN\n(20.8M)", "ReDimNet2-B6\n(light, 12.7M)"]
raw0, fe0 = [8.39, 11.15, 9.94], [4.89, 6.32, 7.58]
raw5, fe5 = [11.06, 14.36, 13.03], [5.56, 7.13, 9.35]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
for ax, raw, fe, t in ((axes[0], raw0, fe0, "0 dB, 3 s"), (axes[1], raw5, fe5, "-5 dB, 3 s")):
    x = np.arange(3); w = 0.36
    b1 = ax.bar(x - w/2, raw, w, color=GRAY, label="Raw (baseline)")
    b2 = ax.bar(x + w/2, fe, w, color=BLUE, label="With TSE front-end")
    for xi, (r, f) in enumerate(zip(raw, fe)):
        ax.annotate(f"-{(r-f)/r*100:.1f}%", (xi + w/2, f), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9.5, color=RED, fontweight="bold")
        ax.annotate(f"{r:.2f}", (xi - w/2, r), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8.5, color="#555555")
        ax.annotate(f"{f:.2f}", (xi + w/2, f), textcoords="offset points", xytext=(14, -2), ha="center", fontsize=8.5, color=BLUE)
    ax.set_xticks(x); ax.set_xticklabels(backs, fontsize=9)
    ax.set_title(t); ax.set_ylabel("EER (%)"); ax.set_ylim(0, max(raw) * 1.25)
    ax.legend(fontsize=9, loc="upper left"); style_ax(ax)
fig.suptitle("Delivered front-end vs. untouched backbones (acceptance protocol, 3 s)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{OUT}/fig_backends.png", dpi=150); plt.close(fig)

# ---------- 3. 时长轴 ----------
dur = ["1 s", "3 s", "5 s"]
d_raw0, d_fe0 = [18.56, 8.39, 7.86], [13.57, 4.89, 4.43]
d_raw5, d_fe5 = [21.44, 11.06, 10.08], [14.74, 5.56, 4.90]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
for ax, raw, fe, t in ((axes[0], d_raw0, d_fe0, "0 dB"), (axes[1], d_raw5, d_fe5, "-5 dB")):
    x = np.arange(3); w = 0.36
    ax.bar(x - w/2, raw, w, color=GRAY, label="Raw")
    ax.bar(x + w/2, fe, w, color=BLUE, label="With front-end")
    for xi, (r, f) in enumerate(zip(raw, fe)):
        ax.annotate(f"-{(r-f)/r*100:.1f}%", (xi + w/2, f), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9.5, color=RED, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(dur); ax.set_title(f"ERes2Net-large, {t}")
    ax.set_ylabel("EER (%)"); ax.set_ylim(0, max(raw) * 1.22); ax.legend(fontsize=9); style_ax(ax)
fig.suptitle("Utterance-duration axis (random 1/2/3 s segment augmentation during training)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"{OUT}/fig_duration.png", dpi=150); plt.close(fig)

# ---------- 4. 噪声域 ----------
doms = ["Main pool\n(free-sound)", "Music", "Sound-\nbible", "Speech\n(competing\nspeakers)"]
n_raw0 = [8.39, 9.64, 8.13, 8.63]; n_fe0 = [4.89, 5.79, 4.95, 7.47]
n_raw5 = [11.06, 13.29, 10.16, 10.83]; n_fe5 = [5.56, 7.34, 5.52, 10.92]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
for ax, raw, fe, t in ((axes[0], n_raw0, n_fe0, "0 dB"), (axes[1], n_raw5, n_fe5, "-5 dB")):
    x = np.arange(4); w = 0.36
    ax.bar(x - w/2, raw, w, color=GRAY, label="Raw")
    ax.bar(x + w/2, fe, w, color=BLUE, label="With front-end")
    for xi, (r, f) in enumerate(zip(raw, fe)):
        c = GREEN if f < r * 0.7 else (RED if f > r else ORANGE)
        ax.annotate(f"-{(r-f)/r*100:.1f}%" if f < r else f"+{(f-r)/r*100:.1f}%",
                    (xi + w/2, f), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9, color=c, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(doms, fontsize=8.8); ax.set_title(t)
    ax.set_ylabel("EER (%)"); ax.set_ylim(0, max(raw) * 1.25); ax.legend(fontsize=9); style_ax(ax)
fig.suptitle("Noise-domain generalization (unseen noise files, ERes2Net-large, 3 s)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"{OUT}/fig_noise_domains.png", dpi=150); plt.close(fig)

# ---------- 5. claim 权衡散点 ----------
arms = [  # (label, oracle0, claim0, marker, color)
    ("Delivered r5", 4.88, 10.72, "o", BLUE),
    ("+30% non-tgt", 5.41, 9.12, "s", ORANGE),
    ("+50% non-tgt", 6.05, 8.39, "s", ORANGE),
    ("+50%+pullback", 6.05, 8.48, "v", ORANGE),
    ("+70% non-tgt", 6.69, 8.11, "s", ORANGE),
    ("CE off", 7.83, 7.84, "^", GREEN),
    ("CE halved", 7.82, 7.82, "^", GREEN),
    ("Impostor-mix 0.3", 4.87, 10.56, "D", RED),
    ("Impostor-mix 0.5", 5.51, 9.46, "D", RED),
    ("Mix+rank", 5.54, 9.33, "P", RED),
    ("Mix-strong", 4.79, 11.42, "X", RED),
    ("Neutral-strong", 7.78, 7.79, "v", GREEN),
    ("Silence-mask+rank", 4.73, 12.60, "*", PURPLE := "#7030A0"),
]
raw0db = 8.37
fig, ax = plt.subplots(figsize=(9.6, 6.2))
for lab, o, c, m, col in arms:
    ax.scatter(o, c, s=130, marker=m, color=col, zorder=3, edgecolors="white", linewidths=0.8)
    ax.annotate(lab, (o, c), textcoords="offset points", xytext=(7, 5), fontsize=8.6, color="#333333")
ax.axhline(raw0db, color=GRAY, ls="--", lw=1.4)
ax.text(4.55, raw0db + 0.12, "raw baseline, 0 dB (8.37): claim must get below this line", fontsize=9, color="#555555")
ax.plot([4.4, 8.2], [4.4, 8.2], ls=":", color="#BBBBBB", lw=1.2)
ax.text(6.55, 6.28, "y = x (claim = oracle)", rotation=24, fontsize=8.5, color="#999999")
ax.annotate("", xy=(4.8, 8.0), xytext=(4.8, 12.8),
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
ax.text(4.92, 11.9, "impersonation leakage:\nstronger oracle  ->  worse claim", fontsize=9.5, color=RED)
ax.set_xlabel("Oracle-conditioned EER, 0 dB (%)  [delivery metric]")
ax.set_ylabel("Claim-conditioned EER, 0 dB (%)  [deployment metric]")
ax.set_title("Oracle-claim trade-off frontier: 13 remediation arms, none passes both", fontsize=11.5)
ax.set_xlim(4.3, 8.6); ax.set_ylim(6.8, 13.4); style_ax(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_claim_tradeoff.png", dpi=150); plt.close(fig)

# ---------- 6. 模型架构图 ----------
fig, ax = plt.subplots(figsize=(12.2, 5.0))
ax.set_xlim(0, 122); ax.set_ylim(0, 50); ax.axis("off"); ax.grid(False)

def box(x, y, w, h, text, fc="#EAF0FA", ec=BLUE, fs=9.5, tc="#222222", lw=1.4, rounding=1.6):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.3,rounding_size={rounding}",
                       fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3)

def arrow(x1, y1, x2, y2, color=DARK, style="-|>", lw=1.6, conn="arc3,rad=0.0"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                        color=color, lw=lw, connectionstyle=conn, zorder=1)
    ax.add_patch(a)

# 上路: 注册语音 → 声纹嵌入 → FiLM 条件
box(2, 33, 18, 10, "Enrollment speech\n(clean, 2 utts)", fc="#FFF3E8", ec=ORANGE)
box(26, 33, 20, 10, "Frozen SV backend\n(speaker embedding, 512-d)", fc="#F2F2F2", ec=GRAY)
box(52, 33, 16, 10, "SpeakerFiLM\naffine: (γ, β)", fc="#FFF3E8", ec=ORANGE)
# 下路: 带噪语音 → FullSubNet+
box(2, 8, 18, 10, "Noisy test speech\n(0 dB / -5 dB)")
box(26, 8, 20, 10, "FullSubNet+\nfull-band 512 (LSTM)\n+ sub-band 384", fc="#EAF0FA", ec=BLUE)
box(52, 8, 16, 10, "cIRM mask\n→ enhanced speech", fc="#EAF0FA", ec=BLUE)
box(74, 8, 20, 10, "Frozen SV backend\n→ enhanced embedding", fc="#F2F2F2", ec=GRAY)
box(100, 8, 18, 10, "Cosine score\nvs enrollment emb.", fc="#EAF0FA", ec=GREEN, ec2=None) if False else box(100, 8, 18, 10, "Cosine score\nvs enrollment", fc="#EAF0FA", ec=GREEN)
arrow(20, 38, 26, 38); arrow(46, 38, 52, 38)
arrow(60, 33, 40, 18, conn="arc3,rad=0.25")  # FiLM 注入 FullSubNet+
ax.text(51, 27.5, "FiLM conditioning\n(528K params, last layer zero-init)", fontsize=8.8, color=ORANGE, ha="center")
arrow(20, 13, 26, 13); arrow(46, 13, 52, 13); arrow(68, 13, 74, 13); arrow(94, 13, 100, 13)
# FiLM 也调制 mask 应用
ax.text(2, 2.5, "Trainable: FullSubNet+ + SpeakerFiLM = 9.20M params   |   SV backend frozen (inference reuses the same backend for scoring)",
        fontsize=9.2, color=DARK, ha="left")
ax.set_title("Delivered front-end: enrollment-conditioned target-speaker extraction (VoiceFilter-style)", fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_arch.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------- 7. 研究路线图 ----------
fig, ax = plt.subplots(figsize=(12.4, 4.6))
ax.set_xlim(0, 124); ax.set_ylim(0, 46); ax.axis("off"); ax.grid(False)
box(1, 26, 20, 12, "Route A\nFrozen-layer backend\nfine-tune\n6/8 conditions WORSE\n(8.39 → 9.71)", fc="#FDECEC", ec=RED, fs=8.8)
box(1, 8, 20, 12, "Route B\nGeneric denoiser\nfront-end\n0 dB: 10.11 vs noisy 8.39\nidentity features erased", fc="#FDECEC", ec=RED, fs=8.8)
box(27, 17, 17, 12, "KEY INSIGHT\nsingle-speaker noise\n= interfering SPEECH;\ndenoiser must know\nWHOSE voice to keep", fc="#FFF7E0", ec="#BF9000", fs=8.8, lw=2.0)
box(50, 30, 22, 11, "Enrollment-conditioned\nTSE front-end (FiLM)\nzero-training fusion: 7.38 / 9.32", fc="#EAF0FA", ec=BLUE, fs=8.6)
box(50, 17, 22, 11, "+ speaker-preserve loss\n+ non-target suppression\n7.29 / 9.20  →  6.52 / 8.17", fc="#EAF0FA", ec=BLUE, fs=8.6)
box(50, 4, 22, 11, "hard-example rounds ×3\n5.24 / 6.10\n(round 4 regresses → stop)", fc="#EAF0FA", ec=BLUE, fs=8.6)
box(78, 17, 21, 11, "+ random 1/2/3 s\nduration augmentation\n= DELIVERED front-end", fc="#E2F0D9", ec=GREEN, fs=8.8, lw=2.0)
box(104, 17, 18, 11, "0 dB 4.89  (-41.7%)\n-5 dB 5.56 (-49.7%)\nvs untouched backbone", fc="#E2F0D9", ec=GREEN, fs=8.8, lw=2.0)
arrow(21, 32, 27, 26, conn="arc3,rad=-0.2"); arrow(21, 14, 27, 20, conn="arc3,rad=0.2")
arrow(44, 23, 50, 32, conn="arc3,rad=-0.25")
arrow(61, 30, 61, 28, color=BLUE); arrow(61, 17, 61, 15, color=BLUE)
arrow(72, 9.5, 78, 20, conn="arc3,rad=0.2")
arrow(99, 22.5, 104, 22.5, color=GREEN, lw=2.0)
ax.set_title("How the method was found: two falsified routes and the pivot to enrollment-conditioned extraction", fontsize=11.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_roadmap.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------- 8. 参数量 ----------
names = ["MoE-lite branch\n(rejected)", "ERes2Net-large\n(ckpt w/ head)", "soup5 verifier", "ECAPA-TDNN",
         "ReDimNet2-B6", "WeSep front-end\n(rejected)", "FullSubNet+ base", "TSE front-end\n(DELIVERED)"]
vals = [50.54, 29.06, 29.06, 20.81, 12.72, 27.63, 8.68, 9.20]
cols = [GRAY, DARK, DARK, DARK, DARK, GRAY, GRAY, GREEN]
fig, ax = plt.subplots(figsize=(9.8, 4.6))
y = np.arange(len(names))
ax.barh(y, vals, color=cols, height=0.62)
for yi, v in zip(y, vals):
    ax.text(v + 0.7, yi, f"{v:.2f} M", va="center", fontsize=9.3, color="#333333")
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
ax.invert_yaxis(); ax.set_xlabel("Parameters (M)")
ax.set_title("Parameter census (measured from checkpoints, 2026-09-03)", fontsize=11.5)
ax.set_xlim(0, 58); style_ax(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_params.png", dpi=150); plt.close(fig)

# ---------- 9. 场景 ----------
sc = ["Reverb\n(T60~0.6s)", "2-spk overlap\nChinese", "3-spk overlap\nChinese", "2-spk overlap\nEnglish"]
s_raw = [6.09, 22.42, 28.43, 9.13]; s_fe = [4.87, 16.63, 18.42, 7.76]
fig, ax = plt.subplots(figsize=(9.8, 4.2))
x = np.arange(4); w = 0.36
ax.bar(x - w/2, s_raw, w, color=GRAY, label="Raw")
ax.bar(x + w/2, s_fe, w, color=BLUE, label="With front-end")
for xi, (r, f) in enumerate(zip(s_raw, s_fe)):
    ax.annotate(f"-{(r-f)/r*100:.1f}%", (xi + w/2, f), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=9.5, color=RED, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(sc, fontsize=9)
ax.set_ylabel("EER (%)"); ax.set_ylim(0, 33); ax.legend(fontsize=9)
ax.set_title("Other scenes (delivered front-end, ERes2Net-large, 3 s)", fontsize=11.5)
style_ax(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_scenes.png", dpi=150); plt.close(fig)

print("done:", sorted(os.listdir(OUT)))
