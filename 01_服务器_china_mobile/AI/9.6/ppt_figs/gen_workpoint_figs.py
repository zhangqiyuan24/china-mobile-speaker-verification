# -*- coding: utf-8 -*-
# 9.6 工作点验证图(纯 CPU):minDCF / FRR@FAR5% / DET 曲线 / 失败案例 Top5
# 数据来源:交付 06_指标/roc_det/summary.json 与 4 条 curve.csv、失败案例_failures.json、README §6.2
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

DIR = "/data/home/ligangrui/china_mobile/AI/9.6/ppt_figs"
for f in ["msyh.ttc", "msyhbd.ttc"]:
    fm.fontManager.addfont(f"{DIR}/{f}")
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

NAVY = "#1F3864"; BLUE = "#4472C4"; LBLUE = "#8FAADC"; GRAY = "#A6A6A6"
DARK = "#333333"; FGRAY = "#595959"; RED = "#C00000"
O_RAW = "#F4B183"; O_FE = "#C55A11"


def style_ax(ax, fs=10):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#7F7F7F")
    ax.tick_params(colors=FGRAY, labelsize=fs)
    ax.set_axisbelow(True)


def pair_bars(ax, labels, base, fe, ylim, yticks, fmt, ylab, vfs=9.5):
    x = list(range(len(labels)))
    w = 0.30; off = 0.17
    b1 = ax.bar([i - off for i in x], base, w, color=GRAY, label="Baseline")
    b2 = ax.bar([i + off for i in x], fe, w, color=BLUE, label="Delivered front-end")
    for r, v in zip(b1, base):
        ax.text(r.get_x() + r.get_width() / 2, v + ylim * 0.018, fmt.format(v),
                ha="center", va="bottom", color=FGRAY, fontsize=vfs, fontweight="bold")
    for r, v in zip(b2, fe):
        ax.text(r.get_x() + r.get_width() / 2, v + ylim * 0.018, fmt.format(v),
                ha="center", va="bottom", color=NAVY, fontsize=vfs + 0.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color=DARK)
    ax.set_ylim(0, ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylab, fontsize=10.5, color=FGRAY)
    style_ax(ax)
    return b1, b2


GL = ["主力 0 dB", "主力 −5 dB", "经典 0 dB", "经典 −5 dB"]

fig, axes = plt.subplots(2, 2, figsize=(12.4, 5.6), dpi=200,
                         gridspec_kw=dict(width_ratios=[1, 1], height_ratios=[1, 1],
                                          wspace=0.26, hspace=0.52))
fig.subplots_adjust(left=0.055, right=0.985, top=0.925, bottom=0.105)

# (a) minDCF
M_B = [0.628, 0.687, 0.722, 0.773]; M_F = [0.515, 0.557, 0.580, 0.622]
b1, b2 = pair_bars(axes[0][0], GL, M_B, M_F, 1.05, [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                   "{:.3f}", "minDCF")
axes[0][0].legend(handles=[b1, b2], loc="upper left", frameon=False, fontsize=9.5)
axes[0][0].set_title("minDCF(部署代价 · p_target = 1%)", fontsize=12,
                     fontweight="bold", color=NAVY, pad=8)

# (b) FRR @ FAR = 5%
F_B = [12.46, 18.07, 18.91, 25.75]; F_F = [4.77, 6.20, 7.81, 9.62]
pair_bars(axes[0][1], GL, F_B, F_F, 30, [0, 10, 20, 30], "{:.2f}", "FRR(%)")
axes[0][1].set_title("FRR @ FAR = 5%(固定阈值工作点)", fontsize=12,
                     fontweight="bold", color=NAVY, pad=8)

# (c) DET 曲线(主力后端,全阈值扫描,log-log)
ax = axes[1][0]
CURVES = [
    ("main_eres_0db_raw", "0 dB · Baseline", GRAY, (0, (5, 3)), 1.7),
    ("main_eres_0db_fe", "0 dB · Delivered", NAVY, "solid", 2.0),
    ("main_eres_m5db_raw", "−5 dB · Baseline", O_RAW, (0, (5, 3)), 1.7),
    ("main_eres_m5db_fe", "−5 dB · Delivered", O_FE, "solid", 2.0),
]
for name, lab, c, ls, lw in CURVES:
    d = np.loadtxt(f"{DIR}/{name}_curve.csv", skiprows=1, delimiter=",")
    far = d[:, 1] * 100.0; fnr = d[:, 2] * 100.0
    m = (far >= 0.05) & (fnr >= 0.05)
    ax.plot(far[m], fnr[m], ls=ls, color=c, lw=lw, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.05, 100); ax.set_ylim(0.05, 100)
ax.set_xticks([0.1, 1, 10, 100]); ax.set_yticks([0.1, 1, 10, 100])
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:g}"))
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:g}"))
ax.minorticks_off()
ax.set_xlabel("FAR(%)", fontsize=10, color=FGRAY)
ax.set_ylabel("FRR(%)", fontsize=10, color=FGRAY)
ax.legend(loc="lower left", frameon=False, fontsize=8.8, handlelength=2.4)
style_ax(ax)
ax.set_title("DET 曲线 · 全阈值扫描(主力后端)", fontsize=12, fontweight="bold",
             color=NAVY, pad=8)

# (d) 失败案例:最困难说话人 Top5(raw 0 dB 漏报率)
ax = axes[1][1]
SPK = ["06207", "06851", "06820", "06852", "06157"]
RT = [45.6, 43.8, 38.8, 37.5, 34.2]
CNT = ["197/432", "63/144", "141/363", "136/363", "77/225"]
ys = list(range(len(SPK)))
bars = ax.barh(ys, RT, 0.52, color=GRAY)
for y, v, c in zip(ys, RT, CNT):
    ax.text(v + 1.2, y, f"{v:.1f}", ha="left", va="center", color=NAVY,
            fontsize=10, fontweight="bold")
    ax.text(v + 7.5, y, c, ha="left", va="center", color=FGRAY, fontsize=8.5)
ax.set_yticks(ys)
ax.set_yticklabels(SPK, fontsize=10, color=DARK)
ax.invert_yaxis()
ax.set_xlim(0, 62)
ax.set_xticks([0, 20, 40, 60])
ax.set_xlabel("漏报率(%)(漏报 / 该说话人目标 trial 数)", fontsize=9.5, color=FGRAY)
style_ax(ax)
ax.set_title("失败案例 · 最困难说话人 Top 5(raw 0 dB)", fontsize=12,
             fontweight="bold", color=NAVY, pad=8)

fig.savefig(f"{DIR}/fig_workpoint.png")
plt.close(fig)
print("DONE workpoint fig")
