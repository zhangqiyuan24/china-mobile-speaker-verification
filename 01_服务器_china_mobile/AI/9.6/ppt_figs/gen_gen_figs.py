# -*- coding: utf-8 -*-
# 9.6 扩展验证(泛化)三图:时长泛化 / 场景泛化 / 注册端鲁棒性
# 数据来源:README §2.5(时长官方表)、§2.6(场景官方表)、§6.1(注册端网格,代理验证集)
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

NAVY = "#1F3864"; BLUE = "#4472C4"; GRAY = "#A6A6A6"
DARK = "#333333"; FGRAY = "#595959"


def style_ax(ax, fs=9):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#7F7F7F")
    ax.tick_params(colors=FGRAY, labelsize=fs)
    ax.set_axisbelow(True)


def pair_bars(ax, labels, base, fe, ylim, yticks, vfs=8.8, lfs=8.5):
    x = np.arange(len(labels))
    w = 0.32; off = 0.19
    b1 = ax.bar(x - off, base, w, color=GRAY, label="Baseline")
    b2 = ax.bar(x + off, fe, w, color=BLUE, label="Delivered front-end")
    for r, v in zip(b1, base):
        ax.text(r.get_x() + r.get_width() / 2, v + ylim * 0.015, f"{v:.2f}",
                ha="center", va="bottom", color=FGRAY, fontsize=vfs, fontweight="bold")
    for r, v in zip(b2, fe):
        ax.text(r.get_x() + r.get_width() / 2, v + ylim * 0.015, f"{v:.2f}",
                ha="center", va="bottom", color=NAVY, fontsize=vfs + 0.4, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color=DARK)
    ax.set_ylim(0, ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel("EER(%)", fontsize=9.5, color=FGRAY)
    ax.legend(loc="upper right", frameon=False, fontsize=lfs, handlelength=1.4,
              borderaxespad=0.1)
    style_ax(ax)


# ---- 1) 时长泛化(测试 0 dB · 主力后端) ----
fig, ax = plt.subplots(figsize=(5.83, 4.44), dpi=200)
fig.subplots_adjust(left=0.135, right=0.97, top=0.90, bottom=0.09)
pair_bars(ax, ["1 s", "3 s", "5 s"], [18.56, 8.39, 7.86], [13.57, 4.89, 4.43],
          21.5, [0, 5, 10, 15, 20], vfs=9.2, lfs=9)
ax.set_title("语音时长泛化(测试 0 dB · 主力后端)", fontsize=12.5,
             fontweight="bold", color=NAVY, pad=9)
fig.savefig(f"{DIR}/fig_gen_duration.png")
plt.close(fig)

# ---- 2) 复杂声学场景泛化(3 s · 主力后端) ----
fig, ax = plt.subplots(figsize=(6.11, 2.33), dpi=200)
fig.subplots_adjust(left=0.088, right=0.985, top=0.86, bottom=0.235)
pair_bars(ax, ["混响", "二人重叠\n中文", "三人重叠\n中文", "二人重叠\n英文"],
          [6.09, 22.42, 28.43, 9.13], [4.87, 16.63, 18.42, 7.76],
          33, [0, 10, 20, 30], vfs=8.2, lfs=8)
ax.tick_params(axis="x", labelsize=8.8)
ax.set_title("复杂声学场景泛化(3 s · 主力后端)", fontsize=11.5,
             fontweight="bold", color=NAVY, pad=6)
fig.savefig(f"{DIR}/fig_gen_scenes.png")
plt.close(fig)

# ---- 3) 注册端鲁棒性(注册 SNR 扫描 · 测试 0 dB · 代理验证集) ----
fig, ax = plt.subplots(figsize=(6.11, 2.33), dpi=200)
fig.subplots_adjust(left=0.088, right=0.985, top=0.86, bottom=0.265)
XS = np.arange(5)
XLAB = ["clean", "15 dB", "10 dB", "5 dB", "0 dB"]
RAW = [5.50, 6.00, 6.24, 7.24, 9.00]
FE = [1.81, 2.48, 2.26, 2.52, 4.50]
ax.plot(XS, RAW, ls="--", marker="o", ms=4.5, lw=1.6, color=GRAY, label="Baseline")
ax.plot(XS, FE, ls="-", marker="o", ms=4.5, lw=2.0, color=NAVY,
        label="Delivered front-end")
for x, v in zip(XS, RAW):
    ax.text(x, v + 0.42, f"{v:.2f}", ha="center", va="bottom", color=FGRAY,
            fontsize=8.2, fontweight="bold")
for x, v in zip(XS, FE):
    ax.text(x, v - 0.42, f"{v:.2f}", ha="center", va="top", color=NAVY,
            fontsize=8.6, fontweight="bold")
ax.set_xticks(XS)
ax.set_xticklabels(XLAB, fontsize=9, color=DARK)
ax.set_ylim(0, 11)
ax.set_yticks([0, 3, 6, 9])
ax.set_ylabel("EER(%)", fontsize=9.5, color=FGRAY)
ax.set_xlabel("注册语音 SNR(测试 0 dB · 代理验证集 100 人)", fontsize=8.8, color=FGRAY)
ax.legend(loc="upper left", frameon=False, fontsize=8.2, handlelength=1.8,
          borderaxespad=0.1)
style_ax(ax)
ax.set_title("注册端鲁棒性(注册语音质量扫描)", fontsize=11.5,
             fontweight="bold", color=NAVY, pad=6)
fig.savefig(f"{DIR}/fig_gen_enroll.png")
plt.close(fig)

print("DONE gen figs x3")
