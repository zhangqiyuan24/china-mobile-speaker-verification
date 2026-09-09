#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fig_claim_tradeoff v2 — 重画:编号点+分族配色+双达标区(空)+右侧全臂图例(中文,SimHei)"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import font_manager

font_manager.fontManager.addfont("/data/home/ligangrui/china_mobile/AI/9.4/fonts/simhei.ttf")
plt.rcParams.update({
    "font.family": "SimHei", "axes.unicode_minus": False,
    "font.size": 11, "axes.edgecolor": "#888888", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
BLUE, ORANGE, GRAY, DARK, GREEN, RED, PURPLE = "#4472C4", "#ED7D31", "#7F7F7F", "#44546A", "#70AD47", "#C00000", "#7030A0"

# (编号, 名称, oracle0, claim0, marker, color, dx, dy)  dx/dy = 编号徽章偏移(点)
arms = [
    (1,  "交付前端 r5(对照)",   4.88, 10.72, "o", BLUE,   -2, 14),
    (2,  "非目标抑制 +30%",      5.41,  9.12, "s", ORANGE, -7, 14),
    (3,  "非目标抑制 +50%",      6.05,  8.39, "s", ORANGE,-18, -6),
    (4,  "+50% + 反拉项",        6.05,  8.48, "s", ORANGE, 14,  7),
    (5,  "非目标抑制 +70%",      6.69,  8.11, "s", ORANGE, 10,-14),
    (6,  "判别损失清零",          7.83,  7.84, "^", GREEN,   0, 14),
    (7,  "判别损失减半",          7.82,  7.82, "^", GREEN,  15, -3),
    (12, "中性化配方强化",        7.78,  7.79, "v", GREEN, -20, -8),
    (8,  "冒认条件混合 0.3",      4.87, 10.56, "D", RED,    11,-13),
    (9,  "冒认条件混合 0.5",      5.51,  9.46, "D", RED,    14,  4),
    (10, "混样 + 排序损失",       5.54,  9.33, "P", RED,    16,-14),
    (11, "混样配方强化",          4.79, 11.42, "X", RED,     9,  5),
    (13, "冒认静音掩蔽 + 排序",   4.73, 12.60, "*", PURPLE,  10,  4),
    (14, "能量门控(最好点)",     8.28, 14.69, "p", GRAY,   -20, -3),
]
RAW, LINE = 8.37, 5.873   # raw 基线 / oracle 30% 达标线

fig, ax = plt.subplots(figsize=(11.4, 6.2))

# 双达标区(空)
ax.add_patch(Rectangle((4.2, 6.9), LINE - 4.2, RAW - 6.9, facecolor=GREEN,
                       alpha=0.10, edgecolor=GREEN, ls="--", lw=1.6, zorder=1))
ax.text(4.32, 7.05, "双达标区(空):oracle 达标 且 claim 优于 raw\n15 个补救臂无一进入", fontsize=10,
        color=GREEN, zorder=4, bbox=dict(fc="white", ec=GREEN, alpha=0.85, boxstyle="round,pad=0.35"))

# 参考线
ax.axvline(LINE, color=BLUE, ls="--", lw=1.4, alpha=0.7)
ax.text(LINE + 0.06, 14.85, "oracle 达标线 5.87(较 raw 降 30%)", fontsize=9, color=BLUE, va="top")
ax.axhline(RAW, color=DARK, ls="--", lw=1.4, alpha=0.8)
ax.text(8.52, RAW + 0.12, "raw 基线 8.37(claim 需低于此线)", fontsize=9, color=DARK, ha="right")

# 权衡前沿:1 → 2 → 3 → 5 → 6
fx = [4.88, 5.41, 6.05, 6.69, 7.83]
fy = [10.72, 9.12, 8.39, 8.11, 7.84]
ax.plot(fx, fy, ls="--", lw=1.6, color="#B0B0B0", zorder=2)
ax.annotate("", xy=(8.05, 7.79), xytext=(7.55, 7.99),
            arrowprops=dict(arrowstyle="-|>", color="#B0B0B0", lw=1.6))
ax.text(5.78, 9.95, "权衡前沿:沿前沿滑动\noracle 越强 → claim 越差", fontsize=10, color="#666666",
        rotation=-33, ha="center", zorder=4)

# 散点 + 编号徽章
for n, lab, o, c, m, col, dx, dy in arms:
    ax.scatter(o, c, s=170 if n != 13 else 260, marker=m, color=col,
               zorder=3, edgecolors="white", linewidths=1.0)
    ax.annotate(str(n), (o, c), textcoords="offset points", xytext=(dx, dy),
                fontsize=9.5, fontweight="bold", color=col, ha="center", va="center", zorder=5,
                bbox=dict(fc="white", ec=col, boxstyle="circle,pad=0.22", lw=1.1))
ax.annotate("oracle 全场最优 4.73\nclaim 全场最差 12.60", (4.73, 12.60),
            textcoords="offset points", xytext=(14, 10), fontsize=9, color=PURPLE)
# 6/7/12 三点几乎重合的说明
ax.text(7.80, 7.30, "6·7·12 三点几乎重合", fontsize=8.5, color=GREEN, ha="center")
# 融合臂(无坐标,注释说明)
ax.text(4.28, 15.55, "15 raw+claim 分数融合(验证集调权):最优 claim 8.23 ≈ raw,无净增益(不画点)",
        fontsize=9, color=GRAY, va="top",
        bbox=dict(fc="#F2F2F2", ec=GRAY, boxstyle="round,pad=0.3"))

ax.set_xlabel("Oracle 口径 EER(交付 / 验收口径,0 dB,%)—— 越左越好", fontsize=11.5)
ax.set_ylabel("Claim 口径 EER(部署真实口径,0 dB,%)—— 越下越好", fontsize=11.5)
ax.set_title("Oracle–Claim 权衡前沿:15 种补救(13 训练侧 + 2 推理侧)无一同时过线(0 dB,3 s)",
             fontsize=12.5, pad=10)
ax.set_xlim(4.2, 8.6); ax.set_ylim(6.9, 15.8)

handles = [plt.Line2D([], [], marker=m, ls="", color=col, markersize=10,
                      label=f"{n} {lab}") for n, lab, o, c, m, col, dx, dy in arms]
leg = ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.012, 0.5),
                fontsize=9.6, frameon=True, title="编号 = 补救臂(颜色 = 路线族)", title_fontsize=10,
                borderpad=0.8, labelspacing=0.75)
leg._legend_box.align = "left"

for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("/data/home/ligangrui/china_mobile/AI/9.4/figs/fig_claim_tradeoff_v2.png", dpi=150,
            bbox_inches="tight", pad_inches=0.15)
print("saved v2")
