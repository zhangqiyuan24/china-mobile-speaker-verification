# -*- coding: utf-8 -*-
# 9.5 重画 PPT 结果图(纯 CPU,无 GPU)  v4
# 变更:去 ↓% 标注 / 去 30% 线文字(留虚线)/ 去横向网格线 / 面板标题居中且明确表述 / 全图文字缩小一号
# 数据来源:E:\china_mobile_review\交付材料_9.3\03_结果汇总\README_实验结果汇总.md(§2.5/§2.6/工作点表/§8)
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

DIR = "/data/home/ligangrui/china_mobile/AI/9.5/ppt_figs"
for f in ["msyh.ttc", "msyhbd.ttc"]:
    fm.fontManager.addfont(f"{DIR}/{f}")
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

NAVY = "#1F3864"; BLUE = "#4472C4"; LBLUE = "#8FAADC"; GRAY = "#A6A6A6"
DARK = "#333333"; FGRAY = "#595959"; RED = "#C00000"


def style_ax(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#7F7F7F")
    ax.tick_params(colors=FGRAY, labelsize=11)
    ax.set_axisbelow(True)


def group_bars(ax, labels, base, fe, ylim, yticks, tick_fs=11):
    """灰柱=Baseline(数值柱顶上方,深灰);蓝柱=Delivered front-end(数值柱顶上方,navy)。"""
    x = list(range(len(labels)))
    w = 0.28; off = 0.185
    b1 = ax.bar([i - off for i in x], base, w, color=GRAY, label="Baseline")
    b2 = ax.bar([i + off for i in x], fe, w, color=BLUE, label="Delivered front-end")
    for r, v in zip(b1, base):
        ax.text(r.get_x() + r.get_width() / 2, v + ylim * 0.018, f"{v:.2f}",
                ha="center", va="bottom", color=FGRAY, fontsize=10, fontweight="bold")
    for r, v in zip(b2, fe):
        ax.text(r.get_x() + r.get_width() / 2, v + ylim * 0.018, f"{v:.2f}",
                ha="center", va="bottom", color=NAVY, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=tick_fs, color=DARK)
    ax.set_ylim(0, ylim)
    ax.set_yticks(yticks)
    style_ax(ax)
    return b1, b2


# ============ 图 A:第18页 = 三面板(0dB / −5dB / 相对降幅vs30%线) ============
BK = ["ERes2Net-large\n(29.1M)", "ECAPA-TDNN\n(20.8M)", "ReDimNet2-B6\n(12.7M)"]
BKS = ["ERes2Net-L", "ECAPA", "ReDimNet-B6"]
BASE0 = [8.39, 11.15, 9.94];  FE0 = [4.89, 6.32, 7.58]
BASE5 = [11.06, 14.36, 13.03]; FE5 = [5.56, 7.13, 9.35]
D0 = [41.7, 43.3, 23.7]; D5 = [49.7, 50.3, 28.2]

fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), dpi=200,
                         gridspec_kw=dict(width_ratios=[1, 1, 0.95], wspace=0.30))
fig.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.175)

b1, b2 = group_bars(axes[0], BK, BASE0, FE0, 12.5, [0, 4, 8, 12])
axes[0].set_title("信噪比 0 dB · 语音 3 s", fontsize=13, fontweight="bold",
                  color=NAVY, pad=10)
axes[0].set_ylabel("EER (%)", fontsize=11.5, color=FGRAY)

group_bars(axes[1], BK, BASE5, FE5, 16.5, [0, 4, 8, 12, 16])
axes[1].set_title("信噪比 −5 dB · 语音 3 s", fontsize=13, fontweight="bold",
                  color=NAVY, pad=10)
axes[1].set_ylabel("EER (%)", fontsize=11.5, color=FGRAY)

# 面板3:相对降幅,30% 验收线只画虚线不写文字
ax3 = axes[2]
xx = list(range(3)); w = 0.30
r1 = ax3.bar([i - 0.17 for i in xx], D0, w, color=BLUE, label="0 dB")
r2 = ax3.bar([i + 0.17 for i in xx], D5, w, color=LBLUE, label="−5 dB")
for r, v in list(zip(r1, D0)) + list(zip(r2, D5)):
    _off = 1.2
    if v + _off < 30 < v + _off + 4.5:   # 数值标签会骑在验收线上 → 抬到线上方
        _off = 3.2
    ax3.text(r.get_x() + r.get_width() / 2, v + _off, f"{v:.1f}", ha="center", va="bottom",
             color=NAVY, fontsize=10.5, fontweight="bold")
ax3.axhline(30, color=RED, ls=(0, (5, 3)), lw=1.6)
ax3.set_xticks(xx)
ax3.set_xticklabels(BKS, fontsize=11, color=DARK)
ax3.set_ylim(0, 62)
ax3.set_yticks([0, 20, 40, 60])
style_ax(ax3)
ax3.set_title("相对降幅 vs 30% 验收线", fontsize=13, fontweight="bold", color=NAVY, pad=10)
ax3.set_ylabel("相对降幅 (%)", fontsize=11.5, color=FGRAY)

fig.legend(handles=[b1, b2], loc="upper center", ncol=2, frameon=False,
           fontsize=12.5, bbox_to_anchor=(0.5, 1.0))
fig.savefig(f"{DIR}/fig_ppt_main3.png")
plt.close(fig)

# ============ 图 B:第19页左 = 时长轴两面板(主力后端 ERes2Net) ============
DUR0 = ([18.56, 8.39, 7.86], [13.57, 4.89, 4.43])
DUR5 = ([21.44, 11.06, 10.08], [14.74, 5.56, 4.90])
TL = ["1 s", "3 s", "5 s"]

fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.55), dpi=200,
                         gridspec_kw=dict(wspace=0.26))
fig.subplots_adjust(left=0.09, right=0.985, top=0.80, bottom=0.16)
b1, b2 = group_bars(axes[0], TL, *DUR0, 21, [0, 5, 10, 15, 20], tick_fs=12)
axes[0].set_title("信噪比 0 dB", fontsize=13, fontweight="bold", color=NAVY, pad=10)
axes[0].set_ylabel("EER (%)", fontsize=11.5, color=FGRAY)
axes[0].set_xlabel("语音时长", fontsize=11, color=FGRAY)
group_bars(axes[1], TL, *DUR5, 25, [0, 5, 10, 15, 20, 25], tick_fs=12)
axes[1].set_title("信噪比 −5 dB", fontsize=13, fontweight="bold", color=NAVY, pad=10)
axes[1].set_xlabel("语音时长", fontsize=11, color=FGRAY)
fig.legend(handles=[b1, b2], loc="upper center", ncol=2, frameon=False,
           fontsize=12, bbox_to_anchor=(0.5, 1.0))
fig.savefig(f"{DIR}/fig_ppt_duration2.png")
plt.close(fig)

# ============ 图 C:第19页右 = 复杂场景横向成对条形(主力后端) ============
SC = [("三人重叠 · 中文", 28.427, 18.421),
      ("二人重叠 · 中文", 22.416, 16.631),
      ("回声混响",        6.089,  4.874),
      ("二人重叠 · 英文", 9.133,  7.760)]

fig, ax = plt.subplots(figsize=(6.7, 3.55), dpi=200)
fig.subplots_adjust(left=0.205, right=0.975, top=0.80, bottom=0.13)
ys = list(range(len(SC))); h = 0.30
r1 = ax.barh([y + h / 2 + 0.02 for y in ys], [s[1] for s in SC], h, color=GRAY, label="Baseline")
r2 = ax.barh([y - h / 2 - 0.02 for y in ys], [s[2] for s in SC], h, color=BLUE, label="Delivered front-end")
for r, s in zip(r1, SC):
    ax.text(r.get_width() + 0.35, r.get_y() + r.get_height() / 2, f"{s[1]:.2f}",
            ha="left", va="center", color=FGRAY, fontsize=11)
for r, s in zip(r2, SC):
    ax.text(r.get_width() + 0.35, r.get_y() + r.get_height() / 2, f"{s[2]:.2f}",
            ha="left", va="center", color=NAVY, fontsize=11, fontweight="bold")
ax.set_yticks(ys)
ax.set_yticklabels([s[0] for s in SC], fontsize=12, color=DARK)
ax.invert_yaxis()
ax.set_xlim(0, 32)
ax.set_xticks([0, 10, 20, 30])
ax.tick_params(labelsize=11.5)
style_ax(ax)
ax.set_xlabel("EER (%)", fontsize=11.5, color=FGRAY)
ax.set_title("复杂声学场景(语音 3 s · 主力后端)", fontsize=13, fontweight="bold",
             color=NAVY, pad=10)
fig.legend(handles=[r1, r2], loc="upper center", ncol=2, frameon=False,
           fontsize=12, bbox_to_anchor=(0.55, 1.0))
fig.savefig(f"{DIR}/fig_ppt_scenes2.png")
plt.close(fig)

# ============ 图 D:第21页 = 噪声域泛化两面板(ERes2Net-large · 3 s) ============
NG = ["主验收池\n(自由声源)", "纯音乐域", "音效域\n(SoundBible)", "竞争说话人\n(speech)"]
N0B = [8.39, 9.637, 8.128, 8.633];  N0F = [4.89, 5.788, 4.949, 7.467]
N5B = [11.06, 13.290, 10.158, 10.831]; N5F = [5.56, 7.339, 5.515, 10.917]

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.35), dpi=200,
                         gridspec_kw=dict(wspace=0.22))
fig.subplots_adjust(left=0.06, right=0.985, top=0.84, bottom=0.135)
b1, b2 = group_bars(axes[0], NG, N0B, N0F, 12, [0, 4, 8, 12])
axes[0].set_title("信噪比 0 dB · 语音 3 s", fontsize=13, fontweight="bold", color=NAVY, pad=10)
axes[0].set_ylabel("EER (%)", fontsize=11.5, color=FGRAY)
axes[0].legend(handles=[b1, b2], loc="upper right", frameon=False, fontsize=11.5)
group_bars(axes[1], NG, N5B, N5F, 16, [0, 4, 8, 12, 16])
axes[1].set_title("信噪比 −5 dB · 语音 3 s", fontsize=13, fontweight="bold", color=NAVY, pad=10)
axes[1].text(0.985, 0.975, "训练未见噪声文件 · ERes2Net-large", transform=axes[1].transAxes,
             ha="right", va="top", fontsize=10.5, color=FGRAY)
fig.savefig(f"{DIR}/fig_noise_domains.png")
plt.close(fig)

# ============ 图 E:第20页 = 增量消融阶梯(官方链 · 冻结平均后端) ============
SCX = ["零训练融合", "+说话人保持\n损失", "+难负原型抑制\n与分类损失",
       "同配方续训\n第 1 轮", "同配方续训\n第 2 轮", "同配方续训\n第 3 轮\n(此链最优点)",
       "第4轮续训\n(回归 → 止损)"]
Y0 = [7.61, 7.29, 6.52, 5.77, 5.47, 5.24, 6.17]   # 0 dB
Y5 = [9.71, 9.20, 8.17, 6.97, 6.46, 6.10, 7.60]   # -5 dB
C0 = "#2F5496"; C5 = "#C55A11"

fig, ax = plt.subplots(figsize=(10.0, 5.0), dpi=200)
fig.subplots_adjust(left=0.075, right=0.985, top=0.965, bottom=0.145)
xs = list(range(7))
ax.plot(xs, Y0, "-o", color=BLUE, lw=2.4, ms=9, label="0 dB", zorder=3)
ax.plot(xs, Y5, "-s", color="#ED7D31", lw=2.4, ms=9, label="-5 dB", zorder=3)
L0 = 7.61 * 0.7; L5 = 9.71 * 0.7   # 30% 线 = raw × 0.7(只画虚线,不写文字)
ax.axhline(L5, color="#ED7D31", ls=(0, (5, 3)), lw=1.5, zorder=1)
ax.axhline(L0, color="#8FAADC", ls=(0, (5, 3)), lw=1.5, zorder=1)
for k, v in enumerate(Y0):   # 数值标签在点上方
    ax.text(k, v + 0.20, f"{v:.2f}", ha="center", va="bottom",
            color=C0, fontsize=12, fontweight="bold")
for k, v in enumerate(Y5):
    below = (k == 4)   # 6.46 标签在点下方,避免贴虚线
    ax.text(k, v - 0.26 if below else v + 0.20, f"{v:.2f}",
            ha="center", va="top" if below else "bottom",
            color=C5, fontsize=12, fontweight="bold")
ax.set_xticks(xs)
ax.set_xticklabels(SCX, fontsize=11.5, color=DARK)
ax.set_xlim(-0.45, 6.55)
ax.set_ylim(4.4, 10.45)
ax.set_yticks([5, 6, 7, 8, 9, 10])
style_ax(ax)
ax.set_ylabel("EER (%)", fontsize=12, color=FGRAY)
ax.legend(loc="upper right", frameon=False, fontsize=12.5)
fig.savefig(f"{DIR}/fig_staircase.png")
plt.close(fig)

print("DONE v4: main3 / duration2 / scenes2 / noise_domains / staircase")
