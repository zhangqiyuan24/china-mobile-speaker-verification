#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""channel_aug_v2.py — ChannelAug 加强版（8.25, 地板线/汤成员 D）

背景(8.25 地板-缝隙分解): 0dB 的既有增益全部来自 clean 地板(信道鲁棒性),
gap 零压缩 → 专攻跨设备信道失配的增强成员杠杆被低估。
v1 已是 soup3 贡献最重成员(B); v2 = 同 pretrained 初始化(保汤线性连通性)
+ 更强/更多样的信道扰动, 目标产出第 4 个成员 D。

相对 v1(channel_aug.py) 的差异:
  概率: rir 0.5→0.7, band 0.3→0.4, eq 0.3→0.4, codec 0.2→0.3
  强度: 带宽 hp 上限 400→500Hz / lp 下限 3400→2800Hz; EQ ±6→±9dB
  新效果: AM 泵动(AGC wobble) / 50Hz 工频哼声 / 20% 概率双 RIR 级联
  继承: 全部作用于 3 条件(含 clean), 末端峰值归一(v1 同款)
"""
import numpy as np

from channel_aug import ChannelAug, SR


class ChannelAugV2(ChannelAug):

    def __init__(self, rir_list_path, rir_cache=32):
        super().__init__(rir_list_path, p_rir=0.7, p_band=0.4, p_eq=0.4,
                         p_codec=0.3, rir_cache=rir_cache)
        self.p["am"] = 0.3
        self.p["hum"] = 0.2

    # ---- v1 效果加强 ----
    def _apply_band(self, w):
        hp = float(np.random.uniform(100, 500))
        lp = float(np.random.uniform(2800, 7500))
        return self._freq_mask(w, hp=hp, lp=lp)

    def _apply_eq(self, w):
        F = np.fft.rfft(w)
        f = np.fft.rfftfreq(len(w), 1.0 / SR)
        knots_f = np.linspace(0, SR / 2, 9)
        knots_g = np.random.uniform(-9, 9, 9)
        gain_db = np.interp(f, knots_f, knots_g)
        F = F * (10 ** (gain_db / 20.0))
        return np.fft.irfft(F, len(w)).astype(np.float32)

    # ---- 新效果 ----
    def _apply_am(self, w):
        """AM 泵动: 模拟设备 AGC 周期性增益摆动(0.5-4Hz, 深度 0.1-0.25)。"""
        f = float(np.random.uniform(0.5, 4.0))
        depth = float(np.random.uniform(0.1, 0.25))
        ph = float(np.random.uniform(0, 2 * np.pi))
        t = np.arange(len(w)) / SR
        return (w * (1.0 + depth * np.sin(2 * np.pi * f * t + ph))).astype(np.float32)

    def _apply_hum(self, w):
        """工频哼声: 50Hz + 谐波, 相对信号 RMS -26~-34dB(低端供电/接地噪声)。"""
        t = np.arange(len(w)) / SR
        p_rms = float(np.sqrt(np.mean(w ** 2)) + 1e-10)
        snr_db = float(np.random.uniform(-34, -26))
        h_rms = p_rms * (10 ** (snr_db / 20.0))
        hum = np.zeros(len(w), dtype=np.float32)
        for k, wgt in ((1, 1.0), (2, 0.5), (3, 0.25)):
            ph = float(np.random.uniform(0, 2 * np.pi))
            hum += wgt * np.sin(2 * np.pi * 50 * k * t + ph)
        hum *= h_rms / (float(np.sqrt(np.mean(hum ** 2))) + 1e-10)
        return w + hum

    def _apply_rir2(self, w):
        """双 RIR 级联: 20% 概率二次卷积(更长混响尾, 大房间/远讲)。"""
        w2 = self._apply_rir(w)
        if np.random.rand() < 0.2:
            w2 = self._apply_rir(w2)
        return w2

    def __call__(self, wav):
        w = wav
        if np.random.rand() < self.p["rir"]:
            w = self._apply_rir2(w)
        if np.random.rand() < self.p["band"]:
            w = self._apply_band(w)
        if np.random.rand() < self.p["eq"]:
            w = self._apply_eq(w)
        if np.random.rand() < self.p["codec"]:
            w = self._apply_codec(w)
        if np.random.rand() < self.p["am"]:
            w = self._apply_am(w)
        if np.random.rand() < self.p["hum"]:
            w = self._apply_hum(w)
        # 幅度安全(v1 同款): 峰值回到 <=0.95
        peak = float(np.max(np.abs(w))) if len(w) else 0.0
        if peak > 0.95:
            w = w * (0.95 / peak)
        # dtype 收敛: v1 的 codec(np.random.uniform float64)会泄 float64, 统一回 float32
        return w.astype(np.float32, copy=False)
