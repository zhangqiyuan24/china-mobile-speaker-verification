#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""channel_aug.py — 信道/设备域增强（8.24 备用弹药②）

动机: 协议是 trials_cross_device(注册/测试设备不同), 信道失配是一阶问题;
现有鲁棒训练只做混噪, 对信道零扰动。本模块在波形域叠加四类轻量信道效果:
  1) RIR 卷积(模拟房间/麦克风位置; 数据: sv-ecapa/data/rirs_noises, 6 万条模拟 RIR)
  2) 带宽限制(随机高通 100-400Hz + 低通 3400-7500Hz, 模拟电话/低端设备)
  3) 随机频响 EQ(FFT 域 8 段随机增益 ±6dB, 模拟麦克风频响差异)
  4) 量化/削波微扰(8bit 量化 + dither + 软削波, 模拟低端 ADC)

依赖: 仅 numpy(FFT 域处理, 无 scipy/librosa)。wav 加载用 8.20 的 load_wav
(torchaudio, 与训练管线一致)。

用法(插入点: random_crop 之后, mix_noise 之前, 对全部 3 个条件生效——
"clean"条件在跨设备协议下同样有信道失配):
    from channel_aug import ChannelAug
    aug = ChannelAug(rir_list_path, rir_base)
    wav3 = aug(wav3)
"""
import os

import numpy as np

SR = 16000


def _load_rir(path):
    from common_utils import load_wav  # 8.20 的 torchaudio 加载器
    rir = load_wav(path).astype(np.float32)
    if len(rir) == 0:
        return None
    # 单位能量归一 + 截尾 0.5s(RIR 能量集中在前段)
    e = np.sqrt(np.sum(rir ** 2)) + 1e-10
    rir = rir / e
    max_len = int(0.5 * SR)
    if len(rir) > max_len:
        rir = rir[:max_len]
    return rir


class ChannelAug:
    """每次调用按独立概率叠加各效果。RNG 用 np.random(worker_init_fn 已按
    worker 派生种子, 与 8.20 训练管线随机源一致)。"""

    def __init__(self, rir_list_path, p_rir=0.5, p_band=0.3, p_eq=0.3,
                 p_codec=0.2, rir_cache=32):
        self.rir_paths = [l.strip() for l in open(rir_list_path) if l.strip()]
        self.p = dict(rir=p_rir, band=p_band, eq=p_eq, codec=p_codec)
        self.cache = {}          # LRU: RIR 很小, 预载避免每条重复 IO
        self.cache_n = rir_cache

    def _get_rir(self):
        p = self.rir_paths[np.random.randint(len(self.rir_paths))]
        if p not in self.cache:
            if len(self.cache) >= self.cache_n:
                self.cache.pop(next(iter(self.cache)))
            r = _load_rir(p)
            if r is None:
                return None
            self.cache[p] = r
        return self.cache[p]

    # ---- 各效果 ----
    def _apply_rir(self, w):
        rir = self._get_rir()
        if rir is None:
            return w
        n = len(w) + len(rir) - 1
        W = np.fft.rfft(w, n)
        H = np.fft.rfft(rir, n)
        y = np.fft.irfft(W * H, n).astype(np.float32)
        # 峰值对齐(去掉传播延迟), 保持原长
        peak = int(np.argmax(np.abs(rir)))
        return y[peak:peak + len(w)]

    def _freq_mask(self, w, hp=None, lp=None):
        F = np.fft.rfft(w)
        f = np.fft.rfftfreq(len(w), 1.0 / SR)
        if hp is not None:  # 平滑高通(过渡带 ~100Hz)
            g = 1.0 / (1.0 + np.exp(-(f - hp) / 50.0))
            F = F * g
        if lp is not None:  # 平滑低通
            g = 1.0 / (1.0 + np.exp((f - lp) / 150.0))
            F = F * g
        return np.fft.irfft(F, len(w)).astype(np.float32)

    def _apply_band(self, w):
        hp = float(np.random.uniform(100, 400))
        lp = float(np.random.uniform(3400, 7500))
        return self._freq_mask(w, hp=hp, lp=lp)

    def _apply_eq(self, w):
        # 8 段随机增益(±6dB), 线性插值成平滑曲线
        F = np.fft.rfft(w)
        f = np.fft.rfftfreq(len(w), 1.0 / SR)
        knots_f = np.linspace(0, SR / 2, 9)
        knots_g = np.random.uniform(-6, 6, 9)
        gain_db = np.interp(f, knots_f, knots_g)
        F = F * (10 ** (gain_db / 20.0))
        return np.fft.irfft(F, len(w)).astype(np.float32)

    def _apply_codec(self, w):
        q = np.round(w * 128.0) / 128.0          # 8bit 量化
        d = np.random.uniform(-1, 1, len(w)) * (1.0 / 256.0)  # dither
        y = q + d
        return np.tanh(y * 1.1) / 1.1            # 软削波

    def __call__(self, wav):
        w = wav
        if np.random.rand() < self.p["rir"]:
            w = self._apply_rir(w)
        if np.random.rand() < self.p["band"]:
            w = self._apply_band(w)
        if np.random.rand() < self.p["eq"]:
            w = self._apply_eq(w)
        if np.random.rand() < self.p["codec"]:
            w = self._apply_codec(w)
        # 幅度安全: 峰值回到 <=0.95
        peak = float(np.max(np.abs(w))) if len(w) else 0.0
        if peak > 0.95:
            w = w * (0.95 / peak)
        return w.astype(np.float32)


def make_rir_list(rir_root, out_list, n=2000, seed=2026):
    """从 simulated_rirs 随机抽 n 条生成清单(确定性)。"""
    import glob
    pats = ["simulated_rirs/smallroom/Room*/*.wav",
            "simulated_rirs/mediumroom/Room*/*.wav",
            "simulated_rirs/largeroom/Room*/*.wav"]
    paths = []
    for pat in pats:
        paths.extend(glob.glob(os.path.join(rir_root, pat)))
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(paths))[:n]
    with open(out_list, "w") as f:
        for i in idx:
            f.write(paths[i] + "\n")
    print(f"[rir-list] {min(n, len(paths))}/{len(paths)} -> {out_list}")
    return out_list


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rir-root", default="/data/home/ligangrui/china_mobile/code/"
                    "3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa/data/rirs_noises/RIRS_NOISES")
    ap.add_argument("--out", default="exp/channel_aug/rir_list.txt")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    make_rir_list(args.rir_root, args.out, args.n)
