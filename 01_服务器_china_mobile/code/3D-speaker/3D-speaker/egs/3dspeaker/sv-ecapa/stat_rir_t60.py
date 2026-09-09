#!/usr/bin/env python
"""
RIR T60 统计脚本
================

统计 simulated_rirs 和 real_rirs_isotropic_noises 中各 RIR 的 T60。
T60 计算使用 Schroeder 反向积分 + 线性回归（ISO 3382 标准方法）。

用法：python stat_rir_t60.py
"""

import os, sys
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

SAMPLE_RATE = 16000
RIR_BASE = Path("data/rirs_noises/RIRS_NOISES")
SIM_DIR = RIR_BASE / "simulated_rirs"
REAL_DIR = RIR_BASE / "real_rirs_isotropic_noises"


def compute_t60(rir, sr, max_db_drop=35):
    """
    用 Schroeder 反向积分 + 线性回归计算 T60。

    步骤:
      1. 能量 = rir²
      2. 反向累积求和 → Schroeder 能量衰减曲线
      3. 取 dB，从 -5dB 到 min(-5-max_db_drop, noise_floor)dB 做线性回归
      4. T60 = -60 / 斜率
    """
    # 找峰值位置作为起始点
    peak_idx = np.argmax(np.abs(rir))
    rir_trim = rir[peak_idx:]

    if len(rir_trim) < sr * 0.05:  # 太短，无法计算
        return None

    # 能量反向积分
    energy = rir_trim ** 2
    sch = np.cumsum(energy[::-1])[::-1]

    max_val = sch[0]
    if max_val < 1e-15:
        return None

    sch_db = 10.0 * np.log10(np.maximum(sch, max_val * 1e-10) / max_val)

    # 找到 -5dB 的位置（避免初始瞬态）
    t5_idx = np.argmax(sch_db <= -5.0)
    if t5_idx == 0 or sch_db[t5_idx] > -5.0:
        # 没有衰减到 -5dB 以下，混响极轻
        return 0.05  # 极小值

    # 找噪声地板：sch_db 趋于平稳的位置
    # 用滑动窗口检测斜率趋于 0
    window = int(sr * 0.02)  # 20ms
    if t5_idx + window * 2 >= len(sch_db):
        return None

    # 取 -5dB 到 -35dB 之间的区域做线性回归（如果噪声地板更高则提前截断）
    max_drop = min(max_db_drop, int(abs(sch_db[t5_idx] - sch_db[-1]) * 0.8))
    if max_drop < 10:
        return None

    # 用 sch_db 一阶差分找噪声地板
    diff = np.diff(sch_db[t5_idx:])
    noise_start = None
    for k in range(len(diff) - window):
        if abs(np.mean(diff[k:k + window])) < 0.05:  # 斜率接近 0
            noise_start = k
            break

    if noise_start is not None and noise_start > 10:
        end_idx = t5_idx + noise_start
    else:
        end_idx = len(sch_db) - 1

    # 确保 end_idx 对应至少 10dB 的衰减
    if end_idx - t5_idx < int(sr * 0.02):
        return None

    # 线性回归
    x = np.arange(t5_idx, end_idx) / sr  # 时间（秒）
    y = sch_db[t5_idx:end_idx]

    if len(x) < 5:
        return None

    slope, intercept = np.polyfit(x, y, 1)

    if slope >= 0:  # 异常：没衰减反而上升
        return None

    # T60 = 降到 -60dB 的时间（从 0dB 算）
    # 拟合线: y = slope * t + intercept
    # t60 满足 slope * t60 + intercept = -60
    # 但 intercept ≈ -5（-5dB 起点），所以 t60 ≈ (-60 - (-5)) / slope = -55 / slope
    # 更准确：t60 = (-60 - intercept) / slope
    t60 = (-60.0 - intercept) / slope

    return max(0.01, min(t60, 10.0))  # 限制在 0.01s ~ 10s


def scan_and_calc(root_dir, tag, label_fn=None):
    """
    扫描目录下所有 .wav，计算 T60。
    label_fn: 从路径提取分组标签（如房间大小）
    """
    wavs = list(root_dir.rglob("*.wav"))
    if not wavs:
        print(f"  [{tag}] 未找到 wav 文件: {root_dir}")
        return {}

    results = defaultdict(list)  # label → [t60 values]

    for path in tqdm(wavs, desc=f"  [{tag}]"):
        try:
            rir, sr = sf.read(str(path))
        except Exception:
            continue
        if rir.ndim > 1:
            rir = rir[:, 0]
        rir = rir.astype(np.float32)

        t60 = compute_t60(rir, sr)
        if t60 is not None:
            if label_fn:
                label = label_fn(path)
            else:
                label = "all"
            results[label].append(t60)

    return results


def label_simulated(path):
    """从路径提取房间大小: smallroom / mediumroom / largeroom"""
    parts = path.parts
    for p in parts:
        if p in ("smallroom", "mediumroom", "largeroom"):
            return p
    return "unknown"


def label_real(path):
    """从文件名提取房间类型"""
    name = path.name.lower()
    if "largeroom" in name:
        return "largeroom"
    elif "mediumroom" in name:
        return "mediumroom"
    elif "smallroom" in name:
        return "smallroom"
    elif "simroom" in name:
        return "simroom"
    elif "booth" in name:
        return "booth"
    elif "lecture" in name:
        return "lecture"
    elif "meeting" in name:
        return "meeting"
    elif "office" in name:
        return "office"
    elif "corridor" in name or "stairway" in name:
        return "corridor_stairway"
    elif "bathroom" in name or "kitchen" in name:
        return "bathroom_kitchen"
    else:
        return "other"


def print_stats(label, t60s):
    arr = np.array(t60s)
    print(f"  {label:20s}  n={len(arr):5d}  "
          f"mean={np.mean(arr):.3f}s  median={np.median(arr):.3f}s  "
          f"min={np.min(arr):.3f}s  max={np.max(arr):.3f}s  "
          f"std={np.std(arr):.3f}s")


def main():
    print("=" * 60)
    print("  RIR T60 统计")
    print("=" * 60)

    # ── Simulated RIRs ──
    print("\n[1] Simulated RIRs")
    print("-" * 40)
    sim_results = scan_and_calc(SIM_DIR, "simulated", label_simulated)
    for label in ["smallroom", "mediumroom", "largeroom"]:
        if label in sim_results:
            print_stats(label, sim_results[label])
    all_sim = []
    for v in sim_results.values():
        all_sim.extend(v)
    if all_sim:
        print_stats("(全部)", all_sim)

    # ── Real RIRs ──
    print("\n[2] Real RIRs (Isotropic Noises)")
    print("-" * 40)
    real_results = scan_and_calc(REAL_DIR, "real", label_real)
    for label in sorted(real_results.keys()):
        print_stats(label, real_results[label])
    all_real = []
    for v in real_results.values():
        all_real.extend(v)
    if all_real:
        print_stats("(全部)", all_real)

    # ── 对比结论 ──
    print(f"\n{'=' * 60}")
    print("  对比")
    print(f"{'=' * 60}")
    if all_sim:
        arr_sim = np.array(all_sim)
        print(f"  Simulated:  {len(all_sim)} 个, T60 = {np.mean(arr_sim):.3f}s ± {np.std(arr_sim):.3f}s")
    if all_real:
        arr_real = np.array(all_real)
        print(f"  Real:       {len(all_real)} 个, T60 = {np.mean(arr_real):.3f}s ± {np.std(arr_real):.3f}s")


if __name__ == "__main__":
    main()
