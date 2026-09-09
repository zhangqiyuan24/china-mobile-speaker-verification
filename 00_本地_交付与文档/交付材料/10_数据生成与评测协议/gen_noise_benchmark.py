#!/usr/bin/env python
"""
Noise Benchmark 生成脚本（子课题一）
======================================

基于 clean 数据，为 A 和 D 叠加自由声场噪声（MUSAN free-sound）。

规则：
  - 噪声来源：MUSAN noise/free-sound/ 目录
  - 噪声音频按文件名排序，后 20% 作为测试专用噪声池，前 80% 预留训练
  - 每条 A/D 语音随机选一条测试噪声，随机截取与目标等长的片段
  - 0dB 和 -5dB 使用**完全相同**的噪声文件和截取位置，仅 SNR 缩放不同
  - 文件名与原 clean 完全一致（复用同一套 trial/enrollment）
  - seed=2026，噪声分配完全可复现

输出：
  data/data_for_test/noise_0db/{1s,3s,5s}/{A,D}/
  data/data_for_test/noise_m5db/{1s,3s,5s}/{A,D}/
  data/divide/noise_test_freesound.txt   ← 测试噪声清单
  每个时长目录下附带 noise_metadata.csv

用法：python gen_noise_benchmark.py
      先运行 gen_clean_benchmark.py
"""

import os, sys, csv, random
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

SEED = 2026
SNR_LEVELS = [0, -5]              # dB
SAMPLE_RATE = 16000
NOISE_SPLIT_RATIO = 0.2           # 后 20% 用作测试
DURATIONS = [1, 3, 5]

MUSAN_NOISE_DIR = Path("data/musan/noise/free-sound")
CLEAN_DIR = Path("data/data_for_test/clean")
OUT_BASE = Path("data/data_for_test")
SPLIT_DIR = Path("data/divide")

TARGET_ROLES = ["A", "D"]

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# 噪声工具
# ============================================================

def load_wav(path):
    data, sr = sf.read(path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"采样率不匹配: {path}")
    return data.astype(np.float32) if data.ndim == 1 else data[:, 0].astype(np.float32)


def save_wav(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, data, SAMPLE_RATE)


def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)


def mix_noise(clean, noise_full, snr_db, noise_start):
    """将 clean 与 noise_full[noise_start:noise_start+len(clean)] 按 snr_db 混合"""
    # 截取噪声
    if noise_start + len(clean) <= len(noise_full):
        noise_seg = noise_full[noise_start:noise_start + len(clean)]
    else:
        # 噪声不够长则循环补齐
        seg = noise_full[noise_start:]
        reps = int(np.ceil(len(clean) / len(noise_full)))
        noise_seg = np.tile(np.concatenate([seg, np.tile(noise_full, reps)]), 1)[:len(clean)]

    # 按 SNR 缩放
    desired_rms = rms(clean) / (10 ** (snr_db / 20.0))
    noise_rms = rms(noise_seg)
    if noise_rms < 1e-12:
        scale = 0.0
    else:
        scale = desired_rms / noise_rms
    noise_scaled = noise_seg * scale

    mixed = clean + noise_scaled

    # 防削波
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed / peak * 0.99

    return mixed.astype(np.float32), round(scale, 6)


# ============================================================
# 噪声池构建
# ============================================================

def build_noise_pool():
    """
    扫描 MUSAN free-sound，按文件名排序，后 20% 作为测试池。
    → test_pool: [wav_path, ...]
    → 写出 data/divide/noise_test_freesound.txt
    """
    os.makedirs(SPLIT_DIR, exist_ok=True)

    all_noise = sorted(MUSAN_NOISE_DIR.rglob("*.wav"))
    all_noise = [str(p) for p in all_noise]

    if len(all_noise) == 0:
        print(f"[ERROR] 未找到噪声文件: {MUSAN_NOISE_DIR}")
        sys.exit(1)

    n_test = max(1, int(len(all_noise) * NOISE_SPLIT_RATIO))
    train_pool = all_noise[:-n_test]    # 前 80%
    test_pool = all_noise[-n_test:]     # 后 20%

    # 写出
    test_list_path = SPLIT_DIR / "noise_test_freesound.txt"
    with open(test_list_path, "w") as f:
        for p in test_pool:
            f.write(p + "\n")

    train_list_path = SPLIT_DIR / "noise_train_freesound.txt"
    with open(train_list_path, "w") as f:
        for p in train_pool:
            f.write(p + "\n")

    print(f"\n  MUSAN free-sound: {len(all_noise)} 条")
    print(f"    训练池 (前80%): {len(train_pool)} 条 → {train_list_path}")
    print(f"    测试池 (后20%): {len(test_pool)} 条 → {test_list_path}")

    return test_pool


# ============================================================
# 扫描 clean 数据（与 gen_overlap_benchmark.py 相同逻辑）
# ============================================================

def scan_clean():
    inventory = {d: {r: defaultdict(list) for r in ["A", "B", "C", "D", "E", "F"]}
                 for d in DURATIONS}

    for dur in DURATIONS:
        role_dir = CLEAN_DIR / f"{dur}s"
        if not role_dir.exists():
            continue
        for role in ["A", "B", "C", "D", "E", "F"]:
            spk_dir = role_dir / role
            if not spk_dir.exists():
                continue
            for spk in sorted(spk_dir.iterdir()):
                if not spk.is_dir():
                    continue
                for wav in sorted(spk.glob("*.wav")):
                    stem = wav.stem
                    base = stem
                    for d in DURATIONS:
                        suffix = f"_{d}s"
                        if stem.endswith(suffix):
                            base = stem[:-len(suffix)]
                            break
                    inventory[dur][role][spk.name].append((base, str(wav)))
    return inventory


# ============================================================
# 混合生成
# ============================================================

def generate_noise_condition(noise_test_pool, inventory):
    """
    对每个 A/D 语音：
      1. 随机选一条噪声 + 随机起始位置
      2. 以 0dB 混合 → noise_0db
      3. 以 -5dB 混合 → noise_m5db（同噪声、同位置）
    """
    metadata_rows = []

    for dur in DURATIONS:
        inv = inventory[dur]

        out_dirs = {
            0:  OUT_BASE / "noise_0db" / f"{dur}s",
            -5: OUT_BASE / "noise_m5db" / f"{dur}s",
        }
        for d in out_dirs.values():
            os.makedirs(d, exist_ok=True)

        for role in TARGET_ROLES:
            # 收集所有 (base_stem, clean_path)
            items = []
            for spk, lst in inv[role].items():
                items.extend(lst)

            # 用确定性的随机状态（seed 已固定）
            # 为每条语音预生成噪声选择
            for base_stem, clean_path in tqdm(items, desc=f"  noise [{role}] {dur}s"):
                clean = load_wav(clean_path)
                spk_id = os.path.basename(os.path.dirname(clean_path))
                lang = "en" if role in ("E", "F") else "zh"

                # 随机选噪声 + 起始位置
                noise_file = random.choice(noise_test_pool)
                noise_full = load_wav(noise_file)
                max_start = max(0, len(noise_full) - len(clean))
                if max_start > 0:
                    noise_start = random.randint(0, max_start)
                else:
                    noise_start = 0

                # 两个 SNR 共用同一个噪声片段
                for snr in SNR_LEVELS:
                    mixed, scale = mix_noise(clean, noise_full, snr, noise_start)

                    cond_name = f"noise_{'m5db' if snr == -5 else '0db'}"
                    out_dir = out_dirs[snr]

                    rel = os.path.relpath(clean_path, str(CLEAN_DIR / f"{dur}s" / role))
                    out_path = os.path.join(str(out_dir / role), rel)
                    save_wav(out_path, mixed)

                    metadata_rows.append({
                        "wav_path": out_path,
                        "speaker_id": spk_id,
                        "role": role,
                        "language": lang,
                        "duration_sec": dur,
                        "condition": cond_name,
                        "SNR_db": snr,
                        "target_wav": clean_path,
                        "noise_wav": noise_file,
                        "noise_offset_sample": noise_start,
                        "noise_scale": scale,
                        "seed": SEED,
                    })

        # 每个 SNR 按时长写 metadata + wav_list
        for snr in SNR_LEVELS:
            cond_name = f"noise_{'m5db' if snr == -5 else '0db'}"
            out_dir = OUT_BASE / cond_name / f"{dur}s"
            dur_rows = [r for r in metadata_rows
                        if r["duration_sec"] == dur and r["condition"] == cond_name]

            # metadata
            meta_path = out_dir / "noise_metadata.csv"
            with open(meta_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "wav_path", "speaker_id", "role", "language",
                    "duration_sec", "condition", "SNR_db",
                    "target_wav", "noise_wav", "noise_offset_sample",
                    "noise_scale", "seed",
                ])
                w.writeheader()
                w.writerows(dur_rows)

            # wav_list
            wav_list_path = out_dir / "wav_list.txt"
            with open(wav_list_path, "w") as f:
                for r in dur_rows:
                    f.write(r["wav_path"] + "\n")

            n_a = sum(1 for r in dur_rows if r["role"] == "A")
            n_d = sum(1 for r in dur_rows if r["role"] == "D")
            print(f"    {cond_name} [{dur}s]: A={n_a}, D={n_d}")

    return metadata_rows


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("  Noise Benchmark 生成")
    print(f"  噪声源: MUSAN free-sound（后 20% 测试）")
    print(f"  SNR: {SNR_LEVELS},  seed: {SEED}")
    print("=" * 60)

    # 1. 构建噪声测试池
    noise_test_pool = build_noise_pool()

    # 2. 扫描 clean
    inventory = scan_clean()

    # 3. 生成 0dB / -5dB
    rows = generate_noise_condition(noise_test_pool, inventory)

    print(f"\n{'=' * 60}")
    print(f"  完成！共生成 {len(rows)} 条噪声语音")
    print(f"  输出: {OUT_BASE}/noise_0db/, {OUT_BASE}/noise_m5db/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
