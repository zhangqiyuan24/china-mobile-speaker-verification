#!/usr/bin/env python
"""
Reverb Benchmark 生成脚本（子课题一）
========================================

基于 clean 数据，为 A 和 D 添加混响。

规则：
  - RIR 来源：simulated_rirs/mediumroom（T60 ≈ 0.63s）
  - 按文件名排序，后 20% 作为测试专用 RIR 池
  - 每条 A/D 语音随机选一个 RIR 做 fftconvolve 卷积
  - 保留原长 + 0.5s 拖尾，不做能量归一化，仅防削波
  - 文件名与原 clean 完全一致（复用同一套 trial/enrollment）
  - seed=2026，RIR 分配完全可复现

输出：
  data/data_for_test/reverb/{1s,3s,5s}/{A,D}/
  data/divide/rir_test_mediumroom.txt    ← 测试 RIR 清单
  每个时长目录下附带 reverb_metadata.csv

用法：python gen_reverb_benchmark.py
      先运行 gen_clean_benchmark.py
"""

import os, sys, csv, random
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

SEED = 2026
SAMPLE_RATE = 16000
REVERB_TAIL_SEC = 0.5
RIR_SPLIT_RATIO = 0.2
DURATIONS = [1, 3, 5]

RIR_MEDIUMROOM = Path("data/rirs_noises/RIRS_NOISES/simulated_rirs/mediumroom")
CLEAN_DIR = Path("data/data_for_test/clean")
OUT_BASE = Path("data/data_for_test")
SPLIT_DIR = Path("data/divide")

TARGET_ROLES = ["A", "D"]

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# 工具
# ============================================================

def load_wav(path):
    data, sr = sf.read(path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"采样率不匹配: {path}")
    return data.astype(np.float32) if data.ndim == 1 else data[:, 0].astype(np.float32)


def save_wav(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, data, SAMPLE_RATE)


def compute_t60(rir, sr):
    """Schroeder 反向积分法计算 T60（秒）。失败返回 None。"""
    peak_idx = np.argmax(np.abs(rir))
    rir_trim = rir[peak_idx:]
    if len(rir_trim) < int(sr * 0.05):
        return None

    energy = rir_trim ** 2
    sch = np.cumsum(energy[::-1])[::-1]
    max_val = sch[0]
    if max_val < 1e-15:
        return None

    sch_db = 10.0 * np.log10(np.maximum(sch, max_val * 1e-10) / max_val)
    t5_idx = np.argmax(sch_db <= -5.0)
    if t5_idx == 0:
        return 0.05

    # 用差分找噪声地板
    diff = np.diff(sch_db[t5_idx:])
    window = int(sr * 0.02)
    noise_start = None
    for k in range(len(diff) - window):
        if abs(np.mean(diff[k:k + window])) < 0.05:
            noise_start = k
            break
    end_idx = t5_idx + noise_start if noise_start and noise_start > 10 else len(sch_db) - 1

    if end_idx - t5_idx < 5:
        return None

    x = np.arange(t5_idx, end_idx) / sr
    y = sch_db[t5_idx:end_idx]
    slope, intercept = np.polyfit(x, y, 1)
    if slope >= 0:
        return None
    t60 = (-60.0 - intercept) / slope
    return round(max(0.01, min(t60, 10.0)), 4)


# ============================================================
# RIR 池
# ============================================================

def build_rir_pool():
    """→ (test_pool, t60_map): test_pool=[path,...], t60_map={path: t60_sec}"""
    os.makedirs(SPLIT_DIR, exist_ok=True)

    rir_list_file = RIR_MEDIUMROOM / "rir_list"
    if not rir_list_file.exists():
        print(f"[ERROR] 未找到: {rir_list_file}")
        sys.exit(1)

    all_rirs = []
    with open(rir_list_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                path = Path("data/rirs_noises") / parts[4]
                if path.exists():
                    all_rirs.append(str(path))

    all_rirs.sort()
    n_test = max(1, int(len(all_rirs) * RIR_SPLIT_RATIO))
    train_pool = all_rirs[:-n_test]
    test_pool = all_rirs[-n_test:]

    # 预计算 T60
    t60_map = {}
    print(f"\n  预计算 {len(test_pool)} 个 RIR 的 T60...")
    for p in tqdm(test_pool, desc="  T60"):
        try:
            r, sr = sf.read(p)
            if r.ndim > 1:
                r = r[:, 0]
            t60 = compute_t60(r.astype(np.float32), sr)
            t60_map[p] = t60
        except Exception:
            t60_map[p] = None

    # 写出
    test_list = SPLIT_DIR / "rir_test_mediumroom.txt"
    with open(test_list, "w") as f:
        for p in test_pool:
            f.write(f"{p}\t{t60_map.get(p, 'N/A')}\n")

    train_list = SPLIT_DIR / "rir_train_mediumroom.txt"
    with open(train_list, "w") as f:
        for p in train_pool:
            f.write(p + "\n")

    valid_t60 = [v for v in t60_map.values() if v is not None]
    print(f"\n  mediumroom RIR 测试池: {len(test_pool)} 条")
    print(f"    T60: mean={np.mean(valid_t60):.3f}s, "
          f"median={np.median(valid_t60):.3f}s, "
          f"range=[{np.min(valid_t60):.3f}, {np.max(valid_t60):.3f}]s")

    return test_pool, t60_map


# ============================================================
# 扫描 clean
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
                        if stem.endswith(f"_{d}s"):
                            base = stem[:-len(f"_{d}s")]
                            break
                    inventory[dur][role][spk.name].append((base, str(wav)))
    return inventory


# ============================================================
# 混响生成
# ============================================================

def generate_reverb(rir_test_pool, t60_map, inventory):
    metadata_rows = []
    out_dir = OUT_BASE / "reverb"
    tail_samples = int(REVERB_TAIL_SEC * SAMPLE_RATE)

    for dur in DURATIONS:
        inv = inventory[dur]
        dur_dir = out_dir / f"{dur}s"
        os.makedirs(dur_dir, exist_ok=True)

        for role in TARGET_ROLES:
            items = []
            for spk, lst in inv[role].items():
                items.extend(lst)

            for base_stem, clean_path in tqdm(items, desc=f"  reverb [{role}] {dur}s"):
                clean = load_wav(clean_path)
                spk_id = os.path.basename(os.path.dirname(clean_path))
                lang = "en" if role in ("E", "F") else "zh"

                # 随机选 RIR
                rir_path = random.choice(rir_test_pool)
                try:
                    rir, rir_sr = sf.read(rir_path)
                except Exception:
                    continue
                if rir_sr != SAMPLE_RATE:
                    continue
                if rir.ndim > 1:
                    rir = rir[:, 0]
                rir = rir.astype(np.float32)

                # RIR 归一化到单位能量（避免脉冲响应幅度过小导致输出静音）
                rir_rms = np.sqrt(np.mean(rir ** 2) + 1e-12)
                rir = rir / rir_rms

                # 卷积
                full_reverb = fftconvolve(clean, rir)
                reverb = full_reverb[:len(clean) + tail_samples]

                # 恢复到与 clean 相近的响度，仅防削波
                scale = np.sqrt(np.mean(clean ** 2) + 1e-12) / (np.sqrt(np.mean(reverb ** 2) + 1e-12))
                reverb = reverb * scale
                peak = np.max(np.abs(reverb))
                if peak > 0.99:
                    reverb = reverb / peak * 0.99

                rel = os.path.relpath(clean_path, str(CLEAN_DIR / f"{dur}s" / role))
                out_path = os.path.join(str(dur_dir / role), rel)
                save_wav(out_path, reverb.astype(np.float32))

                metadata_rows.append({
                    "wav_path": out_path,
                    "speaker_id": spk_id,
                    "role": role,
                    "language": lang,
                    "duration_sec": dur,
                    "condition": "reverb",
                    "rir_wav": rir_path,
                    "rir_room": "mediumroom",
                    "rir_t60_sec": t60_map.get(rir_path, "N/A"),
                    "tail_sec": REVERB_TAIL_SEC,
                    "target_wav": clean_path,
                    "seed": SEED,
                })

        # 写 metadata + wav_list
        dur_rows = [r for r in metadata_rows if r["duration_sec"] == dur]
        meta_path = dur_dir / "reverb_metadata.csv"
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "wav_path", "speaker_id", "role", "language",
                "duration_sec", "condition",
                "rir_wav", "rir_room", "rir_t60_sec", "tail_sec",
                "target_wav", "seed",
            ])
            w.writeheader()
            w.writerows(dur_rows)

        wav_list_path = dur_dir / "wav_list.txt"
        with open(wav_list_path, "w") as f:
            for r in dur_rows:
                f.write(r["wav_path"] + "\n")

        n_a = sum(1 for r in dur_rows if r["role"] == "A")
        n_d = sum(1 for r in dur_rows if r["role"] == "D")
        print(f"    [{dur}s] A={n_a}, D={n_d}")

    return metadata_rows


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("  Reverb Benchmark 生成")
    print(f"  RIR: simulated mediumroom (T60 ≈ 0.63s)")
    print(f"  拖尾: {REVERB_TAIL_SEC}s,  seed: {SEED}")
    print("=" * 60)

    rir_test_pool, t60_map = build_rir_pool()
    inventory = scan_clean()
    rows = generate_reverb(rir_test_pool, t60_map, inventory)

    print(f"\n{'=' * 60}")
    print(f"  完成！共生成 {len(rows)} 条混响语音")
    print(f"  输出: {OUT_BASE}/reverb/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
