#!/usr/bin/env python
"""
Clean Benchmark 生成脚本（子课题一）
======================================

基于 ori/{A,B,C,D,E,F} 的原始语音，生成 clean/{1s,3s,5s}/{A,B,C,D,E,F}。

核心逻辑：
  1. 对每条语音跑能量 VAD，找到有效语音段
  2. 选取能量最大的连续语音段作为截取锚点
  3. 原始语音 ≥ 目标 → 以最佳 VAD 段为中心截取目标时长
  4. 原始语音 < 目标 → 重复整段原始语音（保持自然节奏），余数用最佳 VAD 段补齐
  5. 所有随机过程固定 seed=2026，结果完全可复现

输出：
  data/data_for_test/clean/
  ├── 1s/{A,B,C,D,E,F}/{speaker_id}/{utt_id}_{1s}.wav
  ├── 3s/{A,B,C,D,E,F}/{speaker_id}/{utt_id}_{3s}.wav
  ├── 5s/{A,B,C,D,E,F}/{speaker_id}/{utt_id}_{5s}.wav
  └── clean_metadata.csv       ← 每条生成语音的完整 metadata

用法（在 sv-ecapa 目录下运行）：
  python gen_clean_benchmark.py

依赖：numpy, soundfile, tqdm（已在你的环境里）
"""

import os
import sys
import csv
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm


# ============================================================
# 全局配置
# ============================================================

SEED = 2026
SAMPLE_RATE = 16000
ORI_DIR = Path("data/data_for_test/ori")
CLEAN_DIR = Path("data/data_for_test/clean")
DURATIONS = [1, 3, 5]                 # 秒
ROLES = ["A", "B", "C", "D", "E", "F"]

# VAD 参数
VAD_FRAME_MS = 30                      # 帧长 (ms)
VAD_ENERGY_RATIO = 0.08                # 能量阈值 = 最大帧能量 × ratio
VAD_MIN_SEGMENT_SEC = 0.3              # 最短有效段 (s)，太短的段可能是噪音尖峰

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# 工具函数
# ============================================================

def load_mono(path):
    """加载 wav 为 float32 单声道"""
    data, sr = sf.read(path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"采样率不匹配: {path} is {sr}Hz")
    if data.ndim > 1:
        data = data[:, 0]
    return data.astype(np.float32)


def save_wav(path, data):
    """保存 wav，自动创建目录"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, data, SAMPLE_RATE)


def frame_energy(wav, frame_len):
    """分帧计算每帧能量"""
    n_frames = max(1, (len(wav) - frame_len) // frame_len + 1)
    energy = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * frame_len
        end = start + frame_len
        energy[i] = np.sum(wav[start:end] ** 2)
    return energy


def merge_segments(frame_indices, frame_len, min_frames, total_samples):
    """合并相邻有效帧为连续语音段"""
    if len(frame_indices) == 0:
        return []

    segments = []
    run_start = frame_indices[0]
    prev = frame_indices[0]

    for idx in frame_indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            n_frames = prev - run_start + 1
            if n_frames >= min_frames:
                seg_start = run_start * frame_len
                seg_end = min((prev + 1) * frame_len, total_samples)
                segments.append((seg_start, seg_end))
            run_start = idx
            prev = idx

    # 最后一段
    n_frames = prev - run_start + 1
    if n_frames >= min_frames:
        seg_start = run_start * frame_len
        seg_end = min((prev + 1) * frame_len, total_samples)
        segments.append((seg_start, seg_end))

    return segments


def energy_vad(wav, frame_ms=VAD_FRAME_MS, energy_ratio=VAD_ENERGY_RATIO,
               min_seg_sec=VAD_MIN_SEGMENT_SEC):
    """
    能量 VAD：返回有效语音段列表 [(start_sample, end_sample), ...]。
    选择能量最高的连续段，不是简单截中间。
    """
    total = len(wav)
    frame_len = int(SAMPLE_RATE * frame_ms / 1000)
    if frame_len <= 0 or total < frame_len:
        return [(0, total)]

    energy = frame_energy(wav, frame_len)

    if len(energy) == 0 or np.max(energy) < 1e-12:
        return [(0, total)]

    # 动态阈值：取最大帧能量的一定比例
    threshold = np.max(energy) * energy_ratio

    # 有效帧索引
    valid_frames = np.where(energy >= threshold)[0]
    min_frames = int(min_seg_sec * 1000 / frame_ms)

    segments = merge_segments(valid_frames, frame_len, min_frames, total)

    # 兜底：如果什么都没找到，整段当作一个段
    if not segments:
        return [(0, total)]

    return segments


def pick_best_segment(wav, segments):
    """从所有有效语音段中选取平均能量最大的那一个"""
    best_seg = None
    best_energy = -1.0

    for start, end in segments:
        seg = wav[start:end]
        if len(seg) > 0:
            avg_energy = float(np.mean(seg ** 2))
            if avg_energy > best_energy:
                best_energy = avg_energy
                best_seg = (start, end)

    # 如果没有找到有效段，回退到整段
    if best_seg is None:
        return (0, len(wav))

    return best_seg


def crop_or_repeat(wav, target_samples, segments):
    """
    截取 / 重复补齐至 target_samples。

    策略：
      - 原始语音 ≥ 目标时长：以最佳 VAD 段为中心截取 target_samples
      - 原始语音 < 目标时长：重复整段原始语音填满主部，
        末尾余数从最佳 VAD 段截取补齐（而非重复碎片产生不自然拼接）

    Returns:
        output_wav, is_repeated, (seg_start, seg_end)
    """
    best_start, best_end = pick_best_segment(wav, segments)
    best_seg = wav[best_start:best_end]
    orig_len = len(wav)

    if orig_len >= target_samples:
        # ── 原始语音够长：以最佳 VAD 段为中心截取 ──
        if len(best_seg) >= target_samples:
            # 最佳段本身就够长，直接居中截取
            offset = (len(best_seg) - target_samples) // 2
            crop_start = best_start + offset
        else:
            # 最佳段不够长但整段够长，以最佳段为中心向外扩展
            center = (best_start + best_end) // 2
            half = target_samples // 2
            crop_start = max(0, center - half)
            crop_end = min(orig_len, crop_start + target_samples)
            if crop_end - crop_start < target_samples:
                crop_start = max(0, crop_end - target_samples)

        result = wav[crop_start:crop_start + target_samples].copy()
        repeated = False

    else:
        # ── 原始语音短于目标：重复整段，余数用最佳 VAD 段补齐 ──
        n_full = target_samples // orig_len             # 整段重复次数
        remainder = target_samples - n_full * orig_len   # 末尾剩余采样数

        result = np.tile(wav, n_full)
        if remainder > 0:
            # 从最佳 VAD 段取前 remainder 个采样接在末尾
            result = np.concatenate([result, best_seg[:remainder]])

        repeated = True

    return result, repeated, (best_start, best_end)


def parse_3dspeaker_path(wav_path, speaker_dir):
    """
    解析 3D-Speaker 文件名，返回 dict。
    文件名格式: 3D_SPK_XXXXX_NNN_DeviceXX_DistanceYY_DialectZZ.wav
    """
    filename = os.path.basename(wav_path)
    stem = filename.replace(".wav", "")
    parts = stem.split("_")

    info = {"speaker_id": os.path.basename(speaker_dir),
            "device": None, "distance": None, "dialect": None,
            "utt_id": stem}

    # 尝试解析 _Device, _Distance, _Dialect
    for p in parts:
        if p.startswith("Device"):
            info["device"] = p
        elif p.startswith("Distance"):
            info["distance"] = p
        elif p.startswith("Dialect"):
            info["dialect"] = p

    return info


def parse_librivox_name(wav_path):
    """
    LibriVox 文件名: speech-librivox-XXXX.wav
    每个文件视为一个独立伪说话人。
    """
    filename = os.path.basename(wav_path)
    stem = filename.replace(".wav", "")
    # speech-librivox-0000 → librivox_0000
    spk_id = stem.replace("speech-", "").replace("-", "_")
    return {"speaker_id": spk_id, "device": None, "distance": None,
            "dialect": None, "utt_id": stem}


def get_role_language(role):
    """A/B/C/D → zh, E/F → en"""
    return "en" if role in ("E", "F") else "zh"


# ============================================================
# 主流程
# ============================================================

def generate_clean_for_role(role, duration_sec, metadata_rows):
    """为一个角色生成指定时长的 clean 语音"""
    role_ori = ORI_DIR / role
    target_samples = int(duration_sec * SAMPLE_RATE)

    if not role_ori.exists():
        print(f"  [跳过] {role_ori} 不存在")
        return

    lang = get_role_language(role)

    # 扫描原始 wav 列表
    # A/B/C/D: 有子目录，每个子目录 = 一个说话人
    # E/F: LibriVox 扁平文件，每个文件 = 一个伪说话人
    wav_tasks = []

    if role in ("E", "F"):
        # LibriVox 扁平结构
        for f in sorted(role_ori.glob("*.wav")):
            wav_tasks.append((str(f), str(f.parent)))  # parent 当 "speaker_dir"
    else:
        # 3D-Speaker: speaker_dir / *.wav
        for spk_dir in sorted(role_ori.iterdir()):
            if spk_dir.is_dir():
                for f in sorted(spk_dir.glob("*.wav")):
                    wav_tasks.append((str(f), str(spk_dir)))

    desc = f"Clean {duration_sec}s [{role}] ({lang})"
    for wav_path, spk_dir in tqdm(wav_tasks, desc=desc):
        # 解析路径信息
        if role in ("E", "F"):
            info = parse_librivox_name(wav_path)
        else:
            info = parse_3dspeaker_path(wav_path, spk_dir)

        speaker_id = info["speaker_id"]
        utt_id = info["utt_id"]

        # 加载原始音频
        try:
            wav = load_mono(wav_path)
        except Exception as e:
            print(f"  [警告] 加载失败 {wav_path}: {e}")
            continue

        orig_dur = len(wav) / SAMPLE_RATE

        # VAD
        segments = energy_vad(wav)

        # 截取或重复补齐
        out_wav, repeated, (vad_start, vad_end) = crop_or_repeat(wav, target_samples, segments)

        # 输出路径
        out_dir = CLEAN_DIR / f"{duration_sec}s" / role / speaker_id
        out_filename = f"{utt_id}_{duration_sec}s.wav"
        out_path = out_dir / out_filename
        save_wav(str(out_path), out_wav)

        # 写 metadata
        metadata_rows.append({
            "wav_path": str(out_path),
            "speaker_id": speaker_id,
            "role": role,
            "language": lang,
            "device": info["device"] or "",
            "distance": info["distance"] or "",
            "dialect": info["dialect"] or "",
            "original_wav": wav_path,
            "original_duration_sec": round(orig_dur, 3),
            "target_duration_sec": duration_sec,
            "vad_start_sec": round(vad_start / SAMPLE_RATE, 3),
            "vad_end_sec": round(vad_end / SAMPLE_RATE, 3),
            "vad_segment_sec": round((vad_end - vad_start) / SAMPLE_RATE, 3),
            "repeated": repeated,
            "condition": "clean",
            "seed": SEED,
        })


def main():
    print("=" * 60)
    print("  Clean Benchmark 生成")
    print(f"  seed={SEED},  时长={DURATIONS},  角色={ROLES}")
    print("=" * 60)

    metadata_rows = []

    for dur in DURATIONS:
        print(f"\n{'─' * 50}")
        print(f"  时长: {dur}s")
        print(f"{'─' * 50}")
        for role in ROLES:
            generate_clean_for_role(role, dur, metadata_rows)

    # 保存 metadata
    meta_path = CLEAN_DIR / "clean_metadata.csv"
    os.makedirs(str(CLEAN_DIR), exist_ok=True)
    fieldnames = [
        "wav_path", "speaker_id", "role", "language",
        "device", "distance", "dialect",
        "original_wav", "original_duration_sec", "target_duration_sec",
        "vad_start_sec", "vad_end_sec", "vad_segment_sec",
        "repeated", "condition", "seed",
    ]
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    # 统计
    print(f"\n{'=' * 60}")
    print(f"  生成完毕")
    print(f"{'=' * 60}")
    print(f"  总条数:  {len(metadata_rows)}")
    print(f"  Metadata: {meta_path}")

    # 按时长 / 角色 / 语言统计
    for dur in DURATIONS:
        subset = [r for r in metadata_rows if r["target_duration_sec"] == dur]
        n_zh = sum(1 for r in subset if r["language"] == "zh")
        n_en = sum(1 for r in subset if r["language"] == "en")
        n_rep = sum(1 for r in subset if r["repeated"])
        print(f"  {dur}s: {len(subset)} 条 (zh={n_zh}, en={n_en}, repeated={n_rep})")

    print(f"\n  输出目录: {CLEAN_DIR.resolve()}")


if __name__ == "__main__":
    main()
