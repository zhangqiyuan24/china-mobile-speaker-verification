#!/usr/bin/env python
"""
Overlap Benchmark 生成脚本（子课题一）
========================================

基于 clean 数据，为 A 和 D 生成重叠语音。

规则：
  - A/D 的每条 utterance 混入指定干扰说话人的语音
  - overlap2_cn:  A/D + B (中文干扰 ×1)
  - overlap2_en:  A/D + E (英文干扰 ×1)
  - overlap3_cn:  A/D + B + C (中文干扰 ×2)
  - overlap3_en:  A/D + E + F (英文干扰 ×2)
  - TIR = 0dB（目标与每个干扰能量 1:1），seed=2026
  - 文件名与原 clean 完全一致（复用同一套 trial/enrollment）
  - enrollment 侧的 A 保持 clean，仅测试端混合（评测时 --enrol_data 指向 clean，--test_data 指向 overlap）

输出：
  data/data_for_test/overlap2_cn/{1s,3s,5s}/{A,D}/
  data/data_for_test/overlap2_en/{1s,3s,5s}/{A,D}/
  data/data_for_test/overlap3_cn/{1s,3s,5s}/{A,D}/
  data/data_for_test/overlap3_en/{1s,3s,5s}/{A,D}/
  每个时长目录下附带 overlap_metadata.csv

用法：python gen_overlap_benchmark.py
      先运行 gen_clean_benchmark.py
"""

import os, sys, csv, random
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

SEED = 2026
TIR_DB = 0.0       # 目标能量 / 干扰能量
SAMPLE_RATE = 16000
DURATIONS = [1, 3, 5]

CLEAN_DIR = Path("data/data_for_test/clean")
OUT_BASE = Path("data/data_for_test")

# 条件定义: 输出目录名 → (interferer roles, language label)
CONDITIONS = {
    "overlap2_cn":  (["B"],       "cn"),
    "overlap2_en":  (["E"],       "en"),
    "overlap3_cn":  (["B", "C"],   "cn"),
    "overlap3_en":  (["E", "F"],   "en"),
}

TARGET_ROLES = ["A", "D"]  # 需要混合的角色

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


def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)


def overlay(target, interferers, tir_db=TIR_DB):
    """
    混合 target + 多个 interferer，每个干扰按 tir_db 缩放。
    返回 mixed_wav, individual_scales
    """
    mixed = target.copy().astype(np.float64)
    scales = []

    for int_wav in interferers:
        # 对齐到 target 长度
        if len(int_wav) < len(target):
            reps = int(np.ceil(len(target) / len(int_wav)))
            int_wav = np.tile(int_wav, reps)[:len(target)]
        else:
            start = random.randint(0, len(int_wav) - len(target))
            int_wav = int_wav[start:start + len(target)]

        # 缩放到 TIR
        scale = rms(target) / (rms(int_wav) * 10 ** (tir_db / 20.0))
        scaled = int_wav * scale
        mixed += scaled
        scales.append(round(scale, 4))

    # 防削波
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed / peak * 0.99

    return mixed.astype(np.float32), scales


# ============================================================
# 扫描 clean 数据
# ============================================================

def scan_clean():
    """
    → {
        dur: {
            role: {
                speaker_id: [(base_stem, wav_path), ...]
            }
        }
      }
    base_stem: 去掉时长后缀，如 "3D_SPK_06154_001_Device03_Distance00_Dialect00"
    因为 1s/3s/5s 共享相同 base stem，后续按时长索引时用 base_stem + f"_{dur}s"
    """
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
                if not spk_dir.is_dir():
                    continue
                for wav in sorted(spk.glob("*.wav")):
                    stem = wav.stem  # e.g. "3D_SPK_06154_..._1s"
                    base = stem
                    for d in DURATIONS:
                        suffix = f"_{d}s"
                        if stem.endswith(suffix):
                            base = stem[:-len(suffix)]
                            break
                    inventory[dur][role][spk.name].append((base, str(wav)))

    return inventory


# ============================================================
# 主流程
# ============================================================

def generate_overlap_condition(cond_name, int_roles, lang, inventory):
    """生成单个 overlap 条件（所有时长）"""
    print(f"\n{'=' * 60}")
    print(f"  {cond_name}  (干扰: {'+'.join(int_roles)}, {lang})")
    print(f"{'=' * 60}")

    metadata_rows = []

    for dur in DURATIONS:
        out_dir = OUT_BASE / cond_name / f"{dur}s"
        inv = inventory[dur]

        # 构建干扰池: base_stem → wav_path
        int_pool = []  # [(base_stem, wav_path), ...]
        for role in int_roles:
            for spk, items in inv[role].items():
                int_pool.extend(items)

        if not int_pool:
            print(f"  [{dur}s] 干扰池为空，跳过")
            continue

        # 对每个需要混合的角色
        for role in TARGET_ROLES:
            all_items = []
            for spk, items in inv[role].items():
                all_items.extend(items)

            desc = f"  {cond_name} {dur}s [{role}]"
            for base_stem, clean_path in tqdm(all_items, desc=desc):
                target = load_wav(clean_path)

                # 随机选干扰（同一时长）
                chosen = random.sample(int_pool,
                                       min(len(int_roles), len(int_pool)))
                if len(chosen) < len(int_roles):
                    continue  # 干扰不够，理论上不应发生

                int_wavs = []
                int_bases = []
                for int_base, int_path in chosen:
                    int_wavs.append(load_wav(int_path))
                    int_bases.append(int_base)

                # 混合
                mixed, scales = overlay(target, int_wavs)

                # 输出路径（保持与 clean 相同的子目录和文件名）
                rel = os.path.relpath(clean_path, str(CLEAN_DIR / f"{dur}s" / role))
                out_path = os.path.join(str(out_dir / role), rel)
                save_wav(out_path, mixed)

                # metadata
                metadata_rows.append({
                    "wav_path": out_path,
                    "speaker_id": os.path.basename(os.path.dirname(out_path)),
                    "role": role,
                    "language": "zh" if role in ("A", "B", "C", "D") else "en",
                    "duration_sec": dur,
                    "condition": cond_name,
                    "interferer_language": lang,
                    "num_interferers": len(int_roles),
                    "TIR_db": TIR_DB,
                    "target_wav": clean_path,
                    "interferer_wavs": "|".join(chosen_paths := [p for _, p in chosen]),
                    "interferer_speakers": "|".join(
                        [os.path.basename(os.path.dirname(p)) for _, p in chosen]),
                    "interferer_scales": "|".join([str(s) for s in scales]),
                    "seed": SEED,
                })

        n = len(metadata_rows)
        n_a = sum(1 for r in metadata_rows if r["role"] == "A" and r["duration_sec"] == dur)
        n_d = sum(1 for r in metadata_rows if r["role"] == "D" and r["duration_sec"] == dur)
        print(f"    [{dur}s] A={n_a}, D={n_d}")

        # 每个时长写一份 metadata
        dur_meta = [r for r in metadata_rows if r["duration_sec"] == dur]
        meta_path = out_dir / "overlap_metadata.csv"
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "wav_path", "speaker_id", "role", "language",
                "duration_sec", "condition", "interferer_language",
                "num_interferers", "TIR_db",
                "target_wav", "interferer_wavs", "interferer_speakers",
                "interferer_scales", "seed",
            ])
            w.writeheader()
            w.writerows(dur_meta)

        # 生成 wav_list.txt
        wav_list = out_dir / "wav_list.txt"
        with open(wav_list, "w") as f:
            for r in dur_meta:
                f.write(r["wav_path"] + "\n")

    return metadata_rows


def main():
    print("=" * 60)
    print("  Overlap Benchmark 生成")
    print(f"  TIR = {TIR_DB}dB, seed = {SEED}")
    print("=" * 60)

    inventory = scan_clean()
    total = 0

    for cond_name, (int_roles, lang) in CONDITIONS.items():
        rows = generate_overlap_condition(cond_name, int_roles, lang, inventory)
        total += len(rows)

    print(f"\n{'=' * 60}")
    print(f"  完成！共生成 {total} 条 overlap 语音")
    print(f"  输出: {OUT_BASE}/overlap2_cn/, overlap2_en/, overlap3_cn/, overlap3_en/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
