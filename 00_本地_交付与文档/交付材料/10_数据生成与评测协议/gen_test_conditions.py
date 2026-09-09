"""
多条件测试数据集生成脚本（子课题一）

功能：基于 3D-Speaker 干净测试集，生成多种声学条件下的测试数据：
  1-6. 单说话人噪声（3 种噪声类型 × 2 档 SNR）：
       music / noise / speech 各 SNR=-5dB, 0dB
  7.   回声/混响
  8-9. 2/3 人重叠语音（TIR=0dB）
  10-12. 短语音（1s / 3s / 5s）

每个场景输出：
  - 新的 wav 文件（存放在 data/data_for_test/test_<condition>/ 目录下）
  - 新的 wav_list.txt（每行一个 wav 路径）

用法（在 sv-ecapa 目录下运行）：
  # 噪声（按类型 × SNR 分 6 种）
  python gen_test_conditions.py --scenario noise_m5db_music
  python gen_test_conditions.py --scenario noise_m5db_noise
  python gen_test_conditions.py --scenario noise_m5db_speech
  python gen_test_conditions.py --scenario noise_0db_music
  python gen_test_conditions.py --scenario noise_0db_noise
  python gen_test_conditions.py --scenario noise_0db_speech

  # 其他场景
  python gen_test_conditions.py --scenario reverb
  python gen_test_conditions.py --scenario overlap_2
  python gen_test_conditions.py --scenario overlap_3
  python gen_test_conditions.py --scenario short_1s
  python gen_test_conditions.py --scenario short_3s
  python gen_test_conditions.py --scenario short_5s

  # 一键生成全部 12 种条件：
  python gen_test_conditions.py --scenario all

  # 推荐：使用 gen_all_test_data.sh 一键完成 split + gen 全流程
  bash gen_all_test_data.sh

依赖：soundfile, scipy, numpy, tqdm（已在你的环境里）
"""

import os
import sys
import argparse
import random
import numpy as np
import soundfile as sf
from tqdm import tqdm
from pathlib import Path

# 固定随机种子，保证每次生成的测试数据完全一致
random.seed(42)
np.random.seed(42)

# ============================================================
# 配置
# ============================================================

SAMPLE_RATE = 16000
CLEAN_WAV_LIST = "data/test/wav_list.txt"                  # 干净测试集路径列表
NOISE_TEST_MUSIC  = "data/divide/noise_test_music.txt"      # 测试专用音乐噪声（MUSAN 20%）
NOISE_TEST_NOISE  = "data/divide/noise_test_noise.txt"      # 测试专用环境噪声（MUSAN 20%）
NOISE_TEST_SPEECH = "data/divide/noise_test_speech.txt"     # 测试专用语音噪声（MUSAN 20%）
NOISE_TRAIN_SCP   = "data/divide/noise_train.scp"           # 训练专用噪声（MUSAN 80%）
RIR_TEST_LIST     = "data/divide/rir_test.txt"              # 测试专用混响（真实 RIR）
RIR_TRAIN_SCP     = "data/divide/rir_train.scp"             # 训练专用混响（模拟 RIR）
OUT_BASE          = "data/data_for_test"                    # 所有测试数据统一输出目录

parser = argparse.ArgumentParser(description="Generate multi-condition test data for 子课题一")
parser.add_argument("--scenario", required=True,
                    choices=["noise_m5db_music", "noise_m5db_noise", "noise_m5db_speech",
                             "noise_0db_music", "noise_0db_noise", "noise_0db_speech",
                             "reverb",
                             "overlap_2", "overlap_3",
                             "short_1s", "short_3s", "short_5s", "all"],
                    help="Which condition to generate")
parser.add_argument("--nj", type=int, default=4, help="Number of parallel workers")
args = parser.parse_args()


# ============================================================
# 工具函数
# ============================================================

def load_wav(path):
    """加载 wav 并确保 16kHz 单声道"""
    data, sr = sf.read(path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Sample rate mismatch: {path} is {sr}Hz, expected {SAMPLE_RATE}")
    if data.ndim > 1:
        data = data[:, 0]  # 取第一声道
    return data.astype(np.float32)


def save_wav(path, data):
    """保存 wav"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, data, SAMPLE_RATE)


def compute_rms(signal):
    """计算均方根能量"""
    return np.sqrt(np.mean(signal ** 2) + 1e-12)


def adjust_snr(clean, noise, snr_db):
    """将噪声缩放到指定 SNR"""
    rms_clean = compute_rms(clean)
    rms_noise = compute_rms(noise)
    desired_rms_noise = rms_clean / (10 ** (snr_db / 20.0))
    if rms_noise < 1e-12:
        return np.zeros_like(clean)
    return noise * (desired_rms_noise / rms_noise)


# ============================================================
# 场景生成函数
# ============================================================

def generate_noise(wav_list, snr_db, out_dir, noise_pool):
    """对每条语音叠加随机 MUSAN 噪声"""
    os.makedirs(out_dir, exist_ok=True)
    new_list = []

    for wav_path in tqdm(wav_list, desc=f"Noise SNR={snr_db}dB"):
        clean = load_wav(wav_path)

        # 随机选一段噪声
        noise_file = random.choice(noise_pool)
        noise_full = load_wav(noise_file)

        # 从噪声中随机截取和语音等长的片段
        if len(noise_full) < len(clean):
            noise = np.tile(noise_full, int(np.ceil(len(clean) / len(noise_full))))[:len(clean)]
        else:
            start = random.randint(0, len(noise_full) - len(clean))
            noise = noise_full[start:start + len(clean)]

        noise_scaled = adjust_snr(clean, noise, snr_db)
        mixed = clean + noise_scaled

        # 防削波
        max_val = np.max(np.abs(mixed))
        if max_val > 0.99:
            mixed = mixed / max_val * 0.99

        # 保持和原文件相同的目录结构
        rel_path = os.path.relpath(wav_path, "data/test")
        out_path = os.path.join(out_dir, rel_path)
        save_wav(out_path, mixed)
        new_list.append(out_path)

    return new_list


def generate_reverb(wav_list, out_dir, rir_pool):
    """对每条语音卷积随机 RIR"""
    from scipy.signal import fftconvolve
    os.makedirs(out_dir, exist_ok=True)
    new_list = []

    for wav_path in tqdm(wav_list, desc="Reverb"):
        clean = load_wav(wav_path)

        # 随机选一个 RIR
        rir_path = random.choice(rir_pool)
        try:
            rir, rir_sr = sf.read(rir_path)
        except Exception:
            continue
        if rir_sr != SAMPLE_RATE:
            continue
        if rir.ndim > 1:
            rir = rir[:, 0]
        rir = rir.astype(np.float32)

        # 卷积
        full_reverb = fftconvolve(clean, rir)
        # 保留原始长度 + 0.5 秒的混响拖尾（让混响效果可感知）
        tail_samples = int(0.5 * SAMPLE_RATE)
        reverb = full_reverb[:len(clean) + tail_samples]

        # 仅防削波，不做能量归一化（保留混响的自然衰减特征）
        max_val = np.max(np.abs(reverb))
        if max_val > 0.99:
            reverb = reverb / max_val * 0.99

        rel_path = os.path.relpath(wav_path, "data/test")
        out_path = os.path.join(out_dir, rel_path)
        save_wav(out_path, reverb)
        new_list.append(out_path)

    return new_list


def generate_overlap(wav_list, out_dir, num_speakers, tir_db, utt2spk_path="data/test/utt2spk",
                     all_wav_list=None):
    """生成多人重叠语音：
    - 目标说话人 + (num_speakers-1) 个干扰说话人
    - TIR = 目标能量 / 干扰能量
    """
    os.makedirs(out_dir, exist_ok=True)
    new_list = []

    # 构建 spk → [wav_paths] 映射
    if all_wav_list is None:
        all_wav_list = wav_list

    # 读 utt2spk
    spk_map = {}  # utt_id → spk_id
    with open(utt2spk_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                spk_map[parts[0]] = parts[1]

    # spk → [wav paths]
    spk_to_wavs = {}
    for wp in all_wav_list:
        utt_id = os.path.splitext(os.path.basename(wp))[0]
        spk = spk_map.get(utt_id, "unknown")
        spk_to_wavs.setdefault(spk, []).append(wp)

    all_speakers = list(spk_to_wavs.keys())

    for wav_path in tqdm(wav_list, desc=f"Overlap {num_speakers}p"):
        target = load_wav(wav_path)
        utt_id = os.path.splitext(os.path.basename(wav_path))[0]
        target_spk = spk_map.get(utt_id, "unknown")

        # 随机选 num_speakers-1 个不同的干扰说话人
        other_spks = [s for s in all_speakers if s != target_spk]
        if len(other_spks) < num_speakers - 1:
            continue
        chosen_interferers = random.sample(other_spks, num_speakers - 1)

        mixed = target.copy()
        for int_spk in chosen_interferers:
            int_wav = random.choice(spk_to_wavs[int_spk])
            interference = load_wav(int_wav)

            # 对齐长度
            if len(interference) < len(mixed):
                interference = np.tile(interference, int(np.ceil(len(mixed) / len(interference))))[:len(mixed)]
            else:
                start = random.randint(0, len(interference) - len(mixed))
                interference = interference[start:start + len(mixed)]

            # 调整到指定的 TIR
            interference_scaled = adjust_snr(target, interference, tir_db)
            mixed = mixed + interference_scaled

        max_val = np.max(np.abs(mixed))
        if max_val > 0.99:
            mixed = mixed / max_val * 0.99

        rel_path = os.path.relpath(wav_path, "data/test")
        out_path = os.path.join(out_dir, rel_path)
        save_wav(out_path, mixed)
        new_list.append(out_path)

    return new_list


def generate_short(wav_list, out_dir, duration):
    """裁剪到指定时长"""
    os.makedirs(out_dir, exist_ok=True)
    new_list = []
    target_samples = int(duration * SAMPLE_RATE)

    for wav_path in tqdm(wav_list, desc=f"Short {duration}s"):
        clean = load_wav(wav_path)

        if len(clean) >= target_samples:
            # 语音够长：取中间一段
            start = (len(clean) - target_samples) // 2
            short = clean[start:start + target_samples]
        else:
            # 语音太短：末尾补零
            short = np.zeros(target_samples, dtype=np.float32)
            short[:len(clean)] = clean

        rel_path = os.path.relpath(wav_path, "data/test")
        base, ext = os.path.splitext(rel_path)
        out_path = os.path.join(out_dir, f"{base}_{duration}s{ext}")
        save_wav(out_path, short)
        new_list.append(out_path)

    return new_list


# ============================================================
# 主流程
# ============================================================

def main():
    # 读取干净语音列表
    with open(CLEAN_WAV_LIST) as f:
        clean_list = [line.strip() for line in f if line.strip()]
    print(f"干净语音: {len(clean_list)} 条")

    scenarios = [args.scenario] if args.scenario != "all" else [
        "noise_m5db_music", "noise_m5db_noise", "noise_m5db_speech",
        "noise_0db_music", "noise_0db_noise", "noise_0db_speech",
        "reverb",
        "overlap_2", "overlap_3",
        "short_1s", "short_3s", "short_5s"
    ]

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"  生成场景: {scenario}")
        print(f"{'='*60}")

        if scenario.startswith("noise"):
            snr = -5 if "m5db" in scenario else 0
            noise_type = scenario.split("_")[-1]  # music / noise / speech
            out_dir = os.path.join(OUT_BASE, f"test_{scenario}")

            # 按噪声类型选择对应的测试池
            noise_type_to_file = {
                "music":  NOISE_TEST_MUSIC,
                "noise":  NOISE_TEST_NOISE,
                "speech": NOISE_TEST_SPEECH,
            }
            noise_list_file = noise_type_to_file[noise_type]

            if not os.path.exists(noise_list_file):
                print(f"  [ERROR] 噪声列表不存在: {noise_list_file}")
                print(f"  请先运行: python split_noise_rir.py")
                continue
            with open(noise_list_file) as f:
                noise_pool = [line.strip() for line in f if line.strip()]
            print(f"  噪声类型: {noise_type}, SNR={snr}dB, 测试噪声池: {len(noise_pool)} 条（与训练噪声物理隔离）")
            if not noise_pool:
                print("  [ERROR] 测试噪声池为空")
                continue
            new_list = generate_noise(clean_list, snr, out_dir, noise_pool)

        elif scenario == "reverb":
            out_dir = os.path.join(OUT_BASE, "test_reverb")
            # 从测试专用 RIR 池加载（真实房间 RIR，与训练的模拟 RIR 物理隔离）
            if not os.path.exists(RIR_TEST_LIST):
                print(f"  [ERROR] 测试混响列表不存在: {RIR_TEST_LIST}")
                print(f"  请先运行: python split_noise_rir.py")
                continue
            with open(RIR_TEST_LIST) as f:
                rir_pool = [line.strip() for line in f if line.strip()]
            print(f"  测试混响池: {len(rir_pool)} 条（真实房间测量，训练未见）")
            if not rir_pool:
                print("  [ERROR] 测试混响池为空")
                continue
            new_list = generate_reverb(clean_list, out_dir, rir_pool)

        elif scenario.startswith("overlap"):
            num_spk = 2 if "2" in scenario else 3
            out_dir = os.path.join(OUT_BASE, f"test_{scenario}")
            new_list = generate_overlap(clean_list, out_dir, num_spk, tir_db=0,
                                        utt2spk_path="data/test/utt2spk",
                                        all_wav_list=clean_list)

        elif scenario.startswith("short"):
            dur = float(scenario.split("_")[-1].replace("s", ""))
            out_dir = os.path.join(OUT_BASE, f"test_{scenario}")
            new_list = generate_short(clean_list, out_dir, dur)

        else:
            continue

        # 保存 wav_list.txt
        list_file = os.path.join(out_dir, "wav_list.txt")
        with open(list_file, "w") as f:
            for p in new_list:
                f.write(p + "\n")
        print(f"  生成: {len(new_list)} 条 → {out_dir}/")
        print(f"  列表: {list_file}")


if __name__ == "__main__":
    main()
