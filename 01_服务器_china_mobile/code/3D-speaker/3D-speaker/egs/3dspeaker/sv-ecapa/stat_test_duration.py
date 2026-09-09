#!/usr/bin/env python
"""
统计 data/test 下所有 wav 的采样率和时长分布。
用法：在 sv-ecapa 目录下运行 python stat_test_duration.py
"""

import os
import sys
import soundfile as sf
from tqdm import tqdm
from collections import Counter

TEST_DIR = "data/test"

def main():
    # 收集所有 wav
    wavs = []
    for root, dirs, files in os.walk(TEST_DIR):
        for f in files:
            if f.endswith(".wav"):
                wavs.append(os.path.join(root, f))

    print(f"共找到 {len(wavs)} 条 wav 文件\n")

    sr_counter = Counter()
    dur_bins = Counter()  # 按秒分桶: 0-1, 1-2, 2-3, 3-4, 4+
    total_dur = 0.0
    dur_list = []

    for wav in tqdm(wavs, desc="分析中"):
        try:
            info = sf.info(wav)
            sr = info.samplerate
            dur = info.duration
        except Exception as e:
            print(f"  [跳过] {wav}: {e}")
            continue

        sr_counter[sr] += 1
        total_dur += dur
        dur_list.append(dur)

        # 分桶
        if dur < 1:
            dur_bins["0-1s"] += 1
        elif dur < 2:
            dur_bins["1-2s"] += 1
        elif dur < 3:
            dur_bins["2-3s"] += 1
        elif dur < 4:
            dur_bins["3-4s"] += 1
        elif dur < 5:
            dur_bins["4-5s"] += 1
        else:
            dur_bins["5s+"] += 1

    # 打印结果
    print(f"\n{'='*50}")
    print(f"  采样率分布")
    print(f"{'='*50}")
    for sr, cnt in sorted(sr_counter.items()):
        print(f"  {sr} Hz: {cnt} 条")

    print(f"\n{'='*50}")
    print(f"  时长分布")
    print(f"{'='*50}")
    for bucket in ["0-1s", "1-2s", "2-3s", "3-4s", "4-5s", "5s+"]:
        cnt = dur_bins.get(bucket, 0)
        bar = "█" * (cnt // 500)
        print(f"  {bucket:6s}: {cnt:6d} 条  {bar}")

    print(f"\n{'='*50}")
    print(f"  统计摘要")
    print(f"{'='*50}")
    dur_arr = sorted(dur_list)
    n = len(dur_arr)
    print(f"  总条数:   {n}")
    print(f"  总时长:   {total_dur:.1f} 秒 ({total_dur/3600:.2f} 小时)")
    print(f"  最短:     {dur_arr[0]:.2f}s")
    print(f"  最长:     {dur_arr[-1]:.2f}s")
    print(f"  平均:     {total_dur/n:.2f}s")
    print(f"  中位数:   {dur_arr[n//2]:.2f}s")
    if n >= 10:
        print(f"  P10:      {dur_arr[n//10]:.2f}s")
        print(f"  P90:      {dur_arr[n*9//10]:.2f}s")
    print(f"  ≥1s:      {sum(1 for d in dur_list if d >= 1)} 条")
    print(f"  ≥3s:      {sum(1 for d in dur_list if d >= 3)} 条")
    print(f"  ≥5s:      {sum(1 for d in dur_list if d >= 5)} 条")


if __name__ == "__main__":
    main()
