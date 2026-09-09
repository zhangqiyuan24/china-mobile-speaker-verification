#!/usr/bin/env python
"""
准备微调数据：从训练集中选 60,000 条，按 speaker 8:2 划分。
检查与测试集 speaker 不重叠。

用法：python local/prepare_finetune_data.py \
        --train_dir data/data_for_train/train \
        --test_speakers_file data/data_for_test/test_speakers.txt \
        --output_dir data \
        --num_utts 60000 \
        --seed 2026
"""

import os, sys, csv, random, argparse
from pathlib import Path
from collections import defaultdict


def collect_wavs(train_dir):
    """→ [(wav_path, spk_id), ...]"""
    data = []
    for spk in sorted(os.listdir(train_dir)):
        spk_path = os.path.join(train_dir, spk)
        if not os.path.isdir(spk_path):
            continue
        for f in sorted(os.listdir(spk_path)):
            if f.endswith('.wav'):
                data.append((os.path.join(spk_path, f), spk))
    return data


def load_test_speakers(path):
    """从文件读测试集 speaker ID 列表，每行一个"""
    if not path or not os.path.exists(path):
        return set()
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--test_speakers_file', default='')
    parser.add_argument('--output_dir', default='data')
    parser.add_argument('--num_utts', type=int, default=60000)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed)

    # Collect all training data
    print(f"[1] 扫描训练数据: {args.train_dir}")
    all_data = collect_wavs(args.train_dir)
    print(f"    总计: {len(all_data)} 条语音")

    # Group by speaker
    spk2utts = defaultdict(list)
    for wav, spk in all_data:
        spk2utts[spk].append(wav)

    all_spks = sorted(spk2utts.keys())
    print(f"    说话人数: {len(all_spks)}")

    # Load test speakers
    test_spks = load_test_speakers(args.test_speakers_file)
    print(f"    测试集 speaker 数: {len(test_spks)}")

    # Remove test speakers from pool
    overlap = set(all_spks) & test_spks
    if overlap:
        print(f"    [ERROR] 训练集与测试集 speaker 重叠: {overlap}")
        sys.exit(1)
    print(f"    [✓] 训练/测试 speaker 无重叠")

    # Select speakers until reaching target utterance count
    random.shuffle(all_spks)
    selected_utts = 0
    selected_spks = []
    for spk in all_spks:
        if selected_utts >= args.num_utts:
            break
        selected_spks.append(spk)
        selected_utts += len(spk2utts[spk])

    print(f"\n[2] 选取 speakers: {len(selected_spks)}, utterances: {selected_utts}")

    # 8:2 speaker-level split
    random.shuffle(selected_spks)
    n_train_spks = int(len(selected_spks) * 0.8)
    train_spks = set(selected_spks[:n_train_spks])
    valid_spks = set(selected_spks[n_train_spks:])

    print(f"    Train speakers: {len(train_spks)}, Valid speakers: {len(valid_spks)}")
    assert len(train_spks & valid_spks) == 0, "Train/Valid speaker overlap!"

    # Write CSV files
    os.makedirs(args.output_dir, exist_ok=True)

    def write_csv(spk_set, fname):
        path = os.path.join(args.output_dir, fname)
        count = 0
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['ID', 'path', 'spk'])
            for spk in sorted(spk_set):
                for wav_path in spk2utts[spk]:
                    utt_id = f"{spk}_{Path(wav_path).stem}"
                    w.writerow([utt_id, wav_path, spk])
                    count += 1
        return count

    n1 = write_csv(train_spks, 'train.csv')
    n2 = write_csv(valid_spks, 'valid.csv')
    print(f"\n[3] 输出:")
    print(f"    {args.output_dir}/train.csv: {n1} 条, {len(train_spks)} speakers")
    print(f"    {args.output_dir}/valid.csv: {n2} 条, {len(valid_spks)} speakers")

    # Build spk2utt mapping for overlap augmentation
    spk2utt = {}
    for spk in train_spks:
        spk2utt[spk] = spk2utts[spk]
    import json
    spk2utt_path = os.path.join(args.output_dir, 'train_spk2utt.json')
    with open(spk2utt_path, 'w') as f:
        json.dump(spk2utt, f)
    print(f"    {spk2utt_path}: overlap 干扰池 ({len(spk2utt)} speakers)")

    # Save test speakers for reference
    test_list_path = os.path.join(args.output_dir, 'test_speakers.txt')
    with open(test_list_path, 'w') as f:
        for s in sorted(test_spks):
            f.write(s + '\n')
    print(f"    {test_list_path}: 测试集 speaker 列表")


if __name__ == '__main__':
    main()
