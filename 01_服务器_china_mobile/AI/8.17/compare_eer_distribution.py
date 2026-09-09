#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比降噪前后 target / nontarget 的 cosine 分数分布，定位 EER 退化的原因。

对比三组 test embedding:
  - clean:     exp/clean_bench/eres2net/embeddings_3s          (baseline)
  - noisy:     exp/clean_bench/noise_0db/eres2net/embeddings_3s (带噪)
  - denoised:  exp/clean_bench/denoised_noise_0db/eres2net/embeddings_3s (降噪后)

enrollment 都用 clean。对每组统计 target / nontarget 的平均 cosine + 重叠程度。

用法 (在 sv-ecapa 目录下运行):
  python /path/to/compare_eer_distribution.py
"""

import os
import re
import numpy as np
from kaldiio import ReadHelper


def collect_emb(data_dir):
    """读取一个 embedding 目录，返回 {utt_id: embedding}"""
    d = {}
    for ark in sorted(os.listdir(data_dir)):
        if ark.endswith('.ark'):
            with ReadHelper(f'ark:{os.path.join(data_dir, ark)}') as reader:
                for key, array in reader:
                    d[key] = array
    return d


def main():
    enroll_dir = "exp/clean_bench/eres2net/embeddings_3s"
    trial_file = "data/data_for_test/clean/3s/trials_cross_device"

    test_groups = {
        "clean": enroll_dir,
        "noisy": "exp/clean_bench/noise_0db/eres2net/embeddings_3s",
        "denoised": "exp/clean_bench/denoised_noise_0db/eres2net/embeddings_3s",
    }

    print("=" * 70)
    print("  降噪前后 target / nontarget 分数对比")
    print("=" * 70)

    enroll_emb = collect_emb(enroll_dir)

    # 读 trial
    trials = []
    with open(trial_file) as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 3:
                trials.append((p[0], p[1], p[2]))
    print(f"\n  trial 总数: {len(trials)}")

    for name, test_dir in test_groups.items():
        test_emb = collect_emb(test_dir)
        target_scores = []
        nontarget_scores = []
        for enroll_id, test_id, label in trials:
            if enroll_id not in enroll_emb or test_id not in test_emb:
                continue
            a = enroll_emb[enroll_id]
            b = test_emb[test_id]
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            if label in ("target", "1"):
                target_scores.append(cos)
            else:
                nontarget_scores.append(cos)

        t = np.array(target_scores)
        nt = np.array(nontarget_scores)
        gap = t.mean() - nt.mean()  # target 与 nontarget 的分离度

        print(f"\n  [{name}]  target={len(t)}  nontarget={len(nt)}")
        print(f"    target    均值={t.mean():.4f}  标准差={t.std():.4f}")
        print(f"    nontarget 均值={nt.mean():.4f}  标准差={nt.std():.4f}")
        print(f"    target-nontarget 分离度 = {gap:.4f}  (越大越好)")

    print("\n" + "=" * 70)
    print("  判定: 若 denoised 的分离度 < noisy 的分离度 → 降噪损伤了 speaker 区分度")
    print("=" * 70)


if __name__ == "__main__":
    main()
