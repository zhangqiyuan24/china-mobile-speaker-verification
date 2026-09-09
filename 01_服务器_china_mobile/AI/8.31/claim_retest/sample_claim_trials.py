#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""sample_claim_trials.py — #37 claim 口径重测：nontarget 对抽样（seed 固定可复现）
从官方 trials_cross_device(3s) 抽 30,000 条 nontarget 对；target 对全量保留
（claim=真身，与交付 oracle 增强恒等，无需新算）。trials 两条件共用。
"""
import json
import os
import random
from collections import defaultdict

SV = "/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa"
TRIALS = os.path.join(SV, "data/data_for_test/clean/3s/trials_cross_device")
HERE = os.path.dirname(os.path.abspath(__file__))
N = 30000


def spk_of(stem):
    return "_".join(stem.split("_")[:3])


def main():
    random.seed(2026)
    n_tgt = n_non = 0
    non = []
    with open(TRIALS) as f:
        for line in f:
            e, t, lab = line.split()
            if lab == "target":
                n_tgt += 1
            else:
                non.append((e, t))
                n_non += 1
    sample = random.sample(non, N)
    groups = defaultdict(set)
    for e, t in sample:
        groups[spk_of(e)].add(t)
    with open(os.path.join(HERE, "sampled_nontarget.txt"), "w") as f:
        for e, t in sample:
            f.write(f"{e} {t} nontarget\n")
    with open(os.path.join(HERE, "claim_groups.json"), "w") as f:
        json.dump({k: sorted(v) for k, v in groups.items()}, f)
    print(f"targets={n_tgt} nontargets={n_non} sampled={len(sample)} "
          f"claims={len(groups)} combos={sum(len(v) for v in groups.values())} "
          f"test_utts={len({t for _, t in sample})}")


if __name__ == "__main__":
    main()
