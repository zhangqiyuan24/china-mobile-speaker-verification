#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_proxy_val.py — 构造 proxy-official 验证集（8.27 Phase 0/1 共用）

设计目标（对齐官方协议 trials_cross_device 的关键要素）：
  1. 说话人：100 个从未被任何微调见过的池内说话人（train_utts.csv 9900 之外）
  2. 跨设备：enrol=Device03（clean 3s），test=Device06/07（每人各 2 条）
  3. held-out 噪声：noise_test_freesound.txt（168 条，与训练 615 条不相交）
  4. SNR 档：0 / -5 dB（与官方验收档一致），snr_mix 与训练同实现
  5. trial 构造：3 列官方格式，target:nontarget ≈ 1:5.5（对齐官方 211400:1166806）

输出（OUT 下）：
  enrol/                      每人 2 条 Device03 clean 3s（trials 只用第 1 条；条件化可用 2 条）
  noise_0db/ noise_m5db/      每人 4 条带噪 3s（文件名 {spk}__{原stem}.wav）
  trials_noise_0db.txt trials_noise_m5db.txt
  meta.json

用法: python build_proxy_val.py [--n-test-per-spk 4 --nontarget-ratio 5.5]
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torchaudio

_HERE = os.path.dirname(os.path.abspath(__file__))
_AI = os.path.dirname(_HERE)
for _p in (os.path.join(_AI, "8.18"), os.path.join(_AI, "8.21"), _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from trainer_spk_preserve import snr_mix  # 与训练同实现（RMS SNR + 防削波）

SV_ECAPA = ("/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker"
            "/egs/3dspeaker/sv-ecapa")
TRAIN_DIR = os.path.join(SV_ECAPA, "data", "data_for_train", "train")
USED_CSV = "/data/home/ligangrui/china_mobile/AI/8.22/exp/data_90k/data/train_utts.csv"
NOISE_TEST_LIST = os.path.join(SV_ECAPA, "data", "divide", "noise_test_freesound.txt")
OUT = os.path.join(_HERE, "exp", "proxy")

SR, SEC = 16000, 3
TARGET_LEN = SR * SEC


def load_wav_np(path):
    w, _ = torchaudio.load(path)
    return w[0].numpy()


def save_wav(path, arr):
    import torch
    torchaudio.save(path, torch.from_numpy(arr).unsqueeze(0), SR,
                    encoding="PCM_S", bits_per_sample=16)


def crop3s(arr, rng):
    if len(arr) >= TARGET_LEN:
        st = rng.randint(0, len(arr) - TARGET_LEN)
        return arr[st:st + TARGET_LEN]
    return np.pad(arr, (0, TARGET_LEN - len(arr)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-enrol", type=int, default=2)
    ap.add_argument("--n-test-dev", type=int, default=2, help="每设备取几条 test")
    ap.add_argument("--nontarget-ratio", type=float, default=5.5)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    for d in ("enrol", "noise_0db", "noise_m5db"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)

    used = set()
    with open(USED_CSV, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i and line.strip():
                used.add(line.split(",")[1])
    pool = sorted(os.listdir(TRAIN_DIR))
    unseen = [s for s in pool if s not in used and s.startswith("3D_SPK")]
    print(f"[pool] {len(pool)} spks, 已用 {len(used)}, 未见 {len(unseen)}", flush=True)

    noise_paths = [ln.strip() for ln in open(NOISE_TEST_LIST) if ln.strip()]
    noise_paths = [os.path.join(SV_ECAPA, p) for p in noise_paths]
    print(f"[noise] held-out {len(noise_paths)} 条", flush=True)

    picked, skipped = [], 0
    # 未见说话人只有 Device03/04/05（官方测试的 D06/07 不在此池）——
    # 保留"1 注册设备 → 2 不同测试设备"的跨设备结构，通道差异在 README 记录
    TEST_DEVS = (4, 5)
    for spk in unseen:
        utts = sorted(os.listdir(os.path.join(TRAIN_DIR, spk)))
        by_dev = {d: [u for u in utts if f"Device{d:02d}" in u]
                  for d in (3,) + TEST_DEVS}
        if (len(by_dev[3]) < args.n_enrol
                or any(len(by_dev[d]) < args.n_test_dev for d in TEST_DEVS)):
            skipped += 1
            continue
        picked.append((spk, by_dev))
    print(f"[pick] {len(picked)} spks 可用（缺设备跳过 {skipped}）"
          f" enrol=Device03 test=Device{'/'.join(str(d) for d in TEST_DEVS)}",
          flush=True)

    # ---- enrol 侧（clean 3s）----
    enrol0 = {}          # spk -> trials 用的 enrol stem
    for spk, by_dev in picked:
        for u in by_dev[3][:args.n_enrol]:
            arr = crop3s(load_wav_np(os.path.join(TRAIN_DIR, spk, u)), rng)
            stem = f"{spk}__{os.path.splitext(u)[0]}"
            save_wav(os.path.join(OUT, "enrol", stem + ".wav"), arr)
            enrol0.setdefault(spk, stem)

    # ---- test 侧（Device04/05 各 n 条 × 2 SNR 档）----
    test_stems = {"noise_0db": [], "noise_m5db": []}
    for spk, by_dev in picked:
        tests = by_dev[4][:args.n_test_dev] + by_dev[5][:args.n_test_dev]
        for u in tests:
            clean = load_wav_np(os.path.join(TRAIN_DIR, spk, u))
            for cond, snr in (("noise_0db", 0), ("noise_m5db", -5)):
                noise = load_wav_np(rng.choice(noise_paths))
                noisy, _ = snr_mix(clean, noise, snr, TARGET_LEN)
                stem = f"{spk}__{os.path.splitext(u)[0]}"
                save_wav(os.path.join(OUT, cond, stem + ".wav"), noisy)
                test_stems[cond].append(stem)

    # ---- trials（官方 3 列格式, 1:5.5）----
    for cond in ("noise_0db", "noise_m5db"):
        stems_by_spk = {}
        for s in test_stems[cond]:
            stems_by_spk.setdefault(s.split("__")[0], []).append(s)
        spks = sorted(stems_by_spk)
        targets, nontargets = [], []
        for spk in spks:
            e = enrol0[spk]
            for t in stems_by_spk[spk]:
                targets.append((e, t))
        n_nt = int(len(targets) * args.nontarget_ratio)
        seen = set()
        while len(nontargets) < n_nt:
            spk = rng.choice(spks)
            other = rng.choice([s for s in spks if s != spk])
            e = enrol0[spk]
            t = rng.choice(stems_by_spk[other])
            if (e, t) in seen:
                continue
            seen.add((e, t))
            nontargets.append((e, t))
        trials = [(e, t, "target") for e, t in targets] + \
                 [(e, t, "nontarget") for e, t in nontargets]
        rng.shuffle(trials)
        path = os.path.join(OUT, f"trials_{cond}.txt")
        with open(path, "w") as f:
            for e, t, lab in trials:
                f.write(f"{e} {t} {lab}\n")
        print(f"[trials] {cond}: {len(targets)} target + {len(nontargets)} nontarget"
              f" -> {path}", flush=True)

    meta = {"n_spk": len(picked), "n_enrol_per_spk": args.n_enrol,
            "n_test_per_spk": 2 * args.n_test_dev,
            "nontarget_ratio": args.nontarget_ratio, "seed": args.seed,
            "noise_pool": len(noise_paths)}
    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[done] {meta}", flush=True)


if __name__ == "__main__":
    main()
