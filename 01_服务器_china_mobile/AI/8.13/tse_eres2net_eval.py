#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WeSep TSE 前端 + ERes2Net-large — overlap 评测

流程 (每个 trial):
  enroll_id (clean A) ──┬──→ WeSep reference ──┐
                        │                       ├→ extracted target wav
  test_id (mixture) ────┘──→ WeSep mixture ────┘
                                                   ↓
                                         ERes2Net → test embedding
                                                   ↓
                       enroll → ERes2Net → enroll embedding (预计算缓存)
                                                   ↓
                                         cosine → EER

用法:
  # 全量
  python tse_eres2net_eval.py --condition overlap2_en --out_dir ./eval_out

  # 平衡抽样 (100 target + 100 nontarget, 固定 seed)
  python tse_eres2net_eval.py --condition overlap2_en --balanced --out_dir ./eval_out
"""

import os
import sys
import argparse
import numpy as np
import yaml
from tqdm import tqdm

import common


def build_stem_map(wav_list_path, base_dir=None):
    """wav_list.txt (每行一个路径) → {stem: path}, 检测重复"""
    m = {}
    with open(wav_list_path) as f:
        for line in f:
            p = line.strip()
            if not p:
                continue
            # wav_list 里的路径是相对 sv-ecapa 的，转成绝对路径
            if base_dir and not os.path.isabs(p):
                p = os.path.join(base_dir, p)
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem in m:
                raise ValueError(f"[ERROR] wav_list 存在重复 utt-id: {stem}")
            m[stem] = p
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--condition", default="overlap2_en",
                        help="overlap2_cn / overlap2_en / overlap3_cn / overlap3_en")
    parser.add_argument("--balanced", action="store_true",
                        help="平衡抽样 (固定 seed, target/nontarget 各 N 条)")
    parser.add_argument("--n_target", type=int, default=100)
    parser.add_argument("--n_nontarget", type=int, default=100)
    parser.add_argument("--out_dir", default="./eval_out")
    args = parser.parse_args()

    with open(args.config) as f:
        raw_cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = common.resolve_paths(raw_cfg)

    os.makedirs(args.out_dir, exist_ok=True)
    dur = cfg["data"]["duration"]
    sv_ecapa = cfg["data"]["sv_ecapa"]
    dft = os.path.join(sv_ecapa, cfg["data"]["data_for_test"])

    # ---- 路径 ----
    clean_wav_list = os.path.join(dft, "clean", dur, "wav_list.txt")
    overlap_wav_list = os.path.join(dft, args.condition, dur, "wav_list.txt")
    trial_file = os.path.join(sv_ecapa, cfg["data"]["trial"])

    if not os.path.exists(overlap_wav_list):
        print(f"[ERROR] 找不到 {overlap_wav_list}")
        sys.exit(1)

    print("=" * 60)
    print(f"  WeSep TSE + ERes2Net  overlap 评测")
    print(f"  条件: {args.condition}  时长: {dur}")
    print("=" * 60)

    # ---- 映射 ----
    print("[1] 建立 utt_id → wav 映射 ...")
    clean_map = build_stem_map(clean_wav_list, sv_ecapa)
    overlap_map = build_stem_map(overlap_wav_list, sv_ecapa)
    print(f"    clean: {len(clean_map)}, overlap: {len(overlap_map)}")

    # ---- 读 trial ----
    print("[2] 读 trial ...")
    trials = []
    with open(trial_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            trials.append((parts[0], parts[1], parts[2]))

    if args.balanced:
        trials = common.balanced_sample_trials(
            trials, n_target=args.n_target, n_nontarget=args.n_nontarget, seed=2026)
        print(f"    平衡抽样: {len(trials)} trials "
              f"(target={args.n_target}, nontarget={args.n_nontarget})")
    else:
        print(f"    全量: {len(trials)} trials")

    # ---- 加载模型 ----
    print("[3] 加载 WeSep + ERes2Net ...")
    extractor = common.load_wesep_extractor(cfg)
    eres2net, fbank, device = common.load_eres2net(cfg)

    # ---- 预计算 enrollment embedding ----
    print("[4] 预计算 enrollment embedding ...")
    unique_enroll = sorted(set(t[0] for t in trials))
    enroll_emb = {}
    for enroll_id in tqdm(unique_enroll, desc="enroll embed"):
        if enroll_id not in clean_map:
            continue
        wav = common.load_wav_mono(clean_map[enroll_id])
        emb = common.extract_embeddings_batch(eres2net, fbank, device, [wav])[0]
        enroll_emb[enroll_id] = emb
    print(f"    预计算 {len(enroll_emb)} 个 enrollment embedding")

    # ---- 逐 batch 评测 ----
    print("[5] 批量 TSE 评测 ...")
    batch_size = cfg["eval"].get("batch_size", 16)
    target_scores = []
    nontarget_scores = []

    def process_batch(batch_trials):
        mix_wavs = []
        enroll_wavs = []
        valid = []
        for enroll_id, test_id, label in batch_trials:
            if enroll_id not in enroll_emb or test_id not in overlap_map:
                continue
            mix_wavs.append(common.load_wav_mono(overlap_map[test_id]))
            enroll_wavs.append(common.load_wav_mono(clean_map[enroll_id]))
            valid.append((enroll_id, test_id, label))
        if not valid:
            return

        extracted = common.tse_extract_batch(extractor, mix_wavs, enroll_wavs)
        test_embs = common.extract_embeddings_batch(eres2net, fbank, device, extracted)

        for (enroll_id, test_id, label), test_emb in zip(valid, test_embs):
            score = common.cosine(enroll_emb[enroll_id], test_emb)
            if label in ("target", "1"):
                target_scores.append(score)
            else:
                nontarget_scores.append(score)

    n_batches = (len(trials) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(trials), batch_size), total=n_batches, desc="tse scoring"):
        process_batch(trials[i:i + batch_size])

    # ---- EER ----
    target_scores = np.array(target_scores)
    nontarget_scores = np.array(nontarget_scores)
    if len(target_scores) == 0 or len(nontarget_scores) == 0:
        print("[ERROR] target 或 nontarget 为空，无法计算 EER")
        sys.exit(1)

    eer = common.compute_eer(target_scores, nontarget_scores)
    print("\n" + "=" * 60)
    print(f"  TSE 前端 EER ({args.condition}, {dur}): {eer:.4f}%")
    print(f"  target={len(target_scores)}, nontarget={len(nontarget_scores)}")
    print("=" * 60)

    np.save(os.path.join(args.out_dir, "target_scores.npy"), target_scores)
    np.save(os.path.join(args.out_dir, "nontarget_scores.npy"), nontarget_scores)
    with open(os.path.join(args.out_dir, "eer.txt"), "w") as f:
        f.write(f"{eer:.4f}\n")
    print(f"\n结果已保存到 {args.out_dir}/")


if __name__ == "__main__":
    main()
