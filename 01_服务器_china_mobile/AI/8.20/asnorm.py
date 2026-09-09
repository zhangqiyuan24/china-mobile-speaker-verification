# -*- coding: utf-8 -*-
"""
E1/E2 后端：AS-Norm（自适应分数归一化）独立脚本，三个子功能：

  1) --make-cohort   从训练说话人（train_utts.csv）确定性抽 cohort 语料，写 wav 清单。
                     cohort 与正式 test 说话人天然不重叠，无需排除同说话人噪声。
  2) --select-k      在验证集上选 K：用选定 checkpoint 提验证 bench embedding +
                     cohort embedding（需先由 evaluate_robust.py --extract-only 提好），
                     对 K ∈ {100,200,300} 计算 AS-Norm 后 EER，选噪声两档之和最小的 K。
  3) --apply         对正式评测的 (enrol ark, test ark, trials) 施加 AS-Norm，
                     报告 raw / AS-Norm 的 EER 与 minDCF，写 .asnorm.score 文件。

用法示例：
  python asnorm.py --config config_robust.toml --make-cohort --cohort-spks 500 --cohort-utts 2
  python evaluate_robust.py --config config_robust.toml --checkpoint exp/robust/checkpoints/checkpoint_best.tar \
      --extract-only --crop-seconds 3 \
      --wav-list exp/robust/data/asnorm/cohort_wavs.txt \
      --out-dir exp/robust/eval/finetuned/embeddings/cohort
  python asnorm.py --config config_robust.toml --select-k
  python asnorm.py --config config_robust.toml --apply --k auto
"""

import os
import json
import glob
import argparse
import numpy as np
import torch
import toml

from common_utils import load_trials, eer_minDCF, load_ark_dir, extract_embeddings, setup_speakerlab
from validate_robust import load_model_for_validation, build_fbank, COND_DIRS

KS = (100, 200, 300)


# ============================================================
# cohort 构造
# ============================================================

def make_cohort(data_dir, out_path, cohort_spks, cohort_utts):
    """从 train_utts.csv 抽 cohort：确定性（spk 排序取前 M 人 × 每人前 N 句）。"""
    by_spk = {}
    with open(os.path.join(data_dir, "train_utts.csv")) as f:
        next(f)
        for line in f:
            utt_id, spk, path = line.strip().split(",")
            by_spk.setdefault(spk, []).append((utt_id, path))
    spks = sorted(by_spk)[:cohort_spks]
    wavs = []
    for s in spks:
        for utt_id, path in sorted(by_spk[s])[:cohort_utts]:
            wavs.append(path)
    with open(out_path, "w") as f:
        f.write("\n".join(wavs) + "\n")
    print(f"[cohort] {len(spks)} 人 × {cohort_utts} 句 = {len(wavs)} 条 -> {out_path}")
    return wavs


# ============================================================
# AS-Norm 核心
# ============================================================

def _norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def cohort_stats(embs, cohort, k):
    """embs [N,D], cohort [C,D]（均已归一化）。返回 (mu, std) [N]，top-K 均值/标准差。"""
    scores = embs @ cohort.T                       # [N, C]
    k = min(k, scores.shape[1])
    part = np.sort(scores, axis=1)[:, -k:]         # 每行 top-K
    return part.mean(axis=1), part.std(axis=1)


def asnorm_scores(enrol_embs, test_embs, cohort, k):
    """对称 AS-Norm：0.5*[(s-mu_e)/std_e + (s-mu_t)/std_t]。输入未归一化也行，内部归一化。"""
    enrol_n, test_n, cohort_n = _norm(enrol_embs), _norm(test_embs), _norm(cohort)
    raw = np.sum(enrol_n * test_n, axis=1)
    mu_e, std_e = cohort_stats(enrol_n, cohort_n, k)
    mu_t, std_t = cohort_stats(test_n, cohort_n, k)
    return 0.5 * ((raw - mu_e) / (std_e + 1e-8) + (raw - mu_t) / (std_t + 1e-8))


# ============================================================
# select-k：验证集选 K
# ============================================================

def select_k(cfg, base, args, device):
    p_cfg = cfg["paths"]
    exp_dir = os.path.join(base, p_cfg["exp_dir"])
    data_dir = os.path.join(exp_dir, "data")
    bench_dir = os.path.join(data_dir, "val_bench")
    trials = load_trials(os.path.join(data_dir, "val_trials.txt"))
    ckpt = args.checkpoint or os.path.join(exp_dir, "checkpoints", "checkpoint_best.tar")
    cohort_dir = args.cohort_dir or os.path.join(exp_dir, "eval", "finetuned",
                                                 "embeddings", "cohort")
    cohort = load_ark_dir(cohort_dir)
    cohort_mat = np.stack(list(cohort.values()))
    print(f"[select-k] cohort = {cohort_mat.shape}")
    _meta_warn_if_mismatch(cohort_dir, ckpt)

    eres2net_ckpt = p_cfg["eres2net_ckpt"]
    if not os.path.isabs(eres2net_ckpt):
        eres2net_ckpt = os.path.join(base, eres2net_ckpt)
    model = load_model_for_validation(ckpt, eres2net_ckpt, device)
    fbank = build_fbank()

    enroll_files = sorted(glob.glob(os.path.join(bench_dir, "enroll", "*.wav")))
    ek, ee = extract_embeddings(model, fbank, enroll_files, device, args.batch_size, "selK-enroll")
    emap = dict(zip(ek, ee))

    print(f"\n{'K':>5} | {'clean EER%':>10} {'0dB EER%':>9} {'-5dB EER%':>9} | {'noise_sum':>9}")
    best = None
    table = {}
    cond_cache = {}
    for k in KS:
        row = {}
        noise_sum = 0.0
        for cond, subdir in COND_DIRS.items():
            if cond not in cond_cache:
                files = sorted(glob.glob(os.path.join(bench_dir, subdir, "*.wav")))
                tk, te = extract_embeddings(model, fbank, files, device, args.batch_size,
                                            f"selK-{cond}")
                cond_cache[cond] = dict(zip(tk, te))
            tmap = cond_cache[cond]
            e_idx, t_idx, labels = [], [], []
            for e, t, lab in trials:
                if e in emap and t in tmap:
                    e_idx.append(e)
                    t_idx.append(t)
                    labels.append(lab)
            E = np.stack([emap[e] for e in e_idx])
            T = np.stack([tmap[t] for t in t_idx])
            scores = asnorm_scores(E, T, cohort_mat, k)
            eer, mindcf = eer_minDCF(scores, np.array(labels))
            row[cond] = {"eer": eer, "mindcf": mindcf}
            if cond in ("0db", "m5db"):
                noise_sum += eer
        table[k] = {"metrics": row, "noise_sum": noise_sum}
        print(f"{k:>5} | {row['clean']['eer']:>10.4f} {row['0db']['eer']:>9.4f} "
              f"{row['m5db']['eer']:>9.4f} | {noise_sum:>9.4f}")
        if best is None or noise_sum < table[best]["noise_sum"]:
            best = k

    out = {"best_k": best, "table": table, "checkpoint": ckpt, "cohort_dir": cohort_dir}
    out_path = os.path.join(data_dir, "asnorm_selected_k.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n[select-k] 最优 K = {best}（noise_sum 最小）-> {out_path}")


# ============================================================
# apply：正式分数 AS-Norm
# ============================================================

def _print_meta(ark_dir, name):
    """打印 embedding 目录的 checkpoint 溯源，缺失时提醒。"""
    meta = os.path.join(ark_dir, "meta.json")
    if os.path.isfile(meta):
        print(f"[apply] {name} ark 提取自 checkpoint: {json.load(open(meta)).get('checkpoint')}")
    else:
        print(f"[apply] 提醒: {name} 目录没有 meta.json，无法确认 checkpoint 是否一致"
              f"（旧版提取的 ark？建议用 evaluate_robust.py 重提）")


def _meta_warn_if_mismatch(ark_dir, ckpt):
    meta = os.path.join(ark_dir, "meta.json")
    if os.path.isfile(meta):
        used = json.load(open(meta)).get("checkpoint")
        if used and os.path.abspath(used) != os.path.abspath(ckpt):
            print(f"[select-k] 警告: cohort ark 用的是 {used}，与本次 checkpoint {ckpt} 不一致！"
                  f"AS-Norm 需要 cohort 与验证同模型，请重提 cohort。")
    else:
        print("[select-k] 提醒: cohort 目录没有 meta.json，无法确认 checkpoint 一致性")


def apply_asnorm(cfg, base, args, device, conds=("clean", "noise_0db", "noise_m5db")):
    p_cfg = cfg["paths"]
    exp_dir = os.path.join(base, p_cfg["exp_dir"])
    sv_ecapa = p_cfg["sv_ecapa"] if os.path.isabs(p_cfg["sv_ecapa"]) else \
        os.path.join(base, p_cfg["sv_ecapa"])
    test_root = os.path.join(sv_ecapa, "data", "data_for_test")
    tag = args.tag
    emb_root = os.path.join(exp_dir, "eval", tag, "embeddings")
    cohort_dir = args.cohort_dir or os.path.join(emb_root, "cohort")
    _print_meta(cohort_dir, "cohort")
    _print_meta(os.path.join(emb_root, "clean"), "enrol/clean")

    k = args.k
    if k in (None, "auto", 0):
        sel = os.path.join(exp_dir, "data", "asnorm_selected_k.json")
        assert os.path.isfile(sel), "未指定 --k 且没有 select-k 结果，请先运行 --select-k"
        k = json.load(open(sel))["best_k"]
        print(f"[apply] 使用 select-k 结果 K={k}")
    k = int(k)

    cohort = load_ark_dir(cohort_dir)
    cohort_mat = np.stack(list(cohort.values()))
    enrol = load_ark_dir(os.path.join(emb_root, "clean"))
    baseline = cfg.get("baseline", {})

    print(f"\n===== [AS-Norm K={k}] {tag} 正式评测（cohort={cohort_mat.shape[0]}） =====")
    print(f"{'条件':<12}{'raw EER%':>9}{'AS EER%':>9}{'raw mDCF':>9}{'AS mDCF':>9}{'相对基线降幅':>12}")
    # 所有条件统一用 clean/3s 的 trials（与官方 evaluate_noise.sh 一致）
    trials_path = os.path.join(test_root, "clean", "3s", "trials_cross_device")
    for cond in conds:
        test = load_ark_dir(os.path.join(emb_root, cond))
        trials = load_trials(trials_path)
        pairs, labels = [], []
        for e, t, lab in trials:
            if e in enrol and t in test:
                pairs.append((e, t))
                labels.append(lab)
        E = np.stack([enrol[e] for e, _ in pairs])
        T = np.stack([test[t] for _, t in pairs])
        raw = np.sum(_norm(E) * _norm(T), axis=1)
        adapted = asnorm_scores(E, T, cohort_mat, k)
        raw_eer, raw_dc = eer_minDCF(raw, np.array(labels))
        as_eer, as_dc = eer_minDCF(adapted, np.array(labels))

        scores_dir = os.path.join(exp_dir, "eval", tag, "scores", cond)
        os.makedirs(scores_dir, exist_ok=True)
        with open(os.path.join(scores_dir, "trials_cross_device.asnorm.score"), "w") as f:
            for (e, t), s in zip(pairs, adapted):
                f.write(f"{e} {t} {s:.5f}\n")

        b = baseline.get(cond)
        rel = f"{100.0*(b-as_eer)/b:.1f}%" if b else "-"
        print(f"{cond:<12}{raw_eer:>9.4f}{as_eer:>9.4f}{raw_dc:>9.4f}{as_dc:>9.4f}{rel:>12}")
        print(f"             （raw EER 应与 evaluate_robust.py 官方打分一致：当前 {raw_eer:.4f}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_robust.toml")
    ap.add_argument("--make-cohort", action="store_true")
    ap.add_argument("--cohort-spks", type=int, default=500)
    ap.add_argument("--cohort-utts", type=int, default=2)
    ap.add_argument("--select-k", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--cohort-dir", default=None, help="cohort ark 目录（默认 exp/eval/<tag>/embeddings/cohort）")
    ap.add_argument("--tag", default="finetuned", help="apply 模式对应的评测 tag")
    ap.add_argument("--k", default="auto", help="top-K；auto = 用 select-k 的结果")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cfg = toml.load(args.config)
    base = os.path.dirname(os.path.abspath(args.config))
    p_cfg = cfg["paths"]
    setup_speakerlab(p_cfg["sv_ecapa"], base)   # 保证 speakerlab 可导入
    exp_dir = os.path.join(base, p_cfg["exp_dir"])
    data_dir = os.path.join(exp_dir, "data")

    if args.make_cohort:
        os.makedirs(os.path.join(data_dir, "asnorm"), exist_ok=True)
        make_cohort(data_dir, os.path.join(data_dir, "asnorm", "cohort_wavs.txt"),
                    args.cohort_spks, args.cohort_utts)
        print("[make-cohort] 下一步：evaluate_robust.py --extract-only 提 cohort embedding，"
              "再 --select-k / --apply")
        return
    if args.select_k:
        select_k(cfg, base, args, device)
        return
    if args.apply:
        apply_asnorm(cfg, base, args, device)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
