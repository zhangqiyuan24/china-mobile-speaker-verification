# -*- coding: utf-8 -*-
"""
8.21 AS-Norm（自适应分数归一化）：复用 8.20 asnorm.py 的核心函数，
路径换成 8.21 体系——cohort 从 8.20 train_utts.csv **只读**抽取、用 robust
checkpoint 提取；K 在 8.20 验证 bench（只读）上选；apply 直接吃
evaluate_combo.py 已落盘的官方 ark（enrol_clean + raw/<cond>），不重提。

三步（也可 --all 一次跑完）：
  python asnorm_proto.py --config config_proto.toml --gpu 3 --make-cohort
  python asnorm_proto.py --config config_proto.toml --gpu 3 --select-k
  python asnorm_proto.py --config config_proto.toml --gpu 3 --apply
  python asnorm_proto.py --config config_proto.toml --gpu 3 --all

说明：
  - cohort 必须与 enrol/test 同一 checkpoint 提取（meta.json 校验）。
  - apply 的 raw EER 应与 evaluate_combo.py 官方打分一致（一致性自检行）。
  - 分块打分：官方 1.38M trials 不会一次堆成 1.38M×512 的大矩阵。
  - 注意：AS-Norm 是打分层操作（不碰测试数据/trials），是否计入验收口径需与老师确认。
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import torch
import toml

_HERE = os.path.dirname(os.path.abspath(__file__))
_V820 = os.path.join(os.path.dirname(_HERE), "8.20")
if _V820 not in sys.path:
    sys.path.insert(0, _V820)

from common_utils import (load_trials, eer_minDCF, load_ark_dir, save_ark,
                          extract_embeddings, setup_speakerlab, resolve_existing_path)
from validate_robust import load_model_for_validation, build_fbank, COND_DIRS
from asnorm import make_cohort, cohort_stats, asnorm_scores, _norm, KS

CONDS_APPLY = ("clean", "noise_0db", "noise_m5db")


def _ckpt_path(cfg, base, args):
    ckpt = args.checkpoint or os.path.join(base, cfg["paths"]["exp_dir"],
                                           "checkpoints", "checkpoint_best.tar")
    assert os.path.isfile(ckpt), f"checkpoint 不存在: {ckpt}"
    return ckpt


def _load_model(cfg, base, ckpt, device):
    eres = cfg["paths"]["eres2net_ckpt"]
    if not os.path.isabs(eres):
        eres = os.path.join(base, eres)
    return load_model_for_validation(ckpt, eres, device)


def _data820(cfg, base):
    return resolve_existing_path(cfg["paths"]["data_dir_820"], base)


# ============================================================
# ① cohort：清单（读 8.20 train_utts.csv）+ robust ckpt 提取（居中裁 3s）
# ============================================================

def step_make_cohort(cfg, base, args, device):
    exp_dir = os.path.join(base, cfg["paths"]["exp_dir"])
    out_root = os.path.join(exp_dir, "eval", "asnorm")
    os.makedirs(out_root, exist_ok=True)
    wav_list = os.path.join(out_root, "cohort_wavs.txt")
    if os.path.isfile(wav_list):
        print(f"[cohort] {wav_list} 已存在，跳过生成")
    else:
        make_cohort(_data820(cfg, base), wav_list,
                    args.cohort_spks, args.cohort_utts)
    wavs = [l.strip() for l in open(wav_list) if l.strip()]

    ckpt = _ckpt_path(cfg, base, args)
    coh_dir = args.cohort_dir or os.path.join(out_root, "cohort")
    ark, meta_p = os.path.join(coh_dir, "emb.ark"), os.path.join(coh_dir, "meta.json")
    if os.path.isfile(ark):
        used = json.load(open(meta_p)).get("checkpoint") if os.path.isfile(meta_p) else None
        if used and os.path.abspath(used) == os.path.abspath(ckpt):
            print(f"[cohort] ark 已存在且 checkpoint 一致，跳过提取: {ark}")
            return coh_dir
        print(f"[cohort] ark 的 checkpoint（{used}）与本次不一致，重提")
    model = _load_model(cfg, base, ckpt, device)
    keys, embs = extract_embeddings(model, build_fbank(), wavs, device,
                                    args.batch_size, desc="cohort",
                                    crop_len=int(3.0 * 16000))
    save_ark(coh_dir, keys, embs)
    json.dump({"checkpoint": ckpt, "num_utts": len(keys), "crop_seconds": 3.0,
               "cohort_spks": args.cohort_spks, "cohort_utts": args.cohort_utts},
              open(meta_p, "w"), indent=1)
    print(f"[cohort] {len(keys)} 条（居中裁剪 3s，robust ckpt）-> {ark}")
    return coh_dir


# ============================================================
# ② select-k：8.20 验证 bench（只读）上选 K ∈ {100,200,300}
# ============================================================

def step_select_k(cfg, base, args, device):
    data820 = _data820(cfg, base)
    exp_dir = os.path.join(base, cfg["paths"]["exp_dir"])
    bench_dir = os.path.join(data820, "val_bench")
    trials = load_trials(os.path.join(data820, "val_trials.txt"))
    ckpt = _ckpt_path(cfg, base, args)
    coh_dir = args.cohort_dir or os.path.join(exp_dir, "eval", "asnorm", "cohort")
    cohort = np.stack(list(load_ark_dir(coh_dir).values()))
    _warn_cohort_ckpt(coh_dir, ckpt)

    model = _load_model(cfg, base, ckpt, device)
    fbank = build_fbank()
    enroll_files = sorted(glob.glob(os.path.join(bench_dir, "enroll", "*.wav")))
    ek, ee = extract_embeddings(model, fbank, enroll_files, device,
                                args.batch_size, desc="selK-enroll")
    emap = dict(zip(ek, ee))

    print(f"\n[select-k] cohort={cohort.shape}，bench trials={len(trials)}（seen-by-pretrained，仅用于选 K）")
    print(f"{'K':>5} | {'clean EER%':>10} {'0dB EER%':>9} {'-5dB EER%':>9} | {'noise_sum':>9}")
    best, table, cond_cache = None, {}, {}
    for k in KS:
        row, noise_sum = {}, 0.0
        for cond, subdir in COND_DIRS.items():
            if cond not in cond_cache:
                files = sorted(glob.glob(os.path.join(bench_dir, subdir, "*.wav")))
                tk, te = extract_embeddings(model, fbank, files, device,
                                            args.batch_size, desc=f"selK-{cond}")
                cond_cache[cond] = dict(zip(tk, te))
            tmap = cond_cache[cond]
            e_idx = [e for e, t, _ in trials if e in emap and t in tmap]
            t_idx = [t for e, t, _ in trials if e in emap and t in tmap]
            labs = [l for e, t, l in trials if e in emap and t in tmap]
            E = np.stack([emap[e] for e in e_idx])
            T = np.stack([tmap[t] for t in t_idx])
            eer, dcf = eer_minDCF(asnorm_scores(E, T, cohort, k), np.array(labs))
            row[cond] = {"eer": eer, "mindcf": dcf}
            if cond in ("0db", "m5db"):
                noise_sum += eer
        table[k] = {"metrics": row, "noise_sum": noise_sum}
        print(f"{k:>5} | {row['clean']['eer']:>10.4f} {row['0db']['eer']:>9.4f} "
              f"{row['m5db']['eer']:>9.4f} | {noise_sum:>9.4f}")
        if best is None or noise_sum < table[best]["noise_sum"]:
            best = k
    out = os.path.join(exp_dir, "eval", "asnorm", "selected_k.json")
    json.dump({"best_k": best, "table": table, "checkpoint": ckpt,
               "cohort_dir": coh_dir}, open(out, "w"), indent=2)
    print(f"\n[select-k] 最优 K = {best} -> {out}")
    return best


def _warn_cohort_ckpt(coh_dir, ckpt):
    meta_p = os.path.join(coh_dir, "meta.json")
    if os.path.isfile(meta_p):
        used = json.load(open(meta_p)).get("checkpoint")
        if used and os.path.abspath(used) != os.path.abspath(ckpt):
            print(f"[warn] cohort ark 用的是 {used}，与本次 {ckpt} 不一致！"
                  f"AS-Norm 要求同模型，请先重跑 --make-cohort")
    else:
        print("[warn] cohort 目录没有 meta.json，无法确认 checkpoint 一致性")


# ============================================================
# ③ apply：官方 ark（evaluate_combo.py 产物）上做 AS-Norm 打分
# ============================================================

def _pairwise_asnorm(E_all, T_all, e_idx, t_idx, cohort, k, chunk=200000):
    """
    E_all [Ne,D]=enrol 全库，T_all [Nt,D]=test 全库，e_idx/t_idx=[P] pair 行号。
    与 asnorm_scores 数学等价，但 cohort 统计只对唯一向量算一次、pair 打分分块，
    避免 1.38M×512 的中间矩阵。返回 (raw, adapted) 各 [P]。
    """
    En, Tn, Cn = _norm(E_all), _norm(T_all), _norm(cohort)
    mu_e, std_e = cohort_stats(En, Cn, k)
    mu_t, std_t = cohort_stats(Tn, Cn, k)
    raw_l, ad_l = [], []
    for i in range(0, len(e_idx), chunk):
        e, t = En[e_idx[i:i + chunk]], Tn[t_idx[i:i + chunk]]
        raw = np.einsum("ij,ij->i", e, t)
        ad = 0.5 * ((raw - mu_e[e_idx[i:i + chunk]]) / (std_e[e_idx[i:i + chunk]] + 1e-8)
                   + (raw - mu_t[t_idx[i:i + chunk]]) / (std_t[t_idx[i:i + chunk]] + 1e-8))
        raw_l.append(raw)
        ad_l.append(ad)
    return np.concatenate(raw_l), np.concatenate(ad_l)


def step_apply(cfg, base, args, device):
    p_cfg = cfg["paths"]
    exp_dir = os.path.join(base, p_cfg["exp_dir"])
    combo_root = os.path.join(exp_dir, "eval", "combo")
    sv_ecapa = setup_speakerlab(p_cfg["sv_ecapa"], base)
    trials_path = os.path.join(sv_ecapa, "data", "data_for_test",
                               "clean", "3s", "trials_cross_device")
    trials = load_trials(trials_path)
    ckpt = _ckpt_path(cfg, base, args)
    coh_dir = args.cohort_dir or os.path.join(exp_dir, "eval", "asnorm", "cohort")
    _warn_cohort_ckpt(coh_dir, ckpt)
    cohort = np.stack(list(load_ark_dir(coh_dir).values()))

    k = args.k
    if k in (None, "auto", 0):
        sel = os.path.join(exp_dir, "eval", "asnorm", "selected_k.json")
        assert os.path.isfile(sel), "未显式给 --k 且没有 select-k 结果，先跑 --select-k"
        k = json.load(open(sel))["best_k"]
        print(f"[apply] 使用 select-k 结果 K={k}")
    k = int(k)

    enrol = load_ark_dir(os.path.join(combo_root, "common", "enrol_clean"))
    ekeys = sorted(enrol)
    E_all = np.stack([enrol[e] for e in ekeys])
    erow = {e: i for i, e in enumerate(ekeys)}
    bl = cfg.get("baseline", {})

    print(f"\n===== [AS-Norm K={k}] 正式 open-set（cohort={cohort.shape[0]}，trials={len(trials)}） =====")
    print(f"{'条件':<12}{'raw EER%':>9}{'AS EER%':>9}{'raw mDCF':>9}{'AS mDCF':>9}{'相对基线':>10}")
    summary = {"k": k, "checkpoint": ckpt, "cohort_dir": coh_dir, "conds": {}}
    for cond in CONDS_APPLY:
        test = load_ark_dir(os.path.join(combo_root, "raw", cond))
        tkeys = sorted(test)
        T_all = np.stack([test[t] for t in tkeys])
        trow = {t: i for i, t in enumerate(tkeys)}
        e_idx, t_idx, labs, kept, miss = [], [], [], [], 0
        for e, t, lab in trials:
            if e in erow and t in trow:
                e_idx.append(erow[e])
                t_idx.append(trow[t])
                labs.append(lab)
                kept.append((e, t))
            else:
                miss += 1
        assert miss == 0, f"{cond}: {miss} 条 trial 的 key 在 ark 里找不到（提取不完整？）"
        raw, ad = _pairwise_asnorm(E_all, T_all,
                                   np.asarray(e_idx, dtype=np.int64),
                                   np.asarray(t_idx, dtype=np.int64), cohort, k)
        raw_eer, raw_dcf = eer_minDCF(raw, np.array(labs))
        as_eer, as_dcf = eer_minDCF(ad, np.array(labs))
        b = bl.get(cond)
        rel = f"{100.0 * (b - as_eer) / b:.1f}%" if b else "-"
        print(f"{cond:<12}{raw_eer:>9.4f}{as_eer:>9.4f}{raw_dcf:>9.4f}{as_dcf:>9.4f}{rel:>10}")
        print(f"             （raw EER 自检，应≈evaluate_combo 官方打分：{raw_eer:.4f}）")
        sdir = os.path.join(combo_root, "scores", f"raw_{cond}")
        os.makedirs(sdir, exist_ok=True)
        with open(os.path.join(sdir, "trials_cross_device.asnorm.score"), "w") as f:
            for (e, t), s in zip(kept, ad):
                f.write(f"{e} {t} {s:.5f}\n")
        summary["conds"][cond] = {"raw_eer": raw_eer, "as_eer": as_eer,
                                  "raw_mindcf": raw_dcf, "as_mindcf": as_dcf,
                                  "rel_to_baseline": rel}
    out = os.path.join(exp_dir, "eval", "asnorm", "apply_summary.json")
    json.dump(summary, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\n[apply] 明细 -> {out}；AS-Norm 分数文件在各 scores/raw_<cond>/ 下")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_proto.toml")
    ap.add_argument("--make-cohort", action="store_true")
    ap.add_argument("--select-k", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--all", action="store_true", help="依次跑三步")
    ap.add_argument("--cohort-spks", type=int, default=500)
    ap.add_argument("--cohort-utts", type=int, default=2)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--cohort-dir", default=None)
    ap.add_argument("--k", default="auto")
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = toml.load(args.config)
    base = os.path.dirname(os.path.abspath(args.config))
    setup_speakerlab(cfg["paths"]["sv_ecapa"], base)   # 任何一步都可能建模型，统一注入 speakerlab

    if args.all or args.make_cohort:
        step_make_cohort(cfg, base, args, device)
    if args.all or args.select_k:
        step_select_k(cfg, base, args, device)
    if args.all or args.apply:
        step_apply(cfg, base, args, device)
    if not (args.all or args.make_cohort or args.select_k or args.apply):
        ap.print_help()


if __name__ == "__main__":
    main()
