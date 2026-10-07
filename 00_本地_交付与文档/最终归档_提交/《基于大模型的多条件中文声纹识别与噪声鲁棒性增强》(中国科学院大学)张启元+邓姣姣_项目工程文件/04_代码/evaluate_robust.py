# -*- coding: utf-8 -*-
"""
E2 正式评测：同一个 checkpoint 重提注册侧 + 测试侧 embedding，三条件完整打分。

协议（与官方 evaluate_clean.sh / results_summary.csv 完全一致）：
  - 注册侧 = data_for_test/clean/3s 的 A 角色（clean），用【微调后的模型】重提
  - 测试侧 = data_for_test/{clean,noise_0db,noise_m5db}/3s 的 A+D 全部 wav
  - trials = 各条件目录下的 trials_cross_device（原始 3s 跨设备 trial）
  - 打分 = 原版 speakerlab/bin/compute_score_metrics.py（cosine，EER + minDCF）
  - 汇总时对照 results_summary.csv 的 eres2net 3s 基线，输出相对降幅

修复的旧问题：8.18 用 modelscope infer_sv_batch 提 embedding，它强制加载预训练
权重，换 checkpoint 不生效（注册/测试两侧模型不一致的隐患）。本脚本自己加载
指定 checkpoint 提两侧 embedding，保证“同一模型”。

附加功能：
  --extract-only --wav-list xx.txt   只提 embedding 出 ark（给 AS-Norm cohort 用）
  --also-pretrained                  同时跑一遍 pretrained 基线（应复现官方数字，
                                     用于验证本提取管线无误）
"""

import os
import sys
import glob
import json
import argparse
import subprocess
import numpy as np
import torch
import toml

from common_utils import extract_embeddings, save_ark, resolve_existing_path, setup_speakerlab
from validate_robust import build_fbank, load_model_for_validation


def get_wav_list(cond_dir, sv_ecapa):
    """
    优先用官方生成的 wav_list.txt（A+D 两角色；行是 sv-ecapa 相对路径，需解析），
    否则现场枚举 A/D 子目录。
    """
    wl = os.path.join(cond_dir, "wav_list.txt")
    if os.path.isfile(wl):
        with open(wl) as f:
            return [resolve_existing_path(l.strip(), sv_ecapa)
                    for l in f if l.strip().endswith(".wav")]
    wavs = sorted(glob.glob(os.path.join(cond_dir, "A", "*.wav"))) + \
           sorted(glob.glob(os.path.join(cond_dir, "D", "*.wav")))
    assert wavs, f"{cond_dir} 下没有 wav（A/D 子目录）"
    return wavs


def extract_and_save(model, wav_paths, out_dir, device, batch_size, desc,
                     ckpt_path=None, crop_len=None):
    fbank = build_fbank()
    keys, embs = extract_embeddings(model, fbank, wav_paths, device, batch_size,
                                    desc=desc, crop_len=crop_len)
    save_ark(out_dir, keys, embs)
    if ckpt_path:
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump({"checkpoint": ckpt_path, "num_utts": len(keys),
                       "crop_len": crop_len}, f, indent=1)
    print(f"[extract] {len(keys)} -> {out_dir}/emb.ark")
    return out_dir


def run_official_scorer(speakerlab_root, enrol_dir, test_dir, scores_dir, trials):
    """调用原版 compute_score_metrics.py（等价于官方 cd sv-ecapa && source path.sh）。"""
    os.makedirs(scores_dir, exist_ok=True)
    cmd = [sys.executable,
           os.path.join(speakerlab_root, "speakerlab", "bin", "compute_score_metrics.py"),
           "--enrol_data", enrol_dir, "--test_data", test_dir,
           "--scores_dir", scores_dir, "--trials", trials]
    env = dict(os.environ)
    env["PYTHONPATH"] = speakerlab_root + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "1")
    subprocess.run(cmd, check=True, cwd=speakerlab_root, env=env)


def read_metrics(scores_dir):
    eer = mindcf = None
    with open(os.path.join(scores_dir, "result.metrics")) as f:
        for line in f:
            if line.startswith("EER"):
                eer = float(line.split("=")[1])
            elif line.startswith("minDCF"):
                mindcf = float(line.split("=")[1].split("(")[0])
    return eer, mindcf


def evaluate_model(tag, model, cfg, base, device, args, conditions, ckpt_path=None):
    p_cfg = cfg["paths"]
    exp_dir = os.path.join(base, p_cfg["exp_dir"])
    sv_ecapa = p_cfg["sv_ecapa"] if os.path.isabs(p_cfg["sv_ecapa"]) else \
        os.path.join(base, p_cfg["sv_ecapa"])
    # sv-ecapa -> 3dspeaker -> egs -> 3D-Speaker 仓库根（与官方 ../../../speakerlab/bin 等价）
    speakerlab_root = os.path.dirname(os.path.dirname(os.path.dirname(sv_ecapa)))
    test_root = os.path.join(sv_ecapa, "data", "data_for_test")
    out_root = os.path.join(exp_dir, "eval", tag)
    os.makedirs(out_root, exist_ok=True)

    emb_root = os.path.join(out_root, "embeddings")
    cond_dirs = {}
    for cond in conditions:
        cdir = os.path.join(test_root, cond, "3s")
        assert os.path.isdir(cdir), f"测试目录不存在: {cdir}"
        cond_dirs[cond] = cdir

    # 1) 提 embedding（clean 供注册侧 + 自身测试侧；噪声条件仅测试侧）
    for cond, cdir in cond_dirs.items():
        out_dir = os.path.join(emb_root, cond)
        if os.path.isfile(os.path.join(out_dir, "emb.ark")):
            meta_p = os.path.join(out_dir, "meta.json")
            used = json.load(open(meta_p)).get("checkpoint") if os.path.isfile(meta_p) else None
            if ckpt_path and used and os.path.abspath(used) != os.path.abspath(ckpt_path):
                raise RuntimeError(
                    f"{out_dir} 的 ark 是用 {used} 提的，与本次 checkpoint {ckpt_path} 不一致。\n"
                    f"换模型评测必须重提：rm -rf {out_dir} 后重跑。")
            print(f"[{tag}] {cond} ark 已存在，跳过提取（如需重提请删除 {out_dir}）")
            continue
        wavs = get_wav_list(cdir, sv_ecapa)
        print(f"[{tag}] 提取 {cond}: {len(wavs)} 条 wav")
        extract_and_save(model, wavs, out_dir, device, args.batch_size,
                         desc=f"{tag}-{cond}", ckpt_path=ckpt_path)

    # 2) 官方打分：enrol = clean（同模型重提），test = 各条件
    #    trials 与官方 evaluate_noise.sh 一致：所有条件都用 clean/3s 的
    #    trials_cross_device（噪声目录本身不生成 trials 文件）
    trials_path = os.path.join(test_root, "clean", "3s", "trials_cross_device")
    assert os.path.isfile(trials_path), f"缺 trials: {trials_path}"
    enrol_dir = os.path.join(emb_root, "clean")
    results = {}
    for cond in conditions:
        scores_dir = os.path.join(out_root, "scores", cond)
        run_official_scorer(speakerlab_root, enrol_dir,
                            os.path.join(emb_root, cond), scores_dir, trials_path)
        eer, mindcf = read_metrics(scores_dir)
        results[cond] = {"eer": eer, "mindcf": mindcf}

    # 3) 汇总（对照官方基线）
    baseline = cfg.get("baseline", {})
    print(f"\n===== [{tag}] 正式评测结果（trials_cross_device, 3s） =====")
    print(f"{'条件':<12}{'EER%':>9}{'minDCF':>9}{'基线EER%':>10}{'相对降幅':>10}")
    for cond, r in results.items():
        b = baseline.get(cond)
        if b:
            rel = 100.0 * (b - r["eer"]) / b
            print(f"{cond:<12}{r['eer']:>9.4f}{r['mindcf']:>9.4f}{b:>10.4f}{rel:>9.1f}%")
        else:
            print(f"{cond:<12}{r['eer']:>9.4f}{r['mindcf']:>9.4f}")
    ok = all(
        results[c]["eer"] <= baseline[c] * 0.7
        for c in ("noise_0db", "noise_m5db") if c in results and c in baseline
    ) if baseline else False
    print(f"验收判据（0dB 与 -5dB 相对降幅均 ≥30%）: {'达标 ✓' if ok else '未达标（见上行降幅）'}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_robust.toml")
    ap.add_argument("--checkpoint", default=None,
                    help="默认取 exp 下 checkpoints/checkpoint_best.tar")
    ap.add_argument("--conditions", nargs="+",
                    default=["clean", "noise_0db", "noise_m5db"])
    ap.add_argument("--also-pretrained", action="store_true",
                    help="附带评测 pretrained 基线（验证管线可复现官方数字）")
    ap.add_argument("--extract-only", action="store_true",
                    help="只提 embedding 不打分（配合 --wav-list / --out-dir，供 AS-Norm cohort）")
    ap.add_argument("--wav-list", default=None, help="extract-only 模式的 wav 清单")
    ap.add_argument("--out-dir", default=None, help="extract-only 模式的 ark 输出目录")
    ap.add_argument("--crop-seconds", type=float, default=0.0,
                    help="extract-only 时对每条 wav 居中裁剪到该秒数（cohort 全长语料必填 3.0，"
                         "否则变长帧数无法按 batch 堆叠）")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cfg = toml.load(args.config)
    base = os.path.dirname(os.path.abspath(args.config))
    p_cfg = cfg["paths"]
    setup_speakerlab(p_cfg["sv_ecapa"], base)   # 保证 speakerlab 可导入
    exp_dir = os.path.join(base, p_cfg["exp_dir"])
    eres2net_ckpt = p_cfg["eres2net_ckpt"]
    if not os.path.isabs(eres2net_ckpt):
        eres2net_ckpt = os.path.join(base, eres2net_ckpt)

    ckpt = args.checkpoint
    if ckpt and not os.path.isabs(ckpt):
        ckpt = os.path.join(base, ckpt)
    if not ckpt:
        ckpt = os.path.join(exp_dir, "checkpoints", "checkpoint_best.tar")
    assert os.path.isfile(ckpt), f"checkpoint 不存在: {ckpt}"

    model = load_model_for_validation(ckpt, eres2net_ckpt, device)

    if args.extract_only:
        assert args.wav_list and args.out_dir, "--extract-only 需要 --wav-list 与 --out-dir"
        wavs = [l.strip() for l in open(args.wav_list) if l.strip()]
        crop_len = int(args.crop_seconds * 16000) if args.crop_seconds > 0 else None
        print(f"[extract-only] {len(wavs)} 条 -> {args.out_dir}"
              + (f"（居中裁剪 {args.crop_seconds}s）" if crop_len else ""))
        extract_and_save(model, wavs, args.out_dir, device, args.batch_size,
                         desc="extract-only", ckpt_path=ckpt, crop_len=crop_len)
        return

    conds = args.conditions
    if args.also_pretrained:
        pre_model = load_model_for_validation(None, eres2net_ckpt, device)
        evaluate_model("pretrained", pre_model, cfg, base, device, args, conds,
                       ckpt_path=eres2net_ckpt)
    evaluate_model("finetuned", model, cfg, base, device, args, conds, ckpt_path=ckpt)


if __name__ == "__main__":
    main()
