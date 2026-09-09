# -*- coding: utf-8 -*-
"""
E2 验证：clean 注册 vs {clean, 0dB, -5dB} 测试，报告 EER/minDCF。

用途：
  1. 训练脚本每个 epoch 调用 run_validation()（同一批落盘的验证 bench，epoch 间可比）。
  2. 独立复测任意 checkpoint：python validate_robust.py --config config_robust.toml \
       --checkpoint exp/robust/checkpoints/checkpoint_best.tar
     不带 --checkpoint 则评测 pretrained 基线（epoch0）。

验证集与训练/正式测试三方隔离（见 prepare_multicond_data.py 的断言）。
"""

import os
import glob
import argparse
import numpy as np
import torch
import toml

from common_utils import load_wav, load_trials, eer_minDCF, extract_embeddings, setup_speakerlab
from trainer_eres2net_robust import build_eres2net_large, load_pretrained_embedding

COND_DIRS = {"clean": "test_clean", "0db": "test_0db", "m5db": "test_m5db"}


def build_fbank():
    from speakerlab.process.processor import FBank
    return FBank(n_mels=80, sample_rate=16000, mean_nor=True)


def cosine_scores(emb_a, emb_b):
    a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-8)
    b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-8)
    return a @ b.T


def run_validation(embedding_model, bench_dir, val_trials_path, device,
                   batch_size=64, verbose=True):
    """
    返回 {cond: {"eer": EER%, "mindcf": minDCF}}，cond in {clean, 0db, m5db}。
    embedding_model 需已在 device 上；本函数内部切 eval / no_grad。
    """
    fbank = build_fbank()
    embedding_model.eval()

    enroll_files = sorted(glob.glob(os.path.join(bench_dir, "enroll", "*.wav")))
    assert enroll_files, f"验证 bench 为空: {bench_dir}/enroll"
    enroll_keys, enroll_emb = extract_embeddings(
        embedding_model, fbank, enroll_files, device, batch_size, desc="val-enroll")
    enroll_map = dict(zip(enroll_keys, enroll_emb))

    trials = load_trials(val_trials_path)
    conds = {c: sorted(glob.glob(os.path.join(bench_dir, d, "*.wav")))
             for c, d in COND_DIRS.items()}
    results = {}
    for cond, files in conds.items():
        keys, emb = extract_embeddings(
            embedding_model, fbank, files, device, batch_size, desc=f"val-{cond}")
        emb_map = dict(zip(keys, emb))
        scores, labels = [], []
        for e, t, lab in trials:
            if e in enroll_map and t in emb_map:
                s = float(np.dot(enroll_map[e], emb_map[t]) /
                          (np.linalg.norm(enroll_map[e]) * np.linalg.norm(emb_map[t]) + 1e-8))
                scores.append(s)
                labels.append(lab)
        eer, mindcf = eer_minDCF(np.array(scores), np.array(labels))
        results[cond] = {"eer": eer, "mindcf": mindcf}
        if verbose:
            print(f"   [val] {cond:>5}: EER = {eer:.4f}%  minDCF = {mindcf:.4f}  (trials={len(labels)})")
    return results


def load_model_for_validation(ckpt_path=None, eres2net_ckpt=None, device="cpu"):
    model = build_eres2net_large()
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("embedding_model", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, \
            f"checkpoint 加载不完整: missing={missing[:5]} unexpected={unexpected[:5]}"
        print(f"[validate] 已加载 checkpoint: {ckpt_path} (epoch={ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'})")
    else:
        assert eres2net_ckpt, "未提供 --checkpoint 时必须提供 pretrained eres2net_ckpt"
        load_pretrained_embedding(model, eres2net_ckpt)
        print(f"[validate] 评测 pretrained 基线: {eres2net_ckpt}")
    return model.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_robust.toml")
    ap.add_argument("--checkpoint", default=None, help="不填则评测 pretrained 基线")
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
    bench_dir = os.path.join(exp_dir, "data", "val_bench")
    val_trials = os.path.join(exp_dir, "data", "val_trials.txt")
    assert os.path.isdir(bench_dir), f"验证 bench 不存在，请先运行 prepare_multicond_data.py: {bench_dir}"

    eres2net_ckpt = p_cfg["eres2net_ckpt"]
    if not os.path.isabs(eres2net_ckpt):
        eres2net_ckpt = os.path.join(base, eres2net_ckpt)

    model = load_model_for_validation(args.checkpoint, eres2net_ckpt, device)
    print(f"[validate] bench={bench_dir}")
    results = run_validation(model, bench_dir, val_trials, device, args.batch_size)
    noise_sum = results["0db"]["eer"] + results["m5db"]["eer"]
    print(f"[validate] EER(0db)+EER(m5db) = {noise_sum:.4f}  "
          f"clean = {results['clean']['eer']:.4f}%")


if __name__ == "__main__":
    main()
