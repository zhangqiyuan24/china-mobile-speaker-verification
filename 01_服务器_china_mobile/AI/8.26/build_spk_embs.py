#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_spk_embs.py — 每说话人注册嵌入表（8.26 TSE 线配套）

用冻结验证器(soup5)从 train_utts.csv 的 clean wav 提每说话人平均嵌入,
L2 归一化后存 npz。训练时作 FiLM 条件; 评测侧(enhance_tse.py)用同一
验证器从官方 enrol clean 提测试说话人嵌入 —— 训练/推理条件同分布。

用法: python build_spk_embs.py --config train_tse.toml --gpu 2
输出: <exp_dir>/spk_embs.npz {embs:(N,512) f32, spk_ids:[...]}
"""
import argparse
import os
import sys

import numpy as np
import torch
import torchaudio
import toml

_HERE = os.path.dirname(os.path.abspath(__file__))
_V818 = os.path.join(os.path.dirname(_HERE), "8.18")
for _p in (_V818, _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from train_spk_preserve import _setup
from trainer_spk_preserve import load_eres2net
from frontend_spk import DiffFBank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train_tse.toml")
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--max-utt", type=int, default=3, help="每说话人最多用几条")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    cfg = toml.load(os.path.join(_HERE, args.config))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda:0"
    base = _HERE
    sv_ecapa = os.path.abspath(os.path.join(base, cfg["paths"]["sv_ecapa"]))
    _setup(os.path.abspath(cfg["paths"]["fullsubnet_root"]), sv_ecapa)

    # ---- 说话人 -> utt 清单(确定性: 每人前 max_utt 条) ----
    import csv as _csv
    spk2paths = {}
    with open(os.path.join(cfg["paths"]["data_dir_820"], "train_utts.csv"),
              newline="", encoding="utf-8") as f:
        for row in _csv.reader(f):
            if not row or row[0] == "utt_id":
                continue
            spk2paths.setdefault(row[1], []).append(row[2])
    spk_ids = sorted(spk2paths)
    utts, owner = [], []
    for s in spk_ids:
        for p in spk2paths[s][:args.max_utt]:
            utts.append(p)
            owner.append(s)
    print(f"[data] {len(spk_ids)} spks / {len(utts)} utts", flush=True)

    verifier = load_eres2net(sv_ecapa, cfg["paths"]["verifier_ckpt"], device)
    verifier.eval()
    fbank = DiffFBank(n_mels=80, sample_rate=16000, n_fft=512,
                      win_length=400, hop_length=160, mean_nor=True).to(device)

    embs = []
    with torch.no_grad():
        for i in range(0, len(utts), args.batch_size):
            chunk = utts[i:i + args.batch_size]
            waves = []
            for p in chunk:
                w, _ = torchaudio.load(p)
                w = w[0]
                if w.numel() < 48000:
                    w = torch.nn.functional.pad(w, (0, 48000 - w.numel()))
                else:  # 中心裁 3s
                    st = (w.numel() - 48000) // 2
                    w = w[st:st + 48000]
                waves.append(w)
            fb = fbank(torch.stack(waves).to(device))
            e = verifier(fb)
            embs.append(e.cpu().numpy())
            if (i // args.batch_size) % 20 == 0:
                print(f"  {i}/{len(utts)}", flush=True)
    embs = np.concatenate(embs, axis=0)

    # ---- 每说话人平均 + L2 归一化 ----
    dim = embs.shape[1]
    out = np.zeros((len(spk_ids), dim), dtype=np.float32)
    idx = {s: i for i, s in enumerate(spk_ids)}
    cnt = np.zeros(len(spk_ids), dtype=np.int64)
    for e, s in zip(embs, owner):
        out[idx[s]] += e
        cnt[idx[s]] += 1
    out /= np.maximum(cnt[:, None], 1)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-8

    exp_dir = cfg["paths"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    np.savez(os.path.join(exp_dir, "spk_embs.npz"), embs=out, spk_ids=np.array(spk_ids))
    print(f"[done] {out.shape} -> {exp_dir}/spk_embs.npz", flush=True)


if __name__ == "__main__":
    main()
