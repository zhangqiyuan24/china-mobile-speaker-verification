#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_ecapa_protos.py — ECAPA 空间的说话人原型表（8.26 ECAPA 线配套）

8.21 的 prototypes.npz 是 ERes2Net 空间, ECAPA 线需要自己的 192 维原型
(分类器 proto 初始化用)。用 pretrained ECAPA 提 train_utts.csv 每说话人
平均嵌入(中心裁 3s), L2 归一化, 存 {embs, spk_ids} —— 与 build_spk_embs
同构, 仅模型/维度不同。

用法: python build_ecapa_protos.py --config config_ecapa_robust.toml --gpu 1
输出: exp/ecapa_protos/prototypes.npz
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torchaudio
import toml

_HERE = os.path.dirname(os.path.abspath(__file__))
_V820 = os.path.join(os.path.dirname(_HERE), "8.20")
_V821 = os.path.join(os.path.dirname(_HERE), "8.21")
for _p in (_V820, _V821, _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from common_utils import setup_speakerlab
from validate_robust import build_fbank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_ecapa_robust.toml")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--max-utt", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=48)
    args = ap.parse_args()

    cfg = toml.load(os.path.join(_HERE, args.config))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda:0"
    sv_ecapa = cfg["paths"]["sv_ecapa"]
    setup_speakerlab(sv_ecapa, _HERE)

    from speakerlab.models.ecapa_tdnn.ECAPA_TDNN import ECAPA_TDNN
    model = ECAPA_TDNN(input_size=80, lin_neurons=192, channels=[1024, 1024, 1024, 1024, 3072])
    sd = torch.load(cfg["paths"]["ecapa_ckpt"], map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)          # 231 张量, 应 strict 全中
    model.to(device).eval()
    print("[ecapa] pretrained 装载完成(20.8M, emb 192)", flush=True)

    spk2paths = {}
    with open(os.path.join(cfg["paths"]["data_dir_820"], "train_utts.csv"),
              newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
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

    fbank = build_fbank()
    embs = []
    with torch.no_grad():
        for i in range(0, len(utts), args.batch_size):
            waves = []
            for p in utts[i:i + args.batch_size]:
                w, _ = torchaudio.load(p)
                w = w[0]
                if w.numel() < 48000:
                    w = torch.nn.functional.pad(w, (0, 48000 - w.numel()))
                else:
                    st = (w.numel() - 48000) // 2
                    w = w[st:st + 48000]
                waves.append(w)
            fb = torch.stack([fbank(w) for w in waves]).to(device)
            embs.append(model(fb).cpu().numpy())
            if (i // args.batch_size) % 30 == 0:
                print(f"  {i}/{len(utts)}", flush=True)
    embs = np.concatenate(embs, axis=0)

    out = np.zeros((len(spk_ids), embs.shape[1]), dtype=np.float32)
    idx = {s: i for i, s in enumerate(spk_ids)}
    cnt = np.zeros(len(spk_ids), dtype=np.int64)
    for e, s in zip(embs, owner):
        out[idx[s]] += e
        cnt[idx[s]] += 1
    out /= np.maximum(cnt[:, None], 1)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-8

    od = os.path.join(_HERE, "exp", "ecapa_protos")
    os.makedirs(od, exist_ok=True)
    np.savez(os.path.join(od, "prototypes.npz"), embs=out, spk_ids=np.array(spk_ids))
    print(f"[done] {out.shape} -> {od}/prototypes.npz", flush=True)


if __name__ == "__main__":
    main()
