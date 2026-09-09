#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""score_proxy.py — proxy 验证集全成员打分（8.27 Phase 0）

成员 = cross_fusion_v3 全集（wavlm 除外——v3 grid 给了 0 权重，且异构抽取另走一条
管线；在 README 记录理由）+ 新增 tse_soup5（soup5 后端 × TSE 增强输入）。

每成员：enrol clean + 对应变体带噪 → 各后端提嵌入 → trial 余弦分 → .score（4 列
官方同构）。打印各成员单路 proxy EER（与官方单路排序做 sanity 对照）。

用法: python score_proxy.py --gpu 1
"""
import argparse
import os
import sys

import numpy as np
import torch
import toml

_HERE = os.path.dirname(os.path.abspath(__file__))
_AI = os.path.dirname(_HERE)
for _p in (os.path.join(_AI, "8.18"), os.path.join(_AI, "8.20"),
           os.path.join(_AI, "8.21"), os.path.join(_AI, "8.26"), _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import torchaudio

from trainer_spk_preserve import load_eres2net
from frontend_spk import DiffFBank
from common_utils import setup_speakerlab

PROXY = os.path.join(_HERE, "exp", "proxy")
OUT = os.path.join(PROXY, "scores")
CFG_TSE = os.path.join(_AI, "8.26", "train_tse.toml")

CKPT = {
    "821": "/data/home/ligangrui/china_mobile/AI/8.21/exp/proto_robust/checkpoints/checkpoint_best.tar",
    "30k": "/data/home/ligangrui/china_mobile/AI/8.22/exp/matched_robust/checkpoints/checkpoint_best.tar",
    "90k": "/data/home/ligangrui/china_mobile/AI/8.22/exp/matched_robust_90k/checkpoints/checkpoint_best.tar",
    "soup": "/data/home/ligangrui/china_mobile/AI/8.24/exp/soup_821_90k/checkpoints/checkpoint_soup_w50.tar",
    "p90k": "/data/home/ligangrui/china_mobile/AI/8.24/exp/proto_robust_90k/checkpoints/checkpoint_best.tar",
    "soup5": "/data/home/ligangrui/china_mobile/AI/8.24/exp/soup5/checkpoints/checkpoint_soup5.tar",
}
# (成员名, 后端, 输入变体)  raw=原始带噪 / enh818 / enhtse
MEMBERS = [
    ("821_raw", "821", "raw"), ("821_fe", "821", "enh818"),
    ("30k_raw", "30k", "raw"), ("30k_fe", "30k", "enh818"),
    ("90k_raw", "90k", "raw"), ("90k_fe", "90k", "enh818"),
    ("soup_raw", "soup", "raw"), ("soup_fe", "soup", "enh818"),
    ("p90k_raw", "p90k", "raw"), ("p90k_fe", "p90k", "enh818"),
    ("tse_soup5", "soup5", "enhtse"),
]


def eer_of(scores, labels):
    order = np.argsort(-scores)
    labs = labels[order]
    P = int(labels.sum())
    N = int(len(labels) - labels.sum())
    tp = np.cumsum(labs)
    fp = np.cumsum(1 - labs)
    fpr = fp / float(N)
    miss = (P - tp) / float(P)
    idx = int(np.argmin(np.abs(fpr - miss)))
    return (fpr[idx] + miss[idx]) / 2.0 * 100.0


@torch.no_grad()
def extract(model, fbank, files, device, batch=16):
    embs = {}
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        waves = []
        for p in chunk:
            w, _ = torchaudio.load(p)
            waves.append(w[0])
        e = model(fbank(torch.stack(waves).to(device))).cpu().numpy()
        for p, row in zip(chunk, e):
            embs[os.path.splitext(os.path.basename(p))[0]] = row
    return embs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    args = ap.parse_args()

    cfg = toml.load(CFG_TSE)
    sv_ecapa = cfg["paths"]["sv_ecapa"]
    setup_speakerlab(sv_ecapa, _HERE)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)

    enrol_files = [os.path.join(PROXY, "enrol", f)
                   for f in sorted(os.listdir(os.path.join(PROXY, "enrol")))]
    trials = {}
    for cond in ("noise_0db", "noise_m5db"):
        rows = []
        with open(os.path.join(PROXY, f"trials_{cond}.txt")) as f:
            for line in f:
                p = line.split()
                if len(p) == 3:
                    rows.append((p[0], p[1], p[2]))
        trials[cond] = rows

    fbank = DiffFBank(n_mels=80, sample_rate=16000, n_fft=512,
                      win_length=400, hop_length=160, mean_nor=True).to(device)

    models = {}
    for key, path in CKPT.items():
        models[key] = load_eres2net(sv_ecapa, path, device)
        models[key].eval()
        print(f"[backend] {key} <- {os.path.basename(path)}", flush=True)

    for name, backend, variant in MEMBERS:
        e_enrol = extract(models[backend], fbank, enrol_files, device)
        for cond in ("noise_0db", "noise_m5db"):
            d = os.path.join(PROXY, cond if variant == "raw"
                             else f"{variant}_{cond}")
            files = [os.path.join(d, f) for f in sorted(os.listdir(d))]
            e_test = extract(models[backend], fbank, files, device)
            out_path = os.path.join(OUT, f"{name}_{cond}.score")
            labs, scs = [], []
            with open(out_path, "w") as f:
                for e_stem, t_stem, lab in trials[cond]:
                    a = e_enrol[e_stem] if e_stem in e_enrol else e_enrol[
                        e_stem.replace("__", "__")]  # enrol 唯一前缀匹配兜底
                    b = e_test[t_stem]
                    cos = float(np.dot(a, b) /
                                (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
                    f.write(f"{e_stem} {t_stem} {lab} {cos:.6f}\n")
                    labs.append(1 if lab == "target" else 0)
                    scs.append(cos)
            eer = eer_of(np.asarray(scs), np.asarray(labs))
            print(f"[score] {name:12s} {cond:11s} EER={eer:.3f}% -> {out_path}",
                  flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
