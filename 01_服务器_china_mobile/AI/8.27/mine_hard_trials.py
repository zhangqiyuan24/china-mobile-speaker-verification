#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""mine_hard_trials.py — proxy v2: 困难负样本挖掘（8.27）

动机（用户 8.26 指示 + proxy_official_consistency 证据）：proxy 对官方系统性
乐观 ~5pp，0dB 档 EER 钉死 2.000 无选点判别力；根因之一是 nontarget 为跨说话
人随机对，远易于官方百万级负对分布。

本脚本不改任何音频（零重新混合），只重挖 trial：
  对每条带噪 test，用 soup5（选点后端）嵌入取跨说话人余弦最相似的
  top-ceil(5.5)=6 个 enrol 做候选，再全局按相似度降序取 1:5.5 配额为
  hard nontarget。同性别/相近音色自动浮现。target 对保持全集不变。

输出 trials_hard_{0db,m5db}.txt + soup5 嵌入缓存（供后续复用）。
只读音频、只新增文件。用法: python mine_hard_trials.py --gpu 1
"""
import argparse
import math
import os
import random
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_AI = os.path.dirname(_HERE)
for _p in (os.path.join(_AI, "8.18"), os.path.join(_AI, "8.20"),
           os.path.join(_AI, "8.21"), os.path.join(_AI, "8.26"), _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from trainer_spk_preserve import load_eres2net
from frontend_spk import DiffFBank
from common_utils import setup_speakerlab

PROXY = os.path.join(_HERE, "exp", "proxy")
SV_ECAPA = ("/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker"
            "/egs/3dspeaker/sv-ecapa")
SOUP5 = ("/data/home/ligangrui/china_mobile/AI/8.24/exp/soup5/checkpoints"
         "/checkpoint_soup5.tar")


def eer_of(scores, labels):
    order = np.argsort(-scores)
    labs = labels[order]
    P = int(labels.sum())
    N = int(len(labels) - P)
    tp = np.cumsum(labs)
    fp = np.cumsum(1 - labs)
    fpr = fp / float(N)
    miss = (P - tp) / float(P)
    idx = int(np.argmin(np.abs(fpr - miss)))
    return (fpr[idx] + miss[idx]) / 2.0 * 100.0


@torch.no_grad()
def extract(model, fbank, files, device, batch=8):
    embs = {}
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        import torchaudio
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
    ap.add_argument("--ratio", type=float, default=5.5)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda:0"
    setup_speakerlab(SV_ECAPA, _HERE)

    fbank = DiffFBank(n_mels=80, sample_rate=16000, n_fft=512,
                      win_length=400, hop_length=160, mean_nor=True).to(device)
    model = load_eres2net(SV_ECAPA, SOUP5, device)
    model.eval()

    # ---- enrol 嵌入 + 从旧 trials 恢复每人实际使用的 enrol stem ----
    enrol_files = [os.path.join(PROXY, "enrol", f)
                   for f in sorted(os.listdir(os.path.join(PROXY, "enrol")))]
    e_enrol = extract(model, fbank, enrol_files, device)
    enrol0 = {}
    for cond in ("noise_0db", "noise_m5db"):
        with open(os.path.join(PROXY, f"trials_{cond}.txt")) as f:
            for line in f:
                p = line.split()
                if len(p) == 3 and p[2] == "target":
                    enrol0.setdefault(p[0].split("__")[0], p[0])
    print(f"[enrol] {len(e_enrol)} 条, 每人 trial 注册 {len(enrol0)}", flush=True)

    np.savez(os.path.join(PROXY, "emb_soup5_enrol.npz"),
             stems=np.array(sorted(e_enrol)),
             embs=np.stack([e_enrl for e_enrl in
                            (e_enrol[s] for s in sorted(e_enrol))]))

    for cond in ("noise_0db", "noise_m5db"):
        test_files = [os.path.join(PROXY, cond, f)
                      for f in sorted(os.listdir(os.path.join(PROXY, cond)))]
        e_test = extract(model, fbank, test_files, device)
        tst = sorted(e_test)
        t_spk = [t.split("__")[0] for t in tst]
        spks = sorted(enrol0)
        spk_idx = {s: i for i, s in enumerate(spks)}

        def l2(x):
            return x / (np.linalg.norm(x) + 1e-8)

        E = np.stack([l2(e_enrol[enrol0[s]]) for s in spks])
        T = np.stack([l2(e_test[t]) for t in tst])
        S = E @ T.T                                   # (n_spk, n_test)

        # target 全集（同旧 trials）+ soup5 sanity EER
        t_sc = [S[spk_idx[t_spk[j]], j] for j in range(len(tst))]
        # ---- hard nontarget: 每 test 跨说话人 top-k 候选 → 全局配额 ----
        k = int(math.ceil(args.ratio))
        cand = []
        for j in range(len(tst)):
            col = S[:, j].copy()
            col[spk_idx[t_spk[j]]] = -2.0             # 排除自身说话人
            for i in np.argsort(-col)[:k]:
                cand.append((enrol0[spks[i]], tst[j], float(col[i])))
        cand.sort(key=lambda x: -x[2])
        targets = [(enrol0[t_spk[j]], tst[j]) for j in range(len(tst))]
        n_nt = int(len(targets) * args.ratio)
        chosen = cand[:n_nt]

        # sanity: soup5 在 hard trials 上的 EER（期望显著高于旧 2.0/4.26）
        labs = [1] * len(targets) + [0] * len(chosen)
        scs = t_sc + [c[2] for c in chosen]
        eer_hard = eer_of(np.asarray(scs), np.asarray(labs))

        rng = random.Random(2027)
        trials = [(e, t, "nontarget") for e, t, _ in chosen] + \
                 [(e, t, "target") for e, t in targets]
        rng.shuffle(trials)
        path = os.path.join(PROXY, f"trials_hard_{cond}.txt")
        with open(path, "w") as f:
            for e, t, lab in trials:
                f.write(f"{e} {t} {lab}\n")
        hard_nt_sim = np.mean([c[2] for c in chosen])
        print(f"[hard] {cond}: {len(targets)} target + {len(chosen)} nontarget"
              f" (负对均余弦 {hard_nt_sim:.3f}) -> {path}", flush=True)
        print(f"[sanity] soup5 hard-trials EER = {eer_hard:.3f}%"
              f"（旧随机负对 proxy: "
              f"{'2.000' if cond == 'noise_0db' else '4.261'}%）", flush=True)

        np.savez(os.path.join(PROXY, f"emb_soup5_{cond}.npz"),
                 stems=np.array(tst), embs=T)

    print("[done]", flush=True)


if __name__ == "__main__":
    main()
