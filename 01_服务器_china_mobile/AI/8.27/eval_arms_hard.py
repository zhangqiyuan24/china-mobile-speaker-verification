#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_arms_hard.py — Phase 1 三臂 ckpt 在 hard proxy 上的离线评测（8.27）

终判仪器：对每个臂的每个 epoch ckpt —— 增强 proxy 带噪集 → 冻结 soup5 提嵌
入 → trials_hard_{cond} 打分 → hard EER。对照锚点：
  tse_soup5（TSE ep3 原模型）hard = 4.716 / 11.193（score_proxy_hard 记录）
enrol 嵌入直接复用 mine_hard_trials 的缓存（同一 soup5 同一批 enrol）。

只读 ckpt/音频，只新增 exp/proxy/val_{tag}_ep{n}/ 增强音频。
用法: python eval_arms_hard.py --gpu 1 [--tags lam01,lam03,lam05]
                          [--epochs 1,2,3,best]
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

from train_tse import build_tse_fullsubnet
from train_spk_preserve import _setup
from trainer_spk_preserve import load_eres2net
from frontend_spk import DiffFBank
from common_utils import setup_speakerlab
from enhance_proxy import enhance, spk_embs_from_enrol

PROXY = os.path.join(_HERE, "exp", "proxy")
CFG = os.path.join(_AI, "8.26", "train_tse.toml")
ANCHOR = {"noise_0db": 4.716, "noise_m5db": 11.193}   # tse_soup5 hard


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
    import torchaudio
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
    ap.add_argument("--tags", default="lam01,lam03,lam05")
    ap.add_argument("--epochs", default="1,2,3,best")
    ap.add_argument("--batch", type=int, default=4,
                    help="enhance batch（与训练共卡时防 OOM）")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda:0"

    cfg = toml.load(CFG)
    sv_ecapa = os.path.abspath(cfg["paths"]["sv_ecapa"])
    fullsubnet_root = os.path.abspath(cfg["paths"]["fullsubnet_root"])
    base = os.path.join(_AI, "8.27")
    _setup(fullsubnet_root, sv_ecapa)
    setup_speakerlab(sv_ecapa, base)

    # soup5 验证后端 + enrol 缓存
    fbank = DiffFBank(n_mels=80, sample_rate=16000, n_fft=512,
                      win_length=400, hop_length=160, mean_nor=True).to(device)
    verifier = load_eres2net(sv_ecapa, cfg["paths"]["verifier_ckpt"], device)
    verifier.eval()
    z = np.load(os.path.join(PROXY, "emb_soup5_enrol.npz"), allow_pickle=True)
    e_enrol = {s: e for s, e in zip(z["stems"], z["embs"])}

    trials = {}
    for cond in ("noise_0db", "noise_m5db"):
        rows = []
        with open(os.path.join(PROXY, f"trials_hard_{cond}.txt")) as f:
            for line in f:
                p = line.split()
                if len(p) == 3:
                    rows.append((p[0], p[1], p[2]))
        trials[cond] = rows

    emb_by_spk = spk_embs_from_enrol(device)
    n_fft, hop, win = (cfg["acoustics"][k]
                       for k in ("n_fft", "hop_length", "win_length"))

    def cos(a, b):
        return float(np.dot(a, b) /
                     (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    print(f"{'ckpt':<16}{'0dB hard':>10}{'-5dB hard':>11}{'avg':>8}"
          f"   (锚点 tse_soup5: 4.716/11.193 avg 7.955)", flush=True)
    for tag in args.tags.split(","):
        for ep in args.epochs.split(","):
            ck = ("checkpoint_best.tar" if ep == "best"
                  else f"checkpoint_epoch{ep}.tar")
            path = os.path.join(_HERE, "exp", f"tse_v2_{tag}", ck)
            if not os.path.exists(path):
                print(f"[skip] {path} 不存在", flush=True)
                continue
            model, stft_fn, istft_fn, dec, _ = build_tse_fullsubnet(
                fullsubnet_root, path, cfg["model"], device)
            model.eval()
            res = {}
            for cond in ("noise_0db", "noise_m5db"):
                out_dir = os.path.join(PROXY, "val_hard",
                                        f"{tag}_ep{ep}", cond)
                with torch.no_grad():
                    enhance(model, stft_fn, istft_fn, dec,
                            n_fft, hop, win,
                            os.path.join(PROXY, cond), out_dir,
                            emb_by_spk, device, batch=args.batch)
                files = [os.path.join(out_dir, f)
                         for f in sorted(os.listdir(out_dir))]
                e_test = extract(verifier, fbank, files, device)
                labs, scs = [], []
                for e_stem, t_stem, lab in trials[cond]:
                    scs.append(cos(e_enrol[e_stem], e_test[t_stem]))
                    labs.append(1 if lab == "target" else 0)
                res[cond] = eer_of(np.asarray(scs), np.asarray(labs))
            avg = (res["noise_0db"] + res["noise_m5db"]) / 2
            print(f"{tag+'_ep'+ep:<16}{res['noise_0db']:>10.3f}"
                  f"{res['noise_m5db']:>11.3f}{avg:>8.3f}", flush=True)
            del model
            torch.cuda.empty_cache()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
