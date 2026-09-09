#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""enhance_proxy.py — proxy 验证集双前端增强（8.27 Phase 0）

--frontend 818 : 8.18 FullSubNet+(spk_preserve e10)。经 build_tse_fullsubnet 装载，
                 FiLM 无 ckpt 键保持零初始化 = 恒等变换，emb 传零向量即可 —— 与
                 官方 8.18 前端数学等价。
--frontend tse : 8.26 TSE ep3。条件向量 = proxy enrol clean(每人 2 条) 过冻结 soup5
                 取均值 + L2（与训练 spk_embs 同空间，与官方 enhance_tse 同口径）。

输出: exp/proxy/enh818_noise_{0db,m5db} / enhtse_noise_{0db,m5db}
用法: python enhance_proxy.py --frontend tse --gpu 1
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import toml

_HERE = os.path.dirname(os.path.abspath(__file__))
_AI = os.path.dirname(_HERE)
for _p in (os.path.join(_AI, "8.18"), os.path.join(_AI, "8.21"),
           os.path.join(_AI, "8.26"), _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import torchaudio

from train_tse import build_tse_fullsubnet
from train_spk_preserve import _setup
from trainer_spk_preserve import load_eres2net
from frontend_spk import DiffFBank
from common_utils import setup_speakerlab

CFG_TSE = os.path.join(_AI, "8.26", "train_tse.toml")
CKPT_818 = "/home/ligangrui/china_mobile/AI/8.18/exp/spk_preserve/checkpoint_epoch10.tar"
CKPT_TSE = "/data/home/ligangrui/china_mobile/AI/8.26/exp/tse/checkpoint_epoch3.tar"
PROXY = os.path.join(_HERE, "exp", "proxy")


@torch.no_grad()
def spk_embs_from_enrol(device):
    """proxy enrol(每人 2 条 clean 3s) → soup5 → 均值 + L2。返回 {spk: (512,) np}。"""
    cfg = toml.load(CFG_TSE)
    sv_ecapa = cfg["paths"]["sv_ecapa"]
    verifier = load_eres2net(sv_ecapa, cfg["paths"]["verifier_ckpt"], device)
    verifier.eval()
    fbank = DiffFBank(n_mels=80, sample_rate=16000, n_fft=512,
                      win_length=400, hop_length=160, mean_nor=True).to(device)
    files = sorted(os.listdir(os.path.join(PROXY, "enrol")))
    groups = defaultdict(list)
    for fn in files:
        spk = fn.split("__")[0]
        groups[spk].append(os.path.join(PROXY, "enrol", fn))
    out = {}
    for spk, paths in groups.items():      # 每人最多 2 条, 直接逐人小批
        waves = []
        for p in paths:
            w, _ = torchaudio.load(p)
            waves.append(w[0])
        e = verifier(fbank(torch.stack(waves).to(device))).cpu().numpy()
        m = e.mean(0)
        out[spk] = (m / (np.linalg.norm(m) + 1e-8)).astype(np.float32)
    print(f"[cond] {len(out)} spk 嵌入(soup5, 均值+L2)", flush=True)
    return out


@torch.no_grad()
def enhance(model, stft_fn, istft_fn, decompress_cIRM, n_fft, hop, win,
            in_dir, out_dir, emb_by_spk, device, batch=8, emb_is_zero=False):
    os.makedirs(out_dir, exist_ok=True)
    items = []
    for fn in sorted(os.listdir(in_dir)):
        if os.path.isfile(os.path.join(out_dir, fn)):
            continue
        items.append(fn)
    print(f"[enhance] 待处理 {len(items)}", flush=True)
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        waves, embs = [], []
        for fn in chunk:
            w, _ = torchaudio.load(os.path.join(in_dir, fn))
            waves.append(w[0])
            if not emb_is_zero:
                embs.append(emb_by_spk[fn.split("__")[0]])
        noisy = torch.stack(waves).to(device)
        e = (torch.zeros(len(chunk), 512) if emb_is_zero else
             torch.from_numpy(np.stack(embs))).to(device)
        c = stft_fn(noisy, n_fft, hop, win)
        r = model(c.abs().unsqueeze(1), c.real.unsqueeze(1), c.imag.unsqueeze(1), e)
        r = r.permute(0, 2, 3, 1)
        dec = decompress_cIRM(r)
        er = dec[..., 0] * c.real - dec[..., 1] * c.imag
        ei = dec[..., 1] * c.real + dec[..., 0] * c.imag
        enh = istft_fn(torch.stack([er, ei], dim=-1), n_fft, hop, win,
                       length=noisy.size(-1))
        for j, fn in enumerate(chunk):
            torchaudio.save(os.path.join(out_dir, fn), enh[j].unsqueeze(0).cpu(),
                            16000, encoding="PCM_S", bits_per_sample=16)
        if (i // batch) % 20 == 0:
            print(f"  {i}/{len(items)}", flush=True)
    print(f"[enhance] done -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", choices=("818", "tse"), required=True)
    ap.add_argument("--gpu", default="1")
    args = ap.parse_args()

    cfg = toml.load(CFG_TSE)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _setup(cfg["paths"]["fullsubnet_root"], cfg["paths"]["sv_ecapa"])
    setup_speakerlab(cfg["paths"]["sv_ecapa"], _HERE)

    ckpt = CKPT_818 if args.frontend == "818" else CKPT_TSE
    model, stft_fn, istft_fn, decompress_cIRM, _ = build_tse_fullsubnet(
        cfg["paths"]["fullsubnet_root"], ckpt, cfg["model"], device)
    model.eval()
    n_fft, hop, win = (cfg["acoustics"][k]
                       for k in ("n_fft", "hop_length", "win_length"))

    emb_by_spk = None
    emb_is_zero = args.frontend == "818"
    if not emb_is_zero:
        emb_by_spk = spk_embs_from_enrol(device)

    tag = "enh818" if args.frontend == "818" else "enhtse"
    for cond in ("noise_0db", "noise_m5db"):
        enhance(model, stft_fn, istft_fn, decompress_cIRM, n_fft, hop, win,
                os.path.join(PROXY, cond),
                os.path.join(PROXY, f"{tag}_{cond}"),
                emb_by_spk, device, emb_is_zero=emb_is_zero)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
