#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_tse_v2.py — Phase 1: TSE + speaker preservation（8.27）

基于 8.26 已训好的 TSE(ep3) 续训，不改动网络/优化器/数据 —— 唯一变量 λ_spk：
    L = L_se(cIRM MSE) + λ_ce·L_ce(AAM-CE) + λ_spk·(1 − cos(e_enh, e_clean))
    e_clean: 同条 clean 过冻结 soup5（no_grad, 常数目标）
    e_enh  : TSE 增强过同一冻结 soup5（不 detach, 梯度穿过 ERes2Net 回传 TSE）
（L_spk 分支在 8.26 train_tse.py 中已实现但 λ=0 未启用 —— 本脚本只开它。）

选点协议：每 epoch 结束在 proxy-official 验证集上增强+打分（soup5 口径），
proxy avg(0dB,m5dB) 最小者为 best。旧小 val 不参与选点。
基线对照（proxy, 同口径）：tse_soup5 = 2.000 / 4.261, avg 3.131。

用法: python train_tse_v2.py --lambda-spk 0.3 --gpu 1 [--epochs 3 --exp-tag lam03]
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import toml

_HERE = os.path.dirname(os.path.abspath(__file__))
_AI = os.path.dirname(_HERE)
for _p in (os.path.join(_AI, "8.18"), os.path.join(_AI, "8.20"),
           os.path.join(_AI, "8.21"), os.path.join(_AI, "8.26"), _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from train_tse import (TSEMixDataset, build_paths_from_csv,
                       build_tse_fullsubnet)
from train_spk_preserve import load_noise_paths, uniform_select, _setup
from trainer_spk_preserve import load_eres2net
from frontend_spk import DiffFBank
from common_utils import setup_speakerlab
from trainer_eres2net_robust import build_aam_margin, build_classifier
from trainer_eres2net_robust_proto import (load_prototypes,
                                           init_classifier_prototypes)
from enhance_proxy import enhance, spk_embs_from_enrol
from score_proxy import extract, eer_of

PROXY = os.path.join(_HERE, "exp", "proxy")
BASELINE_PROXY = {"noise_0db": 2.000, "noise_m5db": 4.261}  # tse_soup5(ep3)


def load_proxy_trials():
    trials = {}
    for cond in ("noise_0db", "noise_m5db"):
        rows = []
        with open(os.path.join(PROXY, f"trials_{cond}.txt")) as f:
            for line in f:
                p = line.split()
                if len(p) == 3:
                    rows.append((p[0], p[1], p[2]))
        trials[cond] = rows
    return trials


@torch.no_grad()
def proxy_validate(model, verifier, fbank, emb_by_spk, stft_fn, istft_fn,
                   decompress_cIRM, acoustics, device, tag, trials):
    """当前 TSE 模型在 proxy 上的 0dB/m5dB EER（增强→soup5→余弦→EER）。"""
    model.eval()
    n_fft, hop, win = (acoustics[k] for k in ("n_fft", "hop_length", "win_length"))
    out_root = os.path.join(PROXY, f"val_{tag}")
    res = {}
    for cond in ("noise_0db", "noise_m5db"):
        out_dir = os.path.join(out_root, cond)
        enhance(model, stft_fn, istft_fn, decompress_cIRM, n_fft, hop, win,
                os.path.join(PROXY, cond), out_dir, emb_by_spk, device)
        enrol_files = [os.path.join(PROXY, "enrol", f)
                       for f in sorted(os.listdir(os.path.join(PROXY, "enrol")))]
        test_files = [os.path.join(out_dir, f)
                      for f in sorted(os.listdir(out_dir))]
        e_enrol = extract(verifier, fbank, enrol_files, device)
        e_test = extract(verifier, fbank, test_files, device)
        labs, scs = [], []
        for e_stem, t_stem, lab in trials[cond]:
            a, b = e_enrol[e_stem], e_test[t_stem]
            scs.append(float(np.dot(a, b) /
                             (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)))
            labs.append(1 if lab == "target" else 0)
        res[cond] = eer_of(np.asarray(scs), np.asarray(labs))
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        _AI, "8.26", "train_tse.toml"))     # 数据/声学/模型段照抄 8.26
    ap.add_argument("--lambda-spk", type=float, required=True)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--exp-tag", default=None)
    args = ap.parse_args()

    tag = args.exp_tag or f"lam{int(round(args.lambda_spk * 10)):02d}"
    cfg = toml.load(args.config)
    # 关键差异：从 8.26 TSE ep3 续训（其余不变）
    cfg["paths"]["fullsubnet_ckpt"] = (
        "/data/home/ligangrui/china_mobile/AI/8.26/exp/tse/checkpoint_epoch3.tar")
    cfg["paths"]["exp_dir"] = os.path.join(_HERE, "exp", f"tse_v2_{tag}")
    if args.max_train:
        cfg["dataset"]["max_train_samples"] = args.max_train

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda:0"
    base = _HERE
    sv_ecapa = os.path.abspath(cfg["paths"]["sv_ecapa"])
    fullsubnet_root = os.path.abspath(cfg["paths"]["fullsubnet_root"])
    _setup(fullsubnet_root, sv_ecapa)
    setup_speakerlab(sv_ecapa, base)
    exp_dir = cfg["paths"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)

    clean_paths, labels, spk2id = build_paths_from_csv(
        os.path.join(cfg["paths"]["data_dir_820"], "train_utts.csv"))
    emb_npz = np.load(cfg["paths"]["spk_embs"], allow_pickle=True)
    assert list(spk2id.keys()) == list(emb_npz["spk_ids"]), \
        "spk_embs.npz 与 train_utts.csv 顺序不一致"
    emb_table = emb_npz["embs"]

    noise_train = load_noise_paths(cfg["dataset"]["noise_list"], sv_ecapa)
    max_train = int(cfg["dataset"].get("max_train_samples", 0))
    if 0 < max_train < len(clean_paths):
        sel = uniform_select(list(range(len(clean_paths))), max_train)
        clean_paths = [clean_paths[i] for i in sel]
        labels = [labels[i] for i in sel]

    sr = cfg["acoustics"]["sr"]
    target_len = int(cfg["dataset"]["sub_sample_length"] * sr)
    ds = TSEMixDataset(clean_paths, labels, emb_table, noise_train,
                       cfg["dataset"]["snr_range"], target_len, sr)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg["dataset"]["batch_size"], shuffle=True,
        num_workers=cfg["dataset"]["num_workers"], drop_last=True,
        pin_memory=True)

    fullsubnet, stft_fn, istft_fn, decompress_cIRM, build_cIRM = \
        build_tse_fullsubnet(fullsubnet_root, cfg["paths"]["fullsubnet_ckpt"],
                             cfg["model"], device)
    print(f"[init] 续训自 {cfg['paths']['fullsubnet_ckpt']}", flush=True)

    verifier = load_eres2net(sv_ecapa, cfg["paths"]["verifier_ckpt"], device)
    verifier.eval()
    classifier = build_classifier(512, len(spk2id))
    protos, proto_spk_ids, _ = load_prototypes(cfg["paths"]["proto_npz"])
    init_classifier_prototypes(classifier, protos, proto_spk_ids, spk2id)
    classifier = classifier.to(device)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    margin_loss = build_aam_margin(32.0, 0.2).to(device)
    diff_fbank = DiffFBank(n_mels=80, sample_rate=sr, n_fft=512,
                           win_length=400, hop_length=160, mean_nor=True,
                           ).to(device)

    trainable = [p for p in fullsubnet.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=cfg["training"]["lr"])
    lam_ce = float(cfg["training"]["lambda_ce"])
    lam_spk = args.lambda_spk
    n_fft, hop, win = (cfg["acoustics"][k]
                       for k in ("n_fft", "hop_length", "win_length"))
    print(f"[train] {len(ds)} 样本 {len(loader)} it/ep × {args.epochs} ep, "
          f"lambda_ce={lam_ce}, lambda_spk={lam_spk} -> {exp_dir}", flush=True)

    emb_by_spk = spk_embs_from_enrol(device)       # proxy 条件嵌入（一次）
    trials = load_proxy_trials()
    best = 1e9

    def enhance_train(noisy, emb):
        noisy_complex = stft_fn(noisy, n_fft, hop, win)
        cRM_raw = fullsubnet(noisy_complex.abs().unsqueeze(1),
                             noisy_complex.real.unsqueeze(1),
                             noisy_complex.imag.unsqueeze(1), emb)
        cRM_raw = cRM_raw.permute(0, 2, 3, 1)
        dec = decompress_cIRM(cRM_raw)
        er = dec[..., 0] * noisy_complex.real - dec[..., 1] * noisy_complex.imag
        ei = dec[..., 1] * noisy_complex.real + dec[..., 0] * noisy_complex.imag
        enhanced = istft_fn(torch.stack([er, ei], dim=-1), n_fft, hop, win,
                            length=noisy.size(-1))
        return enhanced, noisy_complex, cRM_raw

    for ep in range(1, args.epochs + 1):
        fullsubnet.train()
        sums = {"L": 0., "L_se": 0., "L_ce": 0., "L_spk": 0., "cos": 0., "n": 0}
        for it, (noisy, clean, lb, emb) in enumerate(loader):
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            lb = lb.to(device, non_blocking=True)
            emb = emb.to(device, non_blocking=True)
            opt.zero_grad()
            enhanced, noisy_complex, cRM_raw = enhance_train(noisy, emb)
            clean_complex = stft_fn(clean, n_fft, hop, win)
            L_se = F.mse_loss(cRM_raw, build_cIRM(noisy_complex, clean_complex))
            e_enh = verifier(diff_fbank(enhanced))          # 梯度穿冻结 verifier
            L_ce = margin_loss(classifier(e_enh).float(), lb)
            with torch.no_grad():                            # clean 目标 = 常数
                e_clean = verifier(diff_fbank(clean))
            L_spk = (1 - F.cosine_similarity(e_clean, e_enh, dim=1)).mean()
            L = L_se + lam_ce * L_ce + lam_spk * L_spk
            L.backward()
            torch.nn.utils.clip_grad_norm_(fullsubnet.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                cos = F.cosine_similarity(e_clean, e_enh, dim=1).mean()
            sums["L"] += float(L); sums["L_se"] += float(L_se)
            sums["L_ce"] += float(L_ce); sums["L_spk"] += float(L_spk)
            sums["cos"] += float(cos); sums["n"] += 1
            if (it + 1) % 200 == 0:
                n = sums["n"]
                print(f"  ep{ep} it {it+1}/{len(loader)} L={sums['L']/n:.4f} "
                      f"se={sums['L_se']/n:.4f} ce={sums['L_ce']/n:.4f} "
                      f"spk={sums['L_spk']/n:.4f} cos={sums['cos']/n:.4f}",
                      flush=True)
                sums = {"L": 0., "L_se": 0., "L_ce": 0., "L_spk": 0.,
                        "cos": 0., "n": 0}
        ck = {"model": fullsubnet.state_dict(), "epoch": ep,
              "lambda_spk": lam_spk}
        torch.save(ck, os.path.join(exp_dir, f"checkpoint_epoch{ep}.tar"))
        res = proxy_validate(fullsubnet, verifier, diff_fbank, emb_by_spk,
                             stft_fn, istft_fn, decompress_cIRM,
                             cfg["acoustics"], device, f"{tag}_ep{ep}", trials)
        avg = (res["noise_0db"] + res["noise_m5db"]) / 2
        print(f"  [val-proxy] ep{ep} 0db={res['noise_0db']:.3f} "
              f"m5db={res['noise_m5db']:.3f} avg={avg:.3f} "
              f"(基线 tse_soup5: 2.000/4.261 avg=3.131)", flush=True)
        ck["proxy_val"] = res
        torch.save(ck, os.path.join(exp_dir, f"checkpoint_epoch{ep}.tar"))
        if avg < best:
            best = avg
            torch.save(ck, os.path.join(exp_dir, "checkpoint_best.tar"))
            print(f"  [val-proxy] 新最优 avg={avg:.3f} -> checkpoint_best.tar",
                  flush=True)
    print(f"[done] λ_spk={lam_spk} best_proxy_avg={best:.3f} "
          f"(基线 3.131)", flush=True)


if __name__ == "__main__":
    main()
