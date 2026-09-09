#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_tse_v4.py — 时长适配臂（#25，8.28）

背景：#19 实测 1s FE 降幅 −19.4% vs 3s 主线 −34.8%。主因 = 前端固定 3s 裁剪
训练（train_tse.toml sub_sample_length=3.0），1s 推理是长度维度的分布外。
B6 后端同因（嵌入空间外）另案处理（#23 multi-verifier，9 月）。

方案：v3 全套损失不动（L_se + λ_ce·L_ce + 0.3·L_spk + λ_ns·L_ns），唯一差异
--dur-aug：每 dur_seg 个 iteration 把 dataset.target_len 在 {3,2,1}s 中按
{0.5,0.25,0.25} 切一段（段级而非样本级 —— 同 batch 等长天然保持；worker
预取使段边界 1-2 个 batch 的 dur 错位，但 batch 内自洽，损失与 dur 无关，
无害）。3s 占一半保主线；proxy hard val（3s）即门槛①监控。

双门槛预声明（8.28）：
  ① 3s hard proxy 不降：best avg ≥ base ckpt 的 avg − 0.3pp 之内
  ② 1s 官方 FE 降幅 ≥ 28%（从 −19.4% 起步）
两门槛都过才启用；不过则报 base ckpt 诚实数字。

用法: python train_tse_v4.py --lambda-ns 0.5 --base-ckpt <ext3best|ext2ep9> \
        --exp-tag ns05duraug --epochs 3 --gpu 3 --dur-aug
"""
import argparse
import os
import random
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
from score_proxy import eer_of

PROXY = os.path.join(_HERE, "exp", "proxy")
TAU = 0.30
TOPK = 8
DUR_CHOICES = (3, 2, 1)
DUR_PROBS = (0.5, 0.25, 0.25)


def load_hard_trials():
    trials = {}
    for cond in ("noise_0db", "noise_m5db"):
        rows = []
        with open(os.path.join(PROXY, f"trials_hard_{cond}.txt")) as f:
            for line in f:
                p = line.split()
                if len(p) == 3:
                    rows.append((p[0], p[1], p[2]))
        trials[cond] = rows
    return trials


@torch.no_grad()
def proxy_validate(model, verifier, fbank, emb_by_spk, stft_fn, istft_fn,
                   decompress_cIRM, acoustics, device, tag, trials):
    model.eval()
    n_fft, hop, win = (acoustics[k] for k in ("n_fft", "hop_length", "win_length"))
    out_root = os.path.join(PROXY, f"val_v4_{tag}")
    res = {}
    for cond in ("noise_0db", "noise_m5db"):
        out_dir = os.path.join(out_root, cond)
        enhance(model, stft_fn, istft_fn, decompress_cIRM, n_fft, hop, win,
                os.path.join(PROXY, cond), out_dir, emb_by_spk, device)
        enrol_files = [os.path.join(PROXY, "enrol", f)
                       for f in sorted(os.listdir(os.path.join(PROXY, "enrol")))]
        test_files = [os.path.join(out_dir, f)
                      for f in sorted(os.listdir(out_dir))]
        e_enrol = {}
        for i in range(0, len(enrol_files), 16):
            import torchaudio
            chunk = enrol_files[i:i + 16]
            waves = [torchaudio.load(p)[0][0] for p in chunk]
            e = verifier(fbank(torch.stack(waves).to(device))).cpu().numpy()
            for p, row in zip(chunk, e):
                e_enrol[os.path.splitext(os.path.basename(p))[0]] = row
        e_test = {}
        for i in range(0, len(test_files), 16):
            import torchaudio
            chunk = test_files[i:i + 16]
            waves = [torchaudio.load(p)[0][0] for p in chunk]
            e = verifier(fbank(torch.stack(waves).to(device))).cpu().numpy()
            for p, row in zip(chunk, e):
                e_test[os.path.splitext(os.path.basename(p))[0]] = row
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
        _AI, "8.26", "train_tse.toml"))
    ap.add_argument("--lambda-ns", type=float, required=True)
    ap.add_argument("--lambda-spk", type=float, default=0.3)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--exp-tag", default=None)
    ap.add_argument("--base-ckpt", required=True,
                    help="续训基座（ext3 best 或 ext2 ep9，由官方数字定）")
    ap.add_argument("--dur-aug", action="store_true",
                    help="开启段级随机时长 {3,2,1}s @ {0.5,0.25,0.25}")
    ap.add_argument("--dur-seg", type=int, default=200,
                    help="每多少个 iteration 换一段时长")
    ap.add_argument("--seed", type=int, default=25)
    args = ap.parse_args()

    tag = args.exp_tag or f"ns{int(round(args.lambda_ns * 10)):02d}"
    cfg = toml.load(args.config)
    cfg["paths"]["fullsubnet_ckpt"] = args.base_ckpt
    cfg["paths"]["exp_dir"] = os.path.join(_HERE, "exp", f"tse_v3_{tag}")
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
    assert list(spk2id.keys()) == list(emb_npz["spk_ids"])
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

    proto_t = F.normalize(classifier.weight.detach().float(), dim=1)
    print(f"[proto] {proto_t.shape[0]} 行 × {proto_t.shape[1]} 维 (τ={TAU}, K={TOPK})",
          flush=True)

    margin_loss = build_aam_margin(32.0, 0.2).to(device)
    diff_fbank = DiffFBank(n_mels=80, sample_rate=sr, n_fft=512,
                           win_length=400, hop_length=160, mean_nor=True,
                           ).to(device)

    trainable = [p for p in fullsubnet.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=cfg["training"]["lr"])
    lam_ce = float(cfg["training"]["lambda_ce"])
    lam_spk = args.lambda_spk
    lam_ns = args.lambda_ns
    n_fft, hop, win = (cfg["acoustics"][k]
                       for k in ("n_fft", "hop_length", "win_length"))
    dur_rng = random.Random(args.seed)
    dur_plan = []          # [(it 起点, 秒数)]，供日志与复现
    print(f"[train] {len(ds)} 样本 {len(loader)} it/ep × {args.epochs} ep, "
          f"lambda_ce={lam_ce}, lambda_spk={lam_spk}, lambda_ns={lam_ns}, "
          f"dur_aug={'ON seg=%d it, 3/2/1s @ 0.5/0.25/0.25' % args.dur_seg if args.dur_aug else 'OFF'} "
          f"-> {exp_dir}", flush=True)

    emb_by_spk = spk_embs_from_enrol(device)
    trials = load_hard_trials()
    best = 1e9

    for ep in range(1, args.epochs + 1):
        fullsubnet.train()
        sums = {"L": 0., "L_se": 0., "L_ce": 0., "L_spk": 0., "L_ns": 0.,
                "cos": 0., "ns": 0., "n": 0}
        for it, (noisy, clean, lb, emb) in enumerate(loader):
            if args.dur_aug and it % args.dur_seg == 0:
                dur = dur_rng.choices(DUR_CHOICES, weights=DUR_PROBS, k=1)[0]
                ds.target_len = int(dur * sr)
                dur_plan.append((ep, it, dur))
                print(f"  [dur] ep{ep} it{it} -> {dur}s 段", flush=True)
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            lb = lb.to(device, non_blocking=True)
            emb = emb.to(device, non_blocking=True)
            opt.zero_grad()
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
            clean_complex = stft_fn(clean, n_fft, hop, win)
            L_se = F.mse_loss(cRM_raw, build_cIRM(noisy_complex, clean_complex))
            e_enh = verifier(diff_fbank(enhanced))
            L_ce = margin_loss(classifier(e_enh).float(), lb)
            with torch.no_grad():
                e_clean = verifier(diff_fbank(clean))
            L_spk = (1 - F.cosine_similarity(e_clean, e_enh, dim=1)).mean()
            cos_np = F.linear(F.normalize(e_enh, dim=1), proto_t)
            cos_np.scatter_(1, lb.unsqueeze(1), -2.0)
            hard_cos = cos_np.topk(TOPK, dim=1).values
            L_ns = F.relu(hard_cos - TAU).mean()
            L = L_se + lam_ce * L_ce + lam_spk * L_spk + lam_ns * L_ns
            L.backward()
            torch.nn.utils.clip_grad_norm_(fullsubnet.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                cos = F.cosine_similarity(e_clean, e_enh, dim=1).mean()
            sums["L"] += float(L); sums["L_se"] += float(L_se)
            sums["L_ce"] += float(L_ce); sums["L_spk"] += float(L_spk)
            sums["L_ns"] += float(L_ns)
            sums["cos"] += float(cos)
            sums["ns"] += float(hard_cos.mean()); sums["n"] += 1
            if (it + 1) % 200 == 0:
                n = sums["n"]
                print(f"  ep{ep} it {it+1}/{len(loader)} L={sums['L']/n:.4f} "
                      f"se={sums['L_se']/n:.4f} ce={sums['L_ce']/n:.4f} "
                      f"spk={sums['L_spk']/n:.4f} ns={sums['L_ns']/n:.4f} "
                      f"hardcos={sums['ns']/n:.4f} cos={sums['cos']/n:.4f} "
                      f"dur={ds.target_len // sr}s",
                      flush=True)
                sums = {k: 0. for k in sums}
                sums["n"] = 0
        ck = {"model": fullsubnet.state_dict(), "epoch": ep,
              "lambda_ns": lam_ns, "lambda_spk": lam_spk,
              "dur_aug": bool(args.dur_aug)}
        res = proxy_validate(fullsubnet, verifier, diff_fbank, emb_by_spk,
                             stft_fn, istft_fn, decompress_cIRM,
                             cfg["acoustics"], device, f"{tag}_ep{ep}", trials)
        avg = (res["noise_0db"] + res["noise_m5db"]) / 2
        print(f"  [val-hard] ep{ep} 0db={res['noise_0db']:.3f} "
              f"m5db={res['noise_m5db']:.3f} avg={avg:.3f} "
              f"(门槛①: base avg − 0.3 以内)", flush=True)
        ck["proxy_hard"] = res
        torch.save(ck, os.path.join(exp_dir, f"checkpoint_epoch{ep}.tar"))
        if avg < best:
            best = avg
            torch.save(ck, os.path.join(exp_dir, "checkpoint_best.tar"))
            print(f"  [val-hard] 新最优 avg={avg:.3f} -> checkpoint_best.tar",
                  flush=True)
    with open(os.path.join(exp_dir, "dur_plan.txt"), "w") as f:
        for row in dur_plan:
            f.write(f"ep{row[0]} it{row[1]} {row[2]}s\n")
    print(f"[done] λ_ns={lam_ns} dur_aug={args.dur_aug} "
          f"best_hard_avg={best:.3f} (门槛①线见上)", flush=True)


if __name__ == "__main__":
    main()
