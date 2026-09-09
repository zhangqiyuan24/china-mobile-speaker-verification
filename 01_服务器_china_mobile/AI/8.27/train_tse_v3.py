#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_tse_v3.py — Phase 2: TSE + nontarget suppression（8.27）

Phase 1 结论：L_spk 官方 −0.089/−0.114pp（饱和带内）——cosine 拉近只抬 target
侧，margin 的另一半（压 nontarget 相似度）未动。本脚本补上这一半：

    L = L_se + λ_ce·L_ce + 0.3·L_spk + λ_ns·L_ns
    L_ns = mean_k max(0, cos(e_enh, proto_k) − τ)   k = 除真说话人外最难 top-K

原型 = 8.21 prototypes.npz（9900×512 说话人质心，L2 归一后当负目标锚，冻结）。
基座 = lam03_ep3（Phase 1 最优，官方 7.2933/9.2020）。除 λ_ns 外一切不变。

选点协议：每 epoch 在 hard proxy（trials_hard_*，困难负对）上增强+打分，
hard avg 最小为 best。对照锚 = lam03_ep3 hard 4.307/10.500 avg 7.403。
经验换算：hard→官方 ≈ ×0.2 —— hard 移动 <1pp 不值得动官方。

用法: python train_tse_v3.py --lambda-ns 0.2 --gpu 1 [--epochs 3 --exp-tag ns02]
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
from score_proxy import eer_of

PROXY = os.path.join(_HERE, "exp", "proxy")
LAM03_EP3 = os.path.join(_HERE, "exp", "tse_v2_lam03", "checkpoint_epoch3.tar")
ANCHOR = {"noise_0db": 4.307, "noise_m5db": 10.500}   # lam03_ep3 hard
TAU = 0.30          # nontarget hinge 阈值（proxy 难负对均余弦 ~0.29 同量级）
TOPK = 8            # 每样本最难负原型数


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
    out_root = os.path.join(PROXY, f"val_v3_{tag}")
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
    ap.add_argument("--lambda-spk", type=float, default=0.3,
                    help="Phase 1 最优, 默认保持")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--exp-tag", default=None)
    ap.add_argument("--base-ckpt", default=None,
                    help="续训基座 TSE ckpt；默认 lam03_ep3。"
                         "ns05_ext 用 ns05_ep3 做官方加速曲线的延长线")
    args = ap.parse_args()

    tag = args.exp_tag or f"ns{int(round(args.lambda_ns * 10)):02d}"
    cfg = toml.load(args.config)
    # 关键差异：基座 = lam03_ep3（Phase 1 最优）, 其余不变
    cfg["paths"]["fullsubnet_ckpt"] = args.base_ckpt or LAM03_EP3
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

    # ---- L_ns 原型表：init_classifier_prototypes 已把 proto 行按 spk2id 对齐进
    # classifier.weight —— 直接取（行序=标签序, 无歧义）, L2 归一后当负目标锚 ----
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
    print(f"[train] {len(ds)} 样本 {len(loader)} it/ep × {args.epochs} ep, "
          f"lambda_ce={lam_ce}, lambda_spk={lam_spk}, lambda_ns={lam_ns} "
          f"-> {exp_dir}", flush=True)

    emb_by_spk = spk_embs_from_enrol(device)
    trials = load_hard_trials()
    best = 1e9

    for ep in range(1, args.epochs + 1):
        fullsubnet.train()
        sums = {"L": 0., "L_se": 0., "L_ce": 0., "L_spk": 0., "L_ns": 0.,
                "cos": 0., "ns": 0., "n": 0}
        for it, (noisy, clean, lb, emb) in enumerate(loader):
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
            # ---- L_ns: 最难非目标原型 hinge（梯度只穿 e_enh）----
            cos_np = F.linear(F.normalize(e_enh, dim=1), proto_t)   # B×9900
            cos_np.scatter_(1, lb.unsqueeze(1), -2.0)               # 排除真说话人
            hard_cos = cos_np.topk(TOPK, dim=1).values              # B×K
            L_ns = F.relu(hard_cos - TAU).mean()
            L = L_se + lam_ce * L_ce + lam_spk * L_spk + lam_ns * L_ns
            L.backward()
            torch.nn.utils.clip_grad_norm_(fullsubnet.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                cos = F.cosine_similarity(e_clean, e_enh, dim=1).mean()
            sums["L"] += float(L); sums["L_se"] += float(L_se)
            sums["L_ce"] += float(L_ce); sums["L_spk"] += float(L_spk)
            sums["L_ns"] += float(L_ns); sums["cos"] += float(cos)
            sums["ns"] += float(hard_cos.mean()); sums["n"] += 1
            if (it + 1) % 200 == 0:
                n = sums["n"]
                print(f"  ep{ep} it {it+1}/{len(loader)} L={sums['L']/n:.4f} "
                      f"se={sums['L_se']/n:.4f} ce={sums['L_ce']/n:.4f} "
                      f"spk={sums['L_spk']/n:.4f} ns={sums['L_ns']/n:.4f} "
                      f"hardcos={sums['ns']/n:.4f} cos={sums['cos']/n:.4f}",
                      flush=True)
                sums = {k: 0. for k in sums}
                sums["n"] = 0
        ck = {"model": fullsubnet.state_dict(), "epoch": ep,
              "lambda_ns": lam_ns, "lambda_spk": lam_spk}
        res = proxy_validate(fullsubnet, verifier, diff_fbank, emb_by_spk,
                             stft_fn, istft_fn, decompress_cIRM,
                             cfg["acoustics"], device, f"{tag}_ep{ep}", trials)
        avg = (res["noise_0db"] + res["noise_m5db"]) / 2
        print(f"  [val-hard] ep{ep} 0db={res['noise_0db']:.3f} "
              f"m5db={res['noise_m5db']:.3f} avg={avg:.3f} "
              f"(锚 lam03_ep3: 4.307/10.500 avg=7.403)", flush=True)
        ck["proxy_hard"] = res
        torch.save(ck, os.path.join(exp_dir, f"checkpoint_epoch{ep}.tar"))
        if avg < best:
            best = avg
            torch.save(ck, os.path.join(exp_dir, "checkpoint_best.tar"))
            print(f"  [val-hard] 新最优 avg={avg:.3f} -> checkpoint_best.tar",
                  flush=True)
    print(f"[done] λ_ns={lam_ns} best_hard_avg={best:.3f} (锚 7.403)", flush=True)


if __name__ == "__main__":
    main()
