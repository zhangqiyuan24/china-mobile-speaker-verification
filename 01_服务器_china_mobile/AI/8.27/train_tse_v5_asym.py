#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_tse_v5_asym.py — 冒认防御专项：非对称条件化训练（#38，8.31）

背景：claim 口径重测（AI/8.31/claim_retest）实锤——训练条件恒=真身（AS-only），
模型学到的行为是"输出音色向条件靠拢"；部署时条件=声称者≠真身，L_spk 的对称
拉力把冒认语音拉向声称者：claim EER 10.72/12.85 反超 raw 8.37/11.09，冒认对
余弦中位 ×3（0.047→0.129）。文献：Delcroix Interspeech 2022（TSE-IS/TSE-V）。

方案：r5 配方不动（L_se + λ_ce·L_ce + 0.3·L_spk + λ_ns·L_ns + dur-aug），
唯一差异 --imp-prob：每个训练样本以 p 概率把 FiLM 条件换成**随机他人**的
注册嵌入（冒认条件样本）。三个损失锚全部不变、仍锚在**真身**（L_se→真身
clean、L_ce→真身标签、L_spk→真身 clean 嵌入锚），即教模型："条件匹配→按
条件提取；条件不匹配→忠实增强在场说话人，**不向条件变声**"。
可选 --lam-imp>0：反拉项 relu(cos(e_enh, 条件嵌入) − τ_imp)，仅对冒认样本
生效（把"像声称者"本身变成被罚项）。

双门槛预声明（8.31）：
  ① oracle 不降：proxy hard val avg ≤ r5 基线 + 0.3pp；> +0.5pp 红灯回退 r5
     （r5 基线以 --eval-only 实测为准，历史 hard avg=2.847）
  ② claim 修复：proxy claim EER 显著低于 r5 基线值，冒认余弦中位数回落向 raw
官方评测只在臂胜出后收口一次（官方数字永不回流训练决策）。

用法: python train_tse_v5_asym.py --imp-prob 0.3 --lambda-ns 0.5 \
        --base-ckpt .../tse_v3_ns05duraug_r5/checkpoint_best.tar \
        --exp-tag asym30 --epochs 3 --gpu 0 --dur-aug [--lam-imp 0.3 --tau-imp 0.1]
      python train_tse_v5_asym.py --imp-prob 0 --eval-only --base-ckpt ... # 只测基线
"""
import argparse
import json
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

from train_tse import TSEMixDataset, build_paths_from_csv, build_tse_fullsubnet
from train_spk_preserve import load_noise_paths, uniform_select, _setup
from trainer_spk_preserve import load_eres2net
from frontend_spk import DiffFBank
from common_utils import setup_speakerlab
from trainer_eres2net_robust import build_aam_margin, build_classifier
from trainer_eres2net_robust_proto import (load_prototypes,
                                           init_classifier_prototypes)
from enhance_proxy import spk_embs_from_enrol
from score_proxy import eer_of

PROXY = os.path.join(_HERE, "exp", "proxy")
TAU = 0.30
TOPK = 8
DUR_CHOICES = (3, 2, 1)
DUR_PROBS = (0.5, 0.25, 0.25)


class TSEMixDatasetAsym(TSEMixDataset):
    """TSEMixDataset + 以 imp_prob 概率把条件嵌入换成随机他人（冒认条件样本）。

    追加返回 imp 标志（1.0=冒认样本）；三个损失的锚（clean 重建 / 分类标签 /
    L_spk 锚）全部仍是真身，与 v4 完全一致——冒认样本教的是"别向条件变声"。
    """

    def __init__(self, *a, imp_prob=0.0, imp_seed=1234, **kw):
        super().__init__(*a, **kw)
        self.imp_prob = float(imp_prob)
        self.imp_rng = random.Random(imp_seed + os.getpid())  # worker 级随机
        self.n_spk = int(self.emb_table.shape[0])

    def __getitem__(self, idx):
        noisy, clean, lb, emb = super().__getitem__(idx)
        if self.imp_prob > 0 and self.imp_rng.random() < self.imp_prob:
            other = self.imp_rng.randrange(self.n_spk - 1)
            if other >= int(lb):
                other += 1
            emb = torch.from_numpy(self.emb_table[other].copy())
            return noisy, clean, lb, emb, torch.tensor(1.0)
        return noisy, clean, lb, emb, torch.tensor(0.0)


def load_hard_trials():
    trials = {}
    for cond in ("noise_0db", "noise_m5db"):
        rows = []
        with open(os.path.join(PROXY, f"trials_hard_{cond}.txt")) as f:
            for line in f:
                e, t, lab = line.split()
                rows.append((os.path.splitext(e)[0], os.path.splitext(t)[0], lab))
        trials[cond] = rows
    return trials


def _embed_dir(verifier, fbank, files, device):
    import torchaudio
    out = {}
    for i in range(0, len(files), 16):
        chunk = files[i:i + 16]
        waves = [torchaudio.load(p)[0][0] for p in chunk]
        e = verifier(fbank(torch.stack(waves).to(device))).cpu().numpy()
        for p, row in zip(chunk, e):
            out[os.path.splitext(os.path.basename(p))[0]] = row
    return out


def _enhance_map(model, stft_fn, istft_fn, decompress_cIRM, n_fft, hop, win,
                 in_dir, out_dir, emb_of, device, batch=8):
    """enhance() 变体：条件嵌入由 emb_of(文件名) 显式给出（支持冒认映射）。"""
    import torchaudio
    os.makedirs(out_dir, exist_ok=True)
    items = [fn for fn in sorted(os.listdir(in_dir))
             if not os.path.exists(os.path.join(out_dir, fn))]
    print(f"[enhance] {os.path.basename(in_dir)} 待处理 {len(items)}", flush=True)
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        waves = [torchaudio.load(os.path.join(in_dir, fn))[0][0] for fn in chunk]
        e = torch.from_numpy(
            np.stack([emb_of(fn) for fn in chunk]).astype(np.float32)).to(device)
        noisy = torch.stack(waves).to(device)
        c = stft_fn(noisy, n_fft, hop, win)
        r = model(c.abs().unsqueeze(1), c.real.unsqueeze(1), c.imag.unsqueeze(1), e)
        r = r.permute(0, 2, 3, 1)
        dec = decompress_cIRM(r)
        er = dec[..., 0] * c.real - dec[..., 1] * c.imag
        ei = dec[..., 1] * c.real + dec[..., 0] * c.imag
        enh = istft_fn(torch.stack([er, ei], dim=-1), n_fft, hop, win,
                       length=noisy.size(-1)).cpu()
        for fn, w in zip(chunk, enh):
            torchaudio.save(os.path.join(out_dir, fn), w.unsqueeze(0), 16000)


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def validate_dual(model, verifier, fbank, emb_by_spk, stft_fn, istft_fn,
                  decompress_cIRM, acoustics, device, tag, trials):
    """oracle（hard trials，同 v4 proxy_validate）+ claim（冒认条件）双评测。

    claim nontarget：每个测试句用**错位说话人**（排序后移一位，确定性）的
    条件增强，与该条件者的注册嵌入算余弦 → 冒认分数；target 分沿用 oracle
    增强的 hard trials target 对——与官方 claim 重测同构
    （target=FE·oracle，nontarget=FE·claim）。
    """
    model.eval()
    n_fft, hop, win = (acoustics[k] for k in ("n_fft", "hop_length", "win_length"))
    spks = sorted(emb_by_spk)
    wrong = {s: spks[(i + 1) % len(spks)] for i, s in enumerate(spks)}
    res, claim = {}, {}
    for cond in ("noise_0db", "noise_m5db"):
        out_or = os.path.join(PROXY, f"val_v5_{tag}", cond)
        out_cl = os.path.join(PROXY, f"val_v5_{tag}_claim", cond)
        _enhance_map(model, stft_fn, istft_fn, decompress_cIRM, n_fft, hop, win,
                     os.path.join(PROXY, cond), out_or,
                     lambda fn: emb_by_spk[fn.split("__")[0]], device)
        _enhance_map(model, stft_fn, istft_fn, decompress_cIRM, n_fft, hop, win,
                     os.path.join(PROXY, cond), out_cl,
                     lambda fn: emb_by_spk[wrong[fn.split("__")[0]]], device)
        enrol_files = sorted(os.listdir(os.path.join(PROXY, "enrol")))
        e_enrol = _embed_dir(verifier, fbank,
                             [os.path.join(PROXY, "enrol", f) for f in enrol_files],
                             device)
        e_or = _embed_dir(verifier, fbank,
                          [os.path.join(out_or, f)
                           for f in sorted(os.listdir(out_or))], device)
        e_cl = _embed_dir(verifier, fbank,
                          [os.path.join(out_cl, f)
                           for f in sorted(os.listdir(out_cl))], device)
        labs, scs, tgt_sc = [], [], []
        for e_stem, t_stem, lab in trials[cond]:
            sc = _cos(e_enrol[e_stem], e_or[t_stem])
            scs.append(sc)
            labs.append(1 if lab == "target" else 0)
            if lab == "target":
                tgt_sc.append(sc)
        res[cond] = eer_of(np.asarray(scs), np.asarray(labs))
        nt = np.asarray([_cos(emb_by_spk[wrong[stem.split("__")[0]]], b)
                         for stem, b in e_cl.items()])
        sc_cl = np.concatenate([np.asarray(tgt_sc), nt])
        lb_cl = np.concatenate([np.ones(len(tgt_sc)), np.zeros(len(nt))])
        claim[cond] = {"eer": eer_of(sc_cl, lb_cl),
                       "nt_med": float(np.median(nt)),
                       "nt_p90": float(np.percentile(nt, 90))}
        print(f"  [val-claim] {cond} eer={claim[cond]['eer']:.3f} "
              f"nt_med={claim[cond]['nt_med']:.3f} nt_p90={claim[cond]['nt_p90']:.3f}",
              flush=True)
    model.train()
    return res, claim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(_AI, "8.26", "train_tse.toml"))
    ap.add_argument("--lambda-ns", type=float, required=True)
    ap.add_argument("--lambda-spk", type=float, default=0.3)
    ap.add_argument("--lambda-ce", type=float, default=None,
                    help="L_ce 权重覆盖（None=用 config）")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--exp-tag", default=None)
    ap.add_argument("--base-ckpt", required=True,
                    help="续训基座（r5 交付 ckpt）")
    ap.add_argument("--dur-aug", action="store_true")
    ap.add_argument("--dur-seg", type=int, default=200)
    ap.add_argument("--seed", type=int, default=25)
    ap.add_argument("--imp-prob", type=float, default=0.0,
                    help="冒认条件样本比例 p")
    ap.add_argument("--lam-imp", type=float, default=0.0,
                    help="反拉项权重（0=关闭）")
    ap.add_argument("--tau-imp", type=float, default=0.10,
                    help="反拉项余弦裕度")
    ap.add_argument("--eval-only", action="store_true",
                    help="只跑双口径验证（测基座），不训练")
    args = ap.parse_args()

    tag = args.exp_tag or f"asym{int(round(args.imp_prob * 100)):02d}"
    cfg = toml.load(args.config)
    cfg["paths"]["fullsubnet_ckpt"] = args.base_ckpt
    cfg["paths"]["exp_dir"] = os.path.join(_HERE, "exp", f"tse_v5_{tag}")
    if args.max_train:
        cfg["dataset"]["max_train_samples"] = args.max_train
    exp_dir = cfg["paths"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda:0"
    base = _HERE
    sv_ecapa = os.path.abspath(cfg["paths"]["sv_ecapa"])
    fullsubnet_root = os.path.abspath(cfg["paths"]["fullsubnet_root"])
    _setup(fullsubnet_root, sv_ecapa)
    setup_speakerlab(sv_ecapa, base)

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

    margin_loss = build_aam_margin(32.0, 0.2).to(device)
    diff_fbank = DiffFBank(n_mels=80, sample_rate=sr, n_fft=512,
                           win_length=400, hop_length=160, mean_nor=True,
                           ).to(device)

    emb_by_spk = spk_embs_from_enrol(device)
    trials = load_hard_trials()

    if args.eval_only:
        base_tag = os.path.basename(
            os.path.dirname(os.path.abspath(args.base_ckpt)))
        res, claim = validate_dual(fullsubnet, verifier, diff_fbank, emb_by_spk,
                                   stft_fn, istft_fn, decompress_cIRM,
                                   cfg["acoustics"], device, f"base_{base_tag}",
                                   trials)
        avg = (res["noise_0db"] + res["noise_m5db"]) / 2
        ceer = (claim["noise_0db"]["eer"] + claim["noise_m5db"]["eer"]) / 2
        out = {"base": base_tag, "oracle_hard": res, "oracle_avg": avg,
               "claim": claim, "claim_eer_avg": ceer}
        with open(os.path.join(exp_dir, "claimval_base.json"), "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[eval-only] base={base_tag} oracle_avg={avg:.3f} "
              f"claim_eer_avg={ceer:.3f} "
              f"nt_med={claim['noise_0db']['nt_med']:.3f}/"
              f"{claim['noise_m5db']['nt_med']:.3f} -> {exp_dir}/claimval_base.json",
              flush=True)
        return

    trainable = [p for p in fullsubnet.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=cfg["training"]["lr"])
    lam_ce = (args.lambda_ce if args.lambda_ce is not None
              else float(cfg["training"]["lambda_ce"]))
    lam_spk = args.lambda_spk
    lam_ns = args.lambda_ns
    lam_imp = args.lam_imp
    n_fft, hop, win = (cfg["acoustics"][k]
                       for k in ("n_fft", "hop_length", "win_length"))
    dur_rng = random.Random(args.seed)
    dur_plan = []

    ds = TSEMixDatasetAsym(clean_paths, labels, emb_table, noise_train,
                           cfg["dataset"]["snr_range"], target_len, sr,
                           imp_prob=args.imp_prob)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg["dataset"]["batch_size"], shuffle=True,
        num_workers=cfg["dataset"]["num_workers"], drop_last=True,
        pin_memory=True)
    print(f"[train] {len(ds)} 样本 {len(loader)} it/ep × {args.epochs} ep, "
          f"lambda_ce={lam_ce}, lambda_spk={lam_spk}, lambda_ns={lam_ns}, "
          f"imp_prob={args.imp_prob}, lam_imp={lam_imp}, tau_imp={args.tau_imp}, "
          f"dur_aug={'ON seg=%d it' % args.dur_seg if args.dur_aug else 'OFF'} "
          f"-> {exp_dir}", flush=True)

    best = 1e9
    best_claim = None
    for ep in range(1, args.epochs + 1):
        fullsubnet.train()
        sums = {"L": 0., "L_se": 0., "L_ce": 0., "L_spk": 0., "L_ns": 0.,
                "L_imp": 0., "cos": 0., "ns": 0., "ccond": 0., "nimp": 0., "n": 0}
        for it, (noisy, clean, lb, emb, impf) in enumerate(loader):
            if args.dur_aug and it % args.dur_seg == 0:
                dur = dur_rng.choices(DUR_CHOICES, weights=DUR_PROBS, k=1)[0]
                ds.target_len = int(dur * sr)
                dur_plan.append((ep, it, dur))
                print(f"  [dur] ep{ep} it{it} -> {dur}s 段", flush=True)
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            lb = lb.to(device, non_blocking=True)
            emb = emb.to(device, non_blocking=True)
            impf = impf.to(device, non_blocking=True)
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
            # L_spk 锚 = 真身 clean 嵌入（冒认样本也锚真身 → 教"不向条件变声"）
            L_spk = (1 - F.cosine_similarity(e_clean, e_enh, dim=1)).mean()
            cos_np = F.linear(F.normalize(e_enh, dim=1), proto_t)
            cos_np.scatter_(1, lb.unsqueeze(1), -2.0)
            hard_cos = cos_np.topk(TOPK, dim=1).values
            L_ns = F.relu(hard_cos - TAU).mean()
            # 反拉项：仅冒认样本，罚 e_enh 过度靠近条件（声称者）嵌入
            cos_cond = F.cosine_similarity(e_enh, emb, dim=1)
            n_imp = impf.sum()
            L_imp = ((impf * F.relu(cos_cond - args.tau_imp)).sum()
                     / (n_imp + 1e-6)) if lam_imp > 0 else torch.zeros((), device=device)
            L = L_se + lam_ce * L_ce + lam_spk * L_spk + lam_ns * L_ns + lam_imp * L_imp
            L.backward()
            torch.nn.utils.clip_grad_norm_(fullsubnet.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                cos = F.cosine_similarity(e_clean, e_enh, dim=1).mean()
            sums["L"] += float(L); sums["L_se"] += float(L_se)
            sums["L_ce"] += float(L_ce); sums["L_spk"] += float(L_spk)
            sums["L_ns"] += float(L_ns); sums["L_imp"] += float(L_imp)
            sums["cos"] += float(cos)
            sums["ns"] += float(hard_cos.mean())
            sums["ccond"] += float((cos_cond * impf).sum())
            sums["nimp"] += float(n_imp)
            sums["n"] += 1
            if (it + 1) % 200 == 0:
                n = sums["n"]
                ni = max(sums["nimp"], 1e-6)
                print(f"  ep{ep} it {it+1}/{len(loader)} L={sums['L']/n:.4f} "
                      f"se={sums['L_se']/n:.4f} ce={sums['L_ce']/n:.4f} "
                      f"spk={sums['L_spk']/n:.4f} ns={sums['L_ns']/n:.4f} "
                      f"imp={sums['L_imp']/n:.4f} hardcos={sums['ns']/n:.4f} "
                      f"cos={sums['cos']/n:.4f} "
                      f"ccond_imp={sums['ccond']/ni:.4f}({int(sums['nimp'])}个) "
                      f"dur={ds.target_len // sr}s", flush=True)
                sums = {k: 0. for k in sums}
                sums["n"] = 0
        ck = {"model": fullsubnet.state_dict(), "epoch": ep,
              "lambda_ns": lam_ns, "lambda_spk": lam_spk,
              "imp_prob": args.imp_prob, "lam_imp": lam_imp,
              "tau_imp": args.tau_imp, "dur_aug": bool(args.dur_aug)}
        res, claim = validate_dual(fullsubnet, verifier, diff_fbank, emb_by_spk,
                                   stft_fn, istft_fn, decompress_cIRM,
                                   cfg["acoustics"], device, f"{tag}_ep{ep}",
                                   trials)
        avg = (res["noise_0db"] + res["noise_m5db"]) / 2
        ceer = (claim["noise_0db"]["eer"] + claim["noise_m5db"]["eer"]) / 2
        print(f"  [val-hard] ep{ep} 0db={res['noise_0db']:.3f} "
              f"m5db={res['noise_m5db']:.3f} avg={avg:.3f} | "
              f"[val-claim] eer_avg={ceer:.3f} "
              f"(门槛①: 基线 avg+0.3 内；门槛②: eer_avg 显著降)", flush=True)
        ck["proxy_hard"] = res
        ck["proxy_claim"] = claim
        torch.save(ck, os.path.join(exp_dir, f"checkpoint_epoch{ep}.tar"))
        if avg < best:
            best = avg
            best_claim = claim
            torch.save(ck, os.path.join(exp_dir, "checkpoint_best.tar"))
            print(f"  [val-hard] 新最优 avg={avg:.3f} -> checkpoint_best.tar",
                  flush=True)
    with open(os.path.join(exp_dir, "dur_plan.txt"), "w") as f:
        for row in dur_plan:
            f.write(f"ep{row[0]} it{row[1]} {row[2]}s\n")
    print(f"[done] imp_prob={args.imp_prob} lam_imp={lam_imp} "
          f"best_hard_avg={best:.3f} "
          f"best_claim_eer={(best_claim['noise_0db']['eer'] + best_claim['noise_m5db']['eer']) / 2 if best_claim else -1:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
