#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_tse_v7_rank.py — 冒认防御第三格：v6 混叠/非对称 + 排序铰链 L_rank（#38，9.1）

新增 L_rank（仅冒认样本）：要求 cos(e_enh, 真身) ≥ cos(e_enh, 条件) + margin。
与 EER 排序判据同构（target 分必须排在 nontarget 分之上）——这是第一个直接
优化 claim 排序本身的损失项；此前五个损失全部优化代理目标（重建/分类/绝对
阈值），claim 的排序从未被直接训练过（proxy claim ep1 定型的 suspected 根因）。
关键性质：
  ① 铰链饱和：排序满足即不再用力（L_imp 推绝对阈值 τ 永续施压伤 oracle）；
  ② 非冒认样本被 impf 掩蔽、梯度恒零 → oracle 构造性无损；
  ③ 锚 = e_clean（detach），梯度只走 e_enh；cos_cond 与 v6 逐位同源。

v6 机制（A+B 混叠 + 非对称条件）全文保留：
用户方案（8.31 晚）：训练中也混入 A+B（注册=A，约束提取 A）——条件被迫从
"生成目标"变"检索钥匙"：输出必须从输入已有成分拼出，A 不在场就没有原料
可变声，变声通路被堵。文献对位（AI/8.31/lit_notes_0831.md）：VoiceFilter/
SpeakerBeam/SpEx+/USEF 系全部混叠训练；USEV 的 QS 场景（目标缺席+干扰人在
说）即 claim 冒认的训练形态，由 v5 的 imp 样本承担——两机制互补：
  mix 样本（mix_prob）：A+B+noise，条件=A（匹配），锚=A —— 教"从混叠中
      检索条件指定的声音"；
  imp 样本（imp_prob，v5 机制保留）：A+noise，条件=随机他人（不匹配），
      锚=A —— 教"条件不在场时忠实增强在场者，不向条件变声"。
两币独立掷（可同时命中：A+B 混叠+错位条件，锚仍=真身，语义一致）。
三损失锚/L_ns/反拉项与 v5 完全一致，唯一新自由度 = --mix-prob 与 SCR 范围。

用法: python train_tse_v7_rank.py --imp-prob 0.5 --mix-prob 0.4 --lambda-ns 0.5 \
        --lam-rank 0.3 --rank-margin 0.10 \
        --base-ckpt .../tse_v6_mix40i50/checkpoint_epoch1.tar \
        --exp-tag rk03_mix40i50 --epochs 3 --gpu 1 --dur-aug
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import toml
import torchaudio

_HERE = os.path.dirname(os.path.abspath(__file__))
from train_tse_v5_asym import (TSEMixDatasetAsym, load_hard_trials,
                               _embed_dir, _enhance_map, _cos)
from train_tse import build_paths_from_csv, build_tse_fullsubnet
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
SCR_CHOICES = (-5, 0, 5, 10)  # 真身:竞争者功率比 dB，负值=干扰更响


class TSEMixDatasetV6(TSEMixDatasetAsym):
    """v5 非对称 + 以 mix_prob 概率叠加随机他人 clean（A+B 混叠，锚仍=真身）。

    追加第 6 位 mix 标志；混叠只改输入 noisy，clean 目标/标签/条件嵌入全部
    沿用 v5 逻辑（imp 币已在 super() 内掷过）。
    """

    def __init__(self, *a, mix_prob=0.0, mix_seed=5678, **kw):
        super().__init__(*a, **kw)
        self.mix_prob = float(mix_prob)
        self.mix_rng = random.Random(mix_seed + os.getpid())
        self.scr_choices = SCR_CHOICES

    def __getitem__(self, idx):
        noisy, clean, lb, emb, impf = super().__getitem__(idx)
        if self.mix_prob > 0 and self.mix_rng.random() < self.mix_prob:
            n = len(self.labels)
            j = self.mix_rng.randrange(n)
            while self.labels[j] == self.labels[idx]:
                j = (j + 1) % n
            b, _ = torchaudio.load(self.clean_paths[j])
            b = b[0].numpy()
            L = noisy.shape[-1]
            if len(b) > L:
                off = self.mix_rng.randrange(len(b) - L + 1)
                b = b[off:off + L]
            else:
                b = np.pad(b, (0, L - len(b)))
            p_a = float(np.mean(clean.numpy() ** 2))
            p_b = float(np.mean(b ** 2))
            if p_a < 1e-8 or p_b < 1e-8:  # 静音句防炸（scale 发散）
                return noisy, clean, lb, emb, impf, torch.tensor(0.0)
            scr = self.mix_rng.choice(self.scr_choices)
            scale = float(np.sqrt(p_a / p_b * 10 ** (-scr / 10.0)))
            noisy = noisy + torch.from_numpy(
                (b * scale).astype(np.float32))
            return noisy, clean, lb, emb, impf, torch.tensor(1.0)
        return noisy, clean, lb, emb, impf, torch.tensor(0.0)


def validate_dual(model, verifier, fbank, emb_by_spk, stft_fn, istft_fn,
                  decompress_cIRM, acoustics, device, tag, trials):
    """与 v5 validate_dual 相同，仅 val 输出目录前缀改 val_v6_（不覆写 v5 产物）。"""
    model.eval()
    n_fft, hop, win = (acoustics[k] for k in ("n_fft", "hop_length", "win_length"))
    spks = sorted(emb_by_spk)
    wrong = {s: spks[(i + 1) % len(spks)] for i, s in enumerate(spks)}
    res, claim = {}, {}
    for cond in ("noise_0db", "noise_m5db"):
        out_or = os.path.join(PROXY, f"val_v7_{tag}", cond)
        out_cl = os.path.join(PROXY, f"val_v7_{tag}_claim", cond)
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
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(_HERE), "8.26", "train_tse.toml"))
    ap.add_argument("--lambda-ns", type=float, required=True)
    ap.add_argument("--lambda-spk", type=float, default=0.3)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--exp-tag", default=None)
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--dur-aug", action="store_true")
    ap.add_argument("--dur-seg", type=int, default=200)
    ap.add_argument("--seed", type=int, default=25)
    ap.add_argument("--imp-prob", type=float, default=0.0)
    ap.add_argument("--mix-prob", type=float, default=0.0,
                    help="A+B 混叠样本比例")
    ap.add_argument("--lam-imp", type=float, default=0.0)
    ap.add_argument("--tau-imp", type=float, default=0.10)
    ap.add_argument("--lam-rank", type=float, default=0.0,
                    help="排序铰链权重（0=v6 行为）")
    ap.add_argument("--rank-margin", type=float, default=0.10)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    tag = args.exp_tag or (f"mix{int(round(args.mix_prob * 100)):02d}"
                           f"i{int(round(args.imp_prob * 100)):02d}")
    cfg = toml.load(args.config)
    cfg["paths"]["fullsubnet_ckpt"] = args.base_ckpt
    cfg["paths"]["exp_dir"] = os.path.join(_HERE, "exp", f"tse_v7_{tag}")
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
              f"claim_eer_avg={ceer:.3f} -> {exp_dir}/claimval_base.json",
              flush=True)
        return

    trainable = [p for p in fullsubnet.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=cfg["training"]["lr"])
    lam_ce = float(cfg["training"]["lambda_ce"])
    lam_spk = args.lambda_spk
    lam_ns = args.lambda_ns
    lam_imp = args.lam_imp
    lam_rank = args.lam_rank
    n_fft, hop, win = (cfg["acoustics"][k]
                       for k in ("n_fft", "hop_length", "win_length"))
    dur_rng = random.Random(args.seed)
    dur_plan = []

    ds = TSEMixDatasetV6(clean_paths, labels, emb_table, noise_train,
                         cfg["dataset"]["snr_range"], target_len, sr,
                         imp_prob=args.imp_prob, mix_prob=args.mix_prob)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg["dataset"]["batch_size"], shuffle=True,
        num_workers=cfg["dataset"]["num_workers"], drop_last=True,
        pin_memory=True)
    print(f"[train] {len(ds)} 样本 {len(loader)} it/ep × {args.epochs} ep, "
          f"lambda_ce={lam_ce}, lambda_spk={lam_spk}, lambda_ns={lam_ns}, "
          f"imp_prob={args.imp_prob}, mix_prob={args.mix_prob}, "
          f"scr={SCR_CHOICES}, lam_imp={lam_imp}, lam_rank={lam_rank}, "
          f"margin={args.rank_margin}, "
          f"dur_aug={'ON seg=%d it' % args.dur_seg if args.dur_aug else 'OFF'} "
          f"-> {exp_dir}", flush=True)

    best = 1e9
    best_claim = None
    for ep in range(1, args.epochs + 1):
        fullsubnet.train()
        sums = {"L": 0., "L_se": 0., "L_ce": 0., "L_spk": 0., "L_ns": 0.,
                "L_imp": 0., "L_rank": 0., "viol": 0., "cos": 0., "ns": 0.,
                "ccond": 0., "nimp": 0., "nmix": 0., "n": 0}
        for it, (noisy, clean, lb, emb, impf, mixf) in enumerate(loader):
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
            # 锚恒=真身：混叠样本教"按条件检索真身"，冒认样本教"不向条件变声"
            L_spk = (1 - F.cosine_similarity(e_clean, e_enh, dim=1)).mean()
            cos_np = F.linear(F.normalize(e_enh, dim=1), proto_t)
            cos_np.scatter_(1, lb.unsqueeze(1), -2.0)
            hard_cos = cos_np.topk(TOPK, dim=1).values
            L_ns = F.relu(hard_cos - TAU).mean()
            cos_cond = F.cosine_similarity(e_enh, emb, dim=1)
            n_imp = impf.sum()
            L_imp = ((impf * F.relu(cos_cond - args.tau_imp)).sum()
                     / (n_imp + 1e-6)) if lam_imp > 0 else torch.zeros((), device=device)
            # 排序铰链（仅冒认样本）：cos(e_enh, 真身) ≥ cos(e_enh, 条件) + m
            # 与 EER 排序判据同构；非冒认样本被 impf 掩蔽 → oracle 构造性无损
            cos_true = F.cosine_similarity(e_clean, e_enh, dim=1)
            viol = (impf * (cos_cond > cos_true).float()).sum()
            L_rank = ((impf * F.relu(args.rank_margin + cos_cond - cos_true)).sum()
                      / (n_imp + 1e-6)) if lam_rank > 0 else torch.zeros((), device=device)
            L = L_se + lam_ce * L_ce + lam_spk * L_spk + lam_ns * L_ns \
                + lam_imp * L_imp + lam_rank * L_rank
            L.backward()
            torch.nn.utils.clip_grad_norm_(fullsubnet.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                cos = F.cosine_similarity(e_clean, e_enh, dim=1).mean()
            sums["L"] += float(L); sums["L_se"] += float(L_se)
            sums["L_ce"] += float(L_ce); sums["L_spk"] += float(L_spk)
            sums["L_ns"] += float(L_ns); sums["L_imp"] += float(L_imp)
            sums["L_rank"] += float(L_rank); sums["viol"] += float(viol)
            sums["cos"] += float(cos)
            sums["ns"] += float(hard_cos.mean())
            sums["ccond"] += float((cos_cond * impf).sum())
            sums["nimp"] += float(n_imp)
            sums["nmix"] += float(mixf.sum())
            sums["n"] += 1
            if (it + 1) % 200 == 0:
                n = sums["n"]
                ni = max(sums["nimp"], 1e-6)
                print(f"  ep{ep} it {it+1}/{len(loader)} L={sums['L']/n:.4f} "
                      f"se={sums['L_se']/n:.4f} ce={sums['L_ce']/n:.4f} "
                      f"spk={sums['L_spk']/n:.4f} ns={sums['L_ns']/n:.4f} "
                      f"imp={sums['L_imp']/n:.4f} rank={sums['L_rank']/n:.4f} "
                      f"hardcos={sums['ns']/n:.4f} "
                      f"cos={sums['cos']/n:.4f} "
                      f"ccond_imp={sums['ccond']/ni:.4f} "
                      f"viol={sums['viol']/ni:.3f} "
                      f"imp={int(sums['nimp'])} mix={int(sums['nmix'])}个 "
                      f"dur={ds.target_len // sr}s", flush=True)
                sums = {k: 0. for k in sums}
                sums["n"] = 0
        ck = {"model": fullsubnet.state_dict(), "epoch": ep,
              "lambda_ns": lam_ns, "lambda_spk": lam_spk,
              "imp_prob": args.imp_prob, "mix_prob": args.mix_prob,
              "lam_imp": lam_imp, "tau_imp": args.tau_imp,
              "lam_rank": lam_rank, "rank_margin": args.rank_margin,
              "dur_aug": bool(args.dur_aug)}
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
    print(f"[done] imp_prob={args.imp_prob} mix_prob={args.mix_prob} "
          f"lam_rank={lam_rank} best_hard_avg={best:.3f} "
          f"best_claim_eer={(best_claim['noise_0db']['eer'] + best_claim['noise_m5db']['eer']) / 2 if best_claim else -1:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
