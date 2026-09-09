#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_spk_voiceid.py — VoiceID 式前端二轮微调（8.25, 文献驱动）

文献依据:
  Shon et al. Interspeech 2019 (VoiceID Loss, MIT): 用冻结 SV 模型的分类损失
  端到端训练增强网络, 比重构损失(L2/cIRM MSE)+一致性更贴下游任务 —
  "PESQ 提升不等于 EER 提升, 增强应为验证定制"。

相对 8.18 spk_preserve 的升级:
  1. 冻结验证器 = soup3(当前最优模型, 8.18 用的原始弱 ckpt)
  2. 损失 = L_se(cIRM MSE, 防退化锚) + lambda_ce * AAM-CE(enhanced 嵌入, 说话人标签)
     —— CE 的类判别梯度比余弦保持强得多(VoiceID 论文核心论点)
  3. 标签来源: 8.20 train_utts.csv 的 spk2id(与 proto 分类器 9900 类对齐)

复用 8.18 全部基础设施(load_fullsubnet/MixDataset/snr_mix/DiffFBank), 只换损失核。
评测路径不变: 训完用 8.18 工具降噪 eval 集 -> evaluate_combo FE 模式。

用法(单卡):
  python train_spk_voiceid.py --config train_spk_voiceid.toml --gpu 2
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import toml

_V818 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "8.18")
_V820 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "8.20")
_V821 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "8.21")
for _p in (_V818, _V820, _V821):
    if osp := _p:
        if os.path.isdir(osp) and osp not in sys.path:
            sys.path.insert(0, osp)

from train_spk_preserve import (          # 8.18 基础设施
    load_noise_paths, uniform_select, _setup,
)
from trainer_spk_preserve import load_fullsubnet, load_eres2net, snr_mix
from frontend_spk import DiffFBank
from common_utils import setup_speakerlab          # 8.20
from trainer_eres2net_robust import build_aam_margin
from trainer_eres2net_robust_proto import (
    load_prototypes, init_classifier_prototypes,
)


class LabeledMixDataset(torch.utils.data.Dataset):
    """8.18 MixDataset + 说话人标签(utt_id -> spk2id, 与 proto 分类器对齐)。"""

    def __init__(self, clean_paths, labels, noise_paths, snr_range, target_len, sr=16000):
        self.clean_paths = clean_paths
        self.labels = labels
        self.noise_paths = noise_paths
        self.snr_range = snr_range
        self.target_len = target_len
        self.sr = sr

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean, _ = torchaudio.load(self.clean_paths[idx])
        clean = clean[0].numpy()
        noise_path = random.choice(self.noise_paths)
        noise, _ = torchaudio.load(noise_path)
        noise = noise[0].numpy()
        snr = random.choice(self.snr_range)
        noisy, clean = snr_mix(clean, noise, snr, self.target_len)
        return (torch.from_numpy(noisy), torch.from_numpy(clean),
                self.labels[idx])


def build_paths_from_csv(train_csv):
    """train_utts.csv(8.20 的 90k 标签子集)自带绝对 wav 路径+说话人列,
    直接作数据源, 免去 scp<->csv 对齐。返回 (paths, labels, spk2id)。"""
    import csv as _csv
    paths, labels, spks = [], [], set()
    with open(train_csv, newline="", encoding="utf-8") as f:
        for row in _csv.reader(f):
            if not row or row[0] == "utt_id":
                continue
            paths.append(row[2])
            labels.append(row[1])
            spks.add(row[1])
    spk2id = {s: i for i, s in enumerate(sorted(spks))}
    labels = [spk2id[s] for s in labels]
    print(f"[labels] {len(paths)} utts / {len(spk2id)} spks (来自 train_utts.csv)",
          flush=True)
    return paths, labels, spk2id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train_spk_voiceid.toml")
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--epochs", type=int, default=None,
                    help="覆盖 config 中的 epochs")
    ap.add_argument("--lambda-ce", type=float, default=None)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--cpu", action="store_true", help="CPU 冒烟测试(慢, 只验通路)")
    args = ap.parse_args()

    cfg = toml.load(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.lambda_ce is not None:
        cfg["training"]["lambda_ce"] = args.lambda_ce
    if args.max_train is not None:
        cfg["dataset"]["max_train_samples"] = args.max_train
    if args.batch_size is not None:
        cfg["dataset"]["batch_size"] = args.batch_size

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = "cuda:0"
    base = os.path.dirname(os.path.abspath(args.config))
    sv_ecapa = os.path.abspath(os.path.join(base, cfg["paths"]["sv_ecapa"]))
    fullsubnet_root = os.path.abspath(os.path.join(base, cfg["paths"]["fullsubnet_root"]))
    _setup(fullsubnet_root, sv_ecapa)
    setup_speakerlab(sv_ecapa, base)

    # ---- 数据: 直接用 8.20 的 90k 标签子集(csv 自带绝对路径) ----
    clean_paths, labels, spk2id = build_paths_from_csv(
        os.path.join(cfg["paths"]["data_dir_820"], "train_utts.csv"))
    assert labels, "train_utts.csv 为空"
    noise_train = load_noise_paths(cfg["dataset"]["noise_list"], sv_ecapa)
    noise_test = load_noise_paths(cfg["dataset"]["noise_test_list"], sv_ecapa)
    assert not (set(noise_train) & set(noise_test)), "train/test noise 重叠!"
    max_train = int(cfg["dataset"].get("max_train_samples", 0))
    if 0 < max_train < len(clean_paths):
        sel = uniform_select(list(range(len(clean_paths))), max_train)
        clean_paths = [clean_paths[i] for i in sel]
        labels = [labels[i] for i in sel]
        print(f"[labels] 均匀抽样 -> {len(clean_paths)} 条", flush=True)

    sr = cfg["acoustics"]["sr"]
    target_len = int(cfg["dataset"]["sub_sample_length"] * sr)
    ds = LabeledMixDataset(clean_paths, labels, noise_train,
                           cfg["dataset"]["snr_range"], target_len, sr)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg["dataset"]["batch_size"], shuffle=True,
        num_workers=cfg["dataset"]["num_workers"], drop_last=True,
        pin_memory=True)

    # ---- 模型: FullSubNet+(可训练) + 冻结 soup3 + proto 分类器 ----
    fullsubnet, stft_fn, istft_fn, decompress_cIRM, build_cIRM = load_fullsubnet(
        fullsubnet_root, cfg["paths"]["fullsubnet_ckpt"], cfg["model"], device)
    print(f"[frontend] init from {cfg['paths']['fullsubnet_ckpt']}", flush=True)

    verifier = load_eres2net(sv_ecapa, cfg["paths"]["verifier_ckpt"], device)
    print(f"[verifier] frozen soup3 = {cfg['paths']['verifier_ckpt']}", flush=True)

    from speakerlab.models.eres2net.ERes2Net import ERes2Net  # noqa: F401
    from trainer_eres2net_robust import build_classifier
    classifier = build_classifier(512, len(spk2id))   # CPU 上装 prototype(避免跨设备写)
    protos, spk_ids, _used = load_prototypes(cfg["paths"]["proto_npz"])
    init_classifier_prototypes(classifier, protos, spk_ids, spk2id)
    classifier = classifier.to(device)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    margin_loss = build_aam_margin(32.0, 0.2).to(device)
    diff_fbank = DiffFBank(
        n_mels=80, sample_rate=sr,
        n_fft=512, win_length=400, hop_length=160, mean_nor=True,
    ).to(device)

    trainable = [p for p in fullsubnet.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=cfg["training"]["lr"])

    lam_ce = float(cfg["training"]["lambda_ce"])
    lam_spk = float(cfg["training"].get("lambda_spk", 0.0))
    n_fft, hop, win = (cfg["acoustics"][k] for k in ("n_fft", "hop_length", "win_length"))
    exp_dir = cfg["paths"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    epochs = int(cfg["training"]["epochs"])
    print(f"[train] {len(ds)} 样本, {len(loader)} steps/epoch x {epochs} epochs, "
          f"lambda_ce={lam_ce}, lambda_spk={lam_spk}", flush=True)

    def enhance(noisy):
        """8.18 forward_enhanced 同款(裁剪为需要的三元返回)。"""
        noisy_complex = stft_fn(noisy, n_fft, hop, win)
        cRM_raw = fullsubnet(noisy_complex.abs().unsqueeze(1),
                             noisy_complex.real.unsqueeze(1),
                             noisy_complex.imag.unsqueeze(1))
        cRM_raw = cRM_raw.permute(0, 2, 3, 1)
        cRM_dec = decompress_cIRM(cRM_raw)
        er = cRM_dec[..., 0] * noisy_complex.real - cRM_dec[..., 1] * noisy_complex.imag
        ei = cRM_dec[..., 1] * noisy_complex.real + cRM_dec[..., 0] * noisy_complex.imag
        enhanced = istft_fn(torch.stack([er, ei], dim=-1), n_fft, hop, win,
                            length=noisy.size(-1))
        return enhanced, noisy_complex, cRM_raw

    for ep in range(1, epochs + 1):
        fullsubnet.train()
        sums = {"L": 0.0, "L_se": 0.0, "L_ce": 0.0, "L_spk": 0.0, "acc": 0.0, "cos": 0.0, "n": 0}
        for it, (noisy, clean, lb) in enumerate(loader):
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            lb = lb.to(device, non_blocking=True)
            opt.zero_grad()
            enhanced, noisy_complex, cRM_raw = enhance(noisy)
            clean_complex = stft_fn(clean, n_fft, hop, win)
            L_se = F.mse_loss(cRM_raw, build_cIRM(noisy_complex, clean_complex))
            # ---- VoiceID 核: 冻结验证器的类判别损失 ----
            e_enh = verifier(diff_fbank(enhanced))
            cosine = classifier(e_enh)
            L_ce = margin_loss(cosine.float(), lb)
            L = L_se + lam_ce * L_ce
            if lam_spk > 0:
                with torch.no_grad():
                    e_clean = verifier(diff_fbank(clean))
                L_spk = (1 - F.cosine_similarity(e_clean, e_enh, dim=1)).mean()
                L = L + lam_spk * L_spk
            else:
                L_spk = torch.zeros(())
            L.backward()
            torch.nn.utils.clip_grad_norm_(fullsubnet.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                acc = (cosine.argmax(1) == lb).float().mean()
                cosm = F.cosine_similarity(
                    verifier(diff_fbank(clean)), e_enh, dim=1).mean()
            sums["L"] += float(L); sums["L_se"] += float(L_se)
            sums["L_ce"] += float(L_ce); sums["L_spk"] += float(L_spk)
            sums["acc"] += float(acc); sums["cos"] += float(cosm)
            sums["n"] += 1
            if (it + 1) % 200 == 0:
                n = sums["n"]
                print(f"  ep{ep} it {it+1}/{len(loader)} L={sums['L']/n:.4f} "
                      f"se={sums['L_se']/n:.4f} ce={sums['L_ce']/n:.4f} "
                      f"acc={sums['acc']/n:.4f} cos={sums['cos']/n:.4f}", flush=True)
                sums = {"L": 0.0, "L_se": 0.0, "L_ce": 0.0, "L_spk": 0.0,
                        "acc": 0.0, "cos": 0.0, "n": 0}
        torch.save({"model": fullsubnet.state_dict()},
                   os.path.join(exp_dir, f"checkpoint_epoch{ep}.tar"))
        print(f"[ep{ep}] saved -> {exp_dir}/checkpoint_epoch{ep}.tar", flush=True)
    print("[done] VoiceID 式前端微调结束", flush=True)


if __name__ == "__main__":
    main()
