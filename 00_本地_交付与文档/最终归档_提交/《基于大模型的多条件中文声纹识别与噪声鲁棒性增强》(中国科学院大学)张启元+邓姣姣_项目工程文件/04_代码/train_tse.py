#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_tse.py — 注册条件化 TSE 前端训练（8.26 结构线②）

VoiceFilter 式: FullSubNet+ 的 fullband 三路经 SpeakerFiLM(注册嵌入) 条件化,
端到端训 cIRM MSE + 冻结验证器(soup5) 的 AAM-CE。
与 8.25 voiceid 的本质区别: 网络拿到了"目标是谁"的信息, 才能在单说话人
噪声(干扰语音)里做定向提取 —— 这是 gap 轴上唯一没试过的武器。

条件向量空间: 训练用 spk_embs.npz(soup5 提的每说话人平均嵌入, L2 归一);
推理时 enhance_tse.py 用同一 soup5 从官方 enrol clean 提测试说话人嵌入。

用法: python train_tse.py --config train_tse.toml --gpu 2
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_V818 = os.path.join(os.path.dirname(_HERE), "8.18")
_V820 = os.path.join(os.path.dirname(_HERE), "8.20")
_V821 = os.path.join(os.path.dirname(_HERE), "8.21")
for _p in (_V818, _V820, _V821, _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from train_spk_preserve import load_noise_paths, uniform_select, _setup
from trainer_spk_preserve import load_eres2net, snr_mix
from frontend_spk import DiffFBank
from common_utils import setup_speakerlab
from trainer_eres2net_robust import build_aam_margin
from trainer_eres2net_robust_proto import (
    load_prototypes, init_classifier_prototypes,
)


class TSEMixDataset(torch.utils.data.Dataset):
    """voiceid 的 LabeledMixDataset + 每样本携带说话人条件嵌入。"""

    def __init__(self, clean_paths, labels, emb_table, noise_paths, snr_range,
                 target_len, sr=16000):
        self.clean_paths = clean_paths
        self.labels = labels
        self.emb_table = emb_table            # (N_spk, 512) np.float32
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
                self.labels[idx], torch.from_numpy(self.emb_table[self.labels[idx]]))


def build_paths_from_csv(train_csv):
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
    print(f"[labels] {len(paths)} utts / {len(spk2id)} spks", flush=True)
    return paths, labels, spk2id


def build_tse_fullsubnet(fullsubnet_root, ckpt_path, model_args, device,
                         emb_dim=512, film_hidden=256):
    """8.18 load_fullsubnet 的 TSE 版: 子类 + strict=False 装载(缺键只许 film.*)。"""
    from audio_zen.acoustics.feature import stft, istft
    from audio_zen.acoustics.mask import decompress_cIRM, build_complex_ideal_ratio_mask
    from tse_model import FullSubNetPlusTSE

    TSE = FullSubNetPlusTSE.build()
    model = TSE(
        num_freqs=model_args["num_freqs"],
        look_ahead=model_args["look_ahead"],
        sequence_model=model_args["sequence_model"],
        fb_num_neighbors=model_args["fb_num_neighbors"],
        sb_num_neighbors=model_args["sb_num_neighbors"],
        fb_output_activate_function=model_args["fb_output_activate_function"],
        sb_output_activate_function=model_args["sb_output_activate_function"],
        fb_model_hidden_size=model_args["fb_model_hidden_size"],
        sb_model_hidden_size=model_args["sb_model_hidden_size"],
        channel_attention_model=model_args["channel_attention_model"],
        norm_type=model_args["norm_type"],
        num_groups_in_drop_band=model_args["num_groups_in_drop_band"],
        kersize=model_args["kersize"],
        subband_num=model_args["subband_num"],
        weight_init=False,
        use_drop_band=False,
        emb_dim=emb_dim,
        film_hidden=film_hidden,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"多余键(ckpt 有模型无): {unexpected[:5]}"
    assert all(k.startswith("film.") for k in missing), \
        f"缺键不只是 film.*: {[k for k in missing if not k.startswith('film.')][:5]}"
    print(f"[tse] 8.18 权重装载: 缺 {len(missing)} 键(全部 film.*, 零初始化恒等)",
          flush=True)
    model.to(device)
    return model, stft, istft, decompress_cIRM, build_complex_ideal_ratio_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train_tse.toml")
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lambda-ce", type=float, default=None)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--exp-tag", default=None, help="exp_dir 后缀(冒烟用)")
    args = ap.parse_args()

    cfg = toml.load(os.path.join(_HERE, args.config))
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.lambda_ce is not None:
        cfg["training"]["lambda_ce"] = args.lambda_ce
    if args.max_train is not None:
        cfg["dataset"]["max_train_samples"] = args.max_train
    if args.batch_size is not None:
        cfg["dataset"]["batch_size"] = args.batch_size
    if args.exp_tag:
        cfg["paths"]["exp_dir"] = cfg["paths"]["exp_dir"] + "_" + args.exp_tag

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda:0"
    base = _HERE
    sv_ecapa = os.path.abspath(os.path.join(base, cfg["paths"]["sv_ecapa"]))
    fullsubnet_root = os.path.abspath(cfg["paths"]["fullsubnet_root"])
    _setup(fullsubnet_root, sv_ecapa)
    setup_speakerlab(sv_ecapa, base)

    # ---- 数据 + 条件表 ----
    clean_paths, labels, spk2id = build_paths_from_csv(
        os.path.join(cfg["paths"]["data_dir_820"], "train_utts.csv"))
    assert labels, "train_utts.csv 为空"
    emb_npz = np.load(cfg["paths"]["spk_embs"], allow_pickle=True)
    emb_spk_ids = list(emb_npz["spk_ids"])
    emb_table = emb_npz["embs"]
    # csv 的 spk2id(排序) 与 npz 的 spk_ids(排序) 应同序 —— assert 兜底
    assert list(spk2id.keys()) == emb_spk_ids, \
        "spk_embs.npz 与 train_utts.csv 的说话人集合/顺序不一致, 先重跑 build_spk_embs"
    print(f"[tse] 条件表 {emb_table.shape} 对齐 {len(emb_spk_ids)} spks", flush=True)

    noise_train = load_noise_paths(cfg["dataset"]["noise_list"], sv_ecapa)
    max_train = int(cfg["dataset"].get("max_train_samples", 0))
    if 0 < max_train < len(clean_paths):
        sel = uniform_select(list(range(len(clean_paths))), max_train)
        clean_paths = [clean_paths[i] for i in sel]
        labels = [labels[i] for i in sel]
        print(f"[labels] 均匀抽样 -> {len(clean_paths)} 条", flush=True)

    sr = cfg["acoustics"]["sr"]
    target_len = int(cfg["dataset"]["sub_sample_length"] * sr)
    ds = TSEMixDataset(clean_paths, labels, emb_table, noise_train,
                       cfg["dataset"]["snr_range"], target_len, sr)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg["dataset"]["batch_size"], shuffle=True,
        num_workers=cfg["dataset"]["num_workers"], drop_last=True,
        pin_memory=True)

    # ---- 模型 ----
    fullsubnet, stft_fn, istft_fn, decompress_cIRM, build_cIRM = build_tse_fullsubnet(
        fullsubnet_root, cfg["paths"]["fullsubnet_ckpt"], cfg["model"], device)
    print(f"[frontend] init from {cfg['paths']['fullsubnet_ckpt']}", flush=True)

    verifier = load_eres2net(sv_ecapa, cfg["paths"]["verifier_ckpt"], device)
    print(f"[verifier] frozen = {cfg['paths']['verifier_ckpt']}", flush=True)

    from trainer_eres2net_robust import build_classifier
    classifier = build_classifier(512, len(spk2id))
    protos, proto_spk_ids, _used = load_prototypes(cfg["paths"]["proto_npz"])
    init_classifier_prototypes(classifier, protos, proto_spk_ids, spk2id)
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
    n_film = sum(p.numel() for n, p in fullsubnet.named_parameters() if n.startswith("film."))
    print(f"[tse] 可训练 {sum(p.numel() for p in trainable)/1e6:.2f}M, 其中 FiLM {n_film/1e3:.0f}K",
          flush=True)
    opt = torch.optim.Adam(trainable, lr=cfg["training"]["lr"])

    lam_ce = float(cfg["training"]["lambda_ce"])
    lam_spk = float(cfg["training"].get("lambda_spk", 0.0))
    n_fft, hop, win = (cfg["acoustics"][k] for k in ("n_fft", "hop_length", "win_length"))
    exp_dir = cfg["paths"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    epochs = int(cfg["training"]["epochs"])
    print(f"[train] {len(ds)} 样本, {len(loader)} steps/epoch x {epochs} epochs, "
          f"lambda_ce={lam_ce}, lambda_spk={lam_spk}", flush=True)

    def enhance(noisy, emb):
        noisy_complex = stft_fn(noisy, n_fft, hop, win)
        cRM_raw = fullsubnet(noisy_complex.abs().unsqueeze(1),
                             noisy_complex.real.unsqueeze(1),
                             noisy_complex.imag.unsqueeze(1), emb)
        cRM_raw = cRM_raw.permute(0, 2, 3, 1)
        cRM_dec = decompress_cIRM(cRM_raw)
        er = cRM_dec[..., 0] * noisy_complex.real - cRM_dec[..., 1] * noisy_complex.imag
        ei = cRM_dec[..., 1] * noisy_complex.real + cRM_dec[..., 0] * noisy_complex.imag
        enhanced = istft_fn(torch.stack([er, ei], dim=-1), n_fft, hop, win,
                            length=noisy.size(-1))
        return enhanced, noisy_complex, cRM_raw

    for ep in range(1, epochs + 1):
        fullsubnet.train()
        sums = {"L": 0.0, "L_se": 0.0, "L_ce": 0.0, "acc": 0.0, "cos": 0.0, "n": 0}
        for it, (noisy, clean, lb, emb) in enumerate(loader):
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            lb = lb.to(device, non_blocking=True)
            emb = emb.to(device, non_blocking=True)
            opt.zero_grad()
            enhanced, noisy_complex, cRM_raw = enhance(noisy, emb)
            clean_complex = stft_fn(clean, n_fft, hop, win)
            L_se = F.mse_loss(cRM_raw, build_cIRM(noisy_complex, clean_complex))
            e_enh = verifier(diff_fbank(enhanced))
            cosine = classifier(e_enh)
            L_ce = margin_loss(cosine.float(), lb)
            L = L_se + lam_ce * L_ce
            if lam_spk > 0:
                with torch.no_grad():
                    e_clean = verifier(diff_fbank(clean))
                L = L + lam_spk * (1 - F.cosine_similarity(e_clean, e_enh, dim=1)).mean()
            L.backward()
            torch.nn.utils.clip_grad_norm_(fullsubnet.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                acc = (cosine.argmax(1) == lb).float().mean()
                cosm = F.cosine_similarity(
                    verifier(diff_fbank(clean)), e_enh, dim=1).mean()
            sums["L"] += float(L); sums["L_se"] += float(L_se)
            sums["L_ce"] += float(L_ce)
            sums["acc"] += float(acc); sums["cos"] += float(cosm)
            sums["n"] += 1
            if (it + 1) % 200 == 0:
                n = sums["n"]
                print(f"  ep{ep} it {it+1}/{len(loader)} L={sums['L']/n:.4f} "
                      f"se={sums['L_se']/n:.4f} ce={sums['L_ce']/n:.4f} "
                      f"acc={sums['acc']/n:.4f} cos={sums['cos']/n:.4f}", flush=True)
                sums = {"L": 0.0, "L_se": 0.0, "L_ce": 0.0, "acc": 0.0, "cos": 0.0, "n": 0}
        torch.save({"model": fullsubnet.state_dict(), "epoch": ep},
                   os.path.join(exp_dir, f"checkpoint_epoch{ep}.tar"))
        print(f"[ep{ep}] saved -> {exp_dir}/checkpoint_epoch{ep}.tar", flush=True)
    print("[done] TSE 前端训练结束", flush=True)


if __name__ == "__main__":
    main()
