#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_channel_aug_v2.py — 8.21 鲁棒配方 + 信道增强 v2（8.25 汤成员 D 线）

与 8.24 train_channel_aug.py 逐行同构, 仅两处差异:
  1) ChannelAug -> ChannelAugV2(概率/强度上调 + AM/哼声/双RIR)
  2) exp_dir 指向 8.25/exp/channel_aug_v2_90k
其余(初始化=pretrained、proto AAM、冻结、lr 2e-5、8 epochs、eff_batch 96)
与 v1 完全一致 — 归因干净: 差异全部来自扰动配方加强。
动机见 8.25 研究笔记第九节(地板-缝隙分解): 0dB 增益全部来自 clean 地板,
信道鲁棒性是被低估的杠杆; 同初始化保汤的线性模式连通性(Wortsman22 前提)。

用法(GPU1, ~10.7GB 可用):
  cd ~/china_mobile/AI/8.25 && python train_channel_aug_v2.py \
      --config config_channel_aug_v2.toml --gpu 1 --batch-size 48 --accum 2
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_V820 = os.path.join(os.path.dirname(_HERE), "8.20")
_V821 = os.path.join(os.path.dirname(_HERE), "8.21")
_V824 = os.path.join(os.path.dirname(_HERE), "8.24")
for _p in (_V820, _V821, _V824):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common_utils import load_wav, random_crop, mix_noise, setup_speakerlab
from channel_aug_v2 import ChannelAugV2
from train_eres2net_robust import read_train_csv, worker_init_fn, CondBatchSampler
from trainer_eres2net_robust import (
    build_eres2net_large, load_pretrained_embedding, build_classifier,
    build_aam_margin, apply_freeze, freeze_bn_stats, freeze_report, JointModel,
)
from trainer_eres2net_robust_proto import (
    load_prototypes, init_classifier_prototypes,
)
from validate_robust import build_fbank, run_validation


class ChannelAugDatasetV2(Dataset):
    """8.24 ChannelAugDataset 同款(aug 换 V2)。"""

    def __init__(self, rows, noise_paths, crop_len, aug, snrs=(0.0, -5.0)):
        self.rows = rows
        self.noise_paths = noise_paths
        self.crop_len = crop_len
        self.snrs = snrs
        self.aug = aug
        self.fbank = build_fbank()
        self.worker_seed_base = 2026

    def __len__(self):
        return 3 * len(self.rows)

    def __getitem__(self, idx):
        base, cond = idx // 3, idx % 3
        utt_id, spk, path, label = self.rows[base]
        wav = load_wav(path)
        wav3 = random_crop(wav, self.crop_len, np.random)
        wav3 = self.aug(wav3)                       # ← 信道扰动 V2
        if cond > 0:
            noise_path = self.noise_paths[np.random.randint(len(self.noise_paths))]
            noise = load_wav(noise_path)
            if len(noise) > self.crop_len:
                start = np.random.randint(0, len(noise) - self.crop_len)
            else:
                start = 0
            wav3, _ = mix_noise(wav3, noise, self.snrs[cond - 1], start)
        feat = self.fbank(torch.from_numpy(wav3))
        return feat, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_channel_aug_v2.toml")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--val-every", type=int, default=1)
    args = ap.parse_args()

    import toml
    cfg = toml.load(args.config)
    setup_speakerlab(cfg["paths"]["sv_ecapa"],
                     os.path.dirname(os.path.abspath(args.config)))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda"
    torch.manual_seed(cfg["data"]["seed"])

    exp_dir = cfg["paths"]["exp_dir"]
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)

    rows, spk2id = read_train_csv(
        os.path.join(cfg["paths"]["data_dir_820"], "train_utts.csv"))
    noise_paths = [os.path.join(cfg["paths"]["sv_ecapa"], ln.strip())
                   for ln in open(os.path.join(cfg["paths"]["data_dir_820"],
                                               "noise_train.txt")) if ln.strip()]

    model = build_eres2net_large()
    load_pretrained_embedding(model, cfg["paths"]["eres2net_ckpt"])
    classifier = build_classifier(cfg["model"]["embedding_size"], len(spk2id))
    protos, spk_ids, _used = load_prototypes(cfg["paths"]["proto_npz"])
    init_classifier_prototypes(classifier, protos, spk_ids, spk2id)
    frozen_mods, trainable_mods = apply_freeze(model)
    freeze_report(model, classifier)
    joint = JointModel(model, classifier).to(device)
    margin_loss = build_aam_margin(cfg["model"]["scale"], cfg["model"]["margin"])

    aug = ChannelAugV2(cfg["paths"]["rir_list"])
    ds = ChannelAugDatasetV2(rows, noise_paths, crop_len=int(3 * 16000), aug=aug)
    print(f"[data] {len(ds)} 样本(3 条件), {len(spk2id)} spks, "
          f"noise {len(noise_paths)} 条, RIR 清单 {len(aug.rir_paths)} 条", flush=True)
    print("[aug-v2] p: " + ", ".join(f"{k}={v}" for k, v in aug.p.items()), flush=True)

    sampler = CondBatchSampler(len(rows), args.batch_size, cfg["data"]["seed"])
    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=sampler, num_workers=args.num_workers,
        pin_memory=True, worker_init_fn=worker_init_fn)

    params = [p for p in joint.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=5e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    bench_dir = os.path.join(cfg["paths"]["data_dir_820"], "val_bench")
    val_trials = os.path.join(cfg["paths"]["data_dir_820"], "val_trials.txt")
    n_batches = len(sampler)
    print(f"[train] {n_batches} micro-batches/epoch x {args.epochs} epochs, "
          f"eff_batch={args.batch_size * args.accum}", flush=True)

    gstep, best = 0, 1e9
    for ep in range(1, args.epochs + 1):
        joint.train()
        freeze_bn_stats(model, frozen_mods)
        sampler.set_epoch(ep)
        ds.worker_seed_base = cfg["data"]["seed"] + ep * 1000
        sums = {"loss": 0.0, "acc": 0.0, "n": 0}
        opt.zero_grad()
        for it, (fb, lb) in enumerate(loader):
            fb = fb.to(device, non_blocking=True)
            lb = lb.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                cosine = joint(fb)
            cosine = cosine.float()
            loss = margin_loss(cosine, lb)
            logits = cosine
            scaler.scale(loss / args.accum).backward()
            if (it + 1) % args.accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                gstep += 1
            sums["loss"] += float(loss)
            sums["acc"] += float((logits.argmax(1) == lb).float().mean())
            sums["n"] += 1
            if (it + 1) % (50 * args.accum) == 0:
                n = sums["n"]
                print(f"  ep{ep} it {it+1}/{n_batches} loss={sums['loss']/n:.4f} "
                      f"acc={sums['acc']/n:.4f} ({gstep})", flush=True)
                sums = {"loss": 0.0, "acc": 0.0, "n": 0}
        if ep % args.val_every == 0 or ep == args.epochs:
            metrics = run_validation(joint.embedding_model, bench_dir, val_trials,
                                     device, batch_size=24)
            avg = (metrics["0db"]["eer"] + metrics["m5db"]["eer"]) / 2
            print(f"  [val] ep{ep} clean={metrics['clean']['eer']:.3f} "
                  f"0db={metrics['0db']['eer']:.3f} m5db={metrics['m5db']['eer']:.3f} "
                  f"avg01={avg:.3f}", flush=True)
            ck = {"embedding_model": joint.embedding_model.state_dict(),
                  "classifier": joint.classifier.state_dict(),
                  "epoch": ep, "val_metrics": metrics, "args": vars(args)}
            torch.save(ck, os.path.join(exp_dir, "checkpoints", "checkpoint_last.tar"))
            torch.save(ck, os.path.join(exp_dir, "checkpoints", f"checkpoint_ep{ep}.tar"))
            if avg < best:
                best = avg
                torch.save(ck, os.path.join(exp_dir, "checkpoints",
                                            "checkpoint_best.tar"))
                print(f"  [val] 新最优 avg={avg:.3f} -> checkpoint_best.tar", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
