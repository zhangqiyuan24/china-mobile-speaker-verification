#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_sam_late.py — SAM 末期微调 + EMA（8.25, 文献驱动新轴）

文献依据:
  - Zhou et al. ICLR 2025 (arXiv 2410.10373): 训练末期几个 epoch 的 SAM
    即可获得与全程 SAM 相当的平坦极小值 -> 已收敛权重(soup)上小步 SAM 微调
    等价于"末期 SAM"。
  - Fathan et al. Interspeech 2025: SAM/EMA 系优化器在 SV 上有先例且有效。
  - Model Soups (Wortsman 2022): 平坦谷内的权重更可平均 -> SAM 产物本身
    也是未来汤的优质成员。

设计:
  从 soup3(或任意 ckpt)初始化, 全参数小 lr fp32 SAM 微调; 同时维护 EMA
  (decay 0.999, 连续汤视角), 每 epoch 官方 bench 验证, raw/EMA 双 ckpt 落盘。

用法(GPU1, 无 teacher, fp32):
  python train_sam_late.py --config config_sam.toml --gpu 1 \
      --init-ckpt exp/soup3_abc/checkpoints/checkpoint_soup3.tar \
      --batch-size 18 --epochs 2 --lr 1e-5 --rho 0.05
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_V820 = os.path.join(os.path.dirname(_HERE), "8.20")
_V821 = os.path.join(os.path.dirname(_HERE), "8.21")
for _p in (_V820, _V821):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common_utils import load_wav, random_crop, mix_noise, setup_speakerlab
from train_eres2net_robust import read_train_csv, worker_init_fn, CondBatchSampler
from trainer_eres2net_robust import (
    build_eres2net_large, build_classifier, build_aam_margin, JointModel,
)
from trainer_eres2net_robust_proto import load_prototypes, init_classifier_prototypes
from validate_robust import build_fbank, run_validation


class MultiCondDataset(Dataset):
    """8.20 MultiCondDataset 同款(3 条件, 无额外增强)。"""

    def __init__(self, rows, noise_paths, crop_len, snrs=(0.0, -5.0)):
        self.rows = rows
        self.noise_paths = noise_paths
        self.crop_len = crop_len
        self.snrs = snrs
        self.fbank = build_fbank()
        self.worker_seed_base = 2026

    def __len__(self):
        return 3 * len(self.rows)

    def __getitem__(self, idx):
        base, cond = idx // 3, idx % 3
        utt_id, spk, path, label = self.rows[base]
        wav = load_wav(path)
        wav3 = random_crop(wav, self.crop_len, np.random)
        if cond > 0:
            noise_path = self.noise_paths[np.random.randint(len(self.noise_paths))]
            noise = load_wav(noise_path)
            if len(noise) > self.crop_len:
                start = np.random.randint(0, len(noise) - self.crop_len)
            else:
                start = 0
            wav3, _ = mix_noise(wav3, noise, self.snrs[cond - 1], start)
        return self.fbank(torch.from_numpy(wav3)), label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_sam.toml")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--init-ckpt",
                    default="exp/soup3_abc/checkpoints/checkpoint_soup3.tar")
    ap.add_argument("--batch-size", type=int, default=18)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--rho", type=float, default=0.05,
                    help="SAM 邻域半径(归一化)")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true",
                    help="混合精度(fp32 激活 ~9.5GB 会 OOM, GPU1 有他人 1.2GB 常驻)")
    ap.add_argument("--val-every-it", type=int, default=4500,
                    help="每 N it 做一次中途验证+双 ckpt(0=关); 8h/epoch 太长, "
                         "2.5h 一个信号点")
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
    ck = torch.load(args.init_ckpt, map_location="cpu")
    model.load_state_dict(ck["embedding_model"])
    classifier = build_classifier(cfg["model"]["embedding_size"], len(spk2id))
    # soup 类 ckpt 无 classifier -> proto-init(与汤成员 A 同源做法)
    if "classifier" in ck:
        classifier.load_state_dict(ck["classifier"])
    else:
        protos, spk_ids, _used = load_prototypes(cfg["paths"]["proto_npz"])
        init_classifier_prototypes(classifier, protos, spk_ids, spk2id)
        print("[init] ckpt 无 classifier, 回退 proto-init", flush=True)
    print(f"[init] {args.init_ckpt} (epoch={ck.get('epoch', '?')})", flush=True)

    joint = JointModel(model, classifier).to(device)
    margin_loss = build_aam_margin(cfg["model"]["scale"], cfg["model"]["margin"])
    trainable = [p for p in joint.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=5e-5)
    # SAM+AMP 关键: unscale_ 一个周期只能调一次, 而第一步的扰动计算无需 unscale —
    # rho*(s*g)/||s*g|| = rho*g/||g||, loss scale s 在归一化中约掉 (s>0 且迭代内不变)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # EMA 初始化(只跟 embedding_model — 评测/入汤只用它)
    ema = {k: v.detach().clone().float()
           for k, v in joint.embedding_model.state_dict().items()}

    ds = MultiCondDataset(rows, noise_paths, crop_len=int(3 * 16000))
    sampler = CondBatchSampler(len(rows), args.batch_size, cfg["data"]["seed"])
    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=sampler, num_workers=args.num_workers,
        pin_memory=True, worker_init_fn=worker_init_fn)
    print(f"[data] {len(ds)} 样本, {len(spk2id)} spks, rho={args.rho}, "
          f"ema={args.ema}", flush=True)

    bench_dir = os.path.join(cfg["paths"]["data_dir_820"], "val_bench")
    val_trials = os.path.join(cfg["paths"]["data_dir_820"], "val_trials.txt")
    n_batches = len(sampler)

    def validate_and_save(tag):
        """官方 bench 双验证(raw+EMA)并落盘; 结束还原 raw 权重回训练态。"""
        torch.cuda.empty_cache()      # 训练期碎片块先释放, 给验证腾形状空间
        joint.eval()
        metrics = run_validation(joint.embedding_model, bench_dir, val_trials,
                                 device, batch_size=16)
        avg = (metrics["0db"]["eer"] + metrics["m5db"]["eer"]) / 2
        print(f"  [val-raw:{tag}] clean={metrics['clean']['eer']:.3f} "
              f"0db={metrics['0db']['eer']:.3f} m5db={metrics['m5db']['eer']:.3f} "
              f"avg01={avg:.3f}", flush=True)
        torch.save({"embedding_model": joint.embedding_model.state_dict(),
                    "epoch": f"sam_{tag}", "args": vars(args)},
                   os.path.join(exp_dir, "checkpoints", f"checkpoint_sam_{tag}.tar"))
        bak = {k: v.detach().clone() for k, v in
               joint.embedding_model.state_dict().items()}
        joint.embedding_model.load_state_dict(
            {k: v.to(bak[k].dtype) for k, v in ema.items()})
        m2 = run_validation(joint.embedding_model, bench_dir, val_trials,
                            device, batch_size=16)
        avg2 = (m2["0db"]["eer"] + m2["m5db"]["eer"]) / 2
        print(f"  [val-ema:{tag}] clean={m2['clean']['eer']:.3f} "
              f"0db={m2['0db']['eer']:.3f} m5db={m2['m5db']['eer']:.3f} "
              f"avg01={avg2:.3f}", flush=True)
        torch.save({"embedding_model": {k: v.to(bak[k].dtype)
                                        for k, v in ema.items()},
                    "epoch": f"ema_{tag}", "args": vars(args)},
                   os.path.join(exp_dir, "checkpoints", f"checkpoint_ema_{tag}.tar"))
        joint.embedding_model.load_state_dict(bak)
        joint.train()

    for ep in range(1, args.epochs + 1):
        joint.train()
        sampler.set_epoch(ep)
        ds.worker_seed_base = cfg["data"]["seed"] + ep * 1000
        sums = {"loss": 0.0, "loss2": 0.0, "acc": 0.0, "n": 0}
        n_skip, t0 = 0, time.time()
        for it, (fb, lb) in enumerate(loader):
            fb = fb.to(device, non_blocking=True)
            lb = lb.to(device, non_blocking=True)
            # ---- SAM 第一步: w 处梯度 ----
            with torch.cuda.amp.autocast(enabled=args.amp):
                cosine = joint(fb)
            loss = margin_loss(cosine.float(), lb)
            scaler.scale(loss).backward()
            with torch.no_grad():
                # 梯度含 loss scale, 但归一化扰动与之无关(见 scaler 注释)
                gnorm = torch.sqrt(sum((p.grad ** 2).sum()
                                       for p in trainable if p.grad is not None))
            gnorm_f = float(gnorm)
            if not math.isfinite(gnorm_f):
                # nan/inf 梯度(fp16 溢出): 绝不能拿去扰动权重 —— 首跑 it~0
                # 即因 nan 梯度使权重 += nan 全毁(loss=nan, acc~0)。
                # unscale_ 记录 found_inf -> update() 自动减半 scale; 本步跳过。
                scaler.unscale_(opt)
                opt.zero_grad()
                scaler.update()
                n_skip += 1
                if n_skip <= 20 or n_skip % 100 == 0:
                    print(f"  [skip] it {it+1} gnorm={gnorm_f} "
                          f"(scale 可能仍偏大, update 已减半)", flush=True)
                sums["n"] += 1
                continue
            with torch.no_grad():
                scale = args.rho / (gnorm_f + 1e-12)
                eps = []                      # 扰动备份(restore 需要同一 delta)
                for p in trainable:
                    if p.grad is not None:
                        e = scale * p.grad.clone()
                        p.add_(e)
                        eps.append((p, e))
            opt.zero_grad()
            # ---- SAM 第二步: w_adv 处梯度 ----
            with torch.cuda.amp.autocast(enabled=args.amp):
                cosine2 = joint(fb)
            loss2 = margin_loss(cosine2.float(), lb)
            scaler.scale(loss2).backward()
            scaler.unscale_(opt)              # 唯一一次 unscale(还原真实梯度)
            with torch.no_grad():
                for p, e in eps:
                    p.sub_(e)                 # 还原 w
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            # ---- EMA ----
            with torch.no_grad():
                d = args.ema
                for k, v in joint.embedding_model.state_dict().items():
                    ema[k].mul_(d).add_(v.float(), alpha=1.0 - d)
            for key, val in (("loss", float(loss)), ("loss2", float(loss2)),
                             ("acc", float((cosine.argmax(1) == lb)
                                           .float().mean()))):
                if math.isfinite(val):
                    sums[key] += val
            sums["n"] += 1
            if (it + 1) % 150 == 0:
                n = max(1, sums["n"])
                dt = (time.time() - t0) / 150
                print(f"  ep{ep} it {it+1}/{n_batches} loss={sums['loss']/n:.4f} "
                      f"sam_loss={sums['loss2']/n:.4f} acc={sums['acc']/n:.4f} "
                      f"({dt:.2f}s/it, skip={n_skip})", flush=True)
                sums = {"loss": 0.0, "loss2": 0.0, "acc": 0.0, "n": 0}
                n_skip, t0 = 0, time.time()
            if args.val_every_it and (it + 1) % args.val_every_it == 0:
                validate_and_save(f"ep{ep}it{it+1}")
        validate_and_save(f"ep{ep}")
    print("[done] SAM 微调结束", flush=True)


if __name__ == "__main__":
    main()
