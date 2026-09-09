#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_consist.py — clean-noisy 一致性正则微调（8.25, NAW-SV 平移）

文献依据:
  - Lim et al. Interspeech 2024 (NAW-SV): clean 前向(no_grad, 当 teacher) 与
    noisy 前向做表征一致性 -> SSL 微调噪声域不失真。平移到 ERes2Net:
    不变量=说话人身份, 一致性比跨模型 KD 更贴任务。
  - Gu et al. Interspeech 2025: 微调+锚定正则优于联合优化 — 我们从 soup3
    (已收敛) 起步, AAM CE 保持判别力 + 一致性项牵引噪声不变性。

设计:
  PairedDataset: 每条 utt 返回同一切片的 (clean_fbank, noisy_fbank, label),
  noisy SNR ∈ {0, -5} 随机; batch = 9 utt -> collate 成 18 items(b18 等量,
  已知 AMP 下显存安全)。anchor = 同批 clean 嵌入 detach。
    loss = AAM_CE(全部 18) + lambda * (1 - cos(emb_noisy, emb_clean.detach()))

用法(GPU2, AMP):
  python train_consist.py --config config_consist.toml --gpu 2 \
      --init-ckpt exp/soup3_abc/checkpoints/checkpoint_soup3.tar \
      --batch-utts 9 --epochs 2 --consist-lambda 1.0
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
for _p in (_V820, _V821):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common_utils import load_wav, random_crop, mix_noise, setup_speakerlab
from train_eres2net_robust import read_train_csv, worker_init_fn
from trainer_eres2net_robust import (
    build_eres2net_large, build_classifier, build_aam_margin, JointModel,
)
from trainer_eres2net_robust_proto import load_prototypes, init_classifier_prototypes
from validate_robust import build_fbank, run_validation


class PairedDataset(Dataset):
    """每 utt 一个样本: 同一切片的 clean + noisy(随机 SNR 0/-5dB)。"""

    def __init__(self, rows, noise_paths, crop_len, snrs=(0.0, -5.0)):
        self.rows = rows
        self.noise_paths = noise_paths
        self.crop_len = crop_len
        self.snrs = snrs
        self.fbank = build_fbank()
        self.worker_seed_base = 2026

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        utt_id, spk, path, label = self.rows[idx]
        wav = load_wav(path)
        wav_c = random_crop(wav, self.crop_len, np.random)   # 同一切片
        noise_path = self.noise_paths[np.random.randint(len(self.noise_paths))]
        noise = load_wav(noise_path)
        if len(noise) > self.crop_len:
            start = np.random.randint(0, len(noise) - self.crop_len)
        else:
            start = 0
        snr = self.snrs[np.random.randint(len(self.snrs))]
        wav_n, _ = mix_noise(wav_c, noise, snr, start)
        fc = self.fbank(torch.from_numpy(wav_c))
        fn = self.fbank(torch.from_numpy(wav_n))
        return fc, fn, label


def paired_collate(batch):
    maxlen = max(max(f0.shape[0], f1.shape[0]) for f0, f1, _ in batch)
    k = len(batch)
    fb = torch.zeros(2 * k, maxlen, batch[0][0].shape[1])   # [2k, T, 80]
    for i, (f0, f1, _) in enumerate(batch):
        fb[2 * i, :f0.shape[0]] = f0                        # 偶数=clean 锚点
        fb[2 * i + 1, :f1.shape[0]] = f1                    # 奇数=noisy
    lb = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return fb, lb.repeat_interleave(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_consist.toml")
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--init-ckpt",
                    default="exp/soup3_abc/checkpoints/checkpoint_soup3.tar")
    ap.add_argument("--batch-utts", type=int, default=9,
                    help="每批 utt 数(前向 items = 2x)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--consist-lambda", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true")
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
    if "classifier" in ck:
        classifier.load_state_dict(ck["classifier"])
    else:
        protos, spk_ids, _used = load_prototypes(cfg["paths"]["proto_npz"])
        init_classifier_prototypes(classifier, protos, spk_ids, spk2id)
        print("[init] ckpt 无 classifier, 回退 proto-init", flush=True)
    print(f"[init] {args.init_ckpt} (epoch={ck.get('epoch', '?')})", flush=True)

    joint = JointModel(model, classifier).to(device)
    margin_loss = build_aam_margin(cfg["model"]["scale"], cfg["model"]["margin"])
    body_params = [p for p in joint.embedding_model.parameters() if p.requires_grad]
    head_params = [p for p in joint.classifier.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": body_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=5e-5)
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    ds = PairedDataset(rows, noise_paths, crop_len=int(3 * 16000))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_utts, shuffle=True, drop_last=True,
        collate_fn=paired_collate, num_workers=args.num_workers,
        pin_memory=True, worker_init_fn=worker_init_fn)
    n_batches = len(loader)
    print(f"[data] {len(ds)} utt(每 utt 2 items), {len(spk2id)} spks, "
          f"lambda={args.consist_lambda}, eff_batch={args.batch_utts * 2 * args.accum}",
          flush=True)

    bench_dir = os.path.join(cfg["paths"]["data_dir_820"], "val_bench")
    val_trials = os.path.join(cfg["paths"]["data_dir_820"], "val_trials.txt")

    gstep = 0
    for ep in range(1, args.epochs + 1):
        joint.train()
        ds.worker_seed_base = cfg["data"]["seed"] + ep * 1000
        sums = {"loss": 0.0, "ce": 0.0, "cons": 0.0, "acc": 0.0, "n": 0}
        opt.zero_grad()
        for it, (fb, lb) in enumerate(loader):
            warm = min(1.0, (gstep + 1) / max(1, args.warmup))
            for g, bl in zip(opt.param_groups, base_lrs):
                g["lr"] = bl * warm
            fb = fb.to(device, non_blocking=True)
            lb = lb.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                emb = joint.embedding_model(fb)          # [2k, 512]
            emb = emb.float()
            cosine = joint.classifier(emb)
            ce = margin_loss(cosine, lb)
            anchor = emb[0::2].detach()                  # clean 锚点(no_grad 语义)
            noisy = emb[1::2]
            cons = (1.0 - torch.nn.functional.cosine_similarity(
                noisy, anchor, dim=1)).mean()
            loss = ce + args.consist_lambda * cons
            scaler.scale(loss / args.accum).backward()
            if (it + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    body_params + head_params, 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                gstep += 1
            sums["loss"] += float(loss)
            sums["ce"] += float(ce)
            sums["cons"] += float(cons)
            sums["acc"] += float((cosine.argmax(1) == lb).float().mean())
            sums["n"] += 1
            if (it + 1) % 600 == 0:
                n = sums["n"]
                print(f"  ep{ep} it {it+1}/{n_batches} loss={sums['loss']/n:.4f} "
                      f"ce={sums['ce']/n:.4f} cons={sums['cons']/n:.4f} "
                      f"acc={sums['acc']/n:.4f} lr={opt.param_groups[0]['lr']:.2e}",
                      flush=True)
                sums = {"loss": 0.0, "ce": 0.0, "cons": 0.0, "acc": 0.0, "n": 0}
        # ---- 每 epoch 官方 bench 验证 ----
        metrics = run_validation(joint.embedding_model, bench_dir, val_trials,
                                 device, batch_size=24)
        avg = (metrics["0db"]["eer"] + metrics["m5db"]["eer"]) / 2
        print(f"  [val] ep{ep} clean={metrics['clean']['eer']:.3f} "
              f"0db={metrics['0db']['eer']:.3f} m5db={metrics['m5db']['eer']:.3f} "
              f"avg01={avg:.3f}", flush=True)
        torch.save({"embedding_model": joint.embedding_model.state_dict(),
                    "classifier": joint.classifier.state_dict(),
                    "epoch": ep, "val_metrics": metrics, "args": vars(args)},
                   os.path.join(exp_dir, "checkpoints", f"checkpoint_ep{ep}.tar"))
    print("[done] 一致性微调结束", flush=True)


if __name__ == "__main__":
    main()
