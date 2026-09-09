#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_moe.py — MoE-lite 训练（8.26 线④, NCMoE 移植）

配方 = 8.21 三条件多条件 + AAM + proto 分类器，骨干换成 MoEERes2Net：
  ERes2Net-large(layer4×K=3 专家) + CondRouter(fbank→3 类温度 softmax)
  L = AAM-CE(混合输出) + 1.0·CE(router_logits, cond)
UMES：expert 全部从 soup5(当前最优)的 layer4 复制，其余骨干同 soup5。

用法: python train_moe.py --config config_moe.toml --gpu 3
"""
import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_V818 = os.path.join(os.path.dirname(_HERE), "8.18")
_V820 = os.path.join(os.path.dirname(_HERE), "8.20")
_V821 = os.path.join(os.path.dirname(_HERE), "8.21")
for _p in (_V818, _V820, _V821, _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from common_utils import load_wav, random_crop, mix_noise, setup_speakerlab
from train_eres2net_robust import read_train_csv, worker_init_fn, CondBatchSampler
from trainer_eres2net_robust import build_classifier, build_aam_margin, JointModel
from trainer_eres2net_robust_proto import (
    load_prototypes, init_classifier_prototypes,
)
from validate_robust import build_fbank, run_validation
from moe_model import MoEERes2Net


class MultiCondDataset(torch.utils.data.Dataset):
    """8.21 三条件数据集 + 返回条件标签（路由器监督用）。"""

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
        feat = self.fbank(torch.from_numpy(wav3))
        return feat, label, cond


class MoEJointModel(torch.nn.Module):
    """JointModel 变体：透传条件标签给 MoE 前向。"""

    def __init__(self, moe, classifier):
        super().__init__()
        self.embedding_model = moe
        self.classifier = classifier

    def forward(self, feat, cond=None):
        if cond is None:
            return self.classifier(self.embedding_model(feat))
        emb, logits = self.embedding_model(feat, return_router=True)
        return self.classifier(emb), logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_moe.toml")
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--accum", type=int, default=4)   # eff 96 ≈ 8.21 的 90
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--amp", action="store_true",
                    help="fp16 混合精度(3xexpert 激活大, 12GB 卡必开)")
    ap.add_argument("--exp-tag", default=None)
    args = ap.parse_args()

    import toml
    cfg = toml.load(os.path.join(_HERE, args.config))
    if args.exp_tag:
        cfg["paths"]["exp_dir"] = cfg["paths"]["exp_dir"] + "_" + args.exp_tag
    setup_speakerlab(cfg["paths"]["sv_ecapa"], _HERE)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda"
    torch.manual_seed(cfg["data"]["seed"])

    exp_dir = os.path.join(_HERE, cfg["paths"]["exp_dir"])
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)

    # ---- 数据 ----
    rows, spk2id = read_train_csv(
        os.path.join(cfg["paths"]["data_dir_820"], "train_utts.csv"))
    if args.max_train:
        rows = rows[:args.max_train]
        spk2id = {s: i for i, s in enumerate(sorted({r[1] for r in rows}))}
        rows = [(u, s, p, spk2id[s]) for u, s, p, _ in rows]
    noise_paths = [os.path.join(cfg["paths"]["sv_ecapa"], ln.strip())
                   for ln in open(os.path.join(cfg["paths"]["data_dir_820"],
                                               "noise_train.txt")) if ln.strip()]

    # ---- 模型：soup5 起步 + layer4 专家化 ----
    from trainer_spk_preserve import load_eres2net
    base = load_eres2net(cfg["paths"]["sv_ecapa"], cfg["paths"]["init_ckpt"], "cpu")
    for p in base.parameters():
        p.requires_grad = True          # load_eres2net 冻结了，这里要全量微调
    moe = MoEERes2Net(base, n_expert=cfg["model"]["n_expert"],
                      gamma=cfg["model"]["gamma"]).to(device)
    n_par = sum(p.numel() for p in moe.parameters())
    n_exp = sum(p.numel() for n, p in moe.named_parameters() if n.startswith("experts."))
    print(f"[moe] 骨架起步 {cfg['paths']['init_ckpt']}", flush=True)
    print(f"[moe] 总参 {n_par/1e6:.2f}M, 其中 experts(含原始 layer4) {n_exp/1e6:.2f}M, "
          f"K={cfg['model']['n_expert']}, gamma={cfg['model']['gamma']}", flush=True)

    classifier = build_classifier(512, len(spk2id))
    protos, proto_spk_ids, _ = load_prototypes(cfg["paths"]["proto_npz"])
    if args.max_train:
        sel = {s: i for i, s in enumerate(proto_spk_ids)}
        sub = np.stack([protos[sel[s]] for s in spk2id])
        init_classifier_prototypes(classifier, sub, list(spk2id.keys()), spk2id)
    else:
        init_classifier_prototypes(classifier, protos, proto_spk_ids, spk2id)
    joint = MoEJointModel(moe, classifier).to(device)
    margin_loss = build_aam_margin(cfg["model"]["scale"], cfg["model"]["margin"]).to(device)
    cond_ce = torch.nn.CrossEntropyLoss()

    ds = MultiCondDataset(rows, noise_paths, crop_len=int(3 * 16000))
    print(f"[data] {len(ds)} 样本(3 条件), {len(spk2id)} spks", flush=True)
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
          f"eff_batch={args.batch_size * args.accum}, lr={args.lr}", flush=True)

    gstep, best = 0, 1e9
    for ep in range(1, args.epochs + 1):
        joint.train()
        sampler.set_epoch(ep)
        ds.worker_seed_base = cfg["data"]["seed"] + ep * 1000
        sums = {"loss": 0.0, "acc": 0.0, "racc": 0.0, "n": 0}
        opt.zero_grad()
        for it, (fb, lb, cond) in enumerate(loader):
            fb = fb.to(device, non_blocking=True)
            lb = lb.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                cosine, r_logits = joint(fb, cond=cond)
                loss = margin_loss(cosine.float(), lb) + cond_ce(r_logits, cond)
            scaler.scale(loss / args.accum).backward()
            if (it + 1) % args.accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                gstep += 1
            sums["loss"] += float(loss)
            sums["acc"] += float((cosine.argmax(1) == lb).float().mean())
            sums["racc"] += float((r_logits.argmax(1) == cond).float().mean())
            sums["n"] += 1
            if (it + 1) % (50 * args.accum) == 0:
                n = sums["n"]
                print(f"  ep{ep} it {it+1}/{n_batches} loss={sums['loss']/n:.4f} "
                      f"acc={sums['acc']/n:.4f} racc={sums['racc']/n:.4f} ({gstep})",
                      flush=True)
                sums = {"loss": 0.0, "acc": 0.0, "racc": 0.0, "n": 0}
        metrics = run_validation(moe, bench_dir, val_trials, device, batch_size=24)
        avg = (metrics["0db"]["eer"] + metrics["m5db"]["eer"]) / 2
        print(f"  [val] ep{ep} clean={metrics['clean']['eer']:.3f} "
              f"0db={metrics['0db']['eer']:.3f} m5db={metrics['m5db']['eer']:.3f} "
              f"avg01={avg:.3f}", flush=True)
        ck = {"embedding_model": moe.state_dict(),
              "classifier": joint.classifier.state_dict(),
              "epoch": ep, "val_metrics": metrics, "args": vars(args),
              "moe_cfg": {"n_expert": cfg["model"]["n_expert"],
                          "gamma": cfg["model"]["gamma"]}}
        torch.save(ck, os.path.join(exp_dir, "checkpoints", "checkpoint_last.tar"))
        torch.save(ck, os.path.join(exp_dir, "checkpoints", f"checkpoint_ep{ep}.tar"))
        if avg < best:
            best = avg
            torch.save(ck, os.path.join(exp_dir, "checkpoints", "checkpoint_best.tar"))
            print(f"  [val] 新最优 avg={avg:.3f} -> checkpoint_best.tar", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
