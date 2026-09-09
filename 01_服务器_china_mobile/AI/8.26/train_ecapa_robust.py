#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_ecapa_robust.py — ECAPA-TDNN 鲁棒多条件训练（8.26 换靶子线）

用户 8.25 深夜指令: 策略用尽前, 把最有效的配方平移到 ECAPA-TDNN, 看能否
对 ECAPA 自己的基线(0dB 11.150 / -5dB 14.360, 8.24 README 口径)拿满 -30%
—— 达标线 0dB<=7.805 / -5dB<=10.052。ECAPA 基线更弱 => 提升空间更大,
且配方(多条件训练+AAM+proto 分类器)完全模型无关。

配方 = 8.21 train_eres2net_robust 镜像(3 条件多条件, 无信道增强):
  - ECAPA-TDNN(input 80 fbank, emb 192) 全量微调 lr 2e-5(20.8M, 不冻结)
  - 分类器 = build_classifier(192, 9900) + ECAPA 原型表 proto 初始化
  - CondBatchSampler b30 accum3 (eff 90)
合规: 配方对所有基线同构, 计分 vs ECAPA 自己的原始成绩。

用法: python train_ecapa_robust.py --config config_ecapa_robust.toml --gpu 1
"""
import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_V820 = os.path.join(os.path.dirname(_HERE), "8.20")
_V821 = os.path.join(os.path.dirname(_HERE), "8.21")
_V824 = os.path.join(os.path.dirname(_HERE), "8.24")
for _p in (_V820, _V821, _V824, _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from common_utils import load_wav, random_crop, mix_noise, setup_speakerlab
from train_eres2net_robust import read_train_csv, worker_init_fn, CondBatchSampler
from trainer_eres2net_robust import build_classifier, build_aam_margin, JointModel
from validate_robust import build_fbank, run_validation


class MultiCondDataset(torch.utils.data.Dataset):
    """8.21 三条件数据集镜像(clean/0dB/-5dB 各 1/3)。"""

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
        return feat, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_ecapa_robust.toml")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--batch-size", type=int, default=30)
    ap.add_argument("--accum", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--max-train", type=int, default=None,
                    help="只用前 N 行(冒烟)")
    ap.add_argument("--exp-tag", default=None, help="exp_dir 后缀(冒烟用)")
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

    rows, spk2id = read_train_csv(
        os.path.join(cfg["paths"]["data_dir_820"], "train_utts.csv"))
    if args.max_train:
        rows = rows[:args.max_train]
        spk2id = {s: i for i, s in enumerate(sorted({r[1] for r in rows}))}
        rows = [(u, s, p, spk2id[s]) for u, s, p, _ in rows]
    noise_paths = [os.path.join(cfg["paths"]["sv_ecapa"], ln.strip())
                   for ln in open(os.path.join(cfg["paths"]["data_dir_820"],
                                               "noise_train.txt")) if ln.strip()]

    from speakerlab.models.ecapa_tdnn.ECAPA_TDNN import ECAPA_TDNN
    model = ECAPA_TDNN(input_size=80, lin_neurons=192, channels=[1024, 1024, 1024, 1024, 3072])
    sd = torch.load(cfg["paths"]["ecapa_ckpt"], map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[ecapa] pretrained 装载完成 {n_par/1e6:.1f}M(emb 192, 全量微调不冻结)",
          flush=True)
    model = model.to(device)

    classifier = build_classifier(192, len(spk2id))
    npz = np.load(os.path.join(_HERE, "exp", "ecapa_protos", "prototypes.npz"),
                  allow_pickle=True)
    protos, proto_spk_ids = npz["embs"], list(npz["spk_ids"])
    proto_idx = {s: i for i, s in enumerate(proto_spk_ids)}
    if args.max_train:   # 冒烟: 只取子集说话人的原型
        sel = np.stack([protos[proto_idx[s]] for s in spk2id])
    else:
        assert list(spk2id.keys()) == proto_spk_ids, \
            "ecapa 原型表与 train_utts.csv 说话人集合不一致, 先跑 build_ecapa_protos"
        sel = protos
    # 原型直装分类器权重(cosine 层): w[i] = proto[spk i]
    with torch.no_grad():
        classifier.weight.copy_(torch.from_numpy(sel))
    classifier = classifier.to(device)
    joint = JointModel(model, classifier)
    margin_loss = build_aam_margin(cfg["model"]["scale"], cfg["model"]["margin"]).to(device)

    ds = MultiCondDataset(rows, noise_paths, crop_len=int(3 * 16000))
    print(f"[data] {len(ds)} 样本(3 条件), {len(spk2id)} spks", flush=True)

    sampler = CondBatchSampler(len(rows), args.batch_size, cfg["data"]["seed"])
    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=sampler, num_workers=args.num_workers,
        pin_memory=True, worker_init_fn=worker_init_fn)

    params = [p for p in joint.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=5e-5)

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
        sums = {"loss": 0.0, "acc": 0.0, "n": 0}
        opt.zero_grad()
        for it, (fb, lb) in enumerate(loader):
            fb = fb.to(device, non_blocking=True)
            lb = lb.to(device, non_blocking=True)
            cosine = joint(fb).float()
            loss = margin_loss(cosine, lb)
            (loss / args.accum).backward()
            if (it + 1) % args.accum == 0:
                opt.step()
                opt.zero_grad()
                gstep += 1
            sums["loss"] += float(loss)
            sums["acc"] += float((cosine.argmax(1) == lb).float().mean())
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
