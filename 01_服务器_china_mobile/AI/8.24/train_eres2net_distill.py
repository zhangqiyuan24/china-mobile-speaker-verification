#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_eres2net_distill.py — WavLM teacher → ERes2Net-large 蒸馏训练（8.24）

背景（验收规则澄清后）:
  交付模型必须是基线骨干+叠加功能(换模型不算)。知识蒸馏是加在基线上的训练法:
  用多条件微调后的 WavLM(teacher)的嵌入作为软目标, 引导 ERes2Net-large(student)
  学习更抗噪的说话人表征。交付物仍是 ERes2Net-large + (前端/鲁棒训练/蒸馏)。

接口已逐一对照服务器 8.20/8.21 真实代码核对（2026-08-24 19:50）:
  - 8.20/train_eres2net_robust.py: read_train_csv->(rows,spk2id), CondBatchSampler,
    worker_init_fn(用 dataset.worker_seed_base)
  - 8.20/trainer_eres2net_robust.py: JointModel.forward=classifier(embedding_model(x))
    即输出余弦 logits; margin_loss=ArcMarginLoss 吃 cosine; build_aam_margin(s,m)
  - 8.21/trainer_eres2net_robust_proto.py: load_prototypes(npz)->(protos,spk_ids,used),
    init_classifier_prototypes(classifier,protos,spk_ids,spk2id)
  - 8.20/validate_robust.py: run_validation(model,bench,trials,dev,bs)->{cond:{eer,mindcf}}
  - common_utils: mix_noise(clean,noise_full,snr,start,rng=None)->(mixed,clip_scale)

文献依据(heo23 Interspeech2023 OS-KDFT, 全文已读):
  1) KD:CE 配比是成败关键 -> 余弦嵌入蒸馏(量纲天然平衡), lambda 可调
  2) 朴素联合训练有严重负迁移 -> 默认 --student-init 从已鲁棒训练 ckpt 初始化,
     KD 只作正则
  3) 分模块 lr 是最大单项增益 -> body/classifier 双组 lr

用法:
  python train_eres2net_distill.py --config config_distill.toml --gpu 3 \
      --teacher-gpu 2 --batch-size 24 --accum 4 \
      --student-init ../8.24/exp/soup_821_90k/checkpoints/checkpoint_soup_w50.tar \
      --teacher-ckpt exp/wavlm_ft/checkpoints/checkpoint_best.tar
"""
import argparse
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_V820 = os.path.join(os.path.dirname(_HERE), "8.20")
_V821 = os.path.join(os.path.dirname(_HERE), "8.21")
for _p in (_V820, _V821):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# torch 1.10 fsdp shim(transformers 4.46 的 WavLM encoder 需要)
if "torch.distributed.fsdp" not in sys.modules:
    try:
        import torch.distributed.fsdp  # noqa: F401
    except ImportError:
        _fake = types.ModuleType("torch.distributed.fsdp")

        class _FSDP:
            pass

        _fake.FullyShardedDataParallel = _FSDP
        sys.modules["torch.distributed.fsdp"] = _fake
        import torch.distributed as _td
        _td.fsdp = _fake

from common_utils import load_wav, random_crop, mix_noise, setup_speakerlab
from train_eres2net_robust import (
    read_train_csv, worker_init_fn, CondBatchSampler,
)
from trainer_eres2net_robust import (
    build_eres2net_large, load_pretrained_embedding, build_classifier,
    build_aam_margin, apply_freeze, freeze_bn_stats, freeze_report, JointModel,
)
from trainer_eres2net_robust_proto import (
    load_prototypes, init_classifier_prototypes,
)
from validate_robust import build_fbank, run_validation


class DistillDataset(Dataset):
    """镜像 8.20 MultiCondDataset(3 条件), 额外返回归一化波形给 WavLM teacher。
    长度 3N：idx -> (base=idx//3, cond=idx%3)。cond 0/1/2 = clean/0dB/-5dB。"""

    def __init__(self, rows, noise_paths, crop_len, snrs=(0.0, -5.0)):
        self.rows = rows                    # [(utt_id, spk, path, label)]
        self.noise_paths = noise_paths
        self.crop_len = crop_len
        self.snrs = snrs
        self.fbank = build_fbank()
        self.worker_seed_base = 2026        # worker_init_fn 需要

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
        feat = self.fbank(torch.from_numpy(wav3))     # [T, 80] mean-norm
        # Wav2Vec2FeatureExtractor 同款 per-utt 归一化(与 wavlm_finetune 一致)
        tw = (wav3 - wav3.mean()) / np.sqrt(wav3.var() + 1e-7)
        return feat, torch.from_numpy(tw.astype(np.float32)), label


def distill_collate(batch):
    feats = [b[0] for b in batch]
    wavs = torch.nn.utils.rnn.pad_sequence([b[1] for b in batch], batch_first=True)
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    maxlen = max(f.shape[0] for f in feats)
    fb = torch.zeros(len(feats), maxlen, feats[0].shape[1])
    for i, f in enumerate(feats):
        fb[i, :f.shape[0]] = f
    return fb, wavs, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_distill.toml")
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--teacher-gpu", default="2")
    ap.add_argument("--teacher-ckpt",
                    default="exp/wavlm_ft/checkpoints/checkpoint_best.tar")
    ap.add_argument("--teacher-model-dir", default="wavlm-base-sv")
    ap.add_argument("--teacher-type", default="wavlm",
                    choices=("wavlm", "eres2net"),
                    help="wavlm: 跨前端 SSL teacher(输入波形); "
                         "eres2net: born-again 自蒸馏/集成 teacher(输入同 fbank, "
                         "朴素嵌入融合已证伪, 只用单强模型=soup)")
    ap.add_argument("--teacher-amp", type=int, default=1,
                    help="teacher 前向 autocast(纯推理, fp16 安全且省显存)")
    ap.add_argument("--student-init", default=None,
                    help="从已鲁棒训练 ckpt 初始化(需含 embedding_model+classifier; "
                         "如 soup_821_90k/checkpoint_soup_w50.tar), KD 作正则 — "
                         "heo23: 朴素联合训练负迁移严重, 此为规避")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5,
                    help="student body(骨干)学习率")
    ap.add_argument("--lr-head", type=float, default=1e-4,
                    help="classifier 学习率(分模块 lr, heo23 最大单项增益)")
    ap.add_argument("--distill-lambda", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true", help="student 混合精度(P100 收益小)")
    ap.add_argument("--val-every", type=int, default=1)
    args = ap.parse_args()

    import toml
    cfg = toml.load(args.config)
    # speakerlab 路径注入(build_eres2net_large 需要, 8.20 管线必备前置)
    setup_speakerlab(cfg["paths"]["sv_ecapa"],
                     os.path.dirname(os.path.abspath(args.config)))

    # CUDA_VISIBLE_DEVICES 顺序: 0=student, 1=teacher(两卡时);
    # 同卡(born-again 常用)必须单设备映射 — "3,3" 重复映射 CUDA 只认 1 个设备,
    # cuda:1 会报 Error 101 invalid device ordinal (8.25 首跑踩坑)
    if args.gpu == args.teacher_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        dev_s, dev_t = "cuda:0", "cuda:0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu},{args.teacher_gpu}"
        dev_s, dev_t = "cuda:0", "cuda:1"
    torch.manual_seed(cfg["data"]["seed"])

    exp_dir = cfg["paths"]["exp_dir"]
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)

    # ---- teacher(冻结, 第二卡) ----
    if not os.path.isfile(args.teacher_ckpt):
        raise FileNotFoundError(f"teacher ckpt 不存在: {args.teacher_ckpt}")
    if args.teacher_type == "wavlm":
        from transformers import WavLMForXVector
        teacher = WavLMForXVector.from_pretrained(args.teacher_model_dir)
        sd = torch.load(args.teacher_ckpt, map_location="cpu")["model"]
        teacher.load_state_dict(sd)
    else:
        # eres2net born-again: 单强 ckpt(如 soup), 吃与 student 相同的 fbank
        ck = torch.load(args.teacher_ckpt, map_location="cpu")
        teacher = build_eres2net_large()
        teacher.load_state_dict(ck["embedding_model"])
    teacher.eval().to(dev_t)
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"[teacher] {args.teacher_type}: {args.teacher_ckpt}", flush=True)

    # ---- 数据(只读复用; 先读, spk2id 供 student 构建用) ----
    rows, spk2id = read_train_csv(
        os.path.join(cfg["paths"]["data_dir_820"], "train_utts.csv"))
    noise_paths = [os.path.join(cfg["paths"]["sv_ecapa"], ln.strip())
                   for ln in open(os.path.join(cfg["paths"]["data_dir_820"],
                                               "noise_train.txt")) if ln.strip()]

    # ---- student: ERes2Net-large ----
    model = build_eres2net_large()
    classifier = build_classifier(cfg["model"]["embedding_size"], len(spk2id))
    frozen_mods = []
    if args.student_init:
        ck = torch.load(args.student_init, map_location="cpu")
        model.load_state_dict(ck["embedding_model"])
        if "classifier" in ck:
            classifier.load_state_dict(ck["classifier"])
        else:
            # soup 类 ckpt 只存 embedding_model — classifier 走 proto-init
            # (与 8.21 配方同源, 评测只用 embedding 不受影响)
            protos, spk_ids, _used = load_prototypes(cfg["paths"]["proto_npz"])
            init_classifier_prototypes(classifier, protos, spk_ids, spk2id)
            print("[student-init] ckpt 无 classifier, 已回退 proto-init", flush=True)
        print(f"[student-init] {args.student_init} (epoch={ck.get('epoch', '?')})",
              flush=True)
        # 全参数小步微调(不冻结) — KD 正则阶段
    else:
        load_pretrained_embedding(model, cfg["paths"]["eres2net_ckpt"])
        protos, spk_ids, _used = load_prototypes(cfg["paths"]["proto_npz"])
        init_classifier_prototypes(classifier, protos, spk_ids, spk2id)
        frozen_mods, trainable_mods = apply_freeze(model)
        freeze_report(model, classifier)
    joint = JointModel(model, classifier).to(dev_s)
    margin_loss = build_aam_margin(cfg["model"]["scale"], cfg["model"]["margin"])
    ds = DistillDataset(rows, noise_paths, crop_len=int(3 * 16000))
    print(f"[data] {len(ds)} 样本(3 条件 x {len(rows)} utt), "
          f"{len(spk2id)} spks, noise {len(noise_paths)} 条", flush=True)

    sampler = CondBatchSampler(len(rows), args.batch_size, cfg["data"]["seed"])
    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=sampler, collate_fn=distill_collate,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=worker_init_fn)

    # 分模块 lr: body 小步 / classifier 大步
    body_params = [p for p in joint.embedding_model.parameters() if p.requires_grad]
    head_params = [p for p in joint.classifier.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": body_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=5e-5)
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    bench_dir = os.path.join(cfg["paths"]["data_dir_820"], "val_bench")
    val_trials = os.path.join(cfg["paths"]["data_dir_820"], "val_trials.txt")
    n_batches = len(sampler)
    print(f"[train] {n_batches} micro-batches/epoch x {args.epochs} epochs, "
          f"lambda={args.distill_lambda}, eff_batch={args.batch_size * args.accum}",
          flush=True)

    gstep, best = 0, 1e9
    for ep in range(1, args.epochs + 1):
        joint.train()
        if frozen_mods:
            freeze_bn_stats(model, frozen_mods)
        sampler.set_epoch(ep)
        ds.worker_seed_base = cfg["data"]["seed"] + ep * 1000
        sums = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "acc": 0.0, "n": 0}
        opt.zero_grad()
        for it, (fb, wv, lb) in enumerate(loader):
            warm = min(1.0, (gstep + 1) / max(1, args.warmup))
            for g, bl in zip(opt.param_groups, base_lrs):
                g["lr"] = bl * warm
            fb = fb.to(dev_s, non_blocking=True)
            wv = wv.to(dev_t, non_blocking=True)
            lb = lb.to(dev_s, non_blocking=True)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=bool(args.teacher_amp)):
                    if args.teacher_type == "wavlm":
                        t_emb = teacher(input_values=wv).embeddings
                    else:
                        t_emb = teacher(fb.to(dev_t))
                t_emb = F.normalize(t_emb.float(), dim=-1)
            with torch.cuda.amp.autocast(enabled=args.amp):
                emb = joint.embedding_model(fb)          # [B, 512]
            # ArcMarginLoss 内部 fp32 混算会炸(8.20 同款注释), 损失统一回 fp32
            emb = emb.float()
            cosine = joint.classifier(emb)               # 余弦 logits
            ce = margin_loss(cosine, lb)                 # 返回标量损失(含 CE, 8.20 同款)
            logits = cosine                              # acc 用 argmax(8.20 同款)
            # teacher 输出是 no_grad 叶子, 跨卡搬到 student 设备零开销
            # (反向图不可跨设备, 绝不能把 emb 搬去 dev_t)
            kd = (1.0 - F.cosine_similarity(
                F.normalize(emb), t_emb.to(dev_s), dim=1)).mean()
            loss = ce + args.distill_lambda * kd
            scaler.scale(loss / args.accum).backward()
            if (it + 1) % args.accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                gstep += 1
            sums["loss"] += float(loss)
            sums["ce"] += float(ce)
            sums["kd"] += float(kd)
            sums["acc"] += float((logits.argmax(1) == lb).float().mean())
            sums["n"] += 1
            if (it + 1) % (25 * args.accum) == 0:
                n = sums["n"]
                print(f"  ep{ep} it {it+1}/{n_batches} "
                      f"loss={sums['loss']/n:.4f} ce={sums['ce']/n:.4f} "
                      f"kd={sums['kd']/n:.4f} acc={sums['acc']/n:.4f} "
                      f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)
                sums = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "acc": 0.0, "n": 0}
        # ---- 每 epoch 官方 bench 验证(与 8.21 同款) ----
        metrics = run_validation(joint.embedding_model, bench_dir, val_trials,
                                 dev_s, batch_size=24)
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
    print(f"[done] best avg(0db,m5db) = {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
