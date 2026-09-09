#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""wavlm_finetune.py — WavLM-base-sv 在 90k 多条件数据上的微调（8.24 大杠杆实验）

背景:
  zero-shot EER 27.8%(clean) —— 英文域 VoxCeleb1 微调的模型在中文跨设备上不可用,
  但 WavLM 的 SSL 预训练特征抗噪性远强于 fbank+ResNet(文献共识)。在 8.22 的
  90k 多条件数据(9900 说话人)上微调, 是单系统路线最有希望突破 -30% 的方向。

设计(对齐 8.21/8.22 recipe 的关键超参, 骨干换成 WavLM):
  - 条件: clean / 0dB / -5dB 三条件均衡(同一 utt 的三个条件进同一 batch)
  - 损失: AAM-softmax m=0.2 s=32(与 8.21/8.22 相同)
  - 冻结: WavLM CNN 前端(wav2vec2 系微调标准做法); transformer 全部可训
  - 双组 lr: body 1e-5 / AAM 头 1e-4, warmup 500 步(线性)
  - 数据增强: 随机 3s 裁剪 + musan/freesound 随机噪声段在线混叠(与 8.22 同源)
  - 验证: 每 epoch 在 val_bench(200 enroll x 3 cond, 800 trials, 说话人与训练不重叠)
    上算 EER, 按 avg(0dB,-5dB) 存 best
  - AMP: P100 无 tensor core 但省显存约 40%

用法:
  python wavlm_finetune.py --gpu 2 --utts-per-batch 6 --epochs 6 --out exp/wavlm_ft
"""
import argparse
import csv
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- torch 1.10 兼容 shim(同 wavlm_sv_scores.py) ----
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

SR = 16000
CROP = 3 * SR


def load_wav_16k(path):
    """与 8.20/common_utils.py 同款 torchaudio 加载（soundfile 读部分 musan wav 会炸）。"""
    import torchaudio
    wav, sr = torchaudio.load(path)
    if wav.ndim == 2:          # (channels, time) -> 取第一声道
        wav = wav[0]
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.numpy().astype(np.float32)


def rand_crop(w, rng):
    if len(w) <= CROP:
        return np.concatenate([w, np.zeros(CROP - len(w), np.float32)])
    s = rng.randint(0, len(w) - CROP)
    return w[s:s + CROP]


def mix_at_snr(speech, noise, snr_db, rng):
    """噪声段随机裁剪到语音长度, 按语音功率定标到目标 SNR。"""
    if len(noise) < len(speech):
        rep = int(np.ceil(len(speech) / len(noise)))
        noise = np.tile(noise, rep)
    s = rng.randint(0, len(noise) - len(speech) + 1)
    n = noise[s:s + len(speech)].astype(np.float32)
    p_s = float(np.mean(speech ** 2)) + 1e-10
    p_n = float(np.mean(n ** 2)) + 1e-10
    scale = np.sqrt(p_s / (10 ** (snr_db / 10.0) * p_n))
    return speech + scale * n


class NoiseBank:
    """LRU 预载噪声 wav, 避免每个 step 重复 IO。"""

    def __init__(self, list_path, base_dir, cache_n=40, seed=2026):
        self.paths = [os.path.join(base_dir, ln.strip())
                      for ln in open(list_path) if ln.strip()]
        self.cache = {}
        self.cache_n = cache_n
        self.rng = np.random.RandomState(seed)

    def get(self):
        p = self.paths[self.rng.randint(len(self.paths))]
        if p not in self.cache:
            if len(self.cache) >= self.cache_n:
                self.cache.pop(next(iter(self.cache)))
            self.cache[p] = load_wav_16k(p).astype(np.float32)
        return self.cache[p]


class AAMSoftmax(nn.Module):
    def __init__(self, in_dim, n_class, m=0.2, s=32.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_class, in_dim) * 0.01)
        self.m, self.s = m, s

    def forward(self, x, labels):
        cos = F.linear(F.normalize(x), F.normalize(self.weight))
        onehot = torch.zeros_like(cos).scatter_(1, labels.unsqueeze(1), 1.0)
        cos_mt = cos - self.m * onehot
        return self.s * (cos * (1 - onehot) + cos_mt * onehot)


class MultiCondSet(torch.utils.data.Dataset):
    """__getitem__ 接受 (utt_idx, cond) 元组(DataLoader fetcher 直接透传)。
    随机种子由 (utt, cond, epoch) 决定 —— worker 副本间天然一致, epoch 间变化。"""

    def __init__(self, csv_path, noise_bank, seed=2026):
        self.items = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                self.items.append((row["spk"], row["wav_path"]))
        self.spk2id = {s: i for i, s in enumerate(sorted({x[0] for x in self.items}))}
        self.noise = noise_bank
        self.epoch = 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        i, cond = idx
        rng = np.random.RandomState(((i + 1) * 1000003 + cond * 101 + self.epoch * 7919)
                                    % (2 ** 32 - 1))
        spk, path = self.items[i]
        w = rand_crop(load_wav_16k(path), rng)
        if cond == 1:
            w = mix_at_snr(w, self.noise.get(), 0.0, rng)
        elif cond == 2:
            w = mix_at_snr(w, self.noise.get(), -5.0, rng)
        # Wav2Vec2FeatureExtractor 的 do_normalize: per-utt 零均值单位方差
        w = (w - w.mean()) / (np.sqrt(w.var() + 1e-7))
        return torch.from_numpy(w.astype(np.float32)), self.spk2id[spk]


def worker_init_fn(worker_id):
    """每个 DataLoader worker 用 torch 派生种子重置噪声选择 RNG。"""
    info = torch.utils.data.get_worker_info()
    info.dataset.noise.rng = np.random.RandomState(torch.initial_seed() % 2 ** 31)


class CondBatchSampler(torch.utils.data.sampler.Sampler):
    """每 step 取 B 个 utt × 3 条件; 无放回轮询整表。"""

    def __init__(self, n, utts_per_batch, seed):
        self.n, self.b, self.seed = n, utts_per_batch, seed
        self.epoch = 0

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        self.epoch += 1
        idx = rng.permutation(self.n)
        for s in range(0, self.n - self.b + 1, self.b):
            chunk = idx[s:s + self.b]
            # 每次必须 yield 一整个 batch(list) —— DataLoader 视其为 batch 本身
            yield [(int(i), cond) for cond in (0, 1, 2) for i in chunk]

    def __len__(self):
        return self.n // self.b


@torch.no_grad()
def embed_val(model, wav_dirs, device, bs=8):
    """wav_dirs: {name: dir}; 返回 {name: {stem: emb}}。"""
    out = {}
    for name, d in wav_dirs.items():
        embs = {}
        files = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
        for s in range(0, len(files), bs):
            chunk = files[s:s + bs]
            ws = [load_wav_16k(os.path.join(d, f)) for f in chunk]
            ws = [rand_crop(w, np.random.RandomState(0)) for w in ws]
            ws = [(w - w.mean()) / (np.sqrt(w.var() + 1e-7)) for w in ws]
            maxlen = max(len(w) for w in ws)
            x = torch.zeros(len(ws), maxlen)
            m = torch.zeros(len(ws), maxlen, dtype=torch.long)
            for i, w in enumerate(ws):
                x[i, :len(w)] = torch.from_numpy(w)
                m[i, :len(w)] = 1
            e = model(input_values=x.to(device),
                      attention_mask=m.to(device)).embeddings
            e = F.normalize(e, dim=-1).cpu().numpy()
            for f, v in zip(chunk, e):
                embs[os.path.splitext(f)[0]] = v
        out[name] = embs
        print(f"    [val] {name}: {len(embs)} 条", flush=True)
    return out


def val_eer(emb_by_cond, trials_path):
    """emb_by_cond: {cond: {stem: emb}}; 返回各 cond EER。"""
    res = {}
    tri = [ln.split() for ln in open(trials_path)]
    for cond, embs in emb_by_cond.items():
        sc, lb = [], []
        for enr, tst, lab in tri:
            sc.append(float(np.dot(embs[enr], embs[tst])))
            lb.append(1 if lab == "target" else 0)
        sc, lb = np.array(sc), np.array(lb)
        order = np.argsort(-sc)
        labs = lb[order]
        P = int(lb.sum()); N = int(len(lb) - P)
        tp = np.cumsum(labs); fp = np.cumsum(1 - labs)
        fpr = fp / float(N); miss = (P - tp) / float(P)
        i = int(np.argmin(np.abs(fpr - miss)))
        res[cond] = (fpr[i] + miss[i]) / 2.0 * 100.0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--model-dir", default="wavlm-base-sv")
    ap.add_argument("--data-dir", default="../8.22/exp/data_90k/data")
    ap.add_argument("--noise-base", default="/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker")
    ap.add_argument("--out", default="exp/wavlm_ft")
    ap.add_argument("--utts-per-batch", type=int, default=6)   # x3 条件 = 18 条/step
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr-body", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--freeze-layers", type=int, default=6,
                    help="冻结 WavLM 底部 N 层 transformer(省显存+正则化, 0=全训)")
    ap.add_argument("--amp", action="store_true",
                    help="fp16 混合精度(WavLM backward 在 fp16 会 inf, 已实测, 默认关)")
    ap.add_argument("--proto-init", default=None,
                    help="protos.npy 路径: AAM 权重用 WavLM clean prototype 初始化(8.21 E2-proto 思想)")
    ap.add_argument("--renorm-aam", type=int, default=100,
                    help="每 N 步把 AAM 权重行重归一化到单位范数(0=关); 防梯度浪费在范数膨胀上")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="梯度检查点: 重算激活换显存, fp32 大 batch 时用")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=1)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(2026)
    os.makedirs(os.path.join(args.out, "checkpoints"), exist_ok=True)

    from transformers import WavLMForXVector
    model = WavLMForXVector.from_pretrained(args.model_dir)
    # 冻结 CNN 前端
    for p in model.wavlm.feature_extractor.parameters():
        p.requires_grad = False
    # 冻结底部 N 层 transformer(激活不需为反向保留, 显存减半)
    n_frozen = 0
    for layer in list(model.wavlm.encoder.layers)[:args.freeze_layers]:
        for p in layer.parameters():
            p.requires_grad = False
            n_frozen += 1
    model.to(device)
    if args.grad_ckpt:
        # HF4.46 默认会注入 use_reentrant=True, torch1.10 的 checkpoint 不认;
        # 传空 kwargs 即纯 checkpoint(fn,*args)。reentrant 模式要求检查点输入
        # 带梯度 —— feature_projection(未冻结)保证了进入第 6 层的 hidden 带 grad。
        model.wavlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={})
        print("[grad-ckpt] 已开启(空 kwargs 兼容 torch1.10)", flush=True)
    print(f"[freeze] CNN 前端 + 底部 {args.freeze_layers} 层 transformer", flush=True)

    nb = NoiseBank(os.path.join(args.data_dir, "noise_train.txt"), args.noise_base)
    ds = MultiCondSet(os.path.join(args.data_dir, "train_utts.csv"), nb)
    n_class = len(ds.spk2id)
    aam = AAMSoftmax(512, n_class).to(device)
    if args.proto_init:
        import json as _json
        protos = np.load(args.proto_init)
        spks = _json.load(open(os.path.join(os.path.dirname(args.proto_init), "spks.json")))
        assert spks == sorted(ds.spk2id.keys()), "prototype 说话人集合与训练集不一致"
        with torch.no_grad():
            aam.weight.copy_(torch.from_numpy(protos).to(aam.weight.device))
        print(f"[proto-init] AAM 已用 {len(spks)} 个 clean prototype 初始化", flush=True)
    print(f"[data] {len(ds)} utts, {n_class} spks, noise {len(nb.paths)} 条", flush=True)

    body_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": body_params, "lr": args.lr_body},
        {"params": aam.parameters(), "lr": args.lr_head},
    ], weight_decay=5e-5)
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    aam_init = aam.weight.detach().clone()
    sampler = CondBatchSampler(len(ds), args.utts_per_batch, 2026)
    steps_per_epoch = len(ds) // args.utts_per_batch
    total_steps = steps_per_epoch * args.epochs
    print(f"[train] {steps_per_epoch} steps/epoch x {args.epochs} epochs", flush=True)

    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=sampler,
        collate_fn=lambda b: (torch.stack([x[0] for x in b]),
                              torch.tensor([x[1] for x in b], dtype=torch.long)),
        num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init_fn)

    val_names = {"enroll": "enroll", "test_clean": "test_clean",
                 "test_0db": "test_0db", "test_m5db": "test_m5db"}
    gstep, best = 0, 1e9
    for ep in range(1, args.epochs + 1):
        ds.epoch = ep  # 新一轮 worker 副本会继承, 换随机种子
        model.train()
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            warm = min(1.0, (gstep + 1) / max(1, args.warmup))
            for g, bl in zip(opt.param_groups, base_lrs):
                g["lr"] = bl * warm
            with torch.cuda.amp.autocast(enabled=args.amp):
                emb = model(input_values=x).embeddings
                logits = aam(emb.float(), y)
                loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            gstep += 1
            if args.renorm_aam and gstep % args.renorm_aam == 0:
                with torch.no_grad():
                    aam.weight.data.copy_(
                        F.normalize(aam.weight.data, dim=1))
            if gstep % 50 == 0:
                acc = (logits.argmax(1) == y).float().mean().item()
                gs = [p.grad for p in body_params if p.grad is not None]
                gnorm = float(torch.sqrt(sum(g.float().norm() ** 2 for g in gs))) if gs else 0.0
                drift = float((aam.weight - aam_init).norm())
                print(f"  ep{ep} step {gstep}/{total_steps} loss={loss.item():.4f} "
                      f"acc={acc:.4f} lr_body={opt.param_groups[0]['lr']:.2e} "
                      f"body_grads={len(gs)}/{len(body_params)} gnorm={gnorm:.4f} "
                      f"aam_drift={drift:.5f} aam_gn="
                      f"{float(aam.weight.grad.norm()) if aam.weight.grad is not None else -1:.5f}",
                      flush=True)
        if ep % args.val_every == 0 or ep == args.epochs:
            model.eval()
            print(f"  [val] epoch {ep}", flush=True)
            emb_map = embed_val(model, {k: os.path.join(args.data_dir, "val_bench", v)
                                        for k, v in val_names.items()}, device)
            eers = {}
            for cond in ("test_clean", "test_0db", "test_m5db"):
                m = {**emb_map["enroll"], **emb_map[cond]}
                eers[cond] = val_eer({cond: m},
                                     os.path.join(args.data_dir, "val_trials.txt"))[cond]
            avg = (eers["test_0db"] + eers["test_m5db"]) / 2
            print(f"  [val] ep{ep} EER: clean={eers['test_clean']:.3f} "
                  f"0db={eers['test_0db']:.3f} m5db={eers['test_m5db']:.3f} avg01={avg:.3f}",
                  flush=True)
            ck = {"model": model.state_dict(), "aam": aam.state_dict(),
                  "epoch": ep, "val_eer": eers,
                  "spk2id": ds.spk2id, "args": vars(args)}
            torch.save(ck, os.path.join(args.out, "checkpoints", "checkpoint_last.tar"))
            if avg < best:
                best = avg
                torch.save(ck, os.path.join(args.out, "checkpoints", "checkpoint_best.tar"))
                print(f"  [val] 新最优 avg={avg:.3f} -> checkpoint_best.tar", flush=True)
    print("[done] best avg(0db,m5db) =", round(best, 3))


if __name__ == "__main__":
    main()
