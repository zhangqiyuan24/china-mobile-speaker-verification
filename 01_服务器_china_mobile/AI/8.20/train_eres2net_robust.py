# -*- coding: utf-8 -*-
"""
E2 主训练：ERes2Net-large 多条件鲁棒微调（clean / 0dB / -5dB = 1:1:1 动态混噪）

数据流：prepare_multicond_data.py 产出的 train_utts.csv + noise_train.txt。
  - 每个 batch 严格 1:1:1（batch_size 必须是 3 的倍数，三种条件各 B/3 条）。
  - 同一条语句在同一 epoch 内三种条件各自独立随机裁剪/选噪（动态混噪，不落盘）。
  - 噪声只来自 train noise pool；SNR 只有 {0, -5}，与验收两档一致。

模型：pretrained ERes2Net-large（strict 加载）→ 冻结 conv1/bn1/layer1/layer2/
layer1_downsample/fuse_mode12（含 BN stats），训练其余 + 新建 CosineClassifier。
损失：AAM-softmax（ArcMarginLoss, margin/scale 固定，无 scheduler）。

选型规则（方案第 6 节）：best = argmin(EER_0db + EER_m5db)，且
EER_clean 不得超过 epoch0（pretrained）验证 clean + clean_degradation_max 个点。
训练准确率只做日志，不参与选型。

用法：
  # 服务器（单卡）：
  python train_eres2net_robust.py --config config_robust.toml --gpu 1
  # 双卡 DDP：
  CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.run --standalone --nproc_per_node=2 \
      train_eres2net_robust.py --config config_robust.toml
  # sanity（同数据只跑 3 epoch，验证管线）：
  python train_eres2net_robust.py --config config_robust.toml --sanity --gpu 1
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import toml
from torch.utils.data import Dataset, DataLoader

from common_utils import (
    load_wav, random_crop, mix_noise, resolve_existing_path, setup_speakerlab, SR,
)
from trainer_eres2net_robust import (
    build_eres2net_large, load_pretrained_embedding, build_classifier,
    build_aam_margin, apply_freeze, freeze_report, JointModel, RobustTrainer,
)
from validate_robust import run_validation, build_fbank


# ============================================================
# 数据
# ============================================================

class MultiCondDataset(Dataset):
    """长度 3N：idx -> (base=idx//3, cond=idx%3)。cond 0/1/2 = clean/0dB/-5dB。"""

    def __init__(self, rows, noise_paths, crop_len, snrs=(0.0, -5.0)):
        self.rows = rows                    # [(utt_id, spk, path, label)]
        self.noise_paths = noise_paths
        self.crop_len = crop_len
        self.snrs = snrs
        self.fbank = build_fbank()

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
        feat = self.fbank(torch.from_numpy(wav3))    # [T, 80], mean-nor
        return feat, label


class CondBatchSampler:
    """
    每个批次严格 B/3 + B/3 + B/3（clean/0dB/-5dB）。每 epoch 打乱各条件内部顺序。
    """

    def __init__(self, num_utts, batch_size, seed=2026):
        assert batch_size % 3 == 0, f"batch_size 必须是 3 的倍数，当前 {batch_size}"
        self.per_cond = batch_size // 3
        self.num_utts = num_utts
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _cond_indices(self, cond, rng):
        # 数据集中条件 cond 的全部索引（idx % 3 == cond）
        idxs = np.arange(cond, 3 * self.num_utts, 3)
        rng.shuffle(idxs)
        return idxs

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        lists = [self._cond_indices(c, rng) for c in range(3)]
        n_batches = min(len(l) for l in lists) // self.per_cond
        for b in range(n_batches):
            batch = []
            for c in range(3):
                s = b * self.per_cond
                batch.extend(lists[c][s:s + self.per_cond].tolist())
            yield batch

    def __len__(self):
        return (self.num_utts // self.per_cond)


def worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    seed = info.dataset.worker_seed_base + worker_id
    np.random.seed(seed)
    import random as _random
    _random.seed(seed)


def read_train_csv(path):
    rows = []
    with open(path) as f:
        header = f.readline()
        for line in f:
            utt_id, spk, wav_path = line.strip().split(",")
            rows.append((utt_id, spk, wav_path, None))
    spks = sorted(set(r[1] for r in rows))
    spk2id = {s: i for i, s in enumerate(spks)}
    rows = [(u, s, p, spk2id[s]) for (u, s, p, _) in rows]
    return rows, spk2id


# ============================================================
# 主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_robust.toml")
    ap.add_argument("--sanity", action="store_true", help="只跑 sanity_epochs 个 epoch（同数据，验证管线）")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--gpu", default=None, help="单卡模式用；DDP 模式由 LOCAL_RANK 决定")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--amp", action="store_true",
                    help="混合精度训练（共享显卡显存紧张时用，约省 40%% 显存）")
    ap.add_argument("--val-batch-size", type=int, default=None,
                    help="验证/提特征 batch（共享卡显存紧张时调小，如 16；不影响结果只影响速度）")
    ap.add_argument("--accum", type=int, default=None,
                    help="梯度累积步数：有效batch = batch_size×accum。共享卡 batch 调小时"
                         "用它对齐配置 batch_size=96（如 batch 12 配 --accum 8），"
                         "否则每 epoch 更新次数变多、参数漂移放大")
    ap.add_argument("--init-from", default=None, help="从已有 checkpoint 续训（加载 embedding+classifier）")
    args = ap.parse_args()

    cfg = toml.load(args.config)
    base = os.path.dirname(os.path.abspath(args.config))
    p_cfg, t_cfg, m_cfg, d_cfg = cfg["paths"], cfg["training"], cfg["model"], cfg["data"]
    sv_ecapa = setup_speakerlab(p_cfg["sv_ecapa"], base)   # 保证 speakerlab 可导入

    epochs = args.epochs or (t_cfg["sanity_epochs"] if args.sanity else t_cfg["epochs"])
    batch_size = args.batch_size or t_cfg["batch_size"]
    lr = args.lr or t_cfg["lr"]
    num_workers = args.num_workers if args.num_workers is not None else t_cfg["num_workers"]
    val_batch_size = args.val_batch_size or t_cfg["val_batch_size"]
    accum = args.accum if args.accum is not None else t_cfg.get("accum", 1)

    # ---- 分布式初始化（torch.distributed.run 启动时自动生效）----
    ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if ddp:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        is_main = rank == 0
        if is_main:
            print(f"[dist] DDP 启动: world_size={world}")
    else:
        if args.gpu:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
        rank, world, is_main = 0, 1, True
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert torch.cuda.is_available(), "训练需要 GPU（未检测到 CUDA）"

    # ---- 路径与数据 ----
    exp_dir = os.path.join(base, p_cfg["exp_dir"])
    data_dir = os.path.join(exp_dir, "data")
    train_csv = os.path.join(data_dir, "train_utts.csv")
    noise_txt = os.path.join(data_dir, "noise_train.txt")
    bench_dir = os.path.join(data_dir, "val_bench")
    val_trials = os.path.join(data_dir, "val_trials.txt")
    for need in (train_csv, noise_txt, bench_dir, val_trials):
        assert os.path.exists(need), f"缺 {need}，请先运行 prepare_multicond_data.py"

    rows, spk2id = read_train_csv(train_csv)
    if is_main:
        json.dump(spk2id, open(os.path.join(data_dir, "labels.json"), "w"), ensure_ascii=False, indent=1)
        print(f"[data] 训练 {len(rows)} 条 / {len(spk2id)} 人；条件=3x -> {3*len(rows)} 样本/epoch")

    noise_paths = [resolve_existing_path(l, sv_ecapa)
                   for l in open(noise_txt) if l.strip()]
    if is_main:
        print(f"[data] train noise pool = {len(noise_paths)} 条（与 test noise 零交集，prepare 已断言）")

    # ---- 模型 ----
    model = build_eres2net_large(embedding_size=m_cfg["embedding_size"],
                                 m_channels=m_cfg["m_channels"])
    eres2net_ckpt = p_cfg["eres2net_ckpt"]
    if not os.path.isabs(eres2net_ckpt):
        eres2net_ckpt = os.path.join(base, eres2net_ckpt)
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu")
        model.load_state_dict(ckpt["embedding_model"])
        print(f"[model] 从 checkpoint 续训: {args.init_from} (epoch={ckpt.get('epoch')})")
    else:
        load_pretrained_embedding(model, eres2net_ckpt)   # strict: missing=unexpected=0
    frozen_mods, trainable_mods = apply_freeze(model)

    classifier = build_classifier(m_cfg["embedding_size"], len(spk2id))
    margin_loss = build_aam_margin(m_cfg["scale"], m_cfg["margin"])
    if is_main:
        freeze_report(model, classifier)
        print(f"[loss] AAM-softmax: margin={m_cfg['margin']} scale={m_cfg['scale']} "
              f"(固定，无 scheduler)  amp={args.amp or t_cfg.get('amp', False)} "
              f"batch_size={batch_size} accum={accum} (有效batch={batch_size * accum}) "
              f"val_batch_size={val_batch_size}")

    joint = JointModel(model, classifier)
    if ddp:
        joint = torch.nn.parallel.DistributedDataParallel(
            joint, device_ids=[local_rank])
    trainer = RobustTrainer(joint, margin_loss, frozen_mods,
                            lr=lr, weight_decay=t_cfg["weight_decay"], device=device,
                            amp=args.amp or t_cfg.get("amp", False), accum=accum)

    # ---- 数据加载 ----
    crop_len = int(d_cfg["crop_seconds"] * SR)
    dataset = MultiCondDataset(rows, noise_paths, crop_len)
    dataset.worker_seed_base = d_cfg["seed"]
    sampler = CondBatchSampler(len(rows), batch_size, seed=d_cfg["seed"])

    def make_loader(epoch):
        sampler.set_epoch(epoch)
        batches = list(iter(sampler))
        if world > 1:
            # 整除切分（不能用 [rank::world]：batch 数不齐会导致 backward 的
            # allreduce 与先到 barrier 的 rank 互等，死锁）
            n = len(batches) // world
            batches = batches[rank * n:(rank + 1) * n]
        dataset.worker_seed_base = d_cfg["seed"] + epoch * 1000  # 每 epoch 换裁剪/选噪
        g = torch.Generator()
        g.manual_seed(d_cfg["seed"] + epoch)
        return DataLoader(dataset, batch_sampler=batches,
                          num_workers=num_workers, worker_init_fn=worker_init_fn,
                          generator=g, pin_memory=True)

    # ---- 冻结守门：第一个“生效”的 step 后验证冻结参数没动、可训练参数动了 ----
    # 两个正常延迟：梯度累积要满 accum 个 micro-batch 才 step；AMP 下 GradScaler
    # 初始 scale=65536，头几次 step 可能因梯度溢出被跳过并自动降 scale。故最多试
    # 64 个 micro-step，其中有一次参数更新即通过；全没动才是真故障。
    def first_step_guard(loader):
        frozen_p = next(p for n, p in model.named_parameters()
                        if not p.requires_grad)
        train_p = next(p for n, p in model.named_parameters() if p.requires_grad)
        f0, t0 = frozen_p.detach().clone(), train_p.detach().clone()
        it = iter(loader)
        trainer.train_mode()
        loss, tried = None, 0
        for _ in range(64):
            feats, labels = next(it)
            loss, _acc = trainer.train_step(feats, labels)
            tried += 1
            assert torch.equal(f0, frozen_p.detach()), \
                "冻结参数被更新了！检查 optimizer 参数表 / requires_grad"
            if not torch.equal(t0, train_p.detach()):
                break
        else:
            raise AssertionError(
                f"连续 {tried} 个 micro-step 可训练参数都没有更新！"
                f"检查 optimizer 参数表 / requires_grad / accum 设置")
        if is_main:
            n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            note = "" if tried == 1 else f"（含 accum 累积与/或 AMP 溢出跳过，正常）"
            print(f"[guard] 第 {tried} 个 micro-step 后参数更新生效: loss={loss:.4f}；"
                  f"冻结 {n_frozen/1e6:.2f}M 参数未变，"
                  f"可训练 {n_train/1e6:.2f}M 参数已更新 ✓{note}")

    # ---- epoch0：pretrained 基线验证 ----
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    history = []
    if is_main:
        print("\n===== epoch 0（pretrained 基线）验证 =====")
        base_metrics = run_validation(model, bench_dir, val_trials, device,
                                      batch_size=val_batch_size)
        clean_gate = base_metrics["clean"]["eer"] + t_cfg["clean_degradation_max"]
        print(f"[gate] clean 门限 = epoch0 clean {base_metrics['clean']['eer']:.4f}% "
              f"+ {t_cfg['clean_degradation_max']}pp = {clean_gate:.4f}%")
        history.append({"epoch": 0, "phase": "pretrained", "val": base_metrics})

    best = {"noise_sum": float("inf"), "epoch": -1}
    log_every = t_cfg.get("log_every", 50)
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        if is_main:
            print(f"\n===== epoch {epoch}/{epochs} (lr={lr}) =====")
        loader = make_loader(epoch)
        if epoch == 1:
            first_step_guard(loader)
        trainer.train_mode()
        sums = {"loss": 0.0, "acc": 0.0, "n": 0}
        t_ep = time.time()
        for step, (feats, labels) in enumerate(loader, 1):
            loss, acc = trainer.train_step(feats, labels)
            sums["loss"] += loss
            sums["acc"] += acc
            sums["n"] += 1
            if is_main and (step % log_every == 0 or step == len(loader)):
                print(f"  step {step}/{len(loader)}  loss={sums['loss']/sums['n']:.4f}  "
                      f"train_acc={sums['acc']/sums['n']:.4f}  "
                      f"({(time.time()-t_ep)/step:.2f}s/step)", flush=True)
        trainer.flush()   # 用掉不足一组的残余梯度（7500 % 8 = 4，会有残留）

        # ---- 验证（rank0）----
        metrics = None
        if is_main:
            metrics = run_validation(model, bench_dir, val_trials, device,
                                     batch_size=val_batch_size)
            noise_sum = metrics["0db"]["eer"] + metrics["m5db"]["eer"]
            clean_ok = bool(metrics["clean"]["eer"] <= clean_gate)
            rec = {"epoch": epoch, "val": metrics,
                   "noise_sum": noise_sum, "clean_gate_ok": clean_ok,
                   "train_loss": sums["loss"] / max(1, sums["n"]),
                   "train_acc": sums["acc"] / max(1, sums["n"])}
            history.append(rec)
            trainer.save_checkpoint(
                os.path.join(ckpt_dir, f"checkpoint_epoch{epoch}.tar"),
                epoch, val_metrics=metrics)
            improved = noise_sum < best["noise_sum"] and clean_ok
            if improved:
                best = {"noise_sum": noise_sum, "epoch": epoch,
                        "clean_ok": clean_ok}
                trainer.save_checkpoint(os.path.join(ckpt_dir, "checkpoint_best.tar"),
                                        epoch, val_metrics=metrics)
            print(f"[epoch {epoch}] noise_sum={noise_sum:.4f} "
                  f"(best={best['noise_sum']:.4f}@ep{best['epoch']}) "
                  f"clean_gate={'PASS' if clean_ok else 'FAIL'}"
                  f"{'  <- new best' if improved else ''}")
        if ddp:
            import torch.distributed as dist
            dist.barrier()

    if is_main:
        with open(os.path.join(exp_dir, "val_history.json"), "w") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        print("\n===== 训练完成 =====")
        print(f"  总耗时 {(time.time()-t_start)/60:.1f} min")
        if best["epoch"] > 0:
            b = next(h for h in history if h["epoch"] == best["epoch"])
            print(f"  best = epoch {best['epoch']}  "
                  f"EER clean/0db/m5db = "
                  f"{b['val']['clean']['eer']:.4f} / {b['val']['0db']['eer']:.4f} / "
                  f"{b['val']['m5db']['eer']:.4f}")
            print(f"  checkpoint: {os.path.join(ckpt_dir, 'checkpoint_best.tar')}")
        else:
            print("  没有任何 epoch 同时满足 noise_sum 改善 + clean 门限；"
                  "建议降低 lr 或减少 epoch 后重跑。")
        print("  逐 epoch 明细见 val_history.json")


if __name__ == "__main__":
    main()
