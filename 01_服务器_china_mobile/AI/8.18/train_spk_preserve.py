#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Speaker Information Preservation fine-tuning 主入口。

流程：
  1. 读取 clean 训练数据 + MUSAN train noise，校验 train/test noise 隔离
  2. frontend consistency sanity test（官方 FBank vs DiffFBank）
  3. 训练循环 + 0dB/-5dB 验证
"""

import os
import sys
import random
import argparse
import numpy as np
import toml
import torch
import torchaudio
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler

from trainer_spk_preserve import SpeakerPreserveTrainer, snr_mix


def resolve(base, p):
    return os.path.abspath(os.path.join(base, p))


def _resolve_existing_path(p, base_dir):
    """解析音频路径，并兼容服务器历史路径 /data/home/... -> /home/...。"""
    p = os.path.expanduser(p.strip())

    candidates = []
    if os.path.isabs(p):
        candidates.append(p)

        # 旧服务器/挂载路径兼容：
        # /data/home/ligangrui/... -> /home/ligangrui/...
        if p.startswith("/data/home/"):
            candidates.append(p[len("/data"):])
    else:
        candidates.append(os.path.join(base_dir, p))

    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "音频文件不存在。尝试过：\n  " + "\n  ".join(candidates)
    )


def load_clean_paths(clean_scp, sv_ecapa):
    """从 scp 文件读取 clean wav，并映射到 data_for_train。"""
    paths = []
    missing = []

    train_root = os.path.join(
        sv_ecapa,
        "data",
        "data_for_train"
    )

    with open(clean_scp) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            p = line.split()[-1]
            p = os.path.expanduser(p)

            if os.path.isabs(p):
                candidate = p
            else:
                # scp 中是 data/train/...，
                # 实际根目录是 data/data_for_train/train/...
                if p.startswith("data/"):
                    p = p[len("data/"):]

                candidate = os.path.join(train_root, p)

            candidate = os.path.abspath(candidate)

            # 兼容旧 /data/home/... 路径
            if not os.path.isfile(candidate) and candidate.startswith("/data/home/"):
                candidate2 = candidate[len("/data"):]
                if os.path.isfile(candidate2):
                    candidate = candidate2

            if not os.path.isfile(candidate):
                missing.append((line_no, candidate))
            else:
                paths.append(candidate)

    if missing:
        preview = "\n".join(
            f"  line {line_no}: {p}"
            for line_no, p in missing[:10]
        )
        raise FileNotFoundError(
            f"clean_scp 中有 {len(missing)} 条无效路径。前10条：\n{preview}"
        )

    return paths



def load_noise_paths(noise_list, sv_ecapa):
    """读取并校验 noise list，同时兼容历史 /data/home 路径。"""
    paths = []
    missing = []

    with open(noise_list) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                paths.append(_resolve_existing_path(line, sv_ecapa))
            except FileNotFoundError:
                missing.append((line_no, line))

    if missing:
        preview = "\n".join(
            f"  line {line_no}: {p}" for line_no, p in missing[:10]
        )
        raise FileNotFoundError(
            f"noise list 中有 {len(missing)} 条无效路径。前10条：\n{preview}"
        )

    return paths


def uniform_select(items, k):
    """等间隔均匀选取 k 条（保序、确定性、无随机种子）。

    scp 按说话人排序，等间隔抽取可覆盖全程的
    说话人/设备/距离/方言分布，而不是偏向文件头部。
    """
    n = len(items)
    if k <= 0 or n <= k:
        return list(items)
    idxs = np.round(np.linspace(0, n - 1, k)).astype(int)
    return [items[i] for i in dict.fromkeys(idxs)]


class MixDataset(torch.utils.data.Dataset):
    """动态混合 clean + noise，返回 (noisy, clean)。"""
    def __init__(self, clean_paths, noise_paths, snr_range, target_len, sr=16000):
        self.clean_paths = clean_paths
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
        return torch.from_numpy(noisy), torch.from_numpy(clean)


def consistency_sanity(config, sv_ecapa, fullsubnet_root):
    """frontend consistency test：官方 FBank vs DiffFBank。"""
    print("\n==== frontend consistency sanity test ====")
    _setup(fullsubnet_root, sv_ecapa)
    from trainer_spk_preserve import load_eres2net
    from frontend_spk import DiffFBank, consistency_test

    eres2net = load_eres2net(sv_ecapa, config["paths"]["eres2net_ckpt"],
                             torch.device(config["training"]["device"]))
    diff_fbank = DiffFBank(n_mels=80, sample_rate=16000, n_fft=512,
                           win_length=400, hop_length=160, mean_nor=True).to(
                               config["training"]["device"])

    clean_paths = load_clean_paths(config["dataset"]["clean_scp"], sv_ecapa)
    random.shuffle(clean_paths)
    wavs = []

    for p in clean_paths:
        if len(wavs) >= 8:
            break
        w, sr = torchaudio.load(p)
        if sr != 16000:
            w = torchaudio.functional.resample(w, sr, 16000)
        if w.numel() == 0:
            continue
        wavs.append(w[0])

    if not wavs:
        raise RuntimeError("frontend consistency test 找不到可用的 clean wav。")

    # pad 到一致长度
    max_len = max(w.shape[0] for w in wavs)
    wavs = [F.pad(w, (0, max_len - w.shape[0])) for w in wavs]
    wav_batch = torch.stack(wavs)
    cos = consistency_test(eres2net, diff_fbank, wav_batch,
                           torch.device(config["training"]["device"]))
    return cos


def _setup(fullsubnet_root, sv_ecapa):
    speakerlab_root = os.path.abspath(os.path.join(sv_ecapa, "..", "..", ".."))
    fullsubnet_pyroot = os.path.join(fullsubnet_root, "speech_enhance")
    for p in [
        fullsubnet_pyroot,
        fullsubnet_root,
        speakerlab_root,
        os.path.dirname(os.path.abspath(__file__)),
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)



def init_distributed():
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        local_rank, rank, world_size = 0, 0, 1
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    return distributed, local_rank, rank, world_size


def reduce_epoch_stats(stats, n, device, distributed):
    keys = list(stats.keys())
    vals = [stats[k] for k in keys] + [float(n)]
    t = torch.tensor(vals, dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    total_n = max(float(t[-1].item()), 1.0)
    return {k: float(t[i].item()) / total_n for i, k in enumerate(keys)}


def cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train_spk_preserve.toml")
    parser.add_argument("--sanity", action="store_true", help="只跑 3 epoch sanity run")
    parser.add_argument("--skip-consistency", action="store_true", help="跳过 frontend consistency test")
    parser.add_argument("--sanity-samples", type=int, default=10000,
                        help="sanity 模式最多使用多少条 clean")
    args = parser.parse_args()

    distributed, local_rank, rank, world_size = init_distributed()
    is_main = (rank == 0)

    config = toml.load(args.config)
    config["training"]["device"] = (
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    base = os.path.dirname(os.path.abspath(args.config))
    sv_ecapa = resolve(base, config["paths"]["sv_ecapa"])
    fullsubnet_root = resolve(base, config["paths"]["fullsubnet_root"])

    _setup(fullsubnet_root, sv_ecapa)

    # ---- 数据准备 ----
    if is_main:
        print("==== 数据准备 ====")
    clean_paths = load_clean_paths(config["dataset"]["clean_scp"], sv_ecapa)
    noise_train = load_noise_paths(config["dataset"]["noise_list"], sv_ecapa)
    noise_test = load_noise_paths(config["dataset"]["noise_test_list"], sv_ecapa)

    # ---- 数据子采样（sanity 与正式训练都均匀抽取，确定性）----
    subset_note = ""
    if args.sanity:
        if 0 < args.sanity_samples < len(clean_paths):
            clean_paths = uniform_select(clean_paths, args.sanity_samples)
            subset_note = f" (sanity, 均匀抽取 {args.sanity_samples} 条)"
    else:
        max_train = int(config["dataset"].get("max_train_samples", 0))
        if 0 < max_train < len(clean_paths):
            clean_paths = uniform_select(clean_paths, max_train)
            subset_note = f" (均匀抽取 {max_train} 条)"

    if is_main:
        print(f"  clean 训练语音: {len(clean_paths)} 条{subset_note}")
        print(f"  train noise: {len(noise_train)} 条, test noise: {len(noise_test)} 条")

    train_set = set(noise_train)
    test_set = set(noise_test)
    overlap = train_set & test_set
    if is_main:
        print(f"  噪声 train/test 重叠: {len(overlap)} 条")
    assert len(overlap) == 0, "train/test noise 存在重叠！"

    sr = config["acoustics"]["sr"]
    target_len = int(config["dataset"]["sub_sample_length"] * sr)

    train_dataset = MixDataset(
        clean_paths, noise_train, config["dataset"]["snr_range"], target_len, sr
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=2026,
        drop_last=True,
    ) if distributed else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config["dataset"]["num_workers"],
        drop_last=True,
        pin_memory=True,
    )

    # ---- frontend consistency test ----
    if not args.skip_consistency and is_main:
        consistency_sanity(config, sv_ecapa, fullsubnet_root)
    if distributed:
        dist.barrier()

    # ---- 构建 trainer ----
    if is_main:
        print("\n==== 构建 trainer ====")
    trainer = SpeakerPreserveTrainer(
        config, fullsubnet_root, sv_ecapa,
        local_rank=local_rank, distributed=distributed
    )
    if is_main:
        os.makedirs(config["paths"]["exp_dir"], exist_ok=True)
    if distributed:
        dist.barrier()

    # ---- 训练 ----
    epochs = config["training"]["num_epochs_sanity"] if args.sanity else config["training"]["epochs"]

    if is_main:
        print(
            f"\n==== 训练 {epochs} epochs "
            f"(world_size={world_size}, per-GPU batch={config['dataset']['batch_size']}, "
            f"global batch={config['dataset']['batch_size'] * world_size}, "
            f"lr={config['training']['lr']}, lambda_spk={config['training']['lambda_spk']}) ===="
        )

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        trainer.fullsubnet.train()
        stats = {
            "L_total": 0.0, "L_se": 0.0, "L_spk": 0.0,
            "lambda_L_spk": 0.0, "mean_cos": 0.0
        }
        n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_main)
        for noisy, clean in pbar:
            r = trainer.train_step(noisy, clean)
            for k in stats:
                stats[k] += r[k]
            n += 1
            if is_main:
                pbar.set_postfix(
                    L_total=f"{r['L_total']:.4f}",
                    L_spk=f"{r['L_spk']:.4f}",
                    cos=f"{r['mean_cos']:.4f}"
                )

        global_stats = reduce_epoch_stats(
            stats, n, trainer.device, distributed
        )

        if is_main:
            print(
                f"[Epoch {epoch}] "
                f"L_total={global_stats['L_total']:.4f} "
                f"L_se={global_stats['L_se']:.4f} "
                f"L_spk={global_stats['L_spk']:.4f} "
                f"lambda*L_spk={global_stats['lambda_L_spk']:.4f} "
                f"mean_cos={global_stats['mean_cos']:.4f} "
                f"lr={trainer.optimizer.param_groups[0]['lr']:.2e}"
            )

        trainer.fullsubnet.eval()
        valid_cos_sum = 0.0
        valid_n = 0
        for i, (noisy, clean) in enumerate(train_loader):
            if i >= 10:
                break
            valid_cos_sum += trainer.validate_step(noisy, clean)
            valid_n += 1

        valid_tensor = torch.tensor(
            [valid_cos_sum, float(valid_n)],
            dtype=torch.float64,
            device=trainer.device
        )
        if distributed:
            dist.all_reduce(valid_tensor, op=dist.ReduceOp.SUM)

        if is_main and valid_tensor[1].item() > 0:
            print(
                f"  [Valid mixed 0/-5 dB] mean_cos="
                f"{valid_tensor[0].item() / valid_tensor[1].item():.4f}"
            )

        if is_main:
            trainer.save_checkpoint(
                os.path.join(
                    config["paths"]["exp_dir"],
                    f"checkpoint_epoch{epoch}.tar"
                )
            )
        if distributed:
            dist.barrier()

    if is_main:
        print("\n训练完成。best checkpoint 由 validate_spk_preserve.py 按 EER_0db+EER_m5db 选择。")
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
