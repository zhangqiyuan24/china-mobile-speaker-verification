# -*- coding: utf-8 -*-
"""
E2-proto 主训练：prototype 初始化 classifier 版多条件鲁棒微调。

与 8.20/train_eres2net_robust.py 的差异（刻意保持最小，做干净对照）：
  1. CosineClassifier 用 clean speaker prototype 初始化（--no-proto 可切回随机初始化，
     同一代码路径下的 random 对照组）
  2. 训练前打印初始化诊断（prototype vs random vs 理论随机水平）
  3. 预留 Option A/B 接口（默认关闭）
数据管线（MultiCondDataset / CondBatchSampler / 1:1:1 动态混噪 / 3s 裁剪）、
冻结范围、AAM、lr/accum/epoch 全部直接复用 8.20，不做任何改动。

验证集说明（重要）：本验证集说话人被 pretrained 原始训练见过
（seen-by-pretrained），clean 绝对值偏乐观（1.25% vs 正式 open-set 5.82%），
只用于 8.20↔8.21 逐 epoch 对照与 checkpoint 选型趋势；
最终结论必须用 evaluate_combo.py 走正式 open-set 协议。

用法（与 8.20 sanity2 完全同参，保证可比）：
  python train_eres2net_robust_proto.py --config config_proto.toml --sanity --gpu 3 \
      --amp --batch-size 12 --accum 8 --num-workers 4 --val-batch-size 8
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import toml
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_V820 = os.path.join(os.path.dirname(_HERE), "8.20")
if _V820 not in sys.path:
    sys.path.insert(0, _V820)

from common_utils import load_wav, center_crop, resolve_existing_path, setup_speakerlab, SR
from train_eres2net_robust import (          # 8.20 数据管线，原样复用
    MultiCondDataset, CondBatchSampler, worker_init_fn, read_train_csv,
)
from trainer_eres2net_robust import (
    build_eres2net_large, load_pretrained_embedding, build_classifier,
    build_aam_margin, apply_freeze, freeze_report, JointModel,
)
from trainer_eres2net_robust_proto import (
    load_prototypes, init_classifier_prototypes, init_diagnostics, ProtoTrainer,
)
from validate_robust import run_validation


# ============================================================
# 初始化诊断探针：前 N 个说话人各 1 条 clean（优先取未参与 prototype 的语句）
# ============================================================

def build_probe(rows, spk2id, used_utt_ids, num_speakers, crop_len, fbank):
    by_spk = {}
    for utt_id, spk, path, _label in rows:
        by_spk.setdefault(spk, []).append((utt_id, path))
    feats, labels, overlap = [], [], 0
    for spk in sorted(spk2id, key=lambda s: spk2id[s])[:num_speakers]:
        pick = next(((u, p) for u, p in by_spk.get(spk, []) if u not in used_utt_ids), None)
        if pick is None:                      # 全部参与过 prototype（仅 train_csv 来源可能出现）
            pick = by_spk[spk][0]
            overlap += 1
        wav = center_crop(load_wav(pick[1]), crop_len)   # 确定性：居中裁剪
        feats.append(fbank(torch.from_numpy(wav)))
        labels.append(spk2id[spk])
    if overlap:
        print(f"[probe] 注意：{overlap}/{num_speakers} 条探针语句与 prototype 语料重叠，"
              f"top1 可能略乐观")
    return torch.stack(feats), torch.tensor(labels, dtype=torch.long)


# ============================================================
# 主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_proto.toml")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--val-batch-size", type=int, default=None)
    ap.add_argument("--accum", type=int, default=None,
                    help="梯度累积步数（共享卡 batch 12 配 --accum 8，有效 batch 96）")
    ap.add_argument("--init-from", default=None, help="从已有 checkpoint 续训（embedding）")
    ap.add_argument("--exp-name", default=None,
                    help="输出子目录名（默认 exp/proto_robust；--no-proto 对照组建议 "
                         "exp/proto_robust_random，避免互相覆盖 checkpoint/val_history）")
    # ---- 本轮新增 ----
    ap.add_argument("--proto", default=None,
                    help="prototype npz 路径（默认 exp/prototypes/prototypes.npz）")
    ap.add_argument("--no-proto", action="store_true",
                    help="对照组：随机初始化 classifier（同一代码路径，仅初始化不同）")
    ap.add_argument("--allow-partial-proto", action="store_true",
                    help="允许 prototype 只覆盖部分说话人（冒烟用；缺的行保持 xavier 随机）")
    # ---- 预留接口（默认关闭）----
    ap.add_argument("--head-warmup-steps", type=int, default=None,
                    help="Option A：前 N 个 optimizer step 只更新 classifier（0=关）")
    ap.add_argument("--head-lr", type=float, default=None,
                    help="Option B：classifier 独立学习率（0=关；启用即突破统一 lr 约束）")
    args = ap.parse_args()

    cfg = toml.load(args.config)
    base = os.path.dirname(os.path.abspath(args.config))
    p_cfg, t_cfg, m_cfg = cfg["paths"], cfg["training"], cfg["model"]
    d_cfg, proto_cfg = cfg["data"], cfg["prototype"]
    sv_ecapa = setup_speakerlab(p_cfg["sv_ecapa"], base)

    epochs = args.epochs or (t_cfg["sanity_epochs"] if args.sanity else t_cfg["epochs"])
    batch_size = args.batch_size or t_cfg["batch_size"]
    lr = args.lr or t_cfg["lr"]
    num_workers = args.num_workers if args.num_workers is not None else t_cfg["num_workers"]
    val_batch_size = args.val_batch_size or t_cfg["val_batch_size"]
    accum = args.accum if args.accum is not None else t_cfg.get("accum", 1)
    head_warmup = args.head_warmup_steps if args.head_warmup_steps is not None \
        else t_cfg.get("head_warmup_steps", 0)
    head_lr = args.head_lr if args.head_lr is not None else t_cfg.get("head_lr", 0.0)
    head_lr = head_lr if head_lr and head_lr > 0 else None

    # ---- 分布式初始化（与 8.20 相同）----
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
    assert torch.cuda.is_available(), "训练需要 GPU"

    # ---- 数据：直接复用 8.20 已生成产物，不重跑 prepare ----
    exp_root = os.path.dirname(os.path.join(base, p_cfg["exp_dir"])) or base
    exp_dir = os.path.join(exp_root, args.exp_name or os.path.basename(p_cfg["exp_dir"]))
    data820 = resolve_existing_path(p_cfg["data_dir_820"], base)
    train_csv = os.path.join(data820, "train_utts.csv")
    noise_txt = os.path.join(data820, "noise_train.txt")
    bench_dir = os.path.join(data820, "val_bench")
    val_trials = os.path.join(data820, "val_trials.txt")
    for need in (train_csv, noise_txt, bench_dir, val_trials):
        assert os.path.exists(need), f"缺 {need}（请先跑 8.20 的 prepare_multicond_data.py）"

    rows, spk2id = read_train_csv(train_csv)
    num_classes = len(spk2id)
    os.makedirs(exp_dir, exist_ok=True)
    if is_main:
        with open(os.path.join(exp_dir, "labels_proto.json"), "w") as f:
            json.dump(spk2id, f, ensure_ascii=False, indent=1)
        print(f"[data] 训练 {len(rows)} 条 / {num_classes} 人（复用 8.20 数据，未重新 prepare）"
              f"；条件=3x -> {3*len(rows)} 样本/epoch")

    noise_paths = [resolve_existing_path(l, sv_ecapa)
                   for l in open(noise_txt) if l.strip()]
    if is_main:
        print(f"[data] train noise pool = {len(noise_paths)} 条")

    # ---- 模型 + 冻结（与 8.20 相同）----
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
        load_pretrained_embedding(model, eres2net_ckpt)
    frozen_mods, trainable_mods = apply_freeze(model)

    # ---- classifier：prototype 初始化（本轮唯一方法改动）----
    classifier = build_classifier(m_cfg["embedding_size"], num_classes)
    margin_loss = build_aam_margin(m_cfg["scale"], m_cfg["margin"])
    used_utt_ids = set()
    if args.no_proto:
        if is_main:
            print("[proto-init] 对照组 --no-proto：classifier 保持 xavier 随机初始化"
                  "（其余与 prototype 轮完全相同）")
    else:
        npz = args.proto or os.path.join(base, p_cfg["proto_dir"], "prototypes.npz")
        assert os.path.isfile(npz), f"缺 {npz}，先运行 prepare_prototypes.py"
        protos, spk_ids, used_utt_ids = load_prototypes(npz)
        # prototype 取 train_spks.txt 全集（9900 人）；30k 训练清单均匀抽取只含其中
        # 9892 人 —— 超集是正常的（多余 prototype 行不用），缺行才是问题。
        no_proto = [s for s in spk2id if s not in set(spk_ids)]
        if no_proto:
            assert args.allow_partial_proto, (
                f"{len(no_proto)} 个训练说话人没有 prototype（如 {no_proto[:3]}）。"
                f"正式训练必须全覆盖：重跑 prepare_prototypes.py；"
                f"确要冒烟请加 --allow-partial-proto（缺的类保持随机初始化）")
        if is_main:
            init_classifier_prototypes(classifier, protos, spk_ids, spk2id,
                                       allow_missing=bool(no_proto))

    if is_main:
        freeze_report(model, classifier)
        proto_tag = "random（对照组）" if args.no_proto else "prototype"
        print(f"[loss] AAM-softmax: margin={m_cfg['margin']} scale={m_cfg['scale']}（固定）"
              f"  amp={args.amp or t_cfg.get('amp', False)}"
              f"  batch_size={batch_size} accum={accum}"
              f"（有效batch={batch_size * accum}）  classifier init = {proto_tag}")
        if head_warmup or head_lr:
            print(f"[loss] 预留接口已启用：head_warmup_steps={head_warmup} "
                  f"head_lr={head_lr or lr}（注意 Option B 突破统一 lr 约束）")

    joint = JointModel(model, classifier)
    if ddp:
        joint = torch.nn.parallel.DistributedDataParallel(joint, device_ids=[local_rank])
    trainer = ProtoTrainer(joint, margin_loss, frozen_mods,
                           lr=lr, weight_decay=t_cfg["weight_decay"], device=device,
                           amp=args.amp or t_cfg.get("amp", False), accum=accum,
                           head_lr=head_lr, head_warmup_steps=head_warmup, verbose=is_main)

    # ---- 数据加载（与 8.20 相同：1:1:1 严格配比 + 每 epoch 换裁剪/噪声）----
    crop_len = int(proto_cfg["crop_seconds"] * SR)
    dataset = MultiCondDataset(rows, noise_paths, crop_len)
    sampler = CondBatchSampler(len(rows), batch_size, seed=d_cfg["seed"])

    def make_loader(epoch):
        sampler.set_epoch(epoch)
        batches = list(iter(sampler))
        if world > 1:
            n = len(batches) // world          # 整除切分，防 DDP 死锁
            batches = batches[rank * n:(rank + 1) * n]
        dataset.worker_seed_base = d_cfg["seed"] + epoch * 1000
        g = torch.Generator()
        g.manual_seed(d_cfg["seed"] + epoch)
        return DataLoader(dataset, batch_sampler=batches,
                          num_workers=num_workers, worker_init_fn=worker_init_fn,
                          generator=g, pin_memory=True)

    # ---- 冻结守门（与 8.20 相同；accum/AMP 会延迟首次更新，最多等 64 个 micro-step）----
    def first_step_guard(loader):
        frozen_p = next(p for n, p in model.named_parameters() if not p.requires_grad)
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

    # ---- 训练前初始化诊断（rank0；不训练任何参数）----
    if is_main:
        from validate_robust import build_fbank
        fbank = build_fbank()
        probe_feats, probe_labels = build_probe(
            rows, spk2id, used_utt_ids, proto_cfg["probe_speakers"], crop_len, fbank)
        probe_joint = trainer.joint                 # 已在 device
        probe_joint.embedding_model.eval()          # 诊断用 eval（BN 统计不动）
        if not args.no_proto:
            init_diagnostics(probe_joint, margin_loss, probe_feats, probe_labels,
                             device, num_classes, tag="prototype",
                             batch_size=val_batch_size)
            # 随机初始化对照组（临时分类器，只读 embedding，不影响训练）
            rand_cls = build_classifier(m_cfg["embedding_size"], num_classes).to(device)
            rand_joint = JointModel(probe_joint.embedding_model, rand_cls)
            init_diagnostics(rand_joint, margin_loss, probe_feats, probe_labels,
                             device, num_classes, tag="random-init(对照)",
                             batch_size=val_batch_size)
        else:
            init_diagnostics(probe_joint, margin_loss, probe_feats, probe_labels,
                             device, num_classes, tag="random-init",
                             batch_size=val_batch_size)

    # ---- epoch0：pretrained 基线验证（seen-by-pretrained，与 8.20 同一 bench 可比）----
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    history = []
    if is_main:
        print("\n===== epoch 0（pretrained 基线）验证 =====")
        print("[val] 提示：本验证集为 seen-by-pretrained（说话人被 pretrained 原始"
              "训练见过），\n      clean 绝对值偏乐观（1.25% vs 正式 open-set 5.82%），"
              "仅用于 8.20↔8.21 对照与选型；\n      正式结论以 evaluate_combo.py 的 "
              "open-set 协议为准。")
        base_metrics = run_validation(model, bench_dir, val_trials, device,
                                      batch_size=val_batch_size)
        clean_gate = base_metrics["clean"]["eer"] + t_cfg["clean_degradation_max"]
        print(f"[gate] clean 门限 = epoch0 clean {base_metrics['clean']['eer']:.4f}% "
              f"+ {t_cfg['clean_degradation_max']}pp = {clean_gate:.4f}%"
              f"（seen-by-pretrained 集上的选型门限，不是 open-set 真实退化）")
        history.append({"epoch": 0, "phase": "pretrained", "val": base_metrics})

    if ddp:
        import torch.distributed as dist
        dist.barrier()                       # 等 rank0 探针/epoch0 验证完成

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
        trainer.flush()

        metrics = None
        if is_main:
            metrics = run_validation(model, bench_dir, val_trials, device,
                                     batch_size=val_batch_size)
            noise_sum = metrics["0db"]["eer"] + metrics["m5db"]["eer"]
            clean_ok = bool(metrics["clean"]["eer"] <= clean_gate)
            rec = {"epoch": epoch, "val": metrics,
                   "noise_sum": noise_sum, "clean_gate_ok": clean_ok,
                   "train_loss": sums["loss"] / max(1, sums["n"]),
                   "train_acc": sums["acc"] / max(1, sums["n"]),
                   "proto_init": not args.no_proto}
            history.append(rec)
            trainer.save_checkpoint(
                os.path.join(ckpt_dir, f"checkpoint_epoch{epoch}.tar"),
                epoch, val_metrics=metrics)
            improved = noise_sum < best["noise_sum"] and clean_ok
            if improved:
                best = {"noise_sum": noise_sum, "epoch": epoch, "clean_ok": clean_ok}
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
            print(f"  best = epoch {best['epoch']}  EER clean/0db/m5db = "
                  f"{b['val']['clean']['eer']:.4f} / {b['val']['0db']['eer']:.4f} / "
                  f"{b['val']['m5db']['eer']:.4f}")
            print(f"  checkpoint: {os.path.join(ckpt_dir, 'checkpoint_best.tar')}")
        else:
            print("  没有 epoch 同时满足 noise_sum 改善 + clean 门限（seen-by-pretrained 门限）")

        # ---- epoch1 对照表：本轮 vs 8.20 random init（sanity 的核心判读）----
        ep1 = next((h for h in history if h["epoch"] == 1), None)
        rnd = cfg.get("compare", {}).get("random_init_ep1", {})
        if ep1 and rnd:
            this_tag = "prototype(8.21)" if not args.no_proto else "random(8.21)"
            print(f"\n===== epoch1 对照（同数据同超参，仅初始化不同） =====")
            print(f"  {'':22s}{'clean':>8s}{'0dB':>8s}{'-5dB':>8s}"
                  f"{'loss':>10s}{'acc':>8s}")
            print(f"  {'random init (8.20)':<22s}{rnd.get('clean_eer', float('nan')):>8.2f}"
                  f"{rnd.get('noise0_eer', float('nan')):>8.2f}"
                  f"{rnd.get('noise5_eer', float('nan')):>8.2f}"
                  f"{rnd.get('train_loss', float('nan')):>10.4f}"
                  f"{rnd.get('train_acc', float('nan')):>8.4f}")
            print(f"  {this_tag:<22s}{ep1['val']['clean']['eer']:>8.2f}"
                  f"{ep1['val']['0db']['eer']:>8.2f}{ep1['val']['m5db']['eer']:>8.2f}"
                  f"{ep1['train_loss']:>10.4f}{ep1['train_acc']:>8.4f}")
            print("  期望：clean 不再跳到 4%+，train_acc 明显高于 0.017，"
                  "0dB/-5dB 初始退化明显减轻。")


if __name__ == "__main__":
    main()
