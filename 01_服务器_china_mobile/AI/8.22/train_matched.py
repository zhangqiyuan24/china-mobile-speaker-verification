# -*- coding: utf-8 -*-
"""
E2-matched 主训练：五条件混合域鲁棒微调（8.22，对照 8.21）。

与 8.21/train_eres2net_robust_proto.py 的差异（刻意保持最小）：
  1. 条件混合 {clean, raw0, raw-5} 均匀 1/3 -> {clean, raw0, raw-5, enh0, enh-5} 均匀 1/5
     - raw 三路：与 8.21 完全相同的动态混噪（每 epoch 重抽裁剪/噪声，不落盘）
     - enh 两路：读 prepare_enh_data.py 预生成的固定增强语料（部署前端生成）
  2. 每 epoch 双 bench 验证：raw bench（clean/0db/m5db，与 8.21 逐 epoch 可比）
     + enh bench（enh0db/enhm5db，选点主路——部署走的就是前端->后端这条路）
  3. best 选择：enh_noise_sum = EER_enh0 + EER_enh5 最小 + clean 门限（raw bench clean）
  4. batch 语义变化：五条件均分 -> batch 必须是 5 的倍数；共享卡配方 batch10×accum10
     （有效 100，与 8.21 的 96 同规模）；每 epoch 样本 90k -> 150k，更新数 937 -> 1500

其余一切与 8.21 相同：pretrained 起点 / prototype 初始化（复用同一份 npz）/
lr=2e-5 / 冻结范围 / AAM margin0.2 scale32 / 8 epoch / 冻结守门 / 初始化诊断。

启动守卫：enh 语料 manifest 必须 complete 且前端 ckpt 与 config 一致——
换前端必须重跑 prepare_enh_data.py，否则这里直接拒绝。

用法：
  # 共享卡（GPU 3，与 8.21 同款配方，batch 10×accum 10）：
  python train_matched.py --config config_matched.toml --gpu 3 \
      --amp --batch-size 10 --accum 10 --num-workers 4 --val-batch-size 8
  # sanity 3 epoch（部分语料冒烟需 --allow-partial-corpus）：
  python train_matched.py --config config_matched.toml --gpu 3 --sanity \
      --amp --batch-size 10 --accum 10 --allow-partial-corpus
"""

import os
import sys
import json
import glob
import time
import argparse
import numpy as np
import torch
import toml
from torch.utils.data import Dataset, DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_V820 = os.path.join(os.path.dirname(_HERE), "8.20")
_V821 = os.path.join(os.path.dirname(_HERE), "8.21")
for _p in (_V820, _V821):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common_utils import (
    load_wav, random_crop, mix_noise, load_trials, eer_minDCF,
    extract_embeddings, resolve_existing_path, setup_speakerlab, SR,
)
from train_eres2net_robust import worker_init_fn, read_train_csv
from trainer_eres2net_robust import (
    build_eres2net_large, load_pretrained_embedding, build_classifier,
    build_aam_margin, apply_freeze, freeze_report, JointModel,
)
from trainer_eres2net_robust_proto import (
    load_prototypes, init_classifier_prototypes, init_diagnostics, ProtoTrainer,
)
from train_eres2net_robust_proto import build_probe
from validate_robust import run_validation, build_fbank


# ============================================================
# 数据：五条件数据集 + 五路均分采样器
# ============================================================

class MatchedDataset(Dataset):
    """长度 5N：idx -> (base=idx//5, cond=idx%5)。
    cond 0/1/2 = clean/0dB/-5dB（动态，与 8.21 完全同实现）
    cond 3/4   = enh0dB/enh-5dB（读预生成固定语料；恰好 3s，crop 兜底原样返回）"""

    def __init__(self, rows, noise_paths, enh_map, crop_len, snrs=(0.0, -5.0)):
        self.rows = rows                    # [(utt_id, spk, path, label)]
        self.noise_paths = noise_paths
        self.enh_map = enh_map              # {utt_id: {"snr0": path, "snrm5": path}}
        self.crop_len = crop_len
        self.snrs = snrs
        self.fbank = build_fbank()

    def __len__(self):
        return 5 * len(self.rows)

    def __getitem__(self, idx):
        base, cond = idx // 5, idx % 5
        utt_id, _spk, path, label = self.rows[base]
        if cond <= 2:                                    # clean / raw0 / raw-5
            wav3 = random_crop(load_wav(path), self.crop_len, np.random)
            if cond > 0:
                noise = load_wav(self.noise_paths[np.random.randint(len(self.noise_paths))])
                if len(noise) > self.crop_len:
                    start = np.random.randint(0, len(noise) - self.crop_len)
                else:
                    start = 0
                wav3, _ = mix_noise(wav3, noise, self.snrs[cond - 1], start)
        else:                                            # enh0 / enh-5（固定语料）
            wav3 = random_crop(load_wav(self.enh_map[utt_id]["snr0" if cond == 3 else "snrm5"]),
                               self.crop_len, np.random)
        feat = self.fbank(torch.from_numpy(wav3))        # [T, 80], mean-nor
        return feat, label


class MatchedBatchSampler:
    """每个批次严格 5 路均分（batch_size 必须是 5 的倍数）。每 epoch 打乱各条件内部顺序。"""

    def __init__(self, num_utts, batch_size, n_cond=5, seed=2026):
        assert batch_size % n_cond == 0, \
            f"batch_size 必须是 {n_cond} 的倍数，当前 {batch_size}"
        self.per_cond = batch_size // n_cond
        self.n_cond = n_cond
        self.num_utts = num_utts
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _cond_indices(self, cond, rng):
        idxs = np.arange(cond, self.n_cond * self.num_utts, self.n_cond)
        rng.shuffle(idxs)
        return idxs

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        lists = [self._cond_indices(c, rng) for c in range(self.n_cond)]
        n_batches = min(len(l) for l in lists) // self.per_cond
        for b in range(n_batches):
            batch = []
            for c in range(self.n_cond):
                s = b * self.per_cond
                batch.extend(lists[c][s:s + self.per_cond].tolist())
            yield batch

    def __len__(self):
        return self.num_utts // self.per_cond


# ============================================================
# 验证：enh bench（增强测试侧 × clean 注册，同一份 val_trials）
# ============================================================

def index_wavs(root):
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith(".wav"):
                out[os.path.splitext(fn)[0]] = os.path.join(dirpath, fn)
    return out


def _score_trials(enroll_map, emb_map, trials):
    scores, labels = [], []
    for e, t, lab in trials:
        if e in enroll_map and t in emb_map:
            s = float(np.dot(enroll_map[e], emb_map[t]) /
                      (np.linalg.norm(enroll_map[e]) * np.linalg.norm(emb_map[t]) + 1e-8))
            scores.append(s)
            labels.append(lab)
    return eer_minDCF(np.array(scores), np.array(labels)), len(labels)


def run_validation_enh(embedding_model, bench_dir, enh_bench_dir, val_trials_path,
                       device, batch_size=64, verbose=True):
    """返回 {cond: {"eer","mindcf"}}，cond in {enh0db, enhm5db}。
    注册侧 = raw bench 的 clean enroll（与 run_validation 完全同一批人/文件）。"""
    fbank = build_fbank()
    embedding_model.eval()
    enroll_files = sorted(glob.glob(os.path.join(bench_dir, "enroll", "*.wav")))
    assert enroll_files, f"验证 bench 为空: {bench_dir}/enroll"
    ek, ee = extract_embeddings(embedding_model, fbank, enroll_files, device,
                                batch_size, desc="val-enroll")
    enroll_map = dict(zip(ek, ee))
    trials = load_trials(val_trials_path)
    results = {}
    for cond, sub in (("enh0db", "test_0db"), ("enhm5db", "test_m5db")):
        files = sorted(glob.glob(os.path.join(enh_bench_dir, sub, "*.wav")))
        assert files, f"增强 bench 为空: {enh_bench_dir}/{sub}（先跑 prepare_enh_data.py --bench）"
        tk, te = extract_embeddings(embedding_model, fbank, files, device,
                                    batch_size, desc=f"val-{cond}")
        (eer, mindcf), n = _score_trials(enroll_map, dict(zip(tk, te)), trials)
        results[cond] = {"eer": eer, "mindcf": mindcf}
        if verbose:
            print(f"   [val] {cond:>6}: EER = {eer:.4f}%  minDCF = {mindcf:.4f}  (trials={n})")
    return results


# ============================================================
# enh 语料加载 + 启动守卫
# ============================================================

def load_enh_corpus(cfg, base, rows, allow_partial):
    """读 manifest + 逐条索引，返回 (enh_map, meta)。守卫不通过直接 assert 报错。"""
    p_cfg = cfg["paths"]
    f_cfg = cfg["frontend"]
    corpus_root = p_cfg["enh_corpus_dir"] if os.path.isabs(p_cfg["enh_corpus_dir"]) \
        else os.path.join(base, p_cfg["enh_corpus_dir"])
    mani_p = os.path.join(corpus_root, "manifest.json")
    assert os.path.isfile(mani_p), \
        f"缺 {mani_p}：先跑 python prepare_enh_data.py --config config_matched.toml --all"

    mani = json.load(open(mani_p))
    ckpt_cfg = f_cfg["ckpt"] if os.path.isabs(f_cfg["ckpt"]) else os.path.join(base, f_cfg["ckpt"])
    ckpt_cfg = resolve_existing_path(ckpt_cfg, base)
    assert os.path.abspath(mani["frontend_ckpt"]) == os.path.abspath(ckpt_cfg), (
        f"enh 语料是别的 ckpt 生成的（manifest: {mani['frontend_ckpt']}，"
        f"config: {ckpt_cfg}）。换前端必须重跑 prepare_enh_data.py，训练/推理前端必须一致")
    if not mani.get("complete"):
        assert allow_partial, \
            ("manifest.complete=false（--limit 冒烟语料）。正式训练请重跑 "
             "prepare_enh_data.py --all（已完成的块会自动跳过）；确要冒烟加 --allow-partial-corpus")

    tags = mani["snr_tags"]
    enh_map = {}
    for tag in tags:
        tag_root = os.path.join(mani["enh_root"], tag)
        assert os.path.isdir(tag_root), \
            f"缺 {tag_root}（先跑 prepare_enh_data.py --enhance）"
        n = 0
        for d in sorted(os.listdir(tag_root)):
            for uid, p in index_wavs(os.path.join(tag_root, d)).items():
                enh_map.setdefault(uid, {})[tag] = p
                n += 1
        print(f"[corpus] enhanced/{tag}: {n} 条（前端={os.path.basename(mani['frontend_ckpt'])}）")

    full = [r for r in rows
            if r[0] in enh_map and all(t in enh_map[r[0]] for t in tags)]
    if len(full) < len(rows):
        assert allow_partial, \
            (f"enh 语料只覆盖 {len(full)}/{len(rows)} 条训练语句，"
             f"正式训练要求全覆盖：重跑 prepare_enh_data.py --all")
        print(f"[corpus] 冒烟模式：覆盖 {len(full)}/{len(rows)}，训练清单截断到覆盖集")
    return enh_map, full, mani


# ============================================================
# 主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_matched.toml")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--val-batch-size", type=int, default=None)
    ap.add_argument("--accum", type=int, default=None)
    ap.add_argument("--init-from", default=None, help="从已有 checkpoint 续训（embedding）")
    ap.add_argument("--no-proto", action="store_true",
                    help="对照组：随机初始化 classifier（默认 prototype，与 8.21 同源）")
    ap.add_argument("--allow-partial-corpus", action="store_true",
                    help="允许 enh 语料不完整（冒烟；正式训练必须完整）")
    args = ap.parse_args()

    cfg = toml.load(args.config)
    base = os.path.dirname(os.path.abspath(args.config))
    p_cfg, t_cfg, m_cfg = cfg["paths"], cfg["training"], cfg["model"]
    d_cfg = cfg["data"]
    sv_ecapa = setup_speakerlab(p_cfg["sv_ecapa"], base)

    epochs = args.epochs or (t_cfg["sanity_epochs"] if args.sanity else t_cfg["epochs"])
    batch_size = args.batch_size or t_cfg["batch_size"]
    lr = args.lr or t_cfg["lr"]
    num_workers = args.num_workers if args.num_workers is not None else t_cfg["num_workers"]
    val_batch_size = args.val_batch_size or t_cfg["val_batch_size"]
    accum = args.accum if args.accum is not None else t_cfg.get("accum", 1)

    # ---- DDP（与 8.20/8.21 相同）----
    ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if ddp:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
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

    # ---- 数据：8.20 产物 + 本目录 enh 语料 ----
    exp_dir = os.path.join(base, p_cfg["exp_dir"])
    data820 = resolve_existing_path(p_cfg["data_dir_820"], base)
    train_csv = os.path.join(data820, "train_utts.csv")
    bench_dir = os.path.join(data820, "val_bench")
    val_trials = os.path.join(data820, "val_trials.txt")
    for need in (train_csv, bench_dir, val_trials):
        assert os.path.exists(need), f"缺 {need}（先跑 8.20 的 prepare_multicond_data.py）"
    enh_bench_dir = p_cfg["enh_bench_dir"] if os.path.isabs(p_cfg["enh_bench_dir"]) \
        else os.path.join(base, p_cfg["enh_bench_dir"])

    rows, spk2id = read_train_csv(train_csv)
    num_classes = len(spk2id)
    enh_map, rows, mani = load_enh_corpus(cfg, base, rows, args.allow_partial_corpus)
    spk2id = {s: i for s, i in spk2id.items() if i in {r[3] for r in rows}}
    os.makedirs(exp_dir, exist_ok=True)
    if is_main:
        json.dump(spk2id, open(os.path.join(exp_dir, "labels_matched.json"), "w"),
                  ensure_ascii=False, indent=1)
        print(f"[data] 训练 {len(rows)} 条 / {len(spk2id)} 人；条件=5x -> "
              f"{5 * len(rows)} 样本/epoch（clean/raw0/raw-5/enh0/enh-5 各 1/5）")

    noise_paths = [resolve_existing_path(l, sv_ecapa)
                   for l in open(os.path.join(data820, "noise_train.txt")) if l.strip()]
    if is_main:
        print(f"[data] train noise pool = {len(noise_paths)} 条（raw 分支动态混噪，与 8.21 同池同法）")

    # ---- 模型 + 冻结（与 8.21 相同）----
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
    frozen_mods, _trainable = apply_freeze(model)

    # ---- classifier：prototype 初始化（复用 8.21 的 npz，同一份）----
    classifier = build_classifier(m_cfg["embedding_size"], num_classes)
    margin_loss = build_aam_margin(m_cfg["scale"], m_cfg["margin"])
    used_utt_ids = set()
    if args.no_proto:
        if is_main:
            print("[proto-init] 对照组 --no-proto：classifier 保持 xavier 随机初始化")
    else:
        npz = p_cfg["prototypes_npz"] if os.path.isabs(p_cfg["prototypes_npz"]) \
            else os.path.join(base, p_cfg["prototypes_npz"])
        assert os.path.isfile(npz), f"缺 {npz}（8.21 的 prepare_prototypes.py 产物）"
        protos, spk_ids, used_utt_ids = load_prototypes(npz)
        missing = [s for s in spk2id if s not in set(spk_ids)]
        assert not missing, (
            f"{len(missing)} 个训练说话人没有 prototype（如 {missing[:3]}）；"
            f"prototype 覆盖 train_spks.txt 全集，正常不应缺行")
        if is_main:
            init_classifier_prototypes(classifier, protos, spk_ids, spk2id,
                                       allow_missing=False)
    if is_main:
        freeze_report(model, classifier)
        print(f"[loss] AAM-softmax: margin={m_cfg['margin']} scale={m_cfg['scale']}（固定）"
              f"  amp={args.amp or t_cfg.get('amp', False)}"
              f"  batch_size={batch_size} accum={accum}"
              f"（有效batch={batch_size * accum}，五路各 {batch_size // 5}）")

    joint = JointModel(model, classifier)
    if ddp:
        joint = torch.nn.parallel.DistributedDataParallel(joint, device_ids=[local_rank])
    trainer = ProtoTrainer(joint, margin_loss, frozen_mods,
                           lr=lr, weight_decay=t_cfg["weight_decay"], device=device,
                           amp=args.amp or t_cfg.get("amp", False), accum=accum,
                           verbose=is_main)

    # ---- 数据加载（五路严格均分 + 每 epoch 换裁剪/噪声；enh 路固定语料）----
    crop_len = int(cfg["enh_corpus"]["crop_seconds"] * SR)
    dataset = MatchedDataset(rows, noise_paths, enh_map, crop_len)
    sampler = MatchedBatchSampler(len(rows), batch_size, seed=d_cfg["seed"])

    def make_loader(epoch):
        sampler.set_epoch(epoch)
        batches = list(iter(sampler))
        if world > 1:
            n = len(batches) // world
            batches = batches[rank * n:(rank + 1) * n]
        dataset.worker_seed_base = d_cfg["seed"] + epoch * 1000
        g = torch.Generator()
        g.manual_seed(d_cfg["seed"] + epoch)
        return DataLoader(dataset, batch_sampler=batches,
                          num_workers=num_workers, worker_init_fn=worker_init_fn,
                          generator=g, pin_memory=True)

    # ---- 冻结守门（与 8.21 相同）----
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
            print(f"[guard] 第 {tried} 个 micro-step 后参数更新生效: loss={loss:.4f} ✓"
                  f"（含 accum 累积与/或 AMP 溢出跳过均正常）")

    # ---- 训练前初始化诊断（rank0，不训练任何参数；与 8.21 同一探针协议）----
    if is_main:
        probe_feats, probe_labels = build_probe(
            rows, spk2id, used_utt_ids, cfg["prototype"]["probe_speakers"]
            if "prototype" in cfg else 256, crop_len, build_fbank())
        probe_joint = trainer.joint
        probe_joint.embedding_model.eval()
        init_diagnostics(probe_joint, margin_loss, probe_feats, probe_labels,
                         device, num_classes, tag="prototype" if not args.no_proto else "random-init",
                         batch_size=val_batch_size)

    # ---- epoch0：pretrained 基线，双 bench ----
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    history = []
    if is_main:
        print("\n===== epoch 0（pretrained 基线）双 bench 验证 =====")
        print("[val] 提示：raw bench 为 seen-by-pretrained（clean 绝对值偏乐观），"
              "仅用于逐 epoch 对照与选型；\n      正式结论以 evaluate_combo 的 open-set 协议为准。")
        raw0 = run_validation(model, bench_dir, val_trials, device, val_batch_size)
        enh0 = run_validation_enh(model, bench_dir, enh_bench_dir, val_trials,
                                  device, val_batch_size)
        clean_gate = raw0["clean"]["eer"] + t_cfg["clean_degradation_max"]
        print(f"[gate] clean 门限 = epoch0 raw-bench clean {raw0['clean']['eer']:.4f}% "
              f"+ {t_cfg['clean_degradation_max']}pp = {clean_gate:.4f}%")
        history.append({"epoch": 0, "phase": "pretrained",
                        "val_raw": raw0, "val_enh": enh0})

    if ddp:
        import torch.distributed as dist
        dist.barrier()

    best = {"enh_noise_sum": float("inf"), "epoch": -1}
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

        if is_main:
            raw_val = run_validation(model, bench_dir, val_trials, device, val_batch_size)
            enh_val = run_validation_enh(model, bench_dir, enh_bench_dir, val_trials,
                                         device, val_batch_size)
            enh_sum = enh_val["enh0db"]["eer"] + enh_val["enhm5db"]["eer"]
            raw_sum = raw_val["0db"]["eer"] + raw_val["m5db"]["eer"]
            clean_ok = bool(raw_val["clean"]["eer"] <= clean_gate)
            rec = {"epoch": epoch, "val_raw": raw_val, "val_enh": enh_val,
                   "enh_noise_sum": enh_sum, "raw_noise_sum": raw_sum,
                   "clean_gate_ok": clean_ok,
                   "train_loss": sums["loss"] / max(1, sums["n"]),
                   "train_acc": sums["acc"] / max(1, sums["n"]),
                   "proto_init": not args.no_proto}
            history.append(rec)
            merged = dict(raw_val)
            merged.update(enh_val)
            trainer.save_checkpoint(os.path.join(ckpt_dir, f"checkpoint_epoch{epoch}.tar"),
                                    epoch, val_metrics=merged)
            improved = enh_sum < best["enh_noise_sum"] and clean_ok
            if improved:
                best = {"enh_noise_sum": enh_sum, "epoch": epoch}
                trainer.save_checkpoint(os.path.join(ckpt_dir, "checkpoint_best.tar"),
                                        epoch, val_metrics=merged)
            print(f"[epoch {epoch}] enh_sum={enh_sum:.4f} "
                  f"(best={best['enh_noise_sum']:.4f}@ep{best['epoch']})  "
                  f"raw_sum={raw_sum:.4f}  clean_gate={'PASS' if clean_ok else 'FAIL'}"
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
            print(f"  best = epoch {best['epoch']}（按 enh 主路 noise_sum 选点）")
            print(f"    enh bench EER 0dB/-5dB = {b['val_enh']['enh0db']['eer']:.4f} / "
                  f"{b['val_enh']['enhm5db']['eer']:.4f}  (sum={b['enh_noise_sum']:.4f})")
            print(f"    raw bench EER clean/0dB/-5dB = {b['val_raw']['clean']['eer']:.4f} / "
                  f"{b['val_raw']['0db']['eer']:.4f} / {b['val_raw']['m5db']['eer']:.4f}")
            print(f"  checkpoint: {os.path.join(ckpt_dir, 'checkpoint_best.tar')}")
        else:
            print("  没有 epoch 同时满足 enh_sum 改善 + clean 门限")
        print("  下一步：SKIP_PREP=1 SKIP_TRAIN=1 bash run_train.sh（正式 open-set 评测）"
              "或直接 bash run_train.sh 一条龙")


if __name__ == "__main__":
    main()
