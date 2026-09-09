# -*- coding: utf-8 -*-
"""
生成 clean speaker prototype，用于 prototype 初始化 CosineClassifier（8.21 核心改动）。

流程（严格按方案）：
  1. 只加载原始 pretrained ERes2Net-large（strict，绝不使用任何 fine-tune 过的模型）
  2. 语料 = 训练说话人（train_spks.txt，9900 人）的 clean 语句；
     不含验证说话人 / 正式 test 说话人（隔离由 8.20 prepare 断言过，这里再断言一次）
  3. 每个说话人等间隔确定性选 ≤ utts_per_spk 条（来源默认全量 all_wav.scp，
     30k 训练清单每人只有 ~3 条，全量才能拿到 8 条）
  4. 每条语句：居中裁剪 3s（确定性，保证可复现）→ fbank → pretrained embedding
  5. prototype = L2(mean(L2(e_1..e_k)))：先逐条归一化，求均值，再归一化
  6. 落盘 exp/prototypes/prototypes.npz（行序 = spk_ids 字段，训练时按 spk2id 映射）

用法：
  python prepare_prototypes.py --config config_proto.toml --gpu 3
  python prepare_prototypes.py --config config_proto.toml --limit-speakers 50   # 冒烟
  python prepare_prototypes.py ... --force                                      # 强制重提
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_V820 = os.path.join(os.path.dirname(_HERE), "8.20")
if _V820 not in sys.path:
    sys.path.insert(0, _V820)

import toml
from common_utils import (resolve_existing_path, setup_speakerlab, spk_of_utt,
                          uniform_select, extract_embeddings)
from trainer_eres2net_robust import build_eres2net_large, load_pretrained_embedding
from validate_robust import build_fbank


def read_scp_utts(clean_scp, sv_ecapa):
    """解析 all_wav.scp -> {spk: [utt_id,...]}（utt_id 排序，路径解析同 8.20 load_clean_paths）。"""
    train_root = os.path.join(sv_ecapa, "data", "data_for_train")
    per_spk = {}
    n_bad = 0
    with open(clean_scp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id, rel = line.split()[0], line.split()[-1]
            if rel.startswith("data/"):
                rel = rel[len("data/"):]
            cand = os.path.abspath(os.path.join(train_root, rel))
            if not os.path.isfile(cand) and cand.startswith("/data/home/"):
                alt = cand[len("/data"):]
                if os.path.isfile(alt):
                    cand = alt
            if not os.path.isfile(cand):
                n_bad += 1
                continue
            spk = spk_of_utt(utt_id)
            per_spk.setdefault(spk, []).append((utt_id, cand))
    for spk in per_spk:
        per_spk[spk].sort()                      # 确定性：按 utt_id 排序
    if n_bad:
        print(f"[warn] scp 中 {n_bad} 条路径无效已跳过（历史路径兼容未命中）")
    return per_spk


def assemble_selection(selected):
    """
    selected: {spk: [(utt_id, wav_path), ...]} —— 两条来源分支统一的结构。
    返回 (all_utts, paths, utt2spk)：
      all_utts = [(spk, (utt_id, path)), ...]（按 spk 排序，spk 内保持选择顺序）
      paths    = [wav_path, ...]（与 all_utts 同序，全部是存在的文件路径字符串）
      utt2spk  = {utt_id: spk}
    在这里做类型/存在性守卫，让"把元组当路径"这类错误在进 torchaudio 前就报清楚。
    """
    all_utts = [(spk, utt) for spk in sorted(selected) for utt in selected[spk]]
    paths = [utt[1] for _, utt in all_utts]
    utt2spk = {utt[0]: spk for spk, utt in all_utts}
    assert all_utts, "选出的语句为空"
    bad_type = [p for p in paths if not isinstance(p, str)]
    assert not bad_type, f"路径列表混入非字符串（{type(bad_type[0]).__name__}），" \
                         f"检查 selected 的元素必须是 (utt_id, path) 二元组"
    bad_file = [p for p in paths if not os.path.isfile(p)]
    assert not bad_file, f"{len(bad_file)} 条路径不存在，例: {bad_file[:3]}"
    utt_ids = [utt[0] for _, utt in all_utts]
    assert len(set(utt_ids)) == len(utt_ids), "utt_id 有重复（prototype 选择异常）"
    return all_utts, paths, utt2spk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_proto.toml")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--force", action="store_true", help="npz 已存在也重提")
    ap.add_argument("--limit-speakers", type=int, default=0,
                    help="只处理前 N 个说话人（冒烟测试用；产出的 npz 不能用于正式训练）")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="提取 batch（默认取配置；共享卡建议 8）")
    args = ap.parse_args()

    cfg = toml.load(args.config)
    base = os.path.dirname(os.path.abspath(args.config))
    p_cfg, proto_cfg = cfg["paths"], cfg["prototype"]
    sv_ecapa = setup_speakerlab(p_cfg["sv_ecapa"], base)

    if args.gpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    out_dir = os.path.join(base, p_cfg["proto_dir"])
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, "prototypes.npz")
    if os.path.isfile(npz_path) and not args.force:
        print(f"[proto] {npz_path} 已存在，跳过（重提请 --force 或删除该文件）")
        return

    # ---- 说话人隔离（train_spks.txt 由 8.20 prepare 生成，三方零交集已断言过）----
    data820 = resolve_existing_path(p_cfg["data_dir_820"], base)
    train_spks = [l.strip() for l in open(os.path.join(data820, "train_spks.txt")) if l.strip()]
    train_spk_set = set(train_spks)
    val_spks = set(l.strip() for l in open(os.path.join(data820, "val_spks.txt")) if l.strip())
    assert not (train_spk_set & val_spks), "train/val 说话人重叠！"
    print(f"[proto] 训练说话人 {len(train_spk_set)} 人（与验证 {len(val_spks)} 人零交集 ✓）")

    # ---- 选语料 ----
    k = proto_cfg["utts_per_spk"]
    source = proto_cfg.get("source", "full_scp")
    if source == "full_scp":
        clean_scp = resolve_existing_path(p_cfg["clean_scp"], base)
        per_spk = read_scp_utts(clean_scp, sv_ecapa)
        selected = {}
        for spk in sorted(train_spk_set):
            utts = per_spk.get(spk, [])
            selected[spk] = uniform_select(utts, k)      # 不足 k 条则全用
        n_avail = [len(per_spk.get(s, [])) for s in train_spks]
        print(f"[proto] 来源 = 全量 all_wav.scp：训练说话人每人可用 clean "
              f"min/mean/max = {min(n_avail)}/{sum(n_avail)/len(n_avail):.1f}/{max(n_avail)} 条")
    elif source == "train_csv":
        selected = {}
        csv_path = os.path.join(data820, "train_utts.csv")
        with open(csv_path) as f:
            f.readline()
            for line in f:
                utt_id, spk, wav_path = line.strip().split(",")
                selected.setdefault(spk, []).append((utt_id, wav_path))
        for spk in selected:
            selected[spk] = uniform_select(sorted(selected[spk]), k)
        print(f"[proto] 来源 = 30k 训练清单（每人约 {30000/len(train_spk_set):.1f} 条，数据不足）")
    else:
        raise ValueError(f"未知 prototype source: {source}")
    missing = [s for s in train_spk_set if not selected.get(s)]
    assert not missing, f"{len(missing)} 个训练说话人在语料里一条 clean 都没有: {missing[:5]}"

    if args.limit_speakers > 0:                       # 冒烟：只留前 N 人
        keep = sorted(train_spk_set)[:args.limit_speakers]
        selected = {s: selected[s] for s in keep}
        print(f"[proto] 冒烟模式：只处理前 {len(keep)} 个说话人")

    all_utts, paths, utt2spk = assemble_selection(selected)
    print(f"[proto] 共 {len(all_utts)} 条 clean 语句待提取 "
          f"（{len(selected)} 人 × ≤{k} 条，居中裁剪 {proto_cfg['crop_seconds']}s）")

    # ---- 只用 pretrained 模型提取（绝不用 fine-tune 权重）----
    model = build_eres2net_large(embedding_size=cfg["model"]["embedding_size"],
                                 m_channels=cfg["model"]["m_channels"])
    ckpt = p_cfg["eres2net_ckpt"]
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(base, ckpt)
    load_pretrained_embedding(model, ckpt)            # strict：missing=unexpected=0
    model = model.to(device).eval()

    fbank = build_fbank()
    crop_len = int(proto_cfg["crop_seconds"] * 16000)
    bs = args.batch_size or proto_cfg["batch_size"]
    keys, embs = extract_embeddings(model, fbank, paths, device,
                                    batch_size=bs, desc="prototypes", crop_len=crop_len)

    # ---- prototype = L2(mean(L2(e_i))) ----
    emb_dim = embs.shape[1]
    spk_ids = sorted(selected)
    rows_by_spk = {}
    for j, kk in enumerate(keys):                     # 一次遍历建 spk -> 行号表
        rows_by_spk.setdefault(utt2spk[kk], []).append(j)
    assert set(rows_by_spk) == set(spk_ids), \
        f"提取结果的说话人覆盖不全（{len(rows_by_spk)}/{len(spk_ids)}），检查 utt_id->spk 映射"
    protos = np.zeros((len(spk_ids), emb_dim), dtype=np.float32)
    counts = []
    for i, spk in enumerate(spk_ids):
        rows = rows_by_spk[spk]
        e = embs[rows]
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)   # 1) 逐条 L2
        m = e.mean(axis=0)                                          # 2) 均值
        protos[i] = m / (np.linalg.norm(m) + 1e-8)                  # 3) 再 L2
        counts.append(len(rows))

    norms = np.linalg.norm(protos, axis=1)
    print(f"\n[prototype 诊断]")
    print(f"  prototype matrix shape = {protos.shape}  (行序与 spk_ids 一一对应)")
    print(f"  每人使用的 clean 语句数 min/mean/max = "
          f"{min(counts)}/{sum(counts)/len(counts):.2f}/{max(counts)}；总 {sum(counts)} 条")
    print(f"  all finite = {bool(np.isfinite(protos).all())}")
    print(f"  行范数 mean/std = {norms.mean():.6f}/{norms.std():.6f}（应 ≈1）")
    assert np.isfinite(protos).all(), "prototype 含 NaN/Inf，禁止使用"
    assert abs(norms.mean() - 1.0) < 1e-3 and norms.max() < 1.01, "prototype 范数异常"

    np.savez_compressed(npz_path, prototypes=protos,
                        spk_ids=np.array(spk_ids),
                        used_utt_ids=np.array([u[0] for _, u in all_utts]),
                        counts=np.array(counts, dtype=np.int64))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"eres2net_ckpt": ckpt, "source": source, "utts_per_spk": k,
                   "crop_seconds": proto_cfg["crop_seconds"],
                   "num_speakers": len(spk_ids), "num_utts": int(sum(counts)),
                   "limit_speakers": args.limit_speakers}, f, indent=1)
    print(f"[proto] 已保存 {npz_path}（训练时用，另含 used_utt_ids 供探针去重）")


if __name__ == "__main__":
    main()
