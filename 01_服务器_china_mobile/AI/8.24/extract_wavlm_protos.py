#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""extract_wavlm_protos.py — 用 WavLM-base-sv 提取 9900 训练说话人的 clean prototype

镜像 8.21 的 prototype 方案(source=full_scp, 每人 8 条 clean, 确定性居中裁剪 3s),
但嵌入空间换成 WavLM。产出 protos.npy [9900,512] + spks.json(与 spk2id 排序一致),
用于 wavlm_finetune.py --proto-init 加速 AAM 对齐(8.21 的 E2-proto 思想)。
"""
import argparse
import csv
import os
import sys
import types

import numpy as np
import torch

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
    import torchaudio
    wav, sr = torchaudio.load(path)
    if wav.ndim == 2:
        wav = wav[0]
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.numpy().astype(np.float32)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--model-dir", default="wavlm-base-sv")
    ap.add_argument("--train-csv", default="../8.22/exp/data_90k/data/train_utts.csv")
    ap.add_argument("--clean-scp",
                    default="../../code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa/data/data_for_train/all_wav.scp")
    ap.add_argument("--scp-base",
                    default="/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa/data",
                    help="scp 相对路径基准; scp 里的 data/ 前缀会被重写为 data_for_train/(目录已改名)")
    ap.add_argument("--utts-per-spk", type=int, default=8)
    ap.add_argument("--out", default="exp/wavlm_protos")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import WavLMForXVector
    model = WavLMForXVector.from_pretrained(args.model_dir).eval().to(device)

    # 目标说话人集合(与微调脚本的 spk2id 排序严格一致: sorted)
    spks = sorted({row["spk"] for row in csv.DictReader(open(args.train_csv))})
    spk2id = {s: i for i, s in enumerate(spks)}
    print(f"[spk] {len(spks)} 训练说话人", flush=True)

    # 解析 all_wav.scp -> {spk: [utt_id]}
    by_spk = {s: [] for s in spks}
    n_scp = 0
    with open(args.clean_scp) as f:
        for ln in f:
            p = ln.split()
            if len(p) < 2:
                continue
            n_scp += 1
            utt_id, rel = p[0], p[1]
            if rel.startswith("data/"):  # scp 写就时目录叫 data/, 现为 data_for_train/
                rel = "data_for_train/" + rel[len("data/"):]
            spk = "_".join(utt_id.split("_")[:3])  # 3D_SPK_00001
            if spk in by_spk:
                by_spk[spk].append((utt_id, rel))
    print(f"[scp] {n_scp} 条, 命中 {sum(len(v) for v in by_spk.values())} 条", flush=True)

    # 每人等间隔选 utts_per_spk 条(与 8.21 同策略), 居中裁剪 3s
    jobs = []  # (spk, path)
    for spk, lst in by_spk.items():
        if not lst:
            raise RuntimeError(f"{spk} 在 scp 中无 clean 语句")
        k = args.utts_per_spk
        idx = np.linspace(0, len(lst) - 1, min(k, len(lst))).astype(int)
        for i in idx:
            utt_id, rel = lst[i]
            jobs.append((spk, os.path.join(args.scp_base, rel)))
    print(f"[jobs] {len(jobs)} 条待提(期望 ~{len(spks) * args.utts_per_spk})", flush=True)

    # 帧数分组批提(全为 3s 居中裁剪, 等长)
    embs = {}  # spk -> [emb]
    B = args.batch_size
    for s in range(0, len(jobs), B):
        chunk = jobs[s:s + B]
        ws = []
        for spk, p in chunk:
            w = load_wav_16k(p)
            if len(w) > CROP:  # 居中
                st = (len(w) - CROP) // 2
                w = w[st:st + CROP]
            else:
                w = np.concatenate([w, np.zeros(CROP - len(w), np.float32)])
            ws.append((w - w.mean()) / np.sqrt(w.var() + 1e-7))
        x = torch.from_numpy(np.stack(ws)).to(device)
        e = model(input_values=x).embeddings
        e = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
        for (spk, _p), v in zip(chunk, e):
            embs.setdefault(spk, []).append(v)
        if (s // B) % 100 == 0:
            print(f"[extract] {s + len(chunk)}/{len(jobs)}", flush=True)

    protos = np.stack([np.mean(embs[spk], axis=0) for spk in spks])
    protos = protos / np.linalg.norm(protos, axis=1, keepdims=True)
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "protos.npy"), protos.astype(np.float32))
    import json
    with open(os.path.join(args.out, "spks.json"), "w") as f:
        json.dump(spks, f)
    print(f"[done] {protos.shape} -> {args.out}/protos.npy", flush=True)


if __name__ == "__main__":
    main()
