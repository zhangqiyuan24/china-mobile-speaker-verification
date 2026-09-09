#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""probe_gate_mem.py — 定位 gate 线 OOM：分阶段打印 CUDA 内存(8.26)

背景: train_feature_gate.py 在 GPU3 上 b30/b27/b18 全 OOM，allocated 均
~10.6-10.8GB 且几乎不随 batch 变 —— 与激活内存规律矛盾。本探针复现其
构建链，逐阶段量内存，并对拍 有/无 GateNet，连跑 10 it 查泄漏。
用法: cd ~/china_mobile/AI/8.26 && python probe_gate_mem.py --gpu 3
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
for _p in (_V820, _V821, _V824):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from feature_gate import GateNet, GatedEmbedding
from trainer_eres2net_robust import (
    build_eres2net_large, load_pretrained_embedding, build_classifier,
    build_aam_margin, apply_freeze, JointModel,
)
from trainer_eres2net_robust_proto import (
    load_prototypes, init_classifier_prototypes,
)
import toml


def mem(tag):
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated() / 2**30
    r = torch.cuda.memory_reserved() / 2**30
    print(f"[mem] {tag:32s} alloc={a:6.2f}GiB reserved={r:6.2f}GiB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--batch", type=int, default=18)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda"

    cfg = toml.load(os.path.join(_HERE, "config_feature_gate.toml"))
    sv = cfg["paths"]["sv_ecapa"]
    CKPT = os.path.join(
        sv, "pretrained/speech_eres2net_large_sv_zh-cn_3dspeaker_16k/"
           "eres2net_large_model.ckpt")
    sys.path.insert(0, os.path.join(sv, "..", "..", ".."))
    print("[probe] 构建 encoder(fp32, 与训练脚本同链)", flush=True)
    encoder = build_eres2net_large()
    load_pretrained_embedding(encoder, CKPT)
    classifier = build_classifier(512, 9900)
    protos, spk_ids, _u = load_prototypes(cfg["paths"]["proto_npz"])
    spk2id = {sid: i for i, sid in enumerate(spk_ids)}
    init_classifier_prototypes(classifier, protos, spk_ids, spk2id)
    apply_freeze(encoder)

    n_spk = 9900
    margin_loss = build_aam_margin(32.0, 0.2).to(device)

    B, T = args.batch, 300
    fb = torch.randn(B, T, 80, device=device)
    lb = torch.randint(0, n_spk, (B,), device=device)
    opt = None

    print("\n[probe] A) 无 GateNet 基线(裸 encoder)", flush=True)
    enc0 = encoder.to(device)
    cls0 = classifier.to(device)
    mem("A-构建后")
    out = enc0(fb); loss = margin_loss(cls0(out), lb)
    mem("A-前向后")
    loss.backward()
    mem("A-反向后")
    opt = torch.optim.AdamW(
        [p for p in list(enc0.parameters()) + list(cls0.parameters())
         if p.requires_grad], lr=2e-5, weight_decay=5e-5)
    opt.step(); opt.zero_grad()
    mem("A-首个 step 后")
    del out, loss
    opt.state_dict().clear() if False else None
    enc0.cpu(); cls0.cpu(); del enc0, cls0, opt
    torch.cuda.empty_cache()
    mem("A-清理后")

    print("\n[probe] B) 含 GateNet(与 train_feature_gate 同构)", flush=True)
    gate = GateNet(n_mels=80)
    encoder2 = build_eres2net_large()
    load_pretrained_embedding(encoder2, CKPT)
    apply_freeze(encoder2)
    classifier2 = build_classifier(512, n_spk)
    init_classifier_prototypes(classifier2, protos, spk_ids, spk2id)
    embd = GatedEmbedding(gate, encoder2).to(device)
    cls2 = classifier2.to(device)
    joint = JointModel(embd, cls2)
    mem("B-构建后")
    enc_params = [p for p in joint.embedding_model.enc.parameters()
                  if p.requires_grad]
    gate_params = list(joint.embedding_model.gate.parameters())
    cls_params = list(joint.classifier.parameters())
    opt = torch.optim.AdamW([
        {"params": enc_params + cls_params, "lr": 2e-5},
        {"params": gate_params, "lr": 1e-4},
    ], weight_decay=5e-5)
    for it in range(10):
        cosine = joint(fb)
        loss = margin_loss(cosine, lb)
        loss.backward()
        opt.step(); opt.zero_grad()
        if it in (0, 1, 4, 9):
            mem(f"B-it{it} step 后")
    print("\n[probe] 完成", flush=True)


if __name__ == "__main__":
    main()
