#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""diag_amp.py — 诊断 WavLM AMP 训练是否每步都被 GradScaler 跳过。"""
import sys
import types

import torch
import torch.nn.functional as F

_fake = types.ModuleType("torch.distributed.fsdp")


class _FSDP:
    pass


_fake.FullyShardedDataParallel = _FSDP
sys.modules["torch.distributed.fsdp"] = _fake
import torch.distributed as _td
_td.fsdp = _fake

from transformers import WavLMForXVector

m = WavLMForXVector.from_pretrained("wavlm-base-sv")
for p in m.wavlm.feature_extractor.parameters():
    p.requires_grad = False
for layer in list(m.wavlm.encoder.layers)[:6]:
    for p in layer.parameters():
        p.requires_grad = False
m.train().cuda()
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-5)
scaler = torch.cuda.amp.GradScaler()
torch.manual_seed(0)
x = torch.randn(18, 48000).cuda() * 0.05
y = torch.randint(0, 9900, (18,)).cuda()
W = torch.randn(9900, 512, device="cuda") * 0.01
for i in range(4):
    with torch.cuda.amp.autocast():
        emb = m(input_values=x).embeddings
        cos = F.linear(F.normalize(emb.float()), F.normalize(W))
        onehot = torch.zeros_like(cos).scatter_(1, y.unsqueeze(1), 1.0)
        logits = 32.0 * (cos * (1 - onehot) + (cos - 0.2 * onehot) * onehot)
        loss = F.cross_entropy(logits, y)
    opt.zero_grad()
    scaler.scale(loss).backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    gn = torch.sqrt(sum(g.float().norm() ** 2 for g in grads)) if grads else None
    bad = bool(torch.isinf(gn).any() or torch.isnan(gn).any()) if gn is not None else None
    scaler.step(opt)
    scaler.update()
    print(f"step{i} loss={loss.item():.3f} scale={scaler.get_scale()} "
          f"emb_absmax={emb.abs().max().item():.1f} "
          f"logit_absmax={logits.abs().max().item():.1f} grad_bad={bad} n_grads={len(grads)}",
          flush=True)
