#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""feature_gate.py — 特征域门控模块（8.26 结构线①，CAM 式）

文献锚点: Yu et al. ICASSP 2021 (Context-Aware Masking) — 在特征层做
context-aware gating 并与 SV 端到端联训, 无需噪净对。
设计(面向 P100 + 可迁移):
  - 4 层空洞卷积(感受野 ~27 帧) → sigmoid 门 (B,T,80)
  - 残差式软门控: y = x * (0.5 + g)   (g∈[0,1] → 系数∈[0.5,1.5],
    保底通过一半, 防训练初期把特征打死)
  - 参数量 ~0.4M, 对三种基线(ECAPA/ERes2Net/ReDimNet)即插即用 —
    它们都吃 80 维 fbank, 本模块不假设编码器任何内部结构。
用法: 与编码器串联联训(见 train_feature_gate.py), 推理时
  emb = encoder(gate(fbank))
"""
import torch
import torch.nn as nn


class GateNet(nn.Module):
    """空洞卷积门控: (B,T,F) → 门控后的 (B,T,F)。F 不写死(卷积按通道)。"""

    def __init__(self, n_mels=80, channels=128, layers=4):
        super().__init__()
        blocks = []
        for i in range(layers):
            d = 2 ** i                      # 空洞率 1,2,4,8 → 感受野 ~27 帧
            blocks += [
                nn.Conv1d(n_mels if i == 0 else channels, channels,
                          kernel_size=3, dilation=d, padding=d),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
            ]
        self.body = nn.Sequential(*blocks)
        self.proj = nn.Conv1d(channels, n_mels, kernel_size=1)
        self.n_mels = n_mels

    def forward(self, x):
        # x: (B, T, F) fbank
        xt = x.transpose(1, 2)              # (B, F, T)
        g = torch.sigmoid(self.proj(self.body(xt)))   # (B, F, T) ∈ [0,1]
        y = xt * (0.5 + g)                  # 残差式软门控
        return y.transpose(1, 2).contiguous()


class GatedEmbedding(nn.Module):
    """gate + 编码器串联成一个 embedding 模型(训练/验证/评测共用一个壳)。

    state_dict 键 = gate.* + enc.* — 官方评测侧用同名壳加载即可,
    避免严格加载失败或静默漏载(strict=False 的坑)。
    """

    def __init__(self, gate, encoder):
        super().__init__()
        self.gate = gate
        self.enc = encoder

    def forward(self, x):
        return self.enc(self.gate(x))
