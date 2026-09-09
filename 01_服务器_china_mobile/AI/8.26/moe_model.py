#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""moe_model.py — 线④ MoE-lite：ERes2Net-layer4 条件专家化（NCMoE 移植, 8.26）

依据 arXiv 2510.18533（ResNet34 stage-2 并联 expert + 噪声分类器温度路由）：
  - ERes2Net-large 的 layer4（3×BasicBlockERes2Net_diff_AFF@512ch）复制 K 份；
  - 3 层小卷积路由器吃 fbank，温度 softmax(γ=0.1) 出门控；
  - 训练=软混合 Σ g·f（梯度全通），测试=硬 argmax（省算力）；
  - UMES：所有 expert 从同一基线 ckpt 的 layer4 复制（通用模型已由基线完成）。
K=3 与训练条件(clean/0dB/-5dB)自然对齐，路由器用条件标签监督。

合规性：纯外挂结构（复制现有段+小路由器），任何基线模型同构可加；
权重从各自基线/改进 ckpt 起步 = 叠加。
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class CondRouter(nn.Module):
    """NCMoE 式噪声分类器：3 层卷积 + 池化 + K 类 logits（输入 fbank (B,1,F,T)）。"""

    def __init__(self, n_expert=3, base_ch=32, in_ch=1, gamma=0.1):
        super().__init__()
        self.gamma = gamma
        ch = base_ch
        layers = [nn.Conv2d(in_ch, ch, 3, stride=2, padding=1),
                  nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
        for _ in range(2):
            layers += [nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
                       nn.BatchNorm2d(ch * 2), nn.ReLU(inplace=True)]
            ch *= 2
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(ch, n_expert)

    def forward(self, x_fbt):
        """x_fbt: (B,T,F) fbank。返回 logits (B,K) 与门控 g (B,K)。"""
        h = self.body(x_fbt.permute(0, 2, 1).unsqueeze(1))   # (B,ch,F',T')
        h = h.mean(dim=(2, 3))
        logits = self.head(h)
        return logits, torch.softmax(logits / self.gamma, dim=1)


class MoEERes2Net(nn.Module):
    """ERes2Net-large + layer4 专家化。forward 与基线一致，仅 out4 换成 MoE。"""

    def __init__(self, base, n_expert=3, gamma=0.1):
        super().__init__()
        self.base = base                      # 已装载权重的 ERes2Net
        self.experts = nn.ModuleList(
            [copy.deepcopy(base.layer4) for _ in range(n_expert)])
        self.router = CondRouter(n_expert=n_expert, gamma=gamma)

    def forward(self, x, return_router=False):
        """x: (B,T,F) fbank。return_router 时额外回路由 logits（训练用）。"""
        b = self.base
        xr = x.permute(0, 2, 1).unsqueeze(1)          # (B,1,F,T)
        logits, g = self.router(x)
        out = F.relu(b.bn1(b.conv1(xr)))
        out1 = b.layer1(out)
        out2 = b.layer2(out1)
        out1_downsample = b.layer1_downsample(out1)
        fuse_out12 = b.fuse_mode12(out2, out1_downsample)
        out3 = b.layer3(out2)
        fuse_out12_downsample = b.layer2_downsample(fuse_out12)
        fuse_out123 = b.fuse_mode123(out3, fuse_out12_downsample)
        if self.training:
            o4 = sum(g[:, i].view(-1, 1, 1, 1) * expert(out3)
                     for i, expert in enumerate(self.experts))
        else:
            sel = g.argmax(dim=1)
            o4 = None
            for i, expert in enumerate(self.experts):
                mask = sel == i
                if not bool(mask.any()):
                    continue
                eo = expert(out3[mask])
                if o4 is None:
                    o4 = torch.zeros(out3.size(0), *eo.shape[1:],
                                     dtype=eo.dtype, device=eo.device)
                o4[mask] = eo
            assert o4 is not None, "路由器全空？"
            # eval 下 BN 走 running stats，分组前向与整批逐样本等价
        fuse_out123_downsample = b.layer3_downsample(fuse_out123)
        fuse_out1234 = b.fuse_mode1234(o4, fuse_out123_downsample)
        stats = b.pool(fuse_out1234)
        embed = b.seg_1(stats)
        if return_router:
            return embed, logits
        return embed
