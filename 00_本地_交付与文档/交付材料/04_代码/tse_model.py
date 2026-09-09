#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tse_model.py — 注册条件化 TSE 前端（8.26 结构线②，VoiceFilter 式 FiLM）

文献锚点: Wang et al. VoiceFilter (ICASSP 2019) — 单说话人噪声(干扰语音)场景,
通用增强无法区分目标与人声噪声; 用注册语音嵌入作条件, 让前端"定向提取目标
说话人"。这是 floor-gap 分解诊断出的 gap 轴对口武器。

设计(继承 8.17 FullSubNet+, 零侵入 8.17 代码):
  - SpeakerFiLM: 注册嵌入(512d) -> MLP -> 3 路 (scale, bias), 每路 F 维
  - 注入点 = 三个 fullband 模型的 channel_attention 之后(fb_input/fbr/fbi)
  - 最后一层零初始化 => scale=1, bias=0, 初始恒等 —— 8.18 预训练行为完整保留,
    条件信息随训练逐步注入
  - FiLM 后的 fb_input 同时流入 sb_model 输入(unfold 复用), 条件信息渗透全链
合规: 输入侧可迁移模块, 对 ECAPA/ReDimNet 同样可作条件源(注册嵌入谁都能提)。
用法: 见 train_tse.py / enhance_tse.py(评测侧)。
注意: import 前需把 8.17/code 加入 sys.path(_setup 已做)。
"""
import torch
import torch.nn as nn
from torch.nn import functional


class SpeakerFiLM(nn.Module):
    """注册嵌入 -> 3 路 per-frequency (scale, bias)。零初始化 = 恒等。"""

    def __init__(self, emb_dim=512, n_freq=257, hidden=256):
        super().__init__()
        self.n_freq = n_freq
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 6 * n_freq),        # 3 路 x (scale, bias)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, emb):
        # emb: (B, emb_dim) -> scale/bias: 各 (B, 3, F)
        h = self.net(emb)
        s, b = h.chunk(2, dim=1)
        B = emb.size(0)
        scale = 1.0 + s.view(B, 3, self.n_freq)
        bias = b.view(B, 3, self.n_freq)
        return scale, bias


class FullSubNetPlusTSE(object):
    """工厂: 延迟到 8.17 路径就绪后再 import 基类并定义子类。"""

    _cls = None

    @classmethod
    def build(cls):
        if cls._cls is not None:
            return cls._cls
        from fullsubnet_plus.model.fullsubnet_plus import FullSubNet_Plus

        class _TSE(FullSubNet_Plus):
            def __init__(self, *a, emb_dim=512, film_hidden=256, **kw):
                super().__init__(*a, **kw)
                assert self.subband_num == 1, "TSE FiLM 只实现了 subband_num=1 路径"
                self.film = SpeakerFiLM(emb_dim, self.num_channels, film_hidden)

            def forward(self, noisy_mag, noisy_real, noisy_imag, spk_emb):
                assert noisy_mag.dim() == 4
                scale, bias = self.film(spk_emb)                     # (B,3,F)
                pad = [0, self.look_ahead]
                noisy_mag = functional.pad(noisy_mag, pad)
                noisy_real = functional.pad(noisy_real, pad)
                noisy_imag = functional.pad(noisy_imag, pad)
                B, C, F, T = noisy_mag.size()
                assert C == 1, "输入须为 [B,1,F,T]"

                fb_input = self.norm(noisy_mag).reshape(B, F, T)
                fb_input = self.channel_attention(fb_input)
                fb_input = fb_input * scale[:, 0].unsqueeze(-1) + bias[:, 0].unsqueeze(-1)
                fb_output = self.fb_model(fb_input).reshape(B, 1, F, T)

                fbr_input = self.norm(noisy_real).reshape(B, F, T)
                fbr_input = self.channel_attention_real(fbr_input)
                fbr_input = fbr_input * scale[:, 1].unsqueeze(-1) + bias[:, 1].unsqueeze(-1)
                fbr_output = self.fb_model_real(fbr_input).reshape(B, 1, F, T)

                fbi_input = self.norm(noisy_imag).reshape(B, F, T)
                fbi_input = self.channel_attention_imag(fbi_input)
                fbi_input = fbi_input * scale[:, 2].unsqueeze(-1) + bias[:, 2].unsqueeze(-1)
                fbi_output = self.fb_model_imag(fbi_input).reshape(B, 1, F, T)

                fb_output_unfolded = self.unfold(fb_output, num_neighbor=self.fb_num_neighbors)\
                    .reshape(B, F, self.fb_num_neighbors * 2 + 1, T)
                fbr_output_unfolded = self.unfold(fbr_output, num_neighbor=self.fb_num_neighbors)\
                    .reshape(B, F, self.fb_num_neighbors * 2 + 1, T)
                fbi_output_unfolded = self.unfold(fbi_output, num_neighbor=self.fb_num_neighbors)\
                    .reshape(B, F, self.fb_num_neighbors * 2 + 1, T)
                noisy_mag_unfolded = self.unfold(
                    fb_input.reshape(B, 1, F, T), num_neighbor=self.sb_num_neighbors)\
                    .reshape(B, F, self.sb_num_neighbors * 2 + 1, T)

                sb_input = torch.cat(
                    [noisy_mag_unfolded, fb_output_unfolded,
                     fbr_output_unfolded, fbi_output_unfolded], dim=2)
                sb_input = self.norm(sb_input)
                # 与 8.18 联训版一致: 不用 drop_band(保完整频轴重建波形)
                sb_input = sb_input.reshape(
                    B * F,
                    (self.sb_num_neighbors * 2 + 1) + 3 * (self.fb_num_neighbors * 2 + 1),
                    T)
                sb_mask = self.sb_model(sb_input)
                sb_mask = sb_mask.reshape(B, F, self.output_size, T)\
                    .permute(0, 2, 1, 3).contiguous()
                return sb_mask[:, :, :, self.look_ahead:]

        cls._cls = _TSE
        return cls._cls
