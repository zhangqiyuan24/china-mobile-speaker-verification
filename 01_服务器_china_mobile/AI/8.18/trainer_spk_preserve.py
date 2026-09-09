#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Speaker Information Preservation Trainer

结构：
    clean ──▶ DiffFBank ──▶ Frozen ERes2Net ──▶ e_clean (no_grad)
    noisy ──▶ FullSubNet+ ──▶ enhanced ──▶ DiffFBank ──▶ Frozen ERes2Net ──▶ e_enh (梯度回传)

Loss:
    L_se   = cIRM MSE（FullSubNet+ 原 loss）
    L_spk  = 1 - cosine(e_clean, e_enh)
    L_total = L_se + lambda_spk * L_spk
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

# 让 FullSubNet+ 代码可导入
def _setup_paths(fullsubnet_root, sv_ecapa):
    speakerlab_root = os.path.abspath(os.path.join(sv_ecapa, "..", "..", ".."))
    for p in [fullsubnet_root, speakerlab_root]:
        if p not in sys.path:
            sys.path.insert(0, p)


def load_fullsubnet(fullsubnet_root, ckpt_path, model_args, device):
    """加载 pretrained FullSubNet+"""
    from audio_zen.acoustics.feature import stft, istft
    from audio_zen.acoustics.mask import decompress_cIRM, build_complex_ideal_ratio_mask
    from fullsubnet_plus.model.fullsubnet_plus import FullSubNet_Plus

    model = FullSubNet_Plus(
        num_freqs=model_args["num_freqs"],
        look_ahead=model_args["look_ahead"],
        sequence_model=model_args["sequence_model"],
        fb_num_neighbors=model_args["fb_num_neighbors"],
        sb_num_neighbors=model_args["sb_num_neighbors"],
        fb_output_activate_function=model_args["fb_output_activate_function"],
        sb_output_activate_function=model_args["sb_output_activate_function"],
        fb_model_hidden_size=model_args["fb_model_hidden_size"],
        sb_model_hidden_size=model_args["sb_model_hidden_size"],
        channel_attention_model=model_args["channel_attention_model"],
        norm_type=model_args["norm_type"],
        num_groups_in_drop_band=model_args["num_groups_in_drop_band"],
        kersize=model_args["kersize"],
        subband_num=model_args["subband_num"],
        weight_init=False,
        use_drop_band=False,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    return model, stft, istft, decompress_cIRM, build_complex_ideal_ratio_mask


def load_eres2net(sv_ecapa, ckpt_path, device):
    """加载 ERes2Net-large（后续会全冻结）"""
    speakerlab_root = os.path.abspath(os.path.join(sv_ecapa, "..", "..", ".."))
    if speakerlab_root not in sys.path:
        sys.path.insert(0, speakerlab_root)
    from speakerlab.models.eres2net.ERes2Net import ERes2Net

    model = ERes2Net(feat_dim=80, embedding_size=512, m_channels=64)
    state = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    if "embedding_model" in state:
        state = state["embedding_model"]
    model.load_state_dict(state, strict=False)
    # 完全冻结
    for p in model.parameters():
        p.requires_grad = False
    model.eval().to(device)
    return model


def snr_mix(clean, noise, snr, target_len, eps=1e-6):
    """动态混合 clean + noise 到指定 SNR，返回 (noisy, clean)。"""
    if len(clean) >= target_len:
        start = random.randint(0, len(clean) - target_len)
        clean = clean[start:start + target_len]
    else:
        clean = np.pad(clean, (0, target_len - len(clean)))

    if len(noise) >= target_len:
        start = random.randint(0, len(noise) - target_len)
        noise = noise[start:start + target_len]
    else:
        noise = np.tile(noise, int(np.ceil(target_len / len(noise))))[:target_len]

    clean_rms = np.sqrt(np.mean(clean ** 2) + eps)
    noise_rms = np.sqrt(np.mean(noise ** 2) + eps)
    scale = clean_rms / (10 ** (snr / 20.0)) / (noise_rms + eps)
    noisy = clean + noise * scale

    # 防削波
    peak = np.max(np.abs(noisy))
    if peak > 0.99:
        noisy = noisy / peak * 0.99
        clean = clean / peak * 0.99
    return noisy.astype(np.float32), clean.astype(np.float32)


class SpeakerPreserveTrainer:
    def __init__(self, config, fullsubnet_root, sv_ecapa, local_rank=0, distributed=False):
        _setup_paths(fullsubnet_root, sv_ecapa)
        self.local_rank = int(local_rank)
        self.distributed = bool(distributed)
        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        self.sr = config["acoustics"]["sr"]
        self.n_fft = config["acoustics"]["n_fft"]
        self.win_length = config["acoustics"]["win_length"]
        self.hop_length = config["acoustics"]["hop_length"]
        self.lambda_spk = config["training"]["lambda_spk"]

        # 模型
        self.fullsubnet, self.stft_fn, self.istft_fn, self.decompress_cIRM, \
            self.build_cIRM = load_fullsubnet(
                fullsubnet_root, config["paths"]["fullsubnet_ckpt"],
                config["model"], self.device)
        self.eres2net = load_eres2net(
            sv_ecapa, config["paths"]["eres2net_ckpt"], self.device)

        from frontend_spk import DiffFBank
        self.diff_fbank = DiffFBank(
            n_mels=80, sample_rate=self.sr,
            n_fft=512, win_length=400, hop_length=160, mean_nor=True,
        ).to(self.device)

        # DDP 只包装可训练的 FullSubNet+；冻结 ERes2Net 不需要同步梯度
        if self.distributed:
            self.fullsubnet = DDP(
                self.fullsubnet,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        # 优化器：只优化 FullSubNet+ 参数
        trainable = [p for p in self.fullsubnet.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=config["training"]["lr"])

    def forward_enhanced(self, noisy):
        """
        noisy [B, T] -> enhanced [B, T]

        Returns:
            enhanced: reconstructed waveform, gradient-connected to FullSubNet+
            noisy_complex: full STFT [B, F, T]
            cRM_raw: compressed cRM predicted by FullSubNet+ [B, F, T, 2]
            cRM_decompressed: decompressed mask used only for waveform reconstruction
        """
        noisy_complex = self.stft_fn(
            noisy, self.n_fft, self.hop_length, self.win_length
        )  # [B, F, T]

        noisy_mag = noisy_complex.abs().unsqueeze(1)
        noisy_real = noisy_complex.real.unsqueeze(1)
        noisy_imag = noisy_complex.imag.unsqueeze(1)

        # use_drop_band=False => full-frequency output [B, 2, F, T]
        cRM_raw = self.fullsubnet(noisy_mag, noisy_real, noisy_imag)
        cRM_raw = cRM_raw.permute(0, 2, 3, 1)  # [B, F, T, 2]

        # Fail early if model/config accidentally re-enables drop-band.
        assert cRM_raw.shape[1] == noisy_complex.shape[1], (
            f"Frequency mismatch: cRM={tuple(cRM_raw.shape)}, "
            f"noisy_complex={tuple(noisy_complex.shape)}. "
            "Speaker-preservation training requires FullSubNet+ use_drop_band=False."
        )
        assert cRM_raw.shape[2] == noisy_complex.shape[2], (
            f"Time mismatch: cRM={tuple(cRM_raw.shape)}, "
            f"noisy_complex={tuple(noisy_complex.shape)}"
        )

        # IMPORTANT:
        # - cRM_raw is the compressed prediction used for the original cIRM MSE.
        # - decompressed cRM is used only to reconstruct the enhanced waveform.
        cRM_decompressed = self.decompress_cIRM(cRM_raw)

        enhanced_real = (
            cRM_decompressed[..., 0] * noisy_complex.real
            - cRM_decompressed[..., 1] * noisy_complex.imag
        )
        enhanced_imag = (
            cRM_decompressed[..., 1] * noisy_complex.real
            + cRM_decompressed[..., 0] * noisy_complex.imag
        )
        enhanced_complex = torch.stack(
            [enhanced_real, enhanced_imag], dim=-1
        )

        enhanced = self.istft_fn(
            enhanced_complex,
            self.n_fft,
            self.hop_length,
            self.win_length,
            length=noisy.size(-1),
        )
        return enhanced, noisy_complex, cRM_raw, cRM_decompressed

    def train_step(self, noisy, clean):
        """返回 loss 分解 + mean cosine，供日志使用。"""
        noisy = noisy.to(self.device)
        clean = clean.to(self.device)

        self.optimizer.zero_grad()

        # FullSubNet+ 前向 + enhanced 恢复
        enhanced, noisy_complex, cRM_raw, _ = self.forward_enhanced(noisy)

        # L_se: keep the original FullSubNet+ objective.
        # Both sides must be in the COMPRESSED cIRM domain.
        clean_complex = self.stft_fn(
            clean, self.n_fft, self.hop_length, self.win_length
        )
        gt_cIRM = self.build_cIRM(
            noisy_complex, clean_complex
        )  # [B, F, T, 2], compressed cIRM

        assert cRM_raw.shape == gt_cIRM.shape, (
            f"cIRM shape mismatch: pred={tuple(cRM_raw.shape)}, "
            f"target={tuple(gt_cIRM.shape)}"
        )
        L_se = F.mse_loss(cRM_raw, gt_cIRM)

        # speaker consistency
        with torch.no_grad():
            e_clean = self.eres2net(self.diff_fbank(clean))  # [B, 512]，no_grad
        e_enh = self.eres2net(self.diff_fbank(enhanced))  # 梯度必须回传

        # e_clean 无梯度(no_grad 已保证)，e_enh 不 detach，梯度回传到 enhanced -> cRM -> FullSubNet+
        L_spk = (1 - F.cosine_similarity(e_clean.detach(), e_enh, dim=1)).mean()

        L_total = L_se + self.lambda_spk * L_spk
        L_total.backward()

        torch.nn.utils.clip_grad_norm_(self.fullsubnet.parameters(), 5.0)
        self.optimizer.step()

        mean_cos = F.cosine_similarity(e_clean, e_enh, dim=1).mean().item()
        return {
            "L_total": L_total.item(),
            "L_se": L_se.item(),
            "L_spk": L_spk.item(),
            "lambda_L_spk": self.lambda_spk * L_spk.item(),
            "mean_cos": mean_cos,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def validate_step(self, noisy, clean):
        """验证：只算 enhanced 与 clean 的 cosine，不反传。"""
        noisy = noisy.to(self.device)
        clean = clean.to(self.device)
        enhanced, _, _, _ = self.forward_enhanced(noisy)

        e_clean = self.eres2net(self.diff_fbank(clean))
        e_enh = self.eres2net(self.diff_fbank(enhanced))
        cos = F.cosine_similarity(e_clean, e_enh, dim=1)
        return cos.mean().item()

    def save_checkpoint(self, path):
        model = self.fullsubnet.module if isinstance(self.fullsubnet, DDP) else self.fullsubnet
        torch.save({"model": model.state_dict()}, path)
