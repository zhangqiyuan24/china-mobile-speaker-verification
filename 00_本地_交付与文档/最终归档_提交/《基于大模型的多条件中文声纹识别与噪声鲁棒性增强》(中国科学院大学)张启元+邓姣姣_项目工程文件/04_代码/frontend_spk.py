#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
纯 PyTorch 可微 80 维 log-mel frontend，参数对齐 3D-Speaker 官方 FBank。

官方 FBank 用 torchaudio.compliance.kaldi.fbank（dither=0），内部依赖 torch.stft，
理论上可微但 torch 1.10 的 stft autograd 有不确定性。这里用 torch.stft(return_complex=True)
+ melscale_fbanks 显式实现，保证梯度能从 ERes2Net 一路回传到 FullSubNet+。

同时提供 frontend consistency test：同批 clean wav 分别过官方 FBank 和本 diff FBank，
比较两者经 ERes2Net 得到的 embedding 的 cosine similarity。
"""

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torchaudio.functional import melscale_fbanks


class DiffFBank(nn.Module):
    """可微 log-mel frontend，参数对齐官方 FBank(80, sr=16000, mean_nor=True)。"""

    def __init__(
        self,
        n_mels: int = 80,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,   # 25ms
        hop_length: int = 160,   # 10ms
        f_min: float = 20.0,
        f_max: float = 8000.0,
        mean_nor: bool = True,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.mean_nor = mean_nor

        # mel filterbank: melscale_fbanks 返回 [n_freqs, n_mels]，转置成 [n_mels, n_freqs]
        fb = melscale_fbanks(
            n_fft // 2 + 1, f_min, f_max, n_mels, sample_rate,
            norm=None, mel_scale="htk",
        )
        self.register_buffer("fb", fb.t().contiguous())

        # povey window = hann_window(periodic=False) ** 0.85（与 torchaudio kaldi 一致）
        window = torch.hann_window(win_length, periodic=False).pow(0.85)
        self.register_buffer("window", window)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav: [B, T] 波形
        Returns:
            feat: [B, T', n_mels] log-mel 特征（可微）
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        # STFT -> 复数谱 [B, F, T]
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )

        # 功率谱 -> mel -> log
        power = spec.abs() ** 2  # [B, F, T]
        mel = torch.matmul(self.fb, power)  # [B, n_mels, T]
        mel = torch.log(mel + 1e-10)

        # CMN（沿时间轴减均值，对齐官方 mean_nor）
        if self.mean_nor:
            mel = mel - mel.mean(dim=2, keepdim=True)

        return mel.transpose(1, 2)  # [B, T', n_mels]


def official_fbank(wav: torch.Tensor) -> torch.Tensor:
    """官方 FBank（torchaudio kaldi），用于 consistency test。"""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    feats = []
    for i in range(wav.shape[0]):
        f = kaldi.fbank(wav[i].unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0)
        f = f - f.mean(0, keepdim=True)  # mean_nor
        feats.append(f)
    return torch.stack(feats)  # [B, T', 80]


def consistency_test(eres2net, diff_fbank, wav_batch, device):
    """
    frontend consistency sanity test：
      同一批 clean wav，官方 FBank -> ERes2Net -> emb_official
                      diff FBank   -> ERes2Net -> emb_diff
    打印两者 cosine similarity（按 batch 平均）。
    """
    eres2net.eval()
    wav_batch = wav_batch.to(device)

    with torch.no_grad():
        feat_official = official_fbank(wav_batch).to(device)  # [B, T, 80]
        emb_official = eres2net(feat_official).cpu().numpy()  # [B, 512]

        feat_diff = diff_fbank(wav_batch)  # [B, T', 80]
        emb_diff = eres2net(feat_diff).cpu().numpy()

    cos = []
    for i in range(emb_official.shape[0]):
        a = emb_official[i]
        b = emb_diff[i]
        c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        cos.append(c)

    cos_arr = np.array(cos)
    print(f"[consistency] 官方 FBank vs diff FBank 的 embedding cosine:")
    print(f"  均值={cos_arr.mean():.6f}  最小={cos_arr.min():.6f}  最大={cos_arr.max():.6f}")
    return cos_arr


if __name__ == "__main__":
    print("frontend_spk.py: DiffFBank 定义 + consistency_test 函数")
    print("在 train_spk_preserve.py 中调用 consistency_test 做 sanity check")
