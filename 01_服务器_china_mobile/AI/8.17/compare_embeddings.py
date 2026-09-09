#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
比较 clean / noisy / denoise 三条音频的 speaker embedding 相似度

用三个声纹模型 (ECAPA-TDNN / ERes2Net-large / ReDimNet2-B6) 分别提取 embedding，
计算三者两两 cosine 相似度。

核心指标: cos(clean, denoise) 应 > cos(clean, noisy)，说明降噪让语音更接近 clean。

用法:
  python compare_embeddings.py
"""

import os
import sys
import numpy as np
import torch
import torchaudio

# ---- 路径 (相对脚本所在目录 AI/8.17/) ----
BASE = os.path.dirname(os.path.abspath(__file__))
SPEAKERLAB_ROOT = os.path.abspath(os.path.join(BASE, "../../code/3D-speaker/3D-speaker"))
REDIMNET2_ROOT = os.path.abspath(os.path.join(BASE, "../../code/redimnet2/redimnet2"))
MODEL_DIR = os.path.abspath(os.path.join(BASE, "../8.15/models"))
COMPARE_DIR = os.path.join(BASE, "compare")

for p in [SPEAKERLAB_ROOT, REDIMNET2_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from speakerlab.process.processor import FBank
from speakerlab.models.eres2net.ERes2Net import ERes2Net
from speakerlab.models.ecapa_tdnn.ECAPA_TDNN import ECAPA_TDNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 16000


# ============================================================
# 工具
# ============================================================
def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def load_wav_mono(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav  # [1, T]


# ============================================================
# 模型加载
# ============================================================
def load_eres2net():
    from speakerlab.models.eres2net.ERes2Net import ERes2Net
    model = ERes2Net(feat_dim=80, embedding_size=512, m_channels=64)
    ckpt = os.path.join(MODEL_DIR, "eres2net_large_model.ckpt")
    state = torch.load(ckpt, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    if "embedding_model" in state:
        state = state["embedding_model"]
    model.load_state_dict(state, strict=False)
    model.eval().to(DEVICE)
    return model, "fbank"  # 需要 FBank 特征


def load_ecapa():
    from speakerlab.models.ecapa_tdnn.ECAPA_TDNN import ECAPA_TDNN
    model = ECAPA_TDNN(input_size=80, lin_neurons=192,
                       channels=[1024, 1024, 1024, 1024, 3072])
    ckpt = os.path.join(MODEL_DIR, "ecapa-tdnn.ckpt")
    state = torch.load(ckpt, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval().to(DEVICE)
    return model, "fbank"


def load_redimnet2():
    from redimnet2.redimnet2 import ReDimNet2Wrap
    ckpt = os.path.join(MODEL_DIR, "b6-vb2+vox2+cnc2_v0-lm.pt")
    full = torch.load(ckpt, map_location="cpu")
    model = ReDimNet2Wrap(**full["model_config"])
    model.load_state_dict(full["state_dict"])
    model.eval().to(DEVICE)
    return model, "raw"  # 直接输入原始波形


# ============================================================
# embedding 提取
# ============================================================
fbank = FBank(80, sample_rate=SR, mean_nor=True)


def extract_embedding(model, input_type, wav_path):
    wav = load_wav_mono(wav_path)
    if input_type == "fbank":
        feat = fbank(wav).unsqueeze(0).to(DEVICE)  # [1, T, 80]
        with torch.no_grad():
            emb = model(feat).squeeze(0).cpu().numpy()
    else:  # raw
        wav = wav.to(DEVICE)
        with torch.no_grad():
            emb = model(wav).squeeze(0).cpu().numpy()
    return emb


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print(f"  比较 clean / noisy / denoise embedding 相似度")
    print(f"  设备: {DEVICE}")
    print("=" * 60)

    clean_path = os.path.join(COMPARE_DIR, "clean",
                              "3D_SPK_06154_001_Device03_Distance00_Dialect00_3s.wav")
    noisy_path = os.path.join(COMPARE_DIR, "noisy",
                              "3D_SPK_06154_001_Device03_Distance00_Dialect00_3s.wav")
    denoise_path = os.path.join(COMPARE_DIR, "denoise",
                                "3D_SPK_06154_001_Device03_Distance00_Dialect00_3s.wav")

    for p in [clean_path, noisy_path, denoise_path]:
        if not os.path.exists(p):
            print(f"[ERROR] 找不到音频: {p}")
            return

    models = [
        ("ERes2Net-large", load_eres2net),
        ("ECAPA-TDNN", load_ecapa),
        ("ReDimNet2-B6", load_redimnet2),
    ]

    for name, loader in models:
        try:
            model, input_type = loader()
        except Exception as e:
            print(f"\n===== {name} =====  [加载失败: {e}]")
            continue

        emb_clean = extract_embedding(model, input_type, clean_path)
        emb_noisy = extract_embedding(model, input_type, noisy_path)
        emb_denoise = extract_embedding(model, input_type, denoise_path)

        cos_cn = cosine(emb_clean, emb_noisy)
        cos_cd = cosine(emb_clean, emb_denoise)
        cos_nd = cosine(emb_noisy, emb_denoise)

        gain = cos_cd - cos_cn  # 降噪相对 noisy 的提升

        print(f"\n===== {name} (dim={emb_clean.shape[0]}) =====")
        print(f"  cos(clean, noisy)   = {cos_cn:.4f}")
        print(f"  cos(clean, denoise) = {cos_cd:.4f}")
        print(f"  cos(noisy, denoise) = {cos_nd:.4f}")
        print(f"  降噪提升: {gain:+.4f}  {'✓ 有效' if gain > 0 else '✗ 无提升/退化'}")

    print("\n" + "=" * 60)
    print("  判定: 降噪提升 > 0 表示 denoise 比 noisy 更接近 clean")
    print("=" * 60)


if __name__ == "__main__":
    main()
