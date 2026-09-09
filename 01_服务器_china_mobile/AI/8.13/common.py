#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WeSep TSE + ERes2Net 评测 — 共享工具
路径解析 / wav 加载 / 模型加载 / 批量推理 / EER
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

# 本文件所在目录 = AI/8.13/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 路径解析: 相对路径 → 绝对路径 (相对于 AI/8.13/)
# ============================================================
def resolve(rel_path):
    return os.path.abspath(os.path.join(BASE_DIR, rel_path))


def resolve_paths(cfg):
    """把 config 里的 _rel 路径解析成绝对路径，返回更新后的 dict"""
    c = dict(cfg)
    c["wesep"]["third_party"] = resolve(c["wesep"]["third_party_rel"])
    c["wesep"]["model_dir"] = resolve(c["wesep"]["model_dir_rel"])
    c["eres2net"]["ckpt"] = resolve(c["eres2net"]["ckpt_rel"])
    c["data"]["sv_ecapa"] = resolve(c["data"]["sv_ecapa_rel"])
    return c


# ============================================================
# wav 工具
# ============================================================
def load_wav_mono(path, target_sr=16000):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).numpy().astype(np.float32)


# ============================================================
# WeSep Extractor 加载
# ============================================================
def load_wesep_extractor(cfg):
    third_party = cfg["wesep"]["third_party"]
    # 友好检查: third_party 下必须有 wesep/ 和 wespeaker/ 两个包
    for pkg in ["wesep", "wespeaker"]:
        if not os.path.isdir(os.path.join(third_party, pkg)):
            raise FileNotFoundError(
                f"\n[ERROR] 找不到 {pkg} 包！\n"
                f"  third_party = {third_party}\n"
                f"  期望存在: {third_party}/{pkg}/\n"
                f"  请先在本地运行 download_local.sh 下载依赖，\n"
                f"  再把整个 AI/8.13 目录传到服务器。"
            )
    if third_party not in sys.path:
        sys.path.insert(0, third_party)
    from wesep.cli.extractor import Extractor

    model_dir = cfg["wesep"]["model_dir"]
    extractor = Extractor(model_dir)
    extractor.set_device(cfg["wesep"]["device"])
    extractor.set_vad(cfg["wesep"].get("apply_vad", False))
    print(f"  [WeSep] joint_training={extractor.joint_training}, "
          f"speaker_feat={extractor.speaker_feat}, "
          f"resample_rate={extractor.resample_rate}")
    return extractor


# ============================================================
# ERes2Net 加载
# ============================================================
def load_eres2net(cfg):
    sv_ecapa = cfg["data"]["sv_ecapa"]
    speakerlab_root = os.path.abspath(os.path.join(sv_ecapa, "..", "..", "..", ".."))
    for c in [speakerlab_root,
              os.path.abspath(os.path.join(sv_ecapa, "..", "..", ".."))]:
        if os.path.exists(os.path.join(c, "speakerlab")) and c not in sys.path:
            sys.path.insert(0, c)

    from speakerlab.models.eres2net.ERes2Net import ERes2Net
    from speakerlab.process.processor import FBank

    ecfg = cfg["eres2net"]
    model = ERes2Net(
        feat_dim=ecfg["feat_dim"],
        embedding_size=ecfg["embedding_size"],
        m_channels=ecfg["m_channels"],
    )
    state = torch.load(ecfg["ckpt"], map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    if "embedding_model" in state:
        state = state["embedding_model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  [ERes2Net] missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
    if len(missing) > 0:
        print(f"    missing: {missing[:5]}...")
    if len(unexpected) > 0:
        print(f"    unexpected: {unexpected[:5]}...")
    model.eval()
    model.to(ecfg["device"])
    fbank = FBank(80, sample_rate=16000, mean_nor=True)
    return model, fbank, ecfg["device"]


# ============================================================
# WeSep 批量推理 — 严格复刻官方 Extractor.extract_speech_from_pcm
# ============================================================
def tse_extract_batch(extractor, mix_wavs, enroll_wavs):
    """
    批量 WeSep 目标说话人提取。

    严格复刻 Extractor.extract_speech_from_pcm 的推理逻辑:
      - joint_training + speaker_feat=True → enrollment 先 compute_fbank
      - joint_training + speaker_feat=False → enrollment 直接波形
      - forward 返回 tuple → 取 [0]
      - output normalize 到 0.9 (防除零)

    Args:
        mix_wavs: list of 1D float32 numpy (mixture)
        enroll_wavs: list of 1D float32 numpy (enrollment)
    Returns:
        list of 1D float32 numpy (extracted target)
    """
    model = extractor.model
    device = extractor.device
    sr = extractor.resample_rate

    # ---- mixture batch → [B, max_len] (pad 到统一长度, 支持长度不一致) ----
    lengths = [w.shape[0] for w in mix_wavs]
    max_mix_len = max(lengths)
    mix_batch = torch.stack([
        F.pad(torch.from_numpy(w), (0, max_mix_len - len(w)))
        for w in mix_wavs
    ]).float().to(device)

    # ---- enrollment 处理 (复刻官方分支) ----
    if extractor.joint_training:
        if extractor.speaker_feat:
            # enrollment → fbank → [B, T', 80]
            feats_list = []
            for w in enroll_wavs:
                w_t = torch.from_numpy(w).unsqueeze(0).float()
                feat = extractor.compute_fbank(w_t, sample_rate=sr, cmn=True)
                feats_list.append(feat)
            max_len = max(f.shape[0] for f in feats_list)
            feats = torch.stack([
                F.pad(f, (0, 0, 0, max_len - f.shape[0])) for f in feats_list
            ]).to(device)
        else:
            # enrollment 直接原始波形
            feats = torch.from_numpy(np.stack(enroll_wavs)).float().to(device)
    else:
        raise ValueError("[WeSep] joint_training=False 的模型不支持")

    # ---- 前向 ----
    with torch.no_grad():
        outputs = model(mix_batch, feats)
    if isinstance(outputs, (list, tuple)):
        outputs = outputs[0]

    # ---- normalize 到 0.9 (与 Extractor 一致, 防除零) ----
    peak = outputs.abs().max(dim=1, keepdim=True).values.clamp_min(1e-8)
    outputs = outputs / peak * 0.9

    # ---- crop 回原长度 ----
    return [outputs[i, :lengths[i]].cpu().numpy().astype(np.float32)
            for i in range(outputs.shape[0])]


# ============================================================
# ERes2Net 批量 embedding
# ============================================================
def extract_embeddings_batch(model, fbank, device, wavs):
    """wavs: list of 1D float32 numpy → list of 1D numpy embedding (512)"""
    feats_list = []
    for w in wavs:
        w_t = torch.from_numpy(w).unsqueeze(0)
        feat = fbank(w_t)  # [T, 80]
        feats_list.append(feat)
    max_len = max(f.shape[0] for f in feats_list)
    feats = torch.stack([
        F.pad(f, (0, 0, 0, max_len - f.shape[0])) for f in feats_list
    ]).to(device)  # [B, T, 80]
    with torch.no_grad():
        embs = model(feats).cpu().numpy()  # [B, 512]
    return [embs[i] for i in range(embs.shape[0])]


# ============================================================
# cosine + EER
# ============================================================
def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def compute_eer(target_scores, nontarget_scores):
    """标准 EER: FAR(t)=mean(nontarget>=t), FRR(t)=mean(target<t)"""
    all_scores = np.concatenate([target_scores, nontarget_scores])
    thresholds = np.sort(np.unique(all_scores))
    min_diff, eer = 1.0, 1.0
    for t in thresholds:
        far = np.mean(nontarget_scores >= t)
        frr = np.mean(target_scores < t)
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2
    return eer * 100


# ============================================================
# 平衡抽样 (固定 seed, 100 target + 100 nontarget)
# ============================================================
def balanced_sample_trials(trials, n_target=100, n_nontarget=100, seed=2026):
    """从 trials 中固定 seed 抽 target/nontarget 平衡样本"""
    rng = random.Random(seed)
    target_trials = [t for t in trials if t[2] in ("target", "1")]
    nontarget_trials = [t for t in trials if t[2] not in ("target", "1")]

    sampled_target = rng.sample(target_trials, min(n_target, len(target_trials)))
    sampled_nontarget = rng.sample(nontarget_trials, min(n_nontarget, len(nontarget_trials)))
    return sampled_target + sampled_nontarget
