# -*- coding: utf-8 -*-
"""
E2 多条件鲁棒微调 —— 公共工具

复用来源：
  - 路径解析 / 均匀抽取：AI/8.19/train_spk_preserve.py
  - 混噪公式：sv-ecapa/gen_noise_benchmark.py（分段 RMS + 防削波，与正式测试集同源）
  - EER/minDCF：speakerlab.utils.score_metrics（与正式评测同一实现）
"""

import os
import sys
import numpy as np
import torch
import torchaudio


# ============================================================
# 路径
# ============================================================

def resolve(base, p):
    return os.path.abspath(os.path.join(base, p))


def setup_speakerlab(sv_ecapa_cfg, base):
    """
    把 3D-Speaker 仓库根加入 sys.path，保证 `from speakerlab...` 可导入
    （等价于官方 sv-ecapa/path.sh 的 PYTHONPATH；幂等）。
    返回解析后的 sv_ecapa 绝对路径。
    """
    sv = resolve_existing_path(
        sv_ecapa_cfg if os.path.isabs(sv_ecapa_cfg) else os.path.join(base, sv_ecapa_cfg),
        base)
    root = os.path.dirname(os.path.dirname(os.path.dirname(sv)))
    if root not in sys.path:
        sys.path.insert(0, root)
    return sv


def resolve_existing_path(p, base_dir):
    """
    解析路径（文件或目录均可），兼容服务器历史路径 /data/home/... -> /home/...。
    """
    p = os.path.expanduser(p.strip())
    candidates = []
    if os.path.isabs(p):
        p = os.path.abspath(p)
        candidates.append(p)
        if p.startswith("/data/home/"):
            candidates.append(p[len("/data"):])
    else:
        candidates.append(os.path.abspath(os.path.join(base_dir, p)))
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("路径不存在（文件/目录都没找到）。尝试过：\n  "
                            + "\n  ".join(candidates))


def load_clean_paths(clean_scp, sv_ecapa):
    """从 all_wav.scp（utt_id 相对路径）解析出绝对路径列表。"""
    train_root = os.path.join(sv_ecapa, "data", "data_for_train")
    paths = []
    missing = []
    with open(clean_scp) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            p = line.split()[-1]
            if os.path.isabs(p):
                candidate = p
            else:
                if p.startswith("data/"):
                    p = p[len("data/"):]
                candidate = os.path.join(train_root, p)
            candidate = os.path.abspath(candidate)
            if not os.path.isfile(candidate) and candidate.startswith("/data/home/"):
                c2 = candidate[len("/data"):]
                if os.path.isfile(c2):
                    candidate = c2
            if not os.path.isfile(candidate):
                missing.append((line_no, candidate))
            else:
                paths.append(candidate)
    if missing:
        preview = "\n".join(f"  line {n}: {p}" for n, p in missing[:10])
        raise FileNotFoundError(f"clean_scp 中有 {len(missing)} 条无效路径。前10条：\n{preview}")
    return paths


def load_list_file(path, base_dir):
    """读取每行一个路径的清单文件并解析。"""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(resolve_existing_path(line, base_dir))
    return out


def uniform_select(items, k):
    """等间隔均匀选取 k 条（保序、确定性）。"""
    n = len(items)
    if k <= 0 or n <= k:
        return list(items)
    idxs = np.round(np.linspace(0, n - 1, k)).astype(int)
    return [items[i] for i in dict.fromkeys(idxs)]


# ============================================================
# 音频 / 混噪（与 gen_noise_benchmark.py 同一公式）
# ============================================================

SR = 16000


def load_wav(path, sr=SR):
    wav, s = torchaudio.load(path)
    if s != sr:
        wav = torchaudio.functional.resample(wav, s, sr)
    return wav[0].numpy().astype(np.float32)


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def random_crop(wav, target_len, rng):
    """随机裁剪到 target_len；恰好等长原样返回；不足则零填充。"""
    if len(wav) > target_len:
        start = rng.randint(0, len(wav) - target_len)
        return wav[start:start + target_len]
    if len(wav) == target_len:
        return wav                      # 恰好等长：randint(0,0) 会抛 high<=0，直接返回
    return np.pad(wav, (0, target_len - len(wav)))


def center_crop(wav, target_len):
    """确定性居中裁剪（cohort 提取用，保证 batch 特征等长可堆叠）。"""
    if len(wav) >= target_len:
        start = (len(wav) - target_len) // 2
        return wav[start:start + target_len]
    return np.pad(wav, (0, target_len - len(wav)))


def mix_noise(clean, noise_full, snr_db, noise_start, rng=None):
    """
    clean 与 noise_full[noise_start:...] 按 snr_db 混合。
    SNR 按噪声分段 RMS 计算；混合后峰值>0.99 时整体缩放（防削波）。
    返回 (mixed, 削波缩放)。实际 SNR 请用 measure_snr(clean, mixed) 事后校验
    （削波会轻微改变实际 SNR，属正常现象，正式测试集生成同样如此）。
    """
    target_len = len(clean)
    if noise_start + target_len <= len(noise_full):
        noise_seg = noise_full[noise_start:noise_start + target_len]
    else:
        seg = noise_full[noise_start:]
        reps = int(np.ceil(target_len / len(noise_full)))
        noise_seg = np.tile(np.concatenate([seg, np.tile(noise_full, reps)]), 1)[:target_len]

    noise_rms = _rms(noise_seg)
    clean_rms = _rms(clean)
    if noise_rms < 1e-12:
        noise_scaled = np.zeros_like(noise_seg)
    else:
        scale = clean_rms / (10 ** (snr_db / 20.0)) / noise_rms
        noise_scaled = noise_seg * scale

    mixed = clean + noise_scaled
    peak = np.max(np.abs(mixed)) if len(mixed) else 0.0
    clip_scale = 1.0
    if peak > 0.99:
        clip_scale = 0.99 / peak
        mixed = mixed * clip_scale

    return mixed.astype(np.float32), float(clip_scale)


def measure_snr(clean, mixed):
    """事后校验：从混合结果反推实际 SNR。"""
    noise = mixed - clean
    return 10 * np.log10((_rms(clean) ** 2) / (_rms(noise) ** 2 + 1e-12) + 1e-12)


# ============================================================
# 说话人 / trial 解析
# ============================================================

def spk_of_utt(utt_id):
    """3D_SPK_07304_012_Device06... -> 3D_SPK_07304"""
    parts = utt_id.split("_")
    return "_".join(parts[:3])


def load_trials(trial_path):
    """返回 [(enroll, test, label01), ...]"""
    out = []
    with open(trial_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 3:
                continue
            out.append((p[0], p[1], 1 if p[2] in ("1", "target") else 0))
    return out


def speaker_ids_from_trials(trial_path):
    ids = set()
    for enr, tst, _ in load_trials(trial_path):
        ids.add(spk_of_utt(enr))
        ids.add(spk_of_utt(tst))
    return ids


# ============================================================
# EER / minDCF（复用 speakerlab 官方实现）
# ============================================================

def eer_minDCF(scores, labels, p_target=0.01, c_miss=1.0, c_fa=1.0):
    """scores/labels 为 1D array。返回 (EER%, minDCF)。"""
    from speakerlab.utils.score_metrics import (
        compute_pmiss_pfa_rbst, compute_eer, compute_c_norm)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
    eer, _ = compute_eer(fnr, fpr, scores)
    min_dcf = compute_c_norm(fnr, fpr, p_target=p_target, c_miss=c_miss, c_fa=c_fa)
    # 统一转 python float：numpy 标量（np.float64/np.bool_）不能直接 json.dump
    return float(100.0 * eer), float(min_dcf)


# ============================================================
# embedding 提取 / ark 读写（key = wav 文件名 stem，与 trial 一致）
# ============================================================

def extract_embeddings(model, fbank, wav_paths, device, batch_size=32, desc="embed",
                       crop_len=None):
    """
    返回 (keys, np.ndarray[N, D])，keys 为 wav stem。
    crop_len 非空时对每条 wav 做确定性居中裁剪（变长语料如 cohort 必须指定，
    否则 batch 堆叠会因帧数不一而报错）。
    """
    from tqdm import tqdm
    model.eval()
    keys, embs = [], []
    buf, buf_keys = [], []

    def flush():
        if not buf:
            return
        with torch.no_grad():
            feat = torch.stack(buf).to(device)          # [B, T, 80]
            out = model(feat).cpu().numpy()
        embs.append(out)
        keys.extend(buf_keys)
        buf.clear()
        buf_keys.clear()

    for p in tqdm(wav_paths, desc=desc, ncols=100):
        wav = load_wav(p)
        if crop_len:
            wav = center_crop(wav, crop_len)
        feat = fbank(torch.from_numpy(wav))
        buf.append(feat)
        buf_keys.append(os.path.splitext(os.path.basename(p))[0])
        if len(buf) >= batch_size:
            flush()
    flush()
    return keys, np.concatenate(embs, axis=0) if embs else np.zeros((0, 0))


def save_ark(out_dir, keys, embs):
    """写 kaldiio ark（compute_score_metrics.py 可直接读取）。"""
    import kaldiio
    os.makedirs(out_dir, exist_ok=True)
    ark_path = os.path.join(out_dir, "emb.ark")
    with kaldiio.WriteHelper(f"ark:{ark_path}") as w:
        for k, e in zip(keys, embs):
            w(k, e.astype(np.float32))
    return ark_path


def load_ark_dir(data_dir):
    """读取目录下全部 ark → {key: ndarray}（与 compute_score_metrics.collect 一致）。"""
    import re
    import kaldiio
    data = {}
    arks = [os.path.join(data_dir, i) for i in os.listdir(data_dir) if re.search(r".ark$", i)]
    if not arks:
        raise FileNotFoundError(f"{data_dir} 下没有 .ark 文件")
    for ark in arks:
        with kaldiio.ReadHelper(f"ark:{ark}") as reader:
            for key, array in reader:
                data[key] = array
    return data
