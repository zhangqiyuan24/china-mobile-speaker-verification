#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""enhance_tse.py — TSE 前端推理：官方测试集带噪 wav 逐条条件化增强（8.26 线①）

条件向量 = 官方注册 clean wav 经冻结 soup5 提取（均值 + L2 归一）—— 与训练时
build_spk_embs.py 的条件空间同分布（单 utt 时即单 utt L2 归一）。
test→enrol 对应关系取自 trials_cross_device 的正对（target）。

输出布局与 8.21 enhance_testset 产物一致（每条件一目录、wav stem 原名），
可直接喂 evaluate_combo.py / evaluate_ecapa.py 的 --enh-0db/--enh-m5db。

用法:
  python enhance_tse.py --config train_tse.toml --gpu 2 \
      --checkpoint exp/tse/checkpoint_epoch3.tar [--out-root exp/tse/enhanced]
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torchaudio
import toml

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_V818 := os.path.join(os.path.dirname(_HERE), "8.18"),
           _V820 := os.path.join(os.path.dirname(_HERE), "8.20"),
           _V821 := os.path.join(os.path.dirname(_HERE), "8.21"),
           _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from train_tse import build_tse_fullsubnet          # 复用训练侧的模型构建
from train_spk_preserve import _setup
from trainer_spk_preserve import load_eres2net
from frontend_spk import DiffFBank
from common_utils import setup_speakerlab
from evaluate_robust import get_wav_list


def speaker_of(stem):
    """3D_SPK_06154_001_Device03... -> 3D_SPK_06154（说话人前缀）。"""
    return "_".join(stem.split("_")[:3])


@torch.no_grad()
def extract_enrol_embs(verifier, fbank, wav_by_stem, speakers, device):
    """逐说话人：注册 clean 各 utt pad/中心裁 3s → fbank → soup5 → 均值 → L2。

    噪声测试目录同时含注册侧 utt（Device03）与打分侧（Device06/07，共 16401
    stem，trials 只打分 9609 条），FE 模式却要求全部可增强 —— 因此条件向量按
    说话人粒度取注册均值，对两种 utt 一致成立。
    """
    groups = defaultdict(list)
    for stem, p in wav_by_stem.items():
        s = speaker_of(stem)
        if s in speakers and len(groups[s]) < 8:   # 质心用前 8 条足够
            groups[s].append(p)
    out = {}
    todo = [(s, ps) for s, ps in groups.items()]
    assert len(todo) == len(speakers), \
        f"注册侧缺说话人: {sorted(set(speakers) - set(groups))[:5]}"
    pairs = [(s, p) for s, ps in groups.items() for p in ps]
    print(f"  [enrol-emb] {len(pairs)} utts / {len(groups)} spks", flush=True)
    sums, cnts = defaultdict(lambda: np.zeros(512, np.float32)), defaultdict(int)
    CHUNK = 8      # 显存紧张时实测峰值 ~2.8GB（探针口径）
    for i in range(0, len(pairs), CHUNK):
        chunk = pairs[i:i + CHUNK]
        waves = []
        for _s, p in chunk:
            w, _ = torchaudio.load(p)
            w = w[0]
            if w.numel() < 48000:
                w = torch.nn.functional.pad(w, (0, 48000 - w.numel()))
            else:
                st = (w.numel() - 48000) // 2
                w = w[st:st + 48000]
            waves.append(w)
        e = verifier(fbank(torch.stack(waves).to(device))).cpu().numpy()
        for row, (s, _p) in zip(e, chunk):
            sums[s] += row
            cnts[s] += 1
        if (i // CHUNK) % 50 == 0:
            print(f"  [enrol-emb] {i}/{len(pairs)}", flush=True)
    for s in groups:
        m = sums[s] / max(cnts[s], 1)
        out[s] = (m / (np.linalg.norm(m) + 1e-8)).astype(np.float32)
    return out


@torch.no_grad()
def enhance_cond(model, emb_by_stem, test2enrol, wavs, out_dir, device,
                 n_fft, hop, win, batch_size, stft_fn, istft_fn,
                 decompress_cIRM):
    """帧长+说话人分组批推理；断点续跑（已存在的 wav 跳过）。"""
    os.makedirs(out_dir, exist_ok=True)
    done = 0
    todo = []
    for p in wavs:
        stem = os.path.splitext(os.path.basename(p))[0]
        out_p = os.path.join(out_dir, stem + ".wav")
        if os.path.isfile(out_p):
            done += 1
        else:
            todo.append((stem, p, out_p))
    print(f"  [enhance] 已完成 {done}，待处理 {len(todo)}", flush=True)
    if not todo:
        return

    # (样本长度, 说话人) 分组 —— 同组才能堆叠成一个 batch 且共享条件向量
    groups = defaultdict(list)
    for stem, p, out_p in todo:
        w, _ = torchaudio.load(p)
        groups[(w.numel(), test2enrol[stem])].append((w[0], stem, out_p, p))
    keys = sorted(groups.keys())
    for gi, key in enumerate(keys):
        items = groups[key]
        emb = torch.from_numpy(emb_by_stem[key[1]]).to(device)
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            noisy = torch.stack([c[0] for c in chunk]).to(device)
            e = emb.unsqueeze(0).expand(noisy.size(0), -1).contiguous()
            noisy_complex = stft_fn(noisy, n_fft, hop, win)
            cRM_raw = model(noisy_complex.abs().unsqueeze(1),
                            noisy_complex.real.unsqueeze(1),
                            noisy_complex.imag.unsqueeze(1), e)
            cRM_raw = cRM_raw.permute(0, 2, 3, 1)
            dec = decompress_cIRM(cRM_raw)
            er = dec[..., 0] * noisy_complex.real - dec[..., 1] * noisy_complex.imag
            ei = dec[..., 1] * noisy_complex.real + dec[..., 0] * noisy_complex.imag
            enh = istft_fn(torch.stack([er, ei], dim=-1), n_fft, hop, win,
                           length=noisy.size(-1))
            for j, (_w, stem, out_p, _p) in enumerate(chunk):
                torchaudio.save(out_p, enh[j].unsqueeze(0).cpu(), 16000,
                                encoding="PCM_S", bits_per_sample=16)
        if gi % 20 == 0:
            print(f"  [enhance] 组 {gi}/{len(keys)}", flush=True)
    print(f"  [enhance] 本轮完成 {len(todo)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train_tse.toml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-root", default=None,
                    help="默认 exp_dir/enhanced")
    ap.add_argument("--limit", type=int, default=None,
                    help="每条件只处理前 N 条（冒烟用）")
    args = ap.parse_args()

    cfg = toml.load(os.path.join(_HERE, args.config))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    sv_ecapa = os.path.abspath(cfg["paths"]["sv_ecapa"])
    fullsubnet_root = os.path.abspath(cfg["paths"]["fullsubnet_root"])
    _setup(fullsubnet_root, sv_ecapa)
    setup_speakerlab(sv_ecapa, _HERE)

    # ---- 条件向量：官方 enrol clean → soup5（按说话人粒度） ----
    test_root = os.path.join(sv_ecapa, "data", "data_for_test")
    clean_wavs = get_wav_list(os.path.join(test_root, "clean", "3s"), sv_ecapa)
    wav_by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in clean_wavs}
    print(f"[enrol] clean wav {len(wav_by_stem)}", flush=True)

    verifier = load_eres2net(sv_ecapa, cfg["paths"]["verifier_ckpt"], device)
    verifier.eval()
    fbank = DiffFBank(n_mels=80, sample_rate=16000, n_fft=512,
                      win_length=400, hop_length=160, mean_nor=True).to(device)

    all_test = []
    for cond in ("noise_0db", "noise_m5db"):
        wl = get_wav_list(os.path.join(test_root, cond, "3s"), sv_ecapa)
        if args.limit:
            wl = wl[:args.limit]
        all_test += [os.path.splitext(os.path.basename(p))[0] for p in wl]
    speakers = sorted({speaker_of(s) for s in all_test})
    emb_by_spk = extract_enrol_embs(verifier, fbank, wav_by_stem, speakers,
                                    device)
    print(f"[cond] {len(emb_by_spk)} 个说话人注册嵌入(512, L2)", flush=True)

    # ---- TSE 模型 ----
    model, stft_fn, istft_fn, decompress_cIRM, _b = build_tse_fullsubnet(
        fullsubnet_root, cfg["paths"]["fullsubnet_ckpt"], cfg["model"], device)
    ck = torch.load(args.checkpoint, map_location="cpu")
    sd = ck["model"] if "model" in ck else ck
    model.load_state_dict(sd)
    model.eval()
    print(f"[tse] 装载 {args.checkpoint}(epoch={ck.get('epoch', '?')})", flush=True)

    n_fft, hop, win = (cfg["acoustics"][k] for k in ("n_fft", "hop_length", "win_length"))
    out_root = args.out_root or os.path.join(cfg["paths"]["exp_dir"], "enhanced")

    for cond in ("noise_0db", "noise_m5db"):
        wavs = get_wav_list(os.path.join(test_root, cond, "3s"), sv_ecapa)
        if args.limit:
            wavs = wavs[:args.limit]
        stems = [os.path.splitext(os.path.basename(p))[0] for p in wavs]
        test2spk = {s: speaker_of(s) for s in stems}
        miss = [s for s in stems if test2spk[s] not in emb_by_spk]
        assert not miss, f"{cond} 有 {len(miss)} 条测试的说话人无注册嵌入"
        print(f"[{cond}] {len(wavs)} 条带噪待增强", flush=True)
        enhance_cond(model, emb_by_spk, test2spk, wavs,
                     os.path.join(out_root, cond), device,
                     n_fft, hop, win, args.batch_size, stft_fn, istft_fn,
                     decompress_cIRM)
    print(f"[done] 增强产物 -> {out_root}/noise_0db noise_m5db", flush=True)


if __name__ == "__main__":
    main()
