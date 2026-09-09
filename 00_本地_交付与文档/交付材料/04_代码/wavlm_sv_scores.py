#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""wavlm_sv_scores.py — microsoft/wavlm-base-sv 零样本评测(中文跨设备噪声)

目的:
  现有 ensemble 全部是 ERes2Net 系(同骨干), 融合增益趋缓。引入异构 SSL 模型
  WavLM 增加多样性, 即使单系统 EER 一般, score 融合也可能带来大幅提升。

方法:
  - trial 完全复用官方 trials_cross_device(3列), 保证与现有评测同一协议
  - enroll 嵌入: data_for_test/clean/3s 下的 wav
  - test  嵌入: data_for_test/{noise_0db,noise_m5db,clean}/3s 下的 wav
  - WavLM 吃原始 16k 波形(feature_extractor 做 per-utt 归一化), 不需要 fbank
  - 输出 4 列 score 文件(与 evaluate_combo.py 的输出同格式),
    可直接加进 cross_system_fusion_v2.py 的成员列表

产出:
  {out}/scores/raw_{cond}/trials_cross_device.score
  {out}/wavlm_summary.json   # 各条件 EER

用法:
  python wavlm_sv_scores.py --gpu 3 --batch-size 4 --out exp/wavlm_base_sv
"""
import argparse
import json
import os
import sys
import types

import numpy as np
import torch

# ---- torch 1.10 兼容 shim（不动共享 venv 的 site-packages）----
# transformers 4.46 的 encoder.forward 会 import torch.distributed.fsdp
# （torch 1.11+ 才有）。我们单卡推理根本不用 FSDP，注入空壳模块即可。
if "torch.distributed.fsdp" not in sys.modules:
    try:
        import torch.distributed.fsdp  # noqa: F401
    except ImportError:
        _fake = types.ModuleType("torch.distributed.fsdp")

        class _FSDP:  # 仅为 isinstance 检查提供类名
            pass

        _fake.FullyShardedDataParallel = _FSDP
        sys.modules["torch.distributed.fsdp"] = _fake
        import torch.distributed as _td
        _td.fsdp = _fake  # fsdp.py 里走 torch.distributed.fsdp.XXX 属性访问


def build_index(root):
    """递归建 {wav stem: 绝对路径} 索引。"""
    idx = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith(".wav"):
                idx[os.path.splitext(fn)[0]] = os.path.join(dirpath, fn)
    assert idx, f"{root} 下没有 wav"
    return idx


def load_wav_16k(path):
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    assert sr == 16000, f"{path} sr={sr}"
    return wav


def eer_of(scores, labels):
    """自实现 EER(与 cross_system_fusion_v2.py 一致, 无 speakerlab 依赖)。"""
    order = np.argsort(-scores)
    labs = labels[order]
    P = int(labels.sum())
    N = int(len(labels) - P)
    tp = np.cumsum(labs)
    fp = np.cumsum(1 - labs)
    fpr = fp / float(N)
    miss = (P - tp) / float(P)
    idx = int(np.argmin(np.abs(fpr - miss)))
    return (fpr[idx] + miss[idx]) / 2.0 * 100.0


@torch.no_grad()
def extract(model, fe, wav_paths, device, batch_size, desc):
    """按长度分组提嵌入(不裁剪), 返回与输入同序的 [N,512] 矩阵(已 L2 归一)。"""
    wavs = [load_wav_16k(p) for p in wav_paths]
    lens = np.array([len(w) for w in wavs])
    order = np.argsort(lens)  # 短的在前, 减少填充浪费
    embs = [None] * len(wavs)
    for s in range(0, len(order), batch_size):
        idxs = order[s:s + batch_size]
        batch = [wavs[i] for i in idxs]
        inputs = fe(batch, sampling_rate=16000, padding=True,
                    return_attention_mask=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        emb = model(**inputs).embeddings          # [B,512]
        emb = torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()
        for j, i in enumerate(idxs):
            embs[i] = emb[j]
        if (s // batch_size) % 50 == 0:
            sys.stdout.write(f"\r[{desc}] {s + len(idxs)}/{len(order)}")
            sys.stdout.flush()
    print()
    return np.stack(embs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="wavlm-base-sv")
    ap.add_argument("--ckpt", default=None,
                    help="微调 checkpoint(wavlm_finetune.py 产出, 取其 model 键)")
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default="exp/wavlm_base_sv")
    ap.add_argument("--sv-ecapa",
                    default="../../code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.model_dir)
    model = WavLMForXVector.from_pretrained(args.model_dir).eval().to(device)
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")["model"]
        model.load_state_dict(sd)
        print(f"[ckpt] 已加载 {args.ckpt} (epoch {torch.load(args.ckpt, map_location='cpu')['epoch']})")
    print(f"[model] wavlm-base-sv on {device}, params="
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    test_root = os.path.join(args.sv_ecapa, "data", "data_for_test")
    trials_path = os.path.join(test_root, "clean", "3s", "trials_cross_device")
    tri = [ln.split() for ln in open(trials_path)]
    enroll_ids = sorted({t[0] for t in tri})
    test_ids = sorted({t[1] for t in tri})
    print(f"[trials] {len(tri)} 条, enroll {len(enroll_ids)}, test {len(test_ids)}")

    # ---- enroll: clean 目录 ----
    clean_idx = build_index(os.path.join(test_root, "clean", "3s"))
    miss = [e for e in enroll_ids if e not in clean_idx]
    assert not miss, f"clean 目录缺 enroll {len(miss)} 条: {miss[:3]}"
    emb_cache = {"enroll": {e: None for e in enroll_ids}}
    ep = [clean_idx[e] for e in enroll_ids]
    E = extract(model, fe, ep, device, args.batch_size, "enroll")
    for e, v in zip(enroll_ids, E):
        emb_cache["enroll"][e] = v

    summary = {}
    for cond in ["clean", "noise_0db", "noise_m5db"]:
        cidx = build_index(os.path.join(test_root, cond, "3s"))
        need = [t for t in test_ids if t not in emb_cache.get(cond, {})]
        if need:
            tp = [cidx[t] for t in need]
            miss = [t for t, p in zip(need, tp) if not os.path.isfile(p)]
            assert not miss, f"{cond} 目录缺 test {len(miss)} 条: {miss[:3]}"
            T = extract(model, fe, tp, device, args.batch_size, cond)
            emb_cache.setdefault(cond, {})
            for t, v in zip(need, T):
                emb_cache[cond][t] = v

        scores, labels = [], []
        for t in tri:
            scores.append(float(np.dot(emb_cache["enroll"][t[0]],
                                       emb_cache[cond][t[1]])))
            labels.append(1 if t[2] == "target" else 0)
        scores = np.array(scores)
        labels = np.array(labels)
        eer = eer_of(scores, labels)

        sdir = os.path.join(args.out, "scores", f"raw_{cond}")
        os.makedirs(sdir, exist_ok=True)
        sf_path = os.path.join(sdir, "trials_cross_device.score")
        with open(sf_path, "w") as f:
            for t, lab, sc in zip(tri, labels, scores):
                f.write(f"{t[0]} {t[1]} {'target' if lab else 'nontarget'} {sc:.5f}\n")
        summary[cond] = {"eer": round(eer, 4)}
        print(f"[EER] raw_{cond}: {eer:.4f}")

    with open(os.path.join(args.out, "wavlm_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[done]", json.dumps(summary))


if __name__ == "__main__":
    main()
