#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""score_proxy_hard.py — 11 成员在 hard trials 上重打分（8.27 proxy v2）

配对 mine_hard_trials.py：同一批音频/后端/变体，仅 trial 集换成挖掘出的
trials_hard_{cond}.txt。嵌入按后端缓存 npz（重评 trial 不必重提）。
输出 exp/proxy/scores_hard/{member}_{cond}.score + 新旧 EER 对照表
（新锚点 = tse_soup5 的 hard EER, 供 Phase 1 三臂终判）。

用法: python score_proxy_hard.py --gpu 1
"""
import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_AI = os.path.dirname(_HERE)
for _p in (os.path.join(_AI, "8.18"), os.path.join(_AI, "8.20"),
           os.path.join(_AI, "8.21"), os.path.join(_AI, "8.26"), _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from trainer_spk_preserve import load_eres2net
from frontend_spk import DiffFBank
from common_utils import setup_speakerlab

PROXY = os.path.join(_HERE, "exp", "proxy")
OUT = os.path.join(PROXY, "scores_hard")
CACHE = os.path.join(PROXY, "emb_cache")
SV_ECAPA = ("/data/home/ligangrui/china_mobile/code/3D-speaker/3D-speaker"
            "/egs/3dspeaker/sv-ecapa")

CKPT = {
    "821": "/data/home/ligangrui/china_mobile/AI/8.21/exp/proto_robust/checkpoints/checkpoint_best.tar",
    "30k": "/data/home/ligangrui/china_mobile/AI/8.22/exp/matched_robust/checkpoints/checkpoint_best.tar",
    "90k": "/data/home/ligangrui/china_mobile/AI/8.22/exp/matched_robust_90k/checkpoints/checkpoint_best.tar",
    "soup": "/data/home/ligangrui/china_mobile/AI/8.24/exp/soup_821_90k/checkpoints/checkpoint_soup_w50.tar",
    "p90k": "/data/home/ligangrui/china_mobile/AI/8.24/exp/proto_robust_90k/checkpoints/checkpoint_best.tar",
    "soup5": "/data/home/ligangrui/china_mobile/AI/8.24/exp/soup5/checkpoints/checkpoint_soup5.tar",
}
MEMBERS = [
    ("821_raw", "821", "raw"), ("821_fe", "821", "enh818"),
    ("30k_raw", "30k", "raw"), ("30k_fe", "30k", "enh818"),
    ("90k_raw", "90k", "raw"), ("90k_fe", "90k", "enh818"),
    ("soup_raw", "soup", "raw"), ("soup_fe", "soup", "enh818"),
    ("p90k_raw", "p90k", "raw"), ("p90k_fe", "p90k", "enh818"),
    ("tse_soup5", "soup5", "enhtse"),
]
OLD_EER = {  # 旧随机负对 proxy EER（score_proxy.log 记录）, 对照用
    ("821_raw", "noise_0db"): 2.239, ("821_raw", "noise_m5db"): 6.000,
    ("821_fe", "noise_0db"): 2.000, ("821_fe", "noise_m5db"): 5.000,
    ("30k_raw", "noise_0db"): 2.239, ("30k_raw", "noise_m5db"): 5.500,
    ("30k_fe", "noise_0db"): 2.239, ("30k_fe", "noise_m5db"): 4.761,
    ("90k_raw", "noise_0db"): 2.477, ("90k_raw", "noise_m5db"): 5.693,
    ("90k_fe", "noise_0db"): 2.500, ("90k_fe", "noise_m5db"): 4.307,
    ("soup_raw", "noise_0db"): 2.239, ("soup_raw", "noise_m5db"): 6.000,
    ("soup_fe", "noise_0db"): 2.023, ("soup_fe", "noise_m5db"): 4.500,
    ("p90k_raw", "noise_0db"): 2.500, ("p90k_raw", "noise_m5db"): 6.045,
    ("p90k_fe", "noise_0db"): 2.239, ("p90k_fe", "noise_m5db"): 5.000,
    ("tse_soup5", "noise_0db"): 2.000, ("tse_soup5", "noise_m5db"): 4.261,
}


def eer_of(scores, labels):
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
def extract(model, fbank, files, device, batch=8):
    import torchaudio
    embs = {}
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        waves = []
        for p in chunk:
            w, _ = torchaudio.load(p)
            waves.append(w[0])
        e = model(fbank(torch.stack(waves).to(device))).cpu().numpy()
        for p, row in zip(chunk, e):
            embs[os.path.splitext(os.path.basename(p))[0]] = row
    return embs


def get_set_embs(backend, model, fbank, device):
    """enrol + raw/enh818/enhtse × 2 cond 的嵌入, npz 缓存。"""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{backend}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        # dict 经 savez 变 0-d object 数组, item() 解包回 dict
        return {k: (z[k].item() if z[k].shape == () else z[k])
                for k in z.files}
    out = {}
    enrol = [os.path.join(PROXY, "enrol", f)
             for f in sorted(os.listdir(os.path.join(PROXY, "enrol")))]
    out["enrol"] = extract(model, fbank, enrol, device)
    for variant in ("raw", "enh818", "enhtse"):
        for cond in ("noise_0db", "noise_m5db"):
            d = os.path.join(PROXY, cond if variant == "raw"
                             else f"{variant}_{cond}")
            files = [os.path.join(d, f) for f in sorted(os.listdir(d))]
            out[f"{variant}_{cond}"] = extract(model, fbank, files, device)
    np.savez(path, **out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda:0"
    setup_speakerlab(SV_ECAPA, _HERE)
    os.makedirs(OUT, exist_ok=True)
    fbank = DiffFBank(n_mels=80, sample_rate=16000, n_fft=512,
                      win_length=400, hop_length=160, mean_nor=True).to(device)

    trials = {}
    for cond in ("noise_0db", "noise_m5db"):
        rows = []
        with open(os.path.join(PROXY, f"trials_hard_{cond}.txt")) as f:
            for line in f:
                p = line.split()
                if len(p) == 3:
                    rows.append((p[0], p[1], p[2]))
        trials[cond] = rows

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    models = {}
    table = {}
    for name, backend, variant in MEMBERS:
        if backend not in models:
            models[backend] = load_eres2net(SV_ECAPA, CKPT[backend], device)
            models[backend].eval()
            print(f"[backend] {backend} <- {os.path.basename(CKPT[backend])}",
                  flush=True)
        embs = get_set_embs(backend, models[backend], fbank, device)
        e_enrol = embs["enrol"]
        for cond in ("noise_0db", "noise_m5db"):
            e_test = embs[f"{variant}_{cond}"]
            labs, scs = [], []
            out_path = os.path.join(OUT, f"{name}_{cond}.score")
            with open(out_path, "w") as f:
                for e_stem, t_stem, lab in trials[cond]:
                    c = cos(e_enrol[e_stem], e_test[t_stem])
                    f.write(f"{e_stem} {t_stem} {lab} {c:.6f}\n")
                    labs.append(1 if lab == "target" else 0)
                    scs.append(c)
            eer = eer_of(np.asarray(scs), np.asarray(labs))
            table[(name, cond)] = eer
            print(f"[score] {name:12s} {cond:11s} hard EER={eer:.3f}%"
                  f"（旧 {OLD_EER.get((name, cond), float('nan')):.3f}）",
                  flush=True)

    print("\n===== hard proxy 汇总（EER%）=====")
    print(f"{'member':<12}{'0dB旧':>8}{'0dBhard':>9}{'-5dB旧':>9}{'-5dBhard':>10}")
    for name, _, _ in MEMBERS:
        print(f"{name:<12}"
              f"{OLD_EER[(name, 'noise_0db')]:>8.3f}"
              f"{table[(name, 'noise_0db')]:>9.3f}"
              f"{OLD_EER[(name, 'noise_m5db')]:>9.3f}"
              f"{table[(name, 'noise_m5db')]:>10.3f}")
    print("[done] 新锚点 = tse_soup5 hard: "
          f"{table[('tse_soup5', 'noise_0db')]:.3f}/"
          f"{table[('tse_soup5', 'noise_m5db')]:.3f}", flush=True)


if __name__ == "__main__":
    main()
