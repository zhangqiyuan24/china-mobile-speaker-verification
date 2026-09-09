#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_ts.py — Plan B: cross-device clean-teacher / noisy-student（8.27）

攻 clean 跨设备地板（ERes2Net 5.08 / ECAPA 9.31）——此前所有路线只攻噪声轴，
通道轴从未被直接动过（预算分解见 8.26 总结/8.27 讨论）。

四损失 + Kendall 不确定性加权（可学习 log-variance, 参考 PolyU uncertainty-aware KD）：
  1. L_anchor: 1 − cos(e_s(noisy x), e_t(clean x))    冻结教师同句 clean 锚
  2. L_xdev  : 1 − cos(e_s(x_i), e_s(x_j))            同说话人跨设备正对
               （batch 由说话人对组成, 正对天然存在, 替代无正对的 SupCon；
                clean 条件下此损失=直接压跨设备地板）
  3. L_hard  : relu(max_cos_cross_spk − 0.30)          in-batch 难负 hinge
  4. L_ce    : AAM-CE（prototype 初始化 classifier, 防漂移）
  L = Σ_i w_i·exp(−s_i)·L_i + s_i,  s_i 可学习（init 0 → 有效权重=w_i）

学生/教师同构同空间：均 soup5（教师冻结）；数据/混噪/lr 与 robust 微调同源
（90k train_utts.csv + noise_train_freesound, snr_mix 同实现）。
骨干可换：--backbone ecapa 为预案一预留（首轨先 eres2net）。

选点：每 epoch hard proxy（trials_hard_*）, 学生自身做 extractor, avg 最小为
best。锚（soup5 本体 hard proxy）= 4.716/11.193 avg 7.955。
用法: python train_ts.py --gpu 2 [--pairs-per-batch 16 --epochs 3]
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
import toml

_HERE = os.path.dirname(os.path.abspath(__file__))
_AI = os.path.dirname(_HERE)
for _p in (os.path.join(_AI, "8.18"), os.path.join(_AI, "8.20"),
           os.path.join(_AI, "8.21"), os.path.join(_AI, "8.26"), _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from train_eres2net_robust import read_train_csv
from train_spk_preserve import load_noise_paths
from trainer_spk_preserve import load_eres2net, snr_mix
from frontend_spk import DiffFBank
from common_utils import setup_speakerlab
from trainer_eres2net_robust import build_classifier, build_aam_margin
from trainer_eres2net_robust_proto import (load_prototypes,
                                           init_classifier_prototypes)
from score_proxy import extract, eer_of

SR = 16000
CROP = SR * 3
PROXY = os.path.join(_HERE, "exp", "proxy")
ANCHOR_HARD = {"noise_0db": 4.716, "noise_m5db": 11.193}   # soup5 本体
TAU_NEG = 0.30
W = {"anchor": 1.0, "xdev": 0.5, "hard": 0.3, "ce": 1.0}   # 先验权重
# ts1 教训：L_hard 恒 0 时 Kendall 的 +s 正则把 s 无界压低 → 有效权重指数发散
# （实测 19→215 仍在涨, ep1 hard val 劣化 +1.4pp）。夹紧到 [0.05, 5.0]。
W_CLAMP = (0.05, 5.0)


def device_of(utt_id):
    for d in range(1, 9):
        if f"Device{d:02d}" in utt_id:
            return d
    return 0


def read_wav(path):
    return torchaudio.load(path)[0][0].numpy().astype(np.float32)


def build_pairs(rows, rng, max_pairs=None):
    """rows[(utt_id, spk, path, None)] -> 同说话人优先跨设备 utt 对。"""
    by_spk = {}
    for r in rows:
        by_spk.setdefault(r[1], []).append(r)
    pairs = []
    for spk, rs in sorted(by_spk.items()):
        by_dev = {}
        for r in rs:
            by_dev.setdefault(device_of(r[0]), []).append(r)
        devs = sorted(by_dev)
        if len(devs) >= 2:                       # 跨设备对（通道不变性主力）
            for i in range(len(devs)):
                a = by_dev[devs[i]][0]
                b = by_dev[devs[(i + 1) % len(devs)]][0]
                pairs.append((a, b))
        elif len(rs) >= 2:                        # 退化：同设备不同 utt
            pairs.append((rs[0], rs[1]))
    rng.shuffle(pairs)
    if max_pairs:
        pairs = pairs[:max_pairs]
    return pairs


class PairCondDataset(torch.utils.data.Dataset):
    """idx -> (pair=idx//3, cond=idx%3)。cond 0/1/2 = clean/0dB/-5dB。
    返回 (noisy_a, noisy_b, clean_a, clean_b, label)：原始波形 [48000]。
    fbank 在 GPU 侧对整批做（DiffFBank 接受 [B,T]）。"""

    def __init__(self, pairs, noise_paths):
        self.pairs = pairs
        self.noise_paths = noise_paths

    def __len__(self):
        return 3 * len(self.pairs)

    def _crop(self, wav):
        if len(wav) > CROP:
            st = np.random.randint(0, len(wav) - CROP)
            return wav[st:st + CROP]
        return np.pad(wav, (0, CROP - len(wav)))   # 恰好 CROP 长也走补零(no-op)

    def _mix(self, wav, snr):
        noise = read_wav(self.noise_paths[np.random.randint(
            len(self.noise_paths))])
        noisy, clean = snr_mix(wav, noise, snr, CROP)
        return noisy, clean

    def __getitem__(self, idx):
        base, cond = idx // 3, idx % 3
        (ida, spa, pa, lb), (idb, _, pb, _) = self.pairs[base]
        ca = self._crop(read_wav(pa))
        cb = self._crop(read_wav(pb))
        if cond > 0:
            snr = 0.0 if cond == 1 else -5.0
            na, ca = self._mix(ca, snr)           # clean 侧用 snr_mix 的裁剪
            nb, cb = self._mix(cb, snr)
        else:
            na, nb = ca, cb
        return (torch.from_numpy(np.ascontiguousarray(na)),
                torch.from_numpy(np.ascontiguousarray(nb)),
                torch.from_numpy(np.ascontiguousarray(ca)),
                torch.from_numpy(np.ascontiguousarray(cb)),
                lb)


@torch.no_grad()
def val_hard_student(model, fbank, device):
    """hard proxy: 学生提 enrol clean + 带噪 test 两侧嵌入 → trials_hard EER。"""
    was_training = model.training
    model.eval()
    res = {}
    enrol_files = [os.path.join(PROXY, "enrol", f)
                   for f in sorted(os.listdir(os.path.join(PROXY, "enrol")))]
    e_enrol = extract(model, fbank, enrol_files, device)
    for cond in ("noise_0db", "noise_m5db"):
        files = [os.path.join(PROXY, cond, f)
                 for f in sorted(os.listdir(os.path.join(PROXY, cond)))]
        e_test = extract(model, fbank, files, device)
        labs, scs = [], []
        with open(os.path.join(PROXY, f"trials_hard_{cond}.txt")) as fh:
            for line in fh:
                e_stem, t_stem, lab = line.split()
                a, b = e_enrol[e_stem], e_test[t_stem]
                scs.append(float(np.dot(a, b) /
                             (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)))
                labs.append(1 if lab == "target" else 0)
        res[cond] = eer_of(np.asarray(scs), np.asarray(labs))
    if was_training:
        model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        _AI, "8.26", "train_tse.toml"))     # paths/dataset 段照抄 8.26
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--backbone", default="eres2net",
                    choices=["eres2net", "ecapa"],
                    help="ecapa=预案一轨道（首轨先 eres2net）")
    ap.add_argument("--student-ckpt", default=None)   # 默认 = cfg verifier
    ap.add_argument("--teacher-ckpt", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--pairs-per-batch", type=int, default=16)
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--exp-tag", default="ts1")
    args = ap.parse_args()
    assert args.backbone == "eres2net", "ecapa 轨道待首轨出结果后接续"

    cfg = toml.load(args.config)
    sv_ecapa = os.path.abspath(cfg["paths"]["sv_ecapa"])
    train_csv = os.path.join(cfg["paths"]["data_dir_820"], "train_utts.csv")
    teacher_ckpt = args.teacher_ckpt or cfg["paths"]["verifier_ckpt"]
    student_ckpt = args.student_ckpt or cfg["paths"]["verifier_ckpt"]
    noise_paths = load_noise_paths(cfg["dataset"]["noise_list"], sv_ecapa)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda:0"
    setup_speakerlab(sv_ecapa, _HERE)
    exp_dir = os.path.join(_HERE, "exp", f"ts_{args.exp_tag}")
    os.makedirs(exp_dir, exist_ok=True)

    rng = random.Random(2026)
    rows, spk2id = read_train_csv(train_csv)      # (rows 含 int label, 官方 id 空间)
    pairs = build_pairs(rows, rng, args.max_pairs)
    xdev_ratio = (sum(1 for a, b in pairs if device_of(a[0]) != device_of(b[0]))
                  / max(len(pairs), 1))
    print(f"[data] {len(rows)} utts / {len(spk2id)} spk -> {len(pairs)} 对"
          f"（跨设备 {xdev_ratio:.1%}）/ 噪声 {len(noise_paths)}", flush=True)

    ds = PairCondDataset(pairs, noise_paths)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.pairs_per_batch, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True)

    student = load_eres2net(sv_ecapa, student_ckpt, device)
    teacher = load_eres2net(sv_ecapa, teacher_ckpt, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student.train()

    classifier = build_classifier(512, len(spk2id))
    protos, proto_spk_ids, _ = load_prototypes(cfg["paths"]["proto_npz"])
    init_classifier_prototypes(classifier, protos, proto_spk_ids, spk2id)
    classifier = classifier.to(device)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    margin_loss = build_aam_margin(32.0, 0.2).to(device)

    log_vars = {k: torch.zeros((), device=device, requires_grad=True)
                for k in W}
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.Adam(
        [{"params": params, "lr": args.lr},
         {"params": list(log_vars.values()), "lr": 1e-3}])
    print(f"[train] {len(loader)} it/ep × {args.epochs} ep, "
          f"batch={args.pairs_per_batch} 对({2*args.pairs_per_batch} 样本), "
          f"lr={args.lr}, 学生={os.path.basename(student_ckpt)}, "
          f"教师={os.path.basename(teacher_ckpt)}", flush=True)

    fbank = DiffFBank(n_mels=80, sample_rate=SR, n_fft=512, win_length=400,
                      hop_length=160, mean_nor=True).to(device)
    best = 1e9
    for ep in range(1, args.epochs + 1):
        sums = {k: 0. for k in list(W) + ["n"]}
        for it, (na, nb, ca, cb, lb) in enumerate(loader):
            na, nb = na.to(device), nb.to(device)
            ca, cb = ca.to(device), cb.to(device)
            lb = lb.to(device)
            opt.zero_grad()
            B = na.size(0)
            e_s = student(fbank(torch.cat([na, nb])))   # 2B 样本
            with torch.no_grad():
                e_t = teacher(fbank(torch.cat([ca, cb])))
            L_anchor = (1 - F.cosine_similarity(e_s, e_t, dim=1)).mean()
            # 跨设备正对：a-b 同说话人
            L_xdev = (1 - F.cosine_similarity(e_s[:B], e_s[B:], dim=1)).mean()
            # in-batch 难负：跨说话人最大余弦 hinge
            en = F.normalize(e_s, dim=1)
            sim = en @ en.t()
            lab2 = torch.cat([lb, lb])
            neg_mask = lab2.unsqueeze(0) != lab2.unsqueeze(1)
            eye = torch.eye(2 * B, dtype=torch.bool, device=device)
            sim.masked_fill_(~neg_mask | eye, -2.0)
            L_hard = F.relu(sim.max(dim=1).values - TAU_NEG).mean()
            L_ce = margin_loss(classifier(e_s).float(), lab2)
            losses = {"anchor": L_anchor, "xdev": L_xdev,
                      "hard": L_hard, "ce": L_ce}
            w_eff = {k: torch.clamp(W[k] * torch.exp(-log_vars[k]),
                                    *W_CLAMP) for k in W}
            L = sum(w_eff[k] * losses[k] + log_vars[k] for k in W)
            L.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            for k in W:
                sums[k] += float(losses[k])
            sums["n"] += 1
            if (it + 1) % 100 == 0:
                n = sums["n"]
                ws = {k: f"{float(w_eff[k]):.2f}" for k in W}
                print(f"  ep{ep} it {it+1}/{len(loader)} "
                      + " ".join(f"{k}={sums[k]/n:.4f}" for k in W)
                      + f" w={ws}", flush=True)
                for k in list(W) + ["n"]:
                    sums[k] = 0.
        res = val_hard_student(student, fbank, device)
        avg = (res["noise_0db"] + res["noise_m5db"]) / 2
        print(f"  [val-hard] ep{ep} 0db={res['noise_0db']:.3f} "
              f"m5db={res['noise_m5db']:.3f} avg={avg:.3f} "
              f"(锚 soup5 本体: 4.716/11.193 avg=7.955)", flush=True)
        ck = {"model": student.state_dict(), "epoch": ep,
              "eff_weights": {k: float(min(max(W[k] * torch.exp(-log_vars[k]),
                                       *W_CLAMP), 1e9)) for k in W}}
        torch.save(ck, os.path.join(exp_dir, f"checkpoint_epoch{ep}.tar"))
        if avg < best:
            best = avg
            torch.save(ck, os.path.join(exp_dir, "checkpoint_best.tar"))
            print(f"  [val-hard] 新最优 avg={avg:.3f} -> checkpoint_best.tar",
                  flush=True)
    print(f"[done] best_hard_avg={best:.3f} (锚 7.955)", flush=True)


if __name__ == "__main__":
    main()
