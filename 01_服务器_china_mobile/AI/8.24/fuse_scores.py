#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fuse_scores.py — 分数层融合快速验证（8.25, CPU 零成本）

动机: 嵌入融合已被证伪(嵌入空间任意旋转), 但分数层同刻度 — 错误去相关即可降 EER。
VoxSRC/NIST SRE 冠军系统标配 2-3 系统分数融合。

输入: N 个已跑过 evaluate_combo 的 exp 目录(含 eval/combo/scores/), 对 5 个条件:
  1. 逐系统按 trial 对齐(enroll,test 为 key, 校验集合一致)
  2. 每系统 z-norm(该条件全部 trial 上标准化 — 融合常规操作, 消量纲差)
  3. 加权平均 -> EER
同时打印各单系统 EER(与官方 combo_summary 对账)。

用法:
  python fuse_scores.py --dirs exp/soup3_abc exp/proto_robust_90k --w 0.5,0.5
  python fuse_scores.py --dirs exp/soup3_abc exp/proto_robust_90k exp/channel_aug_90k
"""
import argparse
import os

import numpy as np

CONDS = ["raw_noise_0db", "raw_noise_m5db", "raw_clean",
         "frontend_noise_0db", "frontend_noise_m5db"]


def load_scores(path):
    """score 文件: 'enroll test label score' -> (keys, labels, scores)"""
    keys, labels, scores = [], [], []
    with open(path) as f:
        for ln in f:
            parts = ln.split()
            if len(parts) != 4:
                continue
            keys.append((parts[0], parts[1]))
            labels.append(1 if parts[2] == "target" else 0)
            scores.append(float(parts[3]))
    return keys, np.array(labels), np.array(scores, dtype=np.float64)


def eer(labels, scores):
    """标准 EER: 降序排列扫阈值, FPR/FNR 交点线性插值。"""
    order = np.argsort(-scores, kind="stable")
    lab = labels[order]
    n_tgt = lab.sum()
    n_non = len(lab) - n_tgt
    if n_tgt == 0 or n_non == 0:
        return float("nan")
    fp = np.cumsum(1 - lab)
    tp = np.cumsum(lab)
    fpr = fp / n_non
    fnr = (n_tgt - tp) / n_tgt
    # 找 fpr-fnr 变号处
    d = fpr - fnr
    idx = np.where(d >= 0)[0]
    if len(idx) == 0:
        return float(fnr[-1])
    i = idx[0]
    if i == 0:
        return float(fpr[0])
    # 线性插值
    d0, d1 = d[i - 1], d[i]
    e = fnr[i - 1] + (0 - d0) / (d1 - d0) * (fnr[i] - fnr[i - 1])
    return float(e * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--w", default=None,
                    help="逗号分隔权重(默认全均匀)")
    ap.add_argument("--no-znorm", action="store_true",
                    help="不做 z-norm 直接平均(cosine 同刻度)")
    ap.add_argument("--pair-raw-fe", action="store_true",
                    help="多视角融合: 同一目录内 raw_noise_X 与 frontend_noise_X "
                         "两套分数互融(去相关来源=增强伪影 vs 噪声本身)")
    ap.add_argument("--base", default=".",
                    help="相对根目录(默认当前目录)")
    args = ap.parse_args()

    n = len(args.dirs)
    ws = ([1.0] * n if not args.w
          else [float(x) for x in args.w.split(",")])
    assert len(ws) == n, f"权重数 {len(ws)} != 目录数 {n}"
    s = sum(ws)
    ws = [w / s for w in ws]
    names = [os.path.basename(d.rstrip("/")) for d in args.dirs]
    print(f"[fusion] {' + '.join(f'{nm}*{w:.2f}' for nm, w in zip(names, ws))}"
          f"{' (no-znorm)' if args.no_znorm else ' (znorm)'}")

    def fuse_list(sys_scores, labels):
        arr = np.stack(sys_scores)
        if not args.no_znorm:
            arr = (arr - arr.mean(1, keepdims=True)) / (arr.std(1, keepdims=True) + 1e-9)
        return eer(labels, np.tensordot(np.array(ws), arr, axes=1))

    if args.pair_raw_fe:
        # 多视角: 同一 exp 目录, raw 与 frontend 两种输入的分数互融
        for d in args.dirs:
            nm = os.path.basename(d.rstrip("/"))
            for cond in ("noise_0db", "noise_m5db"):
                paths = [os.path.join(args.base, d, "eval", "combo", "scores", p,
                                      "trials_cross_device.score")
                         for p in (f"raw_{cond}", f"frontend_{cond}")]
                data = []
                labels = None
                for p in paths:
                    if not os.path.isfile(p):
                        print(f"  {nm} {cond}: 缺 {p}")
                        labels = None
                        break
                    _k, lb, sc = load_scores(p)
                    labels = lb if labels is None else labels
                    data.append(sc)
                if labels is None:
                    continue
                singles = [eer(labels, sc) for sc in data]
                e_f = fuse_list(data, labels)
                e_best = min(singles)
                delta = (e_best - e_f) / e_best * 100 if e_best > 0 else 0
                print(f"  {nm:18s} rawxfe {cond:9s} raw={singles[0]:.4f} "
                      f"fe={singles[1]:.4f} fused={e_f:.4f} "
                      f"({'+' if e_f < e_best else '-'}{abs(delta):.1f}% vs best)")
        return

    for cond in CONDS:
        ref_keys = None
        sys_data = []
        ok = True
        for d in args.dirs:
            p = os.path.join(args.base, d, "eval", "combo", "scores", cond,
                             "trials_cross_device.score")
            if not os.path.isfile(p):
                print(f"  {cond}: 缺 {p} -> 跳过该系统")
                ok = False
                break
            keys, labels, scores = load_scores(p)
            if ref_keys is None:
                ref_keys, ref_labels = keys, labels
            elif keys != ref_keys:
                print(f"  {cond}: trial 集合不一致({d}) -> 跳过")
                ok = False
                break
            sys_data.append(scores)
        if not ok:
            continue
        singles = [eer(ref_labels, sc) for sc in sys_data]
        arr = np.stack(sys_data)                     # [N, T]
        if not args.no_znorm:
            arr = (arr - arr.mean(1, keepdims=True)) / (arr.std(1, keepdims=True) + 1e-9)
        fused = np.tensordot(np.array(ws), arr, axes=1)
        e_fused = eer(ref_labels, fused)
        e_best = min(singles)
        single_str = "/".join(f"{e:.4f}" for e in singles)
        delta = (e_best - e_fused) / e_best * 100 if e_best > 0 else 0
        print(f"  {cond:22s} singles={single_str} fused={e_fused:.4f} "
              f"({'+' if e_fused < e_best else '-'}{abs(delta):.1f}% vs best single)")


if __name__ == "__main__":
    main()
