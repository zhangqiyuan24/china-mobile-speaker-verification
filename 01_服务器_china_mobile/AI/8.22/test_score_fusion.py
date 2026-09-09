# -*- coding: utf-8 -*-
"""
分数融合快测（路线图杠杆 B，零训练）：raw 路 与 enh 路（前端->robust）的
cosine 分数加权融合，看 EER 能不能比单路更低。

数据源：evaluate_combo.py 已落盘的官方 ark（同一 enrol、同一 trials）——
  {combo_root}/common/enrol_clean          注册侧（两路共用）
  {combo_root}/raw/noise_{0db,m5db}        raw 路（原始带噪 -> 模型）
  {combo_root}/frontend/noise_{0db,m5db}   enh 路（前端增强 -> 同一模型）
默认指向 8.21 的 combo 目录（现成数据，训练 8.22 期间就能先测有没有戏）；
8.22 训练完后把 --combo-root 指到 exp/matched_robust/eval/combo 复测。

内存说明：test ark 只按唯一 utt 建矩阵（16401×512），trial 侧用行号索引
点积，绝不按 1.38M 条 trial 堆叠（那会膨胀 ~3GB）。

用法：
  python test_score_fusion.py --config config_matched.toml
  python test_score_fusion.py --config config_matched.toml \
      --combo-root ../8.21/exp/proto_robust/eval/combo --weights 0.3,0.5,0.7
"""

import os
import sys
import argparse
import numpy as np
import toml

_HERE = os.path.dirname(os.path.abspath(__file__))
_V820 = os.path.join(os.path.dirname(_HERE), "8.20")
if _V820 not in sys.path:
    sys.path.insert(0, _V820)

from common_utils import load_ark_dir, load_trials, eer_minDCF, setup_speakerlab

CONDS = ("noise_0db", "noise_m5db")


def _norm_rows(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_matched.toml")
    ap.add_argument("--combo-root", default="../8.21/exp/proto_robust/eval/combo",
                    help="evaluate_combo.py 的输出目录（含 common/ raw/ frontend/）")
    ap.add_argument("--weights", default="0.3,0.5,0.7",
                    help="raw 路权重列表（fused = w*raw + (1-w)*enh）")
    ap.add_argument("--gpu", default="-", help="纯 ark 后处理不用 GPU，参数仅为统一入口")
    args = ap.parse_args()

    cfg = toml.load(args.config)
    base = os.path.dirname(os.path.abspath(args.config))
    sv_ecapa = setup_speakerlab(cfg["paths"]["sv_ecapa"], base)
    trials_path = os.path.join(sv_ecapa, "data", "data_for_test",
                               "clean", "3s", "trials_cross_device")
    root = args.combo_root if os.path.isabs(args.combo_root) \
        else os.path.join(base, args.combo_root)
    weights = [float(w) for w in args.weights.split(",")]

    # 注册侧（两路共用）
    enrol = load_ark_dir(os.path.join(root, "common", "enrol_clean"))
    ekeys = sorted(enrol)
    E = _norm_rows(np.stack([enroll[e] for e in ekeys]))
    erow = {e: i for i, e in enumerate(ekeys)}

    trials = load_trials(trials_path)
    labels = np.array([lab for _, _, lab in trials])
    e_idx = np.asarray([erow[e] for e, _, _ in trials], dtype=np.int64)
    print(f"[fusion] combo_root={root}\n[fusion] trials={len(trials)}  "
          f"enrol={len(ekeys)}  weights(raw)={weights}\n")

    bl = cfg.get("baseline", {})

    def rel(eer, b):
        return f"{100.0 * (b - eer) / b:>9.1f}%" if b else " " * 10

    for cond in CONDS:
        raw = load_ark_dir(os.path.join(root, "raw", cond))
        enh = load_ark_dir(os.path.join(root, "frontend", cond))
        assert set(raw) == set(enh), \
            f"{cond}: raw/enh 两路 ark 的 key 集合不一致（提取不完整？）"
        miss = [t for _, t, _ in trials if t not in raw]
        assert not miss, f"{cond}: ark 缺 {len(miss)} 条 test（如 {miss[:3]}）"

        tkeys = sorted(raw)                                   # 唯一 utt（两路同序）
        trow = {t: i for i, t in enumerate(tkeys)}
        t_idx = np.asarray([trow[t] for _, t, _ in trials], dtype=np.int64)
        Traw = _norm_rows(np.stack([raw[t] for t in tkeys]))
        Tenh = _norm_rows(np.stack([enh[t] for t in tkeys]))
        s_raw = np.einsum("ij,ij->i", E[e_idx], Traw[t_idx])   # [P] 逐 trial 分数
        s_enh = np.einsum("ij,ij->i", E[e_idx], Tenh[t_idx])

        b = bl.get(cond)
        print(f"===== {cond} =====")
        print(f"  {'方案':<28}{'EER%':>9}{'相对基线':>10}")
        eer_r, _ = eer_minDCF(s_raw, labels)
        eer_h, _ = eer_minDCF(s_enh, labels)
        print(f"  {'raw 路（原始带噪）':<28}{eer_r:>9.4f}{rel(eer_r, b)}")
        print(f"  {'enh 路（前端->模型）':<28}{eer_h:>9.4f}{rel(eer_h, b)}")
        best = (min(eer_r, eer_h), "单路最优")
        for w in weights:
            eer_f, _ = eer_minDCF(w * s_raw + (1.0 - w) * s_enh, labels)
            name = f"plain 融合 w_raw={w:.1f}"
            print(f"  {name:<28}{eer_f:>9.4f}{rel(eer_f, b)}")
            if eer_f < best[0]:
                best = (eer_f, name)
        for w in weights:
            eer_f, _ = eer_minDCF(w * _z(s_raw) + (1.0 - w) * _z(s_enh), labels)
            name = f"znorm 融合 w_raw={w:.1f}（参考）"
            print(f"  {name:<28}{eer_f:>9.4f}{rel(eer_f, b)}")
            if eer_f < best[0]:
                best = (eer_f, name)
        tail = f"（基线 {b:.4f}，相对 -{(1 - best[0] / b) * 100:.1f}%）" if b else ""
        print(f"  --> 最优：{best[1]}  EER={best[0]:.4f}%{tail}\n")

    print("[fusion] 说明：plain 融合 = 同一模型两路 cosine 的线性组合，部署时对 test 侧"
          "过两次模型即可；\n        znorm 为传递式操作（用到当批分数分布）仅作参考。"
          "融合是否计入验收口径需与老师确认。")


if __name__ == "__main__":
    main()
