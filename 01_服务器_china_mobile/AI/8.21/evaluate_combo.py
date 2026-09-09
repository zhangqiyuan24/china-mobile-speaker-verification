# -*- coding: utf-8 -*-
"""
组合评测（正式 open-set 协议）：

  A. raw       : noise_0db / noise_m5db 原始带噪 wav ──► robust ERes2Net(8.21)
  B. frontend  : 同一批带噪 wav ──► speaker-aware FullSubNet+(8.18, 只读) ──► robust ERes2Net

协议约束：
  - enrollment 侧 = 官方 clean/3s（不过前端），与 test 侧用【同一个】robust checkpoint 提取
  - trials = 原始 trials_cross_device；打分 = 原版 speakerlab compute_score_metrics.py
  - 前端只作用于 noisy test 侧（clean 条件不增强）
  - 不默认前端有益：A/B 两个模式都跑，输出相对 raw robust 的额外增益（ablation）

带噪 wav 的增强版本由 8.17 的离线推理工具预生成（见 enhance_testset.sh），
本脚本只负责"同一 checkpoint 重提两侧 + 官方打分 + 汇总对照"。

用法：
  python evaluate_combo.py --config config_proto.toml --gpu 3 \
      --checkpoint exp/proto_robust/checkpoints/checkpoint_best.tar \
      --enh-0db exp/enhanced/noise_0db --enh-m5db exp/enhanced/noise_m5db
  # 不带 --enh-* 则只跑 raw 模式
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import toml

_HERE = os.path.dirname(os.path.abspath(__file__))
_V820 = os.path.join(os.path.dirname(_HERE), "8.20")
if _V820 not in sys.path:
    sys.path.insert(0, _V820)

from common_utils import setup_speakerlab, load_wav, save_ark
from validate_robust import load_model_for_validation, build_fbank
from evaluate_robust import (          # 8.20 正式评测构件，原样复用
    get_wav_list, run_official_scorer, read_metrics,
)

CONDS = ("noise_0db", "noise_m5db")


def index_by_stem(root):
    """递归建 {wav stem: 绝对路径} 索引（增强输出目录结构可能保留子目录）。"""
    idx = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith(".wav"):
                idx[os.path.splitext(fn)[0]] = os.path.join(dirpath, fn)
    assert idx, f"{root} 下没有 wav"
    return idx


def extract_len_safe(model, wav_paths, out_dir, device, batch_size, desc,
                     ckpt_path=None):
    """
    变长安全提取。官方 3s 语料实际 2.8~3.0s 不等（如 298/282 帧），8.20 的
    extract_embeddings 按固定 batch 堆叠要求帧数一致，会在这里崩。
    按【帧数分组】再组 batch：不裁剪、不补零，每条 wav 按自然全长提 embedding，
    与逐条提取完全等价；内存只驻留各组未满的 partial buffer。
    """
    from tqdm import tqdm
    fbank = build_fbank()
    model.eval()
    keys_out, embs = [], []
    groups = {}                        # n_frames -> ([feat...], [key...])

    def flush_group(nfr):
        feats, ks = groups.pop(nfr)
        if not feats:
            return
        with torch.no_grad():
            out = model(torch.stack(feats).to(device)).cpu().numpy()
        embs.append(out)
        keys_out.extend(ks)

    for p in tqdm(wav_paths, desc=desc, ncols=100):
        feat = fbank(torch.from_numpy(load_wav(p)))
        nfr = feat.shape[0]
        feats, ks = groups.setdefault(nfr, ([], []))
        feats.append(feat)
        ks.append(os.path.splitext(os.path.basename(p))[0])
        if len(feats) >= batch_size:
            flush_group(nfr)
    for nfr in list(groups):
        flush_group(nfr)

    assert len(keys_out) == len(wav_paths), "提取条数与输入不一致"
    assert len(set(keys_out)) == len(keys_out), "ark key（wav stem）有重复"
    embs = np.concatenate(embs, axis=0) if embs else np.zeros((0, 0), dtype=np.float32)
    save_ark(out_dir, keys_out, embs)
    if ckpt_path:
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump({"checkpoint": ckpt_path, "num_utts": len(keys_out),
                       "length_policy": "full_utterance_grouped"}, f, indent=1)
    print(f"[extract] {len(keys_out)} -> {out_dir}/emb.ark")
    return out_dir


def extract_guarded(model, wav_paths, out_dir, device, args, ckpt, desc):
    """带 checkpoint 一致性守卫的提取（目录里 ark 若是别的 checkpoint 提的必须重提）。"""
    ark = os.path.join(out_dir, "emb.ark")
    meta_p = os.path.join(out_dir, "meta.json")
    if os.path.isfile(ark):
        used = json.load(open(meta_p)).get("checkpoint") if os.path.isfile(meta_p) else None
        if used and os.path.abspath(used) != os.path.abspath(ckpt):
            raise RuntimeError(
                f"{out_dir} 的 ark 是用 {used} 提的，与本次 checkpoint {ckpt} 不一致。\n"
                f"换模型评测必须重提：rm -rf {out_dir} 后重跑。")
        print(f"[extract] {desc}: ark 已存在且 checkpoint 一致，跳过")
        return
    extract_len_safe(model, wav_paths, out_dir, device, args.batch_size,
                     desc=desc, ckpt_path=ckpt)


def run_mode(mode, model, cfg, base, device, args, ckpt, sv_ecapa,
             speakerlab_root, enrol_dir, trials_path, enh_index, out_root):
    """mode in {raw, frontend}；返回 {cond: {eer, mindcf}}（含 clean 仅 raw 模式）。"""
    test_root = os.path.join(sv_ecapa, "data", "data_for_test")
    emb_root = os.path.join(out_root, mode)
    conds = list(CONDS) + (["clean"] if mode == "raw" else [])
    results = {}
    for cond in conds:
        cdir = os.path.join(test_root, cond, "3s")
        wavs = get_wav_list(cdir, sv_ecapa)
        if mode == "frontend":
            assert cond in CONDS
            idx = enh_index[cond]
            missing = [os.path.splitext(os.path.basename(w))[0] for w in wavs
                       if os.path.splitext(os.path.basename(w))[0] not in idx]
            assert not missing, \
                f"增强目录缺 {len(missing)} 条（{missing[:3]}...），先跑 enhance_testset.sh"
            wavs = [idx[os.path.splitext(os.path.basename(w))[0]] for w in wavs]
        out_dir = os.path.join(emb_root, cond)
        extract_guarded(model, wavs, out_dir, device, args, ckpt,
                        desc=f"{mode}-{cond}")
        scores_dir = os.path.join(out_root, "scores", f"{mode}_{cond}")
        if os.path.isfile(os.path.join(scores_dir, "result.metrics")):
            print(f"[score] {mode}-{cond}: result.metrics 已存在，跳过打分"
                  f"（重打请删除 {scores_dir}）")
        else:
            run_official_scorer(speakerlab_root, enrol_dir, out_dir,
                                scores_dir, trials_path)
        eer, mindcf = read_metrics(scores_dir)
        results[cond] = {"eer": eer, "mindcf": mindcf}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_proto.toml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--enh-0db", default=None, help="noise_0db 增强后的 wav 目录")
    ap.add_argument("--enh-m5db", default=None, help="noise_m5db 增强后的 wav 目录")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cfg = toml.load(args.config)
    base = os.path.dirname(os.path.abspath(args.config))
    p_cfg = cfg["paths"]
    sv_ecapa = p_cfg["sv_ecapa"] if os.path.isabs(p_cfg["sv_ecapa"]) else \
        os.path.join(base, p_cfg["sv_ecapa"])
    speakerlab_root = os.path.dirname(os.path.dirname(os.path.dirname(sv_ecapa)))
    setup_speakerlab(p_cfg["sv_ecapa"], base)

    exp_dir = os.path.join(base, p_cfg["exp_dir"])
    ckpt = args.checkpoint or os.path.join(exp_dir, "checkpoints", "checkpoint_best.tar")
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(base, ckpt)
    assert os.path.isfile(ckpt), f"checkpoint 不存在: {ckpt}"
    eres2net_ckpt = p_cfg["eres2net_ckpt"]
    if not os.path.isabs(eres2net_ckpt):
        eres2net_ckpt = os.path.join(base, eres2net_ckpt)

    model = load_model_for_validation(ckpt, eres2net_ckpt, device)

    test_root = os.path.join(sv_ecapa, "data", "data_for_test")
    trials_path = os.path.join(test_root, "clean", "3s", "trials_cross_device")
    assert os.path.isfile(trials_path), f"缺 trials: {trials_path}"
    out_root = os.path.join(exp_dir, "eval", "combo")
    os.makedirs(out_root, exist_ok=True)

    # ---- enrollment 侧：官方 clean/3s，同一个 robust checkpoint，不过前端 ----
    clean_wavs = get_wav_list(os.path.join(test_root, "clean", "3s"), sv_ecapa)
    enrol_dir = os.path.join(out_root, "common", "enrol_clean")
    extract_guarded(model, clean_wavs, enrol_dir, device, args, ckpt,
                    desc="enrol-clean(robust ckpt)")

    modes = ["raw"]
    enh_index = {}
    if args.enh_0db and args.enh_m5db:
        enh_index["noise_0db"] = index_by_stem(
            args.enh_0db if os.path.isabs(args.enh_0db) else os.path.join(base, args.enh_0db))
        enh_index["noise_m5db"] = index_by_stem(
            args.enh_m5db if os.path.isabs(args.enh_m5db) else os.path.join(base, args.enh_m5db))
        modes.append("frontend")
    else:
        print("[combo] 未提供 --enh-0db/--enh-m5db，只跑 raw 模式（前端 ablation 缺失）")

    all_results = {}
    for mode in modes:
        print(f"\n===== 模式 {mode.upper()}："
              + ("原始带噪 → robust 模型" if mode == "raw"
                 else "带噪 → FullSubNet+(8.18) → robust 模型") + " =====")
        all_results[mode] = run_mode(mode, model, cfg, base, device, args, ckpt,
                                     sv_ecapa, speakerlab_root, enrol_dir,
                                     trials_path, enh_index, out_root)

    # ---- 汇总：官方基线 / 前端单独 / raw robust / 前端+robust + 额外增益 ----
    bl = cfg.get("baseline", {})
    rows = [("official baseline", {c: bl[c] for c in ("clean", "noise_0db", "noise_m5db")}),
            ("frontend only (8.18)", {"noise_0db": bl.get("noise_0db_frontend_only"),
                                      "noise_m5db": bl.get("noise_m5db_frontend_only")})]
    if "raw" in all_results:
        rows.append(("raw robust (8.21)", all_results["raw"]))
    if "frontend" in all_results:
        rows.append(("frontend + robust", all_results["frontend"]))

    print("\n===== 组合评测汇总（trials_cross_device, 3s, 官方打分） =====")
    print(f"{'方案':<24}{'0dB EER%':>10}{'-5dB EER%':>10}{'0dB 降幅':>10}{'-5dB 降幅':>10}")

    def _eer(v):
        return v["eer"] if isinstance(v, dict) else v

    for name, res in rows:
        r0, r5 = _eer(res.get("noise_0db")), _eer(res.get("noise_m5db"))
        d0 = f"{100.0*(bl['noise_0db']-r0)/bl['noise_0db']:>9.1f}%" \
            if r0 is not None and bl.get("noise_0db") else " " * 10
        d5 = f"{100.0*(bl['noise_m5db']-r5)/bl['noise_m5db']:>9.1f}%" \
            if r5 is not None and bl.get("noise_m5db") else " " * 10
        print(f"{name:<24}{r0 if r0 is not None else float('nan'):>10.4f}"
              f"{r5 if r5 is not None else float('nan'):>10.4f}{d0}{d5}")
    if "raw" in all_results:
        rc = all_results["raw"].get("clean")
        if rc:
            print(f"{'':<24}clean EER = {rc['eer']:.4f}%"
                  f"（官方 clean 基线 {bl.get('clean')}，open-set 真实 clean 退化）")

    if "raw" in all_results and "frontend" in all_results:
        raw, fe = all_results["raw"], all_results["frontend"]
        print("\n----- 前端 ablation（frontend+robust 相对 raw robust 的额外变化） -----")
        for cond in CONDS:
            delta = raw[cond]["eer"] - fe[cond]["eer"]
            rel = 100.0 * delta / raw[cond]["eer"]
            print(f"  {cond:<10}: raw {raw[cond]['eer']:.4f} -> frontend {fe[cond]['eer']:.4f} "
                  f"（额外 {'-' if delta < 0 else '+'}{abs(delta):.4f}pp / {rel:+.1f}%）"
                  f"{'← 前端有益' if delta > 0 else '← 前端无益（ablation 结论：不采用）'}")
        ok30 = all(fe[c]["eer"] <= bl[c] * 0.7 for c in CONDS)
        ok30_raw = all(raw[c]["eer"] <= bl[c] * 0.7 for c in CONDS)
        print(f"\n验收判据（0dB/-5dB 相对官方基线 ≥30%）："
              f"raw robust {'达标 ✓' if ok30_raw else '未达标'}；"
              f"frontend+robust {'达标 ✓' if ok30 else '未达标'}")
    elif "raw" in all_results:
        raw = all_results["raw"]
        ok30_raw = all(raw[c]["eer"] <= bl[c] * 0.7 for c in CONDS)
        print(f"\n验收判据（0dB/-5dB 相对官方基线 ≥30%）：raw robust "
              f"{'达标 ✓' if ok30_raw else '未达标'}")

    summary = {"checkpoint": ckpt, "results": all_results, "baseline": bl}
    with open(os.path.join(out_root, "combo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[combo] 明细已写 {os.path.join(out_root, 'combo_summary.json')}；"
          f"分数在 {out_root}/scores/")


if __name__ == "__main__":
    main()
