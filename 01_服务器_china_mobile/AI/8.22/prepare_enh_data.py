# -*- coding: utf-8 -*-
"""
E2-matched 语料准备：用【部署前端】(8.18 speaker-preserve, 冻结) 生成
训练增强语料 + 验证 bench 增强版。

为什么必须离线预生成：8.21/8.20 的 raw 分支是动态混噪（每 epoch 重抽裁剪/噪声），
enhancement 无法跟着在线做；所以 enh 分支用一份固定语料（每条语句每个 SNR 一次
固定实现），raw 分支保持动态不变。

三步（--all 默认全跑，各自可独立重跑，块级断点续跑）：
  ① --mix      30k 训练清单 × SNR{0,-5} 固定种子混噪 -> exp/enh_corpus/noisy/<tag>/chunk_*/
                （每条 (行号, snr) 独立 RandomState 种子：跳过任意块重跑结果不变）
  ② --enhance  分块调 8.17 的 speech_enhance.tools.inference（只读代码 + 前端 ckpt）
                -> exp/enh_corpus/enhanced/<tag>/chunk_*/<utt>.wav（拍平到块目录）
  ③ --bench    8.20 val_bench 的 test_0db/test_m5db 过前端 -> exp/enh_bench/<cond>/
                （验证 bench 用的增强测试侧，选 checkpoint 的主路指标）

全部完成后写 exp/enh_corpus/manifest.json（complete=true + 前端 ckpt 绝对路径）。
train_matched.py 启动时校验：manifest complete、前端 ckpt 与 config 一致、逐条覆盖
—— 换前端生成语料必须重跑本脚本，否则训练拒绝启动。

用法：
  python prepare_enh_data.py --config config_matched.toml --gpu 3 --all
  python prepare_enh_data.py --config config_matched.toml --gpu 3 --mix      # 只混噪
  python prepare_enh_data.py --config config_matched.toml --gpu 3 --enhance  # 只增强(续跑)
  python prepare_enh_data.py --config config_matched.toml --gpu 3 --bench
  python prepare_enh_data.py --config config_matched.toml --gpu 3 --all --limit 500  # 冒烟
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
import numpy as np
import toml

_HERE = os.path.dirname(os.path.abspath(__file__))
_V820 = os.path.join(os.path.dirname(_HERE), "8.20")
if _V820 not in sys.path:
    sys.path.insert(0, _V820)

from common_utils import (
    load_wav, random_crop, mix_noise, measure_snr, resolve_existing_path,
    setup_speakerlab, SR,
)
from train_eres2net_robust import read_train_csv


def save_wav(path, wav, sr=SR):
    """与 8.20 prepare_multicond_data.save_wav 同实现（PCM16）。"""
    try:
        import soundfile as sf
        sf.write(path, wav.astype(np.float32), sr, subtype="PCM_16")
    except ImportError:
        from scipy.io import wavfile
        wav16 = np.clip(wav, -1.0, 1.0)
        wavfile.write(path, sr, (wav16 * 32767).astype(np.int16))


def wavs_in(root):
    """root 下全部 wav 的 {stem: 绝对路径}（递归，兼容增强工具的 enhanced_* 子目录）。"""
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith(".wav"):
                out[os.path.splitext(fn)[0]] = os.path.join(dirpath, fn)
    return out


# ============================================================
# ① 固定种子混噪：每条 (row_index, snr_index) 独立种子，块级可续跑
# ============================================================

def step_mix(cfg, base, rows, corpus_root, ecfg):
    snrs = ecfg["snrs"]
    tags = ecfg["snr_tags"]
    crop_len = int(ecfg["crop_seconds"] * SR)
    seed = ecfg["seed"]
    chunk_size = ecfg["chunk_size"]

    # noise_train.txt 的行是相对 sv_ecapa 的路径（8.20 落盘时原样抄自其 noise_train_list，
    # 8.20/8.21 训练脚本同样以 sv_ecapa 为锚解析——锚点必须一致，否则找不到文件）
    sv_ecapa = setup_speakerlab(cfg["paths"]["sv_ecapa"], base)
    noise_paths = [resolve_existing_path(l, sv_ecapa)
                   for l in open(os.path.join(_data820(cfg, base), "noise_train.txt"))
                   if l.strip()]
    print(f"[mix] train noise pool = {len(noise_paths)} 条（与 8.21 raw 分支同池）")
    noise_cache = {}          # path -> wav（615 条全量驻留 ~1GB，混噪 60k 次省 60k 次读盘）

    n_done = n_skip = 0
    snr_checks = {t: [] for t in tags}
    for ci in range(0, len(rows), chunk_size):
        chunk_rows = rows[ci:ci + chunk_size]
        for tag_i, tag in enumerate(tags):
            cdir = os.path.join(corpus_root, "noisy", tag,
                                f"chunk_{ci // chunk_size:03d}")
            marker = os.path.join(cdir, ".done")
            if os.path.isfile(marker):
                # 实数校验：并发/中断曾产出"标记在、文件缺"的假完成块
                # （标记内容 = 落盘时的条数，此处必须逐一核对）
                n_have = len(wavs_in(cdir))
                try:
                    n_want = int(open(marker).read().strip() or 0)
                except ValueError:
                    n_want = -1
                if n_have == n_want == len(chunk_rows):
                    n_skip += len(chunk_rows)
                    continue
                print(f"[mix]   {tag} chunk_{ci // chunk_size:03d}: "
                      f"标记 {n_want} / 实数 {n_have} / 应 {len(chunk_rows)}"
                      f" -> 清掉重做", flush=True)
                for name in os.listdir(cdir):      # 连同 .done 标记一并清掉
                    p = os.path.join(cdir, name)
                    shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
            os.makedirs(cdir, exist_ok=True)
            for k, (utt_id, _spk, path, _label) in enumerate(chunk_rows):
                # 独立种子 = base + 全局行号*2 + snr 序号：与处理顺序/跳过无关，可复现
                rng = np.random.RandomState(seed + (ci + k) * 2 + tag_i)
                wav3 = random_crop(load_wav(path), crop_len, rng)
                noise_path = noise_paths[rng.randint(0, len(noise_paths))]
                noise = noise_cache.get(noise_path)
                if noise is None:
                    noise = noise_cache[noise_path] = load_wav(noise_path)
                start = rng.randint(0, max(1, len(noise) - crop_len)) \
                    if len(noise) > crop_len else 0
                mixed, _ = mix_noise(wav3, noise, snrs[tag_i], start)
                save_wav(os.path.join(cdir, f"{utt_id}.wav"), mixed)
                n_done += 1
                if (k + 1) % 200 == 0:
                    print(f"[mix]   {tag} chunk_{ci // chunk_size:03d}: "
                          f"{k + 1}/{len(chunk_rows)}", flush=True)
                if len(snr_checks[tag]) < 3 and (ci + k) % 5000 == 0:
                    snr_checks[tag].append(
                        (utt_id, measure_snr(wav3, mixed)))
            with open(marker, "w") as f:
                f.write(f"{len(chunk_rows)}\n")
        print(f"[mix] {min(ci + chunk_size, len(rows))}/{len(rows)} 条 × {len(tags)} 档"
              f"（新写 {n_done}，跳过 {n_skip}）", flush=True)
    for tag, expect in zip(tags, ecfg["snrs"]):
        if snr_checks[tag]:
            print(f"[mix] {tag} SNR 抽检（期望 {expect:+.0f} dB）: "
                  + ", ".join(f"{u}:{s:.2f}" for u, s in snr_checks[tag]))


# ============================================================
# ② 调 8.17 推理工具分块增强（块级断点续跑，输出拍平）
# ============================================================

def run_fsn_inference(fsn_root, cfg_inf_abs, ckpt_abs, in_dir, out_dir, gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    subprocess.run(
        [sys.executable, "-m", "speech_enhance.tools.inference",
         "-C", cfg_inf_abs, "-M", ckpt_abs, "-I", in_dir, "-O", out_dir],
        cwd=fsn_root, env=env, check=True)


def flatten_enhanced(out_dir):
    """把 out_dir/enhanced_*/ 下的 wav 拍平到 out_dir 根（stem 不变）。"""
    moved = 0
    for dirpath, _dirs, files in os.walk(out_dir):
        if os.path.samefile(dirpath, out_dir):
            continue
        for fn in files:
            if fn.endswith(".wav"):
                dst = os.path.join(out_dir, fn)
                assert not os.path.exists(dst), f"拍平冲突: {dst}"
                shutil.move(os.path.join(dirpath, fn), dst)
                moved += 1
    for name in os.listdir(out_dir):                     # 清掉空的 enhanced_* 壳
        p = os.path.join(out_dir, name)
        if name.startswith("enhanced_") and os.path.isdir(p):
            shutil.rmtree(p)
    return moved


def step_enhance(cfg, base, rows, corpus_root, ecfg, f_cfg, args):
    fsn_root = resolve_existing_path(
        f_cfg["fsn_code"] if os.path.isabs(f_cfg["fsn_code"])
        else os.path.join(base, f_cfg["fsn_code"]), base)
    cfg_inf = os.path.join(fsn_root, f_cfg["fsn_inference_config"])
    ckpt = f_cfg["ckpt"] if os.path.isabs(f_cfg["ckpt"]) else os.path.join(base, f_cfg["ckpt"])
    ckpt = resolve_existing_path(ckpt, base)
    for need in (fsn_root, cfg_inf, ckpt):
        assert os.path.exists(need), f"缺 {need}（config [frontend] 配置有误）"

    tags = ecfg["snr_tags"]
    chunk_size = ecfg["chunk_size"]
    for tag in tags:
        noisy_root = os.path.join(corpus_root, "noisy", tag)
        enh_root = os.path.join(corpus_root, "enhanced", tag)
        n_chunks = (len(rows) + chunk_size - 1) // chunk_size
        for k in range(n_chunks):
            in_dir = os.path.join(noisy_root, f"chunk_{k:03d}")
            out_dir = os.path.join(enh_root, f"chunk_{k:03d}")
            src = wavs_in(in_dir)
            expect = len(rows[k * chunk_size:(k + 1) * chunk_size])
            assert len(src) == expect, (
                f"{in_dir} 只有 {len(src)}/{expect} 条混噪文件——.done 标记与实数"
                "不符（重跑 --mix 会自动修复），拒绝在此状态下增强")
            assert src, f"{in_dir} 为空（先跑 --mix）"
            os.makedirs(out_dir, exist_ok=True)
            got = wavs_in(out_dir)
            if len(got) == len(src) and set(got) == set(src):
                continue                                    # 该块已完成，续跑跳过
            # 半成品块清掉重做（避免拍平后新旧混杂）
            for name in os.listdir(out_dir):
                p = os.path.join(out_dir, name)
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
            print(f"[enhance] {tag} chunk_{k:03d}: {len(src)} 条 -> {out_dir}", flush=True)
            run_fsn_inference(fsn_root, cfg_inf, ckpt, in_dir, out_dir, args.gpu)
            flatten_enhanced(out_dir)
            got = wavs_in(out_dir)
            assert len(got) == len(src) and set(got) == set(src), \
                (f"块 {out_dir} 增强不完整: {len(got)}/{len(src)}，"
                 f"缺 {sorted(set(src) - set(got))[:3]}")
        print(f"[enhance] {tag}: {n_chunks} 块全部完成", flush=True)


def step_manifest(cfg, base, rows, corpus_root, ecfg, f_cfg, complete):
    f_cfg_ck = f_cfg["ckpt"] if os.path.isabs(f_cfg["ckpt"]) else os.path.join(base, f_cfg["ckpt"])
    mani = {
        "complete": bool(complete and len(rows) > 0),
        "num_utts": len(rows),
        "frontend_ckpt": resolve_existing_path(f_cfg_ck, base),
        "fsn_code": resolve_existing_path(
            f_cfg["fsn_code"] if os.path.isabs(f_cfg["fsn_code"])
            else os.path.join(base, f_cfg["fsn_code"]), base),
        "seed": ecfg["seed"],
        "snrs": ecfg["snrs"],
        "snr_tags": ecfg["snr_tags"],
        "enh_root": os.path.join(corpus_root, "enhanced"),
    }
    # 完整性实测：逐 tag 数一遍
    for tag in ecfg["snr_tags"]:
        n = sum(len(wavs_in(os.path.join(corpus_root, "enhanced", tag, d)))
                for d in os.listdir(os.path.join(corpus_root, "enhanced", tag)))
        mani[f"num_enhanced_{tag}"] = n
        if mani["complete"]:
            assert n == len(rows), f"enhanced/{tag} 只有 {n}/{len(rows)} 条"
    with open(os.path.join(corpus_root, "manifest.json"), "w") as f:
        json.dump(mani, f, indent=2, ensure_ascii=False)
    print(f"[manifest] complete={mani['complete']}  "
          + "  ".join(f"{t}={mani[f'num_enhanced_{t}']}" for t in ecfg["snr_tags"])
          + f"  前端={mani['frontend_ckpt']}")
    if not mani["complete"]:
        print("[manifest] 注意：--limit 冒烟语料，manifest.complete=false，"
              "正式训练（无 --allow-partial-corpus）会拒绝启动")


# ============================================================
# ③ 验证 bench 的两个噪声条件过前端
# ============================================================

def step_bench(cfg, base, corpus_root, ecfg, f_cfg, args):
    data820 = _data820(cfg, base)
    bench = os.path.join(data820, "val_bench")
    out_root = os.path.join(base, cfg["paths"]["enh_bench_dir"])
    fsn_root = resolve_existing_path(
        f_cfg["fsn_code"] if os.path.isabs(f_cfg["fsn_code"])
        else os.path.join(base, f_cfg["fsn_code"]), base)
    cfg_inf = os.path.join(fsn_root, f_cfg["fsn_inference_config"])
    ckpt = resolve_existing_path(
        f_cfg["ckpt"] if os.path.isabs(f_cfg["ckpt"]) else os.path.join(base, f_cfg["ckpt"]),
        base)

    for cond in ("test_0db", "test_m5db"):
        in_dir = os.path.join(bench, cond)
        out_dir = os.path.join(out_root, cond)
        src = wavs_in(in_dir)
        assert src, f"{in_dir} 为空（8.20 val_bench 缺失？）"
        os.makedirs(out_dir, exist_ok=True)
        got = wavs_in(out_dir)
        if len(got) == len(src) and set(got) == set(src):
            print(f"[bench] {cond}: 已增强（{len(got)} 条），跳过")
            continue
        for name in os.listdir(out_dir):                # 半成品重做
            p = os.path.join(out_dir, name)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        print(f"[bench] {cond}: {len(src)} 条 -> {out_dir}", flush=True)
        run_fsn_inference(fsn_root, cfg_inf, ckpt, in_dir, out_dir, args.gpu)
        flatten_enhanced(out_dir)
        got = wavs_in(out_dir)
        assert len(got) == len(src) and set(got) == set(src), \
            f"{out_dir} 增强不完整: {len(got)}/{len(src)}"
    print(f"[bench] 完成：{out_root}/test_0db, test_m5db（enroll 复用 8.20 clean 注册）")


# ============================================================
# 工具
# ============================================================

def _data820(cfg, base):
    return resolve_existing_path(cfg["paths"]["data_dir_820"], base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_matched.toml")
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--mix", action="store_true")
    ap.add_argument("--enhance", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--all", action="store_true", help="mix + enhance + bench + manifest")
    ap.add_argument("--limit", type=int, default=None,
                    help="只处理前 N 条训练语句（冒烟；正式训练语料不要带此参数）")
    args = ap.parse_args()
    if not (args.all or args.mix or args.enhance or args.bench):
        ap.print_help()
        return

    cfg = toml.load(args.config)
    base = os.path.dirname(os.path.abspath(args.config))
    p_cfg = cfg["paths"]
    ecfg, f_cfg = cfg["enh_corpus"], cfg["frontend"]

    data820 = _data820(cfg, base)
    train_csv = os.path.join(data820, "train_utts.csv")
    assert os.path.isfile(train_csv), f"缺 {train_csv}（先跑 8.20 的 prepare_multicond_data.py）"
    rows, _ = read_train_csv(train_csv)
    if args.limit:
        rows = rows[:args.limit]
        print(f"[data] --limit {args.limit}：只处理前 {len(rows)} 条（冒烟）")
    print(f"[data] 训练清单 {len(rows)} 条 × SNR {ecfg['snrs']} -> "
          f"{len(rows) * len(ecfg['snrs'])} 个增强样本")

    corpus_root = os.path.join(base, p_cfg["enh_corpus_dir"]) \
        if not os.path.isabs(p_cfg["enh_corpus_dir"]) else p_cfg["enh_corpus_dir"]
    os.makedirs(corpus_root, exist_ok=True)

    if args.all or args.mix:
        step_mix(cfg, base, rows, corpus_root, ecfg)
    if args.all or args.enhance:
        step_enhance(cfg, base, rows, corpus_root, ecfg, f_cfg, args)
        step_manifest(cfg, base, rows, corpus_root, ecfg, f_cfg,
                      complete=args.limit is None)
    if args.all or args.bench:
        step_bench(cfg, base, corpus_root, ecfg, f_cfg, args)

    print("\n[prepare] 完成。下一步：bash run_train.sh（训练会校验 manifest）")


if __name__ == "__main__":
    main()
