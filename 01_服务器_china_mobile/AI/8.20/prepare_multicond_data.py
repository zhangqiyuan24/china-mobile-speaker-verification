# -*- coding: utf-8 -*-
"""
E2 数据准备：说话人/噪声三池隔离 + 验证集落盘

做四件事（全部确定性，seed=2026）：
  1. 说话人划分：train 说话人（排除正式 test 说话人）中，排序后取末尾 100 人做验证；
     其余做训练。三方隔离：train_spk / val_spk / trial_spk。
  2. 训练语句：从训练说话人的全部语句中等间隔均匀抽取 30k 条 → train_utts.csv。
  3. 噪声划分：noise_train_freesound.txt(675) 排序后末尾 60 条做验证噪声，其余 615 条
     做训练噪声；与 noise_test_freesound.txt(168) 三方隔离（assert 零交集）。
  4. 验证 bench 落盘：每个验证说话人 4 句（2 句 clean 注册 + 2 句测试），测试句用
     验证噪声混 0dB / -5dB（与正式测试集同一公式），3s 随机裁剪，写 val_trials。

注意：
  - 训练用噪声只有 train noise pool；test noise 完全隔离，绝不参与训练/验证。
  - 验证 bench 是“落盘”的固定文件，训练中每个 epoch 用同一批文件评测，保证
     epoch 间可比、且与 epoch0（预训练模型基线）可比。
  - 本脚本只准备数据，不加载任何模型，可在 CPU 上运行。

用法：
  python prepare_multicond_data.py --config config_robust.toml
  python prepare_multicond_data.py --config config_robust.toml --max-train-utts 3000  # 快速试跑
"""

import os
import json
import argparse
import numpy as np
import toml

from common_utils import (
    resolve_existing_path, uniform_select, load_wav, random_crop, mix_noise,
    measure_snr, spk_of_utt, speaker_ids_from_trials, SR,
)


def save_wav(path, wav, sr=SR):
    try:
        import soundfile as sf
        sf.write(path, wav.astype(np.float32), sr, subtype="PCM_16")
    except ImportError:
        from scipy.io import wavfile
        wav16 = np.clip(wav, -1.0, 1.0)
        wavfile.write(path, sr, (wav16 * 32767).astype(np.int16))


def load_scp_with_spk(clean_scp, sv_ecapa):
    """all_wav.scp -> [(utt_id, spk, abs_path), ...]（保序）"""
    train_root = os.path.join(sv_ecapa, "data", "data_for_train")
    out = []
    with open(clean_scp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id, rel = line.split()[0], line.split()[-1]
            if os.path.isabs(rel):
                p = rel
            else:
                if rel.startswith("data/"):
                    rel = rel[len("data/"):]
                p = os.path.join(train_root, rel)
            p = resolve_existing_path(p, sv_ecapa)
            out.append((utt_id, spk_of_utt(utt_id), p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_robust.toml")
    ap.add_argument("--max-train-utts", type=int, default=None,
                    help="覆盖 config 中的 max_train_utts")
    ap.add_argument("--utts-per-spk", type=int, default=None,
                    help="覆盖 config 中的 utts_per_spk（0=不限制；N=每人先均匀取 N 句再总体抽取，"
                         "30k 预算下用于平衡“说话人多样性 vs 每类样本数”）")
    ap.add_argument("--num-val-speakers", type=int, default=None)
    ap.add_argument("--num-val-utts", type=int, default=None)
    args = ap.parse_args()

    cfg = toml.load(args.config)
    p_cfg = cfg["paths"]
    base = os.path.dirname(os.path.abspath(args.config))

    sv_ecapa = resolve_existing_path(p_cfg["sv_ecapa"], base)

    def rel(p):
        """config 里的相对路径按 config 所在目录解析为绝对路径。"""
        return resolve_existing_path(
            p if os.path.isabs(p) else os.path.join(base, p), base)

    clean_scp = rel(p_cfg["clean_scp"])

    noise_train_list = rel(p_cfg["noise_train_list"])
    noise_test_list = rel(p_cfg["noise_test_list"])
    trial_path = rel(p_cfg["trials"])

    d_cfg = cfg["data"]
    num_val_spk = args.num_val_speakers or d_cfg["num_val_speakers"]
    num_val_utts = args.num_val_utts or d_cfg["num_val_utts"]   # 每人 2 注册 + 2 测试
    max_train_utts = args.max_train_utts if args.max_train_utts is not None else d_cfg["max_train_utts"]
    utts_per_spk = args.utts_per_spk if args.utts_per_spk is not None else d_cfg.get("utts_per_spk", 0)
    seed = d_cfg["seed"]

    out_dir = os.path.join(base, p_cfg["exp_dir"], "data")
    bench_dir = os.path.join(out_dir, "val_bench")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(seed)

    # ==========================================================
    # 1) 说话人三方隔离
    # ==========================================================
    print("== 1) 说话人划分 ==")
    utts = load_scp_with_spk(clean_scp, sv_ecapa)
    by_spk = {}
    for u, s, p in utts:
        by_spk.setdefault(s, []).append((u, p))
    all_spks = sorted(by_spk.keys())
    print(f"   全量: {len(utts)} 条 / {len(all_spks)} 人")

    trial_spks = speaker_ids_from_trials(trial_path)
    overlap_with_test = [s for s in all_spks if s in trial_spks]
    print(f"   正式 test 说话人: {len(trial_spks)} 人；与训练集重叠: {len(overlap_with_test)} 人"
          + ("（将剔除，保证开集协议）" if overlap_with_test else "（零重叠，开集协议成立）"))

    # 验证说话人：句数 >= num_val_utts 的说话人中，排序取末尾 num_val_spk 人
    eligible = [s for s in all_spks if s not in trial_spks and len(by_spk[s]) >= num_val_utts]
    val_spks = eligible[-num_val_spk:]
    train_spks = [s for s in all_spks if s not in trial_spks and s not in set(val_spks)]
    assert len(val_spks) == num_val_spk, f"可用验证说话人不足: {len(eligible)} < {num_val_spk}"
    assert not (set(val_spks) & set(train_spks))
    assert not (set(val_spks) & trial_spks)
    assert not (set(train_spks) & trial_spks)
    print(f"   train_spk={len(train_spks)}  val_spk={len(val_spks)}  "
          f"val 取自排序末尾（{val_spks[0]} .. {val_spks[-1]}）")

    # ==========================================================
    # 2) 训练语句（均匀抽取）
    # ==========================================================
    print("== 2) 训练语句抽取 ==")
    train_utts_all = [(u, s, p) for s in train_spks for (u, p) in by_spk[s]]
    if utts_per_spk and utts_per_spk > 0:
        # 每人先确定性取前 utts_per_spk 句（按 utt_id 排序），再总体均匀抽取
        train_utts_all = [(u, s, p) for s in train_spks
                          for (u, p) in sorted(by_spk[s])[:utts_per_spk]]
        print(f"   utts_per_spk={utts_per_spk} -> 池缩小到 {len(train_utts_all)} 条")
    selected = uniform_select(train_utts_all, max_train_utts)
    n_spk_sel = len(set(s for _, s, _ in selected))
    print(f"   训练池 {len(train_utts_all)} 条 -> 均匀抽取 {len(selected)} 条 / {n_spk_sel} 人 "
          f"(平均 {len(selected)/n_spk_sel:.1f} 句/人)")
    with open(os.path.join(out_dir, "train_utts.csv"), "w") as f:
        f.write("utt_id,spk,wav_path\n")
        for u, s, p in selected:
            f.write(f"{u},{s},{p}\n")

    # ==========================================================
    # 3) 噪声三方隔离
    # ==========================================================
    print("== 3) 噪声划分 ==")
    with open(noise_train_list) as f:
        noise_train_all = sorted(l.strip() for l in f if l.strip())
    with open(noise_test_list) as f:
        noise_test = sorted(l.strip() for l in f if l.strip())
    noise_val = noise_train_all[-d_cfg["num_val_noises"]:]
    noise_train = noise_train_all[:-d_cfg["num_val_noises"]]
    assert len(set(noise_train) & set(noise_val)) == 0
    assert len(set(noise_train) & set(noise_test)) == 0
    assert len(set(noise_val) & set(noise_test)) == 0
    print(f"   noise_train={len(noise_train)}  noise_val={len(noise_val)}  noise_test={len(noise_test)}（三方零交集）")

    def write_list(name, items):
        with open(os.path.join(out_dir, name), "w") as f:
            f.write("\n".join(items) + "\n")
    write_list("noise_train.txt", noise_train)
    write_list("noise_val.txt", noise_val)
    with open(os.path.join(out_dir, "train_spks.txt"), "w") as f:
        f.write("\n".join(train_spks) + "\n")
    with open(os.path.join(out_dir, "val_spks.txt"), "w") as f:
        f.write("\n".join(val_spks) + "\n")

    # ==========================================================
    # 4) 验证 bench 落盘（clean 注册 + 三条件测试）
    # ==========================================================
    print("== 4) 验证 bench 落盘 ==")
    crop_len = int(d_cfg["crop_seconds"] * SR)
    sub_dirs = {
        "enroll": os.path.join(bench_dir, "enroll"),      # clean 3s
        "test_clean": os.path.join(bench_dir, "test_clean"),
        "test_0db": os.path.join(bench_dir, "test_0db"),
        "test_m5db": os.path.join(bench_dir, "test_m5db"),
    }
    for d in sub_dirs.values():
        os.makedirs(d, exist_ok=True)

    val_trials = []          # (enroll_stem, test_stem, label)
    enroll_ids_by_spk, test_ids_by_spk = [], []
    n_written = 0
    for spk in val_spks:
        utt_pool = sorted(by_spk[spk])                      # 确定性
        picked = uniform_select(utt_pool, num_val_utts)     # 2 注册 + 2 测试
        enroll_utts, test_utts = picked[:num_val_utts // 2], picked[num_val_utts // 2:]

        enroll_ids = []
        for utt_id, path in enroll_utts:
            wav3 = random_crop(load_wav(path), crop_len, rng)
            save_wav(os.path.join(sub_dirs["enroll"], f"{utt_id}.wav"), wav3)
            enroll_ids.append(utt_id)
            n_written += 1

        # 每个测试句固定 (noise, offset)，0dB 与 -5dB 用同一段噪声（与正式集同设定）
        test_specs = []
        for utt_id, path in test_utts:
            wav3 = random_crop(load_wav(path), crop_len, rng)
            noise_path = noise_val[rng.randint(0, len(noise_val))]
            noise_full = load_wav(resolve_existing_path(noise_path, sv_ecapa))
            noise_start = rng.randint(0, max(1, len(noise_full) - crop_len)) if len(noise_full) > crop_len else 0
            test_specs.append((utt_id, wav3, noise_full, noise_start))

        for cond, snr_db in (("test_clean", None), ("test_0db", 0.0), ("test_m5db", -5.0)):
            for utt_id, wav3, noise_full, noise_start in test_specs:
                if snr_db is None:
                    mixed = wav3
                else:
                    mixed, _ = mix_noise(wav3, noise_full, snr_db, noise_start)
                save_wav(os.path.join(sub_dirs[cond], f"{utt_id}.wav"), mixed)
                n_written += 1

        # target = 同说话人全部 enroll×test 对
        test_ids = [t for t, _, _, _ in test_specs]
        for e in enroll_ids:
            for t in test_ids:
                val_trials.append((e, t, 1))
        enroll_ids_by_spk.append(enroll_ids)
        test_ids_by_spk.append(test_ids)

    # nontarget = 跨说话人环形配对（i 的注册 vs 下一个人的测试），规模 O(N) 且标签清晰
    for i in range(len(val_spks)):
        j = (i + 1) % len(val_spks)
        if j == i:
            continue
        for e in enroll_ids_by_spk[i]:
            for t in test_ids_by_spk[j]:
                val_trials.append((e, t, 0))

    with open(os.path.join(out_dir, "val_trials.txt"), "w") as f:
        for e, t, lab in val_trials:
            f.write(f"{e} {t} {'target' if lab else 'nontarget'}\n")
    n_tgt = sum(1 for _, _, l in val_trials if l == 1)
    print(f"   写出 {n_written} 个 wav；trials={len(val_trials)}（target={n_tgt}, "
          f"nontarget={len(val_trials) - n_tgt}）")

    # ---- SNR 抽检：落盘文件实测 ----
    print("== 5) SNR 抽检（实测落盘文件）==")
    import glob
    for cond, expect in (("test_0db", 0.0), ("test_m5db", -5.0)):
        files = sorted(glob.glob(os.path.join(sub_dirs[cond], "*.wav")))
        picks = files[:: max(1, len(files) // 3)][:3]
        snrs = []
        for fp in picks:
            stem = os.path.splitext(os.path.basename(fp))[0]
            clean_fp = os.path.join(sub_dirs["test_clean"], os.path.basename(fp))
            snrs.append(measure_snr(load_wav(clean_fp), load_wav(fp)))
        print(f"   {cond}: 期望 {expect:+.0f} dB, 实测 {['%.2f' % s for s in snrs]} dB")

    # ==========================================================
    # manifest
    # ==========================================================
    manifest = {
        "seed": seed,
        "num_train_utts": len(selected),
        "num_train_spks_selected": n_spk_sel,
        "utts_per_spk": utts_per_spk,
        "num_train_spks_pool": len(train_spks),
        "num_val_spks": len(val_spks),
        "num_val_utts_per_spk": num_val_utts,
        "num_trials": len(val_trials),
        "num_trials_target": n_tgt,
        "noise_train": len(noise_train),
        "noise_val": len(noise_val),
        "noise_test_isolated": len(noise_test),
        "trial_spk_overlap_with_train": len(overlap_with_test),
        "val_bench_dir": bench_dir,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n[prepare] 完成。输出目录:", out_dir)
    print("[prepare] 隔离检查全部通过：说话人三方零交集、噪声三方零交集。")


if __name__ == "__main__":
    main()
