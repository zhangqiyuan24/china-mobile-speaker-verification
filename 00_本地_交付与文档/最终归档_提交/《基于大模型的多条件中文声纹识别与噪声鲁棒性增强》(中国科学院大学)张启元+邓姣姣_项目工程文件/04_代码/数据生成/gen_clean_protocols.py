#!/usr/bin/env python
"""
Clean Benchmark 评测协议生成脚本（子课题一）
==============================================

基于 gen_clean_benchmark.py 生成的 clean_metadata.csv，生成评测协议。

核心保证：1s / 3s / 5s 三档时长使用**完全相同**的 trial 配对（同一对说话人、
同一对 base utterance），仅 utt_id 的时长后缀不同。确保时长对比实验中，
EER 变化唯一来源于语音长度，不受 trial 差异污染。

对齐官方 3D-Speaker trial 构建方式：
  - target:   A 内同说话人任意两 utterance 组对 C(n,2)
  - nontarget: A vs D（D 是唯一负样本来源，B/C/E/F 不参与 clean 评测）
  - 比例 target:nontarget = 1:5

输出（每个时长目录下，如 clean/1s/）：
  enrollment.csv          — 推荐注册语音（A 每说话人 1 条）
  trial.csv               — 主 trial（含 cross_* 布尔列）
  trials_all              — Kaldi 格式全量
  trials_cross_device     — Kaldi 格式跨设备子集
  trials_cross_distance   — Kaldi 格式跨距离子集
  trials_cross_dialect    — Kaldi 格式跨方言子集

输出（clean/ 层级）：
  speaker_split.csv       — 全部说话人 → role / language

用法：python gen_clean_protocols.py
      先运行 gen_clean_benchmark.py
"""

import os, csv, sys, random
from pathlib import Path
from itertools import combinations
from collections import defaultdict

SEED = 2026
TARGET_RATIO = 5  # nontarget : target

CLEAN_DIR = Path("data/data_for_test/clean")
METADATA_CSV = CLEAN_DIR / "clean_metadata.csv"
DURATIONS = [1, 3, 5]


# ============================================================
# 工具
# ============================================================

def utt_stem(row):
    return Path(row["wav_path"]).stem


def base_stem(row):
    """去除时长后缀，得到跨时长一致的 base id。
    例: 3D_SPK_06154_001_Device03_Distance00_Dialect00_1s
      → 3D_SPK_06154_001_Device03_Distance00_Dialect00
    """
    s = utt_stem(row)
    for d in DURATIONS:
        suffix = f"_{d}s"
        if s.endswith(suffix):
            return s[:-len(suffix)]
    return s  # fallback（不应发生）


def parse_condition_fields(path_or_utt):
    name = os.path.basename(str(path_or_utt))
    info = {"device": None, "distance": None, "dialect": None}
    for token in name.replace(".wav", "").split("_"):
        if token.startswith("Device"):
            info["device"] = token
        elif token.startswith("Distance"):
            info["distance"] = token
        elif token.startswith("Dialect"):
            info["dialect"] = token
    return info


def load_all_metadata():
    """
    → {
        base_stem: {
            "role": ...,
            "speaker_id": ...,
            "device": ..., "distance": ..., "dialect": ...,
            "rows": {1: row, 3: row, 5: row},  # 每个时长一行
        }
      }
    """
    base_map = {}
    with open(METADATA_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dur = int(row["target_duration_sec"])
            if dur not in DURATIONS:
                continue
            bs = base_stem(row)
            if bs not in base_map:
                info = parse_condition_fields(row["wav_path"])
                base_map[bs] = {
                    "role": row["role"],
                    "speaker_id": row["speaker_id"],
                    "device": info["device"],
                    "distance": info["distance"],
                    "dialect": info["dialect"],
                    "rows": {},
                }
            base_map[bs]["rows"][dur] = row
    return base_map


# ============================================================
# Trial 配对生成（只做一次，基于 base stem）
# ============================================================

def build_trial_pairings(base_map):
    """
    从 base_map 中选 A 和 D，构建 trial 配对（base stem 级别）。
    Returns: [(enr_base, test_base, label, device_cross, distance_cross, dialect_cross)]
    """
    random.seed(SEED)

    # ── 收集 A & D ──
    a_bases = defaultdict(list)   # spk_id → [base_stem]
    d_bases = []                  # [base_stem]
    a_all = []                    # [base_stem]（所有 A，用于 nontarget 采样）

    for bs, info in base_map.items():
        if info["role"] == "A":
            a_bases[info["speaker_id"]].append(bs)
            a_all.append(bs)
        elif info["role"] == "D":
            d_bases.append(bs)

    # ── target: A 内同说话人 C(n,2) ──
    target_pairs = []
    for spk, bases in a_bases.items():
        for b1, b2 in combinations(sorted(bases), 2):
            target_pairs.append((b1, b2))

    n_target = len(target_pairs)

    # ── nontarget: A vs D ──
    n_nontarget = n_target * TARGET_RATIO
    random.shuffle(a_all)
    random.shuffle(d_bases)

    nontarget_pairs = []
    seen = set()
    i, attempts = 0, 0
    while len(nontarget_pairs) < n_nontarget and attempts < n_nontarget * 5 and d_bases:
        a_bs = a_all[i % len(a_all)]
        d_bs = d_bases[(i * 7 + 3) % len(d_bases)]
        key = (a_bs, d_bs)
        if key not in seen:
            seen.add(key)
            nontarget_pairs.append((a_bs, d_bs))
        i += 1
        attempts += 1

    # ── 合并：target 在前，nontarget 在后，不 shuffle，便于人工核对 ──
    all_pairs = ([(b1, b2, "target") for b1, b2 in target_pairs] +
                 [(b1, b2, "nontarget") for b1, b2 in nontarget_pairs])

    # ── 附加条件标记 ──
    labelled_pairs = []
    for enr, test, label in all_pairs:
        ei = base_map[enr]
        ti = base_map[test]
        labelled_pairs.append((
            enr, test, label,
            (ei["device"] is not None and ti["device"] is not None
             and ei["device"] != ti["device"]),
            (ei["distance"] is not None and ti["distance"] is not None
             and ei["distance"] != ti["distance"]),
            (ei["dialect"] is not None and ti["dialect"] is not None
             and ei["dialect"] != ti["dialect"]),
            ei, ti,
        ))

    return labelled_pairs, n_target, len(nontarget_pairs)


# ============================================================
# 按时长写出
# ============================================================

def write_duration_files(labelled_pairs, base_map, dur, out_dir):
    """将 base-stem 配对转换为带时长后缀的 utt_id，写出全部文件"""
    os.makedirs(out_dir, exist_ok=True)

    def enr_id(bs, info):
        # info 是 base_map[bs]，直接用其 device/distance/dialect
        return f"{bs}_{dur}s"

    def tst_id(bs):
        return f"{bs}_{dur}s"

    # ── trial.csv ──
    trial_csv = out_dir / "trial.csv"
    with open(trial_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "enrollment_utt_id", "test_utt_id", "label",
            "enr_device", "test_device", "cross_device",
            "enr_distance", "test_distance", "cross_distance",
            "enr_dialect", "test_dialect", "cross_dialect",
        ])
        w.writeheader()
        for enr, test, label, cross_dv, cross_ds, cross_dl, ei, ti in labelled_pairs:
            w.writerow({
                "enrollment_utt_id": enr_id(enr, ei),
                "test_utt_id": tst_id(test),
                "label": label,
                "enr_device": ei["device"] or "",
                "test_device": ti["device"] or "",
                "cross_device": cross_dv,
                "enr_distance": ei["distance"] or "",
                "test_distance": ti["distance"] or "",
                "cross_distance": cross_ds,
                "enr_dialect": ei["dialect"] or "",
                "test_dialect": ti["dialect"] or "",
                "cross_dialect": cross_dl,
            })

    # ── Kaldi 格式 ──
    def write_kaldi(fpath, predicate):
        with open(fpath, "w") as f:
            for enr, test, label, cross_dv, cross_ds, cross_dl, ei, ti in labelled_pairs:
                if predicate(cross_dv, cross_ds, cross_dl):
                    f.write(f"{enr_id(enr, ei)} {tst_id(test)} {label}\n")

    write_kaldi(out_dir / "trials_all",
                lambda dv, ds, dl: True)
    write_kaldi(out_dir / "trials_cross_device",
                lambda dv, ds, dl: dv)
    write_kaldi(out_dir / "trials_cross_distance",
                lambda dv, ds, dl: ds)
    write_kaldi(out_dir / "trials_cross_dialect",
                lambda dv, ds, dl: dl)

    # 统计
    counts = {}
    for fname in ["trials_all", "trials_cross_device", "trials_cross_distance",
                  "trials_cross_dialect"]:
        fpath = out_dir / fname
        n = sum(1 for _ in open(fpath))
        t = sum(1 for l in open(fpath) if l.strip().endswith(" target"))
        counts[fname] = (n, t)
    return counts


def write_enrollment(base_map, dur, out_dir):
    """推荐注册语音：A 每说话人第 1 条（按 base stem 排序）"""
    a_by_spk = defaultdict(list)
    for bs, info in base_map.items():
        if info["role"] == "A":
            a_by_spk[info["speaker_id"]].append(bs)

    enr_csv = out_dir / "enrollment.csv"
    with open(enr_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["speaker_id", "utt_id", "wav_path",
                                          "device", "distance", "dialect"])
        w.writeheader()
        for spk in sorted(a_by_spk):
            # 取 base stem 排序后的第一条
            best_bs = sorted(a_by_spk[spk])[0]
            info = base_map[best_bs]
            dur_row = info["rows"].get(dur)
            wav_path = dur_row["wav_path"] if dur_row else ""
            w.writerow({
                "speaker_id": spk,
                "utt_id": f"{best_bs}_{dur}s",
                "wav_path": wav_path,
                "device": info["device"] or "",
                "distance": info["distance"] or "",
                "dialect": info["dialect"] or "",
            })


def write_speaker_split(base_map):
    """汇总所有 base stem 中的说话人 → speaker_split.csv"""
    out_path = CLEAN_DIR / "speaker_split.csv"
    all_spks = {}
    for bs, info in base_map.items():
        spk = info["speaker_id"]
        if spk not in all_spks:
            all_spks[spk] = {
                "role": info["role"],
                "language": "en" if info["role"] in ("E", "F") else "zh",
            }
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["speaker_id", "role", "language"])
        w.writeheader()
        for spk in sorted(all_spks):
            w.writerow({"speaker_id": spk, **all_spks[spk]})

    rc = defaultdict(int)
    for v in all_spks.values():
        rc[v["role"]] += 1
    print(f"\n  speaker_split: {len(all_spks)} 说话人 → {out_path}")
    for r in ["A", "B", "C", "D", "E", "F"]:
        print(f"    {r}: {rc.get(r, 0)} 人")


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("  Clean Benchmark 评测协议生成")
    print("  1s / 3s / 5s 三档完全共享同一套 trial 配对")
    print("=" * 60)

    if not METADATA_CSV.exists():
        print(f"\n[ERROR] 找不到 {METADATA_CSV}")
        print("  先运行: python gen_clean_benchmark.py")
        sys.exit(1)

    # 1. 加载所有 metadata，按 base stem 合并
    base_map = load_all_metadata()
    print(f"\n  共 {len(base_map)} 个 base utterance")

    # 2. 构建一次 trial 配对（base stem 级别）
    labelled_pairs, n_target, n_nontarget = build_trial_pairings(base_map)
    print(f"  Target:   {n_target}")
    print(f"  Nontarget: {n_nontarget}  (ratio 1:{n_nontarget // max(n_target, 1)})")
    print(f"  Total:    {len(labelled_pairs)}")

    # 3. 每个时长写一份（配对完全相同，仅 utt_id 后缀不同）
    for dur in DURATIONS:
        out_dir = CLEAN_DIR / f"{dur}s"
        print(f"\n{'─' * 50}")
        print(f"  时长: {dur}s → {out_dir}")

        # enrollment
        write_enrollment(base_map, dur, out_dir)

        # trials
        counts = write_duration_files(labelled_pairs, base_map, dur, out_dir)
        for fname in ["trials_all", "trials_cross_device", "trials_cross_distance",
                      "trials_cross_dialect"]:
            n, t = counts[fname]
            print(f"    {fname}: {n} trials (target={t}, nontarget={n-t})")

    # 4. speaker_split
    write_speaker_split(base_map)

    print(f"\n{'=' * 60}")
    print("  完成。1s/3s/5s 三套协议 trial 配对完全一致。")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
