#!/usr/bin/env python
"""
结果汇总脚本
============

遍历 data/result/ 下所有 result.metrics，同时从 metadata CSV 和
condition 名称推导 SNR / TIR / T60 等参数，输出完整汇总 CSV。

用法：python gen_results_summary.py
输出：data/result/results_summary.csv
"""

import os, sys, csv, re
import numpy as np
from pathlib import Path

RESULT_BASE = Path("data/result")
OUTPUT_CSV = RESULT_BASE / "results_summary.csv"


def parse_metrics(filepath):
    eer, mindcf = None, None
    try:
        with open(filepath) as f:
            for line in f:
                m_eer = re.search(r'EER\s*=\s*([\d.]+)', line)
                m_dcf = re.search(r'minDCF.*=\s*([\d.]+)', line)
                if m_eer:
                    eer = float(m_eer.group(1))
                if m_dcf:
                    mindcf = float(m_dcf.group(1))
    except Exception:
        pass
    return eer, mindcf


def load_t60_from_metadata():
    """
    从 data/data_for_test/reverb/*/reverb_metadata.csv 读取 T60 中位数。
    → {"reverb": t60_median}
    """
    data_root = Path("data/data_for_test")
    t60_values = []

    for meta_csv in sorted(data_root.rglob("reverb_metadata.csv")):
        try:
            with open(meta_csv, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    v = row.get("rir_t60_sec", "")
                    if v and v != "" and v != "N/A":
                        try:
                            t60_values.append(float(v))
                        except ValueError:
                            pass
        except Exception:
            continue

    if t60_values:
        return {"reverb": round(np.median(t60_values), 4)}
    return {}


def infer_params(condition):
    """
    从 condition 名推导 SNR / TIR / interferer 信息（无需 metadata 文件）。
    """
    info = {}

    # SNR: noise_0db → 0, noise_m5db → -5
    if condition.startswith("noise_"):
        if "m5db" in condition:
            info["SNR_dB"] = -5
        elif "0db" in condition:
            info["SNR_dB"] = 0

    # TIR + interferer: 所有 overlap 条件 TIR=0dB
    if condition.startswith("overlap"):
        info["TIR_dB"] = 0
        if "overlap2" in condition:
            info["num_interferers"] = 1
        elif "overlap3" in condition:
            info["num_interferers"] = 2
        if "_cn" in condition:
            info["interferer_lang"] = "zh"
        elif "_en" in condition:
            info["interferer_lang"] = "en"

    return info


def scan_results():
    rows = []
    t60_map = load_t60_from_metadata()

    for metrics_file in sorted(RESULT_BASE.rglob("result.metrics")):
        rel = metrics_file.relative_to(RESULT_BASE)
        parts = rel.parts

        eer, mindcf = parse_metrics(metrics_file)
        if eer is None and mindcf is None:
            continue

        # 解析路径层级
        if len(parts) == 3:
            model, dur, _ = parts          # clean: model/dur/result.metrics
            condition = "clean"
        elif len(parts) == 4:
            condition, model, dur, _ = parts
        else:
            continue

        dur_sec = int(dur.replace("s", ""))

        # 合并参数：推导值 + metadata
        params = infer_params(condition)
        if condition in t60_map:
            params["T60_sec"] = t60_map[condition]

        snr = params.get("SNR_dB", "")
        tir = params.get("TIR_dB", "")
        t60 = params.get("T60_sec", "")
        int_lang = params.get("interferer_lang", "")
        int_num = params.get("num_interferers", "")

        rows.append({
            "model": model,
            "condition": condition,
            "duration_sec": dur_sec,
            "EER_%": eer,
            "minDCF": mindcf,
            "SNR_dB": snr,
            "TIR_dB": tir,
            "T60_sec": t60,
            "interferer_lang": int_lang,
            "num_interferers": int_num,
        })

    # 排序
    cond_order = ["clean", "noise_0db", "noise_m5db", "reverb",
                  "overlap2_cn", "overlap2_en", "overlap3_cn", "overlap3_en"]
    model_order = ["ecapa", "eres2net", "redimnet2"]

    def sort_key(r):
        c = cond_order.index(r["condition"]) if r["condition"] in cond_order else 99
        m = model_order.index(r["model"]) if r["model"] in model_order else 99
        return (c, r["duration_sec"], m)

    rows.sort(key=sort_key)
    return rows


def main():
    print("=" * 60)
    print("  结果汇总")
    print("=" * 60)

    rows = scan_results()
    print(f"\n  共找到 {len(rows)} 条结果")

    if not rows:
        print("  [WARN] 未找到任何 result.metrics")
        return

    # 检查 metadata 是否读到
    t60_map = load_t60_from_metadata()
    if t60_map:
        print(f"  T60 (metadata): reverb median = {t60_map.get('reverb', 'N/A')}s")
    else:
        print(f"  [WARN] 未找到 reverb_metadata.csv，T60 列将为空")

    os.makedirs(RESULT_BASE, exist_ok=True)

    fieldnames = [
        "model", "condition", "duration_sec",
        "EER_%", "minDCF",
        "SNR_dB", "TIR_dB", "T60_sec",
        "interferer_lang", "num_interferers",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  汇总表 → {OUTPUT_CSV}")
    print(f"\n{'=' * 92}")
    print(f"{'Model':12s} {'Condition':16s} {'Dur':4s} {'EER%':8s} {'minDCF':8s} "
          f"{'SNR':6s} {'TIR':6s} {'T60':8s} {'IntLang':8s} {'Int#':5s}")
    print(f"{'=' * 92}")

    for r in rows:
        eer_s = f"{r['EER_%']:.2f}" if r['EER_%'] is not None else "N/A"
        dcf_s = f"{r['minDCF']:.4f}" if r['minDCF'] is not None else "N/A"
        snr_s = f"{r['SNR_dB']}" if r['SNR_dB'] != "" else "-"
        tir_s = f"{r['TIR_dB']}" if r['TIR_dB'] != "" else "-"
        t60_s = f"{r['T60_sec']}" if r['T60_sec'] != "" else "-"
        il_s = f"{r['interferer_lang']}" if r['interferer_lang'] != "" else "-"
        in_s = f"{r['num_interferers']}" if r['num_interferers'] != "" else "-"

        print(f"{r['model']:12s} {r['condition']:16s} {r['duration_sec']}s  "
              f"{eer_s:8s} {dcf_s:8s} "
              f"{snr_s:6s} {tir_s:6s} {t60_s:8s} {il_s:8s} {in_s:5s}")


if __name__ == "__main__":
    main()
