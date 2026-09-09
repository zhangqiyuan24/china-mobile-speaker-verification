#!/usr/bin/env python
"""
统计 reverb benchmark 中实际使用的 RIR T60 分布。

用法：python stat_reverb_t60.py
"""

import csv, sys
import numpy as np
from pathlib import Path

METADATA_DIR = Path("data/data_for_test/reverb")


def main():
    t60_all = []

    for meta_csv in sorted(METADATA_DIR.rglob("reverb_metadata.csv")):
        dur = meta_csv.parent.name  # "1s", "3s", "5s"
        values = []
        with open(meta_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                v = row.get("rir_t60_sec", "")
                if v and v != "N/A":
                    try:
                        values.append(float(v))
                    except ValueError:
                        pass
        t60_all.extend(values)

        if values:
            arr = np.array(values)
            print(f"  [{dur}] n={len(arr)}  "
                  f"mean={np.mean(arr):.3f}s  median={np.median(arr):.3f}s  "
                  f"min={np.min(arr):.3f}s  max={np.max(arr):.3f}s  "
                  f"std={np.std(arr):.3f}s")
        else:
            print(f"  [{dur}] 无数据")

    if t60_all:
        arr = np.array(t60_all)
        print(f"\n  [全部] n={len(arr)}  "
              f"mean={np.mean(arr):.3f}s  median={np.median(arr):.3f}s  "
              f"min={np.min(arr):.3f}s  max={np.max(arr):.3f}s")


if __name__ == "__main__":
    main()
