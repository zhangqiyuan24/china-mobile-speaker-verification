# -*- coding: utf-8 -*-
import glob, os, shutil, csv
CM = "/data/home/ligangrui/china_mobile"
SV = CM + "/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa"
R5 = CM + "/AI/8.27/exp/tse_v3_ns05duraug_r5"
ST = CM + "/AI/9.3/交付材料_9.3/07_增强音频样本"

# 清掉之前挑错的文件,重来
for f in glob.glob(os.path.join(ST, "*.wav")) + glob.glob(os.path.join(ST, "样本对照表.tsv")):
    os.remove(f)

def spk_of(stem):
    parts = stem.split("_")
    return "_".join(parts[:3])          # 3D_SPK_XXXXX

def find_one(root, stem):
    g = glob.glob(os.path.join(root, "*", "*", stem + ".wav"))
    return g[0] if g else None

rows = []
def add(cond, idx, stem, enh_path):
    spk = spk_of(stem)
    raw_root = os.path.join(SV, "data", "data_for_test", cond, "3s")
    raw = find_one(raw_root, stem)
    enr = sorted(glob.glob(os.path.join(SV, "data", "data_for_test", "clean", "3s", "*", spk, "*.wav")))[:2]
    tag = "%s_%02d_%s" % (cond, idx, spk)
    dsts = []
    if raw: dsts.append((raw, tag + "_1原始带噪.wav"))
    if enh_path: dsts.append((enh_path, tag + "_2增强后.wav"))
    for j, e in enumerate(enr): dsts.append((e, "%s_reg%d_注册参考.wav" % (tag, j + 1)))
    for src, name in dsts:
        shutil.copy(src, os.path.join(ST, name))
        rows.append((name, os.path.relpath(src, CM)))

def pick(pattern_dir, n, cond):
    wavs = sorted(glob.glob(os.path.join(pattern_dir, "*.wav")))
    step = max(1, len(wavs) // n)
    for i in range(n):
        p = wavs[i * step]
        add(cond, i + 1, os.path.splitext(os.path.basename(p))[0], p)

pick(R5 + "/enhanced_official/noise_0db", 4, "noise_0db")
pick(R5 + "/enhanced_official/noise_m5db", 3, "noise_m5db")
pick(CM + "/AI/9.3/fe_clean/enhanced_clean/clean", 2, "clean")

with open(os.path.join(ST, "样本对照表.tsv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["样本文件", "服务器源路径(相对 china_mobile/)"])
    for r in rows: w.writerow(r)
print("audio files:", len(rows), "speakers:", len(set(r[0].split("_")[2] + r[0].split("_")[3] for r in rows)))
