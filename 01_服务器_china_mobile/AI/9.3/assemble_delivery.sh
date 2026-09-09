#!/bin/bash
# assemble_delivery.sh — 在服务器上装配 交付材料_9.3/(本地交付材料的备份 + 交付物汇集)
set -u
CM=/data/home/ligangrui/china_mobile
AI=$CM/AI
SV=$CM/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa
R5=$AI/8.27/exp/tse_v3_ns05duraug_r5
FIN=$AI/8.28/final_eval_r5
ST=$AI/9.3/交付材料_9.3

mkdir -p $ST/01_报告 $ST/02_汇报PPT与讲稿 $ST/03_结果汇总 $ST/04_代码 $ST/05_权重 $ST/06_指标 $ST/07_增强音频样本 $ST/08_图

# ---- 03 结果汇总 ----
cp $AI/9.3/README_实验结果汇总.md $ST/03_结果汇总/ 2>/dev/null
cp $AI/9.3/模型参数量统计.md $ST/03_结果汇总/ 2>/dev/null
cp $AI/9.3/params_census.json $ST/03_结果汇总/ 2>/dev/null

# ---- 06 指标(逐格 result.metrics,按格名平铺) ----
for d in $FIN/r5_eres_noise_* $FIN/r5_ecapa_noise_* $FIN/r5_b6_noise_* $FIN/raw_eres_noise_* $FIN/raw_ecapa_noise_* $FIN/raw_b6_noise_* $FIN/nd_r5_eres_*; do
  [ -f "$d/result.metrics" ] && cp "$d/result.metrics" "$ST/06_指标/$(basename $d)__result.metrics"
done
for d in $SV/exp/clean_bench/../..//data/data_for_test; do true; done
# 噪声域 raw 锚点
for d in $AI/8.28/noise_domains/eval_raw/*/; do
  [ -f "$d/result.metrics" ] && cp "$d/result.metrics" "$ST/06_指标/ndraw_$(basename $d)__result.metrics"
done
# 场景(交付前端 + 原始)
for d in $AI/8.28/scenes_r5/eres2net/*/; do
  [ -f "$d/result.metrics" ] && cp "$d/result.metrics" "$ST/06_指标/scene_$(basename $d)__result.metrics"
done
# clean 基线与 9.3 直测
cp $AI/8.28/clean_baseline_eres2net_3s/result.metrics $ST/06_指标/clean_未增强基线__result.metrics 2>/dev/null
cp $AI/9.3/fe_clean/fe_clean_eres_3s/result.metrics   $ST/06_指标/clean_过前端_9.3补测__result.metrics 2>/dev/null
# claim 三口径与补救
cp $AI/8.31/claim_retest/claim_summary.txt  $ST/06_指标/claim__summary.txt 2>/dev/null
cp $AI/8.31/claim_retest/claim_summary.json $ST/06_指标/claim__summary.json 2>/dev/null
cp $AI/8.31/claim_fix/energy_gate_v8d.json  $ST/06_指标/claim__energy_gate.json 2>/dev/null
cp $AI/8.31/claim_fix/fusion_summary.json   $ST/06_指标/claim__fusion.json 2>/dev/null
# WavLM 零样本 + 9.3 微调版
cp $AI/8.24/exp/wavlm_base_sv/wavlm_summary.json $ST/06_指标/wavlm__zeroshot_summary.json 2>/dev/null
cp $AI/9.3/wavlm_ft_official/wavlm_summary.json  $ST/06_指标/wavlm__finetuned_official_9.3.json 2>/dev/null
# 失败案例
cp $AI/8.30/failures/failures.json $ST/06_指标/失败案例_failures.json 2>/dev/null
# ROC/DET 曲线数据
cp -r $AI/8.30/roc_det $ST/06_指标/roc_det 2>/dev/null

# ---- 04 代码 ----
cp $AI/8.26/tse_model.py $AI/8.26/train_tse.py $AI/8.26/train_tse.toml $AI/8.26/enhance_tse.py $ST/04_代码/ 2>/dev/null
cp $AI/8.27/enhance_tse_v2.py $AI/8.27/build_proxy_val.py $AI/8.27/diag_enroll_noise.py $ST/04_代码/ 2>/dev/null
cp $AI/8.30/gen_roc_det.py $AI/8.30/mine_failures.py $ST/04_代码/ 2>/dev/null
cp $AI/8.24/wavlm_finetune.py $AI/8.24/wavlm_sv_scores.py $ST/04_代码/ 2>/dev/null
cp $AI/9.3/census_params.py $AI/9.3/launch_fe_clean.sh $ST/04_代码/ 2>/dev/null
for f in train_spk_preserve.py trainer_spk_preserve.py frontend_spk.py common_utils.py evaluate_robust.py; do
  find $AI/8.18 $AI/8.20 $AI/8.21 -maxdepth 1 -name "$f" -exec cp {} $ST/04_代码/ \; 2>/dev/null
done

# ---- 05 权重(交付前端) ----
cp $R5/checkpoint_best.tar $ST/05_权重/tse_frontend_r5_checkpoint_best.tar 2>/dev/null

echo "[stage] metrics=$(ls $ST/06_指标 | wc -l) code=$(ls $ST/04_代码 | wc -l) weight=$(du -sh $ST/05_权重 | cut -f1)"

# ---- 07 增强音频样本(python 挑配对) ----
$CM/zhangqiyuan/bin/python - <<'PYEOF'
# -*- coding: utf-8 -*-
import glob, os, shutil, csv
CM = "/data/home/ligangrui/china_mobile"
SV = CM + "/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa"
R5 = CM + "/AI/8.27/exp/tse_v3_ns05duraug_r5"
ST = CM + "/AI/9.3/交付材料_9.3/07_增强音频样本"

def spk_of(stem): return "_".join(stem.split("_")[:2])   # 3D_SPK_XXXXX

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
    if raw:  dsts.append((raw,  tag + "_1原始带噪.wav"))
    if enh_path: dsts.append((enh_path, tag + "_2增强后.wav"))
    for j, e in enumerate(enr): dsts.append((e, "%s_reg%d.wav" % (tag, j + 1)))
    for src, name in dsts:
        shutil.copy(src, os.path.join(ST, name))
        rows.append((name, os.path.relpath(src, CM)))

def pick(pattern_dir, n, cond, offset=0):
    wavs = sorted(glob.glob(os.path.join(pattern_dir, "*.wav")))
    step = max(1, len(wavs) // n)
    for i in range(n):
        p = wavs[(i * step + offset) % len(wavs)]
        add(cond, i + 1, os.path.splitext(os.path.basename(p))[0], p)

pick(R5 + "/enhanced_official/noise_0db", 4, "noise_0db")
pick(R5 + "/enhanced_official/noise_m5db", 3, "noise_m5db")
pick(CM + "/AI/9.3/fe_clean/enhanced_clean/clean", 2, "clean")

with open(os.path.join(ST, "样本对照表.tsv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["样本文件", "服务器源路径(相对 china_mobile/)"])
    for r in rows: w.writerow(r)
print("audio samples:", len(rows))
PYEOF

echo "[done] $(du -sh $ST | cut -f1)  $(find $ST -type f | wc -l) files"
