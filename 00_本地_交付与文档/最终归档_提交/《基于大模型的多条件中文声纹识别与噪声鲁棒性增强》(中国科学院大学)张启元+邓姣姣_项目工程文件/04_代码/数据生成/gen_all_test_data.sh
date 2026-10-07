#!/bin/bash
# ============================================================
# 子课题一：一键生成全部多条件测试数据
#
# 流程：
#   1. split_noise_rir.py  → 噪声/RIR 训测拆分
#   2. gen_test_conditions.py --scenario all → 生成全部12种测试条件
#
# 输出目录：data/data_for_test/
#
# 用法（在 sv-ecapa 目录下运行）：
#   bash gen_all_test_data.sh
# ============================================================
set -e

echo "============================================"
echo "  子课题一：多条件测试数据生成"
echo "  共 12 种测试条件"
echo "============================================"

# Step 1: 噪声/RIR 训测拆分
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [Step 1/2] 噪声 / RIR 训测拆分"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python split_noise_rir.py

# Step 2: 生成全部测试条件（12 种）
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [Step 2/2] 生成 12 种测试条件"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python gen_test_conditions.py --scenario all

echo ""
echo "============================================"
echo "  全部完成！"
echo "  输出目录：data/data_for_test/"
echo "============================================"
echo ""
echo "生成的内容："
echo "  data/data_for_test/test_noise_m5db_music/"
echo "  data/data_for_test/test_noise_m5db_noise/"
echo "  data/data_for_test/test_noise_m5db_speech/"
echo "  data/data_for_test/test_noise_0db_music/"
echo "  data/data_for_test/test_noise_0db_noise/"
echo "  data/data_for_test/test_noise_0db_speech/"
echo "  data/data_for_test/test_reverb/"
echo "  data/data_for_test/test_overlap_2/"
echo "  data/data_for_test/test_overlap_3/"
echo "  data/data_for_test/test_short_1s/"
echo "  data/data_for_test/test_short_3s/"
echo "  data/data_for_test/test_short_5s/"
echo ""
echo "每个目录下都有 wav_list.txt，可直接用于 infer_sv_new.py 的 --wavs 参数"
