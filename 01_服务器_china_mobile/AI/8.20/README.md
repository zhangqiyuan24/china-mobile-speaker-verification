# E2：ERes2Net-large 多条件鲁棒微调（clean / 0dB / -5dB = 1:1:1）

子课题一的主杠杆实验：不再只做前端增强，而是把**声纹模型本身**在多条件下微调，
让 embedding 对带噪语音不敏感。目标：0dB EER ≤ 5.874%、-5dB EER ≤ 7.739%
（相对官方基线 8.3911 / 11.0557 降幅 ≥30%），clean 退化 < 1pp。

## 目录

| 文件 | 作用 |
|---|---|
| `config_robust.toml` | 全部路径 / 超参 / 基线数字 |
| `common_utils.py` | 路径解析、混噪公式（与 gen_noise_benchmark.py 同源）、EER/minDCF、ark 读写 |
| `prepare_multicond_data.py` | 说话人/噪声三池隔离 + 30k 训练语句抽取 + 验证 bench 落盘 |
| `trainer_eres2net_robust.py` | 冻结（顶层模块名精确匹配 + BN stats 冻结）、AAM、checkpoint 原子保存与回读校验 |
| `train_eres2net_robust.py` | 主训练：1:1:1 动态混噪 batch、逐 epoch 验证、双门限选 best、单卡/DDP |
| `validate_robust.py` | 任意 checkpoint（或 pretrained 基线）在验证集上复测 |
| `evaluate_robust.py` | 正式评测：同一 checkpoint 重提注册+测试两侧，原版 compute_score_metrics 打分 |
| `asnorm.py` | AS-Norm：cohort 构造（训练说话人）/ 验证集选 K / 正式分数归一化 |
| `run_sanity.sh` | prepare + 3 epoch sanity |
| `run_train.sh` | 正式训练 + 评测（+ 可选 AS-Norm） |

## 运行顺序

```bash
cd AI/8.20

# ① sanity（首次必跑）：验证整条管线 + 看 3 epoch 趋势
bash run_sanity.sh                  # 默认 GPU 1；GPU=3 bash run_sanity.sh

# ② 正式训练（8 epoch，约 3~5 小时单卡 / 减半双卡）
bash run_train.sh                   # 单卡
GPUS=1,3 bash run_train.sh          # 双卡 DDP（同 8.19 的卡）
# 可覆盖：EPOCHS=10 LR=1e-5 GPUS=1,3 bash run_train.sh

# ③ 复核最优 checkpoint（可选）
python validate_robust.py --config config_robust.toml \
    --checkpoint exp/robust/checkpoints/checkpoint_best.tar

# ④ AS-Norm（在 ② 之后，可选）
RUN_ASNORM=1 bash run_train.sh      # 或按 asnorm.py 头部注释分步跑
```

## 数据与隔离（prepare 一次性完成，全部断言）

- **说话人**：全量 10,000 train 说话人 → 排序末尾 100 人做验证，其余 9,900 人做训练；
  与正式 test 说话人（trials_cross_device）零交集（重叠则剔除并打印）。三方互斥。
- **噪声**：官方 train pool 675 条 → 末尾 60 条做验证噪声，615 条做训练噪声；
  与官方 test pool 168 条零交集。三方互斥。
- **训练语句**：训练说话人全量语句等间隔均匀抽 30k（`utts_per_spk` 可改成按人采样）。
- **验证 bench**：每人 2 句 clean 注册 + 2 句测试（clean/0dB/-5dB 三条件落盘，
  与正式集同一混噪公式、3s 随机裁剪、0dB/-5dB 用同一段噪声），trials 落盘固定，
  epoch 间严格可比。

## 训练协议

- 输入：同一条语句每 epoch 动态生成 clean/0dB/-5dB 三个版本（随机 3s 裁剪 +
  随机训练噪声 + 随机偏移），每个 batch 严格 1:1:1（batch_size 必须是 3 的倍数）。
- 冻结：`conv1/bn1/layer1/layer1_downsample/layer2/fuse_mode12`（含 BN running stats，
  防带噪统计量漂移污染冻结层）；训练 `layer3/layer4/layer2_downsample/layer3_downsample/
  fuse_mode123/fuse_mode1234/pool/seg_1/seg_2` + 新建 CosineClassifier。
  第一个 step 后有守门断言：冻结参数逐位不变、可训练参数已更新。
- 损失：AAM-softmax（ArcMarginLoss, margin=0.2, scale=32, 固定无 scheduler）。
- 优化：AdamW lr=2e-5（固定），wd=5e-5，8 epoch。
- 选型：`best = argmin(EER_0db + EER_m5db)`，且 `EER_clean ≤ epoch0_clean + 1pp`；
  训练准确率只打印不参与选型。每个 epoch 都存 checkpoint + val_history.json。

## 正式评测协议（evaluate_robust.py）

- **注册侧与测试侧用同一 checkpoint 重新提 embedding**（修复 8.18 依赖
  infer_sv_batch 导致换 checkpoint 不生效的问题）。
- 测试侧 = `data_for_test/{clean,noise_0db,noise_m5db}/3s` 官方 wav_list（A+D），
  trials = 原始 trials_cross_device，打分 = 原版 `compute_score_metrics.py`。
- 汇总自动对照 results_summary.csv 基线输出相对降幅，并给出 30% 判据结论。
- `--also-pretrained`：用 pretrained 跑同一管线，应复现官方基线（管线自检）。

## AS-Norm（asnorm.py）

1. `--make-cohort`：训练说话人确定性抽 500×2=1000 句 cohort（与 test 说话人不重叠）。
2. `evaluate_robust.py --extract-only --crop-seconds 3`：用**同一个 checkpoint**
   提 cohort embedding（cohort 是全长语料，必须居中裁剪到 3s 才能按 batch 堆叠）。
3. `--select-k`：验证集上对 K∈{100,200,300} 算 AS-Norm EER，选噪声两档之和最小的 K。
4. `--apply`：对正式 (enrol, test) ark 做对称 AS-Norm，报告 raw/AS 的 EER 与 minDCF，
   写 `.asnorm.score`；apply 的 raw EER 应与官方打分一致（自检）。

## 首次运行前检查清单（11 项）

1. `python -c "import torch, toml, kaldiio, soundfile, torchaudio"` 依赖齐备
2. `config_robust.toml` 路径存在（sv_ecapa / clean_scp / noise 两个列表 / trials / eres2net_ckpt）
3. pretrained ckpt 只读（本目录不写 3D-Speaker 工程与 pretrained 任何文件）
4. prepare 输出：说话人三方零交集、噪声三方零交集断言通过
5. prepare 输出：SNR 抽检 0dB/-5dB 实测值在目标 ±1.5dB 内
6. `[freeze_report]` 冻结/可训练参数量与预期一致（可训练 ≈ 3/4 的 embedding 参数）
7. `[guard]` 冻结参数未变、可训练参数已更新
8. epoch0 验证：pretrained 在 val bench 的 clean EER 应在 5~8% 量级（过高说明 bench 有问题）
9. sanity 3 epoch：0dB/-5dB 验证 EER 逐 epoch 下降、clean 在门限内
10. checkpoint 可被 validate_robust.py / evaluate_robust.py 重新加载（strict 通过）
11. 评测时 evaluate_robust.py 与 asnorm.py 用的 checkpoint 路径一致（meta.json 会打印）

## 与旧代码的关系

- 复用：speakerlab 的 ERes2Net/FBank/ArcMarginLoss/CosineClassifier/
  score_metrics/compute_score_metrics.py；8.19 的路径解析与均匀抽取；8.18 的
  评测目录结构约定。
- 不覆盖：`AI/8.17`、`AI/8.18`、`AI/8.19`、3D-Speaker 工程、pretrained ckpt。
- 输出全部在本目录 `exp/robust/` 下。
