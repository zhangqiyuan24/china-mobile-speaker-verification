================================================================
  ERes2Net-large Teacher-Student 鲁棒微调 — 代码说明
  日期: 2025-08-11
================================================================

一、设计原理
================================================================

目标：在微调适应 noise/overlap 的同时，用 Teacher 锚定原始 clean embedding 空间，
      防止灾难性遗忘。

架构：

  clean wav --> FBank --> feat_clean --> [Teacher: 全冻结] --> e_teacher
                                                                  |
  same wav --> RobustSpkAug --> FBank --> feat_aug --> [Student] --> e_student --> classifier --> L_arcface
                                                              |
                                              L_cons = 1 - cos(e_student, e_teacher.detach())
                                                              |
                                              L = L_arcface + 0.25 * L_cons

Teacher: 原始 ERes2Net-large pretrained, 全冻结, 始终 eval()
Student: 同样从 pretrained 初始化, 部分冻结, 输入 augmented speech


二、文件清单 & 替换说明
================================================================

[1] eres2net_finetune_ts.yaml
    替换: egs/3dspeaker/sv-eres2net/conf/eres2net_finetune_ts.yaml (新文件)
    作用: Teacher-Student 训练配置
    关键参数:
      - teacher_student: True
      - lambda_cons: 0.25
      - frozen_modules: conv1 / bn1 / layer1 / layer2 / layer3
      - layerwise_lr: layer4/fusion=2e-5, seg_1/classifier=5e-5
      - num_epoch: 15
      - batch_size: 128 (4×P100 即 32/GPU)
      - margin_scheduler fix_epoch: 12

[2] dataset.py
    替换: speakerlab/dataset/dataset.py
    作用: 修改 WavSVDataset.__getitem__(), T-S 模式返回 (feat_clean, feat_aug, spkid)

[3] train_ts.py
    替换: speakerlab/bin/train_ts.py (新文件)
    作用: Teacher-Student 训练主逻辑
    更新:
      - 每个 epoch 第一个 batch 打印各组实际 LR
        (格式: [LR-EPOCH-N] layer4 lr=2.00e-05  seg_1 lr=5.00e-05  ...)
      - scheduler fix 后各组 LR 比例始终不变

[4] scheduler.py  ★ NEW — 修复分层 LR 被覆盖的 bug
    替换: speakerlab/process/scheduler.py
    作用: WarmupCosineScheduler 修复
    Bug:
      原始 set_lr() 将 ALL param_groups 设为同一个 lr 值,
      分层 LR 配置完全无效。
    Fix:
      __init__ 时记录每组初始 lr, set_lr() 时按比例缩放:
        pg['lr'] = new_lr * (base_lr / max_lr)
    效果:
      - layer4 base=2e-5 → ratio=0.1 → peak 时 2e-5, min 时 1e-7
      - seg_1 base=5e-5 → ratio=0.25 → peak 时 5e-5, min 时 2.5e-7
      比例始终恒定, 等价于各组独立余弦衰减

[5] run_eres2net_large_ts.sh
    替换: egs/3dspeaker/sv-eres2net/run_eres2net_large_ts.sh (新文件)
    更新:
      - 噪声池 90/10 划分 (train/valid 隔离, seed=2026)
      - noise_train_freesound.txt 不存在 → exit 1
      - 生成 musan_wav_valid.scp (validation 专用噪声)
      - 生成 valid_spk2utt.json (validation overlap 干扰池)
      - Train/Valid/Test speaker 三重隔离检查
      - Train/Valid/Test noise 三重隔离检查
      - --valid_noise / --valid_spk2utt 传参
      - EPOCHS/LAMBDA_CONS 移到 echo 之前


三、部署步骤
================================================================

在训练服务器上:

1. 备份原始文件 (可选):
   cp speakerlab/bin/train.py speakerlab/bin/train.py.bak
   cp speakerlab/dataset/dataset.py speakerlab/dataset/dataset.py.bak
   cp conf/eres2net_finetune.yaml conf/eres2net_finetune.yaml.bak
   cp run_eres2net_large.sh run_eres2net_large.sh.bak

2. 替换文件:
   cp AI/8.11/dataset.py speakerlab/dataset/dataset.py
   cp AI/8.11/train_ts.py speakerlab/bin/train_ts.py
   cp AI/8.11/scheduler.py speakerlab/process/scheduler.py
   cp AI/8.11/eres2net_finetune_ts.yaml conf/eres2net_finetune_ts.yaml
   cp AI/8.11/run_eres2net_large_ts.sh run_eres2net_large_ts.sh

3. 清理旧数据 (重要!!!):
   rm -rf data/train.csv data/valid.csv data/train_spk2utt.json
   rm -rf exp/eres2net_finetune_ts

4. Debug 验证:
   bash run_eres2net_large_ts.sh --debug
   # 检查 train.log:
   #   - [TEACHER] 日志确认 Teacher 已加载
   #   - [FREEZE] 日志确认冻结参数比例
   #   - 前几个 batch 的 L_arc 和 L_cons 有值且不是 nan

5. 正式训练:
   bash run_eres2net_large_ts.sh

6. 评测:
   修改 evaluate_all_finetune.sh:
     FINETUNED_CKPT=".../eres2net_finetune_ts/models/best/embedding_model.ckpt"
   bash evaluate_all_finetune.sh


四、关键超参数
================================================================

| 参数          | 值      | 说明                              |
|---------------|---------|-----------------------------------|
| lambda_cons   | 0.25    | consistency loss 权重             |
| num_epoch     | 15      | 训练轮数                          |
| batch_size    | 128     | 4×P100 即 32/GPU                  |
| clean_prob    | 0.50    | 50% 样本走 clean 路径             |
| pos_noise     | 0.15    | 15% 正 SNR (0~15 dB)             |
| neg_noise     | 0.15    | 15% 负 SNR (-5~0 dB)             |
| overlap2      | 0.15    | 15% 双人叠音 (TIR=0 dB)          |
| overlap3      | 0.05    | 5% 三人叠音 (TIR=0 dB)           |
| layer4 LR     | 2e-5    | ratio=0.1 × max_lr=2e-4          |
| fusion LR     | 2e-5    | fuse_mode12/123/1234              |
| seg_1 LR      | 5e-5    | embedding 投影层, ratio=0.25      |
| classifier LR | 5e-5    | ArcFace 分类器, ratio=0.25        |
| max_lr        | 2e-4    | scheduler reference (yaml lr)     |
| min_lr        | 1e-6    | scheduler floor                   |

各组 LR = cosine_curve(t) × (base_lr / max_lr), 比例始终不变。


五、冻结 & 可训练参数
================================================================

冻结 (模式匹配, 子串匹配即冻结):
  conv1, bn1, layer1, layer2, layer3
  注意: layer1_downsample, layer2_downsample, layer3_downsample
        因为包含 "layer1"/"layer2"/"layer3" 子串，也会被冻结

可训练:
  layer4 (3 blocks), fuse_mode12, fuse_mode123, fuse_mode1234,
  seg_1 (20480->512 Linear), classifier

预估可训练参数: ~13M (~60% of 22.46M)
(seg_1 占大头: ~10.5M)


六、不修改的文件 (直接复用现有代码)
================================================================

以下文件不需要任何改动:
  - speakerlab/process/processor.py     (WavReader, RobustSpkAug, FBank 不变)
  - speakerlab/process/augmentation.py  (NoiseReverbCorrupter, OverlapCorrupter 不变)
  - speakerlab/models/eres2net/         (ERes2Net 模型架构不变)
  - speakerlab/bin/infer_sv_batch.py    (推理脚本不变, --pretrained 照用)
  - speakerlab/bin/compute_score_metrics.py (评分脚本不变)
  - local/prepare_finetune_data.py      (数据准备不变)
  - evaluate_all_finetune.sh            (评测脚本不变, 只改 FINETUNED_CKPT 路径)


七、Validation — 多条件确定性验证
================================================================

每个 epoch 依次评测 3 个固定条件 (seed=42):

  EER_clean       — 原始 clean 语音, 不用增强
  EER_noise_m5db  — 加 valid noise pool 噪声, SNR=-5dB 固定
  EER_overlap2    — 加 1 个 valid speaker 干扰, TIR=0dB 固定

关键设计:
  - 400 speakers × 4 utts/speaker (可通过 num_spk/utts_per_spk 调整)
  - 固定 seed → 每个 epoch 使用完全相同的 speaker/utterance/noise/interferer
  - random state 在 validation 前后保存/恢复 → 不影响训练随机性
  - Valid noise 来自 noise pool 的 10% 独立划分
  - Valid overlap 干扰者来自 valid speaker (与 target 不同)

每 epoch 输出:
  [VALID] Epoch N: EER_clean=X.XX%  EER_noise_m5db=Y.YY%  EER_overlap2=Z.ZZ%

两个 best checkpoint:
  best_clean:  按 EER_clean 最低选 (传统方式)
  best_robust: 按 robust_score = EER_clean + EER_noise_m5db + EER_overlap2 最低选
  分别保存到 models/best_clean/ 和 models/best_robust/

八、已知风险 & 注意事项
================================================================

1. GPU 内存: Teacher 模型 ~90MB fp32。batch_size=128 (32/GPU) 应在 P100 12GB 内。
   如果仍然 OOM，改为 batch_size=96 (24/GPU)。

2. lambda_cons 调节: 0.25 是初始值。
   - clean EER 退化: 增大 (0.5)
   - noise/overlap 不提升: 减小 (0.1)

3. scheduler.py 替换: 修改了 speakerlab 框架文件。
   如果其他实验也在用 WarmupCosineScheduler + layerwise LR，
   这个修复对它们同样有益 (不会改变单 LR 组的行为)。

4. Validation 耗时: 3 条件 × 400 speakers × 4 utts ≈ 4800 embeddings/epoch。
   预计每 epoch 增加 1-2 分钟 (取决于 GPU)。

5. speed_perturb=false: 保持关闭。

6. 旧数据清理: 删除 data/ 和 exp/eres2net_finetune_ts 后重新运行。
