# E2-proto：prototype 初始化 classifier 的多条件鲁棒微调

基于 `AI/8.20/`（E2）的唯一方法改动：**CosineClassifier 不再随机初始化**，
改用 pretrained ERes2Net 提取的 clean speaker prototype 初始化。其余一切
（1:1:1 动态混噪 / 30k 语句 / 冻结范围 / AAM margin0.2 scale32 / lr=2e-5 /
batch12×accum8 / 3 epoch sanity）与 8.20 完全一致，做干净对照。

## 动机（8.20 sanity 的诊断）

随机 9900 类分类头在 lr=2e-5 下需要 ~2.5 万步才收敛，而 3 epoch sanity 只有
2812 步——训练早期 embedding 一直被半随机的头拖拽：epoch1 clean 1.25→4.00、
0dB 7.00→11.00、-5dB 11.25→16.50，train_acc 仅 0.017。prototype 初始化让
分类头从"接近收敛"的位置出发，AAM 梯度从一开始就小而准。

## 文件清单

| 文件 | 作用 |
|---|---|
| `config_proto.toml` | 全部路径/超参（含 8.20 random-init epoch1 对照数字） |
| `prepare_prototypes.py` | pretrained 模型提 clean prototype，落盘 `exp/prototypes/`（一次性缓存） |
| `trainer_eres2net_robust_proto.py` | prototype 初始化 + 初始化诊断 + ProtoTrainer（含 Option A/B 预留接口） |
| `train_eres2net_robust_proto.py` | 主训练（数据管线/冻结/守卫直接复用 8.20 代码） |
| `evaluate_combo.py` | 正式 open-set 组合评测：raw vs FullSubNet+前端 × robust 模型 |
| `enhance_testset.sh` | 调 8.17 工具离线增强噪声测试集（只读 8.17/8.18） |
| `run_sanity.sh` | prototype 提取 + 3 epoch sanity（+ 可选 ABLATION=1 随机对照组） |
| `run_train.sh` | 8 epoch 正式训练 + 正式评测（+ 可选前端组合） |

复用（import 自 `../8.20`，不复制不覆盖）：`common_utils.py`（混噪/EER/ark）、
`prepare_multicond_data.py` 的全部产物（train_utts.csv / val_bench / trials）、
`train_eres2net_robust.py` 的数据管线类、`validate_robust.py`、
`evaluate_robust.py` 的官方打分构件。**不覆盖** `AI/8.17/8.18/8.19/8.20`、
3D-Speaker 工程、pretrained ckpt；输出全部在本目录 `exp/` 下。

## prototype 生成协议（prepare_prototypes.py）

1. **只用原始 pretrained ERes2Net**（strict 加载；绝不用任何 fine-tune 权重）。
2. 语料 = 训练说话人（train_spks.txt，9900 人）的 clean 语句；与验证/正式 test
   说话人零交集（8.20 prepare 已断言，此处再次断言 train∩val=∅）。
3. 来源默认**全量 all_wav.scp**（64 万条）：30k 训练清单每人只有 ~3 条，全量才能
   满足"每人 5~10 条"（默认 `utts_per_spk=8`，不足则全用，等间隔确定性选取）。
   `source="train_csv"` 可切回仅 30k 清单（每人 ~3 条）。
4. 每条：居中裁剪 3s（确定性，可复现）→ fbank → pretrained embedding。
5. prototype = `L2(mean(L2(e₁..eₖ)))`：逐条归一化 → 均值 → 再归一化。
6. 落盘 `exp/prototypes/prototypes.npz`（prototypes + spk_ids + used_utt_ids + counts），
   训练直接读缓存；诊断打印 matrix shape / 每人条数 min/mean/max / all finite / 行范数 ≈1。

初始化映射：`classifier.weight` 第 j 行 = spk2id 为 j 的说话人的 prototype
（CosineClassifier 为 `Parameter[S,D]` 无 bias、forward 内部 `F.normalize` 双侧
归一化的真 cosine logits，单位 prototype 直接写入即可）。写入后**逐位回读校验**。
注意：prototype 按 train_spks.txt 全集 9900 人生成，而 30k 训练清单均匀抽取只含
9892 人（8 人无训练语句）——**超集是正常情况**，多余的 prototype 行不用；
训练类别数以 train_utts.csv 为准（与 8.20 相同的 9892 类）。

## 训练前初始化诊断（不训练任何参数）

`[prototype init]`：前 256 个说话人各 1 条 clean（优先取未参与 prototype 的语句），
打印 clean top1 acc / AAM loss / mean target cosine / mean max non-target cosine，
并同时打印**随机初始化实测对照**（临时 xavier 分类器，只读 embedding）与理论值
（top1 ≈ 1/9900 ≈ 0.01%，AAM loss ≈ ln(9900)+32×0.2 ≈ 15.6）。
若 prototype top1 仍接近随机，按顺序排查：spk label mapping → weight 维度 →
embedding 归一化 → prototype 行顺序，**不进入训练**。

## sanity 判读：epoch1 对照（本实验的核心问题）

| | clean | 0dB | -5dB | train_loss | train_acc |
|---|---|---|---|---|---|
| random init（8.20 实测） | 4.00 | 11.00 | 16.50 | 15.05 | 0.017 |
| prototype init（本轮） | ? | ? | ? | ? | ? |

期望：clean 不再跳到 4%+；train_acc 明显高于 0.017；0dB/-5dB 初始退化明显减轻；
noise EER 更快低于 pretrained 基线（7.00/11.25）。若 prototype 后 epoch1 仍
clean≈4%+ / 0dB≈11%+ / -5dB≈16%+，说明随机 classifier 不是主矛盾，转向
减少 body 解冻范围 / classifier-only warmup / pretrained embedding anchor，
**不要单纯加 epoch**。训练结束自动打印该对照表（数字来自 config 的
`[compare.random_init_ep1]`）；`ABLATION=1 bash run_sanity.sh` 可在本代码路径下
再跑一组 `--no-proto` 随机对照（输出 `exp/proto_robust_random/`，互不覆盖）。

## 验证集的正确解读（seen-by-pretrained）

验证 bench 的 100 人虽未参与本次微调，但**被 pretrained 原始训练见过**，
故 clean 基线 1.25% 远低于正式 open-set 的 5.8248%，clean 退化会被高估。因此：
1. 保留该 bench 只为与 8.20 逐 epoch 可比、选 checkpoint（noise_sum 最小 + 门限）；
2. 日志已标注 `seen-by-pretrained validation`；
3. `clean ≤ 1.25+1pp` 只是选型门限，**不是** open-set 真实 clean gate；
4. 正式结论必须走 `evaluate_combo.py` 的 open-set 协议（官方 trials + 官方打分）。

## 前端组合评测（训练完成之后；不联合训练）

`enhance_testset.sh`（只读 8.17 代码 + 8.18/8.19 前端 ckpt）先把
`noise_0db / noise_m5db` 的 3s 测试集离线增强；`evaluate_combo.py` 再跑：

```
A. raw       : 原始带噪 ─────────────► robust ERes2Net(8.21)
B. frontend  : 带噪 ─► FullSubNet+ ──► robust ERes2Net
```

协议：enrollment = 官方 clean/3s（**不过前端**），两侧同一个 robust checkpoint；
trials = 原始 trials_cross_device；打分 = 原版 compute_score_metrics.py。
输出四行对照（官方基线 / 前端单独 7.7902/9.9186 / raw robust / 前端+robust）、
相对基线降幅、**前端相对 raw robust 的额外增益（ablation，不默认前端有益）**、
30% 判据结论与 open-set clean EER。结果落 `exp/proto_robust/eval/combo/`。

## 预留接口（本轮全部关闭）

| 开关 | 含义 | 启用 |
|---|---|---|
| `--head-warmup-steps N` | Option A：前 N 个 optimizer step 只更新 classifier（body lr=0，lr 仍 ≤2e-5） | 默认 0=关 |
| `--head-lr 1e-4` | Option B：classifier 独立 lr（body 仍 2e-5）。**突破统一 lr 约束，需老师同意** | 默认 0=关 |

两者关闭时 optimizer 与 8.20 单参数组 AdamW 数值等价。

## 运行顺序

```bash
cd AI/8.21
bash run_sanity.sh                    # ① prototype 提取 + 3 epoch sanity（GPU=3 默认）
GPU=1 SHARED=0 bash run_train.sh      # ② 正式 8 epoch + open-set 评测
bash enhance_testset.sh               # ③（可选）离线增强噪声测试集
ENH_0DB=exp/enhanced/noise_0db ENH_M5DB=exp/enhanced/noise_m5db \
  SKIP_TRAIN=1 bash run_train.sh      # ④（可选）前端组合 ablation
```

共享卡说明同 8.20：batch 调小必须 `--accum` 补偿到有效 96，否则每 epoch 更新
次数成倍增加、漂移放大（run 脚本已内置 batch12×accum8+AMP 配方）。
