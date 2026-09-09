# E2-matched：五条件混合域鲁棒微调（8.22）

## 动机（8.21 组合消融矩阵的结论）

trials_cross_device / 3s / 官方打分，EER%（0dB / -5dB）：

| 测试侧 \ 后端 | pretrained | robust (8.21) |
|---|---|---|
| 原始带噪 | 8.3911 / 11.0557（基线） | 7.6911 / 9.8781 |
| 原版前端 (8.17) | 10.1132 / 12.6667（负增益） | 9.8874 / 12.5284 |
| 微调前端 (8.18 e10) | 7.7902 / 9.9186 | **7.5000 / 9.5085** |

级联（右下角）有效但两路增益重叠：加性上限 ~7.09/8.74，实测 7.50/9.51。
重叠的成因：**robust 后端只被 raw 域校准过**（没见过"本前端增强出来的域"），
且 8.18 前端当初是对着 pretrained 后端调的。8.22 让后端在
{clean, raw0, raw-5, enh0, enh-5} 五域一起微调，收回重叠部分。

## 设计（与 8.21 唯一的方法差异）

- 条件混合：均匀 1/3（三条件）→ **均匀 1/5（五条件）**；batch 必须是 5 的倍数
  （共享卡配方 batch10×accum10，有效 100，与 8.21 的 96 同规模）
- enh 两路 = `prepare_enh_data.py` 预生成的**固定语料**：30k 训练清单 × SNR{0,-5}，
  固定种子混噪后过**部署前端**（8.18 `checkpoint_epoch10.tar`，冻结，只当数据工厂）
  ——动态混噪没法在线跟增强，只能离线；raw 三路保持 8.21 原样的动态混噪
- 其余全部不动：pretrained 起点 / prototype 初始化（复用 8.21 的 npz）/ lr=2e-5 /
  冻结范围 / AAM margin0.2 scale32 / 8 epoch
- 选点准则改为 **enh 主路**：enh_noise_sum = bench EER(enh0)+EER(enh-5) 最小
  + clean 门限（raw bench epoch0 clean +1pp）；raw bench 三条件照常报告（与 8.21 可比）
- 启动守卫：enh 语料 manifest 必须 complete 且前端 ckpt 与 config 一致——
  换前端必须重跑 `prepare_enh_data.py`，训练/推理前端不一致直接拒绝启动

注意两个必然差异（不是 bug）：每 epoch 样本 90k→150k（更新数 937→1500，lr 不变）；
enh 路每条语句每 epoch 是同一份固定 3s 实现（无裁剪/换噪多样性，靠 30k×2 规模兜）。

## 文件清单

| 文件 | 作用 |
|---|---|
| `config_matched_90k.toml` | **主配置**（90k 数据版，run_train.sh 默认） |
| `config_data90k.toml` | 喂给 8.20 prepare 的 90k 数据生成配置（Step 0 自动用） |
| `config_matched.toml` | 30k 版配置（对照/归因） |
| `prepare_enh_data.py` | ①固定种子混噪落盘 ②分块调 8.17 工具过前端增强（块级断点续跑）③验证 bench 增强 ④manifest |
| `train_matched.py` | 五条件数据集/采样器 + 双 bench 验证 + enh 主路选点 |
| `run_train.sh` | 一条龙：数据(可自动补) → 语料 → 训练 → 8.21 evaluate_combo 正式评测 |
| `test_score_fusion.py` | 零训练快测：raw/enh 两路分数加权融合（杠杆 B） |
| `run_frontend_robust.sh` | （已完成）补矩阵最后一格的测量脚本，留档 |

复用（import，不复制不覆盖）：8.20 的 `common_utils`/`read_train_csv`/`worker_init_fn`/
数据产物，8.21 的 `ProtoTrainer`/prototype npz/`evaluate_combo.py`；只读 8.17 代码与
8.18 前端 ckpt。输出全部在本目录 `exp/` 下。

## 运行顺序

```bash
cd AI/8.22
df -h .                                   # 90k 语料约 34GB
bash run_train.sh                         # 全流程，默认 90k（含 Step 0 自动生成 90k 数据）
# 或分步：
SANITY=1 bash run_train.sh                # 冒烟：前 2000 条语料 + 3 epoch（30k 版已冒烟通过）
SKIP_PREP=1 bash run_train.sh             # 语料就绪，训练+评测
SKIP_PREP=1 SKIP_TRAIN=1 bash run_train.sh   # 只评测
CFG=config_matched.toml bash run_train.sh # 30k 对照版（归因用：matched 单变量 vs +数据量）
python test_score_fusion.py --config config_matched.toml   # 融合快测（现在就能跑，用 8.21 的 ark）
```

时间预算（共享 GPU 3，实测 0.24s/micro-step 外推）：

| 阶段 | 90k（默认） | 30k |
|---|---|---|
| Step 0 数据生成（CPU） | ~20 min | —（用 8.20 现成） |
| 混噪 | ~3h | ~1h |
| 前端增强 | ~10-14h（180k 条） | ~3-4h |
| 训练 8 epoch | ~24h（每 epoch 450k 样本） | ~8.5h |
| 正式评测 | ~2h | ~2h |

90k 说明（单变量：只放数据量，lr/冻结/AAM/prototype 全不动）：
- 90k 清单由 `config_data90k.toml` + 8.20 的 prepare 原脚本生成（等间隔确定性选样，
  说话人/噪声三方隔离同一套代码），输出在 `exp/data_90k/`，不碰 8.20 产物
- 验证 bench 与 8.20 那份内容一致 → epoch0 基线、clean 门限、与 8.21 的 bench 数字仍可比
- prototypes.npz 复用：90k 说话人集合 ⊆ npz 的 9900 行，日志应显示 9900/9900 类命中
- 若池子不足 90k，prepare 会取全部并如实打印（以 Step 0 日志为准）
- 全量训练**盯 epoch 1-2**：val 若仍比 epoch0 差且不回头（sanity 里出现过的形态），
  说明不是数据量问题，停下来排查（lr / enh 分支）

## 预期与判读

加性上限 ≈ 7.09/8.74（-15.5%/-21.0%）。30k 版合理预期 **0dB ~7.1-7.4（-13~-17%）、
-5dB ~8.8-9.3（-17~-21%）**；90k 版在其上再拿一截（预期再 -0.3~-0.8pp，
即 0dB 有机会摸到 ~6.5-7.0）——单靠本轮到不了 30%（验收线 5.874/7.739）。
看 raw bench 三条件有没有被 enh 拖垮（对照 8.21 的 ~16.0 noise_sum）；clean 门限兜底。
30% 还差的量走路线图：A 数据 30k→90k（本轮已并入）→ B 分数融合（本目录已可先测）→
C robust 当 teacher 重调前端 → D 更难 SNR → E 端到端联合。

## 验收口径（同 8.21）

0dB ≤ 5.874 且 -5dB ≤ 7.739（相对官方基线 -30%）；clean 退化 <1pp 以
`evaluate_combo` 汇总里的 open-set clean EER 为准（非验证集）。
分数融合（B）是否算系统内改动，用前与老师确认。
