# WeSep TSE 前端 + ERes2Net-large 声纹评测

验证 **TSE（目标说话人提取）能否改善 overlap2/overlap3 条件下的 speaker verification**。

## 两段式离线流程（服务器无网络）

```
本地 (有网络)                    服务器 (无网络)
bash download_local.sh    ──传整个目录──▶   bash setup_offline.sh
下载模型+代码+wheel                         解压模型+离线装依赖
                                            然后跑评测
```

## 目录结构（AI/8.13 自包含）

```
AI/8.13/
├── third_party/               ← 下载产物 (download_local.sh 生成)
│   ├── wesep/                 ← WeSep 代码
│   ├── wespeaker/             ← WeSpeaker 代码 (ECAPA speaker model)
│   └── models/                ← WeSep 模型 (avg_model.pt + config.yaml)
├── wheels/                    ← pip 离线 wheel
├── download_local.sh          ★ 本地下载脚本
├── setup_offline.sh           ★ 服务器离线准备脚本
├── config.yaml                ← 相对路径配置
├── common.py                  ← 共享工具
├── tse_consistency_test.py    ← 接口一致性 + sanity
├── tse_eres2net_eval.py       ← 批量评测
├── run_tse_eval.sh            ← 一键全量评测
└── tse_single_test.py         ← 单条 sanity
```

## 离线依赖清单

| 类型 | 内容 | 大小 |
|------|------|------|
| 模型 | bsrnn_ecapa_vox1.tar.gz | 249MB |
| 代码 | wesep (本地拷贝) + wespeaker (下载) | ~几 MB |
| pip | tqdm PyYAML soundfile silero-vad packaging | ~几 MB |

## 快速开始

### 本地（有网络，Windows）

```bash
cd AI/8.13
bash download_local.sh
# 完成后 third_party/ 和 wheels/ 会填充好
```

### 服务器（无网络）

```bash
# 把整个 AI/8.13/ 目录传到服务器
cd AI/8.13
bash setup_offline.sh

# ① 一致性测试 + sanity
python tse_consistency_test.py \
    --enroll_wav <A_enroll.wav> --target_wav <A_clean.wav> \
    --interferer_wav <B.wav> --interferer2_wav <C.wav> --out_dir ./out

# ② 平衡抽样 EER (200 trial)
python tse_eres2net_eval.py --condition overlap2_en --balanced --out_dir ./eval_out

# ③ 全量评测 (4 条件, 结果存 sv-ecapa/data/result/eres2net_wesep/)
bash run_tse_eval.sh
```

## 模型

| 组件 | 模型 | 来源 |
|------|------|------|
| TSE 前端 | `bsrnn_ecapa_vox1` (BSRNN + ECAPA) | WeSep 官方, ModelScope |
| 声纹 | ERes2Net-large (22.46M, dim=512) | 3D-Speaker |

两个模型**全部冻结**，只做 inference。

## 官方推理调用链（已核对）

```
mixture [B,T] ──→ stft → subband → BN → separator ──┐
                                                      ├→ mask → istft → output [B,T]
enrollment → compute_fbank [B,T',80] → ECAPA(spk_model) [B,256] ──┘
forward 返回 tuple (s, predict_label) → 取 [0]
output normalize 到 0.9
```

关键：model 第二个参数是 **FBank**（`[B,T',80]`），ECAPA 在 model 内部被调用。

## 全量性能说明

TSE 的 WeSep 输出**依赖 enrollment**，每个 (enroll, mixture) 组合都要跑一次，
inference 次数 = trial 数（百万级），比 baseline 慢。

已优化：批量 WeSep（`batch_size=16`）、enrollment embedding 预计算、批量 ERes2Net。

## 注意事项

1. **Python 版本**：`download_local.sh` 里的 `PY_VER` 默认 `38`，按服务器 Python 版本改（3.8→38, 3.9→39）。
2. **英文 vs 中文**：模型在 VoxCeleb1（英文）训练，英文 overlap 效果好，中文可能退化。
3. **target/nontarget 都正确**：nontarget 的 mixture 含 D（无 A），WeSep 用 A 提取不到 → 低 cosine。
4. **ERes2Net ckpt**：服务器需已有（在 `sv-eres2net/pretrained/`），`config.yaml` 里 `ckpt_rel` 指向它。
