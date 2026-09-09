# Speaker Information Preservation fine-tuning

让 FullSubNet+ 在降噪的同时，保留说话人身份信息，使增强语音的 ERes2Net embedding 尽量接近对应 clean 语音的 embedding。

## 训练结构

```
clean ──▶ DiffFBank ──▶ Frozen ERes2Net ──▶ e_clean (no_grad)
noisy ──▶ FullSubNet+ ──▶ enhanced ──▶ DiffFBank ──▶ Frozen ERes2Net ──▶ e_enh (梯度回传)
```

```
L_se   = cIRM MSE（FullSubNet+ 原 loss）
L_spk  = 1 - cosine(e_clean, e_enh)
L_total = L_se + lambda_spk * L_spk
```

- ERes2Net 全冻结，只优化 FullSubNet+
- clean 分支 no_grad，enhanced 分支梯度通过 ERes2Net + DiffFBank 回传到 FullSubNet+

## 文件清单

| 文件 | 功能 |
|------|------|
| `frontend_spk.py` | 可微 80 维 log-mel frontend + consistency test |
| `trainer_spk_preserve.py` | SpeakerPreserveTrainer（模型加载 + 训练/验证 step） |
| `train_spk_preserve.py` | 主入口（数据准备 + consistency test + 训练循环） |
| `validate_spk_preserve.py` | 独立验证（baseline vs 微调的 mean cosine） |
| `train_spk_preserve.toml` | 配置 |
| `run_train.sh` | 训练入口 |

## 使用

```bash
cd AI/8.18

# 先看 frontend consistency（官方 FBank vs DiffFBank 的 embedding cosine）
# 训练脚本会先自动跑，也可以手动跑
python -c "import frontend_spk; print('DiffFBank 定义 OK')"

# 3 epoch sanity run
bash run_train.sh --sanity

# 正式训练 10 epochs
bash run_train.sh
```

## 关键参数（train_spk_preserve.toml）

| 参数 | 值 |
|------|-----|
| lr | 1e-5 |
| epochs | 10 |
| lambda_spk | 0.5 |
| snr_range | [0, -5] |
| reverb_proportion | 0.0 |
| batch_size | 4 |

## 数据

- clean 训练：`sv-ecapa/data/data_for_train/all_wav.scp`（643123 条）
- train noise：`sv-ecapa/data/divide/noise_train_freesound.txt`（675 条）
- test noise：`sv-ecapa/data/divide/noise_test_freesound.txt`（168 条，训练完全隔离）

## 验收目标

| 条件 | baseline EER | 目标 EER |
|------|-------------|----------|
| 0 dB | 8.3911% | ≤ 5.87% |
| -5 dB | 11.06% | ≤ 7.74% |

best checkpoint 按 `EER_0db + EER_m5db` 最小保存（validate_spk_preserve.py）。
