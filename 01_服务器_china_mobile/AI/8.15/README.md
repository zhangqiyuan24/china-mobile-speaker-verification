# 断网跑基线评测（下载模型 + 跳过联网验证）

## 问题

服务器断网，`evaluate_clean.sh` 等评测脚本调用的 `infer_sv_batch.py` / `infer_sv_new.py` 会：
1. `is_official_hub_path()` 联网验证 model_id → 超时
2. `snapshot_download()` / `load_custom()` 联网下载模型 → 超时

## 解决

本地下载 3 个模型的 checkpoint，传到服务器，patch 推理脚本改成「本地加载 + 跳过联网」。

## 模型清单

| 模型 | checkpoint | 来源 |
|------|-----------|------|
| ECAPA-TDNN | ecapa-tdnn.ckpt | ModelScope |
| ERes2Net-large | eres2net_large_model.ckpt | ModelScope |
| ReDimNet2-B6 | b6-vb2+vox2+cnc2_v0-lm.pt | GitHub Release |

## 使用

### 本地（有网络）

```bash
cd AI/8.15
bash download_models.sh
# 下载 3 个模型到 models/
```

### 服务器（无网络）

```bash
# 把整个 AI/8.15/ 传到服务器
cd AI/8.15
bash install.sh
# 自动: 部署模型到 sv-ecapa/pretrained/ + patch 两个推理脚本

# 然后断网跑评测
cd ~/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa
bash evaluate_clean.sh
```

## install.sh 做了什么

1. **部署模型**：3 个 checkpoint 复制到 `sv-ecapa/pretrained/` 对应子目录
2. **替换推理脚本**：把 AI/8.15 里的 `infer_sv_batch.py` / `infer_sv_new.py`（本地已加 `--pretrained` / ReDimNet2 支持）复制到 `speakerlab/bin/`，并备份原脚本为 `.bak`
3. **patch**（针对替换后的脚本）：
   - 跳过 `is_official_hub_path` 联网验证
   - `infer_sv_batch.py` else 分支优先从本地 `pretrained/{model_name}/{model_pt}` 加载
   - `infer_sv_new.py` ReDimNet2 优先从本地 `pretrained/redimnet2/{weight}` 加载

## 注意

1. `install.sh` 顶部 `SV_ECAPA` 变量默认指向 `~/china_mobile/code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa`，按你服务器实际路径改。
2. patch 会**直接修改**原 `infer_sv_batch.py` 和 `infer_sv_new.py`（在 speakerlab/bin 下），建议先备份。
3. ReDimNet2 的代码（`code/redimnet2/`）服务器上需已存在（`_find_redimnet2_root` 会自动找）。
