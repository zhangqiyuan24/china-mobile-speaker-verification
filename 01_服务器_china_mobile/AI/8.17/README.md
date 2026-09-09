# FullSubNet+ 降噪前端 + ERes2Net 声纹评测

用 FullSubNet+（ICASSP 2022 语音增强模型）对 `noise_0db` / `noise_m5db` 条件做降噪前端，
再喂给 ERes2Net 提 embedding。

## 模型

| 组件 | 模型 | 采样率 | 来源 |
|------|------|--------|------|
| 降噪前端 | FullSubNet+ | 16kHz | Google Drive checkpoint |
| 声纹 | ERes2Net-large | 16kHz | 3D-Speaker |

FullSubNet+ 是**通用降噪**，输入带噪语音 → 输出干净语音，**不需要 enrollment**（和 WeSep TSE 不同）。

## 原理

```
noise_0db 的 noisy wav (16kHz)
      ↓ FullSubNet+ 降噪
enhanced wav (16kHz)
      ↓ ERes2Net-large
speaker embedding → cosine → EER
```

## 离线部署（服务器无网络）

### 本地（有网络）

```bash
cd AI/8.17
bash download_local.sh
# 下载: checkpoint (Google Drive) + 代码拷贝 + 依赖 wheel
```

### 服务器（无网络）

```bash
# 把整个 AI/8.17/ 传到服务器
cd AI/8.17
bash setup_offline.sh
# 离线装依赖 + 验证 import + 检查 checkpoint
```

## 推理（降噪）

```bash
cd AI/8.17/code
python -m speech_enhance.tools.inference \
    -C config/inference.toml \
    -M ../models/best_model.tar \
    -I <noisy_wav目录> \
    -O <输出目录>
```

输入是 wav 目录（会递归找所有 .wav），输出也是目录。16kHz 自动处理。

## 依赖

| 包 | 用途 | 是否纯 Python |
|----|------|--------------|
| librosa | 读 wav + STFT | 纯 Python（依赖 numba/scipy） |
| torch_complex | 复数张量 | 纯 Python |
| toml | 读配置 | 纯 Python |
| tqdm / soundfile / torch | 进度/写wav/框架 | 服务器已有 |

## checkpoint 下载备选方案（Google Drive 卡住时）

FullSubNet+ 的 checkpoint 只有 Google Drive 一个源（无国内镜像）。如果 `gdown` 卡住：

### A. 用代理（如果有 VPN/代理）

```bash
export https_proxy=http://127.0.0.1:7890   # 改成你的代理端口
export http_proxy=http://127.0.0.1:7890
bash download_local.sh
```

### B. 浏览器手动下载

浏览器（挂代理）打开：
```
https://drive.google.com/file/d/1UJSt1G0P_aXry-u79LLU_l9tCnNa2u7C/view
```
下载后重命名为 `models/best_model.tar`。

### C. 换 FullSubNet 原始版（不推荐，模型不同）

[Audio-WestlakeU/FullSubNet](https://github.com/Audio-WestlakeU/FullSubNet/releases) 的 GitHub Releases 有 checkpoint（GitHub 国内可访问），但那是 **FullSubNet 原始版**（无 channel attention + 复数谱），不是 FullSubNet+，模型结构不同，需要换代码。

## 文件清单

```
AI/8.17/
├── download_local.sh    # 本地下载 checkpoint + 依赖 + 代码
├── setup_offline.sh     # 服务器离线部署
├── README.md
├── code/                # FullSubNet+ 代码 (download_local.sh 生成)
├── models/              # checkpoint (download_local.sh 生成)
└── wheels/              # 依赖 wheel (download_local.sh 生成)
```
