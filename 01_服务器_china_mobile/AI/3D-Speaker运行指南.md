# 3D-Speaker 代码库完整运行指南

> 基于 `code/3D-speaker/3D-speaker/` 分析整理，2026-07-22

---

## 一、代码库总览

这是阿里巴巴达摩院开源的 **3D-Speaker** 声纹识别工具包，包含 10 种模型架构、9 个任务配方、完整的训练/推理/评测管线。

```
code/3D-speaker/3D-speaker/
├── speakerlab/              ★ 核心 Python 库（你写代码时主要 import 的对象）
│   ├── bin/                 ★ 入口脚本（训练/推理/提取/评测）
│   ├── models/              ★ 模型架构定义
│   ├── dataset/             数据集类
│   ├── loss/                损失函数（ArcMargin / KeLeo / DINO）
│   ├── process/             音频处理 / 数据增强 / 调度器
│   └── utils/               配置 / 断点 / 指标计算 / 文件读写
│
├── egs/3dspeaker/           ★ 任务配方（每个子文件夹是一个独立实验）
│   ├── sv-ecapa/            ECAPA-TDNN 声纹确认
│   ├── sv-cam++/            CAM++ 声纹确认
│   ├── sv-eres2net/         ERes2Net 声纹确认
│   ├── sv-eres2netv2/       ERes2NetV2 声纹确认 ★ 推荐首选
│   ├── sv-res2net/          Res2Net 声纹确认
│   ├── sv-resnet/           ResNet34 声纹确认
│   ├── sv-rdino/            RDINO 自监督声纹确认
│   ├── speaker-diarization/ 说话人日志（含视频多模态）
│   └── language-identification/  语种识别
│
├── runtime/onnxruntime/     ONNX 部署
├── requirements.txt         PyTorch + torchaudio + scipy + sklearn + ...
└── README.md                官方文档
```

### 每个配方（如 `sv-eres2netv2/`）的统一结构

```
sv-eres2netv2/
├── run.sh              ★ 一键运行脚本（分 Stage 1~6）
├── path.sh             环境变量（自动链接 speakerlab）
├── speakerlab          → ../../speakerlab 的软链接
├── conf/
│   ├── eres2netv2.yaml      训练配置
│   └── eres2netv2_lm.yaml   Large-Margin 微调配置
├── local/
│   ├── download_data.sh     下载 3D-Speaker 数据集
│   ├── prepare_data.sh      数据解压与组织
│   └── prepare_data_csv.py  生成训练索引 CSV
└── utils/
    ├── parse_options.sh     Shell 参数解析
    └── ...
```

---

## 二、模型速查表

| 模型 | 参数量 | 3D-Speaker Cross-Device EER | 特点 | 推荐场景 |
|------|--------|---------------------------|------|----------|
| **ERes2NetV2** | 17.8M | 6.52% | 当前最强，含 large-margin 微调 | ★ 子课题一首选 |
| **ERes2Net-large** | 22.5M | 6.34% | 大参数量，效果最好 | 冲刺挑战指标 |
| **CAM++** | 7.2M | 7.75% | 轻量高效，适合做小模型对比 | 小模型方案 |
| **ECAPA-TDNN** | 20.8M | 8.87% | 经典基线，论文标准对比对象 | 基线模型 |
| **ResNet34** | 6.3M | 7.29% | 通用 backbone | 额外对比 |
| **Res2Net** | 4.0M | 8.03% | 最轻量 | 消融实验 |
| **RDINO** | 45.4M | 20.41% | 自监督（不需标签），效果较差 | 自监督探索 |

> EER 数据来自官方 README。完整表格见 `egs/3dspeaker/README.md`。

---

## 三、环境搭建（只需做一次）

### 3.1 安装依赖

```bash
# 进入代码根目录
cd code/3D-speaker/3D-speaker

# 创建虚拟环境（推荐）
conda create -n 3D-Speaker python=3.8
conda activate 3D-Speaker

# 安装 PyTorch（根据你的 CUDA 版本选择）
# CUDA 11.8:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其余依赖
pip install -r requirements.txt

# 额外安装（推理时需要）
pip install modelscope kaldiio
```

### 3.2 验证环境

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "from speakerlab.models.ecapa_tdnn.ECAPA_TDNN import ECAPA_TDNN; print('speakerlab OK')"
```

---

## 四、标准运行流程（5~6 步走）

每个配方 `run.sh` 都遵循完全相同的流水线。以 **sv-eres2netv2** 为例：

### Stage 1: 准备数据

```bash
cd egs/3dspeaker/sv-eres2netv2
bash run.sh --stage 1 --stop_stage 1
```

**实际执行**：`local/prepare_data.sh` → 下载 3D-Speaker 数据集并组织为：
```
data/
├── 3dspeaker/
│   ├── train/        # 训练集 wav + wav.scp + utt2spk
│   ├── test/         # 测试集 wav + wav.scp + utt2spk + trials/
│   └── trials/       # trials_cross_{device,distance,dialect}
├── musan/            # 噪声数据（在线增强用）
│   └── wav.scp
└── rirs/             # RIR 数据（在线混响用）
    └── wav.scp
```

> ⚠️ **如果你的 SpeakerProject 已经解压好了**，需要把数据放到 `data/` 下对应的位置，或修改 `run.sh` 中 `data=你的路径`。

### Stage 2: 生成训练索引

```bash
bash run.sh --stage 2 --stop_stage 2
```

**实际执行**：`python local/prepare_data_csv.py --data_dir data/3dspeaker/train`
→ 生成 `train.csv`（每行：`utt_id, speaker_id, wav_path, duration`）

### Stage 3: 训练模型

```bash
# 单卡训练（修改 run.sh 中 gpus="0"）
bash run.sh --stage 3 --stop_stage 3

# 多卡训练（默认 4 卡）
bash run.sh --stage 3 --stop_stage 3 --gpus "0,1,2,3"
```

**实际执行**：
```bash
torchrun --nproc_per_node=4 speakerlab/bin/train.py \
    --config conf/eres2netv2.yaml \
    --gpu 0 1 2 3 \
    --data data/3dspeaker/train/train.csv \
    --noise data/musan/wav.scp \
    --reverb data/rirs/wav.scp \
    --exp_dir exp/eres2netv2
```

**训练参数（在 `conf/*.yaml` 里改）**：
| 参数 | 默认值 | 含义 |
|------|--------|------|
| `num_epoch` | 70 | 训练轮数 |
| `batch_size` | 256 | 每卡 batch size |
| `wav_len` | 3.0 | 训练片段长度（秒） |
| `aug_prob` | 0.8 | 在线增强概率 |
| `lr` | 0.2 | 初始学习率（SGD） |
| `embedding_size` | 192 | 声纹嵌入维度 |
| `num_classes` | 10000 | 训练说话人数（自动从数据统计） |

### Stage 4: Large-Margin 微调（仅 ERes2Net/ERes2NetV2 有）

```bash
bash run.sh --stage 4 --stop_stage 4
```

用更大的 margin 微调最后的分类头，提升区分性。

### Stage 5: 提取测试集嵌入

```bash
bash run.sh --stage 5 --stop_stage 5
```

**实际执行**：
```bash
torchrun --nproc_per_node=8 speakerlab/bin/extract.py \
    --exp_dir exp/eres2netv2_lm \
    --data data/3dspeaker/test/wav.scp \
    --use_gpu --gpu 0 1 2 3
```

→ 输出 `exp/eres2netv2_lm/embeddings/xvector_*.ark` 和 `xvector_*.scp`

### Stage 6: 计算评测指标

```bash
bash run.sh --stage 6 --stop_stage 6
```

**实际执行**：
```bash
python speakerlab/bin/compute_score_metrics.py \
    --enrol_data exp/eres2netv2_lm/embeddings \
    --test_data exp/eres2netv2_lm/embeddings \
    --scores_dir exp/eres2netv2_lm/scores \
    --trials data/3dspeaker/trials/trials_cross_device \
             data/3dspeaker/trials/trials_cross_distance \
             data/3dspeaker/trials/trials_cross_dialect
```

→ 输出：
- `exp/eres2netv2_lm/scores/result.metrics` — EER / minDCF 结果
- `exp/eres2netv2_lm/scores/trials_cross_device.score` — 每条试验的分数

---

## 五、直接用预训练模型（不需要训练）

如果 GPU 资源有限，可以跳过训练，直接下载官方预训练模型做推理和评测。

### 5.1 交互式推理（两句话比相似度）

```bash
# 在 code/3D-speaker/3D-speaker/ 目录下
python speakerlab/bin/infer_sv.py \
    --model_id iic/speech_eres2netv2_sv_zh-cn_16k-common \
    --wavs 张三的语音.wav 待验证语音.wav

# 输出: The similarity score between two input wavs is 0.8723
```

### 5.2 批量提取嵌入

```bash
python speakerlab/bin/infer_sv_batch.py \
    --model_id iic/speech_eres2netv2_sv_zh-cn_16k-common \
    --wavs wav.list \
    --feat_out_dir ./embeddings \
    --feat_out_format npy
```

### 5.3 可用预训练模型 ID

| model_id | 模型 | 训练数据 |
|----------|------|----------|
| `iic/speech_eres2netv2_sv_zh-cn_16k-common` | ERes2NetV2 | 20 万中文说话人 |
| `iic/speech_eres2net_sv_zh-cn_16k-common` | ERes2Net-huge | 20 万中文说话人 |
| `iic/speech_campplus_sv_zh-cn_16k-common` | CAM++ | 20 万中文说话人 |
| `iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k` | ERes2Net-large | 3D-Speaker |
| `iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k` | ERes2Net-base | 3D-Speaker |
| `iic/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k` | ECAPA-TDNN | 3D-Speaker |
| `iic/speech_campplus_sv_zh_en_16k-common_advanced` | CAM++ | 中英混合 |

> 首次运行会自动从 ModelScope 下载模型权重到 `pretrained/` 文件夹。

---

## 六、如何对接到你的 SpeakerProject 和子课题一

### 6.1 用 SpeakerProject 的测试数据评测模型

你的 `SpeakerProject/` 已经包含了所有需要的测试资源：

| SpeakerProject 路径 | 在 3D-Speaker 流水线中的角色 |
|---------------------|---------------------------|
| `SpeakerProject/test/` | → `data/3dspeaker/test/` 下各说话人文件夹 |
| `SpeakerProject/files/test_wav.scp` | → Stage 5 提取嵌入的输入 |
| `SpeakerProject/files/test_utt2spk` | → 语音到说话人映射 |
| `SpeakerProject/trials_cross_device` | → Stage 6 评测的 trial 文件 |
| `SpeakerProject/trials_cross_dialect` | → Stage 6 评测的 trial 文件 |
| `SpeakerProject/trials_cross_distance` | → Stage 6 评测的 trial 文件 |

### 6.2 最小运行步骤（只用预训练模型 + 你的测试集）

```bash
# 1. 建立软链接，把 SpeakerProject 接到 data 目录下
cd egs/3dspeaker/sv-eres2netv2
mkdir -p data/3dspeaker/test
ln -s /你的路径/SpeakerProject/test data/3dspeaker/test/
mkdir -p data/3dspeaker/trials
ln -s /你的路径/SpeakerProject/trials_cross_device data/3dspeaker/trials/
ln -s /你的路径/SpeakerProject/trials_cross_dialect data/3dspeaker/trials/
ln -s /你的路径/SpeakerProject/trials_cross_distance data/3dspeaker/trials/
cp /你的路径/SpeakerProject/files/test_wav.scp data/3dspeaker/test/

# 2. 用预训练模型提取嵌入（跳过训练，直接提取）
python speakerlab/bin/infer_sv_batch.py \
    --model_id iic/speech_eres2netv2_sv_zh-cn_16k-common \
    --wavs data/3dspeaker/test/test_wav.scp \
    --feat_out_dir exp/eres2netv2_lm/embeddings \
    --feat_out_format ark

# 3. 计算评测指标
python speakerlab/bin/compute_score_metrics.py \
    --enrol_data exp/eres2netv2_lm/embeddings \
    --test_data exp/eres2netv2_lm/embeddings \
    --scores_dir exp/eres2netv2_lm/scores \
    --trials data/3dspeaker/trials/trials_cross_device \
             data/3dspeaker/trials/trials_cross_distance \
             data/3dspeaker/trials/trials_cross_dialect

# 4. 查看结果
cat exp/eres2netv2_lm/scores/result.metrics
```

### 6.3 子课题一的多条件测试扩展

对于 `run.sh` 中原生没有覆盖的测试场景，你需要自己构建：

| 子课题一场景 | 如何构建 |
|-------------|----------|
| **干净语音** | 直接用 SpeakerProject test set + 同设备 trial |
| **-5dB / 0dB 噪声** | 对 SpeakerProject 的 wav 叠加 MUSAN 噪声 → 生成新的 wav.scp → 提取嵌入 → 评测 |
| **回声/混响** | 对 wav 卷积 RIR → 生成新 wav.scp → 提取嵌入 → 评测 |
| **2-3 人重叠** | 混合 2-3 个说话人的 wav → 生成新 wav.scp → 提取嵌入 → 评测 |
| **跨设备** | 已有（`trials_cross_device`） |
| **短语音** | 裁剪 wav 到 1s/3s/5s → 生成新 wav.scp → 提取嵌入 → 评测 |

每个场景的核心工作流：
```
原始 wav → [增强处理] → 生成 wav.scp → extract.py 提取嵌入 → compute_score_metrics.py 评测
```

---

## 七、关键 Python API 速查

### 7.1 核心入口脚本一览

| 脚本 | 功能 | 调用方式 |
|------|------|----------|
| `speakerlab/bin/train.py` | 训练声纹模型 | `torchrun --nproc_per_node=N train.py --config xxx.yaml --gpu 0 1` |
| `speakerlab/bin/extract.py` | 提取测试集嵌入（分布式） | `torchrun --nproc_per_node=N extract.py --exp_dir EXP --data wav.scp --use_gpu` |
| `speakerlab/bin/infer_sv.py` | 单/双文件推理 + 相似度 | `python infer_sv.py --model_id ID --wavs a.wav b.wav` |
| `speakerlab/bin/infer_sv_batch.py` | 批量提取嵌入 | `python infer_sv_batch.py --model_id ID --wavs list --feat_out_dir DIR` |
| `speakerlab/bin/compute_score_metrics.py` | 计算 EER/minDCF | `python compute_score_metrics.py --trials trial.txt --enrol_data ...` |

### 7.2 在自己的代码中调用

```python
import sys
sys.path.append('code/3D-speaker/3D-speaker')

from speakerlab.models.eres2net.ERes2NetV2 import ERes2NetV2
from speakerlab.process.processor import FBank
from speakerlab.utils.score_metrics import compute_eer, compute_c_norm

# 加载模型
model = ERes2NetV2(feat_dim=80, embedding_size=192)
model.load_state_dict(torch.load('pretrained.ckpt'))
model.eval()

# 提取 FBank 特征
fbank = FBank(n_mels=80, sample_rate=16000, mean_nor=True)
feat = fbank(wav_tensor)  # wav_tensor shape: (1, samples)

# 计算嵌入
with torch.no_grad():
    embedding = model(feat.unsqueeze(0))  # → (1, 192)

# 评测
eer, threshold = compute_eer(fnr, fpr, scores)
min_dcf = compute_c_norm(fnr, fpr, p_target=0.01, c_miss=1, c_fa=1)
```

---

## 八、常见问题

### Q1: 我只有一张 GPU 卡，怎么跑？

修改对应 `run.sh` 中的 `gpus` 变量：
```bash
gpus="0"   # 只用 0 号卡
```
训练时自动检测 GPU 数量并适配 DDP。

### Q2: 我连一张 GPU 都没有，能不能跑？

- **训练**：不行，必须 GPU。租 AutoDL 的 RTX 4090 约 3-5 元/小时，训练 ECAPA-TDNN 约 8-12 小时
- **推理/评测**：可以用 CPU，但很慢。`infer_sv.py` 会自动检测并回退到 CPU；`extract.py` 去掉 `--use_gpu` 即可

### Q3: 训练数据 wav.scp 格式是什么？

每行：`<语音ID> <wav路径>`
```
3D_SPK_06154_001_Device03_Distance00_Dialect00 test/3D_SPK_06154/3D_SPK_06154_001_Device03_Distance00_Dialect00.wav
```

### Q4: trial 文件格式是什么？

每行：`<注册语音ID> <测试语音ID> <target/nontarget>`
```
3D_SPK_07304_012_Device06_Distance10_Dialect00 3D_SPK_07304_008_Device03_Distance03_Dialect00 target
```

### Q5: 训练出来的模型在哪个文件？

`exp/<exp_name>/models/CKPT-EPOCH-70-00/` 目录下，`embedding_model.ckpt` 是声纹提取器权重，`classifier.ckpt` 是分类头（推理时不需要）。

### Q6: SpeakerProject 的 wav.scp 里的路径是相对路径，跑不起来？

如果 `wav.scp` 里写的是相对路径（如 `test/3D_SPK_06154/xxx.wav`），需要确保运行脚本时的工作目录包含 `test/` 文件夹。或者用 `sed` 把路径改成绝对路径：
```bash
sed -i "s|^\(.*\) test/|\\1 /绝对路径/SpeakerProject/test/|" test_wav.scp
```
