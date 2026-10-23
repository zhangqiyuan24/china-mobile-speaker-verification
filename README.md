# 多条件中文声纹识别与噪声鲁棒性增强(中国移动"梧桐·鸿鹄"实习 · 子课题一)

面向 AI 眼镜、车载座舱等中文智能终端的**开集声纹验证**:在跨设备、低信噪比(0 dB / -5 dB)、混响、多人重叠等复杂声学条件下,把说话人确认错误率(EER)大幅压低。

> 2026-10-07 备份更新：已同步本地终版交付材料与最终提交归档。最新备份范围、训练表述纠正和指标口径见 [BACKUP_STATUS.md](BACKUP_STATUS.md)。结果表为 oracle 条件化评测，claim 部署检验结果需独立阅读；同配方延长续训没有失败样本加权重采样。服务器本次连接超时，原服务器镜像保持历史版本。

**核心做法**:注册条件化目标说话人提取前端(SpeakerFiLM + FullSubNet+)——用注册语音的声纹嵌入对降噪网络做特征级仿射调制,让前端"只提取注册说话人的语音",配合四损失训练(重建 + 说话人保持 + 难负原型抑制 + AAM 判别分类)。

## 核心结果(官方打分清单 1,378,206 条 trial,3s 验收档)

| 后端 | 0 dB EER | -5 dB EER | 相对降幅 | 验收线 |
|---|---|---|---|---|
| ERes2Net-large(主力) | 8.39 → **4.89** | 11.06 → **5.56** | **-41.7% / -49.7%** | ≥30% ✅ |
| ECAPA-TDNN(经典) | 11.15 → 6.32 | 14.36 → 7.13 | -43.3% / -50.3% | ≥30% ✅ |
| ReDimNet2-B6(轻量) | 9.94 → 7.58 | 13.03 → 9.35 | -23.7% / -28.2% | 未过线 |

clean 语音无退化(5.29 → 4.17,正增益);混响 -20.0%;2-3 人重叠 -15.0%~-35.2%;整链 RTF ≈ 0.024,实时性无虞。完整分项见 `00_本地_交付与文档/交付材料/01_报告/技术报告_跨设备噪声鲁棒声纹识别_v2.pdf`。

## 补充报告（2026-10-17）

[`00_本地_交付与文档/补充_声纹特征对比/`](00_本地_交付与文档/补充_声纹特征对比/) 对“为什么通用 FullSubNet+ 降噪后声纹 EER 反而变差”做了四路对比与指标分析：原方法 EER **10.11%**，TSE **4.89%**；原方法 SI-SDR/STOI 更好，但 embedding 漂移更大（0.714 vs Noisy 0.820）。含 25 页补充 PDF、可试听 `listen.html`、图、逐句数据与复现脚本；原报告的任务/数据/TSE 结构/主 EER 不重复。

## 仓库结构

```
├── README.md                          本文件
├── scripts/download_data.sh           数据与预训练模型一键获取脚本
├── 00_本地_交付与文档/                 交付材料与文档(本地侧全集)
│   ├── 交付材料/                       00_交付清单 + 报告/PPT/讲稿/代码/指标/图/数据生成与评测协议
│   ├── report_v2.html                  技术报告源文件(渲染 PDF 用)
│   ├── 实验结果汇总_9.3.md             全部实验结果总账
│   ├── 邓姣姣PPT_对照问题清单.txt       第二位成员 PPT 的事实核对清单
│   └── figs/ 绘图/                     报告与 PPT 用图
└── 01_服务器_china_mobile/             服务器工作区镜像(实验原貌)
    ├── AI/                             按日期组织的实验目录(8.9-9.6),代码+日志+指标
    │   ├── 8.18/                       FullSubNet+ 通用降噪基座训练
    │   ├── 8.20~8.24/                  后端微调路线(否决)与权重汤 soup5
    │   ├── 8.26/                       条件化前端模型与训练代码(tse_model.py 等)
    │   ├── 8.27/                       四损失训练、消融、注册侧诊断、交付前端 r5
    │   ├── 8.28/                       交付终评大表、场景/噪声域/clean 直测
    │   ├── 8.30/                       ROC/DET 曲线、失败案例挖掘
    │   ├── 8.31/                       部署口径(claim)分析与 15 臂补救
    │   └── 9.3/ 9.4/                   参数量普查、WavLM 对比、clean 基准补全、交付材料
    ├── code/
    │   ├── 3D-speaker/                 3D-Speaker 工具包(官方) + 本项目全部自定义脚本
    │   │   └── .../egs/3dspeaker/sv-ecapa/   ← 项目主工作区(数据生成/评测脚本在此)
    │   ├── FullSubNet-plus-master/     前端骨干官方代码(对照)
    │   ├── redimnet2/ wesep-master/    轻量后端 / WeSep(探索后弃用)
    │   └── 注意.docx
    ├── 周报/                           6 期周报
    └── 任务.xlsx                       任务书
```

## 快速开始(从零复现)

### 0. 环境

Linux + Python 3.8+、PyTorch(实验用 1.x,CUDA GPU 4×Tesla P100)、`pip install modelscope soundfile librosa tqdm`。3D-Speaker 工具包本身无重依赖,进入 `code/3D-speaker/3D-speaker/` 按其 README 安装即可。

### 1. 获取数据与预训练模型

```bash
bash scripts/download_data.sh /path/to/本仓库
```

脚本自动下载 **MUSAN**(OpenSLR 17,噪声)与 **RIRS_NOISES**(OpenSLR 28,混响),自动拉取三个后端的 **3D-Speaker 中文预训练模型**(ModelScope);**3D-Speaker 原始数据集**(约 190 GB)需在 [OpenDataLab](https://opendatalab.com/3DSpeaker) 注册后手动下载,脚本会打印放置路径。全部就位后目录形如:

```
data/musan/  data/rirs_noises/  data/3dspeaker/{train,test}/  pretrained/
```

### 2. 生成统一多条件测试基准

```bash
cd code/3D-speaker/3D-speaker/egs/3dspeaker/sv-ecapa/
bash gen_all_test_data.sh        # 一键:噪声/RIR 训测拆分 → 8 条件 × 3 时长测试集 → trial 清单
```

生成产物:`data/data_for_test/<条件>/<1s|3s|5s>/`(角色 A-F 目录、enrollment.csv、trial.csv、trials_cross_device 官方打分清单 1,378,206 条)。协议细节见 `00_本地_交付与文档/交付材料/10_数据生成与评测协议/评测协议_统一多条件基准.md`。

### 3. 训练通用基座 → 条件化前端

```bash
# 通用降噪基座(FullSubNet+,10 轮) —— AI/8.18/
python train_spk_preserve.py ...          # 配置见 AI/8.18/ 留档
# 注册条件化前端 + 四损失训练(交付版 r5)—— AI/8.26/ AI/8.27/
python train_tse.py --config train_tse.toml    # 模型定义 tse_model.py(SpeakerFiLM)
```

训练超参、代理验证集选点与"官方测试单次实测"纪律见技术报告 §5(逐项对应 `train_tse.toml`)。

### 4. 推理增强与官方评测

```bash
python enhance_tse.py ...                  # 批量增强官方测试集(交付入口)
bash evaluate_noise.sh / evaluate_reverb.sh / evaluate_overlap.sh / evaluate_clean.sh
python gen_results_summary.py              # 汇总全部 result.metrics
```

每格输出 `result.metrics`(EER/minDCF/固定工作点),对照 `01_服务器_china_mobile/AI/8.28/` 留档结果即可核对复现是否一致(全部数字与《实验结果汇总》《技术报告》同源)。

## 未入库内容(体积原因)与再生方式

| 未入库 | 再生方式 |
|---|---|
| 全部音频(wav:原始/增强/样本) | `download_data.sh` + `gen_all_test_data.sh` / `enhance_tse.py` |
| 模型权重(前端 9.2M、后端 29M、soup5 等) | 第 3 步训练重产;后端可用 ModelScope 预训练权重直接对表 |
| 中间特征(嵌入 ark、原型 npz、逐 trial score) | 推理/评测脚本自动重产 |

## 主要参考与第三方代码

3D-Speaker(数据集与工具包,Apache-2.0)、FullSubNet / FullSubNet+(ICASSP 2021/2022)、VoiceFilter(Interspeech 2019)、FiLM(AAAI 2018)、ECAPA-TDNN(Interspeech 2020)、ERes2Net(Interspeech 2023)、AAM-Softmax(CVPR 2019)、WavLM(IEEE JSTSP 2022)。完整文献列表见技术报告文末参考文献。

---
*项目成员:张启元(中国科学院大学,算法与实验主线)、邓姣姣(数据与评测支撑);指导单位:中国移动(梧桐·鸿鹄暑期实习,2026.7-9)。*
