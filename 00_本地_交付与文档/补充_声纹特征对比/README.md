# 声纹特征对比 · 补充报告

本目录是《多条件中文声纹识别与噪声鲁棒性增强》技术报告的**补充分析**，回答：

> 为什么原通用 FullSubNet+ 降噪后，声纹识别 EER 反而变差？

原报告已有的任务背景、数据与 trial 协议、TSE 结构、主 EER 交付大表不在本目录重复；这里只放新增的四路对比、指标口径、失效机理、结果索引与面试口径。

## 文件

- `中国移动实习_声纹特征对比_补充报告.pdf`：25 页补充报告（新版式，深藏青标题/宋体正文）
- `listen.html`：5 条样本四路试听 + 时频图 + 关键结果（推荐先开）
- `分析报告.md`：文字版
- `figs/`：时频图、差异图、target/embedding 分布、SI-SDR/PESQ/STOI、全局 SNR、PCA
- `data/`：逐句分数、120 条指标、全局 SNR、PESQ-SNR 扫描、embedding npz
- `wav_listen/`：等响度试听副本（−23 dBFS RMS）
- `wav/`：原始响度四路 wav（5 条样本）
- `scripts/`：复现脚本
- `report_supplement.html` / `build_supplement_report.py`：报告 HTML/MathML 生成链

## 关键结论（0 dB）

| 指标 | Noisy | FullSubNet+（原） | TSE（现有） |
|---|---:|---:|---:|
| EER (%) | 8.391 | **10.113** | **4.892** |
| target 均值 | 0.422 | 0.391 | 0.471 |
| 同句 cos 到 clean | 0.820 | **0.714** | 0.816 |
| SI-SDRi (dB) | 0 | +8.59 | +7.62 |
| 全局 SNR Δ (dB) | 0 | +3.80 | +3.32 |
| PESQ-WB | 1.305 | 1.889 | 1.958 |
| STOI | 0.654 | 0.750 | 0.739 |

原方法在 SI-SDR/SNR/STOI 上更好，但把 ERes2Net embedding 推离了 clean 的说话人流形，因此 EER 从 8.39% 升到 10.11%；TSE 的信号指标不全面占优，却把 target/embedding/EER 全面拉回（4.89%）。

## 口径

- 主分析为 0 dB；−5 dB 只补充 EER 与降噪后全局 SNR。
- EER/target/drift 覆盖 8,507 条测试句、1,378,206 条官方 trial；SI-SDR/SNR/PESQ/STOI 覆盖 120 条随机测试句。
- **全局 SNR** 为“输出缩放到输入同整体 RMS”的口径；未匹配的 raw 全局 SNR 对原方法为 −18.1 dB，是峰值归一化增益造成，不能直接比较。
- 5 条 wav 为 3D-Speaker 测试样本的短片段，仅用于本补充报告的试听/可视化，不代表完整数据集。
- 本补充报告 TSE 数字为 oracle 口径；claim 口径见原报告附录 C。

## 注意

PDF 内的文件链接是生成时的本机绝对路径（`E:\...`），公开仓库中请使用 `listen.html`、`figs/`、`data/` 的相对路径浏览。
