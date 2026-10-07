# 声纹鲁棒性汇报参考文献包

本压缩包收录前面 PPT 与汇报稿中提及的 13 篇文献原文，均取自 ISCA Archive、arXiv 或 CVF Open Access 的公开版本。编号与 PPT 中的论证顺序一致。

## 一、三条候选路线的选取与淘汰依据

### 路线一：直接增强/适配声纹后端

1. **01_FT_Boosted_SV_2025.pdf**  
   S. Tabatabaee, J. Liu, C. Espy-Wilson, “FT-Boosted SV: Towards Noise Robust Speaker Verification for English Speaking Classroom Environments,” Interspeech 2025. DOI: 10.21437/Interspeech.2025-1002.  
   **在本课题中的作用：**支持“通过噪声增强数据微调声纹模型可改善噪声鲁棒性”这一思路；但本课题后端冻结，因此仅作为路线依据，不作为最终实现。  
   来源：https://arxiv.org/abs/2505.20222

2. **02_NAW_SV_2024.pdf**  
   C.-Y. Lim, H.-S. Shin, J.-H. Kim, J. Heo, K.-W. Koo, S.-B. Kim, H.-J. Yu, “Improving Noise Robustness in Self-supervised Pre-trained Model for Speaker Verification,” Interspeech 2024, pp. 2665–2669. DOI: 10.21437/Interspeech.2024-1630.  
   **在本课题中的作用：**说明自监督预训练声纹模型需要专门的噪声适配策略；同样涉及后端训练，与“冻结后端”的工程约束不一致。  
   来源：https://www.isca-archive.org/interspeech_2024/lim24_interspeech.html

### 路线二：联合语音增强与声纹识别

3. **03_ParaNoise_SV_2025.pdf**  
   M. Kim, K. Jang, H. Kim, “ParaNoise-SV: Integrated Approach for Noise-Robust Speaker Verification with Parallel Joint Learning of Speech Enhancement and Noise Extraction,” Interspeech 2025. DOI: 10.21437/Interspeech.2025-1145.  
   **在本课题中的作用：**支持显式建模噪声、联合优化增强与声纹目标；但整套联合后端训练超出本课题冻结后端的边界。  
   来源：https://arxiv.org/abs/2508.07219

4. **04_URGENT_Challenge_2024.pdf**  
   W. Zhang et al., “URGENT Challenge: Universality, Robustness, and Generalizability For Speech Enhancement,” Interspeech 2024, pp. 4868–4872. DOI: 10.21437/Interspeech.2024-1239.  
   **在本课题中的作用：**说明通用语音增强需要同时考虑降噪、去混响、带宽扩展与反削波等多类退化，并采用多指标评价；为前端增强路线和混响/削波数据构建提供依据。  
   来源：https://www.isca-archive.org/interspeech_2024/zhang24h_interspeech.html

### 路线三：目标说话人提取（最终路线）

5. **05_TSE_Curriculum_Learning_2024.pdf**  
   Y. Liu, X. Liu, X. Miao, J. Yamagishi, “Target Speaker Extraction with Curriculum Learning,” Interspeech 2024, pp. 4348–4352. DOI: 10.21437/Interspeech.2024-1375.  
   **在本课题中的作用：**支持按说话人相似度、性别、信号失真等难度组织 TSE 训练样本；说明重叠语音可通过“目标说话人条件”而非普通降噪处理。  
   来源：https://www.isca-archive.org/interspeech_2024/liu24j_interspeech.html

6. **06_WeSep_2024.pdf**  
   S. Wang et al., “WeSep: A Scalable and Flexible Toolkit Towards Generalizable Target Speaker Extraction,” Interspeech 2024, pp. 4273–4277. DOI: 10.21437/Interspeech.2024-1840.  
   **在本课题中的作用：**提供目标说话人建模、在线数据仿真、训练配方与部署支持的工程参考；支撑最终选择“说话人条件前端 + 冻结声纹后端”。  
   来源：https://www.isca-archive.org/interspeech_2024/wang24fa_interspeech.html

## 二、FullSubNet 前端结构

7. **07_FullSubNet_2021.pdf**  
   X. Hao, X. Su, R. Horaud, X. Li, “FullSubNet: A Full-Band and Sub-Band Fusion Model for Real-Time Single-Channel Speech Enhancement,” ICASSP 2021. DOI: 10.1109/ICASSP39728.2021.9414177.  
   **在本课题中的作用：**全带分支建模跨频率全局依赖，子带分支利用局部频谱邻域与窄带平稳性，两者融合预测掩码；这是本课题前端主干的理论来源。  
   来源：https://arxiv.org/abs/2010.15508

8. **08_FullSubNet_Plus_2022.pdf**  
   J. Chen, Z. Wang, D. Tuo, Z. Wu, S. Kang, H. Meng, “FullSubNet+: Channel Attention FullSubNet with Complex Spectrograms for Speech Enhancement,” ICASSP 2022.  
   **在本课题中的作用：**补充复数谱、相位信息、多尺度通道注意力和 TCN 全带提取器，为理解 FullSubNet 系列的改进方向及 PPT 中参考结构图提供依据。  
   来源：https://arxiv.org/abs/2203.12188

## 三、FiLM 与说话人条件调制

9. **09_FiLM_2018.pdf**  
   E. Perez, F. Strub, H. de Vries, V. Dumoulin, A. Courville, “FiLM: Visual Reasoning with a General Conditioning Layer,” AAAI 2018.  
   **在本课题中的作用：**给出特征级仿射调制的原始定义：条件向量经参数生成器得到 γ、β，并对中间特征执行 γ⊙F+β；说明 γ、β是动态调制参数，不是重新生成语音或重新生成网络。  
   来源：https://arxiv.org/abs/1709.07871

10. **10_Speaker_Conditional_Sinc_Extractor_2024.pdf**  
    E.-L. Yu, K.-H. Ho, J.-W. Hung, S.-C. Huang, B. Chen, “Speaker Conditional Sinc-Extractor for Personal VAD,” Interspeech 2024, pp. 2115–2119. DOI: 10.21437/Interspeech.2024-365.  
    **在本课题中的作用：**展示把高维说话人 d-vector 注入低维声学特征的说话人条件化实例；支撑“小型参数生成器 + 声纹条件调制”的设计动机。  
    来源：https://www.isca-archive.org/interspeech_2024/yu24_interspeech.html

## 四、训练目标与判别约束

11. **11_GE2E_Loss_2018.pdf**  
    L. Wan, Q. Wang, A. Papir, I. Lopez Moreno, “Generalized End-to-End Loss for Speaker Verification,” ICASSP 2018.  
    **在本课题中的作用：**说明如何以批内说话人中心和余弦相似度构造可微判别目标；用于理解增强输出经过冻结声纹网络后，损失梯度如何继续回传到前端。  
    来源：https://arxiv.org/abs/1710.10467

12. **12_Prototypical_Networks_2017.pdf**  
    J. Snell, K. Swersky, R. S. Zemel, “Prototypical Networks for Few-shot Learning,” NeurIPS 2017.  
    **在本课题中的作用：**给出“类原型/中心 + 距离”分类思想；对应本课题负类原型或说话人中心约束的理论参照。  
    来源：https://arxiv.org/abs/1703.05175

13. **13_ArcFace_2019.pdf**  
    J. Deng, J. Guo, N. Xue, S. Zafeiriou, “ArcFace: Additive Angular Margin Loss for Deep Face Recognition,” CVPR 2019.  
    **在本课题中的作用：**提供归一化嵌入空间中的角度间隔判别思想；虽源于人脸识别，但其余弦/角度分类原则可用于解释说话人分类损失如何拉开类间边界。  
    来源：https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html

## 阅读建议

- 准备“为什么选这条路线”：先看 01–06。
- 准备“FullSubNet 为什么有全带和子带”：重点看 07 的方法图，再对照 08 的改进图。
- 准备“γ、β怎么得到、FiLM是否生成新语音”：看 09 的 FiLM 定义，再看 10 的声学条件化用法。
- 准备“四项损失如何通过冻结后端训练前端”：重点看 11–13，并结合本课题报告中的四项损失定义；这三篇只提供相关判别思想，不等同于本课题四项损失的完整配方。

> 说明：论文版权仍归作者或出版方所有。本压缩包用于课题学习和答辩准备，请按各来源页面的许可条款使用与引用。
