# 8.26 — TSE 线 + ECAPA 换靶子线 + GateNet（结构授权后三线）

## 背景与决策（8.25 深夜用户指令）
1. voiceid 负结果结案（+0.06pp），8.18-FE 保留
2. floor-gap 诊断: 全部现有弹药都在抬地板, gap 轴零压缩 → 需要攻 gap 的武器
3. 用户授权加结构 + 指示: 换 ECAPA 靶子试 -30%; clean 提升多找文献广试方法

## 线① TSE（注册条件化目标说话人提取, VoiceFilter 式）— GPU2
- 依据: 单说话人噪声=干扰语音, 通用增强分不清目标; 注册嵌入条件化才能定向提取
- tse_model.py: FullSubNetPlusTSE(继承 8.17 模型, SpeakerFiLM 528K 参数,
  三路 fullband 输入 FiLM, 末层零初始化=恒等起步, 8.18 e10 权重 strict=False 装载)
- build_spk_embs.py: soup5 提的 9900×512 条件表(L2 归一), 训练/推理同分布
- train_tse.py: L_se(cIRM MSE) + 1.0*AAM-CE(冻结 soup5); b2; 30k×3ep
- 排障记录: run_tse_pipeline.sh 曾因 PY 路径笔误/冒烟 exp_tag 连带 npz 路径失败,
  手动补跑冒烟通过后 00:53 全量启动(FiLM 528K, 可训练 9.20M)
- 评测链(待训完): enhance_tse.py(用官方 enrol clean 提测试说话人嵌入, 逐 utt 条件化降噪)
  -> evaluate_combo FE 模式(换前端必先挪 frontend/* 缓存!)

## 线② ECAPA 换靶子 — GPU1(soup5 评测后自动接管, wait_ecapa.sh)
- 靶子: ECAPA 自己的基线 0dB 11.150 / -5dB 14.360 -> 达标线 7.805 / 10.052
  (我们 ERes2Net 线绝对成绩 7.41/9.37 已低于该线 => 配方平移有现实胜算)
- build_ecapa_protos.py: ECAPA 192 维原型表(先跑)
- train_ecapa_robust.py: 8.21 配方镜像(3 条件多条件, 全量微调 lr 2e-5, b30 accum3)
- 待办: evaluate_ecapa.py(ECAPA 版 loader)+官方评测; 后续 ECAPA 汤与 FE 叠加

## 线③ GateNet(8.25 深夜启动, GPU3)
- b18 accum5 AMP(BN 输入梯度 OOM 教训见 8.25 README); ep1 val avg01 8.000
- 判决窗口 ep2-3: 压不进 7.5 则提前停线

## GPU 布局: GPU0 永空 / GPU1 soup5 评测→ECAPA / GPU2 TSE / GPU3 GateNet

## 补记（01:30）
- soup5 官方 raw: 0dB 7.6139 / -5dB 9.7148（FE 模式在跑）
- 新代码: enhance_tse.py(TSE 推理, trials 正对定条件) + evaluate_ecapa.py(ECAPA 官方评测,
  整段复用 evaluate_combo.run_mode) — 均已上传编译通过
- 文献: 研究笔记_文献调研_20260826.md — NCMoE(2510.18533) 精读 =>
  线④ MoE-lite 规划(ERes2Net 中后段 K=2~3 expert + 3 类条件路由器, UMES=从基线
  ckpt 复制起步)。等 GPU 空位后按 TSE>ECAPA>gate 判决>MoE-lite 顺序启动

## 线④ MoE-lite（02:00 建好待点火, 等 GPU3 gate 判决后释放）
- 依据: NCMoE 精读(研究笔记_文献调研_20260826.md) — 条件子空间 clean 不降反升
- moe_model.py: MoEERes2Net = ERes2Net-large + layer4×K=3 专家(deepcopy=soup5,
  UMES 免 Phase I) + CondRouter(fbank→3层卷积→温度softmax γ=0.1)
- train_moe.py: 8.21 配方 + 数据集带条件标签, L = AAM + CE(router, cond);
  训练软混合 Σg·f / 测试硬 argmax(eval BN 走 running stats, 分组等价)
- 启动: bash run_moe_pipeline.sh 3 (先冒烟 64 行)
- 注意: ckpt 是 MoE 包装结构(experts.*/router.*), 官方评测需另写 loader
  (evaluate_moe.py, 模式同 evaluate_ecapa.py), anchor 看 train_moe.log [val] 行

## GateNet 结案 + MoE-lite 点火（02:50）
- gate ep1 avg01 8.000 -> ep2 8.250（劣化且远离 7.5 门槛）=> 按预定停线,
  ckpt 留档 exp/feature_gate_90k/checkpoints/（ep1/ep2/best）
- MoE-lite 排障三连后上线: ①sys.path 漏 8.18(load_eres2net) ②b16 非 3 倍数
  →b15 ③3xexpert 激活 OOM → --amp(GradScaler)。现 b15 accum6(=eff 90) AMP
- MoE 从 soup5 起步: 总参 50.44M(experts 21.33M), 冒烟 val 链路通(8.375)
- 全量观察点: racc(路由准确率)能否从 0.40 爬向 0.7+；[val] avg01 对比 soup5 的 7.375

## enhance_tse.py 冒烟通过（03:40, 三坑修复 + 协议发现）
- 协议发现(评测链坑#4): 噪声测试目录 16401 stem 同时含注册侧(Device03)与
  打分侧(Device06/07) utt, trials 只打分 9609 条, 但 FE 模式要求全部可增强
  => 条件向量改为说话人前缀(3D_SPK_XXXXX)->注册均值嵌入(与训练条件空间同分布)
- 三坑: ①脚本内 env 覆盖导致 CPU 冒烟打到占用中的 GPU2(--gpu "" 绕过)
  ②trials 正对断言对不上(见上, 改 speaker 粒度) ③分组 dict 键含 ndarray 不可哈希
- 冒烟: --limit 8 CPU 通过, 8+8 条 3.00s, rms 0.0637->0.0140(降噪生效),
  corr 0.43-0.80(语音结构保留)
- 全量增强排期: ECAPA 训完让出 GPU1 后执行(约 32802 条, GPU 上 ~1h),
  然后 evaluate_combo FE 模式(换前端必先挪 frontend 缓存!)
- ECAPA 进展: ep1 avg01 8.000 -> ep2 7.500 新最优(clean 1.0/0db 5.5/m5db 9.5)
- MoE 进展: racc 0.40->0.47 爬升中

## 8.26 上午 — TSE 全量增强落袋；ECAPA raw 官方判决；双 FE 评测上线
### 线① TSE 增强产物（exp/tse/enhanced/）
- noise_0db / noise_m5db 各 16401 条（16kHz PCM16, 共 3.0GB），抽查完整（20 条中 2 条音量极低，记录不阻塞）。
- 条件向量 = 官方注册 clean 前 8 utt → 冻结 soup5 → 均值+L2（说话人粒度，覆盖 enrol+test 两侧 stem）。

### 线② ECAPA raw 官方（trials_cross_device, checkpoint_best=ep3）
| 条件 | 基线 | ecapa-raw | 降幅 | 目标(−30%) |
|---|---|---|---|---|
| clean | 9.3100 | 7.9420 | −14.7% | — |
| 0dB | 11.1500 | 11.2024 | −0.5% | ≤7.805 ✗ |
| −5dB | 14.3600 | 13.5076 | +5.9% | ≤10.052 ✗ |
- **关键发现：小 val 与官方严重背离**（小 val ep3: 1.0/5.25/9.75 → 官方 11.2/13.5）。
  小 val=训练域分布（同说话人域/同信道），官方=跨设备+未见说话人 → 微调收益未泛化到跨设备。
  与 ERes2Net 线对照：ERes2Net 小val→官方漂移 ~2pp，ECAPA 漂移 ~6pp（骨干更弱/更易过拟合训练域）。

### 双 FE 评测上线（~08:4x）
- GPU1: evaluate_combo soup5 + TSE增强（换前端缓存已按规则搬至 eval/combo/scores_bak_818fe_0826/）
- GPU2: evaluate_ecapa ep3 + TSE增强（FE 模式同时验证"前端模型无关可迁移"验收要求）
- 日志: tse_fe_eval.log / ecapa_fe_eval.log；结果待出。

### 线④ MoE ep1 val
clean=1.250 0db=6.000 m5db=9.500 avg01=7.750（soup5 锚 7.375，ep1 落后 0.375 属 UMES 起步常态；
路由 racc ep1~0.40 → ep2 尾 0.64 持续学习。判决点 ep3。）

### 线② ECAPA-FE 官方判决（~09:2x, trials_cross_device）
| 方案 | clean | 0dB | -5dB | 0dB降幅 | -5dB降幅 | 判定 |
|---|---|---|---|---|---|---|
| ecapa baseline | 9.3100 | 11.1500 | 14.3600 | — | — | 靶子 |
| ecapa-raw(ep3) | 7.9420 | 11.2024 | 13.5076 | −0.5% | +5.9% | ✗ |
| **ecapa-FE(TSE增强)** | — | **10.5080** | **12.4323** | **−5.8%** | **−13.4%** | ✗(目标7.805/10.052) |
- 结论1: TSE 前端对 ECAPA 正向且优于 ECAPA 自身微调(raw) —— 前端模型无关性得证(验收加分项)，
  但幅度远不够 −30%。
- 结论2: ECAPA 线按计划全部跑完(复刻+迁移+FE), 噪声档未达标, 结案留档。
  最佳: FE 10.5080/12.4323。缓存 exp/ecapa_robust_90k/eval/combo/{raw,frontend}。

### 线① TSE-FE 官方判决（~09:5x, trials_cross_device）— 项目历史最佳组合
| 方案 | 0dB | -5dB | 0dB降幅 | -5dB降幅 |
|---|---|---|---|---|
| official baseline | 8.3911 | 11.0557 | — | — |
| frontend only (8.18) | 7.7902 | 9.9186 | 7.2% | 10.3% |
| raw robust (soup5) | 7.6139 | 9.7148 | 9.3% | 12.1% |
| **TSE前端 + soup5** | **7.3818** | **9.3160** | **12.0%** | **15.7%** |
- TSE 前端全面胜过 8.18 通用降噪前端（0dB −0.093pp / −5dB −0.141pp），且同一前端在 ECAPA 上
  也正向（−5.8%/−13.4%）→ 注册条件化 TSE = 迄今最强单杠杆，模型无关性双骨干实证。
- 但距 −30%（5.874/7.739）仍差 1.51/1.58pp —— 线①计划（训练→全量增强→官方评测）全部跑完，结案。
- 备选下一步（未执行，等指示）：TSE-FE 分数入融合（旧融合最佳 7.2049/9.1533）；TSE 训练加 epoch/加
  SNR 档；前端两版（8.18+TSE）分数级融合。

### 线④ MoE ep3 判决（13:26）→ 止损停训，全部计划收官
| ep | clean | 0dB | -5dB | avg01 |
|---|---|---|---|---|
| 1 | 1.250 | 6.000 | 9.500 | 7.750 |
| 2 | 1.500 | 6.250 | 9.250 | 7.750 |
| 3 | 1.750 | 5.500 | 9.500 | 7.500 |
- ep3 avg01=7.500 未破 soup5 锚 7.375(差 0.125=一个小 val 量子)，按预声明规则止损停训(13:26, pkill)。
- 观察: 0db 5.500 为该线最佳(专家化对 0dB 有选择性收益)，clean/m5db 轻微代价；小 val(800 trials)
  分辨力不足以定论，官方评测未做(用户指示收官)。ckpt 全保留。
- **至此 8.26 三线(TSE/ECAPA/MoE)全部跑完，-30% 未达成，停止新实验转入总结。**
