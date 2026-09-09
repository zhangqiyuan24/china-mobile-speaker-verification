# 8.25 — 前端二轮（VoiceID 式）+ 三线中期信号

## 文件
- train_spk_voiceid.py / .toml — VoiceID 式 FullSubNet+ 二轮微调
  （文献: Shon et al. Interspeech 2019, VoiceID Loss）
- smoke_voiceid.log — CPU 冒烟记录（通过）

## 设计（相对 8.18 spk_preserve 的升级）
1. 损失 = L_se(cIRM MSE, 防退化锚) + 1.0 * AAM-CE(enhanced 嵌入, spk 标签)
   —— CE 类判别梯度经冻结 SV 模型回传到增强网（VoiceID 论文核心）
2. 冻结验证器 = soup3（当前最优, 8.18 用的是原始弱 ckpt）
3. 分类器 = 8.21 prototypes.npz 初始化（9900 类, 行范数=1, 逐位回读校验）
4. 数据 = 8.20 train_utts.csv（90k utts 自带绝对路径+标签, 免 scp 对齐）
   均匀抽样 30k；FullSubNet+ 从 8.18 e10 起步（叠加, 非从零）
5. 训练: b4, 6 ep, lr 1e-5, fp32（8.18 同款, 无 AMP）

## 冒烟记录（CPU, 8 样本 2 步）
- labels: 90000 utts / 9900 spks；proto-init 9900/9900 ✓
- 踩坑修复: ① sv_ecapa 路径 /data/home 且 egs/3dspeaker 小写
  ② 原设计"先抽样后对齐"浪费数据 → 改为 CSV 直读
- 待 GPU 释放后正式启动（--gpu <空卡>）

## 评测链（训完即用, 与 8.22 官方口径一致）
1. 降噪: cd 8.17/code && python -m speech_enhance.tools.inference    -C config/inference.toml -M <新ckpt> -I data_for_test/<cond>/3s -O <out>
   （两条件: noise_0db / noise_m5db, 各 16401 条）
2. 评测: 8.21/evaluate_combo.py --config config_soup_ac.toml    --checkpoint 8.24/exp/soup3_abc/checkpoints/checkpoint_soup3.tar    --enh-0db <dir1> --enh-m5db <dir2> --gpu <卡> --batch-size 32
   → scores/frontend_noise_{0db,m5db}/ + result.metrics

## 8.25 下午 — ep1 验证三线排名 + 杀线换防（地板/缝隙双线）

### 锚点(anchor_val.py, CPU, 与训练内验证同口径 800 trials)
- soup3(SAM/consist 的 init): clean=1.500 0db=5.750 m5db=9.750 avg01=7.750
- soup_w50(distill 的 init): clean=1.500 0db=6.250 m5db=10.000 avg01=8.125

### ep1 验证与处置
| 线 | init 锚点 | ep1 | Δ | 处置 |
|---|---|---|---|---|
| distill | 8.125 | 7.875 | -0.250 | **留守**(m5db 压 1pp, 才 ep1/4) |
| consist | 7.750 | 8.375 | +0.625 | **杀**(0db 恶化 1.25pp, clean 漂 +0.5, 预警全中) |
| SAM | 7.750 | it4500 EMA 7.625 → it9000 EMA 8.375 | 崩坏 | **杀**(越过 8+ 杀线; it4500-EMA ckpt 留档为产物) |

- SAM 负结果定性: 从已收敛汤出发继续 AAM-CE 本身就是漂移源, rho=0.05 的 SAM
  未能阻止 — "late-SAM≈full-SAM"(Zhou25) 的前提是长训练的末端, 不适用我们的场景。
- consist 负结果定性: 纯噪声对(无 clean 条件)+一致性损失, 0db 反被伤 —
  噪声不变量正则的配比/条件覆盖需要重新设计。

### 换防上线(15:10-16:00)
1. **voiceid 前端**(GPU2, b2): 修 device bug(proto 装载 CPU 上做完再搬卡) + b4 OOM 降 b2;
   ~0.6s/it, 15000 it/ep × 3 ep, 预计 ~23:00 完; ep1 ckpt ~18:15。
2. **channel_aug v2**(GPU1, b30 accum3): 地板线/汤成员 D。ChannelAugV2 =
   v1 概率/强度上调 + AM 泵动 + 50Hz 哼声 + 20% 双 RIR; 同 pretrained 初始化保连通性;
   修 v1 codec 泄 float64 的 dtype 出口。8 ep × 9000 it/ep(比 v1 的 22500 快, b30)。
   踩坑: b48 OOM → b32 违反采样器 3 倍数约束 → b30。

### 今晚日程
- ~18:15 voiceid ep1 ckpt → 可先跑官方 FE 评测(需先降噪, 见上方评测链)
- ~23:00 voiceid 完训; channel_aug_v2 ep1 val 对锚点 7.750(注: v2 init=pretrained,
  锚点应为 pretrained 的小验证 — 待补, 或直接看趋势)
- 明早: 汤候选 = soup3 + distill-best + voiceid-FE + (ca2-best 若好) → soup4 官方全表

## 8.25 深夜追加：评测链排障 + GateNet OOM 根因（重要沉淀）

### voiceid FE 评测链两次"假成功"的真因
1. `KeyError: epoch` — 8.17 推理工具 `_load_model` 要求 ckpt 含 epoch 键；train_spk_voiceid 只存了 {model}。
   已就地补键（torch.load → ck["epoch"]=3 → torch.save 回写）。
2. evaluate_combo 的 extract_guarded 拦截：common/enrol_clean ark 是 soup_ac ckpt 提的，与 soup3 不一致。
   已 mv 留档（enrol_clean.bak_soupac_0825）——守卫按设计工作。
- 21:12 重启 watch_voiceid.sh（WATCH_PID 397199），降噪真实推进（~6 it/s，单条件 ~45min）。

### GateNet 三连 OOM 根因（探针二分结论，通用教训）
现象: b30/b27/b18 全 OOM，allocated 恒 ~10.6-10.8GB 不随 batch 变。
探针(probe_gate_mem.py / probe2.py, GPU3):
  B0 无门(JointModel)      → 峰值 reserved 5.92GiB ✓
  B1 门冻结+输出 detach     → 5.93GiB ✓
  B2 完整图(梯度穿门)       → OOM >11.9GiB ✗
机理: fbank 带 requires_grad 进 encoder 后，ERes2Net 每个 BN 反传 input 梯度需
保存完整输入激活(1024/2048 通道 × 300 帧 × b18 ≈ +5-6GB)；输入是叶子时 BN 只存统计量。
ca2 无门前置模块所以从没触发。
解法: --amp(fp16 激活减半; P100 fp16 还有 2x 吞吐) + b18 accum5(eff 90 不变)。
顺带: GateNet ReLU(inplace=True)→ReLU()（BN+inplace 在图内是隐患）。
另修: freeze_report(encoder, classifier) 参数顺序（classifier 先建再传）。
状态: 21:2x 起 GPU3 99% 运行（GATE_PID 401756），15000 micro-batch/ep × 8ep。

### 评测链第三坑（22:2x 补）：FE 缓存不记前端来源
meta.json 只记 ckpt，不记增强 wav 来自哪个前端 → 同 ckpt 换前端时守卫放行、
直接复用旧 FE 嵌入（100 秒出旧数字，静默错误）。
规则：**换前端评测必先 mv combo/frontend combo/frontend.bak_<tag> 且 mv scores/frontend_***。
合法可复用：common/enrol_clean + raw/*（原始波形 + 同 ckpt）。
另修：新建 config_soup3_eval.toml（exp_dir=exp/soup3_abc，缓存按模型隔离，
避免再撞 soup_ac 时期缓存）。
