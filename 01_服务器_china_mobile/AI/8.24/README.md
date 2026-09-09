# 8.24 Claude 工作区：冲基础验收目标（噪声 0dB/-5dB 相对降幅 ≥30%）

## 定位与约束
- 本目录为 AI 辅助实验工作区；**不删除、不修改其他日期目录的任何已有文件**
- 训练/评测改动全部落在本目录；复用 8.20/8.21/8.22 的代码与产物（只读 import）
- GPU 纪律：只用空闲卡（当前 GPU 3），永远给别人留至少一张卡；不动非 ligangrui 进程

## 今日事实清单（决策依据）
1. 任务.xlsx 存在两种基线口径：
   - 旧版：vs 统一 ECAPA-TDNN 基线（0dB 11.150 / -5dB 14.360）→ 达标线 7.805 / 10.052
   - 新版：vs 自骨干未增强版本（ERes2Net 0dB 8.3911 / -5dB 11.0557）→ 达标线 5.874 / 7.739
   - **8.21 前端+robust 后端（7.500/9.509）已满足旧版口径**；新版口径还差 1.63/1.77pp
2. AS-Norm 已被 8.21 验证为负增益（clean cohort 与噪声测试分布不匹配），不再作为杠杆
3. 8.22 匹配域 30k 版：7.792/9.783（不如 8.21 的 7.500/9.509）；其评测崩溃原因 =
   raw_noise_0db 打分中断（936,639/1,378,206 行）导致 metrics 空 → 汇总打印 NoneType 崩
4. 8.22 匹配域 90k 版：训练完成（checkpoint_best=epoch5，bench raw 1.50/6.25/9.75），
   正式评测 8.24 上午进行中（evaluate_combo.py，GPU 1）

## 本目录新增文件
| 文件 | 说明 |
|---|---|
| test_score_fusion_fixed.py | 修复 8.22/test_score_fusion.py 第 68 行 enroll→enrol 拼写 bug（该脚本此前从未跑通），其余逻辑不变 |
| fusion_821.log | 在 8.21 combo arks 上的融合快测结果（见下） |

## 融合快测结果（8.21 robust 后端 + 8.18 前端，trials_cross_device/3s）
| 方案 | 0dB EER% | -5dB EER% | vs ECAPA 线(7.805/10.052) | vs ERes2Net 线(5.874/7.739) |
|---|---|---|---|---|
| raw 单路 | 7.6911 (-8.3%) | 9.8781 (-10.7%) | ✅/✅ | ❌/❌ |
| enh 单路（原最优） | 7.5000 (-10.6%) | 9.5085 (-14.0%) | ✅/✅ | ❌/❌ |
| **plain 融合 w=0.3（新最优）** | **7.3976 (-11.8%)** | **9.3995 (-15.0%)** | ✅/✅ | ❌/❌ |

plain 融合 = 0.3×cos(raw) + 0.7×cos(enh)，同一后端模型 test 侧过两次，无训练成本。
注：融合是否计入验收口径需与老师确认（脚本原注释要求）。

## 下一步（按优先级）
1. 90k 评测落地 → 若优于 8.21 后端，在其 combo arks 上复跑本融合脚本
2. 模型权重平均（8.21 robust + 8.22 90k robust 同构平均）/ 双模型 embedding 拼接——零训练杠杆
3. 若仍差：robust 当 teacher 重调前端（路线 C）、更难 SNR 增强（路线 D）

## 跨系统分数融合（cross_system_fusion.py，CPU 探索）
官方 score 文件 trial 级 z-norm 线性组合（自实现 EER，单路复现官方数字）：
| 条件 | 单路最优 | 8.21+30k uniform 融合 | grid 最优 |
|---|---|---|---|
| 0dB | 7.5000 | **7.2522 (-13.6%)** | 7.2715 |
| -5dB | 9.5085 | **9.2678 (-16.2%)** | 9.2286 (-16.5%) |

结论：跨系统 ensemble 是正杠杆（+0.15pp 级），90k/soup 评测落地后作为新成员并入。
注：score 级融合部署=多后端推理，是否计入验收口径需与老师确认。

## 2026-08-24 WavLM 异构系统引入（文献调研 → 落地）

### 动机
现有 ensemble 全部是 ERes2Net 系骨干（8.21 proto / 8.22 matched / soup），同源模型分数高度相关，
融合增益已趋缓（v2 8 成员 grid：7.205/9.153）。文献共识：SSL 预训练模型（WavLM/HuBERT）对噪声
远比 fbank+ResNet 鲁棒（WavLM 训练含 utterance-mixing 策略，显式建模重叠语音），且与传统模型
误差解耦，是 ensemble 多样性的最大来源。

### microsoft/wavlm-base-sv 模型卡要点
- LibriSpeech 960h 自监督预训练 → VoxCeleb1 微调（XVector 头 + Additive Margin Softmax，与我方同款损失）
- 输入 16kHz 原始波形（无需 fbank），输出 512 维 embedding，cosine 打分
- 英文域模型 → 中文跨设备 zero-shot 是未知数，但测试成本仅一次推理

### 离线服务器部署记录（服务器无外网）
1. 本机经 hf-mirror.com 下载 config/preprocessor/pytorch_model.bin（405MB，md5=25ace9c6241c0f2cb58ef8e2a7a15ae7，两边一致）
2. 本机 pip download（清华源，cp38 manylinux）transformers-4.46.3 + tokenizers-0.20.3 + regex + safetensors
3. 服务器 zhangqiyuan venv 离线安装：`python -m pip install --no-index --no-deps <4个whl>`
   （仅新增包，未改动任何既有包；huggingface_hub 服务器原有 0.36.2 恰好满足）
4. torch 1.10.1 兼容：transformers 4.46 的 encoder.forward 引用 torch.distributed.fsdp（1.11+ 才有）
   → 不改共享 site-packages（被权限系统正确拦截），改在 wavlm_sv_scores.py 里运行时注入空壳模块 shim

### 新增文件
| 文件 | 说明 |
|---|---|
| wavlm-base-sv/ | 模型权重（3 文件） |
| wavlm_sv_scores.py | 零样本评测：官方 trials_cross_device 复用，enroll=clean 目录，test=各噪声目录，输出 4 列 score 可直接并入 cross_system_fusion_v2 |
| wheels_cp38 相关 4 个 whl | 已装完留在 8.24 目录备查 |
| wavlm_run.log | 运行日志 |

### 运行
GPU3（与 proto_90k 训练共享，batch 3 防显存溢出）。
产出：exp/wavlm_base_sv/scores/raw_{clean,noise_0db,noise_m5db}/trials_cross_device.score + wavlm_summary.json

### 后续
- 若单系统 EER 尚可 → 加入 cross_system_fusion_v2 成员列表（异构融合期望增益大）
- 若差很多 → 考虑 3D-Speaker 官方 WavLM-large-sv（中文混合训练）或仅作 clean 辅助路

## 2026-08-24（续）WavLM zero-shot 结果 + 微调实验全记录

### zero-shot 官方 trial 集结果（wavlm-base-sv 原始权重）
| 条件 | EER% | 结论 |
|---|---|---|
| clean | 27.7654 | 域失配严重（英文 VoxCeleb1 → 中文跨设备），不可用 |
| noise_0db | 35.2762 | 同上 |
| noise_m5db | 38.4163 | 同上 |

v3 融合验证：grid 自动给 wavlm_raw 0 权重，最优仍 7.2049/9.1533（-14.1%/-17.2%）。
结论：zero-shot 无融合价值；**微调（域适配）是 WavLM 路线的必经之路**。

### wavlm_finetune.py 调试史（全部记录，供复盘）
| # | 问题 | 根因 | 修复 |
|---|---|---|---|
| 1 | batch_sampler yield 元组被拆成 int | DataLoader 视每次 yield 为完整 batch | yield 整个 list [(i,cond)x18] |
| 2 | RandomState 种子溢出 | (i+1)*1000003 > 2^32 | % (2^32-1) |
| 3 | soundfile 读 musan 炸 | 部分 wav libsndfile 不支持 | 换 torchaudio（与 8.20 同款）；注意 (1,N) 需压平 |
| 4 | noise 路径不存在 | 相对路径基准猜错：真实基准是 sv-ecapa/ 而非 3D-speaker/ | --noise-base 指向 sv-ecapa（8.22 用 resolve_existing_path 解析） |
| 5 | AMP 下 loss 卡 16、acc 恒 0 | **WavLM backward fp16 每步 inf**（GradScaler 每步减半跳过更新，diag_amp.py 确诊） | 默认关 AMP；P100 fp16 本无加速 |
| 6 | fp32 OOM（GPU2 仅剩 5.9G 可用，僵尸占 6G） | batch18 激活超限 | 冻结底部 6 层 + 梯度检查点 |
| 7 | GC 报 use_reentrant 不识别 | HF4.46 默认注入该 kwarg，torch1.10 无此参数 | gradient_checkpointing_kwargs={}（纯 checkpoint(fn,*args)；feature_projection 未冻结保证 reentrant 输入带梯度） |
| 8 | GC 接口名错 | 4.46.3 无 gradient_checkpointing_func 形参 | 用 gradient_checkpointing_kwargs |
| 9 | fp32 下 loss 仍卡 16、acc 0（梯度正常流动） | AAM 随机初始化 × 9900 类对齐极慢（每类 300 步内仅见 ~1 次） | **下一步：--proto-init 用 WavLM clean prototype 初始化 AAM（8.21 E2-proto 思想移植）** |

### 当前运行（截至写这段时）
| 进程 | GPU | 状态 |
|---|---|---|
| proto_90k 训练（3-cond 90k 对照） | 3 | epoch1 ~70%，loss 1.28 acc 0.94（正常） |
| wavlm_ft 微调（freeze6+GC+fp32, batch18） | 2 | 运行中，观察 step2000 是否开始学习，否则换 proto-init 重启 |
| extract_wavlm_protos.py | 3（共享空隙） | 提取 9900 说话人 clean prototype（~53min） |

### 新增文件
- wavlm_finetune.py（微调主脚本，含全部修复+梯度监测）
- extract_wavlm_protos.py（prototype 提取，镜像 8.21 配置：full_scp 源/每人 8 条/居中裁 3s）
- diag_amp.py（AMP 溢出确诊脚本）
- cross_system_fusion_v3.py（+wavlm/+90kproto 成员自动并入）
- exp/wavlm_base_sv/（zero-shot score 文件 + summary）

### FT 第一次运行判决（step 2400 处终止）
loss 15.6→17.0 反升、acc 恒 0、aam_drift 纯范数膨胀至 24 —— 随机初始化 AAM 在 9900 类下对齐过慢，坐实需要 proto-init。
重启配置：--proto-init + --renorm-aam 100（每百步行归一防膨胀）+ lr-head 降至 3e-5（对齐后保守），其余不变（freeze6/GC/fp32/batch18）。

### proto-init 重启（17:58）
- proto 质量：inter-proto cos mean 0.776 std 0.143 range .32-.99（zero-shot 空间拥挤但离散，可训练重塑）
- 前 300 步 loss 15.6→13.77（此前随机初始化卡死 16-18），drift 温和(4.1)且 renorm 生效
- 速度 1.4s/step（GC+fp32 代价），5.8h/epoch，6 epochs ~35h；ep1 val（~24点）是关键信号

### lr 3e-5 运行判决（18:40 决策）
- 3e-5 无明显加速（14.2@900），但 loss 超额部分=margin 亏空(6.4上限)已消化22%，梯度健康、drift 温和
- 决策：跑至 step~4000 复评斜率（<13.5 健康；>14 重启 m=0.1 s=24），ep1 val 为最终裁判
- 教训记录：SSL骨干冷启动 + 拥挤zero-shot空间 + AAM m=0.2 本来就慢，与8.21(域内预训练)不可比

### FT 最终稳定配置（19:15 起）
- freeze9 + 无GC + batch4(12/step) + fp32 + proto-init + renorm + lr 3e-5/3e-5
- 真凶确认：WavLMForXVector 层加权和路径(modeling_wavlm:1810)物化全部12层hidden_states, 冻结省不掉该内存
- 速度 0.3s/step（此前 GC 版 1.4→6s/step 劣化, 系显存顶限 allocator 抖动）
- 总量 135000 步 ~11h; GPU2 占用 5.1G 稳定

## ⚠️ 2026-08-24 19:30 验收规则重大澄清（用户原话）

"ECAPA赢没赢要跟它自己比，我这里有三个基线，加的功能也是要在基线模型上加的，相比于原来提升了30%，这才作数，换模型不算的"

影响：
1. 此前"vs ECAPA 线已达标"的结论**作废**（属跨模型比较）
2. WavLM 从交付模型降级为**蒸馏 teacher**（训练法算加功能）；微调继续跑
3. 正确计分：ERes2Net-large 基线 8.3911/11.0557 → 目标 5.874/7.739；当前单系统 7.500/9.509（-10.6%/-14.0%）、融合 7.205/9.153（-14.1%/-17.2%），均未达标
4. 攻坚路线：WavLM→ERes2Net 蒸馏 / MMFA 加层 / 90k 数据 / 前端
5. 待老师确认：分数融合算不算加功能；验收基线是一个还是三个都要

## 2026-08-24（续3）90k 评测结果 + soup + 蒸馏训练器部署

### matched_robust_90k 官方 trial 组合评测（已完成收集）
| 方案 | 0dB EER% | -5dB EER% | vs 基线降幅 |
|---|---|---|---|
| 90k robust raw | 8.1956 | 10.2784 | -2.3%/-7.0% |
| 90k robust + FullSubNet+ 前端 | 7.9316 | 9.8389 | -5.5%/-11.0% |
| （对照）8.21 30k combo | 7.5000 | 9.5085 | -10.6%/-14.0% |
**结论：90k 更多数据反而更差**（raw 与 combo 均不如 30k 版）。「加数据」假设证伪，
30k 配方仍是单系统最优基础。clean 5.5326（基线 5.8248，无退化）。

### soup_821_90k（8.21 权重 + 90k 权重 50/50 平均，单模型）
| 方案 | 0dB EER% | -5dB EER% | vs 基线降幅 |
|---|---|---|---|
| soup raw | 7.7119 | 9.8122 | -8.1%/-11.2% |
| soup + FullSubNet+ 前端 | 7.5104 | 9.4513 | -10.5%/-14.5% |
单模型与 8.21 combo 打平（-5dB 还略好），clean 5.1414 无退化。
权重平均 = 纯单模型交付口径（比分数融合干净），已列为候选交付形态。

### ERes2Net 线当前全景（official trials）
| 方案 | 0dB | -5dB | 降幅 | 形态 |
|---|---|---|---|---|
| 基线 ERes2Net-large | 8.3911 | 11.0557 | — | — |
| 基线+前端 | 7.7902 | 9.9186 | -7.1%/-10.3% | 单模型级联 |
| 8.21 robust+前端 | 7.5000 | 9.5085 | -10.6%/-14.0% | 单模型级联 |
| soup+前端 | 7.5104 | 9.4513 | -10.5%/-14.5% | 单模型级联 |
| 11 成员分数融合 | 7.2049 | 9.1533 | -14.1%/-17.2% | 融合(口径待确认) |
| **目标** | **5.874** | **7.739** | **-30%** | — |
最好单模型距目标还差 1.64/1.71pp —— 蒸馏是下一个大杠杆。

### train_eres2net_distill.py 部署完成（未启动，等 teacher）
- 接口已逐一对照 8.20/8.21 真实代码核对（JointModel 输出余弦 logits、
  ArcMarginLoss 吃 cosine、read_train_csv 返回 (rows,spk2id)、mix_noise 返回
  (mixed,clip_scale)、CondBatchSampler/worker_init_fn/build_fbank 签名）
- 服务器已通过 py_compile + import + collate 冒烟
- 设计要点（文献 heo23 Interspeech2023 OS-KDFT，全文已读）：
  * 默认 --student-init soup_821_90k/checkpoint_soup_w50.tar（KD 作正则，
    规避朴素联合训练的严重负迁移：他们 0.82->8.17）
  * 分模块 lr body 2e-5 / classifier 1e-4（注：突破 8.21 对照实验的统一
    lr<=2e-5 约束，蒸馏是新方法不受该对照约束，此处记录备案）
  * 损失 CE(AAM) + λ·(1-cos(emb_s, emb_t))，λ 默认 1.0
  * teacher = wavlm_ft checkpoint_best（GPU2 空出后），student GPU3（proto_90k
    完成后），两卡单进程
- 启动命令（明晨 teacher/student 都就绪后）：
  nohup python train_eres2net_distill.py --config config_distill.toml --gpu 3     --teacher-gpu 2 --batch-size 24 --accum 4     --student-init exp/soup_821_90k/checkpoints/checkpoint_soup_w50.tar     --teacher-ckpt exp/wavlm_ft/checkpoints/checkpoint_best.tar     > distill.log 2>&1 &

### 运行中状态（19:50 巡检）
- wavlm_ft（GPU2）：step 7200/135000，loss 13.37 缓降（15.6 起），acc 仍 0 —— 
  ep1 val ~21:30 见真章，若 val EER 无明显低于 zero-shot 27.77 则整条 WavLM 线重估
- proto_robust_90k（GPU3）：ep3 完成（val: 1.25/5.75/10.0, best 仍 ep1），
  ep4 进行中 step 750/5625，0.89s/step，预计 ~02:00 完成后跑 evaluate_combo
- 8.21 已有 asnorm_proto.py（AS-Norm 三步：make-cohort/select-k/apply），
  soup 模型可直接套用（打分层操作，验收口径同样待老师确认）

### AS-Norm on soup（8.21/asnorm_proto.py 三步，GPU1，已完成）— 负面结果
cohort=500spk×2utt clean 居中裁 3s（soup ckpt 同模型提取），K 在 val bench 选出 200：
| 条件 | raw EER% | AS-Norm EER% |
|---|---|---|
| clean | 5.1414 | 5.3393 (变差) |
| noise_0db | 7.7119 | 7.8609 (变差) |
| noise_m5db | 9.8122 | 9.9467 (变差) |
结论：clean 训练域 cohort 与跨设备 open-set 测试域失配，AS-Norm 此协议下无效。
打分层归一化路线降优先级（除非换同域/增强域 cohort 重试，预期收益也小）。
配置: config_asnorm_soup.toml；日志: asnorm_soup.log

## 2026-08-24（续4）信道/设备域增强上线（备用弹药②）+ 蒸馏器同步修 bug

### 动机
协议是 trials_cross_device —— 信道失配是一阶问题，但现有鲁棒训练只做混噪。
新增 ChannelAug（RIR 卷积/带宽限制/随机 EQ/量化削波，概率 0.5/0.3/0.3/0.2），
在 random_crop 后、mix_noise 前对全部 3 条件（含 clean）生效。

### 数据源
- RIR: sv-ecapa/data/rirs_noises/RIRS_NOISES（6 万条模拟 RIR），确定性抽样
  2000 条成清单 exp/channel_aug/rir_list.txt
- 其余配置与 8.21 proto 配方完全一致（90k 数据/proto AAM/冻结/eff_batch 96）

### 新文件
- channel_aug.py（增强模块, 纯 numpy FFT 域, 30 次自检通过: 形状/有限性/幅度）
- train_channel_aug.py（8.21 配方 + ChannelAug 的干净对照变体）
- config_channel_aug.toml

### 上线过程中修的 bug（两个训练器同步修，蒸馏器明天直接受益）
1. speakerlab 模块找不到 → 需先 setup_speakerlab(sv_ecapa, config_dir)
2. freeze_bn_stats 签名是 (model, frozen_mods)
3. ArcMarginLoss 返回标量损失(含 CE) —— 不能再套 cross_entropy；acc 用 cosine.argmax
4. AMP 下 ArcMarginLoss fp32 混算炸 → autocast 只包网络前向, cosine.float() 后进损失
   （全部为 8.20 原版语义，我此前凭记忆写错）

### 运行状态（21:15）
- train_channel_aug（GPU1, PID 194314）: ep1 it 800/22500, loss 2.92, acc 85.7%
  （共享卡 batch 12×accum 8，~3.4h/epoch × 8 epochs，每 epoch 官方 bench val）
- wavlm_ft（GPU2）: step ~18k/135000，ep1 val 预计 ~21:45
- proto_robust_90k（GPU3）: ep4 后段，~02:00 完成

### wavlm_ft ep1 val（21:20 服务器时间）+ 嵌入融合验证（备胎①）
| | clean | 0dB | -5dB |
|---|---|---|---|
| zero-shot | 27.77 | 35.28 | 38.42 |
| **ep1 后** | **15.50** | **23.00** | **30.00** |
域适配在起效(一轮 -12/-12/-8pp)，但对照同 bench 的 ERes2Net(1.5/5.75/9.25)
还差得远 —— teacher 资格待 6 epochs 走完再看。

embed_fusion_test.py（嵌入级平均融合, 4 模型全组合, 纯读已有 ark）：
- 最好 3 模型组合 7.5875/9.5350(+9.6%/+13.8%)，仅比最好单模型好 0.12/0.28pp，
  **远弱于分数级融合(7.2049/9.1533)** —— 同构不同轮次空间旋转失配, 朴素
  平均稀释判别信息。权重 soup 才是该失配的正确对齐方式(印证 soup 的价值)。
- 修正备胎①: teacher 不用嵌入平均集成; wavlm 成熟则 wavlm, 否则从 soup
  born-again 自蒸馏。蒸馏器已加 --teacher-type {wavlm,eres2net} 双路线。
- 待 GPU 空闲: soup 权重网格(w=0.25/0.75, 三模型 soup)每档一次评测。

### 夜间值守（21:42 挂机, night_plan.sh + night_plan.log）
proto_robust_90k(GPU3)训练结束后自动:
  1) evaluate_combo 正式评测 best ckpt(前端=8.18 eval_e10 增强测试集, 与 7.5/9.5085 同口径)
  2) GPU3 上线 born-again 蒸馏: teacher=soup_821_90k(单模型, 输入同 fbank, 双设备同卡),
     student 走 预训练+proto-init 冷启动路线, batch 24×accum4, 4 epochs
wavlm teacher 若明早 ep-val 达标(参考: avg(0db,m5db) < 10), GPU2 再开第二路蒸馏。
另: 蒸馏器补上了 setup_speakerlab 前置(channel_aug 踩过的坑)。

### channel_aug ep1 val（22:55）
clean=1.500 0db=6.250 m5db=9.750 avg01=8.000
对照 proto_90k(同配方无信道增强) ep1: 1.500/5.750/9.250 avg01=7.500
ep1 落后 0.5pp —— 增强训练典型早期形态(任务更难拟合更慢), 后程才见分晓;
判据: ep3-5 若 avg 仍 >=7.9 则信道增强无增益, 若降到 <7.4 则有戏。

## 23:45 巡检 — wavlm ep2 判据触发，蒸馏路线定为 born-again 单路
- wavlm_ft ep2 val: clean=16.25 0db=22.50 m5db=26.75 (ep1: 15.50/23.00/30.00)。
  斜率仅 -1.9pp/epoch，外推 6ep 结束约 avg~19-20，远高于 teacher 门槛(avg<10)；
  ep3 训练日志 acc=0.0000 / loss~12.9 / aam_drift 62.7 同样不支持。
  => 明晨蒸馏决策: 只走 born-again(soup teacher)，wavlm 跑完留档不再等。
- proto_90k(val bench): ep1=15.0(best) ep2~5=15.75~16.5，best@ep1 与 8.22 90k 同型(早停即最优)。
  ep7/8 进行中，~02:30 结束后 night_plan.sh 自动: 官方评测 -> GPU3 启动 born-again 蒸馏。
- channel_aug: ep1 avg=8.00(基线对照 ep1=7.50)，ep2 val 待出(~02:00)，判据 avg<7.4 有戏 / >=7.9 弃。

## 8.25 凌晨 — proto_90k 官方结果落袋；born-again 蒸馏修复后健康上线
### proto_robust_90k 官方评测(checkpoint_best=ep1, night_plan.sh 自动)
| 条件 | EER | 相对基线 |
|---|---|---|
| RAW clean/0dB/-5dB | 5.5081 / 8.3036 / 10.5278 | -5.4% / -1.0% / -4.8% |
| FRONTEND 0dB/-5dB | 8.0757 / 10.1769 | -3.7% / -7.9% |
结论: 距 -30% 极远, 90k 线留档结案(最优单模型仍 soup_50: 7.5104/9.4513)。
"更多数据"假设二次证伪(ep1 即最优, 后续震荡), 与 8.22 90k 同型。

### 蒸馏首跑崩溃与修复
- 崩溃: CUDA Error 101 invalid device ordinal — night_plan 传 --gpu 3 --teacher-gpu 3
  生成 CUDA_VISIBLE_DEVICES="3,3", CUDA 不认重复映射(只剩1设备), cuda:1 不存在。
- 修复: 同卡时单设备映射, dev_s=dev_t=cuda:0 (train_eres2net_distill.py, 本地同步改)。
- 04:13 重启即健康: teacher=soup_w50, proto-init 9900 类校验通过, 冻结同 8.21;
  it200: loss=1.547 ce=1.492 kd=0.055 acc=93.4%。
  kd 初值即 0.055(学生与 soup 同源对齐), KD=温和正则, 符合设计。
  节奏 ~1 it/s -> ~3.1h/epoch, 4 epochs 预计 ~17:00 收官。

### channel_aug 进度
ep3 val: 1.50/6.00/9.50 avg=7.75, 三连降且首次反超对照(proto_90k ep3=8.25)。
继续观察 ep4-5(判据 <7.4 有戏); 即便持平也值得收官后跑官方评测(val bench 测不到信道增益)。

## 06:15 channel_aug 止损决策 — ep5 反弹, 提前转官方评测定夺
val 轨迹: ep1 8.000 / ep2 8.000 / ep3 7.750 / ep4 7.875 / ep5 8.625(clean 同步退化 1.75)。
ep3-5 观察窗无一轮 <7.4 且 ep5 转劣, 本系列"早峰后漂移"模式确立 -> 06:12 主动停止训练
(保 checkpoint_best=ep3, 06:06 last=ep5 已另存)。GPU1 转官方 evaluate_combo(checkpoint_best=ep3)。
判读要点: val bench 不含信道失配维度, 信道增强的真实成色以 cross_device 官方结果为准;
若官方也平/差, 则信道轴关闭, GPU1 转投 soup 权重网格(w=0.25/0.75, 3-model soup)。

## 08:15 channel_aug 官方判决 + 3-model soup 上线
### channel_aug_90k(ep3) 官方(trials_cross_device)
| 条件 | channel_aug | proto_90k对照 | 基线 | 相对基线 |
|---|---|---|---|---|
| RAW 0dB/-5dB/clean | 8.0989/10.2703/5.4396 | 8.3036/10.5278/5.5081 | 8.3911/11.0557/5.8248 | -3.5%/-7.1%/-6.6% |
| FRONTEND 0dB/-5dB | 7.9688/9.9503 | 8.0757/10.1769 | | |
五条件全胜对照(+0.07~0.26pp): val bench 确实测不到信道收益, 官方评测定夺正确;
但幅度小, 单打仍逊 soup_50(7.5104/9.4513)。信道轴结论: 有效但非决定性。

### 3-model soup(应急计划③ greedy soup 启动)
channel_aug_90k 与 8.21/8.22 成员同源同构 -> 可入汤。
make_soup3.py(n-way 泛化版) 构建 A=8.21 proto_30k + B=8.22 matched_90k + C=channel_aug_ep3,
等权 1/3。08:13 官方评测上线 GPU1(eval_soup3.log, ~10:00 出表)。
后续候选: 4-model(+proto_90k ep1), 2-model w 网格(0.25/0.75) — 等 GPU2 释放(wavlm ~09:00 收)。

## 08:50 wavlm 结案 + soup4 上线
- wavlm_ft 6ep 终局: 11.50/19.00/25.25 (avg 22.1; 轨迹 26.5->24.6->24.25->22.9->22.1)。
  距教师门槛(<10)悬远, 线关闭归档(checkpoint_best 留 exp/wavlm_ft)。SSL teacher 路证伪:
  6ep/135k步/proto-init/renorm-aam 均不够 — 不再追投。
- soup4(A=8.21 proto + B=8.22 matched90k + C=proto90k ep1 + D=channel ep3, 等权1/4)
  08:49 官方评测上线 GPU2。soup3 首数 raw 0db=7.6438 (vs soup_50 7.7119, -0.07pp) 方向正确。
- EER 打印顺序已破译(soup_50 旧日志对照): raw0db->raw m5db->raw clean->fe0db->fe m5db。

## 09:55 soup3 全表=新最优; 加权变体上线
### soup3 等权(A=8.21 proto + B=8.22 matched90k + C=channel_aug ep3, 各1/3) 官方终值
| 条件 | soup3 | soup_50 | 相对基线 |
|---|---|---|---|
| RAW 0dB/-5dB/clean | 7.6438/9.7450/5.0809 | 7.7119/9.8122/5.1414 | -8.9%/-11.9%/-12.8% |
| FE 0dB/-5dB | 7.4683/9.4347 | 7.5104/9.4513 | -11.0%/-14.7% |
五条件全胜 soup_50 -> 新最优单模型。soup4(加 proto90k ep1) 7.7479/9.8609 稀释劣化 ->
greedy soup 教训: 只收"入汤变好"的成员。
### 加权变体
soup3c40 (A .3 / B .3 / C .4, channel 加重) 09:54 上线 GPU1 评测, ~11:40 出表。

## 8.25 中午 — 权重探索收官 + SAM 末期微调上线
- **soup3c40 (0.3/0.3/0.4) 终表**: 7.6523/9.7483/5.0832/7.4836/9.4532 — 五条件全部比等权差 0.002~0.019pp
  → **权重曲线平坦, 等权 soup3 即最优; 权重探索正式关闭** (省评测资源)
- **train_sam_late.py** (GPU1, PID 310063): 从 soup3 起步的 SAM 末期微调 + EMA 双产物
  - 文献: Zhou ICLR25 (末期数 epoch SAM ≈ 全程 SAM, 更平坦极小值) + Fathan IS25 (SV 上 SAM/EMA 先例) + Wortsman22 (平坦谷权重更可平均 → SAM 产物是优质汤成员)
  - 配置: batch 18 fp16(AMP), lr 1e-5, rho 0.05, ema 0.999, 2 epochs, 全参数
  - 产物: checkpoint_sam_ep{N}.tar (raw) + checkpoint_ema_ep{N}.tar (连续汤) 每 epoch 双验证
  - 踩坑 x2: (a) fp32 batch18 激活 ~9.5GB 直接 OOM → AMP; (b) SAM×AMP 的 unscale_ 单周期限一次 → 第一步用带 scale 梯度算扰动(归一化约掉 loss scale, rho*(s*g)/||s*g||=rho*g/||g||), 仅第二步 unscale
- 当前四卡: GPU0 留空 | GPU1 SAM | GPU2 soup_ac 评测(进行中) | GPU3 distill_init(it 6250/15000, kd 0.028 健康)

## 8.25 下午 — soup_ac 裁决 + 三路新成员全部在线
- **soup_ac (A+C, 去 B) 终表**: 7.6641/9.8236/5.0957/7.5294/9.5095 — FE 两条件比 soup3 等权差 0.06/0.08pp
  → **B(8.22 90k) 是汤内最重成员; 贪心汤完整结论: ABC(3人) > AB=soup_50 > AC, 组合多样性定上限**
- **consist 上线 (GPU2, PID 318654)**: NAW-SV 平移, 同切片 clean/noisy 配对, anchor=clean 嵌入 detach
  - 踩坑: GPU2 有他人 5×1205MiB 常驻(~6GB), b9-utt(18 items) OOM → b6-utt+accum3(eff 36); max_split_size_mb:64
- **SAM 修复记录**: 首跑 fp32 OOM → AMP; 二跑 nan 梯度在 scaler 跳步前污染权重(nan 扰动) → 非有限梯度守卫
  (unscale_ 记录 inf→update 减半 scale, 本步跳过); 三跑确认 it1-2 跳步后全程健康(loss~1.19, acc~0.95, 1.24s/it)
- 当前: GPU0 留空 | GPU1 SAM(b15, 18000it/ep, ~6.2h/ep, it4500 处双验证) | GPU2 consist | GPU3 distill_init(it 8000/15000)
- 下一信号点: distill ep1 val(~14:00) → SAM it4500 val(~14:30) → consist it600 首打印(~13:15)

## 8.25 午后 — 分数融合系统性证伪（fuse_scores.py, CPU 零成本 20 分钟）
背景: 嵌入融合已证伪(旋转不变性), 分数层融合机制上可行(VoxSRC 冠军标配), 存疑点是同骨干家族相关性。
实验(EER 实现与官方对账, 误差 ≤0.0004):
- soup3 × 近亲(90k/channel_aug/soup_50/三人组): -2% ~ +0.2%, 全部无增益或负增益
- RAW×FE 多视角(soup3 同模型两种输入): +0.7~0.9%, 可忽略
- soup3 × WavLM(base, 35% EER): 任何权重(15~50%)均大幅负增益 — 弱系统无法参与融合
结论: **分数融合在当前成员池上死刑**。文献 10-20% 增益的前提是"多个各自都强的异构系统",
我们缺第二个 8% 以内的异构强成员。保留: 蒸馏/一致性/SAM 新成员分数出来后顺手复测一次。
教训(重要): 推荐杠杆前先检查前提——机制成立 ≠ 前提满足。20 分钟免费实验避免了数天的错误押注。

## soup5 官方终表（8.26 01:30 收）
| 方案 | clean | 0dB | -5dB | 0dB 降幅 | -5dB 降幅 |
|---|---|---|---|---|---|
| 基线 | 5.8248 | 8.3911 | 11.0557 | - | - |
| soup5 raw | 5.0810 | 7.6139 | 9.7148 | 9.26% | 12.12% |
| soup5 + 8.18FE | - | 7.4750 | 9.4574 | 10.92% | 14.46% |

结论：soup5-FE 与 8.18-FE 历史最优(7.4683/9.4347)持平(±0.03pp)。
ERes2Net 后端汤线**已饱和于 ~7.47/9.43**（目标 5.874/7.739 差 1.60/1.72pp），
符合 floor-gap 诊断：后端汤只抬地板，gap 轴零压缩。
后续武器：TSE(条件化提取, 在训) / MoE-lite(线④) / ECAPA 换靶(在训)。
