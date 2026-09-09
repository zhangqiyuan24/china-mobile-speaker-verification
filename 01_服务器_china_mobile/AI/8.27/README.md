# 8.27 — 阶段二：TSE-SV 联合优化（起点 8.26 深夜，按用户阶段指令执行）

## 阶段执行顺序（严格串行，不得并行开多线）
Phase 0 TSE 入融合 → Phase 1 TSE+speaker preservation → Phase 2 +nontarget suppression
→ Plan B cross-device teacher/student → Plan C RAW+TSE dual-view / 显式噪声建模

## 铁律
1. 融合权重只在 proxy validation 上搜；官方测试只允许冻结权重跑一次
2. 每步先 sanity/小规模筛出趋势，再跑官方全量
3. 不覆盖 8.26 及以前任何结果；新代码全在本目录
4. 止损标准（Phase 1/2 vs TSE 基线 7.3818/9.3160）：改善 <0.1~0.2pp = 架构饱和，
   不许靠堆 epoch 硬磨；若进入 0dB≤6.8 / -5dB≤8.6 = 明显潜力，重点推进并做融合

## Phase 0 设计（零训练成本）
- proxy-official 验证集（build_proxy_val.py）：
  100 个未见说话人（池 10000 − 已用 9900）× enrol=Device03 clean 3s ×2 条
  × test=Device06/07 各 2 条 × SNR{0,-5}，噪声 = noise_test_freesound 168 条
  （held-out，与训练 615 条不相交），trial 比例 1:5.5 对齐官方，3 列官方格式
- 双前端增强（enhance_proxy.py）：818 = FullSubNet+(e10, identity-FiLM 等价装载)；
  tse = 8.26 ep3（条件 = proxy enrol 2 条过冻结 soup5 均值+L2）
- 全成员打分（score_proxy.py）：v3 十成员（wavlm 弃用——v3 grid 给 0 权重，README
  存照）+ tse_soup5 = 11 成员；6 后端（821/30k/90k/soup_w50/p90k/soup5）× 3 变体
- 融合（fusion_phase0.py）：proxy z-norm + 0.25 grid → w* → 冻结 → 官方一次应用；
  旧官方权重 w_old=0.25×(821_raw+821_fe+30k_fe+90k_fe) 双侧对照

## 文件
| 文件 | 作用 |
|---|---|
| build_proxy_val.py | 构造 proxy 验证集 → exp/proxy/ |
| enhance_proxy.py | proxy 双前端增强 → exp/proxy/enh818_* enhtse_* |
| score_proxy.py | 11 成员 proxy 打分 → exp/proxy/scores/*.score |
| fusion_phase0.py | proxy 搜权 → 官方一次应用 → exp/phase0/result.json |
| proxy_official_consistency.py | 10 成员双侧 EER 秩一致性检验（--suffix hard 切 hard 口径）|
| mine_hard_trials.py | 困难负样本挖掘（soup5 嵌入 top-K 跨说话人）→ trials_hard_* |
| score_proxy_hard.py | 11 成员 hard trials 重打分（嵌入按后端缓存 emb_cache/）|
| train_tse_v2.py | Phase 1 训练器：续训 TSE ep3 + λ_spk·(1−cos(e_enh,e_clean)) |
| train_tse_v3.py | Phase 2 训练器：+λ_ns·top-K 原型难负 hinge |
| train_ts.py | Plan B 训练器：clean-teacher/noisy-student（四损失+不确定性加权）|

## Phase 1（进行中）
- 三臂 λ_spk=0.1/0.3/0.5（GPU1/3/2），TSE ep3 续训 3ep，选点=旧 proxy avg
- ep1 val（旧 proxy）：lam01 3.250 / lam03 3.131 / lam05 3.250 —— 均未破基线
  3.131，0dB 三臂+基线全部钉死 2.000（地板饱和）
- ep2 val（旧 proxy）：lam01 **1.761**/4.261 avg 3.011、lam05 **1.761**/4.284
  avg 3.023（两臂 0dB 首次脱离 2.000 地板）；lam03 2.000/4.500 反退
- **hard proxy 离线评测**（eval_arms_hard，锚点 tse_soup5 4.716/11.193 avg 7.955）：

  | ckpt | 0dB | -5dB | avg |
  |---|---|---|---|
  | lam01_ep1/2 | 4.500 / 4.477 | 11.239 / 11.239 | 7.869 / 7.858 |
  | lam03_ep1/2 | 4.500 / 4.545 | 11.500 / 11.000 | 8.000 / 7.773 |
  | lam05_ep1/2 | 4.523 / **4.500** | 11.261 / **10.784** | 7.892 / **7.642** |

  读法：① ep2 avg 对 λ 单调（0.5→7.642 < 0.3→7.773 < 0.1→7.858）剂量响应
  干净；② lam05 两口径同向（hard m5dB −0.409pp + 旧 0dB 脱地板）——真信号；
  ③ 量级换算：hard 展开约为官方 2 倍 → −0.409pp hard ≈ −0.2pp 官方，恰在
  止损模糊带上沿；④ lam03 的 best（旧 proxy 选了 ep1）与 hard（ep2 更优）
  不一致——最终官方候选一律以 hard 口径选
- 判决：按预声明 3ep 协议跑完 ep3（已在跑，零额外成本）→ ep3 hard 复评：
  lam05 趋势延续（avg ≤~7.5）= 明显趋势 → 官方单次确认；平了 → 止损转 Phase 2
- **ep3 终判（hard proxy 全表）**：三臂 ep3 全破锚——lam01 7.750 / lam03 **7.403**
  / lam05 7.750；最优 = lam03_ep3（4.307/10.500，ep2→ep3 再降 0.37pp 加速下行，
  无饱和迹象）→ 触发预声明"官方单次"规则
- 旧/hard 分歧记录：lam03 旧口径 ep3=3.250（其最差）而 hard=7.403（其最佳）——
  旧随机负对 EER 由易负对主导，官方 1.17M 负对的 EER 工作点由最难尾部主导，
  hard 口径与官方结构同构，选点以 hard 为准（README 早前已声明）

## Phase 1 官方单次评测（lam03_ep3，8.26 夜启动）
- 触发：预声明规则（最优臂 ep3 < 7.5 且加速下行）
- 链：enhance_tse(lam03_ep3) 增强官方带噪 16401×2 → evaluate_combo FE 模式
  （soup5 ckpt；raw/enrol 缓存自 8.24/exp/soup5 拷入同 ckpt 复用，frontend 全新）
- 配置 eval_lam03ep3.toml；输出 exp/official_lam03ep3/；日志 official_lam03ep3.log
- 铁律：官方一次应用，不调任何权重

## Phase 1 结案（官方判决 8.26 收）
- **lam03_ep3 官方（trials_cross_device）: 0dB 7.2933 / -5dB 9.2020**
  = 相对基线 −13.1%/−16.8%（历史最优单系统），但 vs TSE 基线仅 −0.089/−0.114pp
  → 落在饱和带（<0.1~0.2pp），按规则不堆 epoch、不追加 λ 微调
- 亮点：干净口径下首次超过融合系统（7.2933 < Phase0 w* 7.3600）
- 仪器校准：hard proxy 方向全对（选臂/双档同向）但幅度高估 ~5×
  （预测 −0.27pp 实测 −0.089pp）——后续 hard 移动 <1pp 不动官方
- 官方产物：exp/official_lam03ep3/（frontend 全新，raw/enrol 复用 8.24 soup5
  同 ckpt 缓存）；日志 official_lam03ep3.log；配置 eval_lam03ep3.toml

## Phase 2（8.26 深夜启动）：TSE + nontarget suppression
- 动机：L_spk 只抬 target 相似度，margin 另一半（压 nontarget）未动——
  Phase 1 的结构短板
- L_ns = mean_k max(0, cos(e_enh, proto_k) − 0.30)，k = 冻结 9900 原型中除真
  说话人外最难 top-8（原型 = classifier.weight 行, L2 归一；梯度只穿 e_enh）
- 基座 lam03_ep3，λ_spk=0.3 保持，唯一新变量 λ_ns ∈ {0.2, 0.5}（GPU1/3）
- 选点 = 每 epoch hard proxy avg（锚 7.403）；3ep；冒烟已过（tse_v3_smoke.log）
- train_tse_v3.py / 日志 tse_v3_ns{02,05}.log / exp/tse_v3_ns{02,05}
- **ep1 val-hard**：ns02 7.631 / ns05 7.642 —— 双臂均高于锚（-5dB 掉 2 量子，
  新损失扰动）
- **ep2 val-hard（趋势判决）**：ns02 7.381（=锚, 亚量子）/ ns05 7.597（仍高于
  锚）——无任何超越锚的迹象，λ 剂量无分化。对照 Phase 1（L_spk ep2 有 0.37pp
  真实下探）→ **L_ns 判无效信号**：top-K 原型难负 hinge 不产生 trial 级收益
  （原型近邻 ≠ trial 混淆说话人）。处置：ep3 跑完留档（除非 <7.0 否则不动
  官方）；Phase 2 结案，转 Plan B（设计并行启动）
- **ep3 终判（判决反转）**：ns02 **7.023**（4.261/9.784）/ ns05 **6.881**
  （4.045/9.716）——双臂 ep3 大幅破锚（−0.38/−0.52pp）且剂量响应显形
  （ns05>ns02 两轴一致）、ep2→ep3 加速无饱和、双轴同向（-5dB 主轴 −0.78=3
  量子）。与 Phase 1 lam03 形态完全同构（ep3 兑现官方最优）。**ns05 6.881 <
  7.0 → 预声明规则触发官方单次**。ep2 的"无效信号"判过早——3ep 协议价值
  再次验证；教训：此架构系前 2ep 组织嵌入空间、ep3 才出收益，趋势判决不早于 ep3

## 选点仪器检验（用户 8.26 提出"验证分布须贴近测试端"）
- 旧 proxy vs 官方（10 成员）：系统性乐观 +5.6/+4.7pp；0dB rho=0.939 但那是
  地板压缩下的粗排序（fe<raw），top-1 错配（821_fe vs tse_soup5）；-5dB rho=0.721
  top-1 一致。根因：nontarget 随机跨说话人对远易于官方负对分布
- hard 挖掘（音频零改动，soup5 嵌入挖跨说话人 top-6 难负对，1:5.5 配额）：
  - 量级对齐 ✓：0dB 均值 2.245→5.312（官方 7.809），偏差 +5.56→+2.50pp；
    -5dB 偏差 +4.70→−2.25pp；0dB 离开地板，成员间展开 0.5→2.0pp
  - 新锚点：tse_soup5 hard = **4.716 / 11.193**
  - 但 0dB rho 降至 0.624、top-1 仍错配：主因 p90k_fe（官方 8.08 第 3 差，
    hard proxy 并列最好）——疑为 D06/07 通道敏感性，挖掘无法模拟（unseen
    说话人无 D06/07 录音）；-5dB rho=0.721 不变，top-1 一致（tse_soup5 双侧最优）
  - 已声明偏差：挖掘用 soup5 对 raw 带噪嵌入，而 tse_soup5 在增强音频上打分
- 仪器结论：proxy 可分辨 ≥2 量子（≈0.5pp）效应；止损边界（<0.2pp）低于其
  分辨率，但边界内判决=kill（保守），故足以支撑 ep2 终判。终判口径：
  **hard -5dB 为主，hard 0dB 为辅，旧 proxy 仅 sanity**

## Phase 2 官方单次评测（ns05_ep3，8.27 触发）
- 触发：预声明规则（ep3 <7.0）；ns05_ep3 = 6.881
- 链（official_ns05ep3.log，GPU3）：enhance_tse(ns05_ep3) 增强官方 16401×2 →
  evaluate_combo FE 模式（soup5 ckpt；raw/enrol 缓存自 8.24 soup5 拷入，
  frontend 全新）→ exp/official_ns05ep3/
- 配置 eval_ns05ep3.toml；铁律：官方一次应用，不调任何权重
- **官方判决：0dB 6.5191 / -5dB 8.1732** = 相对基线 **−22.3%/−26.1%**
  ——历史最优单系统（前最优 lam03_ep3 7.2933/9.2020，再降 −0.77/−1.03pp）；
  **两档首次同时逼近 -30% 线**（目标 5.874/7.739，差 0.65/0.43pp）
- raw sanity 三连 ✓（5.0810/7.6139/9.7148 与 8.24 soup5 逐位一致）
- **仪器校准修正**：hard×0.2 换算规则失效——Phase 1 高估 5×（−0.27→−0.089），
  本次反向低估 9×（预测 −0.1 → 实测 −0.90 avg）。结论：hard 方向可靠、
  幅度双向不稳；"hard<1pp 不动官方"作废，官方触发只认各线预声明规则，
  hard 仅作 <0.2pp 亚量子 kill 参考
- **续臂 ns10（λ_ns=1.0）启动**（GPU3, tse_v3_ns10.log）：剂量响应
  （ns02 7.023 → ns05 6.881 随 λ 递降）+ ep3 加速 → λ 轴未饱和；同基座同协议 3ep
- **ns05ext（ns05_ep3 续训 ep4-6, GPU1, tse_v3_ns05ext.log）**：官方加速曲线
  的延长线；v3 加 --base-ckpt 参数（默认不变）。锚 = ns05 hard 6.881；
  **预声明：best epoch hard < 6.64（≥1 量子）→ 触发官方单次**
- **ep4 判决（触发）**：hard **6.239**（3.739/8.739）——比 ep3 再 −0.64pp
  （近 3 量子）, 0dB 首次破 4, -5dB 单轮 −0.98, 加速段延续。预声明线 6.64
  击穿 → 官方单次启动
- 官方链执行（launch_official_ns05ext4.sh 守望模式）：GPU2 试跑 OOM（他人
  6GB 占用下 2.28GiB LSTM 连续块放不下）→ 改挂守望脚本, ns10 退出后自动在
  GPU3 串联 enhance(e4 ckpt)+evaluate_combo；日志 official_ns05ext4.log；
  配置 eval_ns05ext4.toml；ep5/ep6 继续训不打断（若再降 ≥1 量子考虑二次确认）
- **ep5 突破（11:15 val）**：hard **3.500/7.739 avg=5.619**（ep4→ep5 单轮
  −0.62/−1.00，m5db 连续两轮各降 ~1.0 无饱和）——触发预声明"ep5<6.0 →
  收官后二次官方确认"。1.6-2× 兑现外推：0dB ≈ 5.7-6.0（5.874 线边缘）、
  m5db 大概率过 7.739 线（区间宽, 以官方为准）
- **守望器 2 布防（11:44, pid 711949）**：launch_official_ns05ext_best.sh =
  等 ns05ext 训练退出 + 守望器 1 链退出 → GPU3 用 checkpoint_best.tar（收官
  时最优, 现 best=ep5, ep6 更优则自动覆盖）跑二次官方链；eval_ns05ext_best.toml
  （exp_dir=official_ns05ext_best）；缓存 raw/common 拷自 official_ns05ext4
  （同后端 soup5 meta 守卫通过）, frontend 路径不同强制重提；enhance 产物
  入 enhanced_official_best（不覆盖 ep4 的 enhanced_official——留作 ECAPA
  零训练捷径素材）。pgrep 模式自匹配检查：本脚本名 ns05ext_best 不含
  "exp-tag ns05ext"/"launch_official_ns05ext4" 子串 ✓
- 预计时序：~14:25 ns10 ep3 退 → 守望器 1 跑 ep4 官方（~19:30 出）→
  守望器 2 接力 best 官方（~00:30 出）。ns05ext ep6 ~15:15 落盘收官
- ns10 收官（13:00 前后）：ep3 hard 4.000/10.000 avg=7.000, 曲线
  7.619→7.358→7.000——λ=1.0 到顶坐实, 留档不官方 ✓
- **守望器 1 死锁事故与修复（13:30 巡检发现）**：布防用的 `bash -c` heredoc
  外壳（pid 698302）因 nohup 管道未关残留, 其 cmdline 含脚本文本
  "lambda-ns 1.0" → 守望器 1 的 pgrep 永远匹配到自己的祖先 → while 死锁,
  ns10 退出后半小时未点火。修复：kill 698302（单进程, 守望器 bash 698304
  nohup 独立存活）；验证 pgrep 'lambda-ns [1].0' 空（括号 trick 防自匹配）。
  **教训：布防命令行里不得出现守望模式串；heredoc 写脚本+nohup 的组合
  要 `</dev/null` 断管道, 或布防后确认 bash -c 已退出**
- **来历不明文件隔离**：enhanced_official/noise_0db 下 2737 个 wav,
  mtime 09:43-09:46（3 分钟产 2737 个 GPU 增强=物理不可能; md5 抽样
  ≠ ns05_ep3 同名产物, 排除误拷）。enhance_tse.py 续跑="存在即跳过"
  （:103 os.path.isfile）无 ckpt 校验——若保留且来历错误=污染这条只跑
  一次的官方链。处置：mv → enhanced_official_quarantine_0944/ 留档,
  守望器 1 从零全新增强（多花 ~25 min 换 100% 干净, 严格占优）。
  来源未明, 待事后考古
- 守望器 2 死锁风险排查：其等待模式 "launch_official_ns05ext4"（数字 4
  结尾）不匹配自身脚本名 ns05ext_best/其布防 shell ✓；布防父 shell 已退
  （TaskStop 本地端时断开）✓

## Plan B（8.27 启动）：cross-device clean-teacher / noisy-student
- 动机（预算分解结论）：0dB 档噪声轴总预算 2.57pp 需吃 98%——结构上近乎无解；
  clean 地板 5.08 是跨设备通道罚分而非常数，压地板三档同步降——唯一没开采过的轴
- 结构（train_ts.py）：学生=soup5 可训练 / 教师=soup5 冻结吃同句 clean；
  batch = 同说话人**跨设备 utt 对**（环取每设备首条, 90k→~40k+ 对, 跨设备 100%）
  1. L_anchor=1−cos(e_s(noisy), e_t(clean))（冻结教师同句锚）
  2. L_xdev=1−cos(e_s(x_a), e_s(x_b))（跨设备正对——clean 条件下=直接压地板；
     替代无正对的 SupCon：随机 batch 无同说话人对, 配对 batch 天然有）
  3. L_hard=relu(max跨说话人cos−0.30)（in-batch 难负 hinge）
  4. L_ce=AAM-CE（prototype 初始化, 防漂移）
  Kendall 不确定性加权：L=Σ w_i·exp(−s_i)·L_i+s_i, s_i 可学习（PolyU 式）,
  先验 w=(1.0, 0.5, 0.3, 1.0)；lr=2e-5 与 robust 微调同源, 3ep
- 选点：每 epoch hard proxy, 学生自身做 extractor, 锚=soup5 本体 4.716/11.193
- 骨干开关 --backbone（ecapa=预案一轨道, 首轨先 eres2net）
- 冒烟（ts_smoke.log, 400 对 1ep）：全链路通；hard≈0.0005（hinge 少触发,
  batch 小难负不"难"）/ xdev 起点 0.52 / 小样本扰动致 val 9.773>锚（不作信号）
- 陷阱记录：8.20 read_train_csv 返回 (rows, spk2id) 二元组（早期 grep -A6 截尾
  误判签名）；noise_list 是相对路径需 load_noise_paths 重映射
- 状态：**ts1 全量已启动**（GPU1, ts1.log, 5181 it/ep × 3ep ≈ 3h；90k→~41.4k
  对全跨设备）。首跑崩溃两则已修：恰好 3s 音频 randint(0,0) 越界（>CROP 才
  随机裁）；`PY=... && nohup ... &` 赋值进后台子 shell 致 $PY 为空（显式路径）
- Kendall 权重动态（it500）：anchor 1.10→1.63 升 / ce 0.93→0.74 降 /
  xdev 0.55→0.80 升——不确定性加权按噪声水平仲裁 anchor-xdev 张力, 符合设计
- **ts1 ep1 判决（劣化, 停）**：val-hard 5.761/13.000 avg **9.381** vs 锚 7.955
  （+1.4pp 双轴皆坏）；机制确认：平衡态有效权重=1/L_i 使 anchor 稳态 ~7.3,
  且 L_hard 恒 0 时 +s 正则把 s 无界压低 → hard 权重 19→26→**215** 指数发散
  （Kendall+近零损失的标准退化）。罕见触发步被放大成尖峰, 嵌入空间被打歪
- **ts1b（同设计+权重 clamp [0.05,5.0], 其余零改动）**：单变量归因——劣化主因
  是发散尖峰还是 anchor:xdev 配比, 看 ts1b ep1；先验不动（平衡态与先验无关）
- **ts1b ep1 判决（9.489, clamp 后依旧劣化 → 结构层问题）**：anchor 整 epoch
  纹丝不动（0.136→0.135, 教师=学生初始, 该项起点即噪声扰动距离且零下降）、
  ce 缓降、xdev 平——**训练未在任何目标取得进展, 模型纯漂移**。根因判读：
  全参数无冻结微调 + batch 32（robust 线是冻结+batch 100; TSE 线只训前端
  后端冻结——两条成功线都不动后端全参数）。**Plan B 暂停**, 重设计清单：
  ① 学生加 apply_freeze 冻结策略；② 配对改跨条件（enrol-clean×test-noisy
  结构, 现 xdev 是同条件对, 与评测结构不匹配）；③ 或改为只训轻量适配层
- exp/ts_ts1、ts_ts1b 留档；日志 ts1.log / ts1b.log

## speech-noise 臂（8.27 备料, 待 GPU3 释放启动）
- 依据：训练噪声 = MUSAN free-sound（环境, 675 条）而官方测试噪声 = **单人说话
  噪声**（任务书口径）——域不匹配从基线起一直存在；MUSAN speech（librivox/
  us-gov, 426 条）在盘。TSE 的说话人条件化在环境噪声下用不上, 语音干扰才是
  它设计要解的问题——条件机制可能被解锁
- noise_speech_mix.txt = 426 speech + 426 free-sound 等量（均匀采样=50/50）,
  路径全量校验 0 坏；shuf seed=42 固定可复现
- 协议：train_tse_v3.py --lambda-ns 0.5 --base-ckpt lam03_ep3（=ns05 唯一
  变量改噪声池）, 3ep；锚 = ns05 hard 各 epoch（ep1 7.642/ep2 7.597/ep3 6.881）
- 预声明：ep3 hard < 6.64（与 ns05ext 同线）→ 触发官方单次

- **ns05ext 收官（15:10）**：ep6 hard **2.739/7.239 avg=4.989**——三连降
  6.239→5.619→4.989（0dB 首破 3, m5db 仍 −0.5 无饱和）, best=ep6 定格。
  hard avg 较锚 7.403 累计 −2.414（−33%）——兑现率哪怕 1:1 也过双线
- **speech-noise 臂点火（15:20, GPU1, pid 727065）**：nsspeech = λ_ns 0.5 +
  lam03_ep3 基座 + noise_speech_mix.txt（852 条, 唯一变量）；config =
  train_tse_v3_speech.toml；锚 = ns05 hard 曲线（7.642/7.597/6.881）；
  预声明 ep3 <6.64 → 官方。首启崩于 config 笔误（data_dir_820 抄成
  /home/home）, 修正重启 ✓；ep3 收官预计 ~21:30
- ep4 官方链（13:35 enhance 20min 完成 → 14:35 evaluate_combo）：raw_noise_0db
  = 7.6139 与 8.24 soup5 逐位一致（缓存复用零污染 ✓）, frontend 两组为
  ep4 真实成绩, 预计 ~16:00 出

- **ep4 官方双轴落地（16:35）**：0dB **6.1646**（−26.5%, 距线 0.29）/
  -5dB **7.6012**（**−31.3%, 首次过 -30% 线**！余量 0.138）。raw 三组
  与 8.24 soup5 逐位一致（零污染三连 ✓）。兑现率重标定：0dB ~0.55×、
  m5dB ~0.9×（hard avg 降 0.642 → 官方降 0.355/0.572）
- ep6 官方外推：0dB ≈ 6.1646−0.55 = 5.61（过线余量 ~0.26）；
  m5dB ≈ 7.60−1.35 = 6.25（大幅过线）——**双线达成概率高, ~17:45 揭晓**
- 守望器 2 已点火（16:33, pid 733823）：enhance best(=ep6) 20min →
  evaluate ~50min → official_ns05ext_best/ 落地
- nsspeech ep1 过半（it 7600/15000）, ce 2.47→1.78 健康降（语音干扰
  难度高属预期）, ep1 val ~17:00

- nsspeech ep1 val：hard 4.807/11.000 avg **7.903**（vs 锚 +0.5、vs ns05
  同期 ep1 7.642 差 0.26）——在预期带内不判死（架构规律 ep3 出收益），
  判决等 ep3（~21:30）
- **ns05ext2 续训布防（17:50, 用户授权"空卡继续训", 守望器 3 pid 742103）**：
  launch_ns05ext2.sh = 等 best 官方链退出 → GPU3 从 checkpoint_best.tar(=ep6)
  续训 ep7-9（tag ns05ext2, 新 exp 目录不覆盖）；GPU0 空卡铁律不动；
  预声明 ep7-9 对照 ns05ext 曲线, 收官破 4.989 后是否花官方名额视 ep6
  官方兑现+用户决定；布防用 scp+独立 nohup（避开守望器 1 heredoc 死锁坑）✓
- 排期影响：GPU3 = best 官方链(→~18:20) → ns05ext2(→~23:30)；#18 多后端
  评测改排 GPU1（nsspeech 21:30 收官后）或 GPU3（23:30 后）

- **ep6 官方 0dB 落地（18:50）：5.7660 = −31.28%, 过 0dB 线！**（线 5.874,
  余量 0.108；ep4 6.1646→ep6 单步 −0.40, 兑现率实测 0.4×/0dB 轴）。
  双线只差 ep6 m5db（打分中, 外推 ~6.3）。raw 四组逐位一致（零污染四连）
- 待查：pid 740586 进程组（35min 前, 6 进程）身份——收 m5db 时确认

- **🏆 双线达成（19:05 确认）：ns05ext ep6 官方 0dB 5.7660（−31.28%）/
  -5dB 6.9723（−36.93%）, 均过 -30% 线**。基础目标达成。
  完整官方轨迹（TSE 前端+soup5 后端, 全部 3s trials_cross_device）：
  | ckpt | 0dB | 降幅 | -5dB | 降幅 |
  |---|---|---|---|---|
  | 基线 | 8.3911 | — | 11.0557 | — |
  | lam03_ep3（Phase1）| 7.2933 | −13.1% | 9.2020 | −16.8% |
  | ns05_ep3（Phase2）| 6.5191 | −22.3% | 8.1732 | −26.1% |
  | ns05ext ep4 | 6.1646 | −26.5% | 7.6012 | −31.3% |
  | **ns05ext ep6** | **5.7660** | **−31.3%** | **6.9723** | **−36.9%** |
  兑现率终标定：0dB ep4→ep6 hard 降 1.0 → 官方降 0.40（0.4×）;
  m5dB hard 降 1.5 → 官方降 0.63（0.42×）——延长段兑现率收敛到 ~0.4×,
  外推新 epoch 时用此值。距 clean 地板（5.0810）: 0dB 余 0.68、-5dB 余 1.89
- 守望器 3 点火确认（19:05）：ns05ext2 从 ep6 ckpt 续训（747696,
  [init] 日志 ✓）——ep7 val 预计 ~21:00, 收官 ~23:30; 破 4.989 则为
  汇报加码备胎（是否花官方名额待用户决定）
- 740586 进程组确认 = nsspeech 的 DataLoader workers（cmdline 同主进程,
  无害虚惊）

## 实验轨迹
- Phase 0 结案：旧权重复现精确 7.2049/9.1533（管线正确性 ✓）；w*（proxy 搜）
  官方一次应用 7.3600/9.3582，比旧融合差 +2.2% 相对 → 干净口径下 TSE 入融合
  无净增益（旧融合带官方调权乐观偏差）。全程零违规
- hard 链首跑崩于 npz 缓存回读（dict→0-d object 数组未解包），item() 修复后
  完成（proxy_hard_run2.log）

## 8.27 傍晚：任务书对照收官 + 口径闭合布防
- 任务书（E:\china_mobile\任务.xlsx）对照结论：主验收目标 ✓（−31.3/−36.9%，
  fallback 条款）；欠缺分项=1s/5s 时长、reverb/overlap 场景、ROC/DET+FAR/FRR、
  失败案例（#19/#21 排程）。挑战目标 EER≤0.5% 未达（协议允许 fallback）
- 用户三确认（8.27 傍晚）：①基线口径"满足所有的"→#18 扩为三后端（原始
  ERes2Net + ECAPA + ReDimNet2-B6/90k，任务书两处口径全覆盖）②子课题二不做
  ③论文作为长线目标推进（#23：VoxCeleb 复现 + baseline 对比，Interspeech 2027）
- 守望器 4 布防（17:45，pid 755773）：nsspeech 退出后 GPU1 跑 #18 口径闭合
  ——eval_official_pretrained.py（原始官方 eres2net_large ckpt × ep6 best
  增强产物）。预声明自检：raw 复现官方基线 8.3911/11.0557（±0.05）→ FE 按
  -30% 公式直接判定；不复现则 UNVERIFIED。soup5 是否算"改进方法"的口径
  争议就此彻底绕开
- 今晚时间线：nsspeech ep3 ~18:45 收官（hard<6.64 → 官方链走 GPU3）；
  ns05ext2 ep7 hard ~18:40、收官 ~23:30；口径闭合评测 GPU1 ~19:00-21:00
- **nsspeech 判定规则修正**（18:15，诚实记录）：原预声明"ep3 hard<6.64→官方"
  不适配——本臂训练噪声为人声（noise_speech_mix 852 条），hard proxy 是
  freesound 环境噪声集，域正好相反（ep1/ep2 avg 7.90/7.87，m5db 卡 11.0，
  注定过不了线；但 0db 4.8 段已优于锚 4.307 的差距在收窄）。proxy 无法检验
  "人声训练治人声测试"假设 → 改为**假设检验式一次性官方**（无论好坏都报，
  不构成挑数字）。守望器 5 布防（pid 761584）：口径闭合评测退出后 GPU1
  接力 enhance_tse（nsspeech best）→ evaluate_combo FE（soup5，raw/enrol
  缓存自 official_ns05ext_best 复用），预计 ~21:05-23:35 出结果
- **ns05ext2 ep7 破线**（18:35）：hard 4.795（0db 2.807/m5db 6.784）< ep6
  4.989 → 预声明"破 4.989 → 官方加码"触发 ✓。0db 2.807 为全程最优单档。
  续训继续 ep8/ep9（~23:30 收官，checkpoint_best=全程最优），守望器 6 布防
  （pid 765391）：训练退出后 GPU3 接力 enhance（ext2 best）→ evaluate_combo
  FE（soup5），预计 ~02:00 出官方。若优于 5.7660/6.9723 → 汇报数字升级；
  否则 ep6 仍为终版
- 三守望器在线（18:45 清点）：755773 口径闭合（GPU1 第一棒）/ 761584
  nsspeech 官方（GPU1 第二棒）/ 765391 ext2 加码（GPU3），无残留外壳

## 8.28 凌晨：三口径全闭合 + 续训两轮加码
- **ext2 官方加码成功（01:00）**：ep9 best（hard 4.381）官方 0dB 5.4669（−34.8%）/
  -5dB 6.4615（−41.6%），大幅刷新 ep6 纪录。hard 三连降 4.795→4.608→4.381，
  兑现率 ~0.5×。0dB gap 压至 0.39（clean 地板 5.0810）——噪声惩罚消除 85%
- **nsspeech 假设检验证伪**：人声噪声训练官方 7.1840/9.0052（−14.4/−18.5%），
  差于 freesound 训练——"人声池直击域失配"不成立，官方证据入失败路线表
- **#18 三后端官方管线结案（06:10）**：raw 四档逐位级复现基线（11.1481/14.3576/
  8.3903/11.0549），唯一变量=ext2 ep9 增强 wav。**ECAPA −35.6/−41.5%、原始
  ERes2Net −33.5/−40.9% 双口径双档全过**；ReDimNet2-B6 −18.8/−21.7%（bonus，
  如实报告）。方法论：我们链（evaluate_combo）与官方链有 1-1.5% 系统差
  （raw 自检发现），口径闭合以官方 infer 管线为准。结果：
  exp/official_pipeline_3backends/，脚本 eval_official_pipeline.sh
- ext3 再续训（01:20 GPU3）：ep10 回升 4.597 → ep11 破线 4.023（0db 2.545/
  m5db 5.500）→ 守望器 7 布防（843448），ep12 收官后官方加码（~09:40 出）
- GPU 规则更新（用户 22:00）：别人有进程时四卡全用，全消失才留空卡
- #19 勘探：1s/5s 数据+trials 在盘（官方 evaluate_noise.sh 本就三时长设计，
  enrol 随时长变）；enhance_tse.py 三处 3s 硬编码 → 写 v2 参数化（--dur），
  评测走官方管线（enrol 嵌入 exp/clean_bench/{backend}/embeddings_{1s,5s} 现成）

## 8.28 白天：ext3 官方收官（新王）+ 时长梯度 + dur-aug + 分项报告 CPU 全套
- **ext3 官方终值（09:00）**：ep12 best（hard 3.869）官方 **0dB 5.2351（−37.6%）/
  -5dB 6.0979（−44.8%）**，双超 ext2 的 −34.8/−41.6。三连训曲线 ep6 5.766→
  ep9 5.467→ep12 5.235（0dB），兑现率实测 0.58×。ep 终值表（combo 口径 raw
  7.6139/9.7148；官方口径降幅对 8.3911/11.0557）：
  ep4 6.1646/7.6012（−19/−22）→ ep6 5.7660/6.9723（−31/−37）→
  ep9 5.4669/6.4615（−34.8/−41.6）→ **ep12 5.2351/6.0979（−37.6/−44.8）**
- **泄露三方实证（用户质询，实查）**：训练 9900 ∩ 官方测试 210 = 0；
  proxy 100 ∩ 测试 = 0；proxy ∩ 训练 = 0。选点/触发全走 proxy，官方数字
  不回流训练 → 续训增益非泄露。soup5 后端记忆已由官方管线独立复现排除
- **#19 时长梯度收官（ep9 前端，官方管线，10:35）**：0dB 降幅 1s −19.4% <
  3s −33.5% < **5s −35.4%**（fe 5.0747 vs raw 7.8627）；m5dB 1s −22.2% <
  3s −40.9% < **5s −42.6%**（fe 5.7921 vs raw 10.0809）——单调，5s 完全保持
  甚至略超主线（信息量补偿），1s 短板=长度 OOD 实锤 → dur-aug 修复中
- **#18 补遗：ReDimNet2-B6 −18.8/−21.7**（vox2+cnc2 轻量档，同 ext2 产物）：
  三强一弱，B6 差距=域外嵌入空间+容量天花板，multi-verifier 列论文线
- **#25 dur-aug 布防（用户点头"最优方法+适配"思路，最后一个训练实验）**：
  train_tse_v4.py 段级随机时长 {3,2,1}s@{.5,.25,.25}（seg=200it，batch 内
  等长天然保持）；守望器 8 自动定 base——修了降幅口径混用 bug（原 combo
  分子配官方阈值会错退 ext2），改绝对值直比 → base=ext3 best。GPU3，
  09:00 起训 ep1 it5400/15000（36%），~16:00 出。双门槛：①3s hard 不降
  ±0.3pp ②1s 官方降幅 ≥28%。用户指示：此后不再新增训练实验，终评=
  最佳ckpt×{1,3,5}s×三后端大表；ep12 泛化不降可议 ext4（终评后定）
- **#21 分项报告 CPU 全套完成（任务书第4条）**：report_metrics.py 复算
  EER/minDCF 逐位对上官方（7.6139/0.5942）；固定阈值（clean EER 工作点
  0.2266）FAR/FRR：前端 FRR 11.75→7.07（0dB）/ 17.25→9.98（m5dB），
  FAR 持平略降——部署视角核心卖点。DET/ROC/failures（FR/FA top20 含
  Device/Distance/Dialect 分层）落盘 exp/analysis_21/。场景评测（overlap/
  reverb）按用户指示不重跑，留优化阶段；用户 clean/result 库存（denoised
  官方去噪对照 7.79/9.92）收进对照表

## #26 噪声域泛化测试（8.28 完，ext3 ckpt，官方管线，3s）
协议：镜像 gen_noise_benchmark.py（同 seed/clean/SNR/文件名，仅换噪声池）生成 music(660)/speech(426)/soundbible(87) 三域 × {0,-5}dB 测试集（各 16401 条），raw 与 FE（exp/tse_v3_ns05ext3/enhanced_noise_domains）同口径对照。

| 条件 | raw | ext3 FE | 相对降幅 |
|---|---|---|---|
| （域内 0dB / −5dB 参照） | 8.3911 / 11.0557 | 5.2351 / 6.0979 | −37.6% / −44.8% |
| music 0dB | 9.6367 | 6.3326 | −34.3% |
| music −5dB | 13.2895 | 8.2614 | −37.8% |
| speech 0dB | 8.6329 | 8.2474 | −4.5% |
| speech −5dB | 10.8307 | 12.0634 | +11.4%（劣化）⚠ |
| soundbible 0dB | 8.1281 | 5.3736 | −46.2% |
| soundbible −5dB | 10.1582 | 6.1632 | −39.3% |

结论：①非语音噪声域（器乐/音效）增益全额迁移（保值率 84%~103%），成绩非噪声库匹配成分；②人声噪声为结构性盲区（0dB 无效、−5dB 前端反致 11% 劣化）——TSE 的频谱语音/非语音判别对竞争说话人失效，与 nsspeech 训练证伪、overlap 待优化三方互证，指向同一根因；后续 overlap 优化即攻此盲区。数据：AI/8.28/noise_domains/{eval_raw,eval_fe}。dur-aug 收敛后终版大表另跑 FE 侧。
