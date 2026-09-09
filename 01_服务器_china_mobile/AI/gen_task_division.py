"""
生成子课题一合作分工方案 Word 文档
"""
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
import datetime

doc = Document()

# ── 全局样式 ──
style = doc.styles['Normal']
style.font.name = '微软雅黑'
style.font.size = Pt(10.5)
style.element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
style.paragraph_format.space_after = Pt(6)
style.paragraph_format.line_spacing = 1.25

for level in range(1, 4):
    hs = doc.styles[f'Heading {level}']
    hs.font.name = '微软雅黑'
    hs.element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
    hs.font.color.rgb = RGBColor(0x1A, 0x56, 0xDB)

# ── 辅助函数 ──
def add_table(doc, headers, rows, col_widths=None):
    """添加格式化表格"""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # 表头
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(9)

    # 数据行
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.rows[r + 1].cells[c]
            cell.text = str(val)
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)

    doc.add_paragraph()  # 表后空行
    return table

# ══════════════════════════════════════════════════════════════
# 封面
# ══════════════════════════════════════════════════════════════
doc.add_paragraph()
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run('子课题一：多条件中文声纹识别\n与噪声鲁棒性增强')
run.font.size = Pt(22)
run.bold = True
run.font.color.rgb = RGBColor(0x1A, 0x56, 0xDB)

doc.add_paragraph()

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run('双人合作分工方案')
run.font.size = Pt(18)
run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

doc.add_paragraph()

meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
meta.add_run(f'日期：{datetime.date.today().strftime("%Y 年 %m 月 %d 日")}\n').font.size = Pt(11)
meta.add_run('约束条件：两人均无本地 GPU 服务器、无音频采集设备\n').font.size = Pt(11)
meta.add_run('核心策略：云端 GPU 协作 + 按技术栈层级分工').font.size = Pt(11)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════
# 一、分工总览
# ══════════════════════════════════════════════════════════════
doc.add_heading('一、分工总览', level=1)

doc.add_paragraph(
    '本方案的核心思路是按技术栈层级纵向切分，而不是按场景或模型横向切分。'
    '这样每个人拥有一整条完整的技术链路，可以独立汇报、独立成文。'
)

doc.add_heading('分工一句话概述', level=2)

add_table(doc,
    ['角色', '方向', '一句话描述', '核心交付物'],
    [
        ['同学 A\n（你）',
         '模型训练与鲁棒优化',
         '负责"模型长什么样、怎么训、怎么让它抗噪"',
         '训练代码 + 模型权重 + 消融实验报告'],
        ['同学 B\n（出差同学）',
         '数据管线与评测体系',
         '负责"数据从哪来、怎么测、模型好不好使"',
         '评测框架 + 全场景结果 + 对比分析报告'],
    ]
)

doc.add_heading('为什么这样分？', level=2)

reasons = [
    ('纵向切分，各自闭环',
     '每个人从数据→方法→评测→分析，故事线完整。'
     '同学 A 讲"我用了什么训练策略让模型在噪声下更好"，'
     '同学 B 讲"我设计了怎样的评测体系来全面检验模型"。两条线互相引用但不互相依赖。'),
    ('适配硬件约束',
     '同学 B 的工作（数据下载、预处理、评测代码、可视化）全部可以在 CPU 笔记本上完成，'
     '出差途中不受影响。同学 A 的工作需要 GPU，但通过 AutoDL / Colab Pro+ 等云 GPU 解决。'),
    ('解耦开发节奏',
     '同学 B 拿到公开数据就可以立刻开工，不需要等模型训练完成。'
     '评测框架可以用 3D-Speaker 官方预训练模型先跑通，同学 A 的模型出来后再换上正式跑。'),
    ('独立汇报无冲突',
     '两人可以分别写各自的报告，各自有完整的实验设计、结果分析和失败案例。'
     '合并时直接拼接即可，不存在"谁写了哪段"的模糊地带。'),
]
for title, desc in reasons:
    p = doc.add_paragraph()
    run = p.add_run(f'▸ {title}：')
    run.bold = True
    p.add_run(desc)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════
# 二、同学 A 任务详情
# ══════════════════════════════════════════════════════════════
doc.add_heading('二、同学 A（你）：模型训练与鲁棒优化', level=1)

doc.add_heading('2.1 工作定位', level=2)
doc.add_paragraph(
    '你的核心问题是："给定一个声纹模型，怎么训练它才能在 -5dB/0dB 噪声、回声混响、'
    '多人重叠等恶劣条件下依然保持高识别精度？" 你需要交付从基线到最优模型的完整训练链路。'
)

doc.add_heading('2.2 详细任务清单', level=2)

add_table(doc,
    ['阶段', '任务', '具体内容', '预计工作量', '是否需要 GPU'],
    [
        ['1. 环境搭建', '云 GPU 环境配置',
         '注册 AutoDL（autodl.com）或 Colab Pro+，租用 RTX 4090/A5000 实例；'
         '安装 PyTorch + 3D-Speaker 工具链 + 依赖库；'
         '挂载云盘存储数据集和模型权重',
         '1-2 天', '否（配置为主）'],

        ['2. 数据就绪', '训练数据准备',
         '从同学 B 获取预处理好的 3D-Speaker 训练/验证/测试划分；'
         '确认数据划分符合"说话人互不重叠"规则；'
         '验证数据加载 pipeline 正确性',
         '0.5 天', '否'],

        ['3. 基线复现', 'ECAPA-TDNN 基线',
         '在干净语音上复现 ECAPA-TDNN，跑通完整训练+评测流程；'
         '与 3D-Speaker 官方报告结果对齐（EER 误差 ≤0.1%）；'
         '输出：干净语音基线 EER/minDCF',
         '2-3 天', '是（训练）'],

        ['4. 噪声基线', '未增强模型在噪声下的表现',
         '直接对同学 B 构建的多条件测试集（-5dB/0dB 噪声、回声、混响、重叠）'
         '提取嵌入并评测；输出噪声场景基线 EER/minDCF 表',
         '1 天', '是（推理）'],

        ['5. 鲁棒训练', '多条件联合训练',
         '实现 multi-condition training：训练时在线混合噪声/RIR；'
         '对比不同增强策略：SpecAugment、噪声混合、混响卷积；'
         '输出：增强训练后的全场景 EER 对比表',
         '3-4 天', '是（训练）'],

        ['6. 大模型', 'WavLM 方案',
         '加载 WavLM Base+ 预训练权重，接入声纹分类头；'
         '对比 WavLM 在噪声场景下 vs ECAPA-TDNN 的效果；'
         '（可选）尝试 WavLM Large',
         '2-3 天', '是（训练 + 推理）'],

        ['7. 小模型协同', '轻量化方案',
         '训练 CAM++ 或 MobileNet 风格的小模型；'
         '尝试知识蒸馏（大模型指导小模型）；'
         '输出：大小模型在噪声下的对比',
         '2-3 天', '是（训练）'],

        ['8. 消融实验', '贡献度分析',
         '逐一控制变量：数据增强 × 鲁棒训练 × 模型架构 × 后端校准；'
         '量化每个模块对 EER 降幅的贡献；'
         '输出：消融实验表 + 分析',
         '2-3 天', '是（多次训练）'],

        ['9. 后端校准', 'AS-Norm / Adaptive S-Norm',
         '在验证集上实现分数归一化；'
         '验证冻结阈值在测试集上的 FAR/FRR；'
         '对比校准前后的 DET 曲线',
         '1-2 天', '否（CPU 即可）'],

        ['10. 交付', '模型与报告',
         '整理最终模型权重和训练配置；'
         '撰写"模型训练与鲁棒优化"报告（独立成章）；'
         '输出：训练代码 + 模型权重 + 消融报告 + 失败案例分析',
         '2-3 天', '否'],
    ]
)

doc.add_heading('2.3 你的时间线', level=2)

add_table(doc,
    ['周次', '里程碑', '关键产出'],
    [
        ['第 1 周', 'GPU 就绪 + 基线跑通', '干净语音 ECAPA-TDNN 基线 EER'],
        ['第 2 周', '噪声基线 + 多条件训练 V1', '未增强 vs 增强在各场景的 EER 对比'],
        ['第 3 周', 'WavLM + 小模型协同', '大模型 vs 小模型在噪声下的效果对比'],
        ['第 4 周', '消融实验 + 后端校准', '各模块贡献度量化表'],
        ['第 5 周', '冲刺优化 + 报告初稿', '若未达 0.5%，针对性优化'],
        ['第 6-7 周', '报告定稿 + 代码整理', '最终模型 + 完整训练代码'],
    ]
)

doc.add_heading('2.4 你的独立汇报主线', level=2)

doc.add_paragraph(
    '你的报告可以按以下故事线组织：\n\n'
    '"本研究以 ECAPA-TDNN 为基线，系统探索了多条件鲁棒训练、大模型（WavLM）和小模型协同方案'
    '在中文声纹识别噪声鲁棒性上的效果。通过消融实验，我们量化了数据增强、鲁棒训练策略、'
    '模型规模和后端校准各自对 EER 降幅的贡献。最终方案在 -5dB 单说话人噪声场景下'
    '实现 EER = X.X%，相对基线降幅 XX%。"'
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════
# 三、同学 B 任务详情
# ══════════════════════════════════════════════════════════════
doc.add_heading('三、同学 B（出差同学）：数据管线与评测体系', level=1)

doc.add_heading('3.1 工作定位', level=2)
doc.add_paragraph(
    '她的核心问题是："用什么数据测、怎么测才全面、怎么判断模型在不同条件下到底好不好？" '
    '她负责从公开数据集出发，构建覆盖所有要求场景的测试管线，'
    '设计评测指标体系，并对同学 A 交付的模型进行全场景验收。\n\n'
    '关键优势：她的全部工作可以在普通笔记本（CPU）上完成，出差途中不受影响。'
    '评测框架可以用 3D-Speaker 预训练模型先跑通，同学 A 的模型 ready 后直接替换。'
)

doc.add_heading('3.2 详细任务清单', level=2)

add_table(doc,
    ['阶段', '任务', '具体内容', '预计工作量', '是否需要 GPU'],
    [
        ['1. 数据获取', '公开数据集下载与整理',
         '下载 3D-Speaker 中文数据（全部子集）；\n'
         '下载 MUSAN（噪声）、DEMAND（环境噪声）、RIRS_NOISES（RIR）；\n'
         '整理数据目录结构，编写数据清单文档',
         '1-2 天', '否（下载 + 解压）'],

        ['2. 数据划分', '训练/验证/测试集构建',
         '按说话人互不重叠规则划分三集合；\n'
         '生成注册语音-测试语音配对清单（正例/负例）；\n'
         '控制注册时长和测试时长（1s/3s/5s 三档）；\n'
         '输出：数据划分报告 + 划分代码',
         '2-3 天', '否'],

        ['3. 测试集构建', '多条件测试集生成',
         '干净语音测试集（基准）；\n'
         '噪声测试集：MUSAN/DEMAND 叠加，精确控制 SNR=-5dB 和 0dB；\n'
         '回声/混响测试集：RIR 卷积，记录 T60；\n'
         '多人重叠测试集：2-3 人混合，记录 TIR；\n'
         '跨设备测试集：按 3D-Speaker 设备标签分组；\n'
         '短语音测试集：裁剪至不同时长\n'
         '输出：6 种场景的标准化测试集 + 生成脚本',
         '3-4 天', '否（纯音频处理）'],

        ['4. 评测框架', '指标计算与可视化',
         '实现 EER 计算（含置信区间）；\n'
         '实现 minDCF（p_target=0.01/0.001）；\n'
         '实现固定阈值 FAR/FRR（阈值在验证集确定，测试集冻结）；\n'
         '实现 ROC 曲线和 DET 曲线绘制；\n'
         '实现全场景结果汇总表自动生成\n'
         '输出：评测框架代码 + 使用文档',
         '3-4 天', '否（CPU 计算）'],

        ['5. 基线评测', '3D-Speaker 预训练模型评测',
         '下载 3D-Speaker 官方预训练模型；\n'
         '在多条件测试集上提取嵌入；\n'
         '输出各场景下的基线 EER/minDCF/FAR/FRR/ROC/DET\n'
         '输出：基线评测报告（全场景）',
         '2-3 天', '否（推理可用 CPU 或 Colab 免费 GPU）'],

        ['6. 模型评测', '同学 A 模型全场景验收',
         '对同学 A 交付的每个模型版本进行全场景评测；\n'
         '生成与基线的对比表（绝对数值 + 相对降幅）；\n'
         '绘制 ROC/DET 对比曲线；\n'
         '生成失败案例清单（每条场景下 EER 最高的 20 条试验）\n'
         '输出：全场景评测报告 + 对比分析',
         '2-3 天', '否（推理）'],

        ['7. 文献调研', '相关方法梳理',
         '调研近 3 年声纹鲁棒性相关论文（ICASSP/Interspeech 为主）；\n'
         '整理 baseline 方法列表和数据增强策略汇总；\n'
         '输出：文献调研笔记 + 方法对比表',
         '2-3 天', '否'],

        ['8. 报告撰写', '评测与分析报告',
         '汇总全部评测结果；\n'
         '从评测角度分析：哪个场景最难、哪种增强最有效、瓶颈在哪；\n'
         '撰写"数据管线与评测体系"报告（独立成章）；\n'
         '输出：评测报告 + 全部图表 + 失败案例集',
         '3-4 天', '否'],
    ]
)

doc.add_heading('3.3 她的时间线', level=2)

add_table(doc,
    ['周次', '里程碑', '关键产出'],
    [
        ['第 1 周', '数据下载完成 + 数据划分方案确定',
         '3D-Speaker + 噪声/RIR 数据就绪，数据划分代码'],
        ['第 2 周', '多条件测试集构建完成',
         '6 种场景的标准化测试集 + 生成脚本'],
        ['第 3 周', '评测框架完成 + 3D-Speaker 基线评测',
         '评测代码 + 预训练模型在全场景的基线结果'],
        ['第 4 周', '同学 A 模型 V1 全场景评测',
         '第一版增强模型的全场景对比报告'],
        ['第 5 周', '多版本模型评测 + 失败案例分析',
         '各模型版本的趋势对比 + 失败案例清单'],
        ['第 6-7 周', '最终评测 + 报告定稿',
         '最终版评测报告 + 全部可视化图表'],
    ]
)

doc.add_heading('3.4 她的独立汇报主线', level=2)

doc.add_paragraph(
    '她的报告可以按以下故事线组织：\n\n'
    '"本研究构建了一套覆盖 6 种声学条件的标准化中文声纹评测体系，'
    '包括单说话人噪声（-5dB/0dB）、回声混响、多人重叠、跨设备和短语音场景。'
    '基于该体系，我们对 ECAPA-TDNN 基线、多条件鲁棒训练模型、WavLM 大模型和小模型协同方案'
    '进行了系统性对比评测。评测结果表明，[最优方案] 在各场景下均取得显著提升，'
    '其中单说话人噪声场景 EER 降至 X.X%，相对基线降幅 XX%。'
    '通过失败案例分析，我们发现 [主要瓶颈] 是制约进一步优化的关键。"'
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════
# 四、协作接口
# ══════════════════════════════════════════════════════════════
doc.add_heading('四、两人协作接口设计', level=1)

doc.add_paragraph(
    '为了最大程度减少互相等待，两人之间只需要在三个时间点对接。'
    '其他时间完全独立工作。'
)

doc.add_heading('4.1 三个对接节点', level=2)

add_table(doc,
    ['节点', '时机', '传递方', '传递内容', '接收方动作'],
    [
        ['接口 1', '第 1 周初',
         '同学 B → 同学 A',
         '数据划分方案（train/val/test 说话人列表）\n'
         '+ 数据加载代码（DataLoader）',
         '同学 A 以此搭建训练 pipeline，确保双方用同一套数据划分'],
        ['接口 2', '第 2 周末',
         '同学 B → 同学 A',
         '多条件测试集（完整打包，含元数据）\n'
         '+ 评测脚本（API 约定：输入嵌入 .npy，输出指标 dict）',
         '同学 A 可以在本地/云端自测，快速迭代'],
        ['接口 3', '第 3 周起持续',
         '同学 A → 同学 B',
         '各版本模型的测试集嵌入向量（.npy）\n'
         '+ 模型配置说明（架构、训练策略、参数量）',
         '同学 B 运行正式评测，产出对比报告和可视化'],
    ]
)

doc.add_heading('4.2 约定规范（建议）', level=2)

specs = [
    '数据格式：所有音频统一为 16kHz / 16bit / mono / WAV',
    '嵌入格式：Numpy .npy，shape = (N, 192) 或 (N, 256)',
    '评测 API：统一函数签名 evaluate(embeddings, trials, threshold) → dict',
    '阈值规则：同学 B 在验证集上确定阈值，同学 A 不自行调参',
    '代码仓库：GitHub 私有仓库，main 分支放稳定版本，各自开 dev 分支',
    '实验记录：共享腾讯文档/WPS 表格，记录每次实验的配置、结果和备注',
    '沟通频次：每周一次 15 分钟同步会，其他时间异步（微信/飞书）',
]
for s in specs:
    doc.add_paragraph(s, style='List Bullet')

doc.add_page_break()

# ══════════════════════════════════════════════════════════════
# 五、GPU 解决方案
# ══════════════════════════════════════════════════════════════
doc.add_heading('五、云 GPU 解决方案（针对无本地 GPU 约束）', level=1)

add_table(doc,
    ['方案', '配置', '价格', '适用场景', '推荐度'],
    [
        ['AutoDL', 'RTX 4090 (24GB) / A5000 (24GB)',
         '约 3-5 元/小时',
         '训练 ECAPA-TDNN / ReDimNet2；\n单卡 24GB 足够跑 batch_size=64-128',
         '⭐⭐⭐⭐⭐ 首选'],
        ['Colab Pro+', 'A100 (40GB) 或 V100 (16GB)',
         '约 50 美元/月',
         '快速实验、小规模训练；\nV100 16GB 对大模型可能不够',
         '⭐⭐⭐⭐ 备选'],
        ['Google Cloud / AWS', 'A100 (40GB/80GB)',
         '约 10-30 元/小时',
         '需要超大 batch 或多卡并行时',
         '⭐⭐⭐ 仅在必要时'],
        ['校内资源', '找导师/实验室申请',
         '免费（可能排队）',
         '所有训练任务',
         '优先尝试'],
    ]
)

doc.add_heading('建议策略', level=2)
doc.add_paragraph(
    '1. 首选 AutoDL 按量计费：RTX 4090 实例，预装 PyTorch 镜像，开箱即用。\n'
    '2. 训练一个 ECAPA-TDNN 大约需要 8-12 小时（单卡），约 30-50 元。\n'
    '3. 整个子课题一预计 GPU 总时长 150-250 小时，总费用约 500-1000 元。\n'
    '4. 同学 B 的推理任务：轻量模型可以 CPU 跑，WavLM 推理用 Colab 免费 GPU 即可。'
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════
# 六、子课题一 关键路径
# ══════════════════════════════════════════════════════════════
doc.add_heading('六、子课题一关键路径与风险', level=1)

doc.add_heading('6.1 关键路径', level=2)

doc.add_paragraph(
    '第 1 周是整条关键路径的起点。同学 B 必须在前 3 天内完成数据划分方案，'
    '交付给同学 A 开始训练。同学 B 的测试集构建（预计第 2 周完成）不阻塞同学 A 的训练启动——'
    '同学 A 可以先用干净语音验证集监控训练收敛。'
)

doc.add_paragraph(
    '真正的"汇合点"在第 3 周：同学 A 的 V1 增强模型出炉，同学 B 的评测框架就绪。'
    '从第 3 周起，两人进入"训练-评测-反馈"迭代循环。'
)

doc.add_heading('6.2 风险矩阵', level=2)

add_table(doc,
    ['风险', '概率', '影响', '影响谁', '应对'],
    [
        ['云 GPU 排队/不可用', '中', '中', '同学 A',
         '提前注册 AutoDL + Colab 双账号，互为备份'],
        ['同学 B 下载数据缓慢\n（出差网络不稳定）', '低', '中', '同学 B',
         '3D-Speaker 约 50GB，可以淘宝找人代下到百度网盘；'
         '噪声数据集都很小（<5GB），直接下载即可'],
        ['数据划分有问题，\n训练/测试说话人泄漏', '低', '致命', '两人',
         '第 1 周交叉验证：用说话人 ID 集合取交集，确保为 0'],
        ['ECAPA-TDNN 基线\n复现不出官方结果', '中', '中', '同学 A',
         '第 1 周就验证；若差距大，检查数据划分和超参；'
         '必要时直接用官方 checkpoint'],
        ['-5dB EER 压不到 0.5%', '较高', '中', '同学 A',
         '保底目标是相对降幅 ≥30%，不需要一定到 0.5%；'
         'WavLM 大模型是降级方案'],
        ['两人实验结果不一致', '中', '中', '两人',
         '统一随机种子 + 统一评测脚本 + 统一阈值确定规则'],
    ]
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════
# 七、交付物清单
# ══════════════════════════════════════════════════════════════
doc.add_heading('七、交付物清单', level=1)

doc.add_heading('7.1 同学 A 交付物', level=2)

items_a = [
    '训练代码（含数据加载、模型定义、训练脚本、配置文件）',
    '所有模型权重（基线 ECAPA-TDNN + 增强模型 + WavLM + 小模型）',
    '消融实验报告（各模块贡献度量化分析）',
    '模型对比报告（不同架构/规模在噪声下的效果对比）',
    '失败案例分析（从模型训练角度）',
    '模型使用说明文档',
]
for item in items_a:
    doc.add_paragraph(item, style='List Bullet')

doc.add_heading('7.2 同学 B 交付物', level=2)

items_b = [
    '数据划分代码 + 数据清单（train/val/test 说话人及语音列表）',
    '多条件测试集生成脚本 + 测试集元数据',
    '评测框架代码（EER/minDCF/FAR/FRR/ROC/DET 全套）',
    '3D-Speaker 预训练模型全场景基线评测报告',
    '同学 A 各版本模型全场景评测对比报告',
    '失败案例分析（从评测角度，含具体试验级别的错误样例）',
    '文献调研笔记 + 方法对比表',
    '最终汇总报告（数据管线 + 评测结果 + 分析结论）',
]
for item in items_b:
    doc.add_paragraph(item, style='List Bullet')

doc.add_heading('7.3 共同交付物', level=2)

items_shared = [
    '最终模型方案说明（最好的模型是什么、怎么训的、为什么好）',
    '全场景最终结果汇总表（干净/噪声/回声/混响/重叠/跨设备/短语音）',
    '与 ECAPA-TDNN 基线的相对 EER 降幅汇总',
]
for item in items_shared:
    doc.add_paragraph(item, style='List Bullet')

doc.add_page_break()

# ══════════════════════════════════════════════════════════════
# 八、附录
# ══════════════════════════════════════════════════════════════
doc.add_heading('八、附录：两人独立汇报大纲参考', level=1)

doc.add_heading('8.1 同学 A 报告大纲', level=2)

outline_a = """1. 研究问题：多条件声学场景下的中文声纹识别鲁棒性
2. 相关工作：声纹模型架构（ECAPA-TDNN / ReDimNet2 / WavLM）、鲁棒训练方法
3. 方法：
   3.1 基线模型选择与复现
   3.2 多条件鲁棒训练策略
   3.3 大模型（WavLM）方案
   3.4 小模型协同方案（知识蒸馏）
   3.5 后端校准方法
4. 实验设置：训练数据、超参、评测协议
5. 结果与分析：
   5.1 基线在干净语音上的表现
   5.2 未增强模型在多条件场景下的退化分析
   5.3 鲁棒训练消融实验
   5.4 大模型 vs 小模型对比
   5.5 后端校准效果
6. 失败案例分析
7. 结论与展望"""

doc.add_paragraph(outline_a)

doc.add_heading('8.2 同学 B 报告大纲', level=2)

outline_b = """1. 研究问题：构建全面的中文声纹鲁棒性评测体系
2. 相关工作：声纹评测标准、多条件测试集构建方法
3. 评测体系设计：
   3.1 数据划分方案
   3.2 六种声学条件测试集构建方法
   3.3 评测指标体系（EER/minDCF/FAR/FRR/ROC/DET）
   3.4 阈值确定与冻结规则
4. 基线评测：
   4.1 ECAPA-TDNN 基线在全场景下的表现
   4.2 各场景相对干净语音的退化程度排序
5. 增强模型评测：
   5.1 多条件鲁棒训练效果
   5.2 大模型方案效果
   5.3 小模型协同方案效果
   5.4 各方案在六种场景下的横向对比
6. 失败案例分析
7. 结论与建议"""

doc.add_paragraph(outline_b)

# ── 保存 ──
output_path = r'e:\china_mobile\AI\子课题一_双人合作分工方案.docx'
doc.save(output_path)
print(f'文档已保存至: {output_path}')
print('Done.')
