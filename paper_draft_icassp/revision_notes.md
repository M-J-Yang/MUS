# 相对原稿的修改说明

日期：2026-09-06。原稿 `ICASSP2027_Paper_Templates/Template.tex`、`dads_refs.bib` 及原图表保持原状。新稿为 [main.tex](main.tex)，成稿为 [paper.pdf](paper.pdf)。

## 1. 保留的核心定位

原引言已经建立了一条有价值的线索：ASR 适配有效 → 表示分析描述变化 → 完整 Delta embedding 有用 → 既有识别器实际依赖哪些变化尚不清楚。新稿保留这条线索，以及输入相关表示差分、Taylor utility、固定原 CTC 头的实验对象。

没有把论文改成纯性能方法，也没有改写为参数空间 task vectors。标题改为 **Which Fine-Tuning Changes Matter? Probing Representation Shifts in Accented Speech Recognition**，明确实际语料范围，并避免尚未识别“最小 functional support”带来的承诺。

## 2. 核心证据的重大调整及依据

原摘要用公开 wav2vec Fold 0 的 10.499% / 118.912% 两个端点，以及 50% 保留后 11.713% 来计算“保留 98.9% 适配收益”。逐项核对缓存源端、适配端和 config 后发现两端在 convolution bias、feature normalization、stable layer normalization 上不一致，无法成立为所声明的同源微调差分。

因此移除了该公开配对的全部主结果、对应图和置信区间，改以有本地初始化记录的 **W1、W2、D0** 为证据。详细字段和路径见 [evidence_map.md](evidence_map.md) 第 5 节。这是依据实验对象定义修复论证，不是因为数字大小而筛选结果。公开 checkpoint 的真实性或独立 ASR 性能没有因此被否定；被否定的是项目当前缓存把它与指定源模型组成 fine-tuning pair 的解释。

引言仍研究理解适配，但结论从“少量变化承载几乎全部适配信息”改为“同等规模的高低分变化组对固定头有很不一样的作用”。在有效本地结果上，这一结论有明显删除差异支持。三个 checkpoint 无法分别代表三种独立架构或三折同模型复现，正文用 W1/W2/D0 避免混淆。

## 3. 全文结构与写法

| 部分 | 调整 | 原因 |
|---|---|---|
| 标题与摘要 | 指向 accented speech；同时给出高低分删除差异及 rescaled random 反例 | 摘要不能只保留最好看的结果；适用范围应从标题可见 |
| 引言 | 从非母语发音适配进入；删去过长的通用 SSL 历史与引文罗列 | 让实际问题先出现；篇幅留给缺口、实验对象和关键证据 |
| 相关工作 | 融入引言，分别处理 probe、representation fine-tuning、Delta fusion、task arithmetic、learnable selection | 按与研究问题的关系组织，避免把所有 representation 方法视为已解决同一问题 |
| 方法 | 先定义 paired features 与坐标回退，再解释 utility 和对照 | 每个步骤回应一个测量问题，而非照搬实现流水线 |
| 实验设置 | 新写完整数据、训练、校准、缓存、指标与随机性说明 | 原稿不足以审查配对、预算和评估公平性 |
| 主结果 | 保留表 → 删除图 → 更完整控制表 | 分别回答保留多少、删掉什么最有影响、排序是否跨预算可靠 |
| 讨论与结论 | 总结固定头依赖，明确 native basis、单次训练、单语料的范围 | 不用未做的机制实验支撑声学/语义解释，也不外推儿童 ASR |

六篇实际存在的参考 PDF 均已阅读；重点三篇的具体原文结构与所采用的写法见 [template_reading_notes.md](template_reading_notes.md)。缺失的 Hyper-adapter 没有冒称阅读。引用精简到与本稿问题直接相关的 12 篇，不为“学过某篇写法”而强行在正文引用。

## 4. 方法定义与数学修订

1. 用 `E0` 表示 **L2-ARCTIC 适配前** 的表示。两端源模型均已接受 LibriSpeech ASR 监督训练，不能使 `pretrained` 被误读为仅 SSL。
2. 明确 NoShift 使用适配后的 head，因此不是原始 recognizer 的 ASR baseline，也不能把该 gap 一概叫作训练获得的收益。
3. 按实际代码补入每句 target-length normalization、valid-frame 聚合和 `abs` 的位置。原稿的全句共享 gate 梯度有帧间求和；代码却先逐帧取绝对值。新稿分开写 signed 一阶近似与实际非负 saliency score，避免把二者写成恒等式。
4. 删除未求解的最小支持集优化目标。现有方法是一次排序；没有组合搜索、最优性证明或预先设定的行为容差。
5. “必要”改为组级删除造成的功能影响；“充分”改为在保留源表示底座时保留多少 WER 表现。删去把联合删除外推为每个坐标独立必要的语言。
6. 保留完整 1024 维、相同 head 和编码器；没有参数压缩、FLOPs 或延迟收益主张。补入随机重缩放的 `d/K` 和期望位移解释。

## 5. 实验解释的收紧

- D0 在 50% 时 rescaled random 的平均 WER 为 11.95%，好于 DADS 12.94%；25% 时 magnitude 和 random 也优于 DADS。现在作为重要结果进入正文，否定“DADS 全预算领先”。
- 删低分 25% 与保留高分 75% 是同一 mask，不再暗示是两项独立证据。
- Utility 与 magnitude 相关在 W1/W2 接近 0.77/0.78，不能笼统称弱相关；正文用匹配预算 WER 检验实际作用。
- `train_utility` 是训练内校准，不写成从 encoder adaptation 留出的独立数据。正常化 transcript 无交集，但 prompt IDs 有交集；不写严格 unseen-prompt/unseen-accent。
- “三次随机种子”仅用于随机 masks 和校准子集，适配 checkpoint 各只有一个 seed。不存在可移植到新主结果的 bootstrap CI。
- W1/W2 与 D0 损失、batch、scheduler、预算均不同；不以不同模型绝对 WER 的排序判断 SSL objective 优劣。
- manifest 帧数与重采样元数据不同步，已由实际音频 header 核实；正文不使用错误小时数，未改原始文件。

## 6. 格式核对与尚需作者填写的内容

现有目录明确为 ICASSP 2027，因而沿用该届 `spconf.sty` / `IEEEbib.bst`，没有改投其他届次。官方 [Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php) 与 [Author Guidelines](https://2027.ieeeicassp.org/author-guidelines/) 已查阅（2026-09-06）。按较保守的共同规则排为 4 页技术正文 + 第 5 页仅参考文献；Letter 双栏，正文和图表不使用低于 9 pt 的文字，无页码。摘要为 100–150 词范围，关键词 5 个。

官方采用 non-blind review，原模板的 `Anonymous Authors` 已替换为可见 **TODO: Author names / Affiliations and contact information**，因为本项目无法确认作者名单和单位。当前 PDF 是完整审阅初稿，作者信息与两项证据 TODO 保留可见，未伪装成最终上传文件。

官方指南还要求对超出文字润色的 AI 生成内容作披露。本次确实包含全文草拟，所以记录这一实际适用项；没有凭空扩大为额外审批流程。供作者定稿时采用的真实措辞草案如下，只有作者完成相应核对后才使用第二句：

> OpenAI Codex was used to assist with drafting and revising the manuscript. The authors verified the experimental results, references, and final text and take responsibility for the paper's content.

**TODO-DISCLOSURE：** 作者核对最终实际使用范围并决定准确表述；按届时官方要求置于 acknowledgment，计入正文允许页数。本文尚未添加这段最终声明，避免代作者声称已完成核验。本稿作者与其后投稿者的核对责任不能由一次自动自审替代。

## 7. 完成的自审修复

已经按新颖性、技术可信度、实验充分性、清晰度和结论边界逐项审读并修改：撤下不成立的公开配对；重写核心论点和摘要；纳入负向控制；更正 Taylor 公式；补全设置；统一指标；去除未经证明的机制、最小支持和压缩措辞；修复源码中的长模型 ID、公式和表格越界。

剩余的实证问题没有以润色掩盖，正文保留显式 TODO，审稿报告按优先级列出其检验目的。没有为填补这些空缺擅自启动实验。
