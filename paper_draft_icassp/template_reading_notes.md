# 写作参考论文原文阅读笔记

阅读日期：2026-09-06。依据是 `writing_templates/` 中六份实际存在的 PDF，包括正文、实验表和分析段落；不是旧 README 的推荐语。全文提取文本保存在 `reading_notes/`，并查看了三篇重点参考的首页版面。下面按论证功能组织笔记，不把固定段落数当作写作规则。

## 1. Mind the Shift: Using Delta SSL Embeddings to Enhance Child ASR

文件：[原 PDF](../writing_templates/01_mind_the_shift_delta_ssl_child_asr.pdf)。重点：Introduction、Delta embedding 定义、fusion strategies、Tables 1–3、CCA 与 MoE 分析。

**问题如何建立。** 文章从儿童与成人 ASR 的具体差距切入，交代儿童声学和语言差异；继而讨论大模型与开放 SSL 模型的取舍。不同 SSL encoder 的互补性引出融合，参数差分的已有研究再为表示差分提供动机。最后才定义 Delta embedding。它不是从“我们想减两个向量”出发寻找用途，而是让这个操作回应“如何利用已有模型的互补性”。

**实验怎样承接。** 首先确定融合策略，再比较单模型和融合结果，随后在相同融合输入维度下比较完整 fine-tuned embedding 与 delta，减少“只是增加维度”的解释空间。CCA 和 MoE 可用于讨论互补性，但没有把相关性分析变成因果机制证明。Table 3 中 full WavLM + W2V2 与相应 delta 的 WER 为 9.67 与 9.64，而低资源条件为 22.80 与 21.81；优势大小随设置变化，不能概括成所有设置都大幅改善。

**对本稿的实际影响。** 保留“完整差分有用 → 其内部依赖怎样分布”的问题衔接；清楚说明该论文训练新的融合头，而本稿干预原有冻结头。不能把它的儿童语音结果写成当前项目的数据，也不能以它已经验证了 Delta 信息为由跳过本稿的 checkpoint lineage 核对。本稿使用它作为最直接的研究起点，而不是把同一差分重新当作新方法。

## 2. Heterogeneous Self-Supervised Acoustic Pre-Training With Local Constraints

文件：[原 PDF](../writing_templates/02_heterogeneous_ssl_acoustic_pretraining_local_constraints.pdf)，本地为 arXiv v2，首页日期 2025-09-08。重点：Introduction、Problem Formulation、Optimization、算法与多域/多语言实验。

**问题如何建立。** 文章从来源异质的无标注语音出发，指出最小化混合数据的平均损失不自动保证每个来源适合后续适配。提出的 local constraints 不是孤立正则项：它们直接把“从共同初始化经 K 步后适合每个来源”写入问题定义。接着用双层问题、一阶近似和伪代码解释如何实现。

**证据怎样组织。** 多域和多语言是两个验证场景，表格按 downstream 条件拆分；K=1/K=3 及不同预训练方式检验设计选择，而不仅给一个平均 WER。更大的 K 并非每项都更好。额外 inner updates 也意味着不同优化成本，读者不能据此自动得到“机制已经完全隔离”或“梯度对齐已被实验证实”。

**对本稿的实际影响。** 方法首先定义“固定哪个对象、回退什么、测量什么”，再给 Taylor 分数，避免把缓存、排序、解码程序逐条堆成方法章节。本文没有与之对应的优化求解证据，因此删除原稿未求解的“最小充分支持集合”优化形式。该文用于学习问题—数学—实验的一致性，不强行塞入本稿相关工作以增加引用数量。

## 3. Learning Rich Speech Representations with Acoustic-Semantic Factorization

文件：[原 PDF](../writing_templates/05_learning_rich_speech_representations_acoustic_semantic_factorization.pdf)。重点：Introduction、factorization 架构、Table II 的 downstream/multitask 对照与参数量、Table III 的 upstream 任务表现。

**问题如何建立。** 引言将两个相关但不同的缺口连接起来：ASR fine-tuning 后声学信息与语义信息的取舍，以及不同任务需要不同层造成的使用复杂度；再进一步讨论信息纠缠与声学变化下的稳健性。两分支设计因此分别承担声学与语义目标，ASR 与 transcript-conditioned 音频重建给出具体训练约束。

**证据怎样承接。** 实验不是只证明 ASR 提升，而是用不同任务检查 richer representation 的含义，并用未显式 factorization 的 multitask 对照区分增加目标与结构分解。固定 downstream 表示维度为 768 有助于控制输入维度，却不意味着 upstream 参数和训练成本相等；Table II 同时列出训练与推理参数量。Table III 检查 ASR 与重建的 upstream 取舍，并指出 factorization 在 upstream 上的区别小于其 downstream 作用。结果覆盖多个任务，但不是所有任务都最好，也不构成因素已完全解耦的数学证明。

**对本稿的实际影响。** 将“表征中可用的信息”与“当前头实际使用的变化”分开。学习其让每项机制解释对应消融的写法：本稿既没有音素/声学语义探针，也没有独立因素验证，因此不会给 DADS 坐标贴上声学或语义标签。此文也进入引言作为 representation-oriented fine-tuning 的具体相关工作。

## 4. Windowed SummaryMixing: An Efficient Fine-Tuning of Self-Supervised Learning Models for Low-Resource Speech Recognition

文件：[原 PDF](../writing_templates/03_windowed_summarymixing_efficient_finetuning_ssl_low_resource_asr.pdf)。重点：windowed summary 的定义、Table 1 的替换策略/层数、Table 2 的跨语言结果、资源曲线。

**结构与证据。** 效率需求引出线性复杂度聚合，但全局 summary 丢失局部上下文，窗口设计对此作出具体修正。Table 1 比较 SM、WSM、随机初始化 attention 与 pretrained attention，并改变替换层数，为之后只替换最后两层提供依据。后续表扩展语言和 encoder，图则直接测输入长度对应的显存和运行时间。

**需要忠于原表的地方。** 例如英语某组 baseline 69.24、Ours 69.26 是例外，不能照抄正文中“consistently”式总括。其效率主张有资源测量支持；本稿虽然屏蔽了部分 shift，仍运行完整 encoder、仍输入 1024 维，不能沿用压缩或推理加速的贡献定位。

**采用的写法。** 先回答设计选择，再扩展结果，最后单独解释资源或代价；本稿用保留表、删除图和控制组表分别承担问题，不让一张大表同时承担所有贡献。

## 5. Speech Recognition Rescoring with Large Speech-Text Foundation Models

文件：[原 PDF](../writing_templates/06_speech_recognition_rescoring_speech_text_foundation_models.pdf)。重点：related work 的功能组织、模态条件概率、MWER、模型规模、OOD 与跨模态转移对照。

**结构与证据。** 相关工作按照 text rescoring、判别式训练、speech–text 模型逐步缩小范围；方法通过 token 顺序和条件概率解释不同评分方式。实验分开比较模态输入、训练目标、规模与域外表现。关于跨模态迁移的解释有对应控制：推理只使用文本的条件，以及跨模态训练资源带来的差异，并非仅凭多模态模型更好就断言机制成立。

**采用的写法。** 让替代解释在同一段结果中出现，保持结论与控制组对齐。文中 OOD 条件也有不利结果，本稿同样保留 D0 rescaled random 超过 DADS 的结果，让它参与界定贡献，而不是只在脚注中称作 limitation。

## 6. Hybrid Pruning: In-Situ Compression of Self-Supervised Speech Models for Speaker Verification and Anti-Spoofing

文件：[原 PDF](../writing_templates/07_hybrid_pruning_ssl_speech_models.pdf)。重点：引言研究问题、hard-concrete gates 与任务/稀疏约束、Sections 4.2–4.5、层级剪枝模式与泛化曲线。

**结构与证据。** 引言由部署开销与已有多阶段压缩流程的不足引出联合剪枝，并用 SV 与 anti-spoofing 的任务差异提出可检验的问题。实验依次报告精度/效率、任务和域对应的剪枝结构、适中剪枝与泛化、模型规模。图中的层级模式是已学习 mask 的观察；泛化解释还需要结合性能曲线，而非仅看哪些层留下来。

**对本稿的实际影响。** 学习按科学问题组织分析，而不按脚本执行顺序写作。该文是训练得到稀疏架构，并报告 FLOPs 和实测加速；本稿是固定已有模型后的输入回退，不能用“pruning”一词暗示同等部署收益。也不把原文某处 3.70% 与表中 3.75% 的差异照搬成未经核实的数字。

## 未阅读的缺失文件

`04_hyper_adapter_multilingual_asr_adaptation.pdf` 不存在。README 中列出了 *Hyper-adapter for Parameter-Efficient Multilingual ASR Adaptation*，但本次不声称读过它，也没有根据旧推荐清单重建其逐段结构。

## 本稿最终采用的叙事原则

具体的 non-native ASR 适配问题 → endpoint WER 的解释空缺 → 与 probe、融合和参数 task vector 区分 → 固定 head 的坐标回退 → 有效配对上的保留和删除 → 缩放控制给出的反例 → 结论只覆盖已完成的功能依赖测试。

借鉴的是问题与证据的排列方式，没有复制参考文的句子、统一段落数、夸张措辞或尚未在本项目成立的机制解释。
