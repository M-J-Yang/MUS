# 实验与主张对应表

核对日期：2026-09-06。项目根目录：`/data/zb/ymj/MUS`。下文实验路径均相对该目录。所有数字来自已有结果文件；本次只读取数据、配置、代码与日志，重新计算汇总和作图，没有运行模型推理、训练或新增消融。可机器读取的核对结果及主要输入文件 SHA-256 见 [artifact_audit.json](artifact_audit.json)，检查程序见 [audit_existing_evidence.py](audit_existing_evidence.py)。

## 1. 最终研究问题与证据范围

**研究问题：一个已经适配的识别器，在多大程度上依赖其最终层表示的不同适配变化？**

本文支持的核心表述是：

> In the evaluated L2-ARCTIC recognizers, the frozen adapted CTC head depends unevenly on groups of final-layer representation shifts in the encoder's native coordinate basis.

“Non-uniform”具体指：**同一个模型中，按同一个训练集校准分数排序，回退同样数量的高分与低分坐标，对测试 WER 的影响很不相同**。单位是最终层 1024 个坐标中的坐标组；不是层、时间、说话人、音素、参数或独立语义因素。比较条件固定原适配头、解码器、输入、完整特征维度。结论覆盖 W1、W2、D0 三个本地 checkpoint；每个只有一个适配训练 seed。

这比候选句中的“useful but highly non-uniform task-relevant information”更准确：“information”容易混淆可被新探针学到的信息与既有头的实际依赖。“Highly”也不是独立测得的集中度指标。本文保留理解适配的方向，以有限的功能干预结果作答，不将 DADS 包装为全预算最优的特征选择方法。

## 2. 主证据的文件注册表

### W1 / W2：本地 wav2vec 2.0，released Fold 1 / 2

将下列路径中的 `{f}` 分别替换为 `1`、`2`：

| 对象 | 路径 |
|---|---|
| 共同源模型 | `checkpoints/wav2vec2_large_960h_pretrained/` |
| 适配模型、配置与训练记录 | `artifacts/runs/l2_arctic_official_ut8/fold{f}/w2v2_large_960h_supcon_local_replica_full_gc/` 中的 `config.json`、`training_summary.json`、`replica_metadata.json` |
| 冻结头主结果 | `artifacts/results/l2_arctic_official_ut8/fold{f}/w2v2_large_960h_oracle_shift_local_replica_core/core_metrics.json` |
| 分数与排序 | `artifacts/results/l2_arctic_official_ut8/fold{f}/w2v2_large_960h_oracle_shift_local_replica_utility/utility_shift_taylor_stats.json`、`utility_shift_taylor.pt`、`utility_shift_taylor_ranking.pt` |
| 特征与来源报告 | `artifacts/features/l2_arctic_official_ut8/fold{f}/w2v2_large_960h_oracle_shift_local_replica/{dev,test,train_utility}/`；读取其中 `extraction_report.json` |
| 划分 | `manifests/l2_arctic_official_ut8/fold{f}/{train,dev,test,train_utility,train_teacher}.jsonl` |

目录名包含 `oracle` 不代表它使用公开 Fold 0 checkpoint。这里通过 **core JSON 的 checkpoint/cache/ranking 字段 → extraction report → training_summary 的 pretrained_path → 两端 config** 核对，不能仅按名字归类。训练记录的 `protocol` 字符串仍含 Fold 0，但实际 `train_manifest`、`dev_manifest`、`test_manifest`、样本数和缓存均分别指向 Fold 1 / 2。

### D0：本地 data2vec，released Fold 0

| 对象 | 路径 |
|---|---|
| 源模型 | `checkpoints/data2vec_audio_large_960h/` |
| 适配模型与训练记录 | `artifacts/runs/l2_arctic_official_ut8/fold0/data2vec_audio_large_960h_ctc_formal_b4/` |
| 冻结头主结果 | `artifacts/results/l2_arctic_official_ut8/fold0/data2vec_large_960h_shift_core/core_metrics.json` |
| 完整控制组 | `artifacts/results/l2_arctic_official_ut8/fold0/data2vec_large_960h_shift_empirical_package/metrics.json` |
| 分数与排序 | `artifacts/results/l2_arctic_official_ut8/fold0/data2vec_large_960h_shift_utility/utility_shift_taylor_stats.json` 及同目录 `.pt` 文件 |
| 特征缓存 | `artifacts/features/l2_arctic_official_ut8/fold0/data2vec_large_960h_shift/{dev,test,train_utility}/` |
| 校准规模检查 | `artifacts/results/l2_arctic_official_ut8/fold0/data2vec_calibration_size/calibration_size_metrics.json` |
| 直接解码与缓存一致性 | `artifacts/results/l2_arctic_official_ut8/fold0/data2vec_identity/{dev,test}.json` |

三个主模型的架构配置与所记录初始化一致，缓存报告的两端模型路径也一致。这个结论是基于本地运行溯源记录与配置核对；不是仅根据重构恒等式作出的推断。

## 3. 正文每项主张的直接证据

下表为 **test WER (%)**，四舍五入至小数点后 6 位。原文件 `.wer` 是比例，正文乘 100。差值单位为百分点。

| 条件 / JSON key | W1 | W2 | D0 |
|---|---:|---:|---:|
| Full / `full` | 15.144952 | 14.455811 | 11.388509 |
| NoShift / `no_shift` | 20.326558 | 19.151434 | 13.765390 |
| DADS 75% / `utility75` | 15.361546 | 14.539661 | 10.961012 |
| DADS 50% / `utility50` | 16.011330 | 15.579406 | 12.944596 |
| Magnitude 50% / `magnitude50` | 16.627791 | 16.484991 | 13.748290 |
| Revert high 25% / `drop_best25` | 22.242586 | 24.316619 | 14.637483 |
| Revert low 25% / `drop_worst25` | 15.361546 | 14.539661 | 10.961012 |

精确索引：每个 `core_metrics.json` 的 `splits.test.conditions.<key>.wer`。Dev 对应 `splits.dev.conditions`，完整保存在机器核对文件中。

| 主张 | 支持它的比较 | 证据性质与正文边界 | 主要缺口 |
|---|---|---|---|
| 完整表示变化对既有适配头有用 | 三个模型 Full 均优于 NoShift | 直接输入干预；不能称为原始识别器到适配识别器的净提升，NoShift 使用的是适配头 | 源头与适配头的交叉比较尚未系统完成 |
| 依赖沿坐标组不均匀 | 高分 25% 回退后相对 Full 为 +7.10、+9.86、+3.25 pp；低分 25% 为 +0.22、+0.08、−0.43 pp | 固定读出的组级干预，本文最强证据；不能推出各单坐标分别必要或具有特定音素功能 | 独立训练重复、说话人层面的不确定性、其他排序的 deletion 对照 |
| 75% DADS 保留大部分识别性能 | Table 2 的 Full 与 DADS75 | 小绝对 WER 差异，不写统计等效、无损或最小充分支持 | 尚无等效界值与有效配对置信区间 |
| 50% 时 DADS 优于 magnitude | 三模型分别低 0.62、0.91、0.80 pp | 同预算、同维度、同头的特征选择比较 | 不代表优于全部控制组 |
| 排序质量与保留预算有关 | D0 的 Table 3；50% Random+Rescale 优于 DADS，25% magnitude/random 也优于 DADS | 现有控制组的直接反例，否定全预算优势 | W1/W2 缺这些完整控制组 |
| 缩放显著改变固定随机 mask 的数值表现 | D0 50% Random 15.08 → Random+Rescale 11.95 | 同 seed、同保留坐标，改变位移尺度；支持 scale sensitivity，未隔离完整适配机制 | DADS/Magnitude 自身的匹配缩放与统一 alpha 扫描 |
| 很小的校准集不是 D0 50% 排序劣势的充分解释 | 128 vs 1640 校准集下 DADS50 都落后于现有 rescaled random 平均值 | 校准规模敏感性检查；不能证明分数已收敛或不受训练内校准影响 | 校准独立性、更多子集、W1/W2 规模稳定性 |

**不重复计数：** `drop_worst25` 与 `utility75` 是同一 mask，WER 完全相同；图和表分别呈现它的删除与保留含义，不是两项独立验证。W1、W2 是不同划分下的两次适配，不是同一划分的重复 seed，也不能报告为三折 wav2vec 平均值。

## 4. D0 控制组与校准规模

完整控制文件索引：`splits.test.retention.methods.<method>.<25|50|75>`；`Utility` 是 DADS 的日志名称。

| 排序 / 控制 | 25% | 50% | 75% |
|---|---:|---:|---:|
| DADS | 15.20 | 12.94 | 10.96 |
| Magnitude | 13.30 | 13.75 | 11.10 |
| Gradient | 15.41 | 13.68 | 11.11 |
| Random | 13.00 ± 0.03 | 15.08 ± 0.31 | 11.35 ± 0.08 |
| Random + rescale | 12.58 ± 0.37 | 11.95 ± 0.33 | 11.58 ± 0.07 |

这里 ± 是同一 checkpoint、三个 mask seeds `1337, 2027, 31415` 的 **sample SD**，不是训练重复或置信区间。程序逐项用 `statistics.mean/stdev` 重算，吻合日志。50% rescaled random 三次分别为 11.816005、11.713406、12.329001%；均低于 DADS 的 12.944596%。Dev 上也存在这个排序反例，未利用 test 选择一个最好的随机 seed 放入正文。

`Random+Rescale` 把所选位移乘 `d/K`，补偿的是随机 mask 的期望位移；并不精确匹配每个样本的总范数，且坐标会越过完整适配值。因此该控制可反驳 DADS 的普遍性能优势，却不足以证明唯一原因是“选择保留的能量不足”。

校准规模既有结果，WER 为三个子集的均值 ± sample SD：

| 校准 utterances | DADS50 | DADS75 | top-25% 与全量排序的平均重叠比例 |
|---|---:|---:|---:|
| 128 | 13.1099 ± 0.1306 | 11.1149 ± 0.1040 | 0.8190 |
| 256 | 12.9389 ± 0.1454 | 10.9895 ± 0.0197 | 0.8750 |
| 512 | 13.1042 ± 0.0494 | 11.0123 ± 0.0452 | 0.9128 |
| 1024 | 12.9731 ± 0.0395 | 10.9724 ± 0.0601 | 0.9492 |
| 1640 | 12.9446 | 10.9610 | 1.0000 |

1640 行重复使用全量集合，三个 seed 下结果相同，不能当作三个独立校准样本。正文仅用 128 与全量对照，避免从有限规模实验外推“少量标签足够”这一更强结论。

Magnitude 与 Utility 的 Spearman 相关为 W1 0.767958、W2 0.781564、D0 0.359629。因此不能统一写成“二者弱相关”；分数相关性也不能替代功能干预。

## 5. 原稿公开 Fold 0 的决定性溯源问题

原稿使用 `artifacts/oracles/wav2vec2-large-l2-arctic-supcon-repeated-8fold-0`，其源端缓存却使用 `checkpoints/wav2vec2_large_960h_pretrained`。直接读取二者 `config.json` 得到：

| 架构字段 | 缓存源端 | 公开适配端 |
|---|---|---|
| `conv_bias` | false | true |
| `feat_extract_norm` | group | layer |
| `do_stable_layer_norm` | false | true |
| `hidden_size` / layers | 1024 / 24 | 1024 / 24 |

这不是本文声明的同一源模型经微调得到的成对 encoder。还不能凭上述字段指定它真正的源 checkpoint；本次没有猜测或替换来源。两端维度相同只使减法可计算，`E0 + (Eft − E0) = Eft` 也只是代数恒等式。

受影响的既有材料包括：

- `artifacts/results/l2_arctic_official_ut8/fold0/w2v2_large_960h_oracle_shift/`；
- `artifacts/results/l2_arctic_official_ut8/fold0/w2v2_large_960h_oracle_shift_empirical_package/metrics.json`；
- `artifacts/results/l2_arctic_official_ut8/fold0/w2v2_large_960h_oracle_shift_bootstrap/paired_bootstrap.json`。

原稿中的 Full 10.499%、NoShift 118.912%、DADS50 11.713%、高分删除后 131.190% 等数值保留在原文件，但 **不再进入正文主结果**。这些数值不能支撑本文的 fine-tuning-shift 解释，也不能用原公开 Fold 0 bootstrap 为 W1/W2/D0 提供显著性。原稿的“保留 98.9% adaptation gain”也一并删除。

## 6. 数据划分、校准与泄漏核查

逐行将本地 manifests 的 speaker、音频 basename、transcript 与 `artifacts/protocol_audit/official_l2_arctic_8fold_{0,1,2}/{train,val,test}.csv` 投影比较：全部对应；不能据旧 split 名称推断 unseen accent 或严格 unseen prompt。

| Fold | Train / Dev / Test | Utility / Teacher | Train–Dev prompt 重叠 | Train–Test prompt 重叠 | Dev–Test prompt 重叠 |
|---|---|---|---:|---:|---:|
| 0 | 16312 / 1867 / 675 | 1640 / 14672 | 31 | 17 | 1 |
| 1 | 16070 / 1979 / 686 | 1615 / 14455 | 29 | 19 | 2 |
| 2 | 15953 / 2086 / 691 | 1600 / 14353 | 29 | 19 | 1 |

- 每个 fold 内，三个主 split 的 utt_id 互不重叠；用实际 `src/usde/text.py` normalizer 得到的完整 transcript 字符串也互不重叠。
- Train 与 Dev 共享 18 位说话人，Test 的 6 位说话人独立。各 split 覆盖相同六个 L1 组，不能称 unseen-accent evaluation。
- canonical prompt ID 存在上述交集，文字差异可能来自录音转写差异。本文仅如实陈述，不以“无完全相同字符串”推导严格提示内容独立。
- `train_utility` 按每位说话人的 utt_id 排序后每十条取一条。它属于实际训练使用的整个 `train.jsonl`；`train_teacher` 名称不代表模型只在 teacher 子集训练。正文明确为训练内校准，不能写成 held-out from adaptation。
- 排序脚本不用 dev/test 标签计算分数；但项目已多次查看 test 上多预算、多策略结果。**不能声称 test 完全未参与研究迭代或只用过一次。** 本稿列出已记录条件，没有把 test 最优点另行宣称为 dev 选定配置。新增确认性比较应先锁定规则。

音频核对另发现：manifest 的 `frames` 仍对应原采样率下的长度，而 `sample_rate` 已写 16000；直接相除会高估时长。读取实际 waveform header 后全部为 16 kHz，训练时长分别为 Fold 0 **16.311 h**、Fold 1 **16.396 h**、Fold 2 **15.895 h**；test 为 0.687、0.653、0.712 h。未修改原 manifests。论文使用确认过的 utterance 数，未引用错误小时数。

## 7. 训练预算与评估实现

检查入口：`train_large_supcon.py`、`src/usde/supcon.py`、`scripts/run_l2_arctic_official_local_replica_memory_safe.sh`；具体取值以每个 checkpoint 的 `training_summary.json`、`replica_metadata.json` 为准。

| 项目 | W1 / W2 | D0 |
|---|---|---|
| 适配损失 | CTC + SupCon | CTC |
| head warm-up | 1 epoch, LR 3e-6, fp32, batch 4/GPU × 2 GPU | 继承 `data2vec_audio_large_960h_ctc_formal/head_warmup/checkpoint-4078`；1 epoch, LR 3e-6, batch 4 |
| joint | LR 1e-5, batch 24/GPU × 2, bf16 | LR 1e-5, batch 4 × 1 GPU, bf16 |
| scheduler / weight decay | linear / 0 | cosine / 0.01；warmup ratio 0.1 |
| SupCon | 6 transcript groups × 4 samples/GPU，lambda .05，temperature .1，projection 256，ramp .1 | 不适用 |
| 最大 epoch / patience | 40 / 5 | 40 / 5 |
| 被选择的 joint step | W1 2204；W2 1406 | 16312 |
| 最后记录的 joint step | W1 2394；W2 1596 | 36702 |
| 训练 seed | 各 1337 一次 | 1337 一次 |

W1/W2 的 transcript-grouped sampler 不等同于普通 full-dataset epoch：singleton groups 不参与该 joint 采样，每组抽取四条。不同 recipe 的名义 epoch 与步数不构成公平训练预算。因此正文不将 W1/W2 与 D0 的绝对 WER 差异归因于 SSL objective。

D0 的 `head_warmup_skipped: true` 表示复用已完成的 warm-up checkpoint，不能误写为完全没有 warm-up。W1 的 warm-up best metric 为 null，不据此虚构头阶段的最优 WER。

W1/W2 训练代码的音频上限为 10 s、标签上限为 128 字符；缓存是完整 utterance。实际头信息显示每个 W1/W2 train 有 2 条超过 10 s，dev/test 没有；所有 split 的规范文本均未超过 128 字符。**这不能解释以下 dev WER 差异**：W1 训练最佳记录 14.836579% vs 缓存 Full 14.646295%；W2 14.656860% vs 14.778560%。精度、批处理和评估路径仍需统一核对，未把差异归因于未经隔离的原因。

D0 训练汇总 test WER 为 11.354309%，缓存/直接恒等评估为 11.388509%。正文只使用后一路径，未混用较低值。D0 已有直接输出与缓存输出的 decode match rate = 1.0；W1/W2 core 只验证缓存代数重构及 logits，尚缺同等的 direct-vs-cache 解码记录。

方法实现核对：

- `src/usde/shift.py`：CTC `reduction='none'`、`zero_infinity=True`，每句除以 target length；反向传播的是归一化逐句 loss 之和；`abs(delta * gradient)` 在 frame 聚合前计算。
- `utility/compute_l2_shift_taylor_utility.py`：只计 valid frames，float64 累积分数，再按分数得到固定排序；不是 forced alignment，也不是仅 nonblank frame 的版本。
- `scripts/evaluate_official_replica_core.py` 与 `scripts/evaluate_official_shift_package.py`：固定原 checkpoint 的 `lm_head`，`E0 + mask * delta`，三种 random seeds，重缩放倍数 `d/K`。
- `src/usde/metrics.py`：corpus-level 总 word edits / 总 reference words。每个模型内所有条件共享评估路径；不平均 utterance WER。
- 旧 YAML、早期重新训练融合头/CTC 头的结果不能与这组实验合并；那些结果回答可学习的下游性能，不能证明固定头下的依赖。

## 8. 已确认与尚未确认的界线

已确认：三个有效本地配对的记录来源；split 样本数与 CSV 对应；主 WER 与所有图表；固定 head 和完整 1024 维输入；Taylor 聚合公式；D0 random SD；训练内校准；没有新增实验。

尚未确认 / TODO：公开 Fold 0 真正源模型；W1/W2 direct-vs-cache 等价解码及训练评估差异；有效配对的 speaker-aware uncertainty；同一 split 多训练 seed；W1/W2 的 gradient/random/rescale 完整控制；独立校准集；音素、层、旋转基和头兼容性解释的区分。正文没有把这些内容写成已完成。
