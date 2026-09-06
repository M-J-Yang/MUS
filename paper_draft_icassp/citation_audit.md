# 引用身份与句子支持范围

核对日期：2026-09-06。优先复用原 `dads_refs.bib` 中与最终论点直接相关的条目，另补充数据集、CTC、实际适配 recipe 和两项最相关的 representation 工作。最终正文 12 个 citation keys 均在 [references.bib](references.bib) 中。没有把目录推荐清单当作已读全文，没有从未核实的二手段落生成引用。

| Key | 核实的身份与原始来源 | 本稿使用范围 |
|---|---|---|
| `baevski2020wav2vec` | Baevski, Zhou, Mohamed, Auli, NeurIPS 2020；[会议原始页面](https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html) | wav2vec 2.0 模型背景；不是本地 checkpoint 适配谱系的证据 |
| `baevski2022data2vec` | Baevski et al., ICML 2022, PMLR 162:1298–1312；[PMLR](https://proceedings.mlr.press/v162/baevski22a.html) | data2vec 模型身份和 SSL 背景；不支持本稿中的跨 recipe 公平性 |
| `thai2026contrastive` | Van-Phat Thai, Aradhya Dhruv, Duc-Thinh Pham, Sameer Alam；[作者 arXiv](https://arxiv.org/abs/2605.03297)，标为 accepted at Interspeech 2026；[作者代码仓库](https://github.com/thaivanphat95/robust-atc-asr) | accented ASR 的 SupCon 研究与 released split 来源；本地超参、来源配对仍以本地记录为准 |
| `pasad2021layerwise` | Ankita Pasad, Ju-Chieh Chou, Karen Livescu, ASRU 2021:914–921；[作者预印本](https://arxiv.org/abs/2107.04734)，[DOI](https://doi.org/10.1109/ASRU51503.2021.9688093) | 层次 probe/表示分析及微调变化；不声称它做过本稿的 frozen-head delta 删除 |
| `pasad2023comparative` | Ankita Pasad, Bowen Shi, Karen Livescu, ICASSP 2023；[作者预印本](https://arxiv.org/abs/2211.03929)，[作者发表列表](https://ankitapasad.github.io/publications/)，[DOI](https://doi.org/10.1109/ICASSP49357.2023.10096149) | 不同 SSL 模型的层次信息分析；避免扩展为适配机制的因果解释 |
| `niu2025factorization` | 本地 PDF 作者为 Minxue Niu 等，末位 S. Elizabeth Norred；[作者机构页面](https://www.amazon.science/publications/learning-rich-speech-representations-with-acoustic-semantic-factorization)，ICASSP 2025 | 声学/语义双分支微调与不同任务的信息可用性；已阅读全文；网页中 Sandy/Liz 的署名变体不混入 PDF 作者名 |
| `ilharco2023task` | Gabriel Ilharco et al., ICLR 2023；[作者预印本](https://arxiv.org/abs/2212.04089) | 参数差分可用于模型行为编辑；本稿显式说明 representation delta 与参数 task vector 不同 |
| `wang2026mind` | Zilai Wang, Natarajan Balaji Shankar, Kaiyuan Zhang, Zihan Wang, Abeer Alwan；[作者预印本](https://arxiv.org/abs/2601.20142)，accepted at ICASSP 2026；本地 PDF 全文核对 | 儿童 ASR 中完整 delta 的可学习融合效用；不把 fusion head 的结果当作原头依赖证据 |
| `chiu2024selection` | Sheng-Chieh Chiu, Chia-Hua Wu, Jih-Kang Hsieh, Yu Tsao, Hsin-Min Wang, Interspeech 2024:3914–3918；[ISCA Archive](https://www.isca-archive.org/interspeech_2024/chiu24_interspeech.html) | 可学习的 layer/dimension-wise source selection 与 fusion；说明其下游学习设置与本稿不同 |
| `molchanov2017pruning` | Pavlo Molchanov, Stephen Tyree, Tero Karras, Timo Aila, Jan Kautz, ICLR 2017；[作者预印本](https://arxiv.org/abs/1611.06440) | Taylor/activation-gradient saliency 的已有基础；不把本稿具体的逐帧 absolute 聚合归为它的精确原公式 |
| `graves2006ctc` | Alex Graves, Santiago Fernández, Faustino Gomez, Jürgen Schmidhuber, ICML 2006:369–376；[作者全文](https://www.cs.toronto.edu/~graves/icml_2006.pdf) | CTC 在有效对齐路径上求和及损失定义；本稿的长度归一化来自实现而非援引原文默认值 |
| `zhao2018l2arctic` | Guanlong Zhao et al., Interspeech 2018:2783–2787；[ISCA Archive](https://www.isca-archive.org/interspeech_2018/zhao18b_interspeech.html) | L2-ARCTIC 语料身份；论文初版规模不能替代本项目 24-speaker manifests 的核对 |

已读的其他四份写作参考用于组织论证，未必与正文形成必要的学术归因，因此没有为增加引用数而全部加入 bibliography。`references.bib` 没有凭空补齐未从原始来源确认的 Niu 论文页码/DOI；已确认作者、题名、会议和年份即可定位全文。arXiv 两项使用实际标识和已核实的 accepted 注记，未虚构会议页码。

公式和本地实验结果以代码与 artifacts 为依据。文献可支持研究动机和已有进展，不能为本项目缺少的 baseline、训练重复、显著性或跨域结果提供替代证据。
