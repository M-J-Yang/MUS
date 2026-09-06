# ICASSP 英文完整初稿

日期：2026-09-06。标题：**Which Fine-Tuning Changes Matter? Probing Representation Shifts in Accented Speech Recognition**。

优先阅读 [paper.pdf](paper.pdf)。它是沿用现有 ICASSP 2027 模板的完整 5 页稿件：前 4 页技术内容，第 5 页参考文献。原始草稿保持原状。作者信息和必须补充的关键证据使用明确 TODO；当前文件供审阅和继续投稿打磨。

## 交付内容

| 文件 | 内容 |
|---|---|
| [main.tex](main.tex) | 完整英文正文、标题、摘要、方法、设置、结果分析、讨论与结论 |
| [references.bib](references.bib) | 正文实际使用的 12 项文献 |
| [evidence_map.md](evidence_map.md) | 核心主张、精确结果路径、数值、适用范围和缺口；公开 Fold 0 的来源问题 |
| [revision_notes.md](revision_notes.md) | 相对原稿调整的依据、公式修正、格式核对与作者待填写项 |
| [reviewer_audit.md](reviewer_audit.md) | 严格自审、主要拒稿风险、按检验目的排序的补充实验 |
| [template_reading_notes.md](template_reading_notes.md) | 六篇本地 PDF 的原文阅读和叙事分析，明确缺失的第 4 篇未读 |
| [citation_audit.md](citation_audit.md) | 引用身份、原始来源和各引用可支持的句子范围 |
| [artifact_audit.json](artifact_audit.json) | 配对来源、数据核查、完整精度结果、随机 mask 汇总、输入 SHA-256 |
| [build_validation.json](build_validation.json) | 页数、引用、字号记录与编译问题检查 |
| `figures/`、`tables/` | 从已有结果自动生成的主图与两张结果表 |
| `reading_notes/` | 六份本地参考 PDF 的全文提取文本，仅作阅读记录 |

## 重要的证据修正

原稿公开 wav2vec Fold 0 的缓存源模型与适配模型架构不一致，不能作为同源微调差分证据。新稿只使用溯源成立的本地 W1、W2、D0。核心结论是固定适配头对最终层不同变化组的依赖不均匀；不是 DADS 在所有预算最优，也不是已经解释了音素或口音适配机制。

## 重新构建

在项目中直接编译：

```bash
bash /data/zb/ymj/MUS/paper_draft_icassp/build.sh
python3 /data/zb/ymj/MUS/paper_draft_icassp/validate_build.py
```

构建脚本优先使用项目已有 `.texlive/bin/x86_64-linux`，否则使用 PATH 上的 `pdflatex` 和 `bibtex`；编译中间文件仅写入本目录 `build/`。

重新读取现有结果并更新图表（不运行模型）：

```bash
/home/zbzb/.conda/envs/py311/bin/python /data/zb/ymj/MUS/paper_draft_icassp/render_artifacts.py
```

重新执行只读元数据核查（读取音频 header，可能需要一些时间）：

```bash
/home/zbzb/.conda/envs/py311/bin/python /data/zb/ymj/MUS/paper_draft_icassp/audit_existing_evidence.py
```

渲染需要 matplotlib/numpy；证据核查需要项目 normalizer 及 soundfile；PDF 检查使用已有 Poppler 工具。完整构建不需要重新生成图表。没有训练入口、自动新增消融或启动 GPU 的步骤。
