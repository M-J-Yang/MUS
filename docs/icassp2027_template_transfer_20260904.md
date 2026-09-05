# ICASSP 2027 模板迁移记录

**日期：** 2026-09-04（Asia/Shanghai）  
**任务：** 将当前主稿的引言和方法迁移到 `ICASSP2027_Paper_Templates`。

## 方法

以 `iclr2027/iclr2027_conference.tex` 的最新 Introduction 和 Method 为内容来源，使用官方 `spconf` 模板的双栏结构；将 ICLR 的 `\citep`/`\citet` 引用改为 IEEE/ICASSP 的数字引用 `\cite`，并保留 DADS 方法示意图。

## 修改内容

- `ICASSP2027_Paper_Templates/Template.tex`：替换模板示例正文，写入摘要、关键词、Introduction 和 Method。
- `ICASSP2027_Paper_Templates/dads_refs.bib`：加入本稿实际使用的 9 条参考文献，使模板可独立运行 BibTeX。
- `ICASSP2027_Paper_Templates/figures/formal_shift_pruning.pdf`：复制方法示意图，使模板目录可独立携带图形资源。
- `ICASSP2027_Paper_Templates/DADS_ICASSP2027.pdf`：生成迁移后的预览 PDF；原始示例 `Template.pdf` 未覆盖。

## 目的

验证当前论文叙事和 DADS 方法公式能否在 ICASSP 2027 的 `spconf` 双栏模板中直接排版，作为后续补充实验、结果和作者信息的基础。

## 验证结果

在 `ICASSP2027_Paper_Templates` 目录运行：

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/icassp2027-build Template.tex
```

编译成功，生成 4 页 Letter 尺寸 PDF；交叉引用和最终 BibTeX 引用已解析，无 LaTeX 致命错误。编译中间文件写入临时目录，未污染模板目录。

## 结论与下一步

## 2026-09-04 引言压缩更新

为满足版式要求，将 Introduction 从原约 844 个源码词压缩为约 490 词，合并背景、相关工作、研究问题、DADS、主要结果和贡献表述，移除重复解释与项目符号列表。重新编译后共 3 页，Introduction 在第 1 页结束，Method 从第 2 页开始；最终 PDF 无致命错误或未解析引用。

下一步应补充正式作者/单位信息，并决定是否把实验结果章节一并迁移后再按 ICASSP 页数限制压缩全文。
