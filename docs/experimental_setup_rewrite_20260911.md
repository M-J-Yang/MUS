# Experimental Setup 改写记录（2026-09-11）

## 目的

将论文的 Experimental Setup 从偏实验审计/日志式的写法，改为接近 *Mind the Shift* 的读者友好型实验设计说明：先交代比较问题和设计动机，再给出可复现的协议细节。

## 修改

- 修改 `ICASSP2027_Paper_Templates/Template.tex` 的 Experimental Setup。
- 将结构调整为 `Datasets`、`Adapted models` 和 `Calibration and evaluation`；去掉没有对应 Results 问题的独立 `Computation` 小节，以符合 ICASSP 五页版面目标。
- 参照 *Mind the Shift* 的“实验问题—比较设计—结果解释”逻辑，在 Setup 末尾加入实验路线，并将 Results 重排为：主 retention 效果、匹配对照、必要性反事实、native coordinate frame、scale/calibration 鲁棒性。
- 将正文和表格中的内部代号 `R0/W1/W2/D0` 改为 `Rel-W2V2-F0`、`W2V2-F1`、`W2V2-F2` 和 `D2V-F0`；同步更新删除实验图的标签。
- 保留原有数据量、模型、优化设置、随机种子、解码和 bootstrap 数字；scale/calibration 的次要结果保留在正文叙述中，避免重复表格占用一页。
- 保留 `Template.original_20260911.tex` 作为改写前快照。

另增 `ICASSP2027_Paper_Templates/figures/render_deletion_effect.py`，用于可重复生成已更新模型名称的删除实验图。

## 验证

使用 `bash ICASSP2027_Paper_Templates/build.sh` 编译成功，`qpdf --check` 通过，当前 PDF 为 5 页（正文 4 页、参考文献 1 页）。
