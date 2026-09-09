# ICASSP 2027 论文主目录

本目录的 [Template.tex](Template.tex) 是论文主源文件，[Template.pdf](Template.pdf) 是编译稿。已启用模板原生九号字，正文四页，参考文献从第五页开始。

直接编译：

```bash
bash /data/zb/ymj/MUS/ICASSP2027_Paper_Templates/build.sh
```

图表、完整文献数据与证据说明保存在 [paper_draft_icassp](../paper_draft_icassp/README.md)。其 build.sh 同步图表与文献到本目录，编译主文件，并将结果复制回交付目录。作者栏已填入 Mingjun Yang、Chaofan Yang、Jiahao Zhao、Bo Zhang、统一单位和通讯邮箱。编辑正文请修改本目录 Template.tex。

官方示例及此前论文快照已保存在 `../paper_draft_icassp/versions/v3_before_ninept_restructure/`。spconf.sty 和原 IEEEbib.bst 未修改；IEEEbib_initials.bst 使用 IEEE 名字首字母及超过六位作者时 et al. 的显示规则。
