# 2026-09-09 — ICASSP Introduction story rewrite

## Objective

Replace the ICASSP manuscript Introduction with a story-first narrative centered
on decision-aligned utility of fine-tuning-induced representation shifts. The
new structure follows:

```text
SSL adaptation → transferable model/representation differences
→ existing methods treat shifts as a whole
→ representation-to-decision mismatch
→ two supporting observations
→ DADS
→ main L2-ARCTIC result
→ three memorable contributions
```

The rewrite deliberately preserves the full argument and does not optimize for
the current page limit.

## Reference and writing basis

- `writing_templates/01_mind_the_shift_delta_ssl_child_asr.pdf`
- `references/reading_library.md` and its checked primary-source PDFs
- `docs/method_rewrite_20260903.md`
- Task arithmetic, TIES-Merging, DARE, ASR task-vector/model-merging, and
  accent-steering references already present in the manuscript bibliography

The main stylistic lesson taken from *Mind the Shift* is the compact chain of
problem, representation difference, simple operation, result, and interpretation.
LLM pruning references are used for the selection-principle framing, not as a
claim that DADS operates on parameters or matches their experimental scale.

## Manuscript change

Changed `ICASSP2027_Paper_Templates/Template.tex` Introduction to:

- establish SSL adaptation and representation reshaping;
- position parameter-space and representation-space differences;
- state the representation-to-decision boundary as the central insight;
- give the two observations motivating decision-aligned selection;
- introduce DADS before discussing implementation details;
- foreground the 50% retention result and counterfactual reversion;
- express the contributions as decision-critical structure, decision-aligned
  pruning, and counterfactual dependence analysis.

The full-width overview figure remains in the Introduction and is referenced at
the point where the method story is needed.

## Follow-up — Fig. 1 removal

The single-column motivation figure (`figures/dads_motivation.pdf`) was removed
from the manuscript at the author's request. Its explanatory sentence was
retained in prose, while the full-width DADS overview figure remains. The
motivation PDF source was not deleted and remains recoverable as a local asset.

## Follow-up — contribution rewrite

The contribution list was rewritten from a neutral inventory of observation,
method, and analysis into a claim-driven progression:

1. decision-dependent redundancy is the empirical finding;
2. DADS is the selection principle that follows from that finding;
3. frozen-head retention/reversion supplies functional evidence and the main
   retention result.

This mirrors the stronger contribution style in compact LLM systems papers:
phenomenon or asymmetry first, design principle second, concrete system-level
effect third.

The final contribution wording is intentionally title-free and highly
compressed: phenomenon, design principle, and functional payoff are expressed
as three direct claims without repeating the main WER numbers.

## Verification

```text
bash ICASSP2027_Paper_Templates/build.sh
```

Completed successfully. BibTeX and all LaTeX passes completed, with no
undefined citation or reference errors. The long-form manuscript compiles to
six pages. A small overfull box remains for later layout work; no text was
removed to address it.

No new experiment was run. The reported WER and gain-retention values are
existing manuscript results, not new evidence produced by this rewrite.
