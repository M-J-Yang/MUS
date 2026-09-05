# Method rewrite and literature audit

**Date:** 2026-09-03 (Asia/Shanghai)  
**Status:** complete for the current ICLR draft

## Objective

Recast the manuscript around one scientific question: which parts of a
fine-tuning-induced representation shift are used by an already adapted ASR
decision rule? The Method is now organized as

```text
paired representation change → frozen-readout counterfactual
→ functional-support objective → Taylor estimator → sufficiency/necessity tests
```

This removes checkpoint, GPU, evaluator, and identity-gate details from the
methodological story.

## Literature used for writing and positioning

The introduction and method organization were checked against official
conference versions of the following LLM-pruning papers:

- SparseGPT (ICML 2023): one-shot reconstruction and a compact overview before
  the detailed solver.
- LLM-Pruner (NeurIPS 2023): capability/cost tension, explicit constraints,
  then the structural selection principle.
- WANDA (ICLR 2024): a concrete criterion motivated by the failure of generic
  magnitude pruning, followed by a direct no-retraining intervention.
- SliceGPT and Sheared LLaMA (ICLR 2024): post-training/target-shape
  formulations that define the object being reduced before describing the
  pruning procedure.
- SLEB (ICML 2024): an observed redundancy pattern motivates the pruning unit.
- SlimGPT (NeurIPS 2024): the pruning objective is separated from the
  layer-wise allocation and error-accumulation analysis.

The manuscript uses this rhetorical pattern, but does not claim that DADS is a
parameter-pruning method. Its object is an input-conditioned representation
Delta, and its criterion is derived from the frozen fine-tuned CTC objective.

## Manuscript changes

- Replaced the four Method subsections with: paired shifts, frozen-readout
  interventions and functional support, DADS, and retention/deletion tests.
- Added the formal support objective
  $S^\star_\epsilon=\arg\min_{S\subseteq[d]}|S|$ subject to a frozen-readout
  behavioral tolerance.
- Defined a continuous coordinate gate and separated the global gate derivative
  from the implemented non-cancelling framewise score.
- Matched the score to `utility/compute_l2_shift_taylor_utility.py`:
  $U_i=N_u^{-1}\sum_{x,y,t}|\Delta_{t,i}\,\partial\mathcal{L}/\partial
  \Delta_{t,i}|$, with transcript-length-normalized per-example CTC loss and
  padded frames excluded.
- Framed retention as a sufficiency test and DropBest/DropWorst as a necessity
  test.
- Added the LLM-pruning comparison paragraph to the Introduction and cited the
  existing BibTeX entries.
- Removed the stale retrained-head comparison table from the mainline.

## Protocol reconciliation

The official Fold0 artifacts are now the only mainline numbers in the paper:

- train/dev/test: 16,312 / 1,867 / 675;
- No shift / Full shift: 118.912% / 10.499% test WER;
- Taylor-50: 11.713% test WER, retaining 98.9% of the full adaptation gain;
- Taylor-75: 10.585% test WER, statistically indistinguishable from Full under
  the frozen paired bootstrap;
- Magnitude-50: 16.638% test WER;
- DropBest-25 / DropWorst-25: 131.190% / 10.585% test WER.

The older 22.75 / 16.48 / 17.51 / 83.5% result set was removed from the
manuscript because it belongs to a different, stale protocol.

## Files and verification

Changed manuscript assets:

- `iclr2027/iclr2027_conference.tex`
- `scripts/plot_formal_shift_pruning.py`
- `iclr2027/figures/formal_shift_pruning.pdf`

Verification completed:

- Python syntax check passed for the figure script.
- `git diff --check` passed.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error` passed and produced a
  7-page `iclr2027/iclr2027_conference.pdf` with resolved citations.
- The rendered Figure 1 shows the official Fold0 values and the complete
  measure–rank–intervene loop.

## Conclusion

The paper now has one coherent claim: fine-tuning shifts are not functionally
uniform, and a CTC-loss-aligned ranking can identify a compact subset of the
shift used by the original adapted readout. The remaining high-value follow-up
is empirical rather than rhetorical: preserve the fixed protocol while adding
the planned held-out replication and supplementary analyses.
