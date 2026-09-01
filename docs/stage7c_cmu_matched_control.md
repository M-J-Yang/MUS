# Stage 7C — CMU matched-text control

Date recorded: 2026-09-01  
Status: complete; no model was retrained.

## Question and frozen protocol

Stage 7C asks whether the task-useful shifts induced by the same L2
adaptation have different structure on accented and native speech when the
lexical content is controlled.

The primary L2 utility split is reused:

```text
manifests/arctic_step2/l2/train_utility.jsonl
  → 1792 L2 utterances, 282 shared prompt IDs
```

Each L2 row is matched by ARCTIC `prompt_id` to every available CMU native
recording for that prompt. This produces 1128 unique CMU recordings from
BDL/CLB/RMS/SLT and 7168 auditable L2×CMU pair records. No speaker matching
or CMU fine-tuning is used.

The side manifests are written by
[`scripts/prepare_stage7c_matched.py`](../scripts/prepare_stage7c_matched.py):

- `manifests/stage7c/l2.jsonl`
- `manifests/stage7c/cmu.jsonl`
- `manifests/stage7c/matched_pairs.jsonl`

The CMU canonical prompt transcript is used as the shared CTC target in both
side manifests. L2 source transcripts are retained as `source_transcript` in
the L2 manifest and as `l2_transcript` in the pair manifest. There are 26 L2
source rows with harmless prompt-annotation differences; the prompt ID and
canonical target remain fixed for the control.

For both conditions:

```text
Delta(x) = E_ft^L2(x) - E_pt(x)
U_i^v4 = E_frame,utterance[abs(Delta[t,i] * dL_CTC / dDelta[t,i])]
```

`E_ft^L2` is `checkpoints/w2v2_myst_fullfinetune`, `E_pt` is
`checkpoints/w2v2_large_lv60_pretrained`, and the reference stream is the
frozen `checkpoints/wavlm_myst_fullfinetune`. Both utility passes use the same
pure-linear L2 FullDelta CTC teacher at
`artifacts/runs/stage4/full_delta/best.pt`. CMU is never fine-tuned.

The CMU feature cache is isolated at `artifacts/features/stage7c_cmu`; the
existing L2 cache at `artifacts/features/stage3_l2` is reused. The batch
extractor is [`scripts/extract_stage7c_features.py`](../scripts/extract_stage7c_features.py).

## Results

| Condition | Utterances | Valid frames | Mean CTC loss |
|---|---:|---:|---:|
| L2, matched side | 1792 | 319161 | 0.477641 |
| CMU, matched side | 1128 | 177608 | 0.135094 |

The vectors and complete rankings are in
`results/stage7c/utility_v4_l2.pt`,
`results/stage7c/utility_v4_l2_ranking.pt`,
`results/stage7c/utility_v4_cmu.pt`, and
`results/stage7c/utility_v4_cmu_ranking.pt`.

The only requested cross-condition statistics are:

| Statistic | Result |
|---|---:|
| Spearman(`U_L2`, `U_CMU`) | 0.996069 |
| TopK overlap, K=256 | 0.972656 |
| TopK overlap, K=512 | 0.986328 |

The auditable report is
[`results/stage7c/matched_text_control.json`](../results/stage7c/matched_text_control.json).

## Interpretation

The matched-text control shows that the task-sensitive fine-tuning shifts
are overwhelmingly shared across the accented L2 and native CMU inputs in
this first-pass setting. It does not support an accent-specific utility
structure claim. The result is still useful as a boundary condition: the
Stage 7A/B functional-selection result should be described as a shared,
task-aware refinement of a strong magnitude proxy, rather than as evidence
that utility discovers a separate accent-only coordinate set.

This control compares utility rankings under one frozen L2-adapted CTC head;
it is not a native-model ASR optimum and does not redefine the primary Delta
as a cross-condition subtraction.

## Reproduction

The no-training sequence is wrapped by
[`scripts/run_stage7c.sh`](../scripts/run_stage7c.sh). The wrapper prepares
the manifests, reuses completed CMU cache entries, computes both v4 vectors,
and writes the comparison report.
