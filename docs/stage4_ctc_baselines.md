# Stage 4 — Three-condition frozen-feature CTC comparison

**Date:** 2026-08-31  
**Status:** complete for the first-pass L2-ARCTIC final-layer run.

Stage 4 trains one independent pure-linear CTC head per condition. The
reference stream is always the first 1024 coordinates:

```text
A: E_ref
B: [E_ref; E_w2v2_ft]
C: [E_ref; Delta]
```

All runs use the same Stage 2 vocabulary, `train_teacher`/`dev`/`test` split
protocol, seed `1337`, AdamW (`lr=1e-3`, `weight_decay=0.01`), batch size 32,
20 epochs, greedy CTC decoding, and best-dev-WER checkpoint selection. The
test split is read only after training is complete. Using `train_teacher`
keeps the held-out `train_utility` subset available for unbiased attribution
in the next stage.

## Results

| Condition | Input | Dim | Best dev WER | Test WER |
|---|---|---:|---:|---:|
| A | `E_ref` | 1024 | 0.387236 | 0.260991 |
| B | `[E_ref; E_w2v2_ft]` | 2048 | 0.375307 | 0.250725 |
| C | `[E_ref; Delta]` | 2048 | 0.375258 | 0.250872 |

C is the pure-linear teacher for Stage 5. Its checkpoint stores
`linear.weight` with shape `[29, 2048]`; `linear.weight[:, 1024:]` is the
`[29, 1024]` `W_delta` slice.

## Reproduction

The shared entry point is `scripts/train_stage4_ctc.py`; set
`--condition` to `ref`, `full_embedding`, or `full_delta`. The cache audit is
in `artifacts/features/stage3_l2/stage4_cache_audit.json`, and the result
table is in `artifacts/runs/stage4/stage4_summary.json`.
