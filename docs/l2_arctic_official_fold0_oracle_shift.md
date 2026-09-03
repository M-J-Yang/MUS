# Official Fold0 oracle shift pruning

**Status:** formal pipeline entry point for the publicly released strong Fold0 oracle.

The main result uses the public checkpoint
`artifacts/oracles/wav2vec2-large-l2-arctic-supcon-repeated-8fold-0/` as the frozen
fine-tuned target and `checkpoints/wav2vec2_large_960h_pretrained/` as the frozen
W2V2-Large-960h base. Both use the inherited 32-token processor and 1024-dimensional
final encoder representation. The official manifest namespace is
`manifests/l2_arctic_official_ut8/fold0/` (train 16,312; dev 1,867; test 675).

The public oracle was independently checked with the same batch-1 greedy evaluator:
Dev 8.590360%, Test 10.499316%. The oracle's positional-convolution safetensor keys
are remapped before loading; generic Transformers loading is not used for this
checkpoint because it silently initializes those two weights randomly.

## Formal command

```bash
CUDA_VISIBLE_DEVICES=3 DEVICE=cuda:0 NUM_WORKERS=4 \
  bash scripts/run_l2_arctic_official_fold0_oracle_shift.sh
```

The command creates a deterministic 10%-held-out `train_utility` subset, caches
`E0`, `Eft`, and `Delta=Eft-E0` for every official utterance, computes only
`abs(Delta * dL_CTC/dDelta)` on `train_utility`, checks numerical/logit/prediction/WER
identity on dev, and then evaluates frozen-head shift retention at 100%, 75%, and 50%.
It never trains, heals, or refits a CTC head.

Results are written to:

- `artifacts/features/l2_arctic_official_ut8/fold0/w2v2_large_960h_oracle_shift/`
- `artifacts/results/l2_arctic_official_ut8/fold0/w2v2_large_960h_oracle_shift_utility/`
- `artifacts/results/l2_arctic_official_ut8/fold0/w2v2_large_960h_oracle_shift/`

The final test table is `pruning_summary.md`; the JSON contains both dev and
test rows and the identity-gate diagnostics.
