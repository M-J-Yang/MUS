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

## Empirical package command

After the formal cache and Utility ranking exist, run:

```bash
CUDA_VISIBLE_DEVICES=3 DEVICE=cuda:0 NUM_WORKERS=4 \
  bash scripts/run_l2_arctic_official_fold0_empirical_package.sh
```

This evaluates Random, Random+Rescale, Magnitude, Gradient-only, and Taylor
Utility at 25%, 50%, and 75% retention, plus direct DropBest, DropWorst, and
Random deletion at 10%, 25%, and 50%. Random conditions use seeds 1337, 2027,
and 31415 and report mean ± sample standard deviation. All conditions reuse the
same cached streams and original frozen oracle head.

The package writes `metrics.json`, `summary.md`, and reproducible ranking files
under:

`artifacts/results/l2_arctic_official_ut8/fold0/w2v2_large_960h_oracle_shift_empirical_package/`

## Equivalent orthogonal reparameterization control

To test whether coordinate-wise concentration is an intrinsic property of the
ASR function or only of the native final-layer basis, run:

```bash
CUDA_VISIBLE_DEVICES=3 DEVICE=cuda:0 NUM_WORKERS=4 \
  bash scripts/run_l2_arctic_official_fold0_equivalent_rotation.sh
```

For each Haar-style random orthogonal matrix `Q`, the cached streams are
transformed as `E' = Q E` and the frozen linear CTC head as `W' = W Q^T`.
The script checks logits, greedy frame predictions, WER, and the raw CTC
gradient relation before recomputing magnitude, gradient, and
`abs(Delta * dL_CTC/dDelta)` coordinate scores.  It reports native-basis
concentration against the random-rotation null and writes score tensors,
`metrics.json`, and `summary.md` under:

`artifacts/results/l2_arctic_official_ut8/fold0/w2v2_large_960h_oracle_shift_equivalent_rotation/`

The control does not claim that individual dimensions are invariant.  If
concentration changes under equivalent rotations, the result should be
described as a property of the native parameterization; the function-level
alternative is a rotation-stable low-dimensional adaptation subspace.

### Fold0 result (2026-09-11)

The five rotations (seeds 1337, 2027, 31415, 4242, and 9001) all passed the
identity gate.  Dev/test decoded predictions and WER were unchanged (8.590%
and 10.499%).  The largest measured logit error was `1.91e-5`, and the largest
raw-gradient equivariance error was `9.03e-8`; one dev frame under seed 1337
was a numerically explainable argmax tie, with no decoded prediction change.

The native utility score placed `7.78%` of its mass in the top 1% coordinates,
versus `3.37% ± 0.22%` under random rotations; at 10% the corresponding
values were `30.30%` and `21.55% ± 0.37%`.  Native-vs-rotated Top-K Jaccard
overlap was `0.0` at 1% and `0.058 ± 0.021` at 10%.  Thus the experiment
supports retaining only a native-basis engineering observation, not an
invariant claim about individually identifiable decision-critical dimensions.
