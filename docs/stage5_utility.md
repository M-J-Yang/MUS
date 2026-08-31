# Stage 5 — CTC decision-aligned Delta utility

**Date:** 2026-08-31  
**Status:** complete for the first-pass L2-ARCTIC final-layer run.

## Method

Stage 5 is an offline attribution pass over the held-out `train_utility`
split. It loads the pure-linear FullDelta teacher from Stage 4, reuses the
Stage 4 vocabulary and text encoder, and never trains a model. For each
utterance it computes:

```text
ref + Delta → FullDelta logits
ground-truth target + teacher logits → Viterbi CTC alignment
aligned non-blank target a_t + full logits → strongest competitor c_t
q[t,i] = Delta[t,i] * (W_delta[a_t,i] - W_delta[c_t,i])
```

The pass streams `help_count`, `harm_count`, `zero_count`, and aligned
`|Delta|` sums. It does not save frame-level `q` tensors. The alignment is a
small self-contained Viterbi implementation in
`utility/forced_align.py`; this avoids depending on the deprecated
`torchaudio.functional.forced_align` API while keeping the same CTC lattice
definition.

## Run

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTHONPATH=src:/data/zb/ymj/MUS \
/home/zbzb/.conda/envs/py311/bin/python utility/compute_utility.py \
  --device cpu \
  --output-dir results/stage5 \
  --log-every 200
```

Inputs were:

- checkpoint: `artifacts/runs/stage4/full_delta/best.pt`;
- manifest: `manifests/arctic_step2/l2/train_utility.jsonl`;
- cache: `artifacts/features/stage3_l2/train/{wavlm_ft,delta}`;
- vocabulary: `assets/ctc_vocab/vocab.json`;
- teacher checkpoint epoch: 14, input `[E_ref; Delta]` with dimensions
  `1024 + 1024`, vocabulary size 29, blank id 0.

## Results

| Quantity | Result |
|---|---:|
| Utility utterances | 1,792 |
| Successfully aligned | 1,792 |
| Failed | 0 |
| Aligned non-blank frames | 120,805 |
| Delta dimensions | 1,024 |
| Spearman(`U`, `M`) | 0.324019 |
| Overlap@256 | 0.386719 |
| Overlap@512 | 0.646484 |

The correlation and overlap show that signed decision utility and shift
magnitude produce different rankings. This is a diagnostic only; it does not
yet establish that Utility-K improves WER. That claim belongs to Stage 6's
matched-budget Utility-K versus Magnitude-K versus Random-K experiment.

## Outputs

The completed result directory is [results/stage5](/data/zb/ymj/MUS/results/stage5):

- `utility.pt` and `magnitude.pt`: float64 vectors of shape `[1024]`;
- `utility_ranking.pt` and `magnitude_ranking.pt`: int64 permutations of
  `[0, 1023]`;
- `stats.json`: counts, protocol metadata, diagnostics, and failure examples;
- `debug_alignment.txt`: five human-readable teacher-alignment examples.

Sanity checks passed: utility is in `[-1, 1]`; all tensors are finite; both
rankings are complete permutations; and each coordinate satisfies
`help + harm + zero = 120,805`.

## Verification and conclusion

The three new Stage 5 tests and the existing repository tests pass (`7 passed`)
with one pre-existing PyTorch storage deprecation warning. The full attribution
pass has no alignment failures, so no recovery aligner or external acoustic
model was used.

This completes the implementation and attribution artifact gate. It produces
no method-effect conclusion yet. The next experiment is to freeze the two
rankings and train matched-budget `Utility256/512`, `Magnitude256/512`, and
`Random256/512` CTC heads using the existing Stage 4 protocol.
