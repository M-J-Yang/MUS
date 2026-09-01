# Stage 6 — Matched-budget selected-Delta Linear CTC

Date recorded: 2026-09-01  
Status: complete; all six first-pass WER runs finished successfully.

## Hypothesis and controlled variable

Stage 5 rankings are frozen inputs to this stage. At a fixed retained budget
K, Utility-K should be compared with Magnitude-K and Random-K using the same
reference cache, manifests, linear CTC architecture, optimizer, training
seed, decoding rule, and checkpoint-selection rule. The only intended
difference is the selected Delta coordinate set.

The intervention is:

~~~text
cached [E_ref; Delta]
  → Delta[..., selected_indices]
  → [E_ref; Delta_K]
  → newly initialized LinearCTC(reference_dim + K, vocab_size)
~~~

The FullDelta checkpoint is never reused as a masked classifier. No WavLM or
W2V2 weights are changed, and no new feature cache is created.

## Implementation

- src/usde/stage6.py contains frozen-ranking validation,
  deterministic Random-K selection, dimension-only slicing, and the fresh
  SelectedDeltaLinearCTC head.
- scripts/train_selected_delta.py reuses CachedFeatureDataset, collate,
  CTCLoss, AdamW, greedy decoding, and best-dev-WER checkpoint selection from
  Stage 4.
- selected_indices.pt, sanity.json, best.pt, config.json, and metrics.json are
  saved for every run. Random runs also save the complete random_ranking.pt
  and, in the default layout, results/stage6/random_ranking_seed42.pt; K=256
  and K=512 are therefore nested for one seed.
- analysis/collect_stage6_results.py produces the eight-row report with
  Reference only, FullDelta, and the six selected-Delta runs.

## Frozen first-pass protocol

- Data: the existing L2-ARCTIC train_teacher/dev/test manifests.
- Feature cache: artifacts/features/stage3_l2; the same wavlm_ft and delta
  files are reused for all runs.
- Ranking: results/stage5/utility_ranking.pt or
  results/stage5/magnitude_ranking.pt, derived only from train_utility.
- Head: pure linear CTC, freshly initialized for each run.
- Training: seed 1337, AdamW, learning rate 1e-3, weight decay 0.01, batch
  size 32, 20 epochs, gradient clipping 1.0.
- Selection: dev WER chooses best.pt; test WER is computed once after loading
  that checkpoint. Random coordinate selection uses --selection-seed 42.

## Commands

Run one experiment:

~~~bash
PYTHONPATH=src python scripts/train_selected_delta.py \
  --selection utility --k 256 \
  --ranking results/stage5/utility_ranking.pt
~~~

The default output is results/stage6/utility/k256. To run the full 2×3
matrix with the same configuration:

~~~bash
for k in 256 512; do
  PYTHONPATH=src python scripts/train_selected_delta.py \
    --selection random --k "$k" --selection-seed 42
  PYTHONPATH=src python scripts/train_selected_delta.py \
    --selection magnitude --k "$k" \
    --ranking results/stage5/magnitude_ranking.pt
  PYTHONPATH=src python scripts/train_selected_delta.py \
    --selection utility --k "$k" \
    --ranking results/stage5/utility_ranking.pt
done
~~~

After all six runs:

~~~bash
PYTHONPATH=src python analysis/collect_stage6_results.py
~~~

The collector writes results/stage6/stage6_summary.json and prints the
final comparison table.

## First-pass results

All runs used the frozen protocol above and completed 20 epochs. Lower WER is
better.

| Input | Delta dims | Best dev WER | Test WER |
|---|---:|---:|---:|
| Reference only | 0 | 0.387236 | 0.260991 |
| FullDelta | 1024 | 0.375258 | 0.250872 |
| Random | 256 | 0.380167 | 0.254851 |
| Magnitude | 256 | 0.376534 | 0.252567 |
| Utility | 256 | 0.379676 | 0.254458 |
| Random | 512 | 0.377246 | 0.252542 |
| Magnitude | 512 | 0.374963 | 0.251167 |
| Utility | 512 | 0.375920 | 0.251633 |

At both budgets, ranking-based selection beats Random on test WER. Magnitude
is best at K=256 and K=512 in this fixed-seed first pass; Utility is better
than Random at both budgets but does not beat Magnitude. FullDelta remains the
best overall, while selected Delta recovers most of its gain over the
Reference-only baseline.
