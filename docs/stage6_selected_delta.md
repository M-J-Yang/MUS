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
- analysis/collect_stage6_results.py produces the original eight-row report
  and appends Utility v2/v3 rows when those result directories are present.

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

After all six first-pass runs (and any Utility v2/v3 follow-up runs):

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

## Utility v2/v3 follow-up

The first-pass `utility_ranking.pt` remains the binary sign utility
`E[sign(q)]`. The same frozen FullDelta attribution pass now also writes:

- `utility_v2_ranking.pt`: `E[q]`, retaining signed contribution magnitude;
- `utility_v3_ranking.pt`: `E[q * (1 - p(target))]`, emphasizing uncertain
  aligned frames.

Both variants are supported by `scripts/train_selected_delta.py` as
`--selection utility_v2` and `--selection utility_v3`. Their matched-budget
outputs use `results/stage6/utility_v2/k256`,
`results/stage6/utility_v2/k512`, `results/stage6/utility_v3/k256`, and
`results/stage6/utility_v3/k512`. They use the original train/dev/test
manifests, feature cache, seed, optimizer, and checkpoint-selection protocol.

To produce all three rankings in one pass, rerun the attribution command with
`--overwrite`; the original v1 ranking is preserved at its existing path and
the two new ranking files are added alongside it.

### Follow-up results

| Input | Delta dims | Best dev WER | Test WER |
|---|---:|---:|---:|
| Utility v2 | 256 | 0.379062 | 0.253598 |
| Utility v2 | 512 | 0.377958 | 0.252861 |
| Utility v3 | 256 | 0.379062 | 0.253574 |
| Utility v3 | 512 | 0.377688 | 0.252886 |

In this fixed-seed follow-up, both variants beat Random at K=256, but both
are slightly worse than Random at K=512. Neither beats the existing Magnitude
result. Utility v3 is marginally best among the new variants at K=256, while
Utility v2 is marginally best at K=512; these differences are too small to
interpret as a robust ranking without additional seeds.

## Utility v4 — CTC Taylor importance

Utility v4 changes the attribution target from an aligned local margin to the
actual Stage 4 CTC objective. `utility/compute_ctc_taylor_utility.py` freezes
the FullDelta teacher, computes the per-example CTC-loss gradient with respect
to cached Delta, and ranks dimensions by:

~~~text
U_i = E_frame,utterance[ | Delta[t,i] * d L_CTC / d Delta[t,i] | ]
~~~

The pass uses no forced alignment and no strongest competitor. It writes
`results/stage5/utility_v4.pt`, `utility_v4_ranking.pt`, and `stats_v4.json`.
The CTC loss is normalized by target length per utterance, matching the
reduction used by the Stage 4 trainer before averaging examples.

Generate the ranking with:

~~~bash
PYTHONPATH=src:. python utility/compute_ctc_taylor_utility.py \
  --device cpu --batch-size 8 --output-dir results/stage5 --overwrite
~~~

Then run the matched-budget experiments:

~~~bash
for k in 256 512; do
  PYTHONPATH=src:. python scripts/train_selected_delta.py \
    --selection utility_v4 --k "$k" \
    --ranking results/stage5/utility_v4_ranking.pt
done
~~~

The convenience wrapper `scripts/run_stage6_utility_v4.sh` runs both budgets
and refreshes the Stage 6 summary.

### Utility v4 results

The first fixed-seed v4 runs produced:

| Input | Delta dims | Best dev WER | Test WER |
|---|---:|---:|---:|
| Utility v4 (CTC Taylor) | 256 | 0.376436 | 0.252297 |
| Utility v4 (CTC Taylor) | 512 | 0.374914 | 0.250995 |

Utility v4 is slightly better than the existing Magnitude runs at both
budgets in this seed (0.000270 absolute WER at K=256 and 0.000172 at K=512),
while FullDelta remains best overall. The v4 ranking is highly correlated
with full-frame Delta magnitude on this split (Spearman 0.842151; overlap
0.9375 at K=256 and 0.839844 at K=512), so this is an encouraging first
result rather than a robust claim; additional training seeds are still needed.
