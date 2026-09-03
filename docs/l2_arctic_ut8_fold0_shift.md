# Fold 0 W2V2-Large 960h transfer and shift pruning

**Date:** 2026-09-02 (Asia/Shanghai)
**Status:** formal Fold 0 transfer, frozen representation cache, identity/WER gate, and 50% Taylor-utility evaluation complete.
**Scope:** one UT8 Fold 0 W2V2-Large checkpoint, then same-model representation reconstruction and one 50% Taylor-utility intervention.

## Scientific contract

The old Fold 0 pilot used the SSL-only `facebook/wav2vec2-large-lv60` checkpoint with a newly sized 29-class CTC head. Its 25.01% test WER is retained as exploratory evidence, not as the main method baseline.

The corrected run uses the ASR-pretrained `facebook/wav2vec2-large-960h` model and inherits its processor, tokenizer, and CTC head. The model is adapted to the fixed Fold 0 train/dev split with a one-epoch head-only warm-up followed by joint fine-tuning. Test WER is read only after selecting the best dev-greedy-WER checkpoint.

After the checkpoint is frozen, for each identical utterance and final encoder layer:

```text
E0(x)  = f0(x)
Eft(x) = fft(x)
Delta  = Eft(x) - E0(x)
E_m    = E0(x) + m * Delta
```

The first gate is both numerical and behavioral:

```text
E0 + Delta == Eft
CTC_ft(E0 + Delta) == CTC_ft(Eft)
```

Only after that gate passes is `U_i = mean(abs(Delta_i * dL_CTC/dDelta_i))` computed on `train_utility`. The 50% condition retains the top 512 of 1024 coordinates and sends `E0 + M_512 * Delta` through the already-fine-tuned `lm_head`; no new CTC head is trained.

## Reproducible commands

Prepare the held-out utility subset without changing Fold 0 train/dev/test:

```bash
PYTHONPATH=src python scripts/prepare_step2_utility_split.py \
  --train-manifest manifests/l2_arctic_ut8/fold0/train.jsonl \
  --teacher-out manifests/l2_arctic_ut8/fold0/train_teacher.jsonl \
  --utility-out manifests/l2_arctic_ut8/fold0/train_utility.jsonl
```

Run the corrected formal checkpoint with four GPUs:

```bash
bash scripts/run_l2_arctic_fold0_w2v2_960h.sh
```

The launcher intentionally omits `--vocab-dir`; this is the guardrail that selects the inherited processor and CTC head. A local snapshot can be supplied with `PRETRAINED_PATH=/path/to/facebook/wav2vec2-large-960h`.

Cache each split after `training_summary.json` is written:

```bash
for split in train_utility dev test; do
  PYTHONPATH=src python scripts/cache_l2_shift.py \
    --manifest manifests/l2_arctic_ut8/fold0/${split}.jsonl \
    --split "${split}" \
    --output-root artifacts/features/l2_arctic_ut8/fold0/w2v2_large_960h_shift \
    --pretrained-model facebook/wav2vec2-large-960h \
    --fine-tuned-model artifacts/runs/l2_arctic_ut8/fold0/w2v2_large_960h \
    --device cuda:0 --skip-existing
done
```

Compute the ranking only from `train_utility`:

```bash
PYTHONPATH=src python utility/compute_l2_shift_taylor_utility.py \
  --checkpoint artifacts/runs/l2_arctic_ut8/fold0/w2v2_large_960h \
  --manifest manifests/l2_arctic_ut8/fold0/train_utility.jsonl \
  --cache-root artifacts/features/l2_arctic_ut8/fold0/w2v2_large_960h_shift \
  --feature-split train_utility \
  --output-dir artifacts/results/l2_arctic_ut8/fold0/w2v2_large_960h_shift_utility \
  --device cuda:0
```

Run the identity/WER gate on dev first, then report the frozen test result:

```bash
PYTHONPATH=src python scripts/evaluate_l2_shift.py \
  --checkpoint artifacts/runs/l2_arctic_ut8/fold0/w2v2_large_960h \
  --manifest manifests/l2_arctic_ut8/fold0/dev.jsonl \
  --cache-root artifacts/features/l2_arctic_ut8/fold0/w2v2_large_960h_shift \
  --feature-split dev \
  --ranking artifacts/results/l2_arctic_ut8/fold0/w2v2_large_960h_shift_utility/utility_shift_taylor_ranking.pt \
  --output artifacts/results/l2_arctic_ut8/fold0/w2v2_large_960h_shift/dev_metrics.json \
  --device cuda:0
```

Use the same command with `test.jsonl`, `--feature-split test`, and a separate `test_metrics.json` only after the dev gate is accepted.

## Artifacts and conclusion rule

- Corrected checkpoint: `artifacts/runs/l2_arctic_ut8/fold0/w2v2_large_960h/`
- Representation cache: `artifacts/features/l2_arctic_ut8/fold0/w2v2_large_960h_shift/`
- Utility ranking: `artifacts/results/l2_arctic_ut8/fold0/w2v2_large_960h_shift_utility/`
- Evaluation reports: `artifacts/results/l2_arctic_ut8/fold0/w2v2_large_960h_shift/`

The reproducible post-processing wrapper is:

```bash
bash scripts/run_l2_arctic_fold0_w2v2_960h_shift_postprocess.sh
```

## Verified formal result

The inherited-head transfer used `facebook/wav2vec2-large-960h`, one head-only warm-up epoch, and 11 joint epochs before early stopping. The selected checkpoint was `joint/checkpoint-6147`, with dev WER `14.89%`; the final test WER from the same frozen fine-tuned model was `16.48%` under the direct batch-1 evaluator used by this protocol. The `18.60%` test value in `training_summary.json` is the separate Trainer batched-padding metric; it is retained as provenance but is not used for the shift comparison.

The cache identity gate passed for all 4,140 utterances (`train_utility=1646`, `dev=1827`, `test=667`): maximum `|E0 + Delta - Eft|` was `4.77e-7`, and the dev/test reconstruction logit error was `7.63e-6`. Both splits had direct-vs-cache prediction match `1.0`, `wer_identity_pass=true`, and `gate_pass=true`.

At 50% retention, the top 512 of 1024 hidden coordinates were selected using the original fine-tuned CTC head with no head retraining. WER results were:

| split | no shift | full shift | retained 50% | retained gain |
|---|---:|---:|---:|---:|
| dev | 20.73% | 13.25% | 14.08% | 88.9% |
| test | 22.75% | 16.48% | 17.51% | 83.5% |

These values are the accepted conclusion for this single-model Fold 0 protocol; exploratory multi-backbone and Delta-fusion branches remain preserved but are not expanded here.
