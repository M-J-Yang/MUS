# L2-ARCTIC UT-8 Fold 0 Large ASR Baselines

**Date:** 2026-09-01 (Asia/Shanghai)
**Status:** legacy W2V2-Large SSL+new-head run complete (25.01% test WER); WavLM-Large diagnostic active
**Scope:** diagnostic record for the original Fold 0 Large ASR comparison. The corrected 960h transfer and shift-pruning protocol is recorded in `docs/l2_arctic_ut8_fold0_shift.md`.

## Hypothesis and controlled comparison

The Fold 0 pilot tests whether the two specified Large SSL backbones converge under
one shared character-CTC protocol. W2V2-Large and WavLM-Large use the same Fold 0
train/dev/test manifests, train-only vocabulary, seed, batch, optimizer, scheduler,
checkpoint selection rule, and greedy decoder. The only planned main variable is the
pretrained backbone.

## Data protocol

The new `manifests/l2_arctic_ut8/` namespace follows the recent L2-ARCTIC unseen-
transcript setup: four held-out-speaker assignments, each repeated with an independent
prompt partition, produce eight folds. Each fold holds out exactly one speaker per L1
for test, reserves about 10% of the shared prompt IDs for that test set, and assigns the
remaining prompt/transcript groups from the other 18 speakers to train/dev. The source
manifest and old `manifests/arctic_step2/` files are untouched.

`manifests/l2_arctic_ut8/split_audit.json` records source digest, speaker assignment,
utterance counts, per-speaker/L1 counts, prompt overlap, exact transcript overlap, and
manifest digests. Fold 0 currently contains 16,386 train, 1,827 dev, and 667 test
utterances; both prompt and normalized transcript overlaps are zero.

The Fold 0 vocabulary is `assets/ctc_vocab/l2_arctic_ut8/fold0/` and was built only
from the 16,386 training transcripts (29 classes).

## Training contract

- backbone: `facebook/wav2vec2-large-lv60` or `microsoft/wavlm-large`
- character CTC, 16 kHz mono input
- convolutional feature encoder frozen
- 1 epoch CTC-head-only warm-up
- joint fine-tuning, max 20 epochs, early stopping patience 5
- AdamW, LR `1e-5`, weight decay `0.01`
- cosine schedule, warmup ratio `0.1`
- 4-GPU DDP, per-GPU batch 2, gradient accumulation 2, effective global batch 16
- FP16, gradient checkpointing, max gradient norm 1.0, LayerDrop 0
- best checkpoint selected by dev greedy WER; test is accessed only after selection
- seed `1337`

The executable entry point is `scripts/train_large_ctc.py`; the frozen values are also
recorded in `configs/l2_arctic_ut8_fold0_large.yaml`.

## Checks completed before formal training

Both models passed a 4-GPU smoke test using 16 train and 8 dev examples. Each smoke
ran head-only and joint forward/backward passes with finite CTC losses and completed
dev greedy decoding. The conv encoder had zero trainable parameters in both phases.
The first two W2V2 DDP attempts were stopped before training because `torchrun`
automatically selected an unreachable hostname; the successful and formal launches use
single-node static rendezvous at `127.0.0.1`.

## Formal run records

| Model | Output | Status | Best dev WER | Test WER |
|---|---|---|---:|---:|
| W2V2-Large (legacy SSL + new head) | `artifacts/runs/l2_arctic_ut8/fold0/w2v2_large/` | complete | 20.87% | 25.01% |
| WavLM-Large | `artifacts/runs/l2_arctic_ut8/fold0/wavlm_large/` | active | pending | pending |

The W2V2 row is a retained diagnostic result and is not the corrected main baseline. No method conclusion is available until the 960h transfer, reconstruction gate, and retained-shift evaluation are complete.
