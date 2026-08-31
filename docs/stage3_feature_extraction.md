# Stage 3 — Frozen representation-Delta cache

**Date:** 2026-08-31  
**Status:** implementation, one-/20-utterance QA, train/dev cache reuse, test extraction, and full cache audit complete.

## Objective

Turn the three independently fine-tuned SSL encoders into one-time feature
generators. For each utterance, the script computes

```text
e_pt  = W2V2_pt(x)[last]
e_ft  = W2V2_ft(x)[last]
e_ref = WavLM_ft(x)[last]
delta = e_ft - e_pt
```

The W2V2 checkpoints receive exactly the same processor output. All models are
loaded in evaluation mode and run under inference mode. The hard QA gates are

```text
shape(e_pt) == shape(e_ft)
frames(e_ref) == frames(delta)
delta.abs().mean() > 0
```

No pooling, normalization, alignment module, utility score, ranking, or CTC
head is part of this stage.

## Implementation

`scripts/extract_stage3_features.py` accepts both the project TSV files and
JSONL manifests. A dry run prints the three shapes and Delta statistics. A
normal run stores only the two streams needed downstream:

```text
features/
├── train/wavlm_ft/<utt_id>.pt
├── train/delta/<utt_id>.pt
├── dev/wavlm_ft/<utt_id>.pt
├── dev/delta/<utt_id>.pt
└── test/...
```

Each cache file is a plain CPU tensor. `--debug-output-dir` can additionally
save `w2v2_pt`, `w2v2_ft`, `wavlm_ft`, and `delta` for the one-/20-utterance
debug gate.

## Current ARCTIC command

The frozen Step 2 ARCTIC run uses the 1024-dimensional public MyST-compatible
checkpoints and manifests:

```bash
PYTHONPATH=src /home/zbzb/.conda/envs/py311/bin/python scripts/extract_stage3_features.py \
  --manifest manifests/arctic_step2/l2/train.jsonl \
  --output-root artifacts/features/stage3_l2 \
  --wavlm-ft checkpoints/wavlm_myst_fullfinetune \
  --w2v2-ft checkpoints/w2v2_myst_fullfinetune \
  --w2v2-pt checkpoints/w2v2_large_lv60_pretrained \
  --device cuda:0 --max-utterances 1 --dry-run
```

After one utterance passes, repeat with `--max-utterances 20 --dry-run`, then
remove `--dry-run` for each frozen train/dev/test manifest. Use
`--skip-existing` only to resume an interrupted cache.

The `data/*.tsv` files and `checkpoints/{w2v2_ft,wavlm_ft_lr3e-4_b32}` are a
separate 768-dimensional base-model experiment. They must not be mixed with
the 1024-dimensional `w2v2_large_lv60_pretrained` checkpoint when forming a
Delta; a matching 768-dimensional W2V2 pretrained model is required first.

## QA result

The one-utterance gate passed on `l2_ABA_arctic_a0001`: all four tensors were
`(159, 1024)`, with `mean(|Delta|)=0.189124` and `max(|Delta|)=4.893672`.
The first 20 train records also passed. Their frame range was 69–295 and the
mean of per-utterance `mean(|Delta|)` was 0.209605. No final cache was written
by these dry runs.

The full cache is now complete at `artifacts/features/stage3_l2` with train=17,816,
dev=4,527, and test=4,524 utterances. `cache_audit.json` independently
verified manifest coverage, paired files, `[T,1024]` tensors, frame alignment,
finite values, and non-zero Delta for all three splits. This is an
extraction/shape result only; it does not establish a WER or a DADS result.
The project is now ready for the frozen-encoder FullDelta CTC head.
