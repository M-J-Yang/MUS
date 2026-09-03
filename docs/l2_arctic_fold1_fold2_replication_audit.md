# Fold1/Fold2 replication audit

**Date:** 2026-09-03 (Asia/Shanghai)
**Status:** blocked pending held-out fine-tuned oracle checkpoints and matching official manifests

## What is available

The local repository contains the corrected public Fold0 oracle at
`artifacts/oracles/wav2vec2-large-l2-arctic-supcon-repeated-8fold-0/`, its
official Fold0 manifests under `manifests/l2_arctic_official_ut8/fold0/`, and
the complete Fold0 E0/Eft/Delta cache and Utility/Magnitude rankings.

It also contains locally generated UT8 manifests for Fold1 and Fold2:

| Fold | Train | Dev | Test | Test speakers |
|---:|---:|---:|---:|---|
| 1 | 16,266 | 1,814 | 678 | NJS, TLV, TNI, TXHC, YKWK, ZHAA |
| 2 | 16,268 | 1,813 | 678 | MBMPS, NCC, SVBI, THV, YBAA, YDCK |

These are in `manifests/l2_arctic_ut8/fold{1,2}/` and come from the local
`l2_arctic_unseen_transcript_8fold_v1` generator. They are not currently
materialized in the corrected `l2_arctic_official_ut8` namespace. Fold0 itself
shows why the distinction matters: the corrected official split has 16,312 /
1,867 / 675 train/dev/test utterances rather than the local 16,386 / 1,827 /
667.

## Missing inputs

No Fold1 or Fold2 fine-tuned oracle checkpoint is present in the workspace. The
public model card explicitly identifies the available checkpoint as “8-Fold
Split 0”; the associated public repository's published-model table likewise
lists L2-ARCTIC repeated SupCon only for split 0 (plus a separate Arabic L1
holdout), not split 1 or split 2.

Therefore applying the Fold0 oracle to the Fold1/Fold2 manifests would not be a
replication: it would reuse a model adapted on a different fold. Locally
training new Fold1/Fold2 models would also be a new training run, not an exact
evaluation of official held-out oracle checkpoints, and the available local
Fold0 SupCon run does not reproduce the public Fold0 oracle WER.

## Re-entry condition

Resume the frozen replication only after obtaining, for each fold:

1. the matching official train/dev/test manifests (including the same transcript
   and processor vocabulary contract); and
2. the corresponding fine-tuned Wav2Vec2-Large CTC oracle checkpoint.

Then reuse the frozen Utility definition, 75%/50% retention ratios, frozen
original head, no retraining, no healing, and the same identity gates. The
existing cache and evaluation scripts can be parameterized for those inputs.

No Fold1/Fold2 method conclusion is claimed by this audit.
