# Step 1 — Mind the Shift baseline reproduction and environment freeze

**Status:** active  
**Owner:** USDE experiment track  
**Last updated:** 2026-08-28 (Asia/Shanghai)

## Objective and acceptance criterion

This step reproduces the *Mind the Shift* WavLM + final-layer Delta W2V2
baseline before any layer- or feature-utility selection is enabled. The frozen
system is

\[
 [E^{\mathrm{WavLM}}_{\mathrm{ft},24};
 E^{\mathrm{W2V2}}_{\mathrm{ft},24}-E^{\mathrm{W2V2}}_{\mathrm{pt},24}]
 \rightarrow \text{Concat + Linear CTC}.
\]

The primary reproduction condition is MyST **Full**. The same immutable
pipeline is subsequently run under 10 h, 5 h, and 1 h training budgets. No
USDE layer or feature selection is permitted in Step 1.

A Step 1 run is accepted only when all of the following exist and agree:

1. speaker-disjoint `train/dev/test` manifests with a saved split digest;
2. a character vocabulary built from training transcripts only;
3. independently fine-tuned WavLM-Large and W2V2-Large-lv60 checkpoints;
4. versioned, frame-aligned WavLM-ft, W2V2-ft, and W2V2-pt final-layer
   features, with numerical delta checks;
5. a frozen-encoder concat + linear CTC checkpoint and dev/test WER reports;
6. a run manifest containing every model ID/revision, package version, seed,
   configuration digest, and checkpoint digest.

## Frozen experimental contract

| Item | Fixed value for Step 1 |
| --- | --- |
| Dataset | Licensed My Science Tutor (MyST) speech corpus, 16-kHz mono waveform input |
| Split policy | The exact `SPAPL_KidsASR` MyST split protocol; preserve speaker disjointness and record source split files/digests |
| Reference encoder | `microsoft/wavlm-large`, independently fine-tuned with character CTC on the selected training condition |
| Delta encoder | `facebook/wav2vec2-large-lv60`, independently fine-tuned with the identical training condition |
| Fine-tuned public reference checkpoints | `balaji1312/wavlm-large-myst-fullfinetune`, `balaji1312/wav2vec2-large-myst-fullfinetune` (used for parity audit, not silently mixed with newly trained checkpoints) |
| Representations | WavLM-ft layer 24; W2V2-ft layer 24 minus W2V2-pt layer 24; dimension 1024 |
| Temporal handling | no interpolation/pooling/alignment module; reject an utterance if streams do not share `T` |
| Fusion training | all SSL encoders frozen; source-compatible `LayerNorm(2048) -> Dropout(0.1) -> Linear(2048, |V|)` CTC head |
| Evaluation | character decoding followed by the corpus-standard text normalization and WER; dev chooses training epoch, test is final-only |
| Conditions | Full, 10 h, 5 h, 1 h; the subset lists are immutable and nested within the Full train split |

## Execution log

### Step 1-A — workspace and runtime inventory — COMPLETE

**Method.** Read-only inventory of the worktree, nearby candidate data/model
directories, Conda environments, package imports, and GPU availability.

**Observed result.** The initial worktree contained only the ICLR paper source;
it had no training implementation, MyST manifests, local WavLM/W2V2 cache, or
fine-tuned checkpoint. `/home/zbzb/.conda/envs/py311/bin/python` is the
selected runtime: Python 3.11.11, PyTorch 2.0.1+cu118, Transformers 4.37.2,
Datasets 4.4.1, SoundFile 0.14.0, NumPy 1.24.3, and PyYAML 6.0.3. `evaluate`
and `jiwer` are absent. Four NVIDIA RTX 3090 GPUs (24 GB each) were visible,
with at least 17 GB free on every device at inspection time.

**Effect.** Step 1 must create the implementation and cannot start model
training until the licensed MyST location and its official split metadata are
provided. The dependency lock will target the selected Python 3.11 runtime.

### Step 1-B — data contract and manifest interface — IN PROGRESS

**Method.** Establish a strict JSONL manifest schema and validator before
touching audio. The adapter will consume the authorized MyST layout without
copying audio, build the training-only character vocabulary, verify 16-kHz mono
audio, validate speaker disjointness, and save SHA-256 digests for each split.

**Planned output.** `manifests/{full,10h,5h,1h}/{train,dev,test}.jsonl`,
`artifacts/data_audit/<condition>.json`, and a frozen vocabulary JSON.

**Current result.** No MyST corpus or split manifest was discoverable in the
permitted worktree/nearby experiment directories, so the audit cannot yet
produce counts or duration totals. This is an external-data dependency, not a
failed experiment.

### Step 1-C — independent SSL CTC fine-tuning — PENDING

**Method.** Fine-tune WavLM-Large and W2V2-Large-lv60 separately under the
same condition and vocabulary. Each training run writes config, seed,
package/GPU metadata, best-dev checkpoint, and decoded dev WER. A public
checkpoint parity path is retained as a separate audit; it cannot replace the
condition-specific training run.

### Step 1-D — feature extraction and original fusion baseline — PENDING

**Method.** Extract WavLM-ft and both W2V2 states in evaluation mode, form
`delta = w2v2_ft - w2v2_pt` at the final transformer layer, reject unequal
frame lengths, then train a fresh frozen-feature linear CTC head on the 2048-D
concatenation.

### Step 1-E — parity report and freeze — PENDING

**Method.** Check artifacts/digests, decode dev and untouched test once, and
compare the Full result to the paper's reported WavLM + Delta W2V2 reference
(9.64 test WER, subject to identical split/normalization). The report will
separate exact-parity results from runs with any declared deviation.

## Next operator input required

MyST is licensed/restricted and was not present locally. To execute Step 1-B
through Step 1-E, set `MYST_ROOT` to the authorized dataset root that contains
the official split metadata, or place the already-approved split lists under
`manifests/source/`. No audio or transcript is downloaded or redistributed by
this project.
