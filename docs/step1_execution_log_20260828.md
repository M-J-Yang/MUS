# Step 1 execution log — 2026-08-28

This append-only log supplements `step1_baseline_reproduction.md`. It records
the actual method and outcome of each executed sub-step.

## Step 1-A.1 — public-source parity audit — COMPLETE

**Method.** Cloned the authors' public `Zilai-WANG/Delta-Embedding-Fusion`
repository into a temporary, non-project reference directory and inspected its
configuration, precomputation, fusion model, training, and evaluation paths.
No source file was copied as a black box.

**Observed result.** The source uses Kaldi-style `wav.scp`/`text` inputs,
precomputes unpadded hidden states, stores the final hidden state using index
`-1`, and forms `fine-tuned - pretrained` delta tensors. Its fusion optimizer
is AdamW with 10 epochs, learning rate `1e-4`, weight decay `0.01`, batch size
16, and seed 42. The concrete fusion implementation is
`LayerNorm(2048) -> Dropout(0.1) -> Linear(2048, vocab)`, not a bare linear
map. Its evaluation path truncates unequal streams to the shorter length.

**Effect.** Step 1 adopts the source-compatible head and hyperparameters. The
project implementation deliberately strengthens one invariant: it rejects,
rather than truncates, any frame mismatch. This does not add an alignment
method and is behaviorally identical for expected equal-length streams. The
source `-1` convention is semantic transformer layer 24 here
(`hidden_states[24]`). The full source alignment table is in
`docs/step1_source_alignment.md`.

## Step 1-B — data contract implementation — COMPLETE (awaiting corpus audit)

**Method.** Implemented a JSONL manifest validator and a deterministic
training-only character vocabulary builder. The validator requires `utt_id`,
`audio_path`, `transcript`, and `speaker_id`; it detects duplicate utterances,
verifies 16-kHz mono audio when available, asserts speaker-disjoint splits,
and writes split SHA-256 digests and duration/count statistics.

**Observed result.** `scripts/audit_manifests.py` and its unit tests were
executed locally: **2/2 passed**. The implementation has not audited MyST yet,
because no authorized corpus root or official split list is present in the
workspace.

**Effect.** Once `MYST_ROOT`/split files are supplied, the command below makes
the data split immutable before model training:

```bash
PYTHONPATH=src /home/zbzb/.conda/envs/py311/bin/python scripts/audit_manifests.py \
  --condition full --audit-out artifacts/data_audit/full.json \
  --vocab-out artifacts/data_audit/full_vocab.json
