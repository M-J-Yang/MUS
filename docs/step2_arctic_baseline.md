# Step 2 — ARCTIC Full-Delta baseline and layer audit

**Status:** exploratory train/dev audit in progress; test remains untouched.

This experiment follows the objective after the MyST data dependency was
replaced by the locally prepared L2-ARCTIC and CMU ARCTIC conditions. L2 is
the target adaptation condition. CMU is a separate native/reference control
condition and is not mixed into L2 training.

## Frozen split

The split generator is `scripts/prepare_step2_arctic.py`. It writes
`manifests/arctic_step2/{l2,cmu}/{train,dev,test}.jsonl` and
`manifests/arctic_step2/split_audit.json`.

- L2: 16 speakers train (`ABA ASI BWC EBVS ERMS HJK HKK HQTV LXC MBMPS NCC NJS PNV RRBI SKA SVBI`), 4 dev (`THV TLV TNI TXHC`), and 4 test (`YBAA YDCK YKWK ZHAA`).
- CMU control: `bdl/clb` train, `rms` dev, and `slt` test.
- Audio is verified as 16-kHz mono; vocabularies are built from each
  condition's train transcripts only.

The resulting split audits are in `artifacts/step2_data_audit/`; all speaker
intersection sets are empty.

## Representation contract

For layer `l`, the audit stores

```
reference = WavLM_ft(x)[24]
delta_l  = W2V2_ft(x)[l] - W2V2_pt(x)[l]
[reference; delta_l] -> pure-linear CTC
```

The source-compatible reproduction head remains
`LayerNorm(2048) -> Dropout(0.1) -> Linear CTC` and is run separately from the
pure-linear attribution/audit heads. Test labels are not read for layer or
epoch selection.

## Checkpoint compatibility

The local fine-tuned checkpoints were written by a newer runtime using
`parametrizations.weight.original{0,1}`. The pinned runtime expects
`weight_g/weight_v`; `src/usde/features.py` remaps these keys, rejects
non-finite parameters, and writes feature records through atomic replacement.
The three local encoders now produce finite, frame-aligned layer-24 states.

## Reproducible commands

```bash
PYTHONPATH=src /home/zbzb/.conda/envs/py311/bin/python scripts/prepare_step2_arctic.py

PYTHONPATH=src /home/zbzb/.conda/envs/py311/bin/python scripts/extract_step2_layers.py \
  --manifest manifests/arctic_step2/l2/train.jsonl \
  --output-root artifacts/runs/step2_l2_layer_features/train \
  --wavlm-ft checkpoints/wavlm_myst_fullfinetune \
  --w2v2-ft checkpoints/w2v2_myst_fullfinetune \
  --w2v2-pt checkpoints/w2v2_large_lv60_pretrained \
  --layers $(seq 1 24) --storage-dtype float16 --batch-size 8 --device cuda:0
```

For the completed directional check, the same extractor was limited to the
first 1,000 train and 1,000 dev records, then
`scripts/train_step2_layer_audit.py` trained one pure-linear head per layer
for 10 epochs. Its report is
`artifacts/runs/step2_l2_layer_audit_subset1000/audit/layer_audit_report.json`.
This subset result is not a final test result and must not be presented as the
full-corpus Gate A/C evidence.
