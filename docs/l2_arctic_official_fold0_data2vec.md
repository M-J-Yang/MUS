# Official Fold0 Data2Vec backbone replication

This is the second-backbone replication of the frozen representation-shift
experiment. It uses `facebook/data2vec-audio-large-960h` with the official
Fold0 manifests and the same two-stage CTC adaptation recipe: one head-only
warm-up epoch followed by joint fine-tuning with the convolutional feature
encoder frozen. The Data2Vec processor and pretrained CTC head are inherited;
the Wav2Vec2 vocabulary or any post-cache head retraining is forbidden.

## Prepare the checkpoint

```bash
/home/zbzb/.conda/envs/py311/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="facebook/data2vec-audio-large-960h",
    local_dir="checkpoints/data2vec_audio_large_960h",
    allow_patterns=[
        "config.json", "preprocessor_config.json", "pytorch_model.bin",
        "special_tokens_map.json", "tokenizer_config.json", "vocab.json",
    ],
)
PY
```

## Smoke test

```bash
CUDA_VISIBLE_DEVICES=3 SMOKE_TEST=1 \
  bash scripts/run_l2_arctic_official_fold0_data2vec.sh
```

## Formal adaptation

After the smoke test passes, remove `SMOKE_TEST=1` and run on a free GPU:

```bash
CUDA_VISIBLE_DEVICES=3 \
  bash scripts/run_l2_arctic_official_fold0_data2vec.sh
```

The resulting formal checkpoint is then used to cache `E0`, `Eft`, and
`Delta=Eft-E0` under the Data2Vec-specific paths in the protocol config. The
identity gates must pass before computing Taylor Utility or running the seven
frozen-head conditions.

## Frozen-head shift analysis

After formal adaptation, run the complete Data2Vec post-processing pipeline on
GPU 3:

```bash
CUDA_VISIBLE_DEVICES=3 DEVICE=cuda:0 \
  bash scripts/run_l2_arctic_official_fold0_data2vec_shift_postprocess.sh
```

This writes the `E0`/`Eft`/`Delta` cache, the held-out `train_utility` Taylor
ranking, the empirical retention/deletion package, and the exact core-seven
summary under `artifacts/{features,results}/l2_arctic_official_ut8/fold0/`.
The direct-inference identity gate is then run for both dev and test:

```bash
CUDA_VISIBLE_DEVICES=3 DEVICE=cuda:0 \
  bash scripts/run_l2_arctic_official_fold0_data2vec_identity.sh
```

The calibration-size analysis uses deterministic subsets of 128, 256, 512,
1024, and all 1640 `train_utility` utterances with seeds 1337, 2027, and
31415. Its results are written to
`artifacts/results/l2_arctic_official_ut8/fold0/data2vec_calibration_size/`.
