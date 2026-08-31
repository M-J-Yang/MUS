# Stage 2 — L2-ARCTIC CTC fine-tuning

**Date:** 2026-08-31  
**Status:** Wav2Vec2 and WavLM CTC fine-tuning and final evaluation complete. The
initial WavLM recipe collapsed to blank, but Repair A recovered normal convergence.

## Protocol

The two models start from clean SSL checkpoints and use one shared, train-split-only
character vocabulary:

| model | SSL initialization | output |
|---|---|---|
| Wav2Vec2 | `facebook/wav2vec2-base` | `checkpoints/w2v2_ft` |
| WavLM | `microsoft/wavlm-base-plus` | `checkpoints/wavlm_ft_lr3e-4_b32` |

Audio is loaded at training time and resampled to 16 kHz if needed. The convolutional
feature encoder is frozen; the Transformer and CTC head are trainable. Evaluation is
greedy CTC decoding with word error rate. The test split is not used for model or epoch
selection.

## Implemented files

- `scripts/build_ctc_vocab.py` scans `data/train.tsv` and writes `assets/ctc_vocab/`.
- `scripts/finetune_ctc.py` is the shared Wav2Vec2/WavLM training entry point.
- `scripts/evaluate_ctc.py` evaluates dev/test WER and can merge results into one JSON.
- `scripts/diagnose_wavlm_ctc.py` runs a short FP32 overfit diagnostic with SpecAugment off.
- `src/usde/ctc.py` contains TSV/JSONL loading, dynamic resampling, padding, and length grouping.
- `src/usde/metrics.py` provides a dependency-free corpus WER fallback because the current
  environment does not currently have `jiwer` installed.

## Checks completed

- `assets/ctc_vocab/vocab.json` contains 29 classes: `<pad>`, `<unk>`, `|`, and `a-z`.
- Two real L2-ARCTIC waveforms passed dynamic loading, processor encoding, batching, and
  label masking (`-100`).
- CLI help and Python compilation passed.
- Existing tests passed: `2 passed`.

The official snapshots are cached at:

- `artifacts/hf_cache/hub/models--facebook--wav2vec2-base` (~363 MB)
- `artifacts/hf_cache/hub/models--microsoft--wavlm-base-plus` (~361 MB)

With `HF_HOME` and `HF_HUB_CACHE` pointed at that directory, both models load with
`local_files_only=True` and produce finite `(1, 49, 768)` hidden states.

## Final results

| model | best checkpoint | dev WER | test WER | status |
|---|---|---:|---:|---|
| Wav2Vec2-ft | `checkpoints/w2v2_ft/checkpoint-10165` | 14.86% | 14.15% | converged |
| WavLM-ft (Repair A) | `checkpoints/wavlm_ft_lr3e-4_b32/checkpoint-1500` | 7.62% | 8.76% | converged |

The previous failed run is retained for audit at `checkpoints/wavlm_ft/checkpoint-1129`.
The machine-readable final summary is `results/stage2_ctc.json`; both final model roots
reload offline with finite parameters.

Repair A used the successful HF-style optimization scale: 3 epochs, learning rate
`3e-4`, effective batch size 32 (`4 x 8`), fixed 500-step warmup, layerdrop 0,
default SpecAugment, frozen convolutional feature encoder, and trainable Transformer
plus CTC head. The model remained at 100% Dev WER at step 300, then reached 37.59% at
step 600, 17.92% at step 900, and 10.55% at step 1500 before the final best checkpoint
selection at step 1500.

## WavLM diagnostic conclusion

The official WavLM feature extractor reports 16 kHz, `do_normalize=false`, and
`return_attention_mask=true`; the shared tokenizer and model both use pad/blank id 0,
with 29 valid character classes. The model CTC loss exactly matched a hand-computed
`torch.nn.CTCLoss`, and all sampled output lengths exceeded target lengths.

The pre-repair isolated pilots all remained at WER 1.0 or blank-dominated decoding:

- 64-example FP32 head-only and full-backbone overfit diagnostics;
- no attention mask on a one-utterance diagnostic;
- forced input normalization;
- separate CTC-head learning rate and larger head initialization;
- static blank-bias adjustment and a small blank-posterior penalty;
- a 10-epoch low-learning-rate pilot.
- a 10-epoch full-backbone batch pilot with the attention mask omitted.

The Wav2Vec2 control also blanked in the head-only diagnostic, while its formal
full-data run converged to 14.86% dev and 14.15% test WER. This shows that the short
head-only test is not by itself a valid acceptance criterion. Repair A demonstrates
that the original WavLM failure was an optimization-scale failure, not a broken data
or evaluation path.

## Reproducible commands

```bash
export HF_HOME=/data/zb/ymj/MUS/artifacts/hf_cache
export HF_HUB_CACHE=/data/zb/ymj/MUS/artifacts/hf_cache/hub
export HF_HUB_OFFLINE=1

PYTHONPATH=src /home/zbzb/.conda/envs/py311/bin/python scripts/build_ctc_vocab.py \
  --train-tsv data/train.tsv \
  --output-dir assets/ctc_vocab

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/zbzb/.conda/envs/py311/bin/python scripts/finetune_ctc.py \
  --model_name facebook/wav2vec2-base \
  --train_tsv data/train.tsv \
  --dev_tsv data/dev.tsv \
  --vocab_dir assets/ctc_vocab \
  --output_dir checkpoints/w2v2_ft

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src /home/zbzb/.conda/envs/py311/bin/python scripts/finetune_ctc.py \
  --model_name microsoft/wavlm-base-plus \
  --train_tsv data/train.tsv \
  --dev_tsv data/dev.tsv \
  --vocab_dir assets/ctc_vocab \
  --output_dir checkpoints/wavlm_ft_lr3e-4_b32 \
  --epochs 3 \
  --learning_rate 3e-4 \
  --warmup_steps 500 \
  --per_device_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --layerdrop 0 \
  --eval_steps 300

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/zbzb/.conda/envs/py311/bin/python scripts/evaluate_ctc.py \
  --checkpoint checkpoints/w2v2_ft \
  --dev_tsv data/dev.tsv \
  --test_tsv data/test.tsv \
  --vocab_dir assets/ctc_vocab \
  --output_json results/stage2_ctc.json \
  --model_key w2v2

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src /home/zbzb/.conda/envs/py311/bin/python scripts/evaluate_ctc.py \
  --checkpoint checkpoints/wavlm_ft_lr3e-4_b32 \
  --dev_tsv data/dev.tsv \
  --test_tsv data/test.tsv \
  --vocab_dir assets/ctc_vocab \
  --output_json results/stage2_ctc.json \
  --model_key wavlm
```
