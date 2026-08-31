# Compositional Domain Adaptation for Automatic Speech Recognition with Headwise Selective Attention Merging

This repository contains code and model-release pointers for two related
publications on low-resource child automatic speech recognition (ASR):

- **ICASSP 2025:** *Selective Attention Merging for Low Resource Tasks: A Case Study of Child ASR*
- **Computer Speech & Language :** *Compositional Domain Adaptation for Automatic Speech Recognition with Headwise Selective Attention Merging*

The Computer Speech & Language work extends the earlier SA Merge work from a single child-ASR
adaptation setting to compositional domain adaptation: combining task-specific
adaptations for speaking style, acoustic mismatch, cross-corpus transfer,
synthetic data, dialectal variation, and model scaling.

## Repository Layout

- `src/`: training and decoding utilities used by the original child-ASR experiments.
- `egs/`: train/test split files and example scripts for MyST and CMU Kids.
- `merge/sa_merge.py`: original selective-attention merge script from the ICASSP work.
- `merge/hsa_merge.py`: self-contained Headwise Selective Attention (HSA) merge script for the CSL extension.
- `merge/config/`: legacy SA Merge YAML examples.

## Installation

The original experiments used:

```bash
pip install torch transformers evaluate datasets
```

For current Hugging Face checkpoints, a recent `transformers` release is
recommended. The release packaging was checked with `transformers>=4.48` and
`safetensors`.

## Headwise Selective Attention Merge

`merge/hsa_merge.py` is a standalone script for composing two Whisper
fine-tuned checkpoints relative to a pretrained base model.

Let `B` be the pretrained base model, `M1` the primary/anchor fine-tuned model,
and `M2` the secondary fine-tuned model:

```text
tv1 = M1 - B
tv2 = M2 - B
```

For the top-K encoder self-attention query/key/value heads selected by
task-vector magnitude in encoder layer `i`, HSA applies:

```text
B + lambda_i * tv1 + (1 - lambda_i) * tv2
```

with the layer-wise exponential scaling used in the paper:

```text
lambda_i = lambda ** (alpha * (i + 1) / L)
```

where `i` is the zero-based Hugging Face encoder layer index and `L` is the
number of encoder layers. The `i + 1` offset maps checkpoint layer names such
as `encoder.layers.0` through `encoder.layers.L-1` to the paper's layer
position, so the final encoder layer receives `lambda ** alpha`. The default
`--alpha 0.2` matches the primary HSA configuration in the paper experiments.


Example:

```bash
python merge/hsa_merge.py \
  --pretrained-model openai/whisper-small.en \
  --model1 /path/to/primary_or_target_checkpoint \
  --model2 /path/to/secondary_checkpoint \
  --k-percent 0.6 \
  --lambda 0.5 \
  --alpha 0.2 \
  --out-path exp/hsa_small
```

To sweep lambdas:

```bash
python merge/hsa_merge.py \
  --pretrained-model openai/whisper-small.en \
  --model1 /path/to/primary_or_target_checkpoint \
  --model2 /path/to/secondary_checkpoint \
  --k-percent 60 \
  --lambdas 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --alpha 0.2 \
  --out-path exp/hsa_lambda_sweep
```

Use `--model1` for the adaptation that should be
preserved outside selected heads, and `--model2` for the adaptation being
composed into the selected heads.

## Release Models

### Whisper Small Baselines

| Model | Hugging Face repo |
| --- | --- |
| OGI Script 0-2 SFT | [balaji1312/whisper_small_sft_ogi_script_0_2](https://huggingface.co/balaji1312/whisper_small_sft_ogi_script_0_2) |
| OGI Spon 3-5 SFT | [balaji1312/whisper_small_sft_ogi_spon_3_5](https://huggingface.co/balaji1312/whisper_small_sft_ogi_spon_3_5) |
| OGI Spon 0-2 oracle SFT | [balaji1312/whisper_small_sft_ogi_spon_0_2_oracle](https://huggingface.co/balaji1312/whisper_small_sft_ogi_spon_0_2_oracle) |

### Core Whisper Small Merges

| Model | Hugging Face repo |
| --- | --- |
| HSA K=60%, Spon 3-5 + Script 0-2 | [balaji1312/whisper_small_hsa_k60_spon_3_5_script_0_2](https://huggingface.co/balaji1312/whisper_small_hsa_k60_spon_3_5_script_0_2) |
| SA full attention, Spon 3-5 + Script 0-2 | [balaji1312/whisper_small_sa_full_attention_spon_3_5_script_0_2](https://huggingface.co/balaji1312/whisper_small_sa_full_attention_spon_3_5_script_0_2) |
| OGI Spon 6-10 SFT | [balaji1312/whisper_small_sft_ogi_spon_6_10](https://huggingface.co/balaji1312/whisper_small_sft_ogi_spon_6_10) |
| HSA K=60%, Spon 6-10 + Script 0-2 | [balaji1312/whisper_small_hsa_k60_spon_6_10_script_0_2](https://huggingface.co/balaji1312/whisper_small_hsa_k60_spon_6_10_script_0_2) |

### Robustness and Transfer

| Model | Hugging Face repo |
| --- | --- |
| OGI Spon 3-5 noise/RIR SFT | [balaji1312/whisper_small_sft_ogi_spon_3_5_noise_rir](https://huggingface.co/balaji1312/whisper_small_sft_ogi_spon_3_5_noise_rir) |
| HSA noise/RIR Spon 3-5 + Script 0-2 | [balaji1312/whisper_small_hsa_spon_3_5_noise_rir_script_0_2](https://huggingface.co/balaji1312/whisper_small_hsa_spon_3_5_noise_rir_script_0_2) |
| All-TTS SFT | [balaji1312/whisper_small_sft_all_tts](https://huggingface.co/balaji1312/whisper_small_sft_all_tts) |
| HSA all-TTS + Script 0-2 | [balaji1312/whisper_small_hsa_all_tts_script_0_2](https://huggingface.co/balaji1312/whisper_small_hsa_all_tts_script_0_2) |
| AMI SFT | [balaji1312/whisper_small_sft_ami](https://huggingface.co/balaji1312/whisper_small_sft_ami) |
| HSA AMI + Script 0-2 | [balaji1312/whisper_small_hsa_ami_script_0_2](https://huggingface.co/balaji1312/whisper_small_hsa_ami_script_0_2) |

### Intersectional Dialect Generalization

| Model | Hugging Face repo |
| --- | --- |
| MyST SFT | [balaji1312/whisper_small_sft_myst](https://huggingface.co/balaji1312/whisper_small_sft_myst) |
| CORAAL SFT | [balaji1312/whisper_small_sft_coraal](https://huggingface.co/balaji1312/whisper_small_sft_coraal) |
| HSA MyST + CORAAL | [balaji1312/whisper_small_hsa_myst_coraal](https://huggingface.co/balaji1312/whisper_small_hsa_myst_coraal) |
| Noisy LibriSpeech SFT | [balaji1312/whisper_small_sft_noisy_librispeech](https://huggingface.co/balaji1312/whisper_small_sft_noisy_librispeech) |
| HSA Noisy LibriSpeech + CORAAL | [balaji1312/whisper_small_hsa_noisy_librispeech_coraal](https://huggingface.co/balaji1312/whisper_small_hsa_noisy_librispeech_coraal) |

### Scaling-Law Checkpoints

| Size | Script 0-2 SFT | Spon 3-5 SFT | HSA K=60% |
| --- | --- | --- | --- |
| Tiny | [script](https://huggingface.co/balaji1312/whisper_tiny_sft_ogi_script_0_2) | [spon](https://huggingface.co/balaji1312/whisper_tiny_sft_ogi_spon_3_5) | [hsa](https://huggingface.co/balaji1312/whisper_tiny_hsa_k60_spon_3_5_script_0_2) |
| Base | [script](https://huggingface.co/balaji1312/whisper_base_sft_ogi_script_0_2) | [spon](https://huggingface.co/balaji1312/whisper_base_sft_ogi_spon_3_5) | [hsa](https://huggingface.co/balaji1312/whisper_base_hsa_k60_spon_3_5_script_0_2) |
| Medium | [script](https://huggingface.co/balaji1312/whisper_medium_sft_ogi_script_0_2) | [spon](https://huggingface.co/balaji1312/whisper_medium_sft_ogi_spon_3_5) | [hsa](https://huggingface.co/balaji1312/whisper_medium_hsa_k60_spon_3_5_script_0_2) |
| Large v3 | [script](https://huggingface.co/balaji1312/whisper_largev3_sft_ogi_script_0_2) | [spon](https://huggingface.co/balaji1312/whisper_largev3_sft_ogi_spon_3_5) | [hsa](https://huggingface.co/balaji1312/whisper_largev3_hsa_k60_spon_3_5_script_0_2) |

## Legacy ICASSP 2025 Models

| Model | MyST test WER | Hugging Face repo |
| --- | ---: | --- |
| Whisper tiny - SA Merge | 11.52 | [balaji1312/whisper-tiny-myst-samerge](https://huggingface.co/balaji1312/whisper-tiny-myst-samerge) |
| Whisper base - SA Merge | 9.87 | [balaji1312/whisper-base-myst-samerge](https://huggingface.co/balaji1312/whisper-base-myst-samerge) |
| Whisper small - SA Merge | 8.85 | [balaji1312/whisper-small-myst-samerge](https://huggingface.co/balaji1312/whisper-small-myst-samerge) |
| Whisper small - SpecAug + SA Merge | 8.69 | [balaji1312/whisper-small-myst-specaug-samerge](https://huggingface.co/balaji1312/whisper-small-myst-specaug-samerge) |
| Whisper medium - SA Merge | 8.63 | [balaji1312/whisper-medium-myst-samerge](https://huggingface.co/balaji1312/whisper-medium-myst-samerge) |
| Whisper large v3 - SA Merge | 8.74 | [balaji1312/whisper-large-myst-samerge](https://huggingface.co/balaji1312/whisper-large-myst-samerge) |

## Citation

For the Computer Speech & Language work:

```bibtex
@article{shankara2026compositional,
title = {Compositional domain adaptation for automatic speech recognition with headwise selective attention merging},
journal = {Computer Speech & Language},
pages = {102012},
year = {2026},
issn = {0885-2308},
doi = {https://doi.org/10.1016/j.csl.2026.102012},
url = {https://www.sciencedirect.com/science/article/pii/S0885230826000756},
author = {Natarajan Balaji Shankar and Zilai Wang and Eray Eren and Abeer Alwan}
}
```

For the original ICASSP work:

```bibtex
@INPROCEEDINGS{shankar2025selective,
  author={Shankar, Natarajan Balaji and Wang, Zilai and Eren, Eray and Alwan, Abeer},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Selective Attention Merging for low resource tasks: A case study of Child ASR},
  year={2025},
  pages={1-5},
  doi={10.1109/ICASSP49660.2025.10887889}
}
```
