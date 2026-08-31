# Reusable experiment protocol — Decision-Aligned Delta Selection

Date recorded: 2026-08-31  
Status: project memory/configuration frozen for the first-pass experiment; this entry records a protocol and does not itself establish a method result.

## Purpose

This document captures the reusable engineering configuration distilled from the recent activation-steering, MAS-LoRA, and accented-ASR comparisons. The transferable lesson is the evidence chain:

```text
representation analysis → intervention → downstream WER
```

For this project the concrete chain is:

```text
L2-ARCTIC
  → W2V2/WavLM fine-tuning
  → cached W2V2_pt, W2V2_ft, WavLM_ft
  → Delta = E_ft(x) - E_pt(x)
  → FullDelta / Magnitude-K / Utility-K / Random-K
  → same-budget CTC training
  → dev checkpoint selection
  → final test WER
```

The machine-readable contract is [configs/reusable_experiment_protocol.yaml](/data/zb/ymj/MUS/configs/reusable_experiment_protocol.yaml).

## Frozen first-pass rules

### Data separation

- Keep the existing fixed, speaker-disjoint `train.tsv`, `dev.tsv`, and `test.tsv` protocol.
- Fit the attribution teacher on `train_teacher` and derive the coordinate ranking on the held-out `train_utility` subset.
- Use `dev` only for checkpoint/model selection and `test` only for the frozen final report.
- Never derive Utility from `train + dev + test`, and never use test examples to choose a mask.
- Reuse `scripts/prepare_step2_utility_split.py` and record the generated audit/digests with each experiment.

### Representation and cache

- Define the primary representation shift on the same utterance and waveform:

  ```text
  Delta(x) = E_ft(x) - E_pt(x)
  ```

- Extract expensive backbone activations once and reuse them for FullDelta, Magnitude, Utility, Random, Top-K, Drop-Best, and Drop-Worst experiments.
- Preserve the frame axis. Do not mean-pool Delta before CTC attribution.
- Use the last layer in the first paper pass. If the main result is established, extend only to representative early/middle/late layers before considering a broader audit.

### Utility and intervention

- Use Viterbi CTC forced alignment and keep aligned non-blank frames.
- For each frame, identify the highest-logit competitor other than the aligned target.
- Score coordinate `i` by its mean signed contribution in the target-versus-competitor direction:

  ```text
  q[t,i] = Delta[t,i] * (W[target,i] - W[competitor,i])
  U_i = mean(sign(q[t,i]))
  ```

- Compare Random-K, Magnitude-K, and Utility-K at matched retained dimensions (`0.25D` and `0.5D`) against Reference and FullDelta.
- Keep the attribution teacher pure-linear, as required by `scripts/compute_step2_utility.py`; source-compatible fusion is a separate baseline-validation result, not an attribution result.
- Train all intervention variants with the same split, head, seed, decode rule, and budget. Select by `dev_wer`; report `test_wer` once the checkpoint is frozen.

## Deliberately deferred

The following are not part of the first-pass protocol:

- full layer × backbone × ranking × K × accent sweeps;
- steering-strength (`alpha`) sweeps;
- mean-pooled Delta for Utility;
- 8-fold cross-validation;
- using CMU-ARCTIC in the primary Delta definition.

These can be added only after the fixed-split result shows that Utility-K improves over the magnitude-matched baseline. The first expansion is three representative layers or three seeds, chosen to answer a specific paper question.

## CMU-ARCTIC control

CMU-ARCTIC is a secondary native/reference control. After the primary L2 result is established, use matched transcripts to compare existing-model `Delta_L2` and `Delta_CMU`, including Utility vectors, Top-K overlap, and Spearman correlation. This tests whether fine-tuning-induced decision-useful shifts have different structure on accented speech; it does not redefine the primary shift as `E(L2) - E(CMU)`.

## Expected record for each run

Each meaningful run must record the hypothesis, sole main variable, split/manifests, seed, training budget, command/configuration, cache/checkpoint paths, WERs, anomalies, and the resulting conclusion. Pilot numbers must remain labeled as pilot results and must not be presented as final paper results.

## Origin and project alignment

This protocol is an engineering synthesis of the user-provided notes on Activation Steering 2026, MAS-LoRA, and recent L2-ARCTIC ASR sanity checks. It is aligned with the existing DADS positioning in [docs/decision_aligned_delta_positioning.md](/data/zb/ymj/MUS/docs/decision_aligned_delta_positioning.md) and reuses the current utility-split and attribution entry points rather than introducing a parallel implementation.
