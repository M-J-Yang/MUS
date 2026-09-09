# Efficiency metrics for selective reuse

Date: 2026-09-07  
Status: complete for R0/W1/W2/D0; the strict matched-scenario output is recorded
in `artifacts/results/efficiency_benchmark/efficiency_metrics.json`.

## Purpose

The paper reports WER, paired bootstrap confidence intervals, and random-mask
variation. This benchmark adds parameter count, forward FLOPs, inference speed,
and GPU memory for R0/W1/W2/D0 under the same held-out test manifests.

The current DADS intervention is a mask over the final 1024-dimensional
representation. It does not remove encoder layers, channels, or CTC-head input
dimensions. Consequently, DADS, Magnitude, and Random at the same retention
budget have the same computation graph. The benchmark distinguishes:

1. `full`: one adapted encoder and its adapted CTC head;
2. `selective_dual_<method>`: reference encoder + adapted encoder + 50% mask +
   adapted head, with DADS/Magnitude/Random using the same graph;
3. `selective_cached_<method>`: cached reference/adapted representations + mask/head,
   which measures offline reuse only and is not an end-to-end ASR latency
   number.

## Fixed protocol

- batch size 1;
- FP32 by default;
- one RTX 3090 (`cuda:0` unless overridden);
- test-manifest utterance lengths, with samples spread across the duration
  range and including the duration median;
- model forward timing excludes audio file I/O and feature-extractor
  preprocessing;
- FLOPs are analytical Conv1d/Linear/attention-matmul estimates from executed
  tensor shapes, with small normalization/elementwise terms omitted;
- 50% retained coordinates use the saved DADS/Taylor ranking for each model.

## Reproduction

```bash
/home/zbzb/.conda/envs/py311/bin/python scripts/benchmark_efficiency.py \
  --config configs/efficiency_benchmark.yaml \
  --output-dir artifacts/results/efficiency_benchmark \
  --device cuda:0 --dtype fp32 --num-samples 8 --warmup 3 --repeats 5 \
  --overwrite
```

The JSON result is the machine-readable record; the Markdown result is the
paper-facing summary. Parameter count and online FLOPs should be interpreted
as unchanged (or higher for the dual-encoder path) unless a future structured
pruning implementation changes the actual encoder graph.

## Strict result summary

Measured on one NVIDIA GeForce RTX 3090, batch size 1, FP32, with eight test
utterances spread across the duration range and five timed repeats:

| Same-scenario path | Parameters | GFLOPs/utterance | ms/utterance | Peak allocated memory |
|---|---:|---:|---:|---:|
| Full adapted | 313.31--315.47M | 120.09--129.83 | 20.74--29.06 | 1.29--1.33 GiB |
| Online DADS/Magnitude/Random | 626.59--630.90M | 240.16--259.65 | 42.25--47.79 | 2.46--2.54 GiB |
| Cached DADS/Magnitude/Random head-only | 0.033M | 0.011--0.012 | 0.12--0.13 | 0.02--0.03 GiB |

The cached path additionally stores 9.52--10.30 MB for the eight paired final
representations. The measured per-method latency fluctuates with GPU scheduling
(roughly 42--48 ms online and 0.12--0.13 ms cached), but this is not a DADS
advantage: the three methods execute the same graph. These results support the
strict conclusion that DADS demonstrates selective reuse and WER differences,
but does not reduce deployment cost relative to matched masks.
