# Official Fold0 paired-bootstrap freeze

**Date:** 2026-09-03 (Asia/Shanghai)
**Status:** complete; Fold0 pruning protocol frozen after statistical supplement

## Purpose and protocol

This supplement addresses whether the small Utility-75 versus Full difference
could be test-set fluctuation. It replays the accepted official Fold0 cache with
the original frozen oracle CTC head and saved Utility/Magnitude rankings. For each
of 10,000 replicates, 675 complete test utterances are sampled with replacement;
the same sampled utterances are used for both systems in each paired comparison.
Corpus WER is recomputed from summed word-edit counts and reference words. The
reported interval is the percentile 95% confidence interval in percentage points.

Command:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src:. \
  /home/zbzb/.conda/envs/py311/bin/python scripts/bootstrap_shift_pruning.py \
  --checkpoint artifacts/oracles/wav2vec2-large-l2-arctic-supcon-repeated-8fold-0 \
  --manifest manifests/l2_arctic_official_ut8/fold0/test.jsonl \
  --cache-root artifacts/features/l2_arctic_official_ut8/fold0/w2v2_large_960h_oracle_shift \
  --feature-split test \
  --utility-ranking artifacts/results/l2_arctic_official_ut8/fold0/w2v2_large_960h_oracle_shift_empirical_package/rankings/utility_ranking.pt \
  --magnitude-ranking artifacts/results/l2_arctic_official_ut8/fold0/w2v2_large_960h_oracle_shift_empirical_package/rankings/magnitude_ranking.pt \
  --output-json artifacts/results/l2_arctic_official_ut8/fold0/w2v2_large_960h_oracle_shift_bootstrap/paired_bootstrap.json \
  --output-markdown artifacts/results/l2_arctic_official_ut8/fold0/w2v2_large_960h_oracle_shift_bootstrap/paired_bootstrap.md \
  --device cuda:0 --bootstrap-replicates 10000 --seed 1337
```

## Result

| Comparison | Difference | 95% paired-bootstrap CI |
|---|---:|---:|
| Utility 75% − Full | +0.0855 pp | [−0.2051, +0.3791] pp |
| Utility 50% − Full | +1.2141 pp | [+0.7424, +1.7074] pp |
| Utility 50% − Magnitude 50% | −4.9248 pp | [−5.7168, −4.1595] pp |
| Utility 75% − Magnitude 75% | −1.8297 pp | [−2.4174, −1.2504] pp |

The cache identity gate passed with maximum `|E0 + Delta − Eft| = 4.77e-7`.
Utility-75 is statistically indistinguishable from Full under this paired
bootstrap, supporting the near-lossless claim without claiming an improvement.
Utility is clearly better than matched Magnitude at both 50% and 75% retention.

No further Fold0 pruning criterion, retention ratio, seed, or curvature analysis
should be added. The next experiment is held-out Fold1 replication.
