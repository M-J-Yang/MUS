# Official-split Fold1/Fold2 local replicas

**Date:** 2026-09-03 (Asia/Shanghai)
**Status:** official split manifests materialized; local replica training in progress

The released upstream CSVs for Fold1 and Fold2 are used as the data source. They
are mapped onto the local 16-kHz audio without changing the transcript or split
assignment. The resulting manifests are:

- `manifests/l2_arctic_official_ut8/fold1/{train,dev,test}.jsonl`
- `manifests/l2_arctic_official_ut8/fold2/{train,dev,test}.jsonl`

The held-out Utility subsets were created with the same `utility-every=10`
protocol as Fold0:

| Fold | Train | Dev | Test | Teacher | Utility |
|---:|---:|---:|---:|---:|---:|
| 1 | 16,070 | 1,979 | 686 | 14,455 | 1,615 |
| 2 | 15,953 | 2,086 | 691 | 14,353 | 1,600 |

Training uses the frozen Fold0 local recipe through
`scripts/run_l2_arctic_official_local_replica.sh`: W2V2-Large-960h, one CTC
head warm-up, transcript-grouped SupCon, bf16, lambda 0.05, batch/group setup
24/6/4, seed 1337, and dev-WER checkpoint selection. These models are explicitly
**official-split local replicas**, not published official checkpoints.

After each checkpoint is selected, the frozen analysis uses

```text
Delta = E_ft - E0
Utility = mean(abs(Delta * dL_CTC/dDelta))
```

with the original fine-tuned CTC head, no retraining, no healing, and only the
seven requested conditions: Full, NoShift, Utility-75, Utility-50, Magnitude-50,
DropWorst-25, and DropBest-25. Fold1 is not used to alter the protocol; Fold2
will use the same command and settings with only the fold index changed.
