# L2-ARCTIC Fold1/Fold2 replication audit — corrected provenance

This note supersedes the earlier audit wording that described Fold1 and Fold2
as blocked by missing public splits.  The upstream repository publishes the
official split CSVs under `files/Arctic/8fold/1` and `files/Arctic/8fold/2`.
The public model release is still a Fold0-only checkpoint, so Fold1 and Fold2
are trained locally and must be labelled **official-split local replicas**,
not published official checkpoints.

Materialized official CSVs:

| Fold | Train | Validation | Test |
|---|---:|---:|---:|
| 1 | 16070 | 1979 | 686 |
| 2 | 15953 | 2086 | 691 |

The JSONL manifests in `manifests/l2_arctic_official_ut8/fold1` and `fold2`
are audio-verified projections of those CSVs.  Their provenance audits record
the fold-specific upstream prefix and SHA-256 hashes.

The frozen replication conditions are exactly:

`Full`, `NoShift`, `Utility75`, `Utility50`, `Magnitude50`, `DropWorst25`,
and `DropBest25`.

Fold1 is not used to retune the protocol.  Fold2 repeats the same recipe and
evaluation procedure unchanged.  No Fold1/Fold2 bootstrap, Random, or
Gradient condition is part of this core replication package.
