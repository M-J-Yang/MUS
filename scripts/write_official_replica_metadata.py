#!/usr/bin/env python3
"""Write provenance metadata for an official-split local replica."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(1, 2), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--official-csv-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    csv_files = {name: args.official_csv_root / f"{name}.csv" for name in ("train", "val", "test")}
    missing = [str(path) for path in csv_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing official CSVs: " + ", ".join(missing))
    payload = {
        "protocol": "official_split_local_replica_frozen_core_seven_v1",
        "replica_label": "official-split local replica",
        "fold": args.fold,
        "public_source_prefix": f"files/Arctic/8fold/{args.fold}",
        "official_csvs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in csv_files.items()},
        "manifest_root": str(args.manifest_root),
        "checkpoint": str(args.checkpoint),
        "training": {
            "pretrained": "checkpoints/wav2vec2_large_960h_pretrained",
            "seed": 1337,
            "head_warmup": "fresh per official split, selected on dev WER",
            "supcon": {
                "lambda": 0.05,
                "temperature": 0.1,
                "ramp_ratio": 0.1,
                "projection_dim": 256,
                "batch_size_per_rank": 24,
                "group_size": 6,
                "samples_per_group": 4,
                "learning_rate": 1e-5,
                "weight_decay": 0.0,
                "scheduler": "linear",
                "precision": "bf16",
                "epochs": 40,
                "early_stopping_patience": 5,
                "max_duration_s": 10.0,
                "gradient_checkpointing": bool(args.gradient_checkpointing),
            },
        },
        "evaluation": {
            "conditions": [
                "Full",
                "NoShift",
                "Utility75",
                "Utility50",
                "Magnitude50",
                "DropWorst25",
                "DropBest25",
            ],
            "head": "original fine-tuned CTC lm_head frozen",
            "bootstrap": False,
            "retuning": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "fold": args.fold}, sort_keys=True))


if __name__ == "__main__":
    main()
