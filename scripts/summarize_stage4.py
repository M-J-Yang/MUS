#!/usr/bin/env python3
"""Create the final three-row Stage 4 WER table from run metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONDITIONS = ("ref", "full_embedding", "full_delta")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/runs/stage4"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        metrics_path = args.runs_root / condition / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "condition": condition,
                "feature": {
                    "ref": "E_ref",
                    "full_embedding": "[E_ref; E_w2v2_ft]",
                    "full_delta": "[E_ref; Delta]",
                }[condition],
                "input_dim": metrics["input_dim"],
                "best_epoch": metrics["best_epoch"],
                "best_dev_wer": metrics["best_dev_wer"],
                "test_wer": metrics["test_wer"],
                "train_manifest": metrics["train_manifest"],
                "dev_manifest": metrics["dev_manifest"],
            }
        )
    report = {
        "protocol": "stage4_three_condition_linear_ctc_v1",
        "selection": "best development WER",
        "decoder": "greedy CTC",
        "seed": 1337,
        "rows": rows,
        "deltas": {
            "full_embedding_minus_ref_dev_wer": rows[1]["best_dev_wer"] - rows[0]["best_dev_wer"],
            "full_delta_minus_ref_dev_wer": rows[2]["best_dev_wer"] - rows[0]["best_dev_wer"],
            "full_embedding_minus_ref_test_wer": rows[1]["test_wer"] - rows[0]["test_wer"],
            "full_delta_minus_ref_test_wer": rows[2]["test_wer"] - rows[0]["test_wer"],
        },
    }
    output = args.output or (args.runs_root / "stage4_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
