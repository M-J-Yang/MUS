#!/usr/bin/env python3
"""Aggregate the frozen seven-condition test results across L2-ARCTIC folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CORE_CONDITIONS = (
    "no_shift",
    "full",
    "utility75",
    "utility50",
    "magnitude50",
    "drop_worst25",
    "drop_best25",
)


def fold0_conditions(payload: dict[str, Any]) -> dict[str, float]:
    split = payload["splits"]["test"]
    return {
        "no_shift": float(split["no_shift"]["wer"]),
        "full": float(split["retention"]["full"]["wer"]),
        "utility75": float(split["retention"]["methods"]["Utility"]["75"]["wer"]),
        "utility50": float(split["retention"]["methods"]["Utility"]["50"]["wer"]),
        "magnitude50": float(split["retention"]["methods"]["Magnitude"]["50"]["wer"]),
        "drop_worst25": float(split["deletion"]["methods"]["DropWorst"]["25"]["wer"]),
        "drop_best25": float(split["deletion"]["methods"]["DropBest"]["25"]["wer"]),
    }


def core_conditions(payload: dict[str, Any]) -> dict[str, float]:
    return {
        name: float(payload["splits"]["test"]["conditions"][name]["wer"])
        for name in CORE_CONDITIONS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold0", type=Path, required=True)
    parser.add_argument("--fold1", type=Path, required=True)
    parser.add_argument("--fold2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sources = {
        "Fold0": ("published oracle checkpoint", args.fold0, fold0_conditions),
        "Fold1": ("official-split local replica", args.fold1, core_conditions),
        "Fold2": ("official-split local replica", args.fold2, core_conditions),
    }
    rows: list[dict[str, Any]] = []
    for fold, (label, path, extractor) in sources.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        conditions = extractor(payload)
        rows.append({
            "fold": fold,
            "label": label,
            "metrics_path": str(path),
            "test_examples": int(payload["splits"]["test"].get("examples", 0)),
            "conditions": conditions,
            "identity_pass": bool(payload["splits"]["test"]["identity"].get("pass", payload["splits"]["test"]["identity"].get("identity_pass", False))),
        })

    macro = {
        name: sum(row["conditions"][name] for row in rows) / len(rows)
        for name in CORE_CONDITIONS
    }
    deltas = {
        "utility75_minus_full": [row["conditions"]["utility75"] - row["conditions"]["full"] for row in rows],
        "utility50_minus_full": [row["conditions"]["utility50"] - row["conditions"]["full"] for row in rows],
        "utility50_minus_magnitude50": [row["conditions"]["utility50"] - row["conditions"]["magnitude50"] for row in rows],
        "drop_worst25_minus_full": [row["conditions"]["drop_worst25"] - row["conditions"]["full"] for row in rows],
        "drop_best25_minus_full": [row["conditions"]["drop_best25"] - row["conditions"]["full"] for row in rows],
    }
    payload = {
        "protocol": "official_split_local_replica_frozen_core_crossfold_v1",
        "replica_labels": {"Fold0": "published oracle checkpoint", "Fold1": "official-split local replica", "Fold2": "official-split local replica"},
        "conditions": list(CORE_CONDITIONS),
        "rows": rows,
        "macro_mean": macro,
        "deltas": deltas,
        "macro_mean_deltas": {name: sum(values) / len(values) for name, values in deltas.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "crossfold_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Official-split local-replica cross-fold core summary",
        "",
        "Fold0 is the published oracle checkpoint; Fold1 and Fold2 are locally trained replicas on the upstream official split manifests.",
        "",
        "## Test WER",
        "",
        "| Fold | Provenance | N | NoShift | Full | Utility75 | Utility50 | Magnitude50 | DropWorst25 | DropBest25 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        vals = row["conditions"]
        cells = [row["fold"], row["label"], str(row["test_examples"])] + [f"{100.0 * vals[name]:.3f}%" for name in CORE_CONDITIONS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("| Macro mean | — | — | " + " | ".join(f"{100.0 * macro[name]:.3f}%" for name in CORE_CONDITIONS) + " |")
    lines += [
        "",
        "## Paired condition deltas (percentage points)",
        "",
        "| Fold | U75 − Full | U50 − Full | U50 − M50 | DropWorst25 − Full | DropBest25 − Full |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        vals = row["conditions"]
        delta_values = (
            vals["utility75"] - vals["full"],
            vals["utility50"] - vals["full"],
            vals["utility50"] - vals["magnitude50"],
            vals["drop_worst25"] - vals["full"],
            vals["drop_best25"] - vals["full"],
        )
        lines.append("| " + row["fold"] + " | " + " | ".join(f"{100.0 * value:+.3f}" for value in delta_values) + " |")
    lines.append("| Macro mean | " + " | ".join(f"{100.0 * payload['macro_mean_deltas'][name]:+.3f}" for name in deltas) + " |")
    lines += [
        "",
        "Identity gates: " + ("PASS for all rows." if all(row["identity_pass"] for row in rows) else "REVIEW — at least one row failed."),
        "",
    ]
    (args.output_dir / "crossfold_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(args.output_dir / "crossfold_summary.json"), "markdown": str(args.output_dir / "crossfold_summary.md")}, sort_keys=True))


if __name__ == "__main__":
    main()
