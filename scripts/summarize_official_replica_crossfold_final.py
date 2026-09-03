#!/usr/bin/env python3
"""Aggregate frozen seven-condition test results across L2-ARCTIC folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_official_replica_crossfold import CORE_CONDITIONS, core_conditions


DELTA_NAMES = (
    "utility75_minus_full", "utility50_minus_full", "utility50_minus_magnitude50",
    "drop_worst25_minus_full", "drop_best25_minus_full",
)


def fold0_conditions_legacy(payload: dict[str, object]) -> dict[str, float]:
    """Map the frozen Fold0 package's raw-measured names to core names."""
    raw = payload["splits"]["test"]["raw_measured"]
    return {
        "no_shift": float(raw["no_shift"]["wer"]),
        "full": float(raw["full"]["wer"]),
        "utility75": float(raw["utility_75"]["wer"]),
        "utility50": float(raw["utility_50"]["wer"]),
        "magnitude50": float(raw["magnitude_50"]["wer"]),
        "drop_worst25": float(raw["drop_worst_25"]["wer"]),
        "drop_best25": float(raw["drop_best_25"]["wer"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold0", type=Path, required=True)
    parser.add_argument("--fold1", type=Path, required=True)
    parser.add_argument("--fold2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sources = {
        "Fold0": ("published oracle checkpoint", args.fold0, fold0_conditions_legacy),
        "Fold1": ("official-split local replica", args.fold1, core_conditions),
        "Fold2": ("official-split local replica", args.fold2, core_conditions),
    }
    rows = []
    for fold, (label, path, extractor) in sources.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        split = payload["splits"]["test"]
        count = split.get("examples", split.get("raw_measured", {}).get("full", {}).get("examples", 0))
        identity = split["identity"]
        rows.append({
            "fold": fold, "label": label, "metrics_path": str(path),
            "test_examples": int(count), "conditions": extractor(payload),
            "identity_pass": bool(identity.get("pass", identity.get("identity_pass", False))),
        })

    macro = {name: sum(row["conditions"][name] for row in rows) / len(rows) for name in CORE_CONDITIONS}
    delta_values = {name: [] for name in DELTA_NAMES}
    for row in rows:
        vals = row["conditions"]
        values = (
            vals["utility75"] - vals["full"], vals["utility50"] - vals["full"],
            vals["utility50"] - vals["magnitude50"], vals["drop_worst25"] - vals["full"],
            vals["drop_best25"] - vals["full"],
        )
        for name, value in zip(DELTA_NAMES, values, strict=True):
            delta_values[name].append(value)
    macro_deltas = {name: sum(values) / len(values) for name, values in delta_values.items()}
    result = {
        "protocol": "official_split_local_replica_frozen_core_crossfold_v1",
        "replica_labels": {"Fold0": "published oracle checkpoint", "Fold1": "official-split local replica", "Fold2": "official-split local replica"},
        "conditions": list(CORE_CONDITIONS), "rows": rows, "macro_mean": macro,
        "deltas": delta_values, "macro_mean_deltas": macro_deltas,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "crossfold_summary.json"
    markdown_path = args.output_dir / "crossfold_summary.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Official-split local-replica cross-fold core summary", "",
        "Fold0 is the published oracle checkpoint; Fold1 and Fold2 are locally trained replicas on the upstream official split manifests.", "",
        "## Test WER", "",
        "| Fold | Provenance | N | NoShift | Full | Utility75 | Utility50 | Magnitude50 | DropWorst25 | DropBest25 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        vals = row["conditions"]
        cells = [row["fold"], row["label"], str(row["test_examples"])] + [f"{100.0 * vals[name]:.3f}%" for name in CORE_CONDITIONS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("| Macro mean | — | — | " + " | ".join(f"{100.0 * macro[name]:.3f}%" for name in CORE_CONDITIONS) + " |")
    lines += [
        "", "## Paired condition deltas (percentage points)", "",
        "| Fold | U75 − Full | U50 − Full | U50 − M50 | DropWorst25 − Full | DropBest25 − Full |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        vals = row["conditions"]
        values = (
            vals["utility75"] - vals["full"], vals["utility50"] - vals["full"],
            vals["utility50"] - vals["magnitude50"], vals["drop_worst25"] - vals["full"],
            vals["drop_best25"] - vals["full"],
        )
        lines.append("| " + row["fold"] + " | " + " | ".join(f"{100.0 * value:+.3f}" for value in values) + " |")
    lines.append("| Macro mean | " + " | ".join(f"{100.0 * macro_deltas[name]:+.3f}" for name in DELTA_NAMES) + " |")
    lines += ["", "Identity gates: " + ("PASS for all rows." if all(row["identity_pass"] for row in rows) else "REVIEW — at least one row failed."), ""]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
