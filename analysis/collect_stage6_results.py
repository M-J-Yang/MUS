#!/usr/bin/env python3
"""Collect the Stage 6 matched-budget WER table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SELECTIONS = ("random", "magnitude", "utility")
UTILITY_VARIANTS = ("utility_v2", "utility_v3", "utility_v4")
K_VALUES = (256, 512)


def _read_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing metrics file: {path}; run the corresponding Stage 6 experiment first"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    if payload.get("test_wer") is None:
        raise ValueError(f"{path}: test_wer is not available")
    return payload


def _stage4_row(
    path: Path,
    label: str,
    feature: str,
    delta_dims: int | None,
) -> dict[str, Any]:
    metrics = _read_metrics(path)
    if delta_dims is None:
        input_dim = int(metrics["input_dim"])
        base_dim = int(metrics.get("base_dim", input_dim))
        delta_dims = input_dim - base_dim
        if delta_dims < 0:
            raise ValueError(f"{path}: input_dim is smaller than base_dim")
    return {
        "input": label,
        "feature": feature,
        "selection": "baseline",
        "k": None,
        "delta_dims": delta_dims,
        "best_epoch": metrics.get("best_epoch"),
        "best_dev_wer": metrics.get("best_dev_wer"),
        "test_wer": metrics["test_wer"],
        "metrics_path": str(path),
    }


def _stage6_row(path: Path, selection: str, k: int) -> dict[str, Any]:
    metrics = _read_metrics(path)
    if metrics.get("selection") != selection:
        raise ValueError(
            f"{path}: expected selection {selection!r}, got {metrics.get('selection')!r}"
        )
    if int(metrics.get("k", -1)) != k:
        raise ValueError(f"{path}: expected k={k}, got {metrics.get('k')!r}")
    labels = {
        "utility_v2": "Utility v2",
        "utility_v3": "Utility v3",
        "utility_v4": "Utility v4 (CTC Taylor)",
    }
    return {
        "input": labels.get(selection, selection.capitalize()),
        "feature": f"[E_ref; Delta_{selection}, K={k}]",
        "selection": selection,
        "k": k,
        "delta_dims": k,
        "best_epoch": metrics.get("best_epoch"),
        "best_dev_wer": metrics.get("best_dev_wer"),
        "test_wer": metrics["test_wer"],
        "metrics_path": str(path),
    }


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Input | Delta dims | Best dev WER | Test WER |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['input']} | {row['delta_dims']} | "
            f"{row['best_dev_wer']:.6f} | {row['test_wer']:.6f} |"
        )
    return "\n".join(lines)


def collect(stage4_root: Path, stage6_root: Path) -> dict[str, Any]:
    reference_metrics = stage4_root / "ref" / "metrics.json"
    full_delta_metrics = stage4_root / "full_delta" / "metrics.json"
    rows = [
        _stage4_row(reference_metrics, "Reference only", "E_ref", 0),
        _stage4_row(full_delta_metrics, "FullDelta", "[E_ref; Delta]", None),
    ]
    for k in K_VALUES:
        for selection in SELECTIONS:
            rows.append(
                _stage6_row(stage6_root / selection / f"k{k}" / "metrics.json", selection, k)
            )
    for selection in UTILITY_VARIANTS:
        for k in K_VALUES:
            metrics_path = stage6_root / selection / f"k{k}" / "metrics.json"
            if metrics_path.is_file():
                rows.append(_stage6_row(metrics_path, selection, k))
    return {
        "protocol": "stage6_selected_delta_linear_ctc_v1",
        "stage4_root": str(stage4_root),
        "stage6_root": str(stage6_root),
        "rows": rows,
        "markdown": _markdown(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4-root", type=Path, default=Path("artifacts/runs/stage4"))
    parser.add_argument("--stage6-root", type=Path, default=Path("results/stage6"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = collect(args.stage4_root, args.stage6_root)
    output = args.output or (args.stage6_root / "stage6_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report["markdown"])


if __name__ == "__main__":
    main()
