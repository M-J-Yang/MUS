#!/usr/bin/env python3
"""Summarize frozen-head official Fold0 shift-retention evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RETAINED = (100, 75, 50)


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _row(report: dict[str, Any], retention: int, full_wer: float, no_shift_wer: float) -> dict[str, Any]:
    if retention == 0:
        wer = no_shift_wer
        source = "no_shift_e0"
    else:
        cached = report.get("cached", {})
        retained = cached.get("retained_delta", {})
        if not isinstance(retained, dict) or "wer" not in retained:
            raise ValueError(f"{retention}% report has no retained_delta WER")
        wer = float(retained["wer"])
        source = "retained_delta"
    gain = no_shift_wer - wer
    full_gain = no_shift_wer - full_wer
    return {
        "shift_retained_percent": retention,
        "wer": wer,
        "wer_percent": 100.0 * wer,
        "delta_wer_vs_full": wer - full_wer,
        "delta_wer_vs_full_percentage_points": 100.0 * (wer - full_wer),
        "gain_retained": gain,
        "gain_retained_fraction": None if full_gain == 0.0 else gain / full_gain,
        "gain_retained_percent": None if full_gain == 0.0 else 100.0 * gain / full_gain,
        "source": source,
    }


def _format_percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}%"


def _format_pp(value: float) -> str:
    return "0.000 pp" if abs(value) < 0.0005 else f"{value:+.3f} pp"


def summarize_split(results_dir: Path, split: str) -> dict[str, Any]:
    reports = {retention: _read(results_dir / f"{split}_{retention}.json") for retention in RETAINED}
    for retention, report in reports.items():
        if not bool(report.get("gate_pass")):
            raise RuntimeError(f"{split} {retention}% report failed the identity gate")
    full = float(reports[100]["cached"]["full_delta_reconstruction"]["wer"])
    no_shift = float(reports[100]["cached"]["no_shift_e0"]["wer"])
    rows = [_row(reports[100], 0, full, no_shift)]
    rows.extend(_row(reports[retention], retention, full, no_shift) for retention in (50, 75, 100))
    return {
        "split": split,
        "checkpoint": reports[100]["checkpoint"],
        "manifest": reports[100]["manifest"],
        "full_wer": full,
        "full_wer_percent": 100.0 * full,
        "no_shift_wer": no_shift,
        "no_shift_wer_percent": 100.0 * no_shift,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    splits = {split: summarize_split(args.results_dir, split) for split in ("dev", "test")}
    result: dict[str, Any] = {
        "protocol": "official_fold0_oracle_frozen_head_shift_pruning_summary_v1",
        "retentions_percent": [0, 50, 75, 100],
        "splits": splits,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    test_rows = splits["test"]["rows"]
    lines = [
        "# Official Fold0 oracle frozen-head shift pruning",
        "",
        f"Checkpoint: `{splits['test']['checkpoint']}`",
        "",
        "| Shift retained | Test WER | ΔWER vs full | Gain retained |",
        "|---:|---:|---:|---:|",
    ]
    for row in test_rows:
        gain = row["gain_retained_percent"]
        gain_text = "—" if gain is None else f"{gain:.1f}%"
        lines.append(
            f"| {row['shift_retained_percent']}% | "
            f"{_format_percent(row['wer_percent'])} | "
            f"{_format_pp(row['delta_wer_vs_full_percentage_points'])} | "
            f"{gain_text} |"
        )
    lines.extend(
        [
            "",
            "The 0% row is the same frozen oracle head applied to E0; 50/75/100% "
            "retain the top Taylor-utility coordinates from official train_utility.",
            "",
        ]
    )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
