#!/usr/bin/env python3
"""Freeze speaker-disjoint train/dev/test manifests for the ARCTIC Step 2 audit.

L2-ARCTIC and CMU ARCTIC are written as separate conditions. The CMU control
condition is never concatenated with L2 training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

L2_SPLITS = {
    "train": ["ABA", "ASI", "BWC", "EBVS", "ERMS", "HJK", "HKK", "HQTV", "LXC", "MBMPS", "NCC", "NJS", "PNV", "RRBI", "SKA", "SVBI"],
    "dev": ["THV", "TLV", "TNI", "TXHC"],
    "test": ["YBAA", "YDCK", "YKWK", "ZHAA"],
}
CMU_SPLITS = {
    "train": ["bdl", "clb"],
    "dev": ["rms"],
    "test": ["slt"],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_condition(rows: list[dict[str, Any]], split_speakers: dict[str, list[str]], output_root: Path, condition: str) -> dict[str, Any]:
    expected = {speaker for speakers in split_speakers.values() for speaker in speakers}
    observed = {str(row["speaker_id"]) for row in rows}
    if observed != expected:
        raise ValueError(f"{condition}: speaker mismatch; missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}")
    assigned = {speaker: split for split, speakers in split_speakers.items() for speaker in speakers}
    if len(assigned) != len(expected):
        raise ValueError(f"{condition}: speaker appears in more than one split")
    condition_root = output_root / condition
    audit: dict[str, Any] = {"condition": condition, "speaker_splits": split_speakers, "splits": {}}
    for split in ("train", "dev", "test"):
        split_rows = sorted((row for row in rows if assigned[str(row["speaker_id"])] == split), key=lambda row: str(row["utt_id"]))
        if not split_rows:
            raise ValueError(f"{condition}/{split}: empty split")
        path = condition_root / f"{split}.jsonl"
        write_jsonl(path, split_rows)
        audit["splits"][split] = {
            "records": len(split_rows),
            "speakers": sorted({str(row["speaker_id"]) for row in split_rows}),
            "records_by_speaker": dict(sorted(Counter(str(row["speaker_id"]) for row in split_rows).items())),
            "sha256": sha256_file(path),
        }
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l2-manifest", type=Path, default=Path("data/processed/arctic/l2_manifest_16k.jsonl"))
    parser.add_argument("--cmu-manifest", type=Path, default=Path("data/processed/arctic/cmu_manifest.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("manifests/arctic_step2"))
    args = parser.parse_args()
    l2_audit = freeze_condition(read_jsonl(args.l2_manifest), L2_SPLITS, args.output_root, "l2")
    cmu_audit = freeze_condition(read_jsonl(args.cmu_manifest), CMU_SPLITS, args.output_root, "cmu")
    audit = {"protocol": "arctic_step2_v1", "l2": l2_audit, "cmu": cmu_audit}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "split_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
