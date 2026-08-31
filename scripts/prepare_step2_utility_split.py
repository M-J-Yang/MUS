#!/usr/bin/env python3
"""Create a deterministic held-out utility split inside the L2 train split."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + chr(10))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=Path("manifests/arctic_step2/l2/train.jsonl"))
    parser.add_argument("--teacher-out", type=Path, default=Path("manifests/arctic_step2/l2/train_teacher.jsonl"))
    parser.add_argument("--utility-out", type=Path, default=Path("manifests/arctic_step2/l2/train_utility.jsonl"))
    parser.add_argument("--utility-every", type=int, default=10, help="hold out every Nth sorted utterance per speaker")
    args = parser.parse_args()
    if args.utility_every < 2:
        raise ValueError("utility-every must be at least 2")
    rows = read_jsonl(args.train_manifest)
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_speaker[str(row["speaker_id"])].append(row)
    teacher: list[dict[str, Any]] = []
    utility: list[dict[str, Any]] = []
    for speaker in sorted(by_speaker):
        speaker_rows = sorted(by_speaker[speaker], key=lambda row: str(row["utt_id"]))
        for index, row in enumerate(speaker_rows):
            (utility if index % args.utility_every == 0 else teacher).append(row)
    teacher.sort(key=lambda row: str(row["utt_id"]))
    utility.sort(key=lambda row: str(row["utt_id"]))
    if not teacher or not utility or {row["utt_id"] for row in teacher} & {row["utt_id"] for row in utility}:
        raise ValueError("invalid teacher/utility partition")
    write_jsonl(args.teacher_out, teacher)
    write_jsonl(args.utility_out, utility)
    audit = {
        "protocol": "l2_train_teacher_utility_v1",
        "source_train_manifest": str(args.train_manifest),
        "utility_every": args.utility_every,
        "teacher_records": len(teacher),
        "utility_records": len(utility),
        "teacher_sha256": hashlib.sha256(args.teacher_out.read_bytes()).hexdigest(),
        "utility_sha256": hashlib.sha256(args.utility_out.read_bytes()).hexdigest(),
        "speaker_counts": {"teacher": {speaker: sum(row["speaker_id"] == speaker for row in teacher) for speaker in sorted(by_speaker)}, "utility": {speaker: sum(row["speaker_id"] == speaker for row in utility) for speaker in sorted(by_speaker)}},
    }
    audit_path = args.teacher_out.parent / "train_teacher_utility_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
