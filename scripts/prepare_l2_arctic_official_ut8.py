#!/usr/bin/env python3
"""Materialize the public robust-atc-asr Fold-0 CSV protocol.

The existing ``manifests/l2_arctic_ut8`` namespace is intentionally left
untouched.  This script maps the author's CSV rows onto the locally prepared
16-kHz audio, emits a separate JSONL namespace, and records an explicit
comparison against the previous local Fold-0 split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def stable_text_id(text: str, n_hex: int = 8) -> int:
    return int(hashlib.md5(normalize_text(text).upper().encode("utf-8")).hexdigest()[:n_hex], 16)


def audio_key(row: dict[str, Any]) -> tuple[str, str]:
    raw = row.get("audio_filename", row.get("audio_path", ""))
    return (str(row.get("speaker", row.get("speaker_id", ""))), Path(str(raw)).name)


def prompt_key(row: dict[str, Any]) -> str:
    raw = row.get("audio_filename", row.get("audio_path", ""))
    return Path(str(raw)).stem


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_official(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"index", "speaker", "l1", "audio_filename", "transcript"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path}: expected CSV fields {sorted(required)}")
    for row in rows:
        row["index"] = int(row["index"])
        row["transcript"] = normalize_text(row["transcript"])
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def split_stats(rows: list[dict[str, Any]], speaker_field: str = "speaker_id") -> dict[str, Any]:
    speakers = Counter(str(row[speaker_field]) for row in rows)
    prompts = {prompt_key(row) for row in rows}
    transcripts = {normalize_text(row["transcript"]) for row in rows}
    return {
        "utterances": len(rows),
        "speakers": sorted(speakers),
        "utterances_by_speaker": dict(sorted(speakers.items())),
        "unique_prompts": len(prompts),
        "unique_transcripts": len(transcripts),
    }


def overlaps(data: dict[str, list[dict[str, Any]]], key_fn) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = key_fn(data[left]) & key_fn(data[right])
        result[f"{left}_{right}"] = {"count": len(shared), "examples": sorted(shared)[:10]}
    return result


def set_for(rows: list[dict[str, Any]], field: str) -> set[str]:
    return {normalize_text(row[field]) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path("artifacts/protocol_audit/official_l2_arctic_8fold_0"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/processed/arctic/l2_manifest_16k.jsonl"),
    )
    parser.add_argument(
        "--current-root",
        type=Path,
        default=Path("manifests/l2_arctic_ut8/fold0"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("manifests/l2_arctic_official_ut8/fold0"),
    )
    parser.add_argument(
        "--audit-out",
        type=Path,
        default=Path("artifacts/protocol_audit/official_l2_arctic_8fold_0/comparison.json"),
    )
    parser.add_argument("--verify-audio", action="store_true")
    args = parser.parse_args()

    source_rows = read_jsonl(args.source_manifest)
    source_by_audio = {audio_key(row): row for row in source_rows}
    if len(source_by_audio) != len(source_rows):
        raise ValueError("source manifest has duplicate speaker+audio keys")

    official: dict[str, list[dict[str, Any]]] = {
        split: read_official(args.official_root / f"{split}.csv")
        for split in ("train", "val", "test")
    }
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_source: list[tuple[str, str]] = []
    missing_audio: list[str] = []
    for split, rows in official.items():
        for row in rows:
            key = audio_key(row)
            source = source_by_audio.get(key)
            if source is None:
                missing_source.append(key)
                continue
            item = dict(source)
            item.update(
                {
                    "audio_path": str(source["audio_path"]),
                    "canonical_transcript": source.get("canonical_transcript", row["transcript"]),
                    "dataset": "l2_arctic",
                    "fold": 0,
                    "l1": str(row["l1"]),
                    "official_index": row["index"],
                    "official_audio_filename": str(row["audio_filename"]),
                    "prompt_id": prompt_key(row),
                    "split": "dev" if split == "val" else split,
                    "supcon_id": stable_text_id(row["transcript"]) if split == "train" else -1,
                    "transcript": row["transcript"],
                    "transcript_source": "official:robust-atc-asr/files/Arctic/8fold/0",
                    "ut_protocol": "official_repeated_transcript_8fold",
                    "utt_id": f"l2_{row['speaker']}_{prompt_key(row)}",
                }
            )
            if args.verify_audio and not Path(str(item["audio_path"])).is_file():
                missing_audio.append(str(item["audio_path"]))
            output[split].append(item)

    if missing_source:
        raise FileNotFoundError(f"source manifest missing {len(missing_source)} official rows; first={missing_source[0]}")
    if missing_audio:
        raise FileNotFoundError(f"missing {len(missing_audio)} emitted audio files; first={missing_audio[0]}")

    for split in ("train", "val", "test"):
        output[split].sort(key=lambda row: int(row["official_index"]))
        write_jsonl(args.output_root / f"{split if split != 'val' else 'dev'}.jsonl", output[split])

    current = {
        "train": read_jsonl(args.current_root / "train.jsonl"),
        "val": read_jsonl(args.current_root / "dev.jsonl"),
        "test": read_jsonl(args.current_root / "test.jsonl"),
    }
    official_audio = {split: {audio_key(row) for row in rows} for split, rows in official.items()}
    current_audio = {split: {audio_key(row) for row in rows} for split, rows in current.items()}
    common = set().union(*official_audio.values()) & set().union(*current_audio.values())
    official_assignment = {key: split for split, keys in official_audio.items() for key in keys}
    current_assignment = {key: split for split, keys in current_audio.items() for key in keys}
    confusion = Counter((official_assignment[key], current_assignment[key]) for key in common)

    grouped = Counter(int(row["supcon_id"]) for row in output["train"])
    group_sizes = Counter(len([x for x in grouped.values() if x == size]) for size in [])
    group_size_histogram = Counter(str(size) for size in grouped.values())
    official_sets = {
        split: {
            "audio": {audio_key(row) for row in rows},
            "prompt": {prompt_key(row) for row in rows},
            "transcript": set_for(rows, "transcript"),
        }
        for split, rows in official.items()
    }
    current_sets = {
        split: {
            "audio": {audio_key(row) for row in rows},
            "prompt": {prompt_key(row) for row in rows},
            "transcript": set_for(rows, "transcript"),
        }
        for split, rows in current.items()
    }
    comparison: dict[str, Any] = {
        "protocol": "official_robust_atc_asr_arctic_8fold_0_vs_local_ut8_fold0",
        "official_source": {
            "repository": "https://github.com/thaivanphat95/robust-atc-asr",
            "prefix": "files/Arctic/8fold/0",
            "files": {split: str(args.official_root / f"{split}.csv") for split in official},
            "sha256": {split: sha256_file(args.official_root / f"{split}.csv") for split in official},
        },
        "official": {
            "splits": {split: split_stats(rows, "speaker") for split, rows in official.items()},
            "prompt_overlap": overlaps(official, lambda rows: {prompt_key(row) for row in rows}),
            "transcript_overlap": overlaps(official, lambda rows: set_for(rows, "transcript")),
            "supcon_train": {
                "total": len(output["train"]),
                "unique_ids": len(grouped),
                "groups_with_at_least_2": sum(1 for value in grouped.values() if value >= 2),
                "group_size_histogram": dict(sorted(group_size_histogram.items(), key=lambda item: int(item[0]))),
                "known_ratio": 1.0,
            },
        },
        "current": {
            "splits": {split: split_stats(rows) for split, rows in current.items()},
            "prompt_overlap": overlaps(current, lambda rows: {prompt_key(row) for row in rows}),
            "transcript_overlap": overlaps(current, lambda rows: set_for(rows, "transcript")),
        },
        "official_vs_current": {
            "audio_intersection_by_split": {
                split: {
                    "intersection": len(official_sets[split]["audio"] & current_sets[split]["audio"]),
                    "official_count": len(official_sets[split]["audio"]),
                    "current_count": len(current_sets[split]["audio"]),
                }
                for split in ("train", "val", "test")
            },
            "exact_test_audio_overlap": len(official_sets["test"]["audio"] & current_sets["test"]["audio"]),
            "split_assignment_confusion_on_common_audio": {
                f"official_{left}__current_{right}": count
                for (left, right), count in sorted(confusion.items())
            },
            "official_train_vs_current_train_transcript_delta": {
                "official_only": len(official_sets["train"]["transcript"] - current_sets["train"]["transcript"]),
                "current_only": len(current_sets["train"]["transcript"] - official_sets["train"]["transcript"]),
            },
        },
        "emitted_manifests": {
            "root": str(args.output_root),
            "paths": {split: str(args.output_root / f"{split if split != 'val' else 'dev'}.jsonl") for split in official},
            "sha256": {
                split: sha256_file(args.output_root / f"{split if split != 'val' else 'dev'}.jsonl")
                for split in official
            },
        },
    }
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
