#!/usr/bin/env python3
"""Build the paper-style L2-ARCTIC unseen-transcript 8-fold manifests.

The recent multi-accent protocol has four held-out-speaker assignments.  Each
assignment is repeated with a different prompt partition, giving eight folds
in total.  In every fold, one speaker per L1 is held out for test; a shared
prompt subset is reserved for that test set, so the remaining speakers cannot
leak those prompts into train or validation.  The held-out speakers' remaining
utterances are intentionally unused for that fold.

The source manifest is never modified.  Outputs are JSONL manifests under a
new namespace, plus a machine-readable split audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


L1_BY_SPEAKER = {
    "ABA": "Arabic",
    "SKA": "Arabic",
    "YBAA": "Arabic",
    "ZHAA": "Arabic",
    "BWC": "Chinese",
    "LXC": "Chinese",
    "NCC": "Chinese",
    "TXHC": "Chinese",
    "ASI": "Hindi",
    "RRBI": "Hindi",
    "SVBI": "Hindi",
    "TNI": "Hindi",
    "HJK": "Korean",
    "HKK": "Korean",
    "YDCK": "Korean",
    "YKWK": "Korean",
    "EBVS": "Spanish",
    "ERMS": "Spanish",
    "MBMPS": "Spanish",
    "NJS": "Spanish",
    "HQTV": "Vietnamese",
    "PNV": "Vietnamese",
    "THV": "Vietnamese",
    "TLV": "Vietnamese",
}

# These are the four assignments used by the public 8-fold L2-ARCTIC setup.
# Fold order is retained so published fold-0 remains the development entry
# point; repeated assignments use independent prompt seeds.
TEST_SPEAKERS_BY_FOLD = (
    ("SKA", "RRBI", "HKK", "LXC", "ERMS", "PNV"),
    ("ZHAA", "TNI", "YKWK", "TXHC", "NJS", "TLV"),
    ("YBAA", "SVBI", "NCC", "YDCK", "THV", "MBMPS"),
    ("ABA", "ASI", "BWC", "HJK", "HQTV", "EBVS"),
    ("ZHAA", "TNI", "YKWK", "TXHC", "NJS", "TLV"),
    ("ABA", "ASI", "BWC", "HJK", "HQTV", "EBVS"),
    ("SKA", "RRBI", "HKK", "LXC", "ERMS", "PNV"),
    ("YBAA", "SVBI", "NCC", "YDCK", "THV", "MBMPS"),
)

SEED = 1337
TEST_RATIO = 0.10
DEV_RATIO = 0.10
SPLITS = ("train", "dev", "test")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"utt_id", "audio_path", "transcript", "speaker_id", "prompt_id"}
            missing = required.difference(row)
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}")
            row["utt_id"] = str(row["utt_id"])
            row["speaker_id"] = str(row["speaker_id"])
            row["prompt_id"] = str(row["prompt_id"])
            row["transcript"] = " ".join(str(row["transcript"]).lower().split())
            if not row["transcript"]:
                raise ValueError(f"{path}:{line_number}: empty transcript")
            if row["speaker_id"] not in L1_BY_SPEAKER:
                raise ValueError(f"{path}:{line_number}: unknown speaker {row['speaker_id']}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no records")
    if len({row["utt_id"] for row in rows}) != len(rows):
        raise ValueError(f"{path}: duplicate utt_id")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def choose_test_prompts(rows: list[dict[str, Any]], fold: int) -> set[str]:
    prompt_ids = sorted({str(row["prompt_id"]) for row in rows})
    n_test = max(1, round(len(prompt_ids) * TEST_RATIO))
    rng = random.Random(SEED + 1009 * (fold + 1))
    return set(rng.sample(prompt_ids, n_test))


def split_train_dev(rows: list[dict[str, Any]], fold: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split prompt/transcript connected groups, not individual rows."""

    # Most speakers share prompt IDs, while a small number of transcript files
    # contain spelling/contraction variants.  Unioning both keys prevents a
    # sentence from crossing train/dev through either representation.
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_by_key: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        for key in (("prompt", str(row["prompt_id"])), ("transcript", str(row["transcript"]))):
            previous = first_by_key.setdefault(key, index)
            union(previous, index)

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[find(index)].append(row)
    groups = list(grouped.values())
    rng = random.Random(SEED + 2003 * (fold + 1))
    rng.shuffle(groups)
    target_dev = max(1, round(len(rows) * DEV_RATIO))
    dev: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    for group in groups:
        if len(dev) < target_dev:
            dev.extend(group)
        else:
            train.extend(group)
    if not train or not dev:
        raise ValueError(f"fold {fold}: train/dev split became empty")
    return sorted(train, key=lambda row: str(row["utt_id"])), sorted(dev, key=lambda row: str(row["utt_id"]))


def validate_fold(
    fold: int,
    split_rows: dict[str, list[dict[str, Any]]],
    test_speakers: tuple[str, ...],
    selected_test_prompts: set[str],
) -> dict[str, Any]:
    split_speakers = {split: {str(row["speaker_id"]) for row in rows} for split, rows in split_rows.items()}
    expected_train_speakers = set(L1_BY_SPEAKER).difference(test_speakers)
    if split_speakers["test"] != set(test_speakers):
        raise ValueError(f"fold {fold}: test speakers mismatch")
    if split_speakers["train"] != expected_train_speakers or split_speakers["dev"] != expected_train_speakers:
        raise ValueError(f"fold {fold}: train/dev speaker pool mismatch")
    if split_speakers["train"] & split_speakers["test"] or split_speakers["dev"] & split_speakers["test"]:
        raise ValueError(f"fold {fold}: speaker leakage")

    transcripts = {split: {str(row["transcript"]) for row in rows} for split, rows in split_rows.items()}
    prompts = {split: {str(row["prompt_id"]) for row in rows} for split, rows in split_rows.items()}
    transcript_overlap = {
        f"{left}_{right}": sorted(transcripts[left] & transcripts[right])
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }
    prompt_overlap = {
        f"{left}_{right}": sorted(prompts[left] & prompts[right])
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }
    if any(transcript_overlap.values()) or any(prompt_overlap.values()):
        raise ValueError(f"fold {fold}: transcript/prompt leakage detected")
    all_ids = [str(row["utt_id"]) for rows in split_rows.values() for row in rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError(f"fold {fold}: duplicate utt_id across splits")

    def counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row[key]) for row in rows).items()))

    return {
        "fold": fold,
        "test_speakers": sorted(test_speakers),
        "test_speakers_by_l1": {l1: speaker for speaker, l1 in sorted((s, L1_BY_SPEAKER[s]) for s in test_speakers)},
        "selected_test_prompt_count": len(selected_test_prompts),
        "splits": {
            split: {
                "utterances": len(split_rows[split]),
                "speakers": sorted(split_speakers[split]),
                "utterances_by_speaker": counts(split_rows[split], "speaker_id"),
                "utterances_by_l1": counts(
                    [{"l1": L1_BY_SPEAKER[str(row["speaker_id"])]} for row in split_rows[split]], "l1"
                ),
                "unique_transcripts": len(transcripts[split]),
                "unique_prompts": len(prompts[split]),
            }
            for split in SPLITS
        },
        "transcript_overlap": {key: {"count": len(value), "examples": value[:10]} for key, value in transcript_overlap.items()},
        "prompt_overlap": {key: {"count": len(value), "examples": value[:10]} for key, value in prompt_overlap.items()},
    }


def build_fold(rows: list[dict[str, Any]], fold: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    test_speakers = TEST_SPEAKERS_BY_FOLD[fold]
    selected_test_prompts = choose_test_prompts(rows, fold)
    test_speaker_set = set(test_speakers)

    test_rows = [
        row for row in rows
        if str(row["speaker_id"]) in test_speaker_set and str(row["prompt_id"]) in selected_test_prompts
    ]
    train_pool = [
        row for row in rows
        if str(row["speaker_id"]) not in test_speaker_set
        and str(row["prompt_id"]) not in selected_test_prompts
        and str(row["transcript"]) not in {str(item["transcript"]) for item in test_rows}
    ]
    train_rows, dev_rows = split_train_dev(train_pool, fold)
    split_rows = {"train": train_rows, "dev": dev_rows, "test": sorted(test_rows, key=lambda row: str(row["utt_id"]))}
    audit = validate_fold(fold, split_rows, test_speakers, selected_test_prompts)
    return split_rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=Path("data/processed/arctic/l2_manifest_16k.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("manifests/l2_arctic_ut8"))
    parser.add_argument("--verify-audio", action="store_true", help="check every emitted WAV exists")
    args = parser.parse_args()

    rows = read_jsonl(args.source_manifest)
    source_sha256 = sha256_file(args.source_manifest)
    source_ids = {str(row["utt_id"]) for row in rows}
    audit: dict[str, Any] = {
        "protocol": "l2_arctic_unseen_transcript_8fold_v1",
        "seed": SEED,
        "source_manifest": str(args.source_manifest),
        "source_sha256": source_sha256,
        "source_utterances": len(rows),
        "source_speakers": sorted({str(row["speaker_id"]) for row in rows}),
        "test_ratio": TEST_RATIO,
        "dev_ratio_of_train_pool": DEV_RATIO,
        "folds": {},
    }

    for fold in range(len(TEST_SPEAKERS_BY_FOLD)):
        split_rows, fold_audit = build_fold(rows, fold)
        fold_root = args.output_root / f"fold{fold}"
        for split in SPLITS:
            if args.verify_audio:
                missing = [str(row["audio_path"]) for row in split_rows[split] if not Path(str(row["audio_path"])).is_file()]
                if missing:
                    raise FileNotFoundError(f"fold {fold}/{split}: missing audio {missing[0]}")
            output_rows = []
            for row in split_rows[split]:
                item = dict(row)
                item["l1"] = L1_BY_SPEAKER[str(row["speaker_id"])]
                item["ut_protocol"] = "unseen_transcript"
                item["fold"] = fold
                item["split"] = split
                output_rows.append(item)
            path = fold_root / f"{split}.jsonl"
            write_jsonl(path, output_rows)
            fold_audit["splits"][split]["path"] = str(path)
            fold_audit["splits"][split]["sha256"] = sha256_file(path)
        fold_audit["emitted_utterances"] = sum(len(split_rows[split]) for split in SPLITS)
        fold_audit["source_utterances_unused"] = len(source_ids.difference({str(row["utt_id"]) for split in split_rows.values() for row in split}))
        audit["folds"][f"fold{fold}"] = fold_audit

    args.output_root.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_root / "split_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
