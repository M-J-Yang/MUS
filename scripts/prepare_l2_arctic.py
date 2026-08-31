#!/usr/bin/env python3
"""Create the frozen speaker-independent L2-ARCTIC TSV manifests.

This is the data-preparation gate for the first experiment stage.  It uses
only the scripted corpus under ``<root>/<speaker>/wav`` and the matching
orthographic transcripts.  The six test speakers are fixed below; dev is a
deterministic 10% per-speaker sample from the remaining 18 speakers.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path
from typing import Iterable


SEED = 42
DEV_RATIO = 0.10

SPEAKER_L1 = {
    # Arabic
    "ABA": "Arabic",
    "SKA": "Arabic",
    "YBAA": "Arabic",
    "ZHAA": "Arabic",
    # Chinese
    "BWC": "Chinese",
    "LXC": "Chinese",
    "NCC": "Chinese",
    "TXHC": "Chinese",
    # Hindi
    "ASI": "Hindi",
    "RRBI": "Hindi",
    "SVBI": "Hindi",
    "TNI": "Hindi",
    # Korean
    "HJK": "Korean",
    "HKK": "Korean",
    "YDCK": "Korean",
    "YKWK": "Korean",
    # Spanish
    "EBVS": "Spanish",
    "ERMS": "Spanish",
    "MBMPS": "Spanish",
    "NJS": "Spanish",
    # Vietnamese
    "HQTV": "Vietnamese",
    "PNV": "Vietnamese",
    "THV": "Vietnamese",
    "TLV": "Vietnamese",
}

TEST_SPEAKERS = {"ZHAA", "TXHC", "TNI", "YKWK", "NJS", "TLV"}

FIELDNAMES = ["utt_id", "wav_path", "transcript", "speaker", "l1"]


def normalize_text(text: str) -> str:
    """Apply the single transcript normalization used by all later stages."""

    text = text.lower()
    text = re.sub(r"[^a-z' ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def collect_speaker_rows(root: Path, speaker: str) -> list[dict[str, str]]:
    """Collect actual scripted WAVs and their matching transcripts."""

    wav_dir = root / speaker / "wav"
    transcript_dir = root / speaker / "transcript"
    if not wav_dir.is_dir():
        raise FileNotFoundError(f"missing WAV directory for {speaker}: {wav_dir}")
    if not transcript_dir.is_dir():
        raise FileNotFoundError(
            f"missing transcript directory for {speaker}: {transcript_dir}"
        )

    rows: list[dict[str, str]] = []
    for wav_path in sorted(wav_dir.glob("*.wav")):
        stem = wav_path.stem
        txt_path = transcript_dir / f"{stem}.txt"
        if not txt_path.exists():
            print(f"Missing transcript: {txt_path}", file=sys.stderr)
            continue

        transcript = normalize_text(txt_path.read_text(encoding="utf-8"))
        if not transcript:
            print(f"Empty transcript: {txt_path}", file=sys.stderr)
            continue

        rows.append(
            {
                "utt_id": f"{speaker}_{stem}",
                "wav_path": str(wav_path.resolve()),
                "transcript": transcript,
                "speaker": speaker,
                "l1": SPEAKER_L1[speaker],
            }
        )
    if not rows:
        raise ValueError(f"no usable scripted utterances found for {speaker}")
    return rows


def write_tsv(rows: Iterable[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def validate_splits(
    train_rows: list[dict[str, str]],
    dev_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
) -> None:
    """Fail before writing manifests if the frozen split contract is broken."""

    split_rows = {"train": train_rows, "dev": dev_rows, "test": test_rows}
    split_speakers = {
        split: {row["speaker"] for row in rows} for split, rows in split_rows.items()
    }

    if split_speakers["test"] != TEST_SPEAKERS:
        raise ValueError(
            "test speakers do not match frozen split: "
            f"expected={sorted(TEST_SPEAKERS)}, "
            f"observed={sorted(split_speakers['test'])}"
        )
    expected_train_pool = set(SPEAKER_L1) - TEST_SPEAKERS
    if split_speakers["train"] != expected_train_pool:
        raise ValueError(
            "train speakers do not match frozen train pool: "
            f"expected={sorted(expected_train_pool)}, "
            f"observed={sorted(split_speakers['train'])}"
        )
    if split_speakers["dev"] != expected_train_pool:
        raise ValueError(
            "dev must contain every training-pool speaker: "
            f"expected={sorted(expected_train_pool)}, "
            f"observed={sorted(split_speakers['dev'])}"
        )

    if split_speakers["train"] & split_speakers["test"]:
        raise ValueError("speaker leakage between train and test")
    if split_speakers["dev"] & split_speakers["test"]:
        raise ValueError("speaker leakage between dev and test")

    all_ids = [row["utt_id"] for rows in split_rows.values() for row in rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("duplicate utt_id across train/dev/test")


def build_splits(root: Path) -> tuple[list[dict[str, str]], ...]:
    """Build train, dev, and test rows from the frozen speaker metadata."""

    train_rows: list[dict[str, str]] = []
    dev_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []

    for speaker in sorted(SPEAKER_L1):
        rows = collect_speaker_rows(root, speaker)
        print(f"{speaker:5s} {SPEAKER_L1[speaker]:10s} {len(rows):4d}")

        if speaker in TEST_SPEAKERS:
            test_rows.extend(rows)
            continue

        rng = random.Random(f"{SEED}_{speaker}")
        rng.shuffle(rows)
        n_dev = round(len(rows) * DEV_RATIO)
        dev_rows.extend(rows[:n_dev])
        train_rows.extend(rows[n_dev:])

    validate_splits(train_rows, dev_rows, test_rows)
    return train_rows, dev_rows, test_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/l2_arctic"),
        help="directory containing the 24 speaker directories",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="directory receiving train.tsv, dev.tsv, and test.tsv",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"L2-ARCTIC root does not exist: {root}")

    train_rows, dev_rows, test_rows = build_splits(root)
    write_tsv(train_rows, args.out_dir / "train.tsv")
    write_tsv(dev_rows, args.out_dir / "dev.tsv")
    write_tsv(test_rows, args.out_dir / "test.tsv")

    print()
    print(f"train: {len(train_rows)}")
    print(f"dev:   {len(dev_rows)}")
    print(f"test:  {len(test_rows)}")
    print(f"wrote: {args.out_dir / 'train.tsv'}")
    print(f"wrote: {args.out_dir / 'dev.tsv'}")
    print(f"wrote: {args.out_dir / 'test.tsv'}")


if __name__ == "__main__":
    main()
