"""MyST manifest validation and immutable audit records."""

from __future__ import annotations

import hashlib
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from usde.text import normalize_text

REQUIRED_FIELDS = {"utt_id", "audio_path", "transcript", "speaker_id"}


@dataclass(frozen=True)
class SplitAudit:
    split: str
    records: int
    speakers: int
    duration_seconds: float
    sha256: str


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED_FIELDS.difference(row)
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}")
            row["utt_id"] = str(row["utt_id"])
            row["speaker_id"] = str(row["speaker_id"])
            row["transcript"] = normalize_text(str(row["transcript"]))
            if not row["transcript"]:
                raise ValueError(f"{path}:{line_number}: empty transcript")
            records.append(row)
    if not records:
        raise ValueError(f"{path}: no records")
    utt_ids = [row["utt_id"] for row in records]
    if len(utt_ids) != len(set(utt_ids)):
        raise ValueError(f"{path}: duplicate utt_id")
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != 16000:
            raise ValueError(f"{path}: expected 16000 Hz, got {wav.getframerate()} Hz")
        if wav.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono, got {wav.getnchannels()} channels")
        return wav.getnframes() / wav.getframerate()


def audit_split(split: str, manifest_path: Path, verify_audio: bool) -> tuple[SplitAudit, set[str]]:
    records = load_jsonl(manifest_path)
    duration = 0.0
    for row in records:
        path = Path(row["audio_path"])
        if verify_audio:
            if not path.is_file():
                raise FileNotFoundError(f"{manifest_path}: missing audio {path}")
            duration += wav_duration_seconds(path)
        else:
            duration += float(row.get("duration_seconds", 0.0))
    speakers = {row["speaker_id"] for row in records}
    return SplitAudit(split, len(records), len(speakers), duration, sha256_file(manifest_path)), speakers


def audit_condition(manifest_root: Path, condition: str, verify_audio: bool) -> dict[str, Any]:
    output: dict[str, Any] = {"condition": condition, "splits": {}}
    all_speakers: dict[str, set[str]] = {}
    for split in ("train", "dev", "test"):
        audit, speakers = audit_split(split, manifest_root / condition / f"{split}.jsonl", verify_audio)
        output["splits"][split] = audit.__dict__
        all_speakers[split] = speakers
    overlaps = {
        "train_dev": sorted(all_speakers["train"] & all_speakers["dev"]),
        "train_test": sorted(all_speakers["train"] & all_speakers["test"]),
        "dev_test": sorted(all_speakers["dev"] & all_speakers["test"]),
    }
    output["speaker_overlap"] = overlaps
    if any(overlaps.values()):
        raise ValueError(f"{condition}: speaker leakage: {overlaps}")
    return output
