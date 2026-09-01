#!/usr/bin/env python3
"""Build the Stage 7C same-text L2/CMU control manifests.

The L2 rows are the frozen Stage 5 ``train_utility`` examples. Every L2 row
is matched by the ARCTIC ``prompt_id`` to every available CMU native recording
for that prompt. The CMU prompt transcript is the shared target; source L2
annotation variants are retained for audit instead of being used as targets.

The output contains two side manifests for feature extraction and one
auditable pair manifest. Side-manifest utterance IDs remain the original IDs
so the existing L2 cache can be reused without copying or recomputing it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from usde.text import normalize_text  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"{path}: no records")
    return rows


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

def _text_key(text: str) -> str:
    """Fold spacing and punctuation while retaining lexical characters."""
    return "".join(character for character in normalize_text(text) if character.isalnum())


def _required(row: dict[str, Any], path: Path, index: int) -> tuple[str, str, str, str, str]:
    try:
        utt_id = str(row["utt_id"])
        speaker_id = str(row["speaker_id"])
        prompt_id = str(row["prompt_id"])
        audio_path = str(row["audio_path"])
        transcript = normalize_text(str(row["transcript"]))
    except KeyError as error:
        raise ValueError(f"{path}:{index}: missing field {error.args[0]!r}") from error
    if not all((utt_id, speaker_id, prompt_id, audio_path, transcript)):
        raise ValueError(f"{path}:{index}: utt/speaker/prompt/audio/transcript must be non-empty")
    return utt_id, speaker_id, prompt_id, audio_path, transcript


def build_matched_manifests(
    l2_rows: list[dict[str, Any]],
    cmu_rows: list[dict[str, Any]],
    l2_path: Path = Path("<l2>"),
    cmu_path: Path = Path("<cmu>"),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return L2 side rows, CMU side rows, pair rows, and an audit summary."""

    cmu_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cmu_text_key_by_prompt: dict[str, str] = {}
    cmu_transcript_by_prompt: dict[str, str] = {}
    cmu_seen_ids: set[str] = set()
    for index, row in enumerate(cmu_rows, start=1):
        utt_id, _speaker_id, prompt_id, _audio_path, transcript = _required(row, cmu_path, index)
        if utt_id in cmu_seen_ids:
            raise ValueError(f"{cmu_path}: duplicate utt_id {utt_id}")
        cmu_seen_ids.add(utt_id)
        previous = cmu_text_key_by_prompt.setdefault(prompt_id, _text_key(transcript))
        cmu_transcript_by_prompt.setdefault(prompt_id, transcript)
        if previous != _text_key(transcript):
            raise ValueError(f"{cmu_path}: prompt {prompt_id} has inconsistent transcripts")
        cmu_by_prompt[prompt_id].append(row)
    for rows in cmu_by_prompt.values():
        rows.sort(key=lambda row: (str(row["speaker_id"]), str(row["utt_id"])))

    l2_side: list[dict[str, Any]] = []
    cmu_side_by_id: dict[str, dict[str, Any]] = {}
    pairs: list[dict[str, Any]] = []
    l2_seen_ids: set[str] = set()
    missing_prompts: list[str] = []
    l2_transcript_difference_count = 0
    for index, source_row in enumerate(l2_rows, start=1):
        utt_id, l2_speaker, prompt_id, l2_audio_path, transcript = _required(source_row, l2_path, index)
        if utt_id in l2_seen_ids:
            raise ValueError(f"{l2_path}: duplicate utt_id {utt_id}")
        l2_seen_ids.add(utt_id)
        candidates = cmu_by_prompt.get(prompt_id, [])
        if not candidates:
            missing_prompts.append(prompt_id)
            continue
        canonical_transcript = cmu_transcript_by_prompt[prompt_id]
        l2_transcript_difference_count += int(_text_key(transcript) != _text_key(canonical_transcript))

        l2_row = dict(source_row)
        l2_row["shared_prompt_id"] = prompt_id
        l2_row["source_transcript"] = transcript
        l2_row["transcript"] = canonical_transcript
        l2_side.append(l2_row)
        for cmu_source_row in candidates:
            cmu_row = dict(cmu_source_row)
            cmu_row["shared_prompt_id"] = prompt_id
            cmu_row["source_transcript"] = normalize_text(str(cmu_source_row["transcript"]))
            cmu_row["transcript"] = canonical_transcript
            cmu_side_by_id[str(cmu_row["utt_id"])] = cmu_row
            pairs.append(
                {
                    "shared_prompt_id": prompt_id,
                    "l2_utt_id": utt_id,
                    "l2_speaker_id": l2_speaker,
                    "l2_audio_path": l2_audio_path,
                    "cmu_utt_id": str(cmu_source_row["utt_id"]),
                    "cmu_speaker_id": str(cmu_source_row["speaker_id"]),
                    "cmu_audio_path": str(cmu_source_row["audio_path"]),
                    "transcript": canonical_transcript,
                    "l2_transcript": transcript,
                    "cmu_transcript": canonical_transcript,
                }
            )

    if missing_prompts:
        counts = dict(Counter(missing_prompts))
        raise ValueError(f"CMU has no recording for L2 prompts: {counts}")
    if not l2_side or not cmu_side_by_id or not pairs:
        raise ValueError("matched-text manifest is empty")

    cmu_side = sorted(cmu_side_by_id.values(), key=lambda row: str(row["utt_id"]))
    pairs.sort(key=lambda row: (row["l2_utt_id"], row["cmu_speaker_id"], row["cmu_utt_id"]))
    summary: dict[str, Any] = {
        "protocol": "stage7c_cmu_matched_text_control_v1",
        "matching_key": "prompt_id",
        "text_validation": "prompt_id match; both side manifests use the CMU canonical prompt transcript",
        "l2_transcript_difference_count": l2_transcript_difference_count,
        "l2_records": len(l2_side),
        "cmu_records": len(cmu_side),
        "matched_pairs": len(pairs),
        "shared_prompts": len({row["shared_prompt_id"] for row in pairs}),
        "l2_speakers": dict(sorted(Counter(str(row["speaker_id"]) for row in l2_side).items())),
        "cmu_speakers": dict(sorted(Counter(str(row["speaker_id"]) for row in cmu_side).items())),
        "pair_count_per_prompt": dict(sorted(Counter(row["shared_prompt_id"] for row in pairs).items())),
        "transcripts_verified": True,
    }
    return l2_side, cmu_side, pairs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l2-manifest", type=Path, default=Path("manifests/arctic_step2/l2/train_utility.jsonl"))
    parser.add_argument("--cmu-manifest", type=Path, default=Path("data/processed/arctic/cmu_manifest.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("manifests/stage7c"))
    args = parser.parse_args()

    l2_rows, cmu_rows = read_jsonl(args.l2_manifest), read_jsonl(args.cmu_manifest)
    l2_side, cmu_side, pairs, summary = build_matched_manifests(
        l2_rows, cmu_rows, args.l2_manifest, args.cmu_manifest
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    l2_path = args.output_dir / "l2.jsonl"
    cmu_path = args.output_dir / "cmu.jsonl"
    pairs_path = args.output_dir / "matched_pairs.jsonl"
    write_jsonl(l2_path, l2_side)
    write_jsonl(cmu_path, cmu_side)
    write_jsonl(pairs_path, pairs)
    summary.update(
        {
            "source_l2_manifest": str(args.l2_manifest),
            "source_cmu_manifest": str(args.cmu_manifest),
            "l2_manifest": str(l2_path),
            "cmu_manifest": str(cmu_path),
            "matched_manifest": str(pairs_path),
            "sha256": {
                "l2_manifest": sha256_file(l2_path),
                "cmu_manifest": sha256_file(cmu_path),
                "matched_manifest": sha256_file(pairs_path),
            },
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
