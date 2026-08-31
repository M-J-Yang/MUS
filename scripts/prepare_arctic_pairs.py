#!/usr/bin/env python3
"""Build auditable CMU ARCTIC and L2-ARCTIC manifests.

The L2-ARCTIC scripted prompts reuse CMU ARCTIC prompt IDs. This utility
keeps original audio untouched and writes JSONL indexes. It can run in
CMU-only mode while an authorized L2 archive is being obtained.
"""
from __future__ import annotations
import argparse, json, re, sys, wave
from collections import Counter
from pathlib import Path
PROMPT_RE = re.compile(r"^\(\s*(\S+)\s+\"(.*)\"\s*\)\s*$")
ID_RE = re.compile(r"(arctic_[ab]\d{4})", re.IGNORECASE)

def parse_cmu_transcripts(cmu_root: Path, speakers: list[str]) -> dict[str, dict[str, str]]:
    result = {}
    for speaker in speakers:
        root = cmu_root / "ARCTIC" / f"cmu_us_{speaker}_arctic"
        transcript_path = root / "etc" / "txt.done.data"
        if not transcript_path.is_file(): raise FileNotFoundError(f"missing CMU transcript: {transcript_path}")
        prompts = {}
        for line_no, line in enumerate(transcript_path.read_text(encoding="utf-8").splitlines(), 1):
            match = PROMPT_RE.match(line.strip())
            if not match: raise ValueError(f"cannot parse {transcript_path}:{line_no}: {line!r}")
            prompt_id, text = match.groups(); prompts[prompt_id.lower()] = text
        result[speaker] = prompts
    return result

def wav_info(path: Path) -> dict[str, int]:
    try:
        with wave.open(str(path), "rb") as handle:
            return {"sample_rate": handle.getframerate(), "channels": handle.getnchannels(), "frames": handle.getnframes()}
    except (wave.Error, OSError): return {}

def rel(path: Path, project_root: Path) -> str:
    try:
        return str(path.absolute().relative_to(project_root.absolute()))
    except ValueError:
        return str(path.absolute())

def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records: handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

def build_cmu_records(cmu_root: Path, project_root: Path, prompts: dict[str, dict[str, str]]) -> tuple[list[dict], list[str]]:
    records = []; missing = []
    for speaker, speaker_prompts in prompts.items():
        audio_root = cmu_root / "ARCTIC" / f"cmu_us_{speaker}_arctic" / "wav"
        for wav_path in sorted(audio_root.glob("*.wav")):
            match = ID_RE.search(wav_path.stem)
            if not match: continue
            prompt_id = match.group(1).lower()
            if prompt_id not in speaker_prompts:
                missing.append(f"{speaker}:{prompt_id}"); continue
            info = wav_info(wav_path)
            record = {"utt_id": f"cmu_{speaker}_{prompt_id}", "dataset": "cmu_arctic", "speaker_id": speaker, "prompt_id": prompt_id, "audio_path": rel(wav_path, project_root), "transcript": speaker_prompts[prompt_id]}
            record.update(info); records.append(record)
    return records, missing

def infer_l2_speaker(wav_path: Path, l2_root: Path) -> str:
    relative = wav_path.absolute().relative_to(l2_root.absolute())
    parts = list(relative.parts)
    for index, part in enumerate(parts[:-1]):
        if part.lower() in {"wav", "wavs", "wave"} and index > 0:
            return parts[index - 1]
    return parts[0] if len(parts) > 1 else "unknown"

def build_l2_records(l2_root: Path, project_root: Path, canonical_prompts: dict[str, str]) -> list[dict]:
    records = []
    for wav_path in sorted(l2_root.rglob("*.wav")):
        match = ID_RE.search(wav_path.stem)
        if not match: continue
        prompt_id = match.group(1).lower(); text = canonical_prompts.get(prompt_id)
        if text is None: continue
        info = wav_info(wav_path); speaker = infer_l2_speaker(wav_path, l2_root)
        transcript_path = wav_path.parent.parent / "transcript" / f"{wav_path.stem}.txt"
        source_text = transcript_path.read_text(encoding="utf-8").strip() if transcript_path.is_file() else text
        record = {"utt_id": f"l2_{speaker}_{prompt_id}", "dataset": "l2_arctic", "speaker_id": speaker, "prompt_id": prompt_id, "audio_path": rel(wav_path, project_root), "transcript": source_text, "canonical_transcript": text, "transcript_source": rel(transcript_path, project_root) if transcript_path.is_file() else "cmu_prompt_fallback"}
        record.update(info); records.append(record)
    return records

def build_suitcase_records(l2_root: Path, project_root: Path) -> list[dict]:
    records = []
    audio_root = l2_root / "suitcase_corpus" / "wav"
    transcript_root = l2_root / "suitcase_corpus" / "transcript"
    for wav_path in sorted(audio_root.glob("*.wav")):
        speaker = wav_path.stem.lower()
        transcript_path = transcript_root / f"{wav_path.stem}.txt"
        if not transcript_path.is_file():
            transcript_path = transcript_root / f"{speaker}.txt"
        text = transcript_path.read_text(encoding="utf-8").strip() if transcript_path.is_file() else ""
        info = wav_info(wav_path)
        record = {"utt_id": f"suitcase_{speaker}", "dataset": "l2_arctic_suitcase", "speaker_id": speaker, "audio_path": rel(wav_path, project_root), "transcript": text}
        record.update(info); records.append(record)
    return records

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmu-root", type=Path, default=Path("data/raw/cmu_arctic")); parser.add_argument("--l2-root", type=Path, default=Path("data/raw/l2_arctic")); parser.add_argument("--output-dir", type=Path, default=Path("data/processed/arctic")); parser.add_argument("--speakers", nargs="+", default=["bdl", "slt", "clb", "rms"]); parser.add_argument("--project-root", type=Path, default=Path(".")); parser.add_argument("--allow-missing-l2", action="store_true")
    args = parser.parse_args(); project_root = args.project_root.resolve(); cmu_root = args.cmu_root.resolve(); l2_root = args.l2_root.resolve(); output_dir = args.output_dir.resolve(); speakers = [s.lower() for s in args.speakers]
    prompts_by_speaker = parse_cmu_transcripts(cmu_root, speakers); cmu_records, missing_cmu = build_cmu_records(cmu_root, project_root, prompts_by_speaker); write_jsonl(output_dir / "cmu_manifest.jsonl", cmu_records)
    expected_counts = {speaker: len(prompts) for speaker, prompts in prompts_by_speaker.items()}; actual_counts = Counter(record["speaker_id"] for record in cmu_records)
    for speaker, expected in expected_counts.items():
        if actual_counts[speaker] != expected: raise RuntimeError(f"CMU {speaker}: {actual_counts[speaker]} audio files, {expected} transcripts")
    sample_rates = Counter(record.get("sample_rate") for record in cmu_records if "sample_rate" in record)
    l2_wavs = list(l2_root.rglob("*.wav")) if l2_root.is_dir() else []
    summary = {"cmu_root": str(cmu_root), "l2_root": str(l2_root), "speakers": speakers, "cmu_transcript_counts": expected_counts, "cmu_audio_counts": dict(actual_counts), "cmu_sample_rates": {str(k): v for k, v in sample_rates.items()}, "cmu_audio_without_transcript": missing_cmu, "l2_available": bool(l2_wavs), "status": "cmu_only_l2_missing"}
    if not l2_wavs:
        if not args.allow_missing_l2: raise FileNotFoundError(f"L2 WAV files not found under {l2_root}; pass --allow-missing-l2 to write CMU-only output")
    else:
        canonical = {}
        for speaker_prompts in prompts_by_speaker.values(): canonical.update(speaker_prompts)
        l2_records = build_l2_records(l2_root, project_root, canonical); write_jsonl(output_dir / "l2_manifest.jsonl", l2_records)
        cmu_by_prompt = {speaker: {r["prompt_id"]: r for r in cmu_records if r["speaker_id"] == speaker} for speaker in speakers}; paired = []
        for l2_record in l2_records:
            for speaker in speakers:
                cmu_record = cmu_by_prompt[speaker].get(l2_record["prompt_id"])
                if cmu_record is None: continue
                paired.append({"utt_id": f"pair_{l2_record['speaker_id']}_{speaker}_{l2_record['prompt_id']}", "prompt_id": l2_record["prompt_id"], "l2_speaker_id": l2_record["speaker_id"], "l2_audio_path": l2_record["audio_path"], "cmu_speaker_id": speaker, "cmu_audio_path": cmu_record["audio_path"], "transcript": cmu_record["transcript"], "l2_transcript": l2_record["transcript"]})
        write_jsonl(output_dir / "paired_manifest.jsonl", paired)
        suitcase_records = build_suitcase_records(l2_root, project_root)
        if suitcase_records: write_jsonl(output_dir / "suitcase_manifest.jsonl", suitcase_records)
        summary.update({"status": "paired", "l2_audio_count": len(l2_records), "paired_count": len(paired), "l2_speaker_counts": dict(Counter(r["speaker_id"] for r in l2_records)), "suitcase_count": len(suitcase_records)})
    output_dir.mkdir(parents=True, exist_ok=True); (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(summary, indent=2, sort_keys=True)); return 0

if __name__ == "__main__": sys.exit(main())
