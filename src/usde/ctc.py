"""Shared data and processor utilities for the Stage 2 CTC experiments."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
)

from usde.text import normalize_text


TARGET_SAMPLE_RATE = 16000


def read_records(path: Path) -> list[dict[str, str]]:
    """Read either the Stage 2 TSV files or the frozen JSONL manifests."""
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, str]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                audio = item.get("audio_path", item.get("wav_path"))
                transcript = item.get("transcript")
                if audio is None or transcript is None:
                    raise ValueError(f"{path}:{line_number}: requires audio_path/wav_path and transcript")
                records.append(
                    {
                        "utt_id": str(item.get("utt_id", line_number)),
                        "wav_path": str(audio),
                        "transcript": normalize_text(str(transcript)),
                    }
                )
    else:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError(f"{path}: missing TSV header")
            audio_column = "wav_path" if "wav_path" in reader.fieldnames else "audio_path"
            required = {audio_column, "transcript"}
            missing = required.difference(reader.fieldnames)
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            records = []
            for line_number, item in enumerate(reader, start=2):
                audio = item.get(audio_column, "")
                transcript = normalize_text(item.get("transcript", ""))
                if not audio or not transcript:
                    raise ValueError(f"{path}:{line_number}: empty audio path or transcript")
                records.append(
                    {
                        "utt_id": item.get("utt_id") or str(line_number),
                        "wav_path": audio,
                        "transcript": transcript,
                    }
                )
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def resolve_audio_path(raw_path: str, manifest_path: Path, audio_root: Path | None = None) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if audio_root is not None:
        return audio_root / path
    # Project manifests commonly store paths relative to the repository root.
    project_root = Path(__file__).resolve().parents[2]
    project_path = project_root / path
    return project_path if project_path.exists() else manifest_path.parent / path


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.ndim != 2 or waveform.shape[0] == 0:
        raise ValueError(f"{path}: expected non-empty [channels, time] waveform")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=TARGET_SAMPLE_RATE,
        )
        sample_rate = TARGET_SAMPLE_RATE
    return waveform.squeeze(0), sample_rate


def audio_length(path: Path) -> int:
    """Return the approximate post-resampling length without decoding audio."""
    info = torchaudio.info(str(path))
    return max(1, round(info.num_frames * TARGET_SAMPLE_RATE / info.sample_rate))


def make_processor(model_name: str, vocab_dir: Path | None = None) -> Wav2Vec2Processor:
    """Load an exact pretrained processor or construct the legacy vocabulary path.

    The formal 960h transfer run must inherit the source tokenizer and feature
    extractor. Existing Stage 2 experiments still pass ``vocab_dir`` and keep
    their project-local character vocabulary unchanged.
    """

    if vocab_dir is None:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        feature_extractor = processor.feature_extractor
        if int(getattr(feature_extractor, "sampling_rate", 0)) != TARGET_SAMPLE_RATE:
            raise ValueError(f"{model_name}: feature extractor is not configured for 16 kHz")
        if processor.tokenizer.pad_token_id is None:
            raise ValueError(f"{model_name}: pretrained tokenizer has no CTC blank/pad token")
        return processor

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_dir / "vocab.json"),
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        do_lower_case=False,
        bos_token=None,
        eos_token=None,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    if int(getattr(feature_extractor, "sampling_rate", 0)) != TARGET_SAMPLE_RATE:
        raise ValueError(f"{model_name}: feature extractor is not configured for 16 kHz")
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def prepare_ctc_text(text: str, tokenizer: Any) -> str:
    """Match transcript casing to the supplied tokenizer's character inventory.

    The project manifests are normalized to lowercase, while the public
    ``wav2vec2-large-960h`` tokenizer stores uppercase letters. This keeps the
    model's original vocabulary intact instead of silently mapping every
    lowercase letter to ``<unk>``. Lowercase project vocabularies remain
    unchanged.
    """

    normalized = normalize_text(text)
    vocabulary = {str(token) for token in tokenizer.get_vocab()}
    has_uppercase_letters = any(len(token) == 1 and token.isupper() for token in vocabulary)
    has_lowercase_letters = any(len(token) == 1 and token.islower() for token in vocabulary)
    if has_uppercase_letters and not has_lowercase_letters:
        return normalized.upper()
    if bool(getattr(tokenizer, "do_lower_case", False)):
        return normalized.lower()
    return normalized


class AudioCTCDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: Path,
        processor: Wav2Vec2Processor,
        audio_root: Path | None = None,
        max_examples: int | None = None,
        sample_seed: int = 42,
    ) -> None:
        self.manifest_path = manifest_path
        self.processor = processor
        self.records = read_records(manifest_path)
        if max_examples is not None:
            if max_examples < 1:
                raise ValueError("max_examples must be positive")
            if max_examples < len(self.records):
                generator = random.Random(sample_seed)
                self.records = generator.sample(self.records, max_examples)
        self.paths = [resolve_audio_path(row["wav_path"], manifest_path, audio_root) for row in self.records]
        self.input_lengths = [audio_length(path) for path in self.paths]
        missing = [str(path) for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{manifest_path}: missing audio, first path is {missing[0]}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        waveform, sample_rate = load_audio(self.paths[index])
        encoded_audio = self.processor(
            waveform.numpy(),
            sampling_rate=sample_rate,
            return_attention_mask=True,
        )
        encoded_text = self.processor.tokenizer(
            prepare_ctc_text(self.records[index]["transcript"], self.processor.tokenizer)
        )
        return {
            "input_values": encoded_audio["input_values"][0],
            "attention_mask": encoded_audio.get(
                "attention_mask", [1] * len(encoded_audio["input_values"][0])
            )[0],
            "labels": encoded_text["input_ids"],
        }


@dataclass
class CTCDataCollator:
    processor: Wav2Vec2Processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audio_features = [
            {key: value for key, value in item.items() if key in ("input_values", "attention_mask")}
            for item in features
        ]
        batch = self.processor.feature_extractor.pad(
            audio_features,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        label_features = [{"input_ids": item["labels"]} for item in features]
        labels_batch = self.processor.tokenizer.pad(label_features, padding=True, return_tensors="pt")
        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        return batch
