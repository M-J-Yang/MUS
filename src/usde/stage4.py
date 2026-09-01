"""Datasets, models, and decoding helpers for the Stage 4 baselines.

Stage 4 deliberately keeps the trainable part to one linear CTC classifier.
The only difference between the three conditions is which frozen feature
streams are concatenated before that classifier.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from usde.manifest import load_jsonl
from usde.metrics import word_error_rate
from usde.text import normalize_text


CONDITIONS = ("ref", "full_embedding", "full_delta")
BASE_DIM = 1024


def load_vocab(path: Path) -> dict[str, int]:
    """Load the frozen Stage 2 character vocabulary."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: vocabulary must be a non-empty JSON object")
    vocab = {str(token): int(index) for token, index in raw.items()}
    if len(set(vocab.values())) != len(vocab):
        raise ValueError(f"{path}: vocabulary ids must be unique")
    blank_tokens = [token for token in ("<pad>", "<blank>") if token in vocab]
    if not blank_tokens:
        raise ValueError(f"{path}: expected <pad> or <blank> as the CTC blank token")
    if vocab[blank_tokens[0]] != 0:
        raise ValueError(f"{path}: the CTC blank token must have id 0")
    if "<unk>" not in vocab:
        raise ValueError(f"{path}: vocabulary is missing <unk>")
    if "|" not in vocab and " " not in vocab:
        raise ValueError(f"{path}: expected a word delimiter token ('|' or ' ')")
    return vocab


def blank_id(vocab: dict[str, int]) -> int:
    """Return the blank id for either the Stage 2 or legacy vocabulary name."""

    for token in ("<pad>", "<blank>"):
        if token in vocab:
            return int(vocab[token])
    raise ValueError("vocabulary has no <pad>/<blank> token")


def encode_text(text: str, vocab: dict[str, int]) -> list[int]:
    """Encode text with the same lower-case character contract as Stage 2.

    The current Stage 2 vocabulary uses ``|`` for spaces; an older project
    vocabulary used a literal space.  Supporting both names here makes the
    Stage 4 trainer compatible with either frozen vocabulary without changing
    the actual token ids.
    """

    normalized = normalize_text(text)
    delimiter = "|" if "|" in vocab else " "
    unknown = int(vocab["<unk>"])
    return [int(vocab.get(delimiter if character == " " else character, unknown)) for character in normalized]


def decode_ctc(ids: Iterable[int], vocab: dict[str, int]) -> str:
    """Greedy CTC collapse followed by the frozen text normalization."""

    inverse = {int(index): token for token, index in vocab.items()}
    blank = blank_id(vocab)
    output: list[str] = []
    previous: int | None = None
    for raw_id in ids:
        index = int(raw_id)
        if index != blank and index != previous:
            token = inverse.get(index, "<unk>")
            if token not in {"<unk>", "<pad>", "<blank>"}:
                output.append(" " if token == "|" else token)
        previous = index
    return normalize_text("".join(output))


def _load_tensor(path: Path, label: str) -> torch.Tensor:
    try:
        item = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # compatibility with older PyTorch releases
        item = torch.load(path, map_location="cpu")
    if isinstance(item, dict):
        for key in (label, "tensor", "features"):
            if isinstance(item.get(key), torch.Tensor):
                item = item[key]
                break
    if not isinstance(item, torch.Tensor):
        raise ValueError(f"{path}: expected a CPU tensor cache entry")
    return item


class CachedFeatureDataset(Dataset[dict[str, Any]]):
    """Load one frozen feature condition from the Stage 3 directory tree."""

    def __init__(
        self,
        manifest_path: Path,
        feature_root: Path,
        condition: str,
        vocab: dict[str, int],
        feature_split: str | None = None,
        expected_dim: int = BASE_DIM,
        max_examples: int | None = None,
        sample_seed: int = 1337,
        expected_auxiliary_dim: int | None = None,
        allow_auxiliary_dim_mismatch: bool = False,
    ) -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition!r}; choose from {CONDITIONS}")
        if expected_dim < 1:
            raise ValueError("expected_dim must be positive")
        if expected_auxiliary_dim is not None and expected_auxiliary_dim < 1:
            raise ValueError("expected_auxiliary_dim must be positive when provided")
        if not isinstance(allow_auxiliary_dim_mismatch, bool):
            raise TypeError("allow_auxiliary_dim_mismatch must be a bool")
        self.manifest_path = manifest_path
        self.feature_root = feature_root
        self.condition = condition
        self.vocab = vocab
        self.expected_dim = expected_dim
        self.expected_auxiliary_dim = expected_auxiliary_dim
        self.allow_auxiliary_dim_mismatch = allow_auxiliary_dim_mismatch
        self.records = load_jsonl(manifest_path)
        if max_examples is not None:
            if max_examples < 1:
                raise ValueError("max_examples must be positive")
            if max_examples < len(self.records):
                self.records = random.Random(sample_seed).sample(self.records, max_examples)

        split = feature_split or ("train" if manifest_path.stem.startswith("train") else manifest_path.stem)
        self.feature_split = split
        split_root = feature_root / split
        self.reference_dir = split_root / "wavlm_ft"
        self.auxiliary_dir = {
            "ref": None,
            "full_embedding": split_root / "w2v2_ft",
            "full_delta": split_root / "delta",
        }[condition]
        required_dirs = [self.reference_dir]
        if self.auxiliary_dir is not None:
            required_dirs.append(self.auxiliary_dir)
        missing_dirs = [str(path) for path in required_dirs if not path.is_dir()]
        if missing_dirs:
            raise FileNotFoundError(f"{manifest_path}: missing feature directories {missing_dirs}")

        missing: list[str] = []
        for row in self.records:
            utt_id = str(row["utt_id"])
            if Path(utt_id).name != utt_id:
                raise ValueError(f"{utt_id}: utt_id must not contain path separators")
            if not (self.reference_dir / f"{utt_id}.pt").is_file():
                missing.append(f"{utt_id}:wavlm_ft")
            if self.auxiliary_dir is not None and not (self.auxiliary_dir / f"{utt_id}.pt").is_file():
                missing.append(f"{utt_id}:{condition}")
        if missing:
            raise FileNotFoundError(f"{manifest_path}: missing feature files, e.g. {missing[:3]}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        utt_id = str(row["utt_id"])
        reference = _load_tensor(self.reference_dir / f"{utt_id}.pt", "wavlm_ft")
        if reference.ndim != 2 or reference.shape[1] != self.expected_dim:
            raise ValueError(f"{utt_id}: expected reference [T,{self.expected_dim}], got {tuple(reference.shape)}")
        streams = [reference]
        if self.auxiliary_dir is not None:
            label = "w2v2_ft" if self.condition == "full_embedding" else "delta"
            auxiliary = _load_tensor(self.auxiliary_dir / f"{utt_id}.pt", label)
            expected_auxiliary_dim = (
                self.expected_auxiliary_dim
                if self.expected_auxiliary_dim is not None
                else self.expected_dim
            )
            auxiliary_shape_ok = (
                auxiliary.ndim == 2
                and auxiliary.shape[0] == reference.shape[0]
                and (
                    self.allow_auxiliary_dim_mismatch
                    or auxiliary.shape[1] == expected_auxiliary_dim
                )
            )
            if not auxiliary_shape_ok:
                raise ValueError(
                    f"{utt_id}: frame/dimension mismatch; reference={tuple(reference.shape)}, "
                    f"auxiliary={tuple(auxiliary.shape)}"
                )
            streams.append(auxiliary)
        if not all(torch.isfinite(stream).all() for stream in streams):
            raise ValueError(f"{utt_id}: feature cache contains non-finite values")
        features = torch.cat(streams, dim=-1).to(dtype=torch.float32).contiguous()
        targets = torch.tensor(encode_text(row["transcript"], self.vocab), dtype=torch.long)
        if targets.numel() == 0:
            raise ValueError(f"{utt_id}: transcript encodes to an empty target")
        if targets.numel() > features.shape[0]:
            raise ValueError(
                f"{utt_id}: CTC target ({targets.numel()}) is longer than feature sequence ({features.shape[0]})"
            )
        return {
            "features": features,
            "targets": targets,
            "feature_length": int(features.shape[0]),
            "target_length": int(targets.shape[0]),
            "transcript": normalize_text(row["transcript"]),
            "utt_id": utt_id,
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    features = [item["features"] for item in batch]
    targets = [item["targets"] for item in batch]
    return {
        "features": pad_sequence(features, batch_first=True),
        "feature_lengths": torch.tensor([item["feature_length"] for item in batch], dtype=torch.long),
        "targets": torch.cat(targets),
        "target_lengths": torch.tensor([item["target_length"] for item in batch], dtype=torch.long),
        "transcripts": [item["transcript"] for item in batch],
        "utt_ids": [item["utt_id"] for item in batch],
    }


class LinearCTC(nn.Module):
    """Pure linear frame classifier used by all Stage 4 conditions.

    ``linear`` is the registered module to keep the checkpoint compatible
    with the existing utility loader.  ``classifier`` is the public alias
    used by Stage 5 when slicing ``W_delta``.
    """

    def __init__(self, input_dim: int, vocab_size: int) -> None:
        super().__init__()
        if input_dim < 1 or vocab_size < 2:
            raise ValueError("input_dim must be positive and vocab_size must be at least two")
        self.linear = nn.Linear(input_dim, vocab_size)

    @property
    def classifier(self) -> nn.Linear:
        return self.linear

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.linear.in_features:
            raise ValueError(
                f"expected [B,T,{self.linear.in_features}] features, got {tuple(features.shape)}"
            )
        return self.linear(features)


def greedy_wer(
    model: LinearCTC,
    loader: Any,
    vocab: dict[str, int],
    device: torch.device,
) -> float:
    """Evaluate with one shared greedy CTC decoder."""

    model.eval()
    references: list[str] = []
    hypotheses: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["features"].to(device, non_blocking=True))
            predicted_ids = logits.argmax(dim=-1).cpu()
            for row, length in zip(predicted_ids, batch["feature_lengths"], strict=True):
                hypotheses.append(decode_ctc(row[: int(length)].tolist(), vocab))
            references.extend(batch["transcripts"])
    return float(word_error_rate(references, hypotheses))


__all__ = [
    "BASE_DIM",
    "CONDITIONS",
    "CachedFeatureDataset",
    "LinearCTC",
    "blank_id",
    "collate",
    "decode_ctc",
    "encode_text",
    "greedy_wer",
    "load_vocab",
]
