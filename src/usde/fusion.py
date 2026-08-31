"""Frozen-feature concat + CTC baseline used by Step 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from usde.manifest import load_jsonl
from usde.text import encode


class FrozenFusionDataset(Dataset[dict[str, Any]]):
    def __init__(self, manifest_path: Path, feature_dir: Path, vocab: dict[str, int]) -> None:
        self.records = load_jsonl(manifest_path)
        self.feature_dir = feature_dir
        self.vocab = vocab
        missing = [row["utt_id"] for row in self.records if not (feature_dir / f"{row['utt_id']}.pt").is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} missing feature files, e.g. {missing[:3]}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        item = torch.load(self.feature_dir / f"{row['utt_id']}.pt", map_location="cpu")
        reference, delta = item["wavlm_ft"], item["delta"]
        if reference.ndim != 2 or delta.ndim != 2 or reference.shape != delta.shape or reference.shape[1] != 1024:
            raise ValueError(f"{row['utt_id']}: expected matched [T,1024] tensors")
        # Feature stores may use float16 to keep the multi-layer audit tractable;
        # CTC heads always receive float32 inputs for stable optimization.
        features = torch.cat((reference, delta), dim=-1).to(dtype=torch.float32)
        return {"features": features, "targets": torch.tensor(encode(row["transcript"], self.vocab), dtype=torch.long), "transcript": row["transcript"]}


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    features = [item["features"] for item in batch]
    targets = [item["targets"] for item in batch]
    return {
        "features": pad_sequence(features, batch_first=True),
        "feature_lengths": torch.tensor([item.shape[0] for item in features], dtype=torch.long),
        "targets": torch.cat(targets),
        "target_lengths": torch.tensor([item.shape[0] for item in targets], dtype=torch.long),
        "transcripts": [item["transcript"] for item in batch],
    }


class ConcatLinearCTC(nn.Module):
    """Source-compatible concat head; LayerNorm/dropout can be disabled for pure linear CTC."""
    def __init__(self, vocab_size: int, source_compatible: bool = True) -> None:
        super().__init__()
        self.normalizer = nn.LayerNorm(2048) if source_compatible else nn.Identity()
        self.dropout = nn.Dropout(0.1) if source_compatible else nn.Identity()
        self.linear = nn.Linear(2048, vocab_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(self.normalizer(features)))


def load_vocab(path: Path) -> dict[str, int]:
    return {key: int(value) for key, value in json.loads(path.read_text(encoding="utf-8")).items()}
