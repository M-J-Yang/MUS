from __future__ import annotations

import json
from pathlib import Path

import torch

from usde.stage4 import CachedFeatureDataset, LinearCTC, collate, decode_ctc, encode_text


def _write_manifest(path: Path) -> None:
    rows = []
    for index, text in enumerate(("ab cd", "cab")):
        rows.append(
            {
                "utt_id": f"utt{index}",
                "audio_path": "/not/used.wav",
                "transcript": text,
                "speaker_id": f"s{index}",
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_features(root: Path) -> None:
    for stream in ("wavlm_ft", "w2v2_ft", "delta"):
        directory = root / "train" / stream
        directory.mkdir(parents=True)
        for index in range(2):
            torch.save(torch.randn(5 + index, 4), directory / f"utt{index}.pt")


def test_stage4_conditions_and_collate(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest)
    _write_features(tmp_path)
    vocab = {"<pad>": 0, "<unk>": 1, "|": 2, "a": 3, "b": 4, "c": 5, "d": 6}

    for condition, expected_dim in (("ref", 4), ("full_embedding", 8), ("full_delta", 8)):
        dataset = CachedFeatureDataset(manifest, tmp_path, condition, vocab, expected_dim=4)
        batch = collate([dataset[0], dataset[1]])
        assert tuple(batch["features"].shape) == (2, 6, expected_dim)
        assert batch["feature_lengths"].tolist() == [5, 6]
        model = LinearCTC(expected_dim, len(vocab))
        assert model.classifier is model.linear
        assert tuple(model.classifier.weight.shape) == (len(vocab), expected_dim)


def test_stage4_text_contract() -> None:
    vocab = {"<pad>": 0, "<unk>": 1, "|": 2, "a": 3, "b": 4}
    assert encode_text("ab ba", vocab) == [3, 4, 2, 4, 3]
    assert decode_ctc([0, 3, 3, 0, 2, 4, 0, 3], vocab) == "a ba"
