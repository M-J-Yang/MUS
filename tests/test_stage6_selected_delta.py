from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from analysis.collect_stage6_results import collect
from usde.stage4 import CachedFeatureDataset
from usde.stage6 import (
    SelectedDeltaLinearCTC,
    get_selected_indices,
    select_delta_features,
)


def test_selected_indices_are_topk_and_random_is_nested() -> None:
    utility = torch.tensor([2, 0, 3, 1])
    magnitude = torch.tensor([1, 3, 0, 2])
    assert get_selected_indices("utility", 2, utility_ranking=utility).tolist() == [2, 0]
    assert get_selected_indices("magnitude", 2, magnitude_ranking=magnitude).tolist() == [1, 3]

    random_2 = get_selected_indices("random", 2, delta_dim=8, seed=42)
    random_5 = get_selected_indices("random", 5, delta_dim=8, seed=42)
    assert random_2.tolist() == random_5[:2].tolist()
    assert torch.unique(random_5).numel() == 5


def test_rankings_must_be_complete_permutations() -> None:
    with pytest.raises(ValueError, match="permutation"):
        get_selected_indices(
            "utility",
            2,
            utility_ranking=torch.tensor([0, 0, 1, 2]),
            delta_dim=4,
        )
    with pytest.raises(ValueError, match="contain 4"):
        get_selected_indices(
            "magnitude",
            2,
            magnitude_ranking=torch.tensor([0, 1, 2]),
            delta_dim=4,
        )


def test_selected_delta_slicing_changes_only_feature_dimension() -> None:
    features = torch.arange(2 * 3 * 7, dtype=torch.float32).reshape(2, 3, 7)
    indices = torch.tensor([3, 1])
    selected = select_delta_features(features, indices, reference_dim=3, delta_dim=4)
    expected = torch.cat((features[..., :3], features[..., 3 + indices]), dim=-1)
    assert tuple(selected.shape) == (2, 3, 5)
    assert torch.equal(selected, expected)

    model = SelectedDeltaLinearCTC(3, 4, indices, vocab_size=6)
    assert model.classifier.in_features == 5
    assert tuple(model(features).shape) == (2, 3, 6)
    assert "selected_indices" in model.state_dict()


def test_selected_delta_accepts_distinct_reference_and_delta_dims() -> None:
    features = torch.randn(1, 4, 3 + 5)
    selected = select_delta_features(
        features,
        torch.tensor([4, 0]),
        reference_dim=3,
        delta_dim=5,
    )
    assert tuple(selected.shape) == (1, 4, 5)


def test_cached_dataset_can_infer_a_distinct_delta_dimension(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps({"utt_id": "u0", "transcript": "a", "audio_path": "unused.wav", "speaker_id": "s0"}) + "\n",
        encoding="utf-8",
    )
    delta_dir = tmp_path / "train" / "delta"
    reference_dir = tmp_path / "train" / "wavlm_ft"
    delta_dir.mkdir(parents=True)
    reference_dir.mkdir(parents=True)
    torch.save(torch.randn(2, 3), reference_dir / "u0.pt")
    torch.save(torch.randn(2, 5), delta_dir / "u0.pt")
    vocab = {"<pad>": 0, "<unk>": 1, "|": 2, "a": 3}
    dataset = CachedFeatureDataset(
        manifest,
        tmp_path,
        "full_delta",
        vocab,
        expected_dim=3,
        allow_auxiliary_dim_mismatch=True,
    )
    assert tuple(dataset[0]["features"].shape) == (2, 8)


def test_stage6_collector_builds_the_eight_row_report(tmp_path: Path) -> None:
    stage4_root = tmp_path / "stage4"
    stage6_root = tmp_path / "stage6"
    for condition, input_dim, wer in (("ref", 3, 0.3), ("full_delta", 8, 0.2)):
        run_dir = stage4_root / condition
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "input_dim": input_dim,
                    "base_dim": 3,
                    "best_epoch": 1,
                    "best_dev_wer": wer,
                    "test_wer": wer,
                }
            ),
            encoding="utf-8",
        )
    for selection in ("random", "magnitude", "utility"):
        for k in (256, 512):
            run_dir = stage6_root / selection / f"k{k}"
            run_dir.mkdir(parents=True)
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "selection": selection,
                        "k": k,
                        "best_epoch": 1,
                        "best_dev_wer": 0.1,
                        "test_wer": 0.15,
                    }
                ),
                encoding="utf-8",
            )
    report = collect(stage4_root, stage6_root)
    assert len(report["rows"]) == 8
    assert report["rows"][1]["delta_dims"] == 5
    assert report["rows"][-1]["selection"] == "utility"
    assert report["rows"][-1]["k"] == 512
