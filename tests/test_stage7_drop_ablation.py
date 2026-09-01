from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "evaluate_drop_ablation.py"
    spec = importlib.util.spec_from_file_location("evaluate_drop_ablation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Drop ablation module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mask_delta_coordinates_preserves_reference_and_zeros_only_delta() -> None:
    module = _load_module()
    features = torch.arange(2 * 3 * 7, dtype=torch.float32).reshape(2, 3, 7)
    masked = module.mask_delta_coordinates(features, torch.tensor([2, 0]), reference_dim=3)

    assert torch.equal(masked[..., :3], features[..., :3])
    assert torch.equal(masked[..., 3 + 2], torch.zeros(2, 3))
    assert torch.equal(masked[..., 3 + 0], torch.zeros(2, 3))
    assert torch.equal(masked[..., 3 + 1], features[..., 3 + 1])


def test_drop_ranking_requires_a_complete_permutation() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="permutation"):
        module._validate_ranking(torch.tensor([0, 0, 1, 2]), delta_dim=4)
