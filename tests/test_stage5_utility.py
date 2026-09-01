from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from utility.forced_align import forced_align_ctc


def test_forced_alignment_is_conditioned_on_ground_truth() -> None:
    # Greedy decoding emits only blank, but the best path for target [1, 2]
    # must visit both target labels.
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [5.0, 4.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 4.0],
        ]
    )
    alignment, scores = forced_align_ctc(logits, torch.tensor([1, 2]), blank_id=0)
    assert tuple(alignment.shape) == (4,)
    assert tuple(scores.shape) == (4,)
    assert alignment.tolist().count(1) >= 1
    assert alignment.tolist().count(2) >= 1
    assert alignment.tolist() != logits.argmax(dim=1).tolist()


def test_forced_alignment_handles_repeated_target_labels() -> None:
    logits = torch.tensor(
        [
            [0.0, 5.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
        ]
    )
    alignment, _ = forced_align_ctc(logits, [1, 1], blank_id=0)
    assert alignment.tolist() == [1, 0, 1]


def _load_compute_module():
    path = Path(__file__).parents[1] / "utility" / "compute_utility.py"
    spec = importlib.util.spec_from_file_location("stage5_compute_utility", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Stage 5 utility module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_utility_variants_preserve_signed_effect_size_and_uncertainty() -> None:
    module = _load_compute_module()
    q = torch.tensor([[2.0, -1.0], [0.5, 3.0], [-4.0, 0.0]])
    target_probability = torch.tensor([0.5, 0.25, 1.0])
    sign_sum, contribution_sum, uncertainty_sum = module.frame_utility_sums(
        q, target_probability
    )
    assert torch.equal(sign_sum, torch.tensor([1.0, 0.0]))
    assert torch.allclose(contribution_sum, torch.tensor([-1.5, 2.0]))
    assert torch.allclose(uncertainty_sum, torch.tensor([1.375, 1.75]))


def test_attribution_uses_full_logits_for_strongest_competitor() -> None:
    module = _load_compute_module()
    logits = torch.tensor([[4.0, 5.0, 4.5]])
    alignment = torch.tensor([1])
    delta = torch.tensor([[2.0, -1.0]])
    w_delta = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 5.0]])
    targets, competitors, delta_valid, q = module.attribute_nonblank_frames(
        logits, alignment, delta, w_delta, blank=0
    )
    assert targets.tolist() == [1]
    assert competitors.tolist() == [2]
    assert delta_valid.tolist() == [[2.0, -1.0]]
    assert q.tolist() == [[-4.0, 3.0]]
