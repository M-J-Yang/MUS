from __future__ import annotations

import torch

from usde.reparameterization import (
    concentration_summary,
    haar_orthogonal,
    reparameterize_linear_head,
    rotate_features,
    top_k_jaccard,
)


def test_haar_orthogonal_is_deterministic_and_orthogonal() -> None:
    first = haar_orthogonal(8, 1337)
    second = haar_orthogonal(8, 1337)

    assert torch.equal(first, second)
    assert torch.allclose(first.transpose(0, 1) @ first, torch.eye(8), atol=2e-5, rtol=2e-5)


def test_linear_head_and_features_preserve_logits() -> None:
    head = torch.nn.Linear(5, 4)
    features = torch.randn(2, 3, 5)
    q = haar_orthogonal(5, 2027)
    rotated_head = reparameterize_linear_head(head, q)

    native_logits = head(features)
    rotated_logits = rotated_head(rotate_features(features, q))

    assert torch.allclose(native_logits, rotated_logits, atol=2e-4, rtol=2e-4)


def test_concentration_and_mask_overlap_are_well_defined() -> None:
    scores = torch.tensor([4.0, 2.0, 1.0, 0.0])
    summary = concentration_summary(scores, (0.25, 0.5, 1.0))

    assert summary["dimension"] == 4
    assert summary["top_k_mass"]["0.25"] == 4.0 / 7.0
    assert summary["top_k_mass"]["1"] == 1.0
    assert top_k_jaccard(scores, scores, 2) == 1.0
