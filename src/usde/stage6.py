"""Selected-Delta CTC components for the Stage 6 intervention experiments.

Stage 6 keeps the cached [E_ref; Delta] representation unchanged and
selects Delta coordinates immediately before a newly initialized linear CTC
head. The selection is therefore a feature-dimension intervention, not a
zero-mask applied to a FullDelta checkpoint.
"""

from __future__ import annotations

from typing import Sequence

import torch

from usde.stage4 import LinearCTC


SELECTIONS = ("utility", "magnitude", "random")


def _validate_ranking(ranking: torch.Tensor, delta_dim: int, name: str) -> torch.Tensor:
    if not isinstance(ranking, torch.Tensor):
        raise TypeError(f"{name} ranking must be a torch.Tensor")
    if ranking.ndim != 1:
        raise ValueError(f"{name} ranking must be one-dimensional, got {tuple(ranking.shape)}")
    if ranking.numel() != delta_dim:
        raise ValueError(f"{name} ranking must contain {delta_dim} coordinates, got {ranking.numel()}")
    if ranking.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError(f"{name} ranking must contain integer indices, got {ranking.dtype}")
    ranking = ranking.to(dtype=torch.long, device="cpu")
    if (
        torch.unique(ranking).numel() != delta_dim
        or ranking.min().item() < 0
        or ranking.max().item() >= delta_dim
    ):
        raise ValueError(f"{name} ranking must be a permutation of [0, {delta_dim})")
    return ranking


def get_selected_indices(
    method: str,
    k: int,
    utility_ranking: torch.Tensor | None = None,
    magnitude_ranking: torch.Tensor | None = None,
    delta_dim: int | None = None,
    seed: int = 42,
) -> torch.Tensor:
    """Return one deterministic Top-K coordinate index tensor.

    utility_ranking and magnitude_ranking are frozen Stage 5 permutations.
    Random selection uses one seeded permutation, so calls for different K
    values with the same delta_dim and seed are nested.
    """

    if method not in SELECTIONS:
        raise ValueError(f"unknown selection {method!r}; choose from {SELECTIONS}")
    if k < 1:
        raise ValueError("k must be positive")
    if delta_dim is None:
        candidates = (
            utility_ranking
            if method == "utility"
            else magnitude_ranking
            if method == "magnitude"
            else None
        )
        if candidates is None:
            raise ValueError("delta_dim is required for random selection")
        if candidates.ndim != 1:
            raise ValueError("ranking must be one-dimensional")
        delta_dim = int(candidates.numel())
    if delta_dim < 1 or k > delta_dim:
        raise ValueError(f"k must be in [1, {delta_dim}], got {k}")

    if method == "utility":
        if utility_ranking is None:
            raise ValueError("utility_ranking is required for utility selection")
        ranking = _validate_ranking(utility_ranking, delta_dim, "utility")
    elif method == "magnitude":
        if magnitude_ranking is None:
            raise ValueError("magnitude_ranking is required for magnitude selection")
        ranking = _validate_ranking(magnitude_ranking, delta_dim, "magnitude")
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        ranking = torch.randperm(delta_dim, generator=generator)
    return ranking[:k].clone().long()


def select_delta_features(
    features: torch.Tensor,
    selected_indices: torch.Tensor,
    reference_dim: int,
    delta_dim: int,
) -> torch.Tensor:
    """Concatenate the reference stream with selected Delta coordinates."""

    if features.ndim != 3 or features.shape[-1] != reference_dim + delta_dim:
        raise ValueError(
            f"expected [B,T,{reference_dim + delta_dim}] cached features, got {tuple(features.shape)}"
        )
    if selected_indices.ndim != 1 or selected_indices.numel() < 1:
        raise ValueError("selected_indices must be a non-empty one-dimensional tensor")
    indices = selected_indices.to(device=features.device, dtype=torch.long)
    if indices.min().item() < 0 or indices.max().item() >= delta_dim:
        raise ValueError(f"selected indices must be in [0, {delta_dim})")
    reference = features[..., :reference_dim]
    delta = features[..., reference_dim:]
    delta_selected = delta.index_select(dim=-1, index=indices)
    return torch.cat((reference, delta_selected), dim=-1)


class SelectedDeltaLinearCTC(LinearCTC):
    """Fresh linear CTC head that slices Delta inside forward."""

    def __init__(
        self,
        reference_dim: int,
        delta_dim: int,
        selected_indices: Sequence[int] | torch.Tensor,
        vocab_size: int,
    ) -> None:
        if reference_dim < 1 or delta_dim < 1:
            raise ValueError("reference_dim and delta_dim must be positive")
        indices = torch.as_tensor(selected_indices, dtype=torch.long, device="cpu")
        if indices.ndim != 1 or indices.numel() < 1:
            raise ValueError("selected_indices must be a non-empty one-dimensional sequence")
        if torch.unique(indices).numel() != indices.numel():
            raise ValueError("selected_indices must not contain duplicates")
        if indices.min().item() < 0 or indices.max().item() >= delta_dim:
            raise ValueError(f"selected indices must be in [0, {delta_dim})")
        super().__init__(reference_dim + int(indices.numel()), vocab_size)
        self.reference_dim = int(reference_dim)
        self.delta_dim = int(delta_dim)
        self.register_buffer("selected_indices", indices, persistent=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        selected = select_delta_features(
            features, self.selected_indices, self.reference_dim, self.delta_dim
        )
        return self.linear(selected)


__all__ = [
    "SELECTIONS",
    "SelectedDeltaLinearCTC",
    "get_selected_indices",
    "select_delta_features",
]
