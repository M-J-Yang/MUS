"""Equivalent orthogonal reparameterization helpers.

The final hidden representation of the frozen CTC head can be changed by an
orthogonal basis transform without changing the represented ASR function.  A
row-wise feature tensor ``E`` is transformed as ``E @ Q.T`` and the linear
head weight as ``W @ Q.T``.  This module keeps that convention in one place so
the coordinate-dependence control cannot accidentally rotate only one side of
the interface.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def haar_orthogonal(
    dimension: int,
    seed: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Construct a deterministic Haar-style orthogonal matrix.

    QR factorization of a Gaussian matrix gives a Haar-distributed orthogonal
    basis after fixing the signs of the diagonal of ``R``.  The matrix is
    generated on CPU so the random stream and result do not depend on the
    selected accelerator; callers can move it to the model device afterwards.
    """

    if dimension < 1:
        raise ValueError("dimension must be positive")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }:
        raise TypeError("dtype must be a floating-point dtype")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Use float64 for the QR factorization, then cast once at the interface.
    # This makes the generated basis reproducible and numerically cleaner than
    # performing the factorization directly in half precision.
    gaussian = torch.randn(
        (dimension, dimension), generator=generator, dtype=torch.float64
    )
    q, r = torch.linalg.qr(gaussian, mode="reduced")
    diagonal = torch.diagonal(r)
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    q = q * signs.unsqueeze(0)
    q = q.to(dtype=dtype)
    if device is not None:
        q = q.to(device=device)
    return q


def rotate_features(features: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotate the final feature dimension of a tensor by ``Q``.

    Features use the usual row-wise PyTorch layout ``[..., D]``.  This is the
    row-layout equivalent of the column-vector transform ``e' = Q e``.
    """

    if features.ndim < 1:
        raise ValueError("features must have at least one dimension")
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("q must be a square matrix")
    if features.shape[-1] != q.shape[0]:
        raise ValueError(
            f"feature dimension {features.shape[-1]} does not match q {q.shape[0]}"
        )
    aligned_q = q.to(device=features.device, dtype=features.dtype)
    return features.matmul(aligned_q.transpose(0, 1))


def reparameterize_linear_head(head: nn.Module, q: torch.Tensor) -> nn.Linear:
    """Return the exactly equivalent linear head ``W' = W Q.T``.

    The current formal protocol uses a pure ``nn.Linear`` CTC head.  Refusing
    composite heads here is intentional: a LayerNorm or another nonlinear
    operation would require a different, explicitly justified invariance
    argument.
    """

    if not isinstance(head, nn.Linear):
        raise TypeError(
            "equivalent orthogonal reparameterization requires an nn.Linear head"
        )
    if q.ndim != 2 or q.shape != (head.in_features, head.in_features):
        raise ValueError(
            f"q must have shape {(head.in_features, head.in_features)}, got {tuple(q.shape)}"
        )

    aligned_q = q.to(device=head.weight.device, dtype=head.weight.dtype)
    transformed_weight = head.weight.detach().matmul(aligned_q.transpose(0, 1))
    transformed_bias = head.bias.detach().clone() if head.bias is not None else None
    transformed = nn.Linear(
        head.in_features,
        head.out_features,
        bias=head.bias is not None,
        device=head.weight.device,
        dtype=head.weight.dtype,
    )
    with torch.no_grad():
        transformed.weight.copy_(transformed_weight)
        if transformed.bias is not None and transformed_bias is not None:
            transformed.bias.copy_(transformed_bias)
    transformed.train(head.training)
    for parameter, source in zip(transformed.parameters(), head.parameters(), strict=True):
        parameter.requires_grad_(source.requires_grad)
    return transformed


def top_k_mass(scores: torch.Tensor, k: int) -> float:
    """Return the fraction of nonnegative score mass in the top ``k`` entries."""

    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if scores.numel() < 1:
        raise ValueError("scores must not be empty")
    if k < 1:
        raise ValueError("k must be positive")
    values = scores.detach().to(dtype=torch.float64, device="cpu").clamp_min(0)
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0:
        return 0.0
    count = min(k, values.numel())
    return float(values.topk(count, sorted=False).values.sum() / total)


def normalized_entropy(scores: torch.Tensor) -> float:
    """Return entropy normalized to ``[0, 1]`` for a nonnegative score vector."""

    if scores.ndim != 1 or scores.numel() < 1:
        raise ValueError("scores must be a non-empty one-dimensional tensor")
    values = scores.detach().to(dtype=torch.float64, device="cpu").clamp_min(0)
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0:
        return 0.0
    probabilities = values / total
    entropy = float(-(probabilities * probabilities.clamp_min(torch.finfo(torch.float64).tiny).log()).sum())
    return entropy / math.log(values.numel()) if values.numel() > 1 else 0.0


def gini_coefficient(scores: torch.Tensor) -> float:
    """Return the Gini concentration coefficient of a nonnegative vector."""

    if scores.ndim != 1 or scores.numel() < 1:
        raise ValueError("scores must be a non-empty one-dimensional tensor")
    values = scores.detach().to(dtype=torch.float64, device="cpu").clamp_min(0)
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0:
        return 0.0
    ordered = values.sort().values
    n = ordered.numel()
    positions = torch.arange(1, n + 1, dtype=torch.float64)
    return float(((2 * positions - n - 1) * ordered).sum() / (n * total))


def concentration_summary(
    scores: torch.Tensor,
    top_k_fractions: tuple[float, ...],
) -> dict[str, object]:
    """Summarize coordinate concentration for one score vector."""

    if not top_k_fractions:
        raise ValueError("top_k_fractions must not be empty")
    dimension = int(scores.numel())
    if dimension < 1:
        raise ValueError("scores must not be empty")
    if any(fraction <= 0 or fraction > 1 for fraction in top_k_fractions):
        raise ValueError("top_k_fractions must lie in (0, 1]")
    entropy = normalized_entropy(scores)
    effective_support = math.exp(entropy * math.log(dimension))
    top_mass = {
        f"{fraction:g}": top_k_mass(
            scores, max(1, round(dimension * fraction))
        )
        for fraction in top_k_fractions
    }
    return {
        "dimension": dimension,
        "l1": float(scores.detach().to(dtype=torch.float64).abs().sum()),
        "l2": float(scores.detach().to(dtype=torch.float64).pow(2).sum().sqrt()),
        "gini": gini_coefficient(scores),
        "normalized_entropy": entropy,
        "effective_support": effective_support,
        "effective_support_fraction": effective_support / dimension,
        "top_k_mass": top_mass,
    }


def top_k_jaccard(scores_a: torch.Tensor, scores_b: torch.Tensor, k: int) -> float:
    """Return the Jaccard overlap of two coordinate Top-K masks."""

    if scores_a.ndim != 1 or scores_b.ndim != 1 or scores_a.shape != scores_b.shape:
        raise ValueError("scores must be one-dimensional tensors of equal shape")
    if k < 1:
        raise ValueError("k must be positive")
    count = min(k, scores_a.numel())
    indices_a = set(torch.argsort(scores_a, descending=True, stable=True)[:count].tolist())
    indices_b = set(torch.argsort(scores_b, descending=True, stable=True)[:count].tolist())
    union = indices_a | indices_b
    return len(indices_a & indices_b) / max(len(union), 1)


__all__ = [
    "concentration_summary",
    "gini_coefficient",
    "haar_orthogonal",
    "normalized_entropy",
    "reparameterize_linear_head",
    "rotate_features",
    "top_k_jaccard",
    "top_k_mass",
]
