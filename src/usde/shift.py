"""Single-model fine-tuning-shift cache and pruning primitives."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from usde.ctc import prepare_ctc_text, read_records
from usde.text import normalize_text


SHIFT_STREAMS = ("e0", "eft", "delta")


def _load_tensor(path: Path, label: str | None = None) -> torch.Tensor:
    """Load a tensor cache entry, accepting bare tensors and labeled payloads."""

    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        candidates = (label, "tensor", "features", "e0", "eft", "delta")
        value = next((value[key] for key in candidates if key and isinstance(value.get(key), torch.Tensor)), None)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path}: expected a tensor cache entry")
    return value


def validate_shift_tensors(
    e0: torch.Tensor,
    eft: torch.Tensor,
    delta: torch.Tensor,
    expected_dim: int | None = None,
) -> None:
    """Validate one utterance's E0/Eft/Delta shape and finite-value contract."""

    if any(value.ndim != 2 for value in (e0, eft, delta)):
        raise ValueError(
            f"shift tensors must be [T,D], got e0={tuple(e0.shape)}, "
            f"eft={tuple(eft.shape)}, delta={tuple(delta.shape)}"
        )
    if not (e0.shape == eft.shape == delta.shape):
        raise ValueError(
            f"shift tensors must have identical shapes, got e0={tuple(e0.shape)}, "
            f"eft={tuple(eft.shape)}, delta={tuple(delta.shape)}"
        )
    if e0.shape[0] < 1 or e0.shape[1] < 1:
        raise ValueError("shift tensors must contain at least one frame and one coordinate")
    if expected_dim is not None and e0.shape[1] != expected_dim:
        raise ValueError(f"expected hidden dimension {expected_dim}, got {e0.shape[1]}")
    if not all(torch.isfinite(value).all() for value in (e0, eft, delta)):
        raise ValueError("shift tensors contain non-finite values")


def reconstruction_error(
    e0: torch.Tensor,
    eft: torch.Tensor,
    delta: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Report whether the cached identity E0 + Delta = Eft holds."""

    validate_shift_tensors(e0, eft, delta)
    reconstructed = e0 + delta
    difference = reconstructed - eft
    return {
        "allclose": bool(torch.allclose(reconstructed, eft, atol=atol, rtol=rtol)),
        "atol": float(atol),
        "rtol": float(rtol),
        "max_abs_error": float(difference.abs().max()),
        "mean_abs_error": float(difference.abs().mean()),
        "num_values": int(difference.numel()),
    }


class ShiftFeatureDataset(Dataset[dict[str, Any]]):
    """Load cached E0/Eft/Delta streams for one fixed manifest split."""

    def __init__(
        self,
        manifest_path: Path,
        cache_root: Path,
        tokenizer: Any,
        feature_split: str | None = None,
        expected_dim: int | None = None,
        max_examples: int | None = None,
        sample_seed: int = 1337,
        load_eft: bool = True,
    ) -> None:
        self.manifest_path = manifest_path
        self.cache_root = cache_root
        self.tokenizer = tokenizer
        self.records = read_records(manifest_path)
        if max_examples is not None:
            if max_examples < 1:
                raise ValueError("max_examples must be positive")
            if max_examples < len(self.records):
                self.records = random.Random(sample_seed).sample(self.records, max_examples)
        self.feature_split = feature_split or manifest_path.stem
        self.expected_dim = expected_dim
        self.load_eft = load_eft
        self.stream_dirs = {
            stream: cache_root / self.feature_split / stream for stream in SHIFT_STREAMS
        }
        required_streams = ("e0", "eft", "delta") if load_eft else ("e0", "delta")
        missing_dirs = [str(self.stream_dirs[stream]) for stream in required_streams if not self.stream_dirs[stream].is_dir()]
        if missing_dirs:
            raise FileNotFoundError(f"{manifest_path}: missing shift cache directories {missing_dirs}")
        missing: list[str] = []
        for row in self.records:
            utt_id = str(row["utt_id"])
            if Path(utt_id).name != utt_id:
                raise ValueError(f"{utt_id}: utt_id must not contain path separators")
            for stream in required_streams:
                if not (self.stream_dirs[stream] / f"{utt_id}.pt").is_file():
                    missing.append(f"{utt_id}:{stream}")
        if missing:
            raise FileNotFoundError(f"{manifest_path}: missing shift cache files, e.g. {missing[:3]}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        utt_id = str(row["utt_id"])
        e0 = _load_tensor(self.stream_dirs["e0"] / f"{utt_id}.pt", "e0")
        delta = _load_tensor(self.stream_dirs["delta"] / f"{utt_id}.pt", "delta")
        eft = (
            _load_tensor(self.stream_dirs["eft"] / f"{utt_id}.pt", "eft")
            if self.load_eft
            else e0 + delta
        )
        validate_shift_tensors(e0, eft, delta, self.expected_dim)
        target_ids = self.tokenizer(
            prepare_ctc_text(row["transcript"], self.tokenizer)
        )["input_ids"]
        targets = torch.tensor(target_ids, dtype=torch.long)
        if targets.numel() == 0:
            raise ValueError(f"{utt_id}: transcript encodes to an empty target")
        if targets.numel() > e0.shape[0]:
            raise ValueError(
                f"{utt_id}: CTC target ({targets.numel()}) is longer than cache ({e0.shape[0]})"
            )
        return {
            "e0": e0.to(dtype=torch.float32).contiguous(),
            "eft": eft.to(dtype=torch.float32).contiguous(),
            "delta": delta.to(dtype=torch.float32).contiguous(),
            "feature_length": int(e0.shape[0]),
            "target": targets,
            "target_length": int(targets.shape[0]),
            "transcript": normalize_text(row["transcript"]),
            "utt_id": utt_id,
        }


def collate_shift(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad cached streams while preserving unpadded frame lengths."""

    if not batch:
        raise ValueError("cannot collate an empty batch")
    return {
        "e0": pad_sequence([item["e0"] for item in batch], batch_first=True),
        "eft": pad_sequence([item["eft"] for item in batch], batch_first=True),
        "delta": pad_sequence([item["delta"] for item in batch], batch_first=True),
        "feature_lengths": torch.tensor([item["feature_length"] for item in batch], dtype=torch.long),
        "targets": torch.cat([item["target"] for item in batch]),
        "target_lengths": torch.tensor([item["target_length"] for item in batch], dtype=torch.long),
        "transcripts": [item["transcript"] for item in batch],
        "utt_ids": [item["utt_id"] for item in batch],
    }


def select_delta(
    e0: torch.Tensor,
    delta: torch.Tensor,
    keep_indices: torch.Tensor | None,
) -> torch.Tensor:
    """Construct E0 + M Delta, retaining only the requested coordinates."""

    if e0.shape != delta.shape or e0.ndim != 3:
        raise ValueError("e0 and delta must have the same [B,T,D] shape")
    if keep_indices is None:
        return e0 + delta
    if keep_indices.ndim != 1:
        raise ValueError("keep_indices must be one-dimensional")
    indices = keep_indices.to(device=e0.device, dtype=torch.long)
    if indices.numel():
        if int(indices.min()) < 0 or int(indices.max()) >= delta.shape[-1]:
            raise ValueError(f"keep_indices must be in [0, {delta.shape[-1]})")
        if torch.unique(indices).numel() != indices.numel():
            raise ValueError("keep_indices must not contain duplicates")
    selected_delta = torch.zeros_like(delta)
    if indices.numel():
        selected_delta[..., indices] = delta[..., indices]
    return e0 + selected_delta


def ctc_taylor_batch_sums(
    linear_head: nn.Module,
    e0: torch.Tensor,
    delta: torch.Tensor,
    targets: torch.Tensor,
    feature_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Compute frame-averaged |Delta * dL_CTC/dDelta| coordinate sums."""

    _, taylor_sums, frame_count, mean_loss = ctc_gradient_and_taylor_batch_sums(
        linear_head,
        e0,
        delta,
        targets,
        feature_lengths,
        target_lengths,
        blank,
    )
    return taylor_sums, frame_count, mean_loss


def ctc_gradient_and_taylor_batch_sums(
    linear_head: nn.Module,
    e0: torch.Tensor,
    delta: torch.Tensor,
    targets: torch.Tensor,
    feature_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Compute frame-averaged Gradient-only and Taylor utility sums.

    The returned coordinate sums are, respectively,
    `sum(|dL/dDelta|)` and `sum(|Delta * dL/dDelta|)` over valid CTC
    frames. Keeping this in the same autograd pass makes the Gradient-only
    baseline exactly matched to the Taylor utility protocol.
    """

    if e0.shape != delta.shape or e0.ndim != 3:
        raise ValueError("e0 and delta must have the same [B,T,D] shape")
    if targets.ndim != 1 or feature_lengths.ndim != 1 or target_lengths.ndim != 1:
        raise ValueError("targets and length tensors must be one-dimensional")
    if feature_lengths.numel() != e0.shape[0] or target_lengths.numel() != e0.shape[0]:
        raise ValueError("length tensors must have one entry per batch element")
    if torch.any(feature_lengths < 1) or torch.any(feature_lengths > e0.shape[1]):
        raise ValueError("feature_lengths must be positive and within the padded batch")
    if torch.any(target_lengths < 1):
        raise ValueError("target_lengths must be positive")

    reference = e0.detach()
    delta_leaf = delta.detach().requires_grad_(True)
    logits = linear_head(reference + delta_leaf)
    if logits.ndim != 3:
        raise ValueError(f"linear_head must return [B,T,V] logits, got {tuple(logits.shape)}")
    log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
    losses = torch.nn.CTCLoss(blank=blank, reduction="none", zero_infinity=True)(
        log_probs, targets, feature_lengths, target_lengths
    )
    normalized_losses = losses / target_lengths.to(device=losses.device, dtype=losses.dtype)
    (gradient,) = torch.autograd.grad(normalized_losses.sum(), delta_leaf)
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("CTC Taylor gradient contains non-finite values")
    valid = torch.arange(e0.shape[1], device=e0.device).unsqueeze(0) < feature_lengths.to(e0.device).unsqueeze(1)
    valid_mask = valid.unsqueeze(-1)
    gradient_sums = gradient.abs().masked_fill(~valid_mask, 0.0).sum(dim=(0, 1)).detach()
    taylor_sums = (delta_leaf * gradient).abs().masked_fill(~valid_mask, 0.0).sum(dim=(0, 1)).detach()
    frame_count = int(feature_lengths.sum().detach().cpu())
    mean_loss = normalized_losses.mean().detach()
    if not torch.isfinite(gradient_sums).all() or not torch.isfinite(taylor_sums).all() or not torch.isfinite(mean_loss):
        raise FloatingPointError("CTC Taylor utility contains non-finite values")
    return gradient_sums, taylor_sums, frame_count, mean_loss


__all__ = [
    "SHIFT_STREAMS",
    "ShiftFeatureDataset",
    "collate_shift",
    "ctc_gradient_and_taylor_batch_sums",
    "ctc_taylor_batch_sums",
    "reconstruction_error",
    "select_delta",
    "validate_shift_tensors",
]
