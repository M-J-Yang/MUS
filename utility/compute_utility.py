#!/usr/bin/env python3
"""Compute Stage 5 CTC decision-aligned Delta utility.

This is an offline attribution pass over the held-out ``train_utility``
split.  The FullDelta teacher is frozen; no model is trained here and no
frame-level ``q`` tensor is retained after each utterance.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from usde.stage4 import CachedFeatureDataset, LinearCTC, blank_id, load_vocab  # noqa: E402

try:
    from utility.forced_align import forced_align_ctc  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - exercised by direct execution
    from forced_align import forced_align_ctc  # noqa: E402


PROTOCOL = "stage5_ctc_decision_aligned_delta_utility_v1"


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # compatibility with older PyTorch releases
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a checkpoint dictionary")
    return payload


def load_full_delta_teacher(
    checkpoint_path: Path,
    vocab: dict[str, int],
    reference_dim: int,
    delta_dim: int,
    device: torch.device,
) -> tuple[LinearCTC, dict[str, Any]]:
    """Load and validate the pure-linear Stage 4 FullDelta teacher."""

    payload = _load_checkpoint(checkpoint_path)
    if payload.get("condition") not in (None, "full_delta"):
        raise ValueError(f"{checkpoint_path}: expected a full_delta teacher checkpoint")
    if payload.get("source_compatible", False):
        raise ValueError(
            f"{checkpoint_path}: source-compatible head is not valid for coordinate attribution; "
            "the teacher must be pure linear"
        )
    checkpoint_vocab = payload.get("vocab")
    if checkpoint_vocab is not None:
        checkpoint_vocab = {str(token): int(index) for token, index in checkpoint_vocab.items()}
        if checkpoint_vocab != vocab:
            raise ValueError(
                f"{checkpoint_path}: checkpoint vocabulary differs from --vocab; "
                "Stage 5 must reuse the Stage 4 encoder exactly"
            )

    input_dim = reference_dim + delta_dim
    if payload.get("input_dim") is not None and int(payload["input_dim"]) != input_dim:
        raise ValueError(f"{checkpoint_path}: input_dim does not match {input_dim}")
    if payload.get("base_dim") is not None and int(payload["base_dim"]) != reference_dim:
        raise ValueError(f"{checkpoint_path}: base_dim does not match {reference_dim}")
    if payload.get("blank_id") is not None and int(payload["blank_id"]) != blank_id(vocab):
        raise ValueError(f"{checkpoint_path}: checkpoint blank id differs from --vocab")

    state = payload.get("model")
    if not isinstance(state, dict):
        state = payload
    weight = state.get("linear.weight", state.get("classifier_weight"))
    bias = state.get("linear.bias", state.get("classifier_bias"))
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise ValueError(f"{checkpoint_path}: expected linear.weight and linear.bias")
    expected_shape = (len(vocab), input_dim)
    if tuple(weight.shape) != expected_shape or tuple(bias.shape) != (len(vocab),):
        raise ValueError(
            f"{checkpoint_path}: expected classifier shapes {expected_shape} and {(len(vocab),)}, "
            f"got {tuple(weight.shape)} and {tuple(bias.shape)}"
        )
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError(f"{checkpoint_path}: classifier contains non-finite values")

    model = LinearCTC(input_dim, len(vocab))
    model.load_state_dict({"linear.weight": weight, "linear.bias": bias})
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def attribute_nonblank_frames(
    logits: torch.Tensor,
    alignment: torch.Tensor,
    delta: torch.Tensor,
    w_delta: torch.Tensor,
    blank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return aligned targets, competitors, Delta frames, and coordinate q."""

    if logits.ndim != 2 or delta.ndim != 2 or alignment.ndim != 1:
        raise ValueError("logits, alignment, and delta must have shapes [T,V], [T], and [T]")
    if logits.shape[0] != alignment.shape[0] or delta.shape[0] != alignment.shape[0]:
        raise ValueError("logits, alignment, and delta must have the same frame length")
    if delta.shape[1] != w_delta.shape[1] or w_delta.shape[0] != logits.shape[1]:
        raise ValueError("Delta and classifier weight dimensions do not match")

    valid = alignment != int(blank)
    targets = alignment[valid]
    logits_valid = logits[valid]
    delta_valid = delta[valid]
    if targets.numel():
        if int(targets.min()) < 0 or int(targets.max()) >= logits.shape[1]:
            raise ValueError("alignment contains an id outside the classifier vocabulary")
        competitors_logits = logits_valid.clone()
        competitors_logits.scatter_(1, targets.unsqueeze(1), float("-inf"))
        competitors = competitors_logits.argmax(dim=1)
        q = delta_valid * (w_delta[targets] - w_delta[competitors])
    else:
        competitors = torch.empty(0, dtype=torch.long, device=alignment.device)
        q = delta_valid.new_empty((0, delta.shape[1]))
    return targets, competitors, delta_valid, q


def frame_utility_sums(
    q: torch.Tensor,
    target_probability: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate the three decision-aware utility signals for one utterance.

    ``q[t, i]`` is the Delta coordinate's contribution to the aligned-target
    versus strongest-competitor logit margin. The returned tensors are sums
    over frames, leaving the caller to divide by the global frame count.

    The variants are:

    - v1: ``sign(q)`` (the original binary utility);
    - v2: ``q`` (magnitude-weighted signed contribution);
    - v3: ``q * (1 - p(target))`` (uncertainty-weighted contribution).
    """

    if q.ndim != 2 or target_probability.ndim != 1:
        raise ValueError("q must have shape [frames, dimensions] and target_probability [frames]")
    if q.shape[0] != target_probability.shape[0]:
        raise ValueError("q and target_probability must have the same frame count")
    if not torch.isfinite(q).all() or not torch.isfinite(target_probability).all():
        raise ValueError("q and target_probability must contain finite values")
    if target_probability.numel() and (
        target_probability.min() < 0 or target_probability.max() > 1
    ):
        raise ValueError("target_probability must lie in [0, 1]")

    sign_sum = torch.sign(q).sum(dim=0)
    contribution_sum = q.sum(dim=0)
    uncertainty_sum = (
        q * (1.0 - target_probability).unsqueeze(1)
    ).sum(dim=0)
    return sign_sum, contribution_sum, uncertainty_sum


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, matching the tie behavior used by Spearman correlation."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    index = 0
    while index < values.size:
        end = index + 1
        while end < values.size and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1) + 1.0
        index = end
    return ranks


def spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Compute Spearman's rho without adding a SciPy runtime dependency."""

    if values_a.shape != values_b.shape or values_a.ndim != 1:
        raise ValueError("Spearman inputs must be one-dimensional arrays of equal shape")
    ranks_a = _rankdata(values_a)
    ranks_b = _rankdata(values_b)
    centered_a = ranks_a - ranks_a.mean()
    centered_b = ranks_b - ranks_b.mean()
    denominator = float(np.sqrt(np.dot(centered_a, centered_a) * np.dot(centered_b, centered_b)))
    return float(np.dot(centered_a, centered_b) / denominator) if denominator else float("nan")


def top_k_overlap(values_a: torch.Tensor, values_b: torch.Tensor, k: int) -> float:
    """Return |TopK(a) intersect TopK(b)| / K, as defined by Stage 5."""

    if values_a.ndim != 1 or values_b.ndim != 1 or values_a.shape != values_b.shape:
        raise ValueError("ranking inputs must be one-dimensional arrays of equal shape")
    count = min(max(int(k), 1), values_a.numel())
    top_a = set(torch.argsort(values_a, descending=True, stable=True)[:count].tolist())
    top_b = set(torch.argsort(values_b, descending=True, stable=True)[:count].tolist())
    return len(top_a & top_b) / count


def _token_for_debug(index: int, inverse_vocab: dict[int, str], blank: int) -> str:
    if index == blank:
        return "blank"
    return inverse_vocab.get(index, f"<id:{index}>")


def _format_debug_alignment(
    utt_id: str,
    transcript: str,
    alignment: torch.Tensor,
    greedy: torch.Tensor,
    blank: int,
    inverse_vocab: dict[int, str],
    valid_frames: int,
) -> str:
    aligned_tokens = " ".join(
        _token_for_debug(int(index), inverse_vocab, blank) for index in alignment.cpu().tolist()
    )
    greedy_tokens = " ".join(
        _token_for_debug(int(index), inverse_vocab, blank) for index in greedy.cpu().tolist()
    )
    return (
        f"{utt_id}\n\n"
        f"GT:\n{transcript}\n\n"
        f"alignment:\n{aligned_tokens}\n\n"
        f"greedy:\n{greedy_tokens}\n\n"
        f"valid_frames:\n{valid_frames}\n"
    )


def _atomic_torch_save(value: torch.Tensor, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value.cpu(), temporary)
    temporary.replace(path)


def _json_value(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def compute_utility(args: argparse.Namespace) -> dict[str, Any]:
    """Run the streaming attribution pass and write all Stage 5 artifacts."""

    if args.reference_dim < 1 or args.delta_dim < 1:
        raise ValueError("reference-dim and delta-dim must be positive")
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is non-empty; pass --overwrite to reuse it")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab(args.vocab)
    blank = blank_id(vocab)
    model, checkpoint = load_full_delta_teacher(
        args.checkpoint,
        vocab,
        args.reference_dim,
        args.delta_dim,
        device,
    )
    dataset = CachedFeatureDataset(
        args.manifest,
        args.feature_root,
        "full_delta",
        vocab,
        feature_split=args.feature_split,
        expected_dim=args.reference_dim,
        max_examples=args.max_utterances,
    )
    if not len(dataset):
        raise ValueError("utility dataset is empty")

    dimension = args.delta_dim
    help_count = torch.zeros(dimension, dtype=torch.long)
    harm_count = torch.zeros(dimension, dtype=torch.long)
    zero_count = torch.zeros(dimension, dtype=torch.long)
    contribution_sum = torch.zeros(dimension, dtype=torch.float64)
    uncertainty_contribution_sum = torch.zeros(dimension, dtype=torch.float64)
    magnitude_sum = torch.zeros(dimension, dtype=torch.float64)
    num_frames = 0
    aligned_utterances = 0
    failed: list[dict[str, str]] = []
    debug_examples: list[str] = []
    inverse_vocab = {int(index): str(token) for token, index in vocab.items()}
    w_delta = model.classifier.weight[:, args.reference_dim :].detach()

    for index in range(len(dataset)):
        item: dict[str, Any] | None = None
        try:
            item = dataset[index]
            features = item["features"]
            if features.ndim != 2 or features.shape[1] != args.reference_dim + args.delta_dim:
                raise ValueError(
                    f"expected concatenated feature shape [T,{args.reference_dim + args.delta_dim}], "
                    f"got {tuple(features.shape)}"
                )
            reference = features[:, : args.reference_dim]
            delta = features[:, args.reference_dim :]
            if reference.shape != delta.shape:
                raise ValueError(
                    f"reference/delta shape mismatch: {tuple(reference.shape)} vs {tuple(delta.shape)}"
                )
            with torch.inference_mode():
                logits = model(features.to(device).unsqueeze(0))[0]
                target = item["targets"].to(device)
                alignment, _ = forced_align_ctc(logits, target, blank)
            if alignment.shape != (features.shape[0],):
                raise ValueError(
                    f"alignment length {alignment.shape} does not match logits length {features.shape[0]}"
                )
            aligned_targets, competitors, delta_valid, q = attribute_nonblank_frames(
                logits,
                alignment,
                delta.to(device),
                w_delta,
                blank,
            )
            if not aligned_targets.numel():
                raise ValueError("alignment produced no non-blank frames")
            if aligned_targets.min() < 0 or aligned_targets.max() >= len(vocab):
                raise ValueError("aligned target id is outside the classifier vocabulary")
            if q.shape != delta_valid.shape or q.shape[1] != dimension:
                raise ValueError(f"unexpected q shape {tuple(q.shape)}")

            q_cpu = q.detach().cpu()
            delta_valid_cpu = delta_valid.detach().cpu()
            logits_valid = logits[alignment != blank]
            target_probability = torch.softmax(logits_valid, dim=-1).gather(
                1, aligned_targets.unsqueeze(1)
            ).squeeze(1)
            _, contribution_sum_utt, uncertainty_sum_utt = frame_utility_sums(
                q_cpu,
                target_probability.detach().cpu(),
            )
            help_count += (q_cpu > 0).sum(dim=0)
            harm_count += (q_cpu < 0).sum(dim=0)
            zero_count += (q_cpu == 0).sum(dim=0)
            contribution_sum += contribution_sum_utt.double()
            uncertainty_contribution_sum += uncertainty_sum_utt.double()
            magnitude_sum += delta_valid_cpu.abs().double().sum(dim=0)
            num_frames += int(q.shape[0])
            aligned_utterances += 1

            if len(debug_examples) < args.debug_examples:
                greedy = logits.argmax(dim=1)
                debug_examples.append(
                    _format_debug_alignment(
                        str(item["utt_id"]),
                        str(item["transcript"]),
                        alignment,
                        greedy,
                        blank,
                        inverse_vocab,
                        int(q.shape[0]),
                    )
                )
        except Exception as error:
            failed.append(
                {
                    "utt_id": str(item["utt_id"]) if item is not None else f"index:{index}",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            if args.fail_on_error:
                raise
        if index == 0 or (index + 1) % args.log_every == 0:
            print(
                {
                    "processed": index + 1,
                    "total": len(dataset),
                    "aligned": aligned_utterances,
                    "failed": len(failed),
                },
                flush=True,
            )

    if num_frames == 0:
        raise RuntimeError("utility pass produced no aligned non-blank frames")

    utility = (help_count.double() - harm_count.double()) / num_frames
    utility_v2 = contribution_sum / num_frames
    utility_v3 = uncertainty_contribution_sum / num_frames
    magnitude = magnitude_sum / num_frames
    if utility.min() < -1 or utility.max() > 1:
        raise AssertionError("utility must lie in [-1, 1]")
    utility_ranking = torch.argsort(utility, descending=True, stable=True)
    utility_v2_ranking = torch.argsort(utility_v2, descending=True, stable=True)
    utility_v3_ranking = torch.argsort(utility_v3, descending=True, stable=True)
    magnitude_ranking = torch.argsort(magnitude, descending=True, stable=True)
    _atomic_torch_save(utility, args.output_dir / "utility.pt")
    _atomic_torch_save(utility_v2, args.output_dir / "utility_v2.pt")
    _atomic_torch_save(utility_v3, args.output_dir / "utility_v3.pt")
    _atomic_torch_save(magnitude, args.output_dir / "magnitude.pt")
    _atomic_torch_save(utility_ranking, args.output_dir / "utility_ranking.pt")
    _atomic_torch_save(utility_v2_ranking, args.output_dir / "utility_v2_ranking.pt")
    _atomic_torch_save(utility_v3_ranking, args.output_dir / "utility_v3_ranking.pt")
    _atomic_torch_save(magnitude_ranking, args.output_dir / "magnitude_ranking.pt")
    (args.output_dir / "debug_alignment.txt").write_text(
        "\n".join(debug_examples), encoding="utf-8"
    )

    utility_np = utility.numpy()
    magnitude_np = magnitude.numpy()
    stats: dict[str, Any] = {
        "protocol": PROTOCOL,
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "feature_root": str(args.feature_root),
        "feature_split": args.feature_split,
        "vocab": str(args.vocab),
        "device": str(device),
        "alignment_backend": "torch_viterbi_ctc",
        "reference_dim": args.reference_dim,
        "delta_dim": dimension,
        "vocab_size": len(vocab),
        "blank_id": blank,
        "num_utterances": len(dataset),
        "num_aligned": aligned_utterances,
        "num_failed": len(failed),
        "failure_rate": len(failed) / len(dataset),
        "num_valid_frames": num_frames,
        "help_count": help_count.tolist(),
        "harm_count": harm_count.tolist(),
        "zero_count": zero_count.tolist(),
        "spearman_utility_magnitude": _json_value(spearman(utility_np, magnitude_np)),
        "overlap_at_256": top_k_overlap(utility, magnitude, 256),
        "overlap_at_512": top_k_overlap(utility, magnitude, 512),
        "top_utility": utility_ranking[:10].tolist(),
        "top_utility_v2": utility_v2_ranking[:10].tolist(),
        "top_utility_v3": utility_v3_ranking[:10].tolist(),
        "top_magnitude": magnitude_ranking[:10].tolist(),
        "utility_variants": {
            "v1": {
                "aggregation": "E[sign(q)]",
                "score_path": str(args.output_dir / "utility.pt"),
                "ranking_path": str(args.output_dir / "utility_ranking.pt"),
            },
            "v2": {
                "aggregation": "E[q]",
                "score_path": str(args.output_dir / "utility_v2.pt"),
                "ranking_path": str(args.output_dir / "utility_v2_ranking.pt"),
                "spearman_magnitude": _json_value(spearman(utility_v2.numpy(), magnitude_np)),
                "overlap_at_256": top_k_overlap(utility_v2, magnitude, 256),
                "overlap_at_512": top_k_overlap(utility_v2, magnitude, 512),
            },
            "v3": {
                "aggregation": "E[q * (1 - p(target))]",
                "score_path": str(args.output_dir / "utility_v3.pt"),
                "ranking_path": str(args.output_dir / "utility_v3_ranking.pt"),
                "spearman_magnitude": _json_value(spearman(utility_v3.numpy(), magnitude_np)),
                "overlap_at_256": top_k_overlap(utility_v3, magnitude, 256),
                "overlap_at_512": top_k_overlap(utility_v3, magnitude, 512),
            },
        },
        "failed_examples": failed[:20],
        "checkpoint_protocol": checkpoint.get("protocol"),
        "checkpoint_epoch": checkpoint.get("epoch"),
    }
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "num_aligned": aligned_utterances,
                "num_failed": len(failed),
                "num_valid_frames": num_frames,
                "spearman_utility_magnitude": stats["spearman_utility_magnitude"],
                "overlap_at_256": stats["overlap_at_256"],
                "overlap_at_512": stats["overlap_at_512"],
                "top_utility": stats["top_utility"],
                "top_utility_v2": stats["top_utility_v2"],
                "top_utility_v3": stats["top_utility_v3"],
                "top_magnitude": stats["top_magnitude"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/runs/stage4/full_delta/best.pt"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/arctic_step2/l2/train_utility.jsonl"),
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("artifacts/features/stage3_l2"),
    )
    parser.add_argument("--feature-split", default="train")
    parser.add_argument("--vocab", type=Path, default=Path("assets/ctc_vocab/vocab.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage5"))
    parser.add_argument("--reference-dim", type=int, default=1024)
    parser.add_argument("--delta-dim", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-utterances", type=int, default=None)
    parser.add_argument("--debug-examples", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_utterances is not None and args.max_utterances < 1:
        raise ValueError("max-utterances must be positive")
    if args.debug_examples < 0 or args.log_every < 1:
        raise ValueError("debug-examples must be non-negative and log-every must be positive")
    compute_utility(args)


if __name__ == "__main__":
    main()
