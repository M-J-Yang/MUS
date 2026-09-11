#!/usr/bin/env python3
"""Evaluate coordinate dependence under equivalent orthogonal reparameterizations.

The experiment operates on the official cached final-layer streams.  For each
Haar-style orthogonal matrix Q it rotates E0/Eft/Delta by Q and replaces the
linear CTC head with W' = W Q.T.  It then verifies function equivalence and
recomputes the coordinate-wise magnitude, gradient, and CTC-Taylor scores.

The resulting concentration statistics answer two separate questions:

* Does the ASR function change?  It should not, up to floating-point error.
* Do coordinate-level rankings and Top-K concentration change?  If they do,
  those observations are properties of the native parameterization rather
  than invariant properties of the model function.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from usde.ctc import make_processor  # noqa: E402
from usde.metrics import word_error_rate  # noqa: E402
from usde.model import load_ctc_model  # noqa: E402
from usde.reparameterization import (  # noqa: E402
    concentration_summary,
    haar_orthogonal,
    reparameterize_linear_head,
    rotate_features,
    top_k_jaccard,
)
from usde.shift import (  # noqa: E402
    ShiftFeatureDataset,
    collate_shift,
    ctc_gradient_and_taylor_batch_sums,
)


PROTOCOL = "equivalent_orthogonal_reparameterization_v1"
DEFAULT_ROTATION_SEEDS = (1337, 2027, 31415, 4242, 9001)
DEFAULT_TOP_K_FRACTIONS = (0.01, 0.05, 0.10, 0.25)
SCORE_NAMES = ("magnitude", "gradient", "utility")


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def make_loader(
    manifest_root: Path,
    cache_root: Path,
    tokenizer: Any,
    split: str,
    hidden_dim: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    manifest = manifest_root / f"{split}.jsonl"
    dataset = ShiftFeatureDataset(
        manifest,
        cache_root,
        tokenizer,
        feature_split=split,
        expected_dim=hidden_dim,
        load_eft=True,
    )
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": collate_shift,
        "pin_memory": False,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def compute_scores(
    head: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    blank: int,
    *,
    q: torch.Tensor | None = None,
    log_every: int = 100,
) -> dict[str, Any]:
    """Compute the three coordinate-wise scores on one cached split."""

    gradient_sum: torch.Tensor | None = None
    utility_sum: torch.Tensor | None = None
    magnitude_sum: torch.Tensor | None = None
    frames = processed = 0
    total_loss = 0.0
    for batch_index, batch in enumerate(loader, start=1):
        e0 = batch["e0"].to(device, non_blocking=True)
        delta = batch["delta"].to(device, non_blocking=True)
        if q is not None:
            e0 = rotate_features(e0, q)
            delta = rotate_features(delta, q)
        if gradient_sum is None:
            dimension = int(delta.shape[-1])
            gradient_sum = torch.zeros(dimension, dtype=torch.float64)
            utility_sum = torch.zeros(dimension, dtype=torch.float64)
            magnitude_sum = torch.zeros(dimension, dtype=torch.float64)

        with torch.enable_grad():
            gradient, utility, batch_frames, batch_loss = ctc_gradient_and_taylor_batch_sums(
                head,
                e0,
                delta,
                batch["targets"].to(device, non_blocking=True),
                batch["feature_lengths"].to(device),
                batch["target_lengths"].to(device),
                blank,
            )
        valid = (
            torch.arange(e0.shape[1], device=device).unsqueeze(0)
            < batch["feature_lengths"].to(device).unsqueeze(1)
        )
        assert gradient_sum is not None
        assert utility_sum is not None
        assert magnitude_sum is not None
        gradient_sum += gradient.double().cpu()
        utility_sum += utility.double().cpu()
        magnitude_sum += (
            delta.abs().masked_fill(~valid.unsqueeze(-1), 0.0)
            .sum(dim=(0, 1))
            .double()
            .cpu()
        )
        frames += batch_frames
        count = len(batch["utt_ids"])
        processed += count
        total_loss += float(batch_loss.detach().cpu()) * count
        if batch_index == 1 or processed % log_every < count or processed == len(loader.dataset):
            print(
                {
                    "score_pass_processed": processed,
                    "score_pass_total": len(loader.dataset),
                    "rotated": q is not None,
                },
                flush=True,
            )

    if frames < 1 or gradient_sum is None or utility_sum is None or magnitude_sum is None:
        raise RuntimeError("score pass produced no valid frames")
    return {
        "num_utterances": processed,
        "num_valid_frames": frames,
        "mean_ctc_loss": total_loss / max(processed, 1),
        "scores": {
            "gradient": gradient_sum / frames,
            "utility": utility_sum / frames,
            "magnitude": magnitude_sum / frames,
        },
    }


def _raw_ctc_gradient(
    head: torch.nn.Module,
    e0: torch.Tensor,
    delta: torch.Tensor,
    targets: torch.Tensor,
    feature_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int,
) -> torch.Tensor:
    """Return dL/dDelta for one batch without reducing feature coordinates."""

    delta_leaf = delta.detach().requires_grad_(True)
    logits = head(e0.detach() + delta_leaf)
    log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
    losses = torch.nn.CTCLoss(blank=blank, reduction="none", zero_infinity=True)(
        log_probs,
        targets,
        feature_lengths,
        target_lengths,
    )
    normalized = losses / target_lengths.to(device=losses.device, dtype=losses.dtype)
    (gradient,) = torch.autograd.grad(normalized.sum(), delta_leaf)
    return gradient.detach()


def gradient_equivalence(
    head: torch.nn.Module,
    rotated_head: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    blank: int,
    q: torch.Tensor,
) -> dict[str, Any]:
    """Check g' = Qg on one utility batch in row-layout notation."""

    e0 = batch["e0"].to(device, non_blocking=True)
    delta = batch["delta"].to(device, non_blocking=True)
    targets = batch["targets"].to(device, non_blocking=True)
    feature_lengths = batch["feature_lengths"].to(device)
    target_lengths = batch["target_lengths"].to(device)
    native_gradient = _raw_ctc_gradient(
        head, e0, delta, targets, feature_lengths, target_lengths, blank
    )
    rotated_gradient = _raw_ctc_gradient(
        rotated_head,
        rotate_features(e0, q),
        rotate_features(delta, q),
        targets,
        feature_lengths,
        target_lengths,
        blank,
    )
    expected = rotate_features(native_gradient, q)
    difference = rotated_gradient - expected
    max_abs_error = float(difference.abs().max().detach().cpu())
    scale = max(float(expected.abs().max().detach().cpu()), 1e-12)
    relative_error = max_abs_error / scale
    return {
        "max_abs_error": max_abs_error,
        "relative_error": relative_error,
        "pass": bool(torch.allclose(rotated_gradient, expected, atol=1e-4, rtol=1e-3)),
    }


def decode(tokenizer: Any, logits: torch.Tensor, length: int) -> str:
    ids = logits.argmax(dim=-1) if logits.ndim == 2 else logits
    return str(tokenizer.decode(ids[:length].tolist(), group_tokens=True)).lower().strip()


def evaluate_function_equivalence(
    head: torch.nn.Module,
    rotated_head: torch.nn.Module,
    loader: DataLoader,
    tokenizer: Any,
    device: torch.device,
    q: torch.Tensor,
    logit_atol: float,
) -> dict[str, Any]:
    """Verify logits, greedy predictions, and WER on one split."""

    references: list[str] = []
    native_hypotheses: list[str] = []
    rotated_hypotheses: list[str] = []
    max_logit_error = 0.0
    frame_prediction_mismatches = 0
    stable_frame_prediction_mismatches = 0
    compared_frames = 0
    decoded_prediction_mismatches = 0
    with torch.inference_mode():
        for batch in loader:
            eft = batch["eft"].to(device, non_blocking=True)
            rotated_eft = rotate_features(eft, q)
            native_logits = head(eft)
            rotated_logits = rotated_head(rotated_eft)
            max_logit_error = max(
                max_logit_error,
                float((native_logits - rotated_logits).abs().max().detach().cpu()),
            )
            native_ids = native_logits.argmax(dim=-1)
            rotated_ids = rotated_logits.argmax(dim=-1)
            for native_logit_row, rotated_logit_row, native_id_row, rotated_id_row, length in zip(
                native_logits,
                rotated_logits,
                native_ids,
                rotated_ids,
                batch["feature_lengths"],
                strict=True,
            ):
                frame_count = int(length)
                native_logit_row = native_logit_row[:frame_count]
                rotated_logit_row = rotated_logit_row[:frame_count]
                native_id_row = native_id_row[:frame_count]
                rotated_id_row = rotated_id_row[:frame_count]
                native_hypothesis = decode(tokenizer, native_logit_row.cpu(), frame_count)
                rotated_hypothesis = decode(tokenizer, rotated_logit_row.cpu(), frame_count)
                native_hypotheses.append(native_hypothesis)
                rotated_hypotheses.append(rotated_hypothesis)
                decoded_prediction_mismatches += int(native_hypothesis != rotated_hypothesis)
                differing = native_id_row != rotated_id_row
                frame_prediction_mismatches += int(differing.sum().cpu())
                compared_frames += frame_count
                if bool(differing.any()):
                    # A different argmax is not a stable functional change if
                    # the two competing logits are within the numerical error
                    # budget of the equivalent matrix multiplications.
                    native_choice = native_logit_row.gather(1, native_id_row.unsqueeze(1)).squeeze(1)
                    rotated_choice_in_native = native_logit_row.gather(1, rotated_id_row.unsqueeze(1)).squeeze(1)
                    stable = differing & (
                        (native_choice - rotated_choice_in_native).abs() > 2 * logit_atol
                    )
                    stable_frame_prediction_mismatches += int(stable.sum().cpu())
            references.extend(batch["transcripts"])

    native_wer = float(word_error_rate(references, native_hypotheses))
    rotated_wer = float(word_error_rate(references, rotated_hypotheses))
    return {
        "examples": len(references),
        "compared_frames": compared_frames,
        "max_abs_logit_error": max_logit_error,
        "frame_prediction_mismatches": frame_prediction_mismatches,
        "stable_frame_prediction_mismatches": stable_frame_prediction_mismatches,
        "numerically_explainable_frame_mismatches": (
            frame_prediction_mismatches - stable_frame_prediction_mismatches
        ),
        "frame_prediction_mismatch_rate": frame_prediction_mismatches / max(compared_frames, 1),
        "decoded_prediction_mismatches": decoded_prediction_mismatches,
        "native_wer": native_wer,
        "rotated_wer": rotated_wer,
        "wer_abs_difference": abs(native_wer - rotated_wer),
        "pass": bool(
            max_logit_error <= logit_atol
            and stable_frame_prediction_mismatches == 0
            and decoded_prediction_mismatches == 0
            and native_wer == rotated_wer
        ),
    }


def _summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty list")
    return {
        "mean": float(mean(values)),
        "std": float(stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def coordinate_summary(
    native_scores: dict[str, torch.Tensor],
    rotations: list[dict[str, Any]],
    top_k_fractions: tuple[float, ...],
) -> dict[str, Any]:
    """Aggregate native-vs-rotation concentration and mask instability."""

    result: dict[str, Any] = {}
    for name in SCORE_NAMES:
        native = concentration_summary(native_scores[name], top_k_fractions)
        rotation_concentrations = [item["concentration"][name] for item in rotations]
        null: dict[str, Any] = {}
        for field in (
            "gini",
            "normalized_entropy",
            "effective_support_fraction",
        ):
            values = [float(item[field]) for item in rotation_concentrations]
            null[field] = _summary_stats(values)
            null[field]["native"] = float(native[field])
            null[field]["native_percentile"] = 100.0 * sum(
                value <= float(native[field]) for value in values
            ) / len(values)
        native_top_mass = native["top_k_mass"]
        top_mass_null: dict[str, Any] = {}
        mask_overlap: dict[str, Any] = {}
        for fraction in top_k_fractions:
            key = f"{fraction:g}"
            values = [float(item["top_k_mass"][key]) for item in rotation_concentrations]
            top_mass_null[key] = _summary_stats(values)
            top_mass_null[key]["native"] = float(native_top_mass[key])
            top_mass_null[key]["native_percentile"] = 100.0 * sum(
                value <= float(native_top_mass[key]) for value in values
            ) / len(values)
            count = max(1, round(native_scores[name].numel() * fraction))
            overlaps = [float(item["top_k_jaccard"][name][key]) for item in rotations]
            mask_overlap[key] = _summary_stats(overlaps)
            mask_overlap[key]["native_vs_rotation"] = overlaps
            mask_overlap[key]["k"] = count
        result[name] = {
            "native": native,
            "rotation_null": null,
            "top_k_mass_null": top_mass_null,
            "top_k_jaccard": mask_overlap,
        }
    return result


def write_summary(result: dict[str, Any], path: Path) -> None:
    coordinate = result["coordinate_summary"]
    lines = [
        "# Equivalent orthogonal reparameterization control",
        "",
        "The rotated representation uses E' = Q E and the rotated linear CTC head uses W' = W Q^T.",
        "Function-equivalence checks should pass while coordinate-wise concentration may change.",
        "",
        "## Function equivalence",
        "",
        "| Split | Rotation seed | Max logit error | Frame mismatches (actual/stable) | Decoded mismatches | Native WER | Rotated WER | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for item in result["rotations"]:
        for split, check in item["function_equivalence"].items():
            lines.append(
                f"| {split} | {item['seed']} | {check['max_abs_logit_error']:.3g} | "
                f"{check['frame_prediction_mismatches']}/{check['stable_frame_prediction_mismatches']} | "
                f"{check['decoded_prediction_mismatches']} | {check['native_wer'] * 100:.3f}% | "
                f"{check['rotated_wer'] * 100:.3f}% | "
                f"{'PASS' if check['pass'] else 'FAIL'} |"
            )
    lines += [
        "",
        "## Coordinate concentration",
        "",
        "The native column is the original basis. Rotation-null values are mean ± sample standard deviation.",
        "",
        "| Score | Top-K fraction | Native mass | Rotation-null mass | Native Top-K Jaccard vs rotations |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in SCORE_NAMES:
        summary = coordinate[name]
        for fraction, values in summary["top_k_mass_null"].items():
            overlap = summary["top_k_jaccard"][fraction]
            lines.append(
                f"| {name} | {float(fraction) * 100:g}% | "
                f"{values['native']:.4f} | {values['mean']:.4f} ± {values['std']:.4f} | "
                f"{overlap['mean']:.4f} ± {overlap['std']:.4f} |"
            )
    lines += [
        "",
        "Gradient relation check (rotated gradient vs Q times native gradient):",
        "",
    ]
    for item in result["rotations"]:
        check = item["gradient_equivalence"]
        lines.append(
            f"- seed {item['seed']}: max error {check['max_abs_error']:.3g}, "
            f"relative error {check['relative_error']:.3g}, "
            f"{'PASS' if check['pass'] else 'FAIL'}"
        )
    lines += ["", "Identity gate: " + ("PASS" if result["identity_pass"] else "FAIL"), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1 or args.num_workers < 0 or args.log_every < 1:
        raise ValueError("batch-size must be positive; num-workers non-negative; log-every positive")
    if not args.rotation_seeds:
        raise ValueError("rotation-seeds must not be empty")
    if len(set(args.rotation_seeds)) != len(args.rotation_seeds):
        raise ValueError("rotation-seeds must be unique")
    if not args.top_k_fractions or any(fraction <= 0 or fraction > 1 for fraction in args.top_k_fractions):
        raise ValueError("top-k-fractions must lie in (0, 1]")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is non-empty; pass --overwrite to reuse it")
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")

    required_splits = ["train_utility", *args.verify_splits]
    for split in required_splits:
        manifest = args.manifest_root / f"{split}.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)

    processor = make_processor(str(args.checkpoint), None)
    model = load_ctc_model(args.checkpoint).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hidden_dim = int(getattr(model.config, "hidden_size", 0))
    head = getattr(model, "lm_head", None)
    if hidden_dim < 1 or head is None:
        raise ValueError("checkpoint must provide hidden_size and lm_head")
    if not isinstance(head, torch.nn.Linear) or head.in_features != hidden_dim:
        raise ValueError(
            "equivalent reparameterization requires a pure linear lm_head with "
            f"{hidden_dim} input features"
        )
    blank = processor.tokenizer.pad_token_id
    if blank is None:
        raise ValueError("processor tokenizer must provide a CTC pad/blank token")
    blank = int(blank)

    utility_loader = make_loader(
        args.manifest_root,
        args.cache_root,
        processor.tokenizer,
        "train_utility",
        hidden_dim,
        args.batch_size,
        args.num_workers,
    )
    native_pass = compute_scores(
        head,
        utility_loader,
        device,
        blank,
        log_every=args.log_every,
    )
    native_scores = native_pass["scores"]
    _atomic_torch_save(native_scores, args.output_dir / "native_scores.pt")
    probe_batch = next(iter(utility_loader))

    rotations: list[dict[str, Any]] = []
    for seed in args.rotation_seeds:
        q_cpu = haar_orthogonal(hidden_dim, int(seed), dtype=head.weight.dtype)
        q = q_cpu.to(device=head.weight.device, dtype=head.weight.dtype)
        rotated_head = reparameterize_linear_head(head, q)
        rotated_pass = compute_scores(
            rotated_head,
            utility_loader,
            device,
            blank,
            q=q,
            log_every=args.log_every,
        )
        rotated_scores = rotated_pass["scores"]
        _atomic_torch_save(rotated_scores, args.output_dir / f"rotation_{seed}_scores.pt")

        concentration = {
            name: concentration_summary(rotated_scores[name], tuple(args.top_k_fractions))
            for name in SCORE_NAMES
        }
        top_k_overlap: dict[str, dict[str, float]] = {}
        for name in SCORE_NAMES:
            top_k_overlap[name] = {}
            for fraction in args.top_k_fractions:
                key = f"{fraction:g}"
                count = max(1, round(hidden_dim * fraction))
                top_k_overlap[name][key] = top_k_jaccard(
                    native_scores[name], rotated_scores[name], count
                )

        gradient_check = gradient_equivalence(
            head,
            rotated_head,
            probe_batch,
            device,
            blank,
            q,
        )
        function_equivalence: dict[str, Any] = {}
        for split in args.verify_splits:
            loader = make_loader(
                args.manifest_root,
                args.cache_root,
                processor.tokenizer,
                split,
                hidden_dim,
                args.batch_size,
                args.num_workers,
            )
            function_equivalence[split] = evaluate_function_equivalence(
                head,
                rotated_head,
                loader,
                processor.tokenizer,
                device,
                q,
                args.logit_atol,
            )

        q_cpu_error = float(
            (q_cpu.transpose(0, 1).matmul(q_cpu) - torch.eye(hidden_dim, dtype=q_cpu.dtype))
            .abs()
            .max()
        )
        rotations.append(
            {
                "seed": int(seed),
                "orthogonality_max_abs_error": q_cpu_error,
                "num_utterances": rotated_pass["num_utterances"],
                "num_valid_frames": rotated_pass["num_valid_frames"],
                "mean_ctc_loss": rotated_pass["mean_ctc_loss"],
                "concentration": concentration,
                "top_k_jaccard": top_k_overlap,
                "gradient_equivalence": gradient_check,
                "function_equivalence": function_equivalence,
            }
        )
        print(
            {
                "rotation_seed": int(seed),
                "identity_pass": all(item["pass"] for item in function_equivalence.values()),
                "gradient_pass": gradient_check["pass"],
            },
            flush=True,
        )

    coordinate = coordinate_summary(
        native_scores,
        rotations,
        tuple(args.top_k_fractions),
    )
    identity_pass = all(
        check["pass"]
        for item in rotations
        for check in item["function_equivalence"].values()
    ) and all(item["gradient_equivalence"]["pass"] for item in rotations)
    result: dict[str, Any] = {
        "protocol": PROTOCOL,
        "checkpoint": str(args.checkpoint),
        "manifest_root": str(args.manifest_root),
        "cache_root": str(args.cache_root),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "hidden_dim": hidden_dim,
        "score_definition": {
            "magnitude": "mean_over_valid_frames(abs(Delta))",
            "gradient": "mean_over_valid_frames(abs(dL_CTC/dDelta))",
            "utility": "mean_over_valid_frames(abs(Delta * dL_CTC/dDelta))",
        },
        "transform": {
            "representation_column": "E' = Q E",
            "row_layout": "E'_row = E_row Q.T",
            "head": "W' = W Q.T",
            "bias": "unchanged",
            "rotation_distribution": "Haar-style QR of a Gaussian matrix",
        },
        "rotation_seeds": [int(seed) for seed in args.rotation_seeds],
        "top_k_fractions": [float(fraction) for fraction in args.top_k_fractions],
        "logit_atol": args.logit_atol,
        "native": {
            "num_utterances": native_pass["num_utterances"],
            "num_valid_frames": native_pass["num_valid_frames"],
            "mean_ctc_loss": native_pass["mean_ctc_loss"],
            "concentration": {
                name: concentration_summary(native_scores[name], tuple(args.top_k_fractions))
                for name in SCORE_NAMES
            },
        },
        "coordinate_summary": coordinate,
        "rotations": rotations,
        "identity_pass": identity_pass,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary(result, args.output_dir / "summary.md")
    print(
        json.dumps(
            {
                "metrics": str(metrics_path),
                "summary": str(args.output_dir / "summary.md"),
                "identity_pass": identity_pass,
            },
            sort_keys=True,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rotation-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_ROTATION_SEEDS),
        help="seeds for independent Haar-style orthogonal bases",
    )
    parser.add_argument(
        "--verify-splits",
        nargs="+",
        default=["dev", "test"],
        help="cached splits on which logits and greedy predictions are compared",
    )
    parser.add_argument(
        "--top-k-fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_TOP_K_FRACTIONS),
        help="coordinate fractions for concentration and Top-K mask overlap",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--logit-atol", type=float, default=1e-4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.logit_atol <= 0:
        parser.error("logit-atol must be positive")
    args.top_k_fractions = tuple(args.top_k_fractions)
    args.rotation_seeds = tuple(args.rotation_seeds)
    args.verify_splits = tuple(args.verify_splits)
    run(args)


if __name__ == "__main__":
    main()
