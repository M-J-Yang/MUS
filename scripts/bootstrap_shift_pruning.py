#!/usr/bin/env python3
"""Paired utterance-bootstrap confidence intervals for frozen shift pruning.

This is a post-hoc supplement to the accepted Fold 0 frozen-head evaluation.
It replays the cached E0/Delta streams with the original fine-tuned CTC head,
computes utterance-level edit counts, and resamples complete utterances with
replacement.  Each bootstrap replicate therefore uses one shared sample of
utterances for both systems in a comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from usde.ctc import make_processor  # noqa: E402
from usde.metrics import word_error_rate  # noqa: E402
from usde.model import load_ctc_model  # noqa: E402
from usde.shift import ShiftFeatureDataset, collate_shift  # noqa: E402


PROTOCOL = "official_fold0_frozen_head_paired_utterance_bootstrap_v1"
COMPARISONS = {
    "utility75_vs_full": ("utility75", "full"),
    "utility50_vs_full": ("utility50", "full"),
    "utility50_vs_magnitude50": ("utility50", "magnitude50"),
    "utility75_vs_magnitude75": ("utility75", "magnitude75"),
}


def load_ranking(path: Path, hidden_dim: int, label: str) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        value = next(
            (value[key] for key in ("ranking", "tensor", "utility", "features")
             if isinstance(value.get(key), torch.Tensor)),
            None,
        )
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path}: expected a ranking tensor")
    value = value.to(dtype=torch.long, device="cpu")
    if (
        value.ndim != 1
        or value.numel() != hidden_dim
        or torch.unique(value).numel() != hidden_dim
        or int(value.min()) < 0
        or int(value.max()) >= hidden_dim
    ):
        raise ValueError(f"{label}: expected a complete permutation of {hidden_dim} coordinates")
    return value


def compose(e0: torch.Tensor, delta: torch.Tensor, keep: torch.Tensor | None) -> torch.Tensor:
    if keep is None:
        return e0 + delta
    selected = torch.zeros_like(delta)
    if keep.numel():
        indices = keep.to(device=e0.device, dtype=torch.long)
        selected[..., indices] = delta[..., indices]
    return e0 + selected


def decode(tokenizer: Any, logits: torch.Tensor, length: int) -> str:
    ids = logits.argmax(dim=-1) if logits.ndim == 2 else logits
    return str(tokenizer.decode(ids[:length].tolist(), group_tokens=True)).lower().strip()


def edit_distance(reference: str, hypothesis: str) -> int:
    ref = reference.split()
    hyp = hypothesis.split()
    previous = list(range(len(hyp) + 1))
    for ref_word in ref:
        current = [previous[0] + 1]
        for hyp_index, hyp_word in enumerate(hyp, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_word != hyp_word),
                )
            )
        previous = current
    return previous[-1]


def utterance_counts(
    references: list[str], hypotheses: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have equal length")
    errors = np.asarray([edit_distance(ref, hyp) for ref, hyp in zip(references, hypotheses, strict=True)], dtype=np.int64)
    words = np.asarray([len(ref.split()) for ref in references], dtype=np.int64)
    if not len(words) or np.any(words <= 0):
        raise ValueError("paired bootstrap requires non-empty reference utterances")
    return errors, words


def paired_bootstrap(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    reference_words: np.ndarray,
    replicates: int,
    seed: int,
    chunk_size: int = 512,
) -> np.ndarray:
    """Return paired bootstrap WER differences in percentage points."""

    if errors_a.shape != errors_b.shape or errors_a.shape != reference_words.shape:
        raise ValueError("paired bootstrap arrays must have equal shape")
    if errors_a.ndim != 1 or not len(errors_a):
        raise ValueError("paired bootstrap arrays must be non-empty vectors")
    if replicates < 1 or chunk_size < 1:
        raise ValueError("replicates and chunk_size must be positive")
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        sample = rng.integers(0, len(errors_a), size=(stop - start, len(errors_a)))
        denominator = reference_words[sample].sum(axis=1)
        numerator = errors_a[sample].sum(axis=1) - errors_b[sample].sum(axis=1)
        differences[start:stop] = 100.0 * numerator / denominator
    return differences


def summarize_difference(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    reference_words: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    point_a = float(errors_a.sum() / reference_words.sum())
    point_b = float(errors_b.sum() / reference_words.sum())
    bootstrap = paired_bootstrap(
        errors_a, errors_b, reference_words, replicates=replicates, seed=seed
    )
    return {
        "system_a_wer": point_a,
        "system_a_wer_percent": 100.0 * point_a,
        "system_b_wer": point_b,
        "system_b_wer_percent": 100.0 * point_b,
        "difference_wer": point_a - point_b,
        "difference_percentage_points": 100.0 * (point_a - point_b),
        "ci_level": 0.95,
        "ci_method": "percentile paired bootstrap over utterances",
        "ci_lower_percentage_points": float(np.quantile(bootstrap, 0.025)),
        "ci_upper_percentage_points": float(np.quantile(bootstrap, 0.975)),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def evaluate_predictions(args: argparse.Namespace) -> tuple[dict[str, list[str]], list[str], dict[str, Any]]:
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    processor = make_processor(str(args.checkpoint), None)
    model = load_ctc_model(args.checkpoint).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hidden_dim = int(getattr(model.config, "hidden_size", 0))
    if hidden_dim < 1:
        raise ValueError("checkpoint must provide a positive hidden_size")

    utility = load_ranking(args.utility_ranking, hidden_dim, "utility ranking")
    magnitude = load_ranking(args.magnitude_ranking, hidden_dim, "magnitude ranking")
    count75 = max(1, round(hidden_dim * 0.75))
    count50 = max(1, round(hidden_dim * 0.50))
    conditions = {
        "full": None,
        "utility75": utility[:count75],
        "utility50": utility[:count50],
        "magnitude75": magnitude[:count75],
        "magnitude50": magnitude[:count50],
    }
    dataset = ShiftFeatureDataset(
        args.manifest,
        args.cache_root,
        processor.tokenizer,
        feature_split=args.feature_split,
        expected_dim=hidden_dim,
        load_eft=True,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "collate_fn": collate_shift,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    predictions = {name: [] for name in conditions}
    references: list[str] = []
    max_identity_error = 0.0
    with torch.inference_mode():
        for batch in DataLoader(dataset, **loader_kwargs):
            e0 = batch["e0"].to(device, non_blocking=True)
            eft = batch["eft"].to(device, non_blocking=True)
            delta = batch["delta"].to(device, non_blocking=True)
            max_identity_error = max(max_identity_error, float((e0 + delta - eft).abs().max().cpu()))
            lengths = batch["feature_lengths"].tolist()
            for name, keep in conditions.items():
                logits = model.lm_head(compose(e0, delta, keep))
                for row, length in zip(logits.cpu(), lengths, strict=True):
                    predictions[name].append(decode(processor.tokenizer, row, int(length)))
            references.extend(batch["transcripts"])
    if max_identity_error > 1e-5:
        raise RuntimeError(f"cache identity gate failed with max error {max_identity_error:.6g}")
    return predictions, references, {
        "examples": len(references),
        "hidden_dim": hidden_dim,
        "utility_retained_coordinates": count75,
        "utility_retained_coordinates_50": count50,
        "max_identity_abs_error": max_identity_error,
        "identity_pass": True,
        "device": str(device),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    predictions, references, evaluation = evaluate_predictions(args)
    counts = {
        name: utterance_counts(references, values)
        for name, values in predictions.items()
    }
    full_errors, reference_words = counts["full"]
    measured = {
        name: {
            "wer": float(errors.sum() / reference_words.sum()),
            "wer_percent": 100.0 * float(errors.sum() / reference_words.sum()),
        }
        for name, (errors, _) in counts.items()
    }
    differences: dict[str, Any] = {}
    for index, (label, (system_a, system_b)) in enumerate(COMPARISONS.items()):
        errors_a, words_a = counts[system_a]
        errors_b, words_b = counts[system_b]
        if not np.array_equal(words_a, reference_words) or not np.array_equal(words_b, reference_words):
            raise RuntimeError("reference word counts differ between paired systems")
        differences[label] = {
            "system_a": system_a,
            "system_b": system_b,
            **summarize_difference(
                errors_a,
                errors_b,
                reference_words,
                replicates=args.bootstrap_replicates,
                seed=args.seed + index,
            ),
        }
    result: dict[str, Any] = {
        "protocol": PROTOCOL,
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "cache_root": str(args.cache_root),
        "feature_split": args.feature_split,
        "utility_ranking": str(args.utility_ranking),
        "magnitude_ranking": str(args.magnitude_ranking),
        "evaluation": evaluation,
        "reference_words": int(reference_words.sum()),
        "measured": measured,
        "comparisons": differences,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Fold0 paired utterance-bootstrap confidence intervals",
        "",
        f"Test utterances: {evaluation['examples']}; reference words: {result['reference_words']}",
        "",
        "Positive differences mean the first system has higher WER than the second.",
        "The interval is the 2.5th–97.5th percentile of 10,000 paired resamples of complete utterances.",
        "",
        "| Comparison | WER A | WER B | Difference | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, item in differences.items():
        lines.append(
            f"| {label} | {item['system_a_wer_percent']:.3f}% | "
            f"{item['system_b_wer_percent']:.3f}% | "
            f"{item['difference_percentage_points']:+.3f} pp | "
            f"[{item['ci_lower_percentage_points']:+.3f}, {item['ci_upper_percentage_points']:+.3f}] pp |"
        )
    lines += [
        "",
        f"Identity gate: {'PASS' if evaluation['identity_pass'] else 'FAIL'} "
        f"(max |E0 + Delta - Eft| = {evaluation['max_identity_abs_error']:.3g}).",
        "",
    ]
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_markdown": str(args.output_markdown)}, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--feature-split", required=True)
    parser.add_argument("--utility-ranking", type=Path, required=True)
    parser.add_argument("--magnitude-ranking", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1337)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
