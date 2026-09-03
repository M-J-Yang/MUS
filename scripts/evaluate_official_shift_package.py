#!/usr/bin/env python3
"""Evaluate the official Fold0 frozen-head shift-pruning empirical package.

The evaluator reuses cached E0/Eft/Delta features and the original oracle CTC
head. It reports matched retention baselines and direct deletion interventions;
no CTC head retraining or healing is performed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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
from usde.shift import (  # noqa: E402
    ShiftFeatureDataset,
    collate_shift,
    ctc_gradient_and_taylor_batch_sums,
)


PROTOCOL = "official_fold0_frozen_head_shift_pruning_empirical_package_v1"
RETENTIONS = (25, 50, 75)
DELETIONS = (10, 25, 50)
METHODS = ("Random", "Random+Rescale", "Magnitude", "Gradient", "Utility")
SEEDS = (1337, 2027, 31415)


@dataclass(frozen=True)
class Condition:
    key: str
    method: str
    keep: torch.Tensor | None
    scale: float = 1.0
    seed: int | None = None


def load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        for key in ("ranking", "tensor", "utility", "features"):
            if isinstance(value.get(key), torch.Tensor):
                value = value[key]
                break
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path}: expected a tensor")
    return value


def validate_ranking(value: torch.Tensor, dimension: int, label: str) -> torch.Tensor:
    if value.ndim != 1 or value.numel() != dimension:
        raise ValueError(f"{label}: expected a ranking of {dimension} coordinates")
    value = value.to(dtype=torch.long, device="cpu")
    if (
        torch.unique(value).numel() != dimension
        or int(value.min()) < 0
        or int(value.max()) >= dimension
    ):
        raise ValueError(f"{label}: expected a complete coordinate permutation")
    return value


def save_tensor(value: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value.detach().cpu(), temporary)
    temporary.replace(path)


def decode(tokenizer: Any, logits: torch.Tensor, length: int) -> str:
    ids = logits.argmax(dim=-1) if logits.ndim == 2 else logits
    return str(tokenizer.decode(ids[:length].tolist(), group_tokens=True)).lower().strip()


def make_loader(
    manifest: Path,
    cache_root: Path,
    tokenizer: Any,
    split: str,
    hidden_dim: int,
    batch_size: int,
    num_workers: int,
    load_eft: bool,
) -> DataLoader:
    dataset = ShiftFeatureDataset(
        manifest,
        cache_root,
        tokenizer,
        feature_split=split,
        expected_dim=hidden_dim,
        load_eft=load_eft,
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


def compose(
    e0: torch.Tensor,
    delta: torch.Tensor,
    keep: torch.Tensor | None,
    scale: float = 1.0,
) -> torch.Tensor:
    if keep is None:
        return e0 + delta
    selected = torch.zeros_like(delta)
    if keep.numel():
        indices = keep.to(device=e0.device, dtype=torch.long)
        selected[..., indices] = delta[..., indices] * float(scale)
    return e0 + selected


def compute_rankings(
    head: torch.nn.Module,
    tokenizer: Any,
    manifest: Path,
    cache_root: Path,
    hidden_dim: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    log_every: int,
) -> dict[str, Any]:
    loader = make_loader(
        manifest, cache_root, tokenizer, "train_utility", hidden_dim,
        batch_size, num_workers, load_eft=False,
    )
    gradient_sum = torch.zeros(hidden_dim, dtype=torch.float64)
    utility_sum = torch.zeros(hidden_dim, dtype=torch.float64)
    magnitude_sum = torch.zeros(hidden_dim, dtype=torch.float64)
    frames = processed = 0
    total_loss = 0.0
    for batch_index, batch in enumerate(loader, start=1):
        e0 = batch["e0"].to(device, non_blocking=True)
        delta = batch["delta"].to(device, non_blocking=True)
        with torch.enable_grad():
            gradient, utility, batch_frames, batch_loss = ctc_gradient_and_taylor_batch_sums(
                head,
                e0,
                delta,
                batch["targets"].to(device, non_blocking=True),
                batch["feature_lengths"].to(device),
                batch["target_lengths"].to(device),
                int(tokenizer.pad_token_id),
            )
        valid = (
            torch.arange(e0.shape[1], device=device).unsqueeze(0)
            < batch["feature_lengths"].to(device).unsqueeze(1)
        )
        gradient_sum += gradient.double().cpu()
        utility_sum += utility.double().cpu()
        magnitude_sum += delta.abs().masked_fill(~valid.unsqueeze(-1), 0.0).sum((0, 1)).double().cpu()
        frames += batch_frames
        count = len(batch["utt_ids"])
        processed += count
        total_loss += float(batch_loss.cpu()) * count
        if batch_index == 1 or processed % log_every < count or processed == len(loader.dataset):
            print({"attribution_processed": processed, "attribution_total": len(loader.dataset)}, flush=True)
    if frames < 1:
        raise RuntimeError("attribution pass produced no valid frames")
    values = {
        "gradient": gradient_sum / frames,
        "utility": utility_sum / frames,
        "magnitude": magnitude_sum / frames,
    }
    return {
        "num_utterances": processed,
        "num_valid_frames": frames,
        "mean_ctc_loss": total_loss / max(processed, 1),
        "values": values,
        "rankings": {
            name: torch.argsort(value, descending=True, stable=True)
            for name, value in values.items()
        },
    }


def permutations(dimension: int) -> dict[int, torch.Tensor]:
    result: dict[int, torch.Tensor] = {}
    for seed in SEEDS:
        result[seed] = torch.randperm(
            dimension, generator=torch.Generator().manual_seed(seed)
        )
    return result


def retention_conditions(
    dimension: int,
    rankings: dict[str, torch.Tensor],
    random_orders: dict[int, torch.Tensor],
) -> list[Condition]:
    conditions: list[Condition] = []
    for retention in RETENTIONS:
        count = max(1, round(dimension * retention / 100))
        for method, name in (("Magnitude", "magnitude"), ("Gradient", "gradient"), ("Utility", "utility")):
            conditions.append(Condition(f"{name}_{retention}", method, rankings[name][:count]))
        for seed, order in random_orders.items():
            keep = order[:count]
            conditions.append(Condition(f"random_{retention}_{seed}", "Random", keep, seed=seed))
            conditions.append(Condition(
                f"random_rescale_{retention}_{seed}", "Random+Rescale", keep,
                scale=dimension / count, seed=seed,
            ))
    return conditions


def deletion_conditions(
    dimension: int,
    utility: torch.Tensor,
    random_orders: dict[int, torch.Tensor],
) -> list[Condition]:
    conditions: list[Condition] = []
    for deleted in DELETIONS:
        count = max(1, round(dimension * deleted / 100))
        conditions.extend([
            Condition(f"drop_best_{deleted}", "DropBest", utility[count:]),
            Condition(
                f"drop_worst_{deleted}", "DropWorst",
                utility[:-count] if count < dimension else utility[:0],
            ),
        ])
        for seed, order in random_orders.items():
            conditions.append(Condition(
                f"random_delete_{deleted}_{seed}", "Random", order[count:], seed=seed
            ))
    return conditions


def evaluate_split(
    head: torch.nn.Module,
    tokenizer: Any,
    manifest: Path,
    cache_root: Path,
    split: str,
    hidden_dim: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    conditions: list[Condition],
) -> dict[str, Any]:
    loader = make_loader(
        manifest, cache_root, tokenizer, split, hidden_dim,
        batch_size, num_workers, load_eft=True,
    )
    all_conditions = [
        Condition("full", "Full", None),
        Condition("no_shift", "No shift", torch.empty(0, dtype=torch.long)),
        *conditions,
    ]
    hypotheses = {condition.key: [] for condition in all_conditions}
    references: list[str] = []
    max_feature_error = max_logit_error = 0.0
    with torch.inference_mode():
        for batch in loader:
            e0 = batch["e0"].to(device, non_blocking=True)
            eft = batch["eft"].to(device, non_blocking=True)
            delta = batch["delta"].to(device, non_blocking=True)
            references.extend(batch["transcripts"])
            max_feature_error = max(max_feature_error, float((e0 + delta - eft).abs().max().cpu()))
            eft_logits = head(eft)
            full_logits = head(e0 + delta)
            max_logit_error = max(max_logit_error, float((eft_logits - full_logits).abs().max().cpu()))
            for condition in all_conditions:
                logits = head(compose(e0, delta, condition.keep, condition.scale))
                for row, length in zip(logits.cpu(), batch["feature_lengths"], strict=True):
                    hypotheses[condition.key].append(decode(tokenizer, row, int(length)))
    measured = {
        key: {"wer": float(word_error_rate(references, values)), "examples": len(references)}
        for key, values in hypotheses.items()
    }
    return {
        "examples": len(references),
        "identity": {
            "max_abs_error": max_feature_error,
            "max_logit_abs_error": max_logit_error,
            "pass": max_feature_error <= 1e-5 and max_logit_error <= 1e-4,
        },
        "measured": measured,
    }


def metric(wer: float, full: float) -> dict[str, float]:
    return {
        "wer": wer,
        "wer_percent": wer * 100,
        "delta_vs_full": wer - full,
        "delta_vs_full_percentage_points": (wer - full) * 100,
    }


def aggregate(values: list[tuple[int, float]], full: float) -> dict[str, Any]:
    wers = [value for _, value in values]
    deviation = stdev(wers) if len(wers) > 1 else 0.0
    return {
        **metric(mean(wers), full),
        "std_wer": deviation,
        "std_wer_percentage_points": deviation * 100,
        "seed_results": [{"seed": seed, **metric(wer, full)} for seed, wer in values],
    }


def retention_summary(measured: dict[str, dict[str, Any]]) -> dict[str, Any]:
    full = float(measured["full"]["wer"])
    no_shift = float(measured["no_shift"]["wer"])
    result: dict[str, Any] = {
        "full": metric(full, full),
        "no_shift": metric(no_shift, full),
        "methods": {},
    }
    for method in METHODS:
        rows: dict[str, Any] = {"0": metric(no_shift, full), "100": metric(full, full)}
        for retention in RETENTIONS:
            if method == "Random":
                values = [(seed, float(measured[f"random_{retention}_{seed}"]["wer"])) for seed in SEEDS]
                rows[str(retention)] = aggregate(values, full)
            elif method == "Random+Rescale":
                values = [(seed, float(measured[f"random_rescale_{retention}_{seed}"]["wer"])) for seed in SEEDS]
                rows[str(retention)] = aggregate(values, full)
            else:
                key = f"{method.lower()}_{retention}"
                rows[str(retention)] = metric(float(measured[key]["wer"]), full)
        result["methods"][method] = rows
    return result


def deletion_summary(measured: dict[str, dict[str, Any]]) -> dict[str, Any]:
    full = float(measured["full"]["wer"])
    result: dict[str, Any] = {"full": metric(full, full), "methods": {}}
    for method in ("DropWorst", "Random", "DropBest"):
        rows: dict[str, Any] = {}
        for deleted in DELETIONS:
            if method == "Random":
                values = [(seed, float(measured[f"random_delete_{deleted}_{seed}"]["wer"])) for seed in SEEDS]
                rows[str(deleted)] = aggregate(values, full)
            else:
                key = f"drop_{method[4:].lower()}_{deleted}"
                rows[str(deleted)] = metric(float(measured[key]["wer"]), full)
        result["methods"][method] = rows
    return result


def format_metric(value: dict[str, Any]) -> str:
    if "std_wer_percentage_points" in value:
        return f"{value['wer_percent']:.3f}% ± {value['std_wer_percentage_points']:.3f}"
    return f"{value['wer_percent']:.3f}%"


def write_summary(result: dict[str, Any], path: Path) -> None:
    test = result["splits"]["test"]
    retention = test["retention"]
    deletion = test["deletion"]
    lines = [
        "# Official Fold0 frozen-head shift-pruning empirical package",
        "",
        "All conditions use the same cached E0/Eft/Delta streams and original frozen oracle CTC head.",
        "No head retraining or healing is performed.",
        "",
        "## Retained shift coordinates",
        "",
        "| Retained | Random | Random+Rescale | Magnitude | Gradient | Utility |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for retained in (0, 25, 50, 75, 100):
        lines.append(
            f"| {retained}% | "
            + " | ".join(format_metric(retention["methods"][method][str(retained)]) for method in METHODS)
            + " |"
        )
    lines += [
        "",
        "Random and Random+Rescale are mean ± sample standard deviation over seeds "
        + ", ".join(str(seed) for seed in SEEDS) + ".",
        "",
        "## Deleted utility coordinates",
        "",
        "| Deleted | DropWorst | Random | DropBest |",
        "|---:|---:|---:|---:|",
    ]
    for deleted in DELETIONS:
        lines.append(
            f"| {deleted}% | "
            f"{format_metric(deletion['methods']['DropWorst'][str(deleted)])} | "
            f"{format_metric(deletion['methods']['Random'][str(deleted)])} | "
            f"{format_metric(deletion['methods']['DropBest'][str(deleted)])} |"
        )
    lines += [
        "",
        f"Full-shift test WER: {format_metric(retention['full'])}. "
        f"No-shift test WER: {format_metric(retention['no_shift'])}.",
        "",
        "Identity gate: " + ("PASS" if test["identity"]["pass"] else "FAIL")
        + f" (max feature error {test['identity']['max_abs_error']:.3g}; "
        f"max logit error {test['identity']['max_logit_abs_error']:.3g}).",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1 or args.num_workers < 0 or args.log_every < 1:
        raise ValueError("batch-size must be positive; num-workers non-negative; log-every positive")
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is non-empty; pass --overwrite to reuse it")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in ("train_utility", "dev", "test"):
        if not (args.manifest_root / f"{name}.jsonl").is_file():
            raise FileNotFoundError(args.manifest_root / f"{name}.jsonl")
    utility_path = args.utility_ranking or args.utility_dir / "utility_shift_taylor_ranking.pt"
    if not utility_path.is_file():
        raise FileNotFoundError(utility_path)

    processor = make_processor(str(args.checkpoint), None)
    model = load_ctc_model(args.checkpoint).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hidden_dim = int(getattr(model.config, "hidden_size", 0))
    if hidden_dim < 1 or not hasattr(model, "lm_head"):
        raise ValueError("checkpoint must provide a positive hidden_size and lm_head")
    supplied_utility = validate_ranking(load_tensor(utility_path), hidden_dim, "utility ranking")
    ranking_dir = args.output_dir / "rankings"
    required_ranking_files = {
        name: ranking_dir / f"{name}_ranking.pt"
        for name in ("gradient", "magnitude", "utility")
    }
    if args.reuse_rankings and all(path.is_file() for path in required_ranking_files.values()):
        rankings = {
            name: validate_ranking(load_tensor(path), hidden_dim, f"saved {name} ranking")
            for name, path in required_ranking_files.items()
        }
        if not torch.equal(rankings["utility"], supplied_utility):
            raise RuntimeError("saved Taylor utility ranking differs from --utility-ranking")
        attribution = {
            "num_utterances": None,
            "num_valid_frames": None,
            "mean_ctc_loss": None,
            "values": {
                name: load_tensor(ranking_dir / f"{name}_scores.pt")
                for name in rankings
            },
        }
        print({"attribution": "reused_saved_rankings", "ranking_dir": str(ranking_dir)}, flush=True)
    else:
        attribution = compute_rankings(
            model.lm_head, processor.tokenizer,
            args.manifest_root / "train_utility.jsonl", args.cache_root,
            hidden_dim, args.batch_size, args.num_workers, device, args.log_every,
        )
        rankings = attribution["rankings"]
        if not torch.equal(rankings["utility"], supplied_utility):
            mismatch = int((rankings["utility"] != supplied_utility).sum())
            raise RuntimeError(f"recomputed Taylor utility ranking differs at {mismatch} coordinates")
    for name, ranking in rankings.items():
        save_tensor(attribution["values"][name], ranking_dir / f"{name}_scores.pt")
        save_tensor(ranking, ranking_dir / f"{name}_ranking.pt")
    random_orders = permutations(hidden_dim)
    for seed, order in random_orders.items():
        save_tensor(order, ranking_dir / f"random_seed_{seed}_permutation.pt")

    conditions = retention_conditions(hidden_dim, rankings, random_orders)
    conditions += deletion_conditions(hidden_dim, rankings["utility"], random_orders)
    splits: dict[str, Any] = {}
    for split in ("dev", "test"):
        measured = evaluate_split(
            model.lm_head, processor.tokenizer,
            args.manifest_root / f"{split}.jsonl", args.cache_root, split,
            hidden_dim, args.batch_size, args.num_workers, device, conditions,
        )
        splits[split] = {
            "identity": measured["identity"],
            "retention": retention_summary(measured["measured"]),
            "deletion": deletion_summary(measured["measured"]),
            "raw_measured": measured["measured"],
        }
        print({
            "split": split,
            "examples": measured["examples"],
            "full_wer": measured["measured"]["full"]["wer"],
            "no_shift_wer": measured["measured"]["no_shift"]["wer"],
            "identity_pass": measured["identity"]["pass"],
        }, flush=True)

    result: dict[str, Any] = {
        "protocol": PROTOCOL,
        "checkpoint": str(args.checkpoint),
        "manifest_root": str(args.manifest_root),
        "cache_root": str(args.cache_root),
        "utility_ranking": str(utility_path),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "hidden_dim": hidden_dim,
        "random_seeds": list(SEEDS),
        "retentions_percent": [0, *RETENTIONS, 100],
        "deletions_percent": list(DELETIONS),
        "attribution": {
            "num_utterances": attribution["num_utterances"],
            "num_valid_frames": attribution["num_valid_frames"],
            "mean_ctc_loss": attribution["mean_ctc_loss"],
            "utility_ranking_match": True,
            "score_paths": {name: str(ranking_dir / f"{name}_scores.pt") for name in rankings},
            "ranking_paths": {name: str(ranking_dir / f"{name}_ranking.pt") for name in rankings},
        },
        "splits": splits,
    }
    json_path = args.output_dir / "metrics.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(result, args.output_dir / "summary.md")
    print(json.dumps({"metrics": str(json_path), "summary": str(args.output_dir / "summary.md")}, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--utility-dir", type=Path, required=True)
    parser.add_argument("--utility-ranking", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--reuse-rankings", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
