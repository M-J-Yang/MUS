#!/usr/bin/env python3
"""Train a matched-budget linear CTC head on selected Delta coordinates."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from usde.stage4 import (  # noqa: E402
    BASE_DIM,
    CachedFeatureDataset,
    blank_id,
    collate,
    greedy_wer,
    load_vocab,
)
from usde.stage6 import (  # noqa: E402
    SELECTIONS,
    SelectedDeltaLinearCTC,
    get_selected_indices,
)


PROTOCOL = "stage6_selected_delta_linear_ctc_v1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _save_checkpoint(
    model: SelectedDeltaLinearCTC,
    path: Path,
    payload: dict[str, Any],
) -> None:
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    checkpoint = dict(payload)
    checkpoint["model"] = state
    checkpoint["classifier_weight"] = state["linear.weight"]
    checkpoint["classifier_bias"] = state["linear.bias"]
    _atomic_torch_save(checkpoint, path)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"{path}: invalid Stage 6 checkpoint")
    return payload


def _load_ranking(path: Path, name: str) -> torch.Tensor:
    try:
        ranking = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        ranking = torch.load(path, map_location="cpu")
    if isinstance(ranking, dict):
        ranking = ranking.get("ranking", ranking.get(f"{name}_ranking"))
    if not isinstance(ranking, torch.Tensor):
        raise ValueError(f"{path}: expected a tensor ranking")
    return ranking


def _default_train_manifest() -> Path:
    teacher = Path("manifests/arctic_step2/l2/train_teacher.jsonl")
    return teacher if teacher.is_file() else Path("manifests/arctic_step2/l2/train.jsonl")


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return Path("results/stage6") / args.selection / f"k{args.k}"


def _make_dataset(
    manifest: Path,
    feature_root: Path,
    vocab: dict[str, int],
    feature_split: str,
    reference_dim: int,
    delta_dim: int | None,
    max_examples: int | None,
    seed: int,
) -> CachedFeatureDataset:
    return CachedFeatureDataset(
        manifest,
        feature_root,
        "full_delta",
        vocab,
        feature_split=feature_split,
        expected_dim=reference_dim,
        expected_auxiliary_dim=delta_dim,
        allow_auxiliary_dim_mismatch=delta_dim is None,
        max_examples=max_examples,
        sample_seed=seed,
    )


def _resolve_delta_dim(
    datasets: list[CachedFeatureDataset],
    reference_dim: int,
    requested_delta_dim: int | None,
) -> int:
    if not datasets or len(datasets[0]) == 0:
        raise ValueError("the training manifest must contain at least one example")
    observed = int(datasets[0][0]["features"].shape[-1]) - reference_dim
    if observed < 1:
        raise ValueError(
            f"expected cached [reference; delta] features, got dimension "
            f"{datasets[0][0]['features'].shape[-1]}"
        )
    if requested_delta_dim is not None and requested_delta_dim != observed:
        raise ValueError(
            f"--delta-dim={requested_delta_dim} does not match cached Delta dimension {observed}"
        )
    for dataset in datasets:
        dataset.expected_auxiliary_dim = observed
    return observed


def _build_selection(
    args: argparse.Namespace,
    delta_dim: int,
    output_dir: Path,
) -> tuple[torch.Tensor, Path | None]:
    ranking_path: Path | None = None
    if args.selection in {"utility", "magnitude"}:
        ranking_path = args.ranking or (
            Path("results/stage5") / f"{args.selection}_ranking.pt"
        )
        ranking = _load_ranking(ranking_path, args.selection)
        if args.selection == "utility":
            selected = get_selected_indices(
                "utility",
                args.k,
                utility_ranking=ranking,
                delta_dim=delta_dim,
            )
        else:
            selected = get_selected_indices(
                "magnitude",
                args.k,
                magnitude_ranking=ranking,
                delta_dim=delta_dim,
            )
    else:
        # Materialize one full permutation and slice it. Re-running this for
        # K=256 and K=512 with the same seed gives nested Random-K selections.
        random_ranking = get_selected_indices(
            "random",
            delta_dim,
            delta_dim=delta_dim,
            seed=args.selection_seed,
        )
        _atomic_torch_save(random_ranking, output_dir / "random_ranking.pt")
        # The default results/stage6/<selection>/k<K> layout has one shared
        # artifact so Random-256 and Random-512 visibly use one permutation.
        if output_dir.parent.name == "random":
            root_ranking_path = (
                output_dir.parent.parent
                / f"random_ranking_seed{args.selection_seed}.pt"
            )
            _atomic_torch_save(random_ranking, root_ranking_path)
        ranking_path = output_dir / "random_ranking.pt"
        selected = random_ranking[: args.k]
    _atomic_torch_save(selected, output_dir / "selected_indices.pt")
    return selected, ranking_path


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError(
            "epochs/batch-size/learning-rate must be positive and weight-decay non-negative"
        )
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")

    output_dir = _resolve_output_dir(args)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite_output_dir:
        raise FileExistsError(
            f"{output_dir} is non-empty; pass --overwrite-output-dir to reuse it"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    vocab = load_vocab(args.vocab)
    train_manifest = args.train_manifest or _default_train_manifest()

    datasets: list[CachedFeatureDataset] = []
    train_dataset = _make_dataset(
        train_manifest,
        args.feature_root,
        vocab,
        args.train_feature_split,
        args.reference_dim,
        args.delta_dim,
        args.max_train_examples,
        args.seed,
    )
    datasets.append(train_dataset)
    dev_dataset = _make_dataset(
        args.dev_manifest,
        args.feature_root,
        vocab,
        args.dev_feature_split,
        args.reference_dim,
        args.delta_dim,
        args.max_dev_examples,
        args.seed,
    )
    datasets.append(dev_dataset)
    test_dataset = None
    if args.test_manifest is not None:
        test_dataset = _make_dataset(
            args.test_manifest,
            args.feature_root,
            vocab,
            args.test_feature_split,
            args.reference_dim,
            args.delta_dim,
            args.max_test_examples,
            args.seed,
        )
        datasets.append(test_dataset)

    delta_dim = _resolve_delta_dim(datasets, args.reference_dim, args.delta_dim)
    selected_indices, ranking_path = _build_selection(args, delta_dim, output_dir)

    model = SelectedDeltaLinearCTC(
        args.reference_dim,
        delta_dim,
        selected_indices,
        len(vocab),
    ).to(device)
    input_dim = args.reference_dim + int(selected_indices.numel())

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "collate_fn": collate,
        "pin_memory": device.type == "cuda",
        "num_workers": args.num_workers,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=train_generator,
        **loader_kwargs,
    )
    dev_loader = DataLoader(dev_dataset, shuffle=False, **loader_kwargs)
    test_loader = (
        DataLoader(test_dataset, shuffle=False, **loader_kwargs)
        if test_dataset is not None
        else None
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.CTCLoss(blank=blank_id(vocab), zero_infinity=True)
    best_wer = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    checkpoint_payload: dict[str, Any] = {
        "protocol": PROTOCOL,
        "selection": args.selection,
        "k": int(selected_indices.numel()),
        "selection_seed": args.selection_seed,
        "ranking_path": None if ranking_path is None else str(ranking_path),
        "selected_indices": selected_indices.clone(),
        "feature_root": str(args.feature_root),
        "reference_dim": args.reference_dim,
        "delta_dim": delta_dim,
        "input_dim": input_dim,
        "vocab": vocab,
        "blank_id": blank_id(vocab),
        "source_compatible": False,
        "layer": args.layer,
        "seed": args.seed,
        "train_manifest": str(train_manifest),
        "dev_manifest": str(args.dev_manifest),
        "test_manifest": None if args.test_manifest is None else str(args.test_manifest),
        "train_feature_split": args.train_feature_split,
        "dev_feature_split": args.dev_feature_split,
        "test_feature_split": args.test_feature_split,
    }
    config = dict(vars(args))
    config.update(
        {
            "resolved_output_dir": output_dir,
            "resolved_train_manifest": train_manifest,
            "resolved_ranking_path": ranking_path,
            "device": str(device),
            "delta_dim": delta_dim,
            "input_dim": input_dim,
            "vocab_size": len(vocab),
            "selected_indices": selected_indices,
        }
    )
    _save_json(output_dir / "config.json", config)
    _save_json(
        output_dir / "sanity.json",
        {
            "reference_shape": [args.reference_dim],
            "delta_shape": [delta_dim],
            "selected_indices_shape": [int(selected_indices.numel())],
            "selected_count": int(selected_indices.numel()),
            "input_shape": [input_dim],
            "indices_unique": int(torch.unique(selected_indices).numel())
            == int(selected_indices.numel()),
            "indices_min": int(selected_indices.min()),
            "indices_max": int(selected_indices.max()),
        },
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            cached_features = batch["features"].to(device, non_blocking=True)
            logits = model(cached_features)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = criterion(
                log_probs,
                batch["targets"].to(device, non_blocking=True),
                batch["feature_lengths"],
                batch["target_lengths"],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"epoch {epoch}: CTC loss became non-finite")
            loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.clip_grad_norm,
                )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        dev_wer = greedy_wer(model, dev_loader, vocab, device)
        result = {
            "epoch": epoch,
            "train_ctc_loss": sum(losses) / max(len(losses), 1),
            "dev_wer": dev_wer,
            "train_examples": len(train_dataset),
            "dev_examples": len(dev_dataset),
        }
        history.append(result)
        if dev_wer < best_wer:
            best_wer = dev_wer
            best_epoch = epoch
            _save_checkpoint(
                model,
                output_dir / "best.pt",
                dict(
                    checkpoint_payload,
                    epoch=epoch,
                    best_dev_wer=dev_wer,
                ),
            )
        metrics = {
            **checkpoint_payload,
            "selected_indices": selected_indices,
            "best_dev_wer": best_wer,
            "best_epoch": best_epoch,
            "history": history,
            "test_wer": None,
        }
        _save_json(output_dir / "metrics.json", metrics)
        print(json.dumps(_jsonable(result), sort_keys=True), flush=True)

    if best_epoch == 0:
        raise RuntimeError("training completed without saving a best checkpoint")
    best_checkpoint = _load_checkpoint(output_dir / "best.pt")
    model.load_state_dict(best_checkpoint["model"])
    test_wer = (
        greedy_wer(model, test_loader, vocab, device)
        if test_loader is not None
        else None
    )
    metrics = {
        **checkpoint_payload,
        "selected_indices": selected_indices,
        "best_dev_wer": best_wer,
        "best_epoch": best_epoch,
        "history": history,
        "test_wer": test_wer,
        "test_examples": None if test_dataset is None else len(test_dataset),
    }
    _save_json(output_dir / "metrics.json", metrics)
    print(
        json.dumps(
            {
                "selection": args.selection,
                "k": int(selected_indices.numel()),
                "best_dev_wer": best_wer,
                "test_wer": test_wer,
                "best_epoch": best_epoch,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", choices=SELECTIONS, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument(
        "--ranking",
        type=Path,
        default=None,
        help="frozen Stage 5 ranking; defaults to results/stage5/<selection>_ranking.pt",
    )
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--feature-root", type=Path, default=Path("artifacts/features/stage3_l2"))
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=None,
        help="defaults to train_teacher.jsonl when present",
    )
    parser.add_argument(
        "--dev-manifest",
        type=Path,
        default=Path("manifests/arctic_step2/l2/dev.jsonl"),
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=Path("manifests/arctic_step2/l2/test.jsonl"),
    )
    parser.add_argument("--train-feature-split", default="train")
    parser.add_argument("--dev-feature-split", default="dev")
    parser.add_argument("--test-feature-split", default="test")
    parser.add_argument("--vocab", type=Path, default=Path("assets/ctc_vocab/vocab.json"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument(
        "--reference-dim",
        "--expected-dim",
        dest="reference_dim",
        type=int,
        default=BASE_DIM,
    )
    parser.add_argument("--delta-dim", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-dev-examples", type=int, default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    train(args)


if __name__ == "__main__":
    main()
