#!/usr/bin/env python3
"""Compare L2 and CMU CTC-Taylor utility vectors for Stage 7C."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utility.compute_ctc_taylor_utility import spearman, top_k_overlap  # noqa: E402


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise ValueError(f"{path}: expected a one-dimensional tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{path}: utility contains non-finite values")
    return value.detach().double()


def _load_ranking(path: Path, dimension: int) -> torch.Tensor:
    ranking = _load_tensor(path).long()
    expected = torch.arange(dimension, dtype=torch.long)
    if ranking.shape != expected.shape or not torch.equal(torch.sort(ranking).values, expected):
        raise ValueError(f"{path}: expected a complete ranking permutation of 0..{dimension - 1}")
    return ranking


def compare(
    l2_utility_path: Path,
    cmu_utility_path: Path,
    l2_ranking_path: Path,
    cmu_ranking_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    l2 = _load_tensor(l2_utility_path)
    cmu = _load_tensor(cmu_utility_path)
    if l2.shape != cmu.shape:
        raise ValueError(f"utility shape mismatch: L2={tuple(l2.shape)}, CMU={tuple(cmu.shape)}")
    dimension = l2.numel()
    l2_ranking = _load_ranking(l2_ranking_path, dimension)
    cmu_ranking = _load_ranking(cmu_ranking_path, dimension)
    result: dict[str, Any] = {
        "protocol": "stage7c_cmu_matched_text_control_v1",
        "utility_definition": "E_frame,utterance[abs(Delta * dL_CTC/dDelta)]",
        "comparison": "same L2-adapted FullDelta CTC head on matched L2 and CMU text",
        "delta_definition": "E_ft^L2(x) - E_pt(x), computed separately for each input waveform",
        "dimension": dimension,
        "l2_utility_path": str(l2_utility_path),
        "cmu_utility_path": str(cmu_utility_path),
        "l2_ranking_path": str(l2_ranking_path),
        "cmu_ranking_path": str(cmu_ranking_path),
        "spearman_utility_l2_cmu": float(spearman(l2.numpy(), cmu.numpy())),
        "top_k_overlap": {
            "256": top_k_overlap(l2, cmu, 256),
            "512": top_k_overlap(l2, cmu, 512),
        },
        "top_utility_v4_l2": l2_ranking[:10].tolist(),
        "top_utility_v4_cmu": cmu_ranking[:10].tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l2-utility", type=Path, default=Path("results/stage7c/utility_v4_l2.pt"))
    parser.add_argument("--cmu-utility", type=Path, default=Path("results/stage7c/utility_v4_cmu.pt"))
    parser.add_argument("--l2-ranking", type=Path, default=Path("results/stage7c/utility_v4_l2_ranking.pt"))
    parser.add_argument("--cmu-ranking", type=Path, default=Path("results/stage7c/utility_v4_cmu_ranking.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/stage7c/matched_text_control.json"))
    args = parser.parse_args()
    compare(args.l2_utility, args.cmu_utility, args.l2_ranking, args.cmu_ranking, args.output)


if __name__ == "__main__":
    main()
