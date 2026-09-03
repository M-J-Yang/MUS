#!/usr/bin/env python3
"""Materialize a deterministic magnitude ranking from Taylor-utility stats.

The utility pass already accumulates mean absolute shift magnitude.  This
small helper turns that saved vector into the ranking needed by the frozen
core evaluator without running any additional attribution or exploratory
condition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.stats.read_text(encoding="utf-8"))
    magnitude = torch.tensor(payload["magnitude"], dtype=torch.float64)
    ranking = torch.argsort(magnitude, descending=True, stable=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    torch.save(ranking.cpu(), temporary)
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "hidden_dim": int(ranking.numel())}, sort_keys=True))


if __name__ == "__main__":
    main()
