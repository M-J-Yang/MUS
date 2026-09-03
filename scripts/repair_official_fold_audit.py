#!/usr/bin/env python3
"""Repair fold-specific provenance in an official L2-ARCTIC audit JSON.

The original materializer was written for Fold0 and leaves its source prefix
and protocol label unchanged when reused for another released fold. This small
post-processing command preserves all measured comparison fields while
recomputing the fold-specific source paths and SHA256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repair(fold: int, source_root: Path, audit_path: Path) -> dict[str, Any]:
    if fold < 0:
        raise ValueError("fold must be non-negative")
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    with audit_path.open(encoding="utf-8") as handle:
        audit = json.load(handle)
    if not isinstance(audit, dict):
        raise ValueError(f"{audit_path}: expected a JSON object")
    files = {split: source_root / f"{split}.csv" for split in ("train", "val", "test")}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing official source files: {missing}")
    official_source = audit.setdefault("official_source", {})
    official_source.update(
        {
            "repository": "https://github.com/thaivanphat95/robust-atc-asr",
            "prefix": f"files/Arctic/8fold/{fold}",
            "files": {split: str(path) for split, path in files.items()},
            "sha256": {split: sha256_file(path) for split, path in files.items()},
        }
    )
    audit["protocol"] = f"official_robust_atc_asr_arctic_8fold_{fold}_vs_local_ut8_fold{fold}"
    temporary = audit_path.with_name(audit_path.name + ".tmp")
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(audit_path)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    result = repair(args.fold, args.source_root, args.audit)
    print(json.dumps({"protocol": result["protocol"], "official_source": result["official_source"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
