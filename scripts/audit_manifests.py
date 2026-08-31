#!/usr/bin/env python3
"""Validate a frozen MyST manifest condition and make its audit artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from usde.manifest import audit_condition, load_jsonl
from usde.text import build_vocab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, default=Path("manifests"))
    parser.add_argument("--condition", required=True, choices=("full", "10h", "5h", "1h"))
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--vocab-out", type=Path, required=True)
    parser.add_argument("--skip-audio-validation", action="store_true")
    args = parser.parse_args()

    audit = audit_condition(args.manifest_root, args.condition, not args.skip_audio_validation)
    train_rows = load_jsonl(args.manifest_root / args.condition / "train.jsonl")
    vocab = build_vocab(row["transcript"] for row in train_rows)
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.vocab_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.vocab_out.write_text(json.dumps(vocab, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(args.audit_out), "vocab_size": len(vocab)}, sort_keys=True))


if __name__ == "__main__":
    main()
