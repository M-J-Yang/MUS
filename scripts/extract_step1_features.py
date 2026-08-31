#!/usr/bin/env python3
"""Extract final-layer WavLM and W2V2 delta features for one frozen split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from usde.features import extract_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wavlm-ft", required=True)
    parser.add_argument("--w2v2-ft", required=True)
    parser.add_argument("--w2v2-pt", default="facebook/wav2vec2-large-lv60")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-utterances", type=int, default=None)
    args = parser.parse_args()
    report = extract_split(
        args.manifest, args.output_dir, args.wavlm_ft, args.w2v2_ft,
        args.w2v2_pt, args.layer, args.device, args.skip_existing, args.start_index, args.max_utterances,
    )
    (args.output_dir / "extraction_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
