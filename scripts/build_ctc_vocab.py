#!/usr/bin/env python3
"""Build the shared character vocabulary used by both Stage 2 CTC models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from usde.ctc import read_records


def build_vocab(transcripts: list[str]) -> dict[str, int]:
    characters = sorted({character for text in transcripts for character in text if character != " "})
    tokens = ["<pad>", "<unk>", "|", *characters]
    return {token: index for index, token in enumerate(dict.fromkeys(tokens))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-tsv", type=Path, default=Path("data/train.tsv"))
    parser.add_argument("--output-dir", type=Path, default=Path("assets/ctc_vocab"))
    args = parser.parse_args()

    rows = read_records(args.train_tsv)
    vocab = build_vocab([row["transcript"] for row in rows])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "vocab.json").write_text(
        json.dumps(vocab, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "Wav2Vec2CTCTokenizer",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "word_delimiter_token": "|",
                "do_lower_case": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "special_tokens_map.json").write_text(
        json.dumps(
            {"unk_token": "<unk>", "pad_token": "<pad>", "word_delimiter_token": "|"},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"train_tsv": str(args.train_tsv), "utterances": len(rows), "vocab_size": len(vocab)}))


if __name__ == "__main__":
    main()
