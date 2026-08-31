"""Frozen text and character-CTC vocabulary utilities."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path


# Vendored verbatim from the data-preparation repository specified by the
# Mind-the-Shift source. It preserves that project's choice not to normalize
# spelled-out English numbers.
_NORMALIZER_DIR = Path(__file__).resolve().parents[2] / "third_party" / "spapl_kidsasr"
if str(_NORMALIZER_DIR) not in sys.path:
    sys.path.insert(0, str(_NORMALIZER_DIR))
from english_normalizer import EnglishTextNormalizer

BLANK = "<blank>"
UNK = "<unk>"
_NORMALIZER = EnglishTextNormalizer({})


def normalize_text(text: str) -> str:
    """Apply the same English normalizer as the referenced MyST recipe."""
    return _NORMALIZER(text).strip()


def build_vocab(transcripts: Iterable[str]) -> dict[str, int]:
    """Create a deterministic CTC vocabulary; index zero is always blank."""
    chars = sorted({char for text in transcripts for char in normalize_text(text)})
    return {token: index for index, token in enumerate([BLANK, UNK, *chars])}


def encode(text: str, vocab: dict[str, int]) -> list[int]:
    unknown = vocab[UNK]
    return [vocab.get(char, unknown) for char in normalize_text(text)]


def decode(ids: Iterable[int], vocab: dict[str, int]) -> str:
    inverse = {index: token for token, index in vocab.items()}
    blank = vocab[BLANK]
    output: list[str] = []
    previous: int | None = None
    for idx in ids:
        idx = int(idx)
        if idx != blank and idx != previous:
            token = inverse.get(idx, UNK)
            if token != UNK:
                output.append(token)
        previous = idx
    return normalize_text("".join(output))
