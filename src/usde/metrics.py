"""Small dependency-free metrics used by the reproducible experiment scripts."""

from __future__ import annotations


def word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    """Compute corpus WER as total word edit distance divided by reference words."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    errors = 0
    words = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        ref = reference.split()
        hyp = hypothesis.split()
        words += len(ref)
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
        errors += previous[-1]
    return errors / words if words else 0.0
