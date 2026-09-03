from __future__ import annotations

import numpy as np

from scripts.bootstrap_shift_pruning import edit_distance, paired_bootstrap, utterance_counts


def test_edit_distance_and_utterance_counts_match_corpus_wer() -> None:
    references = ["one two", "one two"]
    hypotheses = ["one two", "one three"]
    errors, words = utterance_counts(references, hypotheses)

    assert errors.tolist() == [0, 1]
    assert words.tolist() == [2, 2]
    assert float(errors.sum() / words.sum()) == 0.25
    assert edit_distance("a b", "a c") == 1


def test_paired_bootstrap_is_zero_for_identical_systems() -> None:
    errors = np.asarray([0, 1, 2], dtype=np.int64)
    words = np.asarray([2, 3, 4], dtype=np.int64)

    samples = paired_bootstrap(errors, errors, words, replicates=100, seed=1337)

    assert np.array_equal(samples, np.zeros(100))
