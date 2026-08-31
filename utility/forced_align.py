"""Small, dependency-free Viterbi forced alignment for CTC emissions.

The alignment is conditioned on the supplied target sequence.  It is not a
greedy decode: the dynamic program searches the valid CTC lattice and returns
one vocabulary id for every input frame.
"""

from __future__ import annotations

from typing import Sequence

import torch


def _as_target_tensor(target_ids: Sequence[int] | torch.Tensor, device: torch.device) -> torch.Tensor:
    target = torch.as_tensor(target_ids, dtype=torch.long, device=device)
    if target.ndim != 1:
        raise ValueError(f"target_ids must be one-dimensional, got shape {tuple(target.shape)}")
    return target


def forced_align_ctc(
    logits: torch.Tensor,
    target_ids: Sequence[int] | torch.Tensor,
    blank_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the best CTC path and its per-frame emission scores.

    Parameters
    ----------
    logits:
        Unnormalized frame emissions with shape ``[T, V]``.
    target_ids:
        Ground-truth CTC labels with shape ``[L]``.  Targets must not contain
        the blank id.
    blank_id:
        Vocabulary id used by the CTC blank symbol.

    Returns
    -------
    alignment, scores:
        Both tensors have shape ``[T]``. ``alignment[t]`` is the aligned
        vocabulary id at frame ``t`` and ``scores[t]`` is its log-softmax
        emission score.  Empty input is supported only with an empty target.

    Notes
    -----
    The expanded CTC state sequence is ``blank, y1, blank, ..., yL, blank``.
    A label may skip the preceding blank only when it differs from the label
    two states earlier; this is the standard repeated-label CTC constraint.
    """

    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [T,V], got {tuple(logits.shape)}")
    frames, vocab_size = logits.shape
    if vocab_size < 2:
        raise ValueError("CTC emissions need at least two vocabulary classes")
    if not 0 <= int(blank_id) < vocab_size:
        raise ValueError(f"blank_id {blank_id} is outside vocabulary size {vocab_size}")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contain non-finite values")

    target = _as_target_tensor(target_ids, logits.device)
    if target.numel() and ((target < 0).any() or (target >= vocab_size).any()):
        raise ValueError("target_ids contain an id outside the vocabulary")
    if target.numel() and (target == blank_id).any():
        raise ValueError("target_ids must not contain the CTC blank id")

    if frames == 0:
        if target.numel():
            raise ValueError("non-empty CTC target cannot align to zero frames")
        empty = torch.empty(0, dtype=torch.long, device=logits.device)
        return empty, logits.new_empty(0)

    # A repeated adjacent target label needs an intervening blank state.
    repeated = int((target[1:] == target[:-1]).sum()) if target.numel() > 1 else 0
    required_frames = int(target.numel()) + repeated
    if frames < required_frames:
        raise ValueError(
            f"CTC target has {target.numel()} labels and {repeated} repeats, "
            f"but only {frames} frames are available (minimum {required_frames})"
        )

    log_probs = torch.log_softmax(logits, dim=-1)
    expanded = torch.full(
        (2 * target.numel() + 1,),
        int(blank_id),
        dtype=torch.long,
        device=logits.device,
    )
    if target.numel():
        expanded[1::2] = target

    states = int(expanded.numel())
    neg_inf = torch.tensor(float("-inf"), dtype=log_probs.dtype, device=logits.device)
    scores = torch.full((frames, states), neg_inf, dtype=log_probs.dtype, device=logits.device)
    backpointers = torch.full(
        (frames, states), -1, dtype=torch.long, device=logits.device
    )

    scores[0, 0] = log_probs[0, blank_id]
    if target.numel():
        scores[0, 1] = log_probs[0, int(target[0])]

    state_ids = torch.arange(states, dtype=torch.long, device=logits.device)
    predecessor = torch.full((3, states), -1, dtype=torch.long, device=logits.device)
    predecessor[0] = state_ids  # stay in the same CTC state
    predecessor[1, 1:] = state_ids[:-1]  # consume the next expanded state
    predecessor[2, 2:] = state_ids[:-2]  # skip blank before a new label
    can_skip = torch.zeros(states, dtype=torch.bool, device=logits.device)
    if states > 2:
        can_skip[2:] = (expanded[2:] != blank_id) & (expanded[2:] != expanded[:-2])

    for frame in range(1, frames):
        previous = scores[frame - 1]
        step = torch.full_like(previous, neg_inf)
        step[1:] = previous[:-1]
        skip = torch.full_like(previous, neg_inf)
        skip[2:] = previous[:-2]
        skip = skip.masked_fill(~can_skip, neg_inf)
        candidates = torch.stack((previous, step, skip), dim=0)
        best, transition = candidates.max(dim=0)
        scores[frame] = best + log_probs[frame, expanded]
        backpointers[frame] = predecessor.gather(0, transition.unsqueeze(0)).squeeze(0)

    if target.numel():
        final_candidates = scores[-1, -2:]
        final_offset = int(final_candidates.argmax())
        end_state = states - 2 + final_offset
    else:
        end_state = 0
    if not torch.isfinite(scores[-1, end_state]):
        raise ValueError("CTC Viterbi alignment has no finite path")

    state_path = torch.empty(frames, dtype=torch.long, device=logits.device)
    state_path[-1] = end_state
    for frame in range(frames - 1, 0, -1):
        previous_state = backpointers[frame, state_path[frame]]
        if previous_state < 0:
            raise ValueError("CTC Viterbi backtrace failed")
        state_path[frame - 1] = previous_state

    alignment = expanded[state_path]
    frame_ids = torch.arange(frames, dtype=torch.long, device=logits.device)
    frame_scores = log_probs[frame_ids, alignment]
    return alignment, frame_scores


__all__ = ["forced_align_ctc"]
