#!/usr/bin/env python3
"""Compute CTC decision-aligned Delta utility on a held-out utility split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from usde.fusion import ConcatLinearCTC, FrozenFusionDataset, collate, load_vocab
from usde.text import BLANK


def viterbi_ctc_alignment(log_probs: torch.Tensor, target: list[int], blank: int) -> list[int]:
    """Return one vocabulary ID per frame under the best CTC path."""
    if not target:
        return [blank] * log_probs.shape[0]
    emissions = [blank]
    for token in target:
        emissions.extend((token, blank))
    steps = len(emissions)
    frames = log_probs.shape[0]
    if frames < len(target):
        raise ValueError(f"CTC target has {len(target)} symbols but only {frames} frames")
    score = torch.full((frames, steps), -torch.inf, device=log_probs.device)
    back = torch.full((frames, steps), -1, dtype=torch.long, device=log_probs.device)
    score[0, 0] = log_probs[0, blank]
    score[0, 1] = log_probs[0, emissions[1]]
    for frame in range(1, frames):
        for state, token in enumerate(emissions):
            candidates = [(score[frame - 1, state], state)]
            if state > 0:
                candidates.append((score[frame - 1, state - 1], state - 1))
            if state > 1 and token != blank and token != emissions[state - 2]:
                candidates.append((score[frame - 1, state - 2], state - 2))
            best_score, best_state = max(candidates, key=lambda item: float(item[0]))
            score[frame, state] = best_score + log_probs[frame, token]
            back[frame, state] = best_state
    end_state = steps - 1
    if steps > 1 and score[-1, steps - 2] > score[-1, steps - 1]:
        end_state -= 1
    if not torch.isfinite(score[-1, end_state]):
        raise ValueError("CTC Viterbi alignment has no finite path")
    states = [end_state]
    for frame in range(frames - 1, 0, -1):
        previous = int(back[frame, states[-1]])
        if previous < 0:
            raise ValueError("CTC Viterbi backtrace failed")
        states.append(previous)
    states.reverse()
    return [emissions[state] for state in states]


def top_indices(values: np.ndarray, count: int) -> list[int]:
    count = min(max(count, 1), values.size)
    return np.argsort(-values, kind="stable")[:count].astype(int).tolist()


def overlap(a: Iterable[int], b: Iterable[int]) -> float:
    a_set, b_set = set(a), set(b)
    return len(a_set & b_set) / max(len(a_set | b_set), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utility-manifest", type=Path, required=True)
    parser.add_argument("--utility-features", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    vocab = load_vocab(args.vocab)
    blank = vocab[BLANK]
    checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu")
    if checkpoint.get("source_compatible", True):
        raise ValueError("utility attribution requires a pure-linear teacher")
    if int(checkpoint.get("layer", args.layer)) != args.layer:
        raise ValueError("teacher checkpoint layer does not match --layer")
    device = torch.device(args.device)
    model = ConcatLinearCTC(len(vocab), source_compatible=False).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    dataset = FrozenFusionDataset(args.utility_manifest, args.utility_features, vocab)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, pin_memory=True)
    dimension = 1024
    sum_q = torch.zeros(dimension, dtype=torch.float64)
    sum_abs_delta = torch.zeros(dimension, dtype=torch.float64)
    sum_abs_delta_aligned = torch.zeros(dimension, dtype=torch.float64)
    sum_sign = torch.zeros(dimension, dtype=torch.float64)
    sum_sign_blank = torch.zeros(dimension, dtype=torch.float64)
    sum_sign_token = torch.zeros(dimension, dtype=torch.float64)
    positive = torch.zeros(dimension, dtype=torch.long)
    negative = torch.zeros(dimension, dtype=torch.long)
    frames_total = aligned_frames = blank_frames = token_frames = 0
    for batch in loader:
        with torch.inference_mode():
            features = batch["features"].to(device)
            logits = model(features)
            log_probs = torch.log_softmax(logits, dim=-1)
        target_offset = 0
        for row, (feature_length, target_length) in enumerate(zip(batch["feature_lengths"], batch["target_lengths"], strict=True)):
            frames = int(feature_length)
            target_count = int(target_length)
            target = batch["targets"][target_offset : target_offset + target_count].tolist()
            target_offset += target_count
            delta = features[row, :frames, 1024:]
            sum_abs_delta += delta.abs().double().sum(dim=0).cpu()
            frames_total += frames
            alignment = viterbi_ctc_alignment(log_probs[row, :frames], target, blank)
            w_delta = model.linear.weight[:, 1024:]
            for frame, aligned_target in enumerate(alignment):
                if aligned_target == blank:
                    continue
                competitor_logits = logits[row, frame].clone()
                competitor_logits[aligned_target] = -torch.inf
                competitor = int(competitor_logits.argmax())
                contribution = delta[frame] * (w_delta[aligned_target] - w_delta[competitor])
                contribution_cpu = contribution.double().cpu()
                signs = torch.sign(contribution_cpu)
                sum_q += contribution_cpu
                sum_abs_delta_aligned += delta[frame].abs().double().cpu()
                sum_sign += signs
                aligned_frames += 1
                if competitor == blank:
                    sum_sign_blank += signs
                    blank_frames += 1
                else:
                    sum_sign_token += signs
                    token_frames += 1
                positive += (contribution_cpu > 0).long()
                negative += (contribution_cpu < 0).long()
    if not aligned_frames:
        raise ValueError("utility split produced no aligned non-blank frames")
    utility = (sum_sign / aligned_frames).numpy()
    magnitude = (sum_abs_delta / max(frames_total, 1)).numpy()
    aligned_magnitude = (sum_abs_delta_aligned / aligned_frames).numpy()
    utility_blank = (sum_sign_blank / max(blank_frames, 1)).numpy()
    utility_token = (sum_sign_token / max(token_frames, 1)).numpy()
    top10 = max(1, round(dimension * 0.10))
    top25 = max(1, round(dimension * 0.25))
    report = {
        "protocol": "ctc_decision_aligned_delta_utility_v1",
        "utility_manifest": str(args.utility_manifest),
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "layer": args.layer,
        "test_used": False,
        "frames_total": frames_total,
        "aligned_nonblank_frames": aligned_frames,
        "blank_competitor_frames": blank_frames,
        "token_competitor_frames": token_frames,
        "blank_competitor_fraction": blank_frames / aligned_frames,
        "token_competitor_fraction": token_frames / aligned_frames,
        "spearman_magnitude_utility": float(spearmanr(magnitude, utility).statistic),
        "spearman_blank_token_utility": float(spearmanr(utility_blank, utility_token).statistic),
        "top10_jaccard_magnitude_utility": overlap(top_indices(magnitude, top10), top_indices(utility, top10)),
        "top25_jaccard_magnitude_utility": overlap(top_indices(magnitude, top25), top_indices(utility, top25)),
        "magnitude_all": magnitude.tolist(),
        "magnitude_aligned": aligned_magnitude.tolist(),
        "utility": utility.tolist(),
        "utility_blank": utility_blank.tolist(),
        "utility_token": utility_token.tolist(),
        "positive_count": positive.tolist(),
        "negative_count": negative.tolist(),
        "top10_magnitude": top_indices(magnitude, top10),
        "top10_utility": top_indices(utility, top10),
        "top10_utility_blank": top_indices(utility_blank, top10),
        "top10_utility_token": top_indices(utility_token, top10),
        "top25_magnitude": top_indices(magnitude, top25),
        "top25_utility": top_indices(utility, top25),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("layer", "frames_total", "aligned_nonblank_frames", "blank_competitor_fraction", "spearman_magnitude_utility", "top10_jaccard_magnitude_utility")}, sort_keys=True))


if __name__ == "__main__":
    main()
