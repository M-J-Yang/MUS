from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load(name: str, relative_path: str):
    path = Path(__file__).parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_matched_manifests_validates_text_and_deduplicates_cmu_side() -> None:
    module = _load("stage7c_prepare", "scripts/prepare_stage7c_matched.py")
    l2 = [
        {"utt_id": "l2_A_a0001", "speaker_id": "A", "prompt_id": "arctic_a0001", "audio_path": "a.wav", "transcript": "Hello world"},
        {"utt_id": "l2_B_a0001", "speaker_id": "B", "prompt_id": "arctic_a0001", "audio_path": "b.wav", "transcript": "Hello world"},
    ]
    cmu = [
        {"utt_id": "cmu_bdl_a0001", "speaker_id": "bdl", "prompt_id": "arctic_a0001", "audio_path": "bdl.wav", "transcript": "Hello, world."},
        {"utt_id": "cmu_slt_a0001", "speaker_id": "slt", "prompt_id": "arctic_a0001", "audio_path": "slt.wav", "transcript": "Hello, world."},
    ]
    l2_side, cmu_side, pairs, summary = module.build_matched_manifests(l2, cmu)
    assert len(l2_side) == 2
    assert len(cmu_side) == 2
    assert len(pairs) == 4
    assert summary["shared_prompts"] == 1
    assert summary["transcripts_verified"] is True


def test_compare_stage7c_requires_complete_same_shape_rankings(tmp_path: Path) -> None:
    module = _load("stage7c_compare", "analysis/compare_stage7c_control.py")
    l2_path = tmp_path / "l2.pt"
    cmu_path = tmp_path / "cmu.pt"
    l2_rank = tmp_path / "l2_rank.pt"
    cmu_rank = tmp_path / "cmu_rank.pt"
    output = tmp_path / "result.json"
    torch.save(torch.tensor([1.0, 3.0, 2.0, 0.5]), l2_path)
    torch.save(torch.tensor([1.1, 2.9, 2.1, 0.4]), cmu_path)
    torch.save(torch.tensor([1, 2, 0, 3]), l2_rank)
    torch.save(torch.tensor([1, 2, 0, 3]), cmu_rank)
    result = module.compare(l2_path, cmu_path, l2_rank, cmu_rank, output)
    assert result["dimension"] == 4
    assert result["top_k_overlap"]["256"] == 1.0
    assert output.is_file()
