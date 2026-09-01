from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_module():
    path = Path(__file__).parents[1] / "utility" / "compute_ctc_taylor_utility.py"
    spec = importlib.util.spec_from_file_location("ctc_taylor_utility", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load CTC Taylor utility module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ctc_taylor_batch_sums_matches_direct_autograd_and_ignores_padding() -> None:
    module = _load_module()
    model = torch.nn.Linear(3, 4, bias=True)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([
            [0.2, -0.1, 0.3],
            [0.4, 0.5, -0.2],
            [-0.3, 0.1, 0.6],
            [0.1, -0.4, 0.2],
        ]))
        model.bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.0]))
    features = torch.tensor([
        [[0.2, 0.1, 1.0], [0.4, -0.2, 0.5], [0.0, 0.3, -0.7], [99.0, 99.0, 99.0]],
        [[-0.1, 0.2, 0.4], [0.5, 0.3, -0.2], [0.2, -0.4, 0.1], [0.0, 0.0, 0.0]],
    ])
    targets = torch.tensor([1, 2, 1], dtype=torch.long)
    feature_lengths = torch.tensor([3, 2], dtype=torch.long)
    target_lengths = torch.tensor([2, 1], dtype=torch.long)

    sums, frame_count, _ = module.ctc_taylor_batch_sums(
        model,
        features,
        targets,
        feature_lengths,
        target_lengths,
        reference_dim=2,
        blank=0,
    )

    delta = features[..., 2:].detach().requires_grad_(True)
    logits = model(torch.cat((features[..., :2], delta), dim=-1))
    log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
    criterion = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)
    losses = criterion(log_probs, targets, feature_lengths, target_lengths)
    gradient = torch.autograd.grad((losses / target_lengths).sum(), delta)[0]
    valid = torch.arange(4).unsqueeze(0) < feature_lengths.unsqueeze(1)
    expected = (delta * gradient).abs().masked_fill(~valid.unsqueeze(-1), 0.0).sum(dim=(0, 1))

    assert frame_count == 5
    assert torch.allclose(sums, expected)
