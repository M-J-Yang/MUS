from __future__ import annotations

import torch

from usde.ctc import prepare_ctc_text
from usde.shift import ctc_gradient_and_taylor_batch_sums, ctc_taylor_batch_sums, reconstruction_error, select_delta


class _Tokenizer:
    def __init__(self, vocabulary: dict[str, int]):
        self._vocabulary = vocabulary
        self.do_lower_case = False

    def get_vocab(self) -> dict[str, int]:
        return self._vocabulary


def test_pretrained_uppercase_tokenizer_receives_uppercase_text() -> None:
    uppercase = _Tokenizer({"<pad>": 0, "<unk>": 1, "|": 2, "A": 3, "B": 4})
    lowercase = _Tokenizer({"<pad>": 0, "<unk>": 1, "|": 2, "a": 3, "b": 4})

    assert prepare_ctc_text("Cab", uppercase) == "CAB"
    assert prepare_ctc_text("Cab", lowercase) == "cab"


def test_reconstruction_error_checks_the_exact_shift_identity() -> None:
    e0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    delta = torch.tensor([[0.5, -1.0], [2.0, 0.25]])
    report = reconstruction_error(e0, e0 + delta, delta)

    assert report["allclose"] is True
    assert report["max_abs_error"] == 0.0


def test_select_delta_reverts_unselected_coordinates_to_e0() -> None:
    e0 = torch.ones(1, 2, 4)
    delta = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
    selected = select_delta(e0, delta, torch.tensor([1, 3]))

    assert torch.equal(selected[..., 0], e0[..., 0])
    assert torch.equal(selected[..., 2], e0[..., 2])
    assert torch.equal(selected[..., 1], e0[..., 1] + delta[..., 1])
    assert torch.equal(selected[..., 3], e0[..., 3] + delta[..., 3])


def test_shift_taylor_batch_sums_matches_direct_autograd() -> None:
    head = torch.nn.Linear(3, 4, bias=True)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[0.2, -0.1, 0.3], [0.4, 0.5, -0.2], [-0.3, 0.1, 0.6], [0.1, -0.4, 0.2]]))
        head.bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.0]))
    e0 = torch.tensor([[[0.2, 0.1, 0.0], [0.4, -0.2, 0.0], [0.0, 0.3, 0.0], [99.0, 99.0, 99.0]], [[-0.1, 0.2, 0.0], [0.5, 0.3, 0.0], [0.2, -0.4, 0.0], [0.0, 0.0, 0.0]]])
    delta = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 0.5], [0.0, 0.0, -0.7], [99.0, 99.0, 99.0]], [[0.0, 0.0, 0.4], [0.0, 0.0, -0.2], [0.0, 0.0, 0.1], [0.0, 0.0, 0.0]]])
    targets = torch.tensor([1, 2, 1], dtype=torch.long)
    feature_lengths = torch.tensor([3, 2], dtype=torch.long)
    target_lengths = torch.tensor([2, 1], dtype=torch.long)

    sums, frame_count, _ = ctc_taylor_batch_sums(
        head, e0, delta, targets, feature_lengths, target_lengths, blank=0
    )

    e0_direct = e0.detach()
    delta_direct = delta.detach().requires_grad_(True)
    logits = head(e0_direct + delta_direct)
    log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
    losses = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)(
        log_probs, targets, feature_lengths, target_lengths
    )
    gradient = torch.autograd.grad((losses / target_lengths).sum(), delta_direct)[0]
    valid = torch.arange(4).unsqueeze(0) < feature_lengths.unsqueeze(1)
    expected = (delta_direct * gradient).abs().masked_fill(~valid.unsqueeze(-1), 0.0).sum(dim=(0, 1))

    assert frame_count == 5
    assert torch.allclose(sums, expected)
def test_gradient_and_taylor_helper_keeps_legacy_taylor_result() -> None:
    head = torch.nn.Linear(2, 3)
    e0 = torch.zeros(1, 3, 2)
    delta = torch.tensor([[[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]]])
    targets = torch.tensor([1], dtype=torch.long)
    feature_lengths = torch.tensor([3], dtype=torch.long)
    target_lengths = torch.tensor([1], dtype=torch.long)

    gradient, taylor, frame_count, _ = ctc_gradient_and_taylor_batch_sums(
        head, e0, delta, targets, feature_lengths, target_lengths, blank=0
    )
    legacy_taylor, legacy_frames, _ = ctc_taylor_batch_sums(
        head, e0, delta, targets, feature_lengths, target_lengths, blank=0
    )

    assert frame_count == legacy_frames == 3
    assert gradient.shape == taylor.shape == torch.Size([2])
    assert torch.all(gradient >= 0)
    assert torch.allclose(taylor, legacy_taylor)
