"""Checkpoint loading helpers for the CTC experiments."""

from __future__ import annotations

from pathlib import Path

from transformers import AutoConfig, AutoModelForCTC


def load_ctc_model(checkpoint: Path | str) -> AutoModelForCTC:
    """Load a CTC checkpoint without silently dropping parametrized weights.

    Some Transformers versions save the weight-normalized positional
    convolution as ``parametrizations.weight.original{0,1}``, while the
    corresponding model module exposes ``weight_g`` and ``weight_v``.  The
    generic ``from_pretrained`` path only warns and initializes those two
    tensors randomly.  Remap that format explicitly and fail on any other
    state-dict mismatch.
    """

    checkpoint = Path(checkpoint)
    safe_path = checkpoint / "model.safetensors"
    if not safe_path.is_file():
        return AutoModelForCTC.from_pretrained(str(checkpoint))

    from safetensors.torch import load_file

    model = AutoModelForCTC.from_config(AutoConfig.from_pretrained(str(checkpoint)))
    state = load_file(str(safe_path), device="cpu")
    remapped = {
        key.replace(
            ".parametrizations.weight.original0", ".weight_g"
        ).replace(
            ".parametrizations.weight.original1", ".weight_v"
        ): value
        for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint load mismatch for {checkpoint}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return model


__all__ = ["load_ctc_model"]
