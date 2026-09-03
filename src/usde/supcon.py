"""Transcript-grouped supervised contrastive fine-tuning for Wav2Vec2 CTC."""

from __future__ import annotations

import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from usde.ctc import TARGET_SAMPLE_RATE, audio_length, load_audio, prepare_ctc_text, resolve_audio_path
from usde.text import normalize_text


def stable_text_id(text: str, n_hex: int = 8) -> int:
    import hashlib

    return int(hashlib.md5(normalize_text(text).upper().encode("utf-8")).hexdigest()[:n_hex], 16)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    import json

    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"{path}: no records")
    return rows


class OfficialSupConDataset(Dataset[dict[str, Any]]):
    """Load the public pipeline's 16-kHz, max-10-second examples."""

    def __init__(
        self,
        manifest_path: Path,
        processor,
        *,
        audio_root: Path | None = None,
        max_duration_s: float = 10.0,
        supcon_enabled: bool = False,
        max_examples: int | None = None,
        sample_seed: int = 1337,
    ) -> None:
        self.manifest_path = manifest_path
        self.processor = processor
        self.records = read_manifest(manifest_path)
        if max_examples is not None:
            if max_examples < 1:
                raise ValueError("max_examples must be positive")
            if max_examples < len(self.records):
                self.records = random.Random(sample_seed).sample(self.records, max_examples)
        self.audio_root = audio_root
        self.max_samples = int(max_duration_s * TARGET_SAMPLE_RATE)
        self.supcon_enabled = bool(supcon_enabled)
        self.paths = [
            resolve_audio_path(str(row.get("audio_path", row.get("wav_path"))), manifest_path, audio_root)
            for row in self.records
        ]
        missing = [str(path) for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{manifest_path}: missing audio, first path is {missing[0]}")
        self.input_lengths = [min(audio_length(path), self.max_samples) for path in self.paths]
        self.supcon_ids = [
            int(row.get("supcon_id", stable_text_id(str(row["transcript"])) if supcon_enabled else -1))
            if supcon_enabled
            else -1
            for row in self.records
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        waveform, sample_rate = load_audio(self.paths[index])
        waveform = waveform[: self.max_samples]
        encoded_audio = self.processor(
            waveform.numpy(),
            sampling_rate=sample_rate,
            return_attention_mask=True,
        )
        encoded_text = self.processor.tokenizer(
            prepare_ctc_text(str(self.records[index]["transcript"]), self.processor.tokenizer),
            truncation=True,
            max_length=128,
        )
        return {
            "input_values": encoded_audio["input_values"][0],
            "attention_mask": encoded_audio.get(
                "attention_mask", [1] * len(encoded_audio["input_values"][0])
            )[0],
            "labels": encoded_text["input_ids"],
            "supcon_id": self.supcon_ids[index],
            "is_synth": 0,
        }


@dataclass
class SupConDataCollator:
    processor: Any
    include_metadata: bool = True

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audio = [
            {key: item[key] for key in ("input_values", "attention_mask") if key in item}
            for item in features
        ]
        batch = self.processor.feature_extractor.pad(
            audio,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        labels = self.processor.tokenizer.pad(
            [{"input_ids": item["labels"]} for item in features],
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        if self.include_metadata:
            batch["supcon_id"] = torch.tensor([int(item.get("supcon_id", -1)) for item in features], dtype=torch.long)
            batch["is_synth"] = torch.tensor([int(item.get("is_synth", 0)) for item in features], dtype=torch.long)
        return batch


class TranscriptGroupedBatchSampler(Sampler[list[int]]):
    """Make batches containing repeated-transcript positive groups.

    The public recipe uses ``group_size * samples_per_group`` examples per
    batch.  Under DDP, complete batches are deterministically sharded across
    ranks so that each rank still computes SupCon on real positive pairs.
    """

    def __init__(
        self,
        dataset: OfficialSupConDataset,
        *,
        batch_size: int = 16,
        group_size: int = 4,
        samples_per_group: int = 4,
        seed: int = 42,
        drop_last: bool = True,
        distributed: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.group_size = int(group_size)
        self.samples_per_group = int(samples_per_group)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.world_size = int(os.environ.get("WORLD_SIZE", "1")) if distributed else 1
        self.rank = int(os.environ.get("RANK", "0")) if distributed else 0
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError(f"invalid distributed state rank={self.rank} world_size={self.world_size}")
        if self.batch_size != self.group_size * self.samples_per_group:
            raise ValueError("batch_size must equal group_size * samples_per_group")

        buckets: dict[int, list[int]] = defaultdict(list)
        for index, supcon_id in enumerate(dataset.supcon_ids):
            if int(supcon_id) != -1:
                buckets[int(supcon_id)].append(index)
        self.buckets = dict(buckets)
        self.keys = [key for key, indices in self.buckets.items() if len(indices) >= 2]
        if not self.keys:
            raise ValueError("no transcript bucket has at least two examples")
        self.global_batches = len(self.keys) // self.group_size if self.drop_last else math.ceil(len(self.keys) / self.group_size)
        self.local_batches = math.ceil(self.global_batches / self.world_size)

    def __len__(self) -> int:
        return self.local_batches

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterable[list[int]]:
        epoch = int(getattr(self, "_epoch", 0))
        self._epoch = epoch + 1
        rng = random.Random(self.seed + epoch)
        keys = self.keys[:]
        rng.shuffle(keys)
        batches: list[list[int]] = []
        for start in range(0, self.global_batches * self.group_size, self.group_size):
            group_keys = keys[start : start + self.group_size]
            if len(group_keys) < self.group_size:
                if self.drop_last:
                    continue
                group_keys += keys[: self.group_size - len(group_keys)]
            batch: list[int] = []
            for key in group_keys:
                indices = self.buckets[key]
                if len(indices) >= self.samples_per_group:
                    chosen = rng.sample(indices, self.samples_per_group)
                else:
                    chosen = [rng.choice(indices) for _ in range(self.samples_per_group)]
                batch.extend(chosen)
            batches.append(batch)
        while len(batches) < self.local_batches * self.world_size:
            batches.append(list(batches[len(batches) % max(1, len(batches))]))
        for batch_index, batch in enumerate(batches):
            if batch_index % self.world_size == self.rank:
                yield batch


class SupConProjection(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, proj_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class W2V2SupCon(nn.Module):
    def __init__(self, base: nn.Module, processor, proj_dim: int = 256) -> None:
        super().__init__()
        self.base = base
        self.processor = processor
        self.config = base.config
        self.proj = SupConProjection(int(base.config.hidden_size), proj_dim=proj_dim)

    def forward(self, input_values, attention_mask=None, labels=None, **kwargs):
        return self.base(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            **kwargs,
        )


def unwrap_model(model: nn.Module) -> W2V2SupCon:
    candidate = model.module if hasattr(model, "module") else model
    return candidate  # type: ignore[return-value]


def masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


def feature_lengths(base: nn.Module, attention_mask: torch.Tensor | None, t_enc: int, device: torch.device, batch_size: int) -> torch.Tensor:
    if attention_mask is None:
        return torch.full((batch_size,), t_enc, dtype=torch.long, device=device)
    wav_lens = attention_mask.to(device).sum(dim=1).to(torch.long)
    lengths = base._get_feat_extract_output_lengths(wav_lens).to(torch.long)
    return torch.clamp(lengths, min=1, max=t_enc)


def make_feature_mask(base: nn.Module, attention_mask: torch.Tensor | None, t_enc: int, device: torch.device, batch_size: int) -> torch.Tensor:
    lengths = feature_lengths(base, attention_mask, t_enc, device, batch_size)
    return torch.arange(t_enc, device=device).unsqueeze(0) < lengths.unsqueeze(1)


def ctc_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor, input_lengths: torch.Tensor, blank_id: int) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
    target_mask = labels != -100
    target_lengths = target_mask.sum(dim=1).to(torch.long)
    targets = labels.masked_select(target_mask).to(torch.long)
    return F.ctc_loss(
        log_probs,
        targets,
        input_lengths.to(torch.long),
        target_lengths,
        blank=blank_id,
        reduction="mean",
        zero_infinity=True,
    )


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    similarity = (z @ z.t()) / temperature
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    labels = labels.view(-1, 1)
    positive_mask = (labels == labels.t()).to(z.dtype)
    logits_mask = torch.ones_like(positive_mask) - torch.eye(z.size(0), device=z.device, dtype=z.dtype)
    positive_mask = positive_mask * logits_mask
    exp_similarity = torch.exp(similarity) * logits_mask
    log_prob = similarity - torch.log(exp_similarity.sum(dim=1, keepdim=True) + 1e-12)
    positive_count = positive_mask.sum(dim=1).clamp(min=1.0)
    return -(positive_mask * log_prob).sum(dim=1).div(positive_count).mean()


class SupConTrainerMixin:
    def __init__(self, *args, supcon_lambda: float = 0.05, supcon_temp: float = 0.1, supcon_ramp_ratio: float = 0.1, batch_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_lambda = float(supcon_lambda)
        self.supcon_temp = float(supcon_temp)
        self.supcon_ramp_ratio = float(supcon_ramp_ratio)
        self._supcon_batch_sampler = batch_sampler

    def get_train_dataloader(self):
        if self._supcon_batch_sampler is None:
            return super().get_train_dataloader()
        return DataLoader(
            self.train_dataset,
            batch_sampler=self._supcon_batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        supcon_ids = inputs.pop("supcon_id", None)
        inputs.pop("is_synth", None)
        wrapped = unwrap_model(model)
        base = wrapped.base
        encoder = getattr(base, "wav2vec2", None) or getattr(base, "wav2vec2_conformer", None)
        if encoder is None:
            raise AttributeError(f"unsupported CTC backbone: {type(base)}")
        hidden = encoder(
            input_values=inputs["input_values"],
            attention_mask=inputs.get("attention_mask"),
            return_dict=True,
        ).last_hidden_state
        logits = base.lm_head(hidden)
        lengths = feature_lengths(
            base,
            inputs.get("attention_mask"),
            hidden.size(1),
            hidden.device,
            hidden.size(0),
        )
        blank_id = int(wrapped.processor.tokenizer.pad_token_id)
        ctc = ctc_loss_from_logits(logits, inputs["labels"], lengths, blank_id)
        total = ctc
        contrastive = None
        lam = 0.0
        valid_n = 0
        if supcon_ids is not None:
            supcon_ids = supcon_ids.to(hidden.device)
            valid = supcon_ids != -1
            valid_n = int(valid.sum().item())
            if valid_n >= 2:
                frame_mask = make_feature_mask(
                    base,
                    inputs.get("attention_mask"),
                    hidden.size(1),
                    hidden.device,
                    hidden.size(0),
                )
                z = wrapped.proj(masked_mean_pool(hidden, frame_mask))[valid]
                contrastive = supcon_loss(z, supcon_ids[valid], self.supcon_temp)
                ramp_steps = max(1, int(self.state.max_steps * self.supcon_ramp_ratio))
                lam = self.supcon_lambda * min(1.0, self.state.global_step / ramp_steps)
                total = total + lam * contrastive
        if model.training and self.args.logging_steps and self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
            self.log(
                {
                    "ctc_loss": float(ctc.detach().cpu()),
                    "supcon_valid_n": float(valid_n),
                    "supcon_lam": float(lam),
                    "supcon_term": 0.0 if contrastive is None else float((lam * contrastive).detach().cpu()),
                }
            )
        if return_outputs:
            return total, {"logits": logits}
        return total
