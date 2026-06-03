from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class LatentNormalizationStats:
    mean: list[float]
    std: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "LatentNormalizationStats | None":
        if payload is None:
            return None
        return cls(mean=[float(v) for v in payload["mean"]], std=[float(v) for v in payload["std"]])


def _stats_tensors(
    stats: LatentNormalizationStats,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(stats.mean, device=device, dtype=dtype).view(1, 1, -1)
    std = torch.tensor(stats.std, device=device, dtype=dtype).view(1, 1, -1)
    return mean, std.clamp(min=1e-6)


def normalize_latents(latents: torch.Tensor, stats: LatentNormalizationStats) -> torch.Tensor:
    mean, std = _stats_tensors(stats, device=latents.device, dtype=latents.dtype)
    return (latents - mean) / std


def denormalize_latents(latents: torch.Tensor, stats: LatentNormalizationStats) -> torch.Tensor:
    mean, std = _stats_tensors(stats, device=latents.device, dtype=latents.dtype)
    return latents * std + mean

