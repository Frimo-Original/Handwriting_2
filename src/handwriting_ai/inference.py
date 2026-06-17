from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from handwriting_ai.checkpoint import load_checkpoint
from handwriting_ai.data.codec import PAD_ID, encode_text
from handwriting_ai.data.dataset import NormalizationStats
from handwriting_ai.data.rendering import render_points_to_image
from handwriting_ai.data.transforms import deltas_to_points
from handwriting_ai.latent_stats import LatentNormalizationStats, denormalize_latents
from handwriting_ai.models import (
    AlignedLatentFlow,
    ContentAlignedLatentFlow,
    InkAutoencoder,
    LatentFlowTransformer,
    LatentRegressorTransformer,
    LocalInkAutoencoder,
    TrajectoryGenerator,
)


def _load_autoencoder(
    path: str | Path,
    device: torch.device,
) -> tuple[InkAutoencoder | LocalInkAutoencoder, NormalizationStats]:
    payload = load_checkpoint(path, map_location=device)
    if payload.get("model_type") == "local_content_autoencoder":
        model = LocalInkAutoencoder(**payload["model_kwargs"]).to(device)
    else:
        model = InkAutoencoder(**payload["model_kwargs"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, NormalizationStats.from_dict(payload["normalization"])


def _load_generator(
    path: str | Path,
    device: torch.device,
) -> tuple[
    AlignedLatentFlow
    | ContentAlignedLatentFlow
    | LatentFlowTransformer
    | LatentRegressorTransformer
    | TrajectoryGenerator,
    dict,
]:
    payload = load_checkpoint(path, map_location=device)
    model_type = payload.get("model_type", "latent_flow")
    if model_type == "content_aligned_latent_flow":
        model = ContentAlignedLatentFlow(**payload["model_kwargs"]).to(device)
    elif model_type == "aligned_latent_flow":
        model = AlignedLatentFlow(**payload["model_kwargs"]).to(device)
    elif model_type == "trajectory_generator":
        model = TrajectoryGenerator(**payload["model_kwargs"]).to(device)
    elif model_type == "latent_regressor":
        model = LatentRegressorTransformer(**payload["model_kwargs"]).to(device)
    elif model_type == "latent_flow":
        model = LatentFlowTransformer(**payload["model_kwargs"]).to(device)
    else:
        raise ValueError(f"Unknown generator model_type in checkpoint {path}: {model_type!r}")
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def generate_points(
    *,
    text: str,
    autoencoder_checkpoint: str | Path,
    generator_checkpoint: str | Path,
    device: torch.device,
    steps: int = 32,
    temperature: float = 0.85,
    latent_length: int | None = None,
    max_latent_length: int = 768,
    pen_threshold: float = 0.5,
) -> np.ndarray:
    generator, generator_payload = _load_generator(generator_checkpoint, device)
    model_type = generator_payload.get("model_type", "latent_flow")
    encoded = torch.tensor([encode_text(text)], dtype=torch.long, device=device)
    text_mask = encoded != PAD_ID
    with torch.no_grad():
        if model_type == "trajectory_generator":
            stats = NormalizationStats.from_dict(generator_payload["normalization"])
            deltas_tensor, point_mask = generator.sample(
                encoded,
                text_mask,
                point_length=latent_length,
                steps=steps,
                temperature=temperature,
                max_point_length=max_latent_length,
            )
            deltas = deltas_tensor[0, : point_mask.shape[1]].detach().cpu().float().numpy()
        else:
            autoencoder, stats = _load_autoencoder(autoencoder_checkpoint, device)
            latent_stats = LatentNormalizationStats.from_dict(generator_payload.get("latent_normalization"))
            latents, latent_mask = generator.sample(
                encoded,
                text_mask,
                latent_length=latent_length,
                steps=steps,
                temperature=temperature,
                max_latent_length=max_latent_length,
            )
            if latent_stats is not None:
                latents = denormalize_latents(latents, latent_stats) * latent_mask.unsqueeze(-1)
            decoded = autoencoder.decode(latents, target_length=latent_mask.shape[1] * autoencoder.downsample_factor)
            deltas = decoded[0].detach().cpu().float().numpy()
    deltas[:, 2] = (1.0 / (1.0 + np.exp(-deltas[:, 2])) > pen_threshold).astype(np.float32)
    if len(deltas):
        deltas[-1, 2] = 1.0
    return deltas_to_points(
        deltas,
        within_mean=stats.mean,
        within_std=stats.std,
        jump_mean=stats.jump_mean,
        jump_std=stats.jump_std,
    )


def save_points_json(path: str | Path, points: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [[float(x), float(y), int(flag > 0.5)] for x, y, flag in points]
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_points_png(path: str | Path, points: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    render_points_to_image(points).save(output)
