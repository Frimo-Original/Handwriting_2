from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from handwriting_ai.alignment import (
    AlignmentCache,
    content_tokens,
    ctc_forced_alignment,
)
from handwriting_ai.checkpoint import load_checkpoint, save_checkpoint
from handwriting_ai.config import DataConfig, ExperimentConfig
from handwriting_ai.data import NormalizationStats, build_dataloaders
from handwriting_ai.data.codec import CTC_BLANK_ID, EOS_ID, VOCAB_TOKENS
from handwriting_ai.generator_checkpoint import CURRENT_GENERATOR_TRAINING_VERSION
from handwriting_ai.latent_stats import (
    LatentNormalizationStats,
    denormalize_latents,
    normalize_latents,
)
from handwriting_ai.metrics import ctc_character_error_rate
from handwriting_ai.models import (
    ContentAlignedLatentFlow,
    LocalInkAutoencoder,
    TrajectoryRecognizer,
)
from handwriting_ai.seed import resolve_device, seed_everything
from handwriting_ai.training.common import (
    average_metric_dicts,
    configure_torch,
    model_state_dict,
    optimizer_step,
    to_plain,
    write_json,
)
from handwriting_ai.training.losses import (
    content_aligned_flow_loss,
    trajectory_for_recognizer,
)


def _generator_kwargs(
    config: ExperimentConfig,
    latent_dim: int,
) -> dict[str, int | float]:
    generator = config.generator
    return {
        "latent_dim": latent_dim,
        "hidden_dim": generator.hidden_dim,
        "text_dim": generator.text_dim,
        "layers": generator.layers,
        "n_heads": generator.n_heads,
        "dropout": generator.dropout,
    }


def _generator_data_config(config: ExperimentConfig) -> DataConfig:
    if config.generator.augment_targets:
        return config.data
    return replace(config.data, augment=False, jitter_std=0.0, scale_jitter=0.0)


def _loss_kwargs(config: ExperimentConfig) -> dict[str, float]:
    generator = config.generator
    return {
        "duration_loss_weight": generator.duration_loss_weight,
        "endpoint_latent_weight": generator.endpoint_latent_weight,
        "decoded_xy_weight": generator.decoded_xy_weight,
        "decoded_path_weight": generator.decoded_path_weight,
        "decoded_pen_weight": generator.decoded_pen_weight,
        "decoded_pen_pos_weight": generator.decoded_pen_pos_weight,
        "decoded_curvature_weight": generator.decoded_curvature_weight,
    }


def _load_autoencoder(
    checkpoint: str | Path,
    device: torch.device,
) -> tuple[LocalInkAutoencoder, dict]:
    payload = load_checkpoint(checkpoint, map_location=device)
    if payload.get("model_type") != "local_content_autoencoder":
        raise ValueError(
            "Generator v7 requires a local_content_autoencoder checkpoint. "
            "Retrain the autoencoder with the current code."
        )
    model = LocalInkAutoencoder(**payload["model_kwargs"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def _load_recognizer(
    checkpoint: str | Path,
    device: torch.device,
) -> tuple[TrajectoryRecognizer, dict]:
    payload = load_checkpoint(checkpoint, map_location=device)
    model = TrajectoryRecognizer(**payload["model_kwargs"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.rnn.dropout = 0.0
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def _validate_normalization(
    autoencoder_payload: dict,
    recognizer_payload: dict,
) -> NormalizationStats:
    ae_stats = NormalizationStats.from_dict(autoencoder_payload["normalization"])
    rec_stats = NormalizationStats.from_dict(recognizer_payload["normalization"])
    if not np.allclose(ae_stats.mean, rec_stats.mean, atol=1e-5) or not np.allclose(
        ae_stats.std,
        rec_stats.std,
        atol=1e-5,
    ):
        raise ValueError(
            "Autoencoder and recognizer normalization differ. "
            "Retrain both stages with the same data configuration."
        )
    return ae_stats


@torch.no_grad()
def compute_latent_normalization(
    autoencoder: LocalInkAutoencoder,
    loader,
    device: torch.device,
) -> LatentNormalizationStats:
    autoencoder.eval()
    total = torch.zeros(autoencoder.latent_dim, device=device, dtype=torch.float64)
    total_sq = torch.zeros_like(total)
    count = torch.zeros((), device=device, dtype=torch.float64)
    for batch in tqdm(loader, desc="Latent stats", leave=False):
        batch = batch.to(device)
        latents, _, _, latent_mask, _ = autoencoder.encode(
            batch.points,
            batch.point_lengths,
            sample=False,
        )
        valid = latents[latent_mask].double()
        total += valid.sum(dim=0)
        total_sq += valid.square().sum(dim=0)
        count += valid.shape[0]
    mean = total / count.clamp(min=1.0)
    variance = (total_sq / count.clamp(min=1.0) - mean.square()).clamp(min=1e-6)
    return LatentNormalizationStats(
        mean=[float(value) for value in mean.cpu()],
        std=[float(value) for value in variance.sqrt().cpu()],
    )


def _alignment_signature(
    config: ExperimentConfig,
    recognizer_checkpoint: str | Path,
) -> dict[str, Any]:
    dataset_path = config.data.dataset_npz.resolve()
    recognizer_path = Path(recognizer_checkpoint).resolve()
    dataset_stat = dataset_path.stat()
    recognizer_stat = recognizer_path.stat()
    return {
        "dataset": str(dataset_path),
        "dataset_size": dataset_stat.st_size,
        "dataset_mtime_ns": dataset_stat.st_mtime_ns,
        "recognizer": str(recognizer_path),
        "recognizer_size": recognizer_stat.st_size,
        "recognizer_mtime_ns": recognizer_stat.st_mtime_ns,
        "resample_spacing": config.data.resample_spacing,
        "smooth_window": config.data.smooth_window,
        "max_points": config.data.max_points,
        "max_text_len": config.data.max_text_len,
    }


@torch.no_grad()
def build_alignment_cache(
    recognizer: TrajectoryRecognizer,
    loaders: tuple,
    device: torch.device,
    *,
    metadata: dict[str, Any],
) -> AlignmentCache:
    entries: dict[int, dict[str, Any]] = {}
    scores: list[float] = []
    recognizer.eval()
    for split_name, loader in zip(("train", "val"), loaders, strict=True):
        for batch in tqdm(loader, desc=f"CTC align {split_name}", leave=False):
            batch = batch.to(device)
            log_probs, output_lengths = recognizer(batch.points, batch.point_lengths)
            for row in range(batch.points.shape[0]):
                sample_id = int(batch.sample_ids[row].item())
                targets = content_tokens(
                    batch.text[row],
                    int(batch.text_lengths[row].item()),
                )
                frame_count = int(output_lengths[row].item())
                alignment = ctc_forced_alignment(
                    log_probs[:frame_count, row].detach().float().cpu(),
                    targets.detach().cpu(),
                    blank_id=CTC_BLANK_ID,
                )
                entries[sample_id] = {
                    "durations": alignment.durations,
                    "score": alignment.score,
                    "frames": frame_count,
                    "tokens": int(targets.numel()),
                }
                scores.append(alignment.score)
    cache_metadata = {
        **metadata,
        "samples": len(entries),
        "mean_log_score": float(np.mean(scores)) if scores else float("-inf"),
        "min_log_score": float(np.min(scores)) if scores else float("-inf"),
    }
    return AlignmentCache(entries=entries, metadata=cache_metadata)


def load_or_build_alignment_cache(
    path: Path,
    recognizer: TrajectoryRecognizer,
    loaders: tuple,
    device: torch.device,
    *,
    signature: dict[str, Any],
) -> AlignmentCache:
    if path.exists():
        cache = AlignmentCache.load(path)
        if all(cache.metadata.get(key) == value for key, value in signature.items()):
            print(
                f"Forced alignments: {path} "
                f"({len(cache.entries)} samples, cached)"
            )
            return cache
        print(f"Forced alignment cache is stale, rebuilding: {path}")
    cache = build_alignment_cache(
        recognizer,
        loaders,
        device,
        metadata=signature,
    )
    cache.save(path)
    print(
        f"Forced alignments: {path} ({len(cache.entries)} samples, "
        f"mean score={cache.metadata['mean_log_score']:.4f})"
    )
    return cache


def _content_lengths(text: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
    return (text_mask & (text != EOS_ID)).sum(dim=1)


def _encode_targets(
    autoencoder: LocalInkAutoencoder,
    batch,
    latent_stats: LatentNormalizationStats,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        latents, _, _, latent_mask, latent_lengths = autoencoder.encode(
            batch.points,
            batch.point_lengths,
            sample=False,
        )
        latents = normalize_latents(latents, latent_stats) * latent_mask.unsqueeze(-1)
    return latents, latent_mask, latent_lengths


def _batch_durations(
    cache: AlignmentCache,
    batch,
    latent_lengths: torch.Tensor,
) -> torch.Tensor:
    content_lengths = _content_lengths(batch.text, batch.text_mask)
    return cache.batch(
        batch.sample_ids,
        content_lengths,
        latent_lengths,
        max_text_length=batch.text.shape[1],
        device=batch.text.device,
    )


def train_generator(
    config: ExperimentConfig,
    autoencoder_checkpoint: str | Path,
    recognizer_checkpoint: str | Path,
) -> Path:
    seed_everything(config.run.seed)
    configure_torch(config)
    device = resolve_device(config.hardware.device)

    autoencoder, ae_payload = _load_autoencoder(autoencoder_checkpoint, device)
    recognizer, recognizer_payload = _load_recognizer(recognizer_checkpoint, device)
    stats = _validate_normalization(ae_payload, recognizer_payload)
    data_config = _generator_data_config(config)
    train_loader, val_loader, _ = build_dataloaders(
        data_config,
        config.hardware,
        stats=stats,
    )
    latent_stats = compute_latent_normalization(autoencoder, train_loader, device)

    run_dir = config.run.out_dir / "generator"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)
    alignment_cache = load_or_build_alignment_cache(
        run_dir / "forced_alignments.pt",
        recognizer,
        (train_loader, val_loader),
        device,
        signature=_alignment_signature(config, recognizer_checkpoint),
    )
    recognizer.train()
    if not config.generator.augment_targets and config.data.augment:
        print("Generator target augmentation: disabled")

    model = ContentAlignedLatentFlow(
        **_generator_kwargs(config, autoencoder.latent_dim)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.generator.learning_rate,
        weight_decay=config.generator.weight_decay,
    )
    amp_enabled = config.hardware.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_selection = float("inf")
    best_train = float("inf")
    best_path = run_dir / "best.pt"
    train_best_path = run_dir / "train_best.pt"
    last_path = run_dir / "last.pt"

    global_step = 0
    for epoch in range(1, config.generator.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        metrics_list: list[dict[str, float]] = []
        progress = tqdm(train_loader, desc=f"Content flow train {epoch}", leave=False)
        for step, batch in enumerate(progress, start=1):
            global_step += 1
            batch = batch.to(device)
            latents, latent_mask, latent_lengths = _encode_targets(
                autoencoder,
                batch,
                latent_stats,
            )
            durations = _batch_durations(alignment_cache, batch, latent_lengths)
            semantic_weight = (
                config.generator.semantic_weight
                if global_step % max(config.generator.semantic_every, 1) == 0
                else 0.0
            )
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                loss, metrics = content_aligned_flow_loss(
                    model,
                    latents,
                    latent_mask,
                    latent_lengths,
                    durations,
                    batch.text,
                    batch.text_mask,
                    batch.text_lengths,
                    batch.points,
                    batch.point_mask,
                    batch.point_lengths,
                    autoencoder=autoencoder,
                    latent_normalization=latent_stats,
                    recognizer=recognizer,
                    semantic_weight=semantic_weight,
                    blank_id=CTC_BLANK_ID,
                    **_loss_kwargs(config),
                )
                scaled_loss = loss / config.generator.grad_accum_steps
            scaler.scale(scaled_loss).backward()
            if step % config.generator.grad_accum_steps == 0 or step == len(train_loader):
                optimizer_step(
                    scaler=scaler,
                    optimizer=optimizer,
                    model=model,
                    max_grad_norm=config.generator.max_grad_norm,
                )
            metrics_list.append(metrics)
            progress.set_postfix(loss=f"{metrics['loss']:.4f}")

        train_avg = average_metric_dicts(metrics_list)
        should_eval = (
            epoch % config.generator.eval_every == 0
            or epoch == config.generator.epochs
        )
        if not should_eval:
            print(
                f"Content flow epoch {epoch}: "
                f"train_loss={train_avg['loss']:.4f} val=skipped"
            )
            continue

        val_metrics = evaluate_generator(
            model,
            autoencoder,
            recognizer,
            val_loader,
            config,
            device,
            latent_stats,
            alignment_cache,
        )
        print(
            f"Content flow epoch {epoch}: train_loss={train_avg['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_velocity={val_metrics['velocity']:.4f} "
            f"val_duration={val_metrics['duration']:.4f} "
            f"val_cer={val_metrics['generation_cer']:.4f}"
        )

        selection_metric = (
            val_metrics["generation_cer"] + 0.01 * val_metrics["loss"]
        )
        payload = {
            "epoch": epoch,
            "generator_training_version": CURRENT_GENERATOR_TRAINING_VERSION,
            "metric": selection_metric,
            "best_metric": min(best_selection, selection_metric),
            "train_metric": train_avg["loss"],
            "best_train_metric": min(best_train, train_avg["loss"]),
            "model_state": model_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "model_type": "content_aligned_latent_flow",
            "model_kwargs": _generator_kwargs(config, autoencoder.latent_dim),
            "autoencoder_checkpoint": str(autoencoder_checkpoint),
            "recognizer_checkpoint": str(recognizer_checkpoint),
            "alignment_cache": str(run_dir / "forced_alignments.pt"),
            "normalization": stats.to_dict(),
            "latent_normalization": latent_stats.to_dict(),
            "vocab_tokens": VOCAB_TOKENS,
            "config": to_plain(config),
            "generator_target_augmentation": config.generator.augment_targets,
            "loss_weights": {
                **_loss_kwargs(config),
                "semantic_weight": config.generator.semantic_weight,
            },
            "train_metrics": train_avg,
            "val_metrics": val_metrics,
        }
        save_checkpoint(last_path, **payload)
        if selection_metric < best_selection:
            best_selection = selection_metric
            save_checkpoint(best_path, **payload)
        if train_avg["loss"] < best_train:
            best_train = train_avg["loss"]
            save_checkpoint(train_best_path, **payload)
        if (
            epoch % config.generator.checkpoint_every == 0
            or epoch == config.generator.epochs
        ):
            save_checkpoint(run_dir / f"epoch_{epoch:04d}.pt", **payload)

    return best_path


@torch.no_grad()
def evaluate_generator(
    model: ContentAlignedLatentFlow,
    autoencoder: LocalInkAutoencoder,
    recognizer: TrajectoryRecognizer,
    loader,
    config: ExperimentConfig,
    device: torch.device,
    latent_stats: LatentNormalizationStats,
    alignment_cache: AlignmentCache,
) -> dict[str, float]:
    model.eval()
    metrics_list: list[dict[str, float]] = []
    generation_cers: list[float] = []
    generated_samples = 0
    max_latent_length = (
        math.ceil(config.data.max_points / autoencoder.downsample_factor)
        if config.data.max_points is not None
        else 1024
    )
    fork_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(config.run.seed + 10_000)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.run.seed + 10_000)
        for batch in loader:
            batch = batch.to(device)
            latents, latent_mask, latent_lengths = _encode_targets(
                autoencoder,
                batch,
                latent_stats,
            )
            durations = _batch_durations(alignment_cache, batch, latent_lengths)
            _, metrics = content_aligned_flow_loss(
                model,
                latents,
                latent_mask,
                latent_lengths,
                durations,
                batch.text,
                batch.text_mask,
                batch.text_lengths,
                batch.points,
                batch.point_mask,
                batch.point_lengths,
                autoencoder=autoencoder,
                latent_normalization=latent_stats,
                recognizer=recognizer,
                semantic_weight=config.generator.semantic_weight,
                blank_id=CTC_BLANK_ID,
                **_loss_kwargs(config),
            )
            metrics_list.append(metrics)

            remaining = config.generator.eval_samples - generated_samples
            if remaining <= 0:
                continue
            count = min(remaining, batch.text.shape[0])
            sampled, sampled_mask = model.sample(
                batch.text[:count],
                batch.text_mask[:count],
                steps=config.generator.eval_flow_steps,
                temperature=config.generator.temperature,
                max_latent_length=max_latent_length,
            )
            decoder_latents = denormalize_latents(
                sampled,
                latent_stats,
            ) * sampled_mask.unsqueeze(-1)
            decoded = autoencoder.decode(decoder_latents)
            generated_lengths = (
                sampled_mask.sum(dim=1) * autoencoder.downsample_factor
            ).long()
            log_probs, output_lengths = recognizer(
                trajectory_for_recognizer(decoded),
                generated_lengths,
            )
            generation_cers.append(
                ctc_character_error_rate(
                    log_probs,
                    output_lengths,
                    batch.text[:count],
                    batch.text_lengths[:count],
                    blank_id=CTC_BLANK_ID,
                    include_eos=False,
                )
            )
            generated_samples += count

    result = average_metric_dicts(metrics_list)
    result["generation_cer"] = (
        float(sum(generation_cers) / len(generation_cers))
        if generation_cers
        else float("inf")
    )
    result["generation_samples"] = float(generated_samples)
    return result
