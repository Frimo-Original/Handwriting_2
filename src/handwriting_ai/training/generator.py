from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from tqdm import tqdm

from handwriting_ai.checkpoint import load_checkpoint, save_checkpoint
from handwriting_ai.config import DataConfig, ExperimentConfig
from handwriting_ai.data import NormalizationStats, build_dataloaders
from handwriting_ai.data.codec import VOCAB_TOKENS
from handwriting_ai.generator_checkpoint import CURRENT_GENERATOR_TRAINING_VERSION
from handwriting_ai.latent_stats import LatentNormalizationStats, normalize_latents
from handwriting_ai.models import AlignedLatentFlow, InkAutoencoder
from handwriting_ai.seed import resolve_device, seed_everything
from handwriting_ai.training.common import (
    average_metric_dicts,
    configure_torch,
    model_state_dict,
    to_plain,
    write_json,
)
from handwriting_ai.training.losses import aligned_flow_matching_loss


def _generator_kwargs(config: ExperimentConfig, latent_dim: int) -> dict[str, int | float]:
    generator = config.generator
    return {
        "latent_dim": latent_dim,
        "hidden_dim": generator.hidden_dim,
        "text_dim": generator.text_dim,
        "layers": generator.layers,
        "n_heads": generator.n_heads,
        "dropout": generator.dropout,
        "alignment_prior_strength": generator.alignment_prior_strength,
        "alignment_prior_width": generator.alignment_prior_width,
    }


def _generator_data_config(config: ExperimentConfig) -> DataConfig:
    if config.generator.augment_targets:
        return config.data
    return replace(config.data, augment=False, jitter_std=0.0, scale_jitter=0.0)


def _generator_loss_kwargs(config: ExperimentConfig) -> dict[str, float]:
    return {
        "alignment_loss_weight": config.generator.alignment_loss_weight,
        "duration_loss_weight": config.generator.duration_loss_weight,
    }


@torch.no_grad()
def compute_latent_normalization(
    autoencoder: InkAutoencoder,
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


def _encode_targets(
    autoencoder: InkAutoencoder,
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


def train_generator(config: ExperimentConfig, autoencoder_checkpoint: str | Path) -> Path:
    seed_everything(config.run.seed)
    configure_torch(config)
    device = resolve_device(config.hardware.device)

    ae_payload = load_checkpoint(autoencoder_checkpoint, map_location=device)
    stats = NormalizationStats.from_dict(ae_payload["normalization"])
    autoencoder = InkAutoencoder(**ae_payload["model_kwargs"]).to(device)
    autoencoder.load_state_dict(ae_payload["model_state"])
    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)

    data_config = _generator_data_config(config)
    train_loader, val_loader, _ = build_dataloaders(data_config, config.hardware, stats=stats)
    latent_stats = compute_latent_normalization(autoencoder, train_loader, device)

    run_dir = config.run.out_dir / "generator"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)
    if not config.generator.augment_targets and config.data.augment:
        print("Generator target augmentation: disabled")
    if config.hardware.torch_compile:
        print("Generator torch.compile: disabled because MAS uses dynamic monotonic paths")

    model = AlignedLatentFlow(**_generator_kwargs(config, autoencoder.latent_dim)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.generator.learning_rate,
        weight_decay=config.generator.weight_decay,
    )
    amp_enabled = config.hardware.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val = float("inf")
    best_train = float("inf")
    best_path = run_dir / "best.pt"
    train_best_path = run_dir / "train_best.pt"
    last_path = run_dir / "last.pt"

    for epoch in range(1, config.generator.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        metrics_list: list[dict[str, float]] = []
        progress = tqdm(train_loader, desc=f"Aligned flow train {epoch}", leave=False)
        for step, batch in enumerate(progress, start=1):
            batch = batch.to(device)
            latents, latent_mask, latent_lengths = _encode_targets(autoencoder, batch, latent_stats)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                loss, metrics = aligned_flow_matching_loss(
                    model,
                    latents,
                    latent_mask,
                    latent_lengths,
                    batch.text,
                    batch.text_mask,
                    batch.text_lengths,
                    **_generator_loss_kwargs(config),
                )
                scaled_loss = loss / config.generator.grad_accum_steps
            scaler.scale(scaled_loss).backward()
            if step % config.generator.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.generator.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            metrics_list.append(metrics)
            progress.set_postfix(loss=f"{metrics['loss']:.4f}")

        train_avg = average_metric_dicts(metrics_list)
        should_eval = epoch % config.generator.eval_every == 0 or epoch == config.generator.epochs
        if not should_eval:
            print(f"Aligned flow epoch {epoch}: train_loss={train_avg['loss']:.4f} val=skipped")
            continue

        val_metrics = evaluate_generator(
            model,
            autoencoder,
            val_loader,
            config,
            device,
            latent_stats,
        )
        print(
            f"Aligned flow epoch {epoch}: train_loss={train_avg['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_velocity={val_metrics['velocity']:.4f} "
            f"val_alignment={val_metrics['alignment']:.4f} "
            f"val_duration={val_metrics['duration']:.4f}"
        )

        payload = {
            "epoch": epoch,
            "generator_training_version": CURRENT_GENERATOR_TRAINING_VERSION,
            "metric": val_metrics["loss"],
            "best_metric": min(best_val, val_metrics["loss"]),
            "train_metric": train_avg["loss"],
            "best_train_metric": min(best_train, train_avg["loss"]),
            "model_state": model_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "model_type": "aligned_latent_flow",
            "model_kwargs": _generator_kwargs(config, autoencoder.latent_dim),
            "autoencoder_checkpoint": str(autoencoder_checkpoint),
            "normalization": stats.to_dict(),
            "latent_normalization": latent_stats.to_dict(),
            "vocab_tokens": VOCAB_TOKENS,
            "config": to_plain(config),
            "generator_target_augmentation": config.generator.augment_targets,
            "loss_weights": _generator_loss_kwargs(config),
            "train_metrics": train_avg,
            "val_metrics": val_metrics,
        }
        save_checkpoint(last_path, **payload)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(best_path, **payload)
        if train_avg["loss"] < best_train:
            best_train = train_avg["loss"]
            save_checkpoint(train_best_path, **payload)
        if epoch % config.generator.checkpoint_every == 0 or epoch == config.generator.epochs:
            save_checkpoint(run_dir / f"epoch_{epoch:04d}.pt", **payload)

    return best_path


@torch.no_grad()
def evaluate_generator(
    model: AlignedLatentFlow,
    autoencoder: InkAutoencoder,
    loader,
    config: ExperimentConfig,
    device: torch.device,
    latent_stats: LatentNormalizationStats,
) -> dict[str, float]:
    model.eval()
    metrics_list: list[dict[str, float]] = []
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
            latents, latent_mask, latent_lengths = _encode_targets(autoencoder, batch, latent_stats)
            _, metrics = aligned_flow_matching_loss(
                model,
                latents,
                latent_mask,
                latent_lengths,
                batch.text,
                batch.text_mask,
                batch.text_lengths,
                **_generator_loss_kwargs(config),
            )
            metrics_list.append(metrics)
    return average_metric_dicts(metrics_list)
