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
from handwriting_ai.models import TrajectoryGenerator
from handwriting_ai.seed import resolve_device, seed_everything
from handwriting_ai.training.common import (
    average_metric_dicts,
    configure_torch,
    maybe_compile,
    model_state_dict,
    to_plain,
    write_json,
)
from handwriting_ai.training.losses import trajectory_generator_loss


def _generator_kwargs(config: ExperimentConfig) -> dict[str, int | float]:
    generator = config.generator
    return {
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


def _generator_loss_kwargs(
    config: ExperimentConfig,
) -> dict[str, object]:
    generator = config.generator
    return {
        "xy_weight": generator.decoded_xy_weight,
        "path_weight": generator.decoded_path_weight,
        "pen_weight": generator.decoded_pen_weight,
        "pen_pos_weight": generator.decoded_pen_pos_weight,
        "curvature_weight": generator.decoded_curvature_weight,
        "length_loss_weight": generator.length_loss_weight,
    }


def _loss_weight_payload(config: ExperimentConfig) -> dict[str, float]:
    generator = config.generator
    return {
        "decoded_xy_weight": generator.decoded_xy_weight,
        "decoded_path_weight": generator.decoded_path_weight,
        "decoded_pen_weight": generator.decoded_pen_weight,
        "decoded_pen_pos_weight": generator.decoded_pen_pos_weight,
        "decoded_curvature_weight": generator.decoded_curvature_weight,
        "length_loss_weight": generator.length_loss_weight,
    }


def train_generator(config: ExperimentConfig, autoencoder_checkpoint: str | Path) -> Path:
    seed_everything(config.run.seed)
    configure_torch(config)
    device = resolve_device(config.hardware.device)

    ae_payload = load_checkpoint(autoencoder_checkpoint, map_location=device)
    stats = NormalizationStats.from_dict(ae_payload["normalization"])
    data_config = _generator_data_config(config)
    train_loader, val_loader, _ = build_dataloaders(data_config, config.hardware, stats=stats)

    run_dir = config.run.out_dir / "generator"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)
    if not config.generator.augment_targets and config.data.augment:
        print("Generator target augmentation: disabled (autoencoder augmentation settings are ignored here)")

    model = TrajectoryGenerator(**_generator_kwargs(config)).to(device)
    model = maybe_compile(model, config.hardware.torch_compile)
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
        progress = tqdm(train_loader, desc=f"Trajectory train {epoch}", leave=False)
        for step, batch in enumerate(progress, start=1):
            batch = batch.to(device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                loss, metrics = trajectory_generator_loss(
                    model,
                    batch.points,
                    batch.point_mask,
                    batch.point_lengths,
                    batch.text,
                    batch.text_mask,
                    **_generator_loss_kwargs(config),
                )
                loss = loss / config.generator.grad_accum_steps
            scaler.scale(loss).backward()
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
            print(f"Trajectory epoch {epoch}: train_loss={train_avg['loss']:.4f} val=skipped")
            continue

        val_metrics = evaluate_generator(model, val_loader, config, device)
        print(
            f"Trajectory epoch {epoch}: train_loss={train_avg['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_xy={val_metrics['xy']:.4f} "
            f"val_path={val_metrics['path']:.4f} val_pen={val_metrics['pen']:.4f}"
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
            "model_type": "trajectory_generator",
            "model_kwargs": _generator_kwargs(config),
            "autoencoder_checkpoint": str(autoencoder_checkpoint),
            "normalization": stats.to_dict(),
            "vocab_tokens": VOCAB_TOKENS,
            "config": to_plain(config),
            "generator_target_augmentation": config.generator.augment_targets,
            "loss_weights": _loss_weight_payload(config),
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
    model: torch.nn.Module,
    loader,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metrics_list: list[dict[str, float]] = []
    for batch in loader:
        batch = batch.to(device)
        _, metrics = trajectory_generator_loss(
            model,
            batch.points,
            batch.point_mask,
            batch.point_lengths,
            batch.text,
            batch.text_mask,
            **_generator_loss_kwargs(config),
        )
        metrics_list.append(metrics)
    return average_metric_dicts(metrics_list)
