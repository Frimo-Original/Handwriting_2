from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from handwriting_ai.checkpoint import save_checkpoint
from handwriting_ai.config import ExperimentConfig
from handwriting_ai.data import build_dataloaders
from handwriting_ai.data.codec import VOCAB_TOKENS
from handwriting_ai.models import InkAutoencoder
from handwriting_ai.seed import resolve_device, seed_everything
from handwriting_ai.training.common import (
    average_metric_dicts,
    configure_torch,
    maybe_compile,
    model_state_dict,
    to_plain,
    write_json,
)
from handwriting_ai.training.losses import autoencoder_loss


def _model_kwargs(config: ExperimentConfig) -> dict[str, int | float]:
    ae = config.autoencoder
    return {
        "input_dim": 3,
        "hidden_dim": ae.hidden_dim,
        "latent_dim": ae.latent_dim,
        "downsample_factor": ae.downsample_factor,
        "bottleneck_layers": ae.bottleneck_layers,
        "n_heads": ae.n_heads,
        "dropout": ae.dropout,
    }


def train_autoencoder(config: ExperimentConfig) -> Path:
    seed_everything(config.run.seed)
    configure_torch(config)
    device = resolve_device(config.hardware.device)
    train_loader, val_loader, stats = build_dataloaders(config.data, config.hardware)
    run_dir = config.run.out_dir / "autoencoder"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)

    model = InkAutoencoder(**_model_kwargs(config)).to(device)
    model = maybe_compile(model, config.hardware.torch_compile)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.autoencoder.learning_rate,
        weight_decay=config.autoencoder.weight_decay,
    )
    amp_enabled = config.hardware.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val = float("inf")
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    for epoch in range(1, config.autoencoder.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_metrics: list[dict[str, float]] = []
        progress = tqdm(train_loader, desc=f"AE train {epoch}", leave=False)
        for step, batch in enumerate(progress, start=1):
            batch = batch.to(device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(batch.points, batch.point_lengths)
                loss, metrics = autoencoder_loss(
                    output,
                    batch.points,
                    batch.point_mask,
                    kl_weight=config.autoencoder.kl_weight,
                    pen_weight=config.autoencoder.pen_weight,
                    curvature_weight=config.autoencoder.curvature_weight,
                    render_weight=config.autoencoder.render_weight,
                )
                loss = loss / config.autoencoder.grad_accum_steps
            scaler.scale(loss).backward()
            if step % config.autoencoder.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.autoencoder.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_metrics.append(metrics)
            progress.set_postfix(loss=f"{metrics['loss']:.4f}")

        val_metrics = evaluate_autoencoder(model, val_loader, config, device)
        train_avg = average_metric_dicts(train_metrics)
        print(
            f"AE epoch {epoch}: train_loss={train_avg['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_xy={val_metrics['xy']:.4f}"
        )

        payload = {
            "epoch": epoch,
            "model_state": model_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "model_kwargs": _model_kwargs(config),
            "normalization": stats.to_dict(),
            "vocab_tokens": VOCAB_TOKENS,
            "config": to_plain(config),
            "val_metrics": val_metrics,
        }
        save_checkpoint(last_path, **payload)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(best_path, **payload)
        if epoch % config.autoencoder.checkpoint_every == 0:
            save_checkpoint(run_dir / f"epoch_{epoch:04d}.pt", **payload)

    return best_path


@torch.no_grad()
def evaluate_autoencoder(
    model: torch.nn.Module,
    loader,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metrics_list: list[dict[str, float]] = []
    for batch in loader:
        batch = batch.to(device)
        output = model(batch.points, batch.point_lengths)
        _, metrics = autoencoder_loss(
            output,
            batch.points,
            batch.point_mask,
            kl_weight=config.autoencoder.kl_weight,
            pen_weight=config.autoencoder.pen_weight,
            curvature_weight=config.autoencoder.curvature_weight,
            render_weight=0.0,
        )
        metrics_list.append(metrics)
    return average_metric_dicts(metrics_list)
