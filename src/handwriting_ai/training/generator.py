from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from handwriting_ai.checkpoint import load_checkpoint, save_checkpoint
from handwriting_ai.config import ExperimentConfig
from handwriting_ai.data import NormalizationStats, build_dataloaders
from handwriting_ai.data.codec import VOCAB_TOKENS
from handwriting_ai.models import InkAutoencoder, LatentFlowTransformer
from handwriting_ai.seed import resolve_device, seed_everything
from handwriting_ai.training.common import (
    average_metric_dicts,
    configure_torch,
    maybe_compile,
    model_state_dict,
    to_plain,
    write_json,
)
from handwriting_ai.training.losses import flow_matching_loss


def _flow_kwargs(config: ExperimentConfig, latent_dim: int) -> dict[str, int | float]:
    flow = config.generator
    return {
        "latent_dim": latent_dim,
        "hidden_dim": flow.hidden_dim,
        "text_dim": flow.text_dim,
        "layers": flow.layers,
        "n_heads": flow.n_heads,
        "dropout": flow.dropout,
    }


def train_generator(config: ExperimentConfig, autoencoder_checkpoint: str | Path) -> Path:
    seed_everything(config.run.seed)
    configure_torch(config)
    device = resolve_device(config.hardware.device)

    ae_payload = load_checkpoint(autoencoder_checkpoint, map_location=device)
    stats = NormalizationStats.from_dict(ae_payload["normalization"])
    train_loader, val_loader, _ = build_dataloaders(config.data, config.hardware, stats=stats)

    autoencoder = InkAutoencoder(**ae_payload["model_kwargs"]).to(device)
    autoencoder.load_state_dict(ae_payload["model_state"])
    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)

    run_dir = config.run.out_dir / "generator"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)

    model = LatentFlowTransformer(**_flow_kwargs(config, autoencoder.latent_dim)).to(device)
    model = maybe_compile(model, config.hardware.torch_compile)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.generator.learning_rate,
        weight_decay=config.generator.weight_decay,
    )
    amp_enabled = config.hardware.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val = float("inf")
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    for epoch in range(1, config.generator.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        metrics_list: list[dict[str, float]] = []
        progress = tqdm(train_loader, desc=f"Flow train {epoch}", leave=False)
        for step, batch in enumerate(progress, start=1):
            batch = batch.to(device)
            with torch.no_grad():
                latents, _, _, latent_mask, latent_lengths = autoencoder.encode(
                    batch.points,
                    batch.point_lengths,
                    sample=False,
                )
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                loss, metrics = flow_matching_loss(
                    model,
                    latents,
                    latent_mask,
                    latent_lengths,
                    batch.text,
                    batch.text_mask,
                    length_loss_weight=config.generator.length_loss_weight,
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

        val_metrics = evaluate_generator(model, autoencoder, val_loader, config, device)
        train_avg = average_metric_dicts(metrics_list)
        print(
            f"Flow epoch {epoch}: train_loss={train_avg['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_velocity={val_metrics['velocity']:.4f}"
        )

        payload = {
            "epoch": epoch,
            "model_state": model_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "model_kwargs": _flow_kwargs(config, autoencoder.latent_dim),
            "autoencoder_checkpoint": str(autoencoder_checkpoint),
            "normalization": stats.to_dict(),
            "vocab_tokens": VOCAB_TOKENS,
            "config": to_plain(config),
            "val_metrics": val_metrics,
        }
        save_checkpoint(last_path, **payload)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(best_path, **payload)
        if epoch % config.generator.checkpoint_every == 0:
            save_checkpoint(run_dir / f"epoch_{epoch:04d}.pt", **payload)

    return best_path


@torch.no_grad()
def evaluate_generator(
    model: torch.nn.Module,
    autoencoder: InkAutoencoder,
    loader,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metrics_list: list[dict[str, float]] = []
    for batch in loader:
        batch = batch.to(device)
        latents, _, _, latent_mask, latent_lengths = autoencoder.encode(
            batch.points,
            batch.point_lengths,
            sample=False,
        )
        _, metrics = flow_matching_loss(
            model,
            latents,
            latent_mask,
            latent_lengths,
            batch.text,
            batch.text_mask,
            length_loss_weight=config.generator.length_loss_weight,
        )
        metrics_list.append(metrics)
    return average_metric_dicts(metrics_list)
