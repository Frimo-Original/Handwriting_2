from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from handwriting_ai.checkpoint import load_checkpoint, save_checkpoint
from handwriting_ai.config import ExperimentConfig
from handwriting_ai.data import build_dataloaders
from handwriting_ai.data.codec import CTC_BLANK_ID, VOCAB_TOKENS
from handwriting_ai.metrics import ctc_character_error_rate
from handwriting_ai.models import LocalInkAutoencoder, TrajectoryRecognizer
from handwriting_ai.seed import resolve_device, seed_everything
from handwriting_ai.training.common import (
    average_metric_dicts,
    configure_torch,
    maybe_compile,
    model_state_dict,
    to_plain,
    write_json,
)
from handwriting_ai.training.losses import (
    autoencoder_loss,
    semantic_ctc_loss,
    trajectory_for_recognizer,
)


def _model_kwargs(config: ExperimentConfig) -> dict[str, int | float]:
    ae = config.autoencoder
    return {
        "input_dim": 3,
        "hidden_dim": ae.hidden_dim,
        "latent_dim": ae.latent_dim,
        "downsample_factor": ae.downsample_factor,
        "bottleneck_layers": ae.bottleneck_layers,
        "local_kernel_size": ae.local_kernel_size,
        "dropout": ae.dropout,
    }


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


def _validate_normalization(stats, recognizer_payload: dict) -> None:
    rec_stats = recognizer_payload.get("normalization")
    if rec_stats is None:
        raise ValueError("Recognizer checkpoint has no normalization statistics")
    if not np.allclose(stats.mean, rec_stats["mean"], atol=1e-5) or not np.allclose(
        stats.std,
        rec_stats["std"],
        atol=1e-5,
    ):
        raise ValueError(
            "Recognizer and autoencoder use different normalization statistics. "
            "Retrain the recognizer with the same data config."
        )


def train_autoencoder(
    config: ExperimentConfig,
    recognizer_checkpoint: str | Path,
) -> Path:
    seed_everything(config.run.seed)
    configure_torch(config)
    device = resolve_device(config.hardware.device)
    train_loader, val_loader, stats = build_dataloaders(config.data, config.hardware)
    recognizer, recognizer_payload = _load_recognizer(recognizer_checkpoint, device)
    _validate_normalization(stats, recognizer_payload)
    run_dir = config.run.out_dir / "autoencoder"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)

    model = LocalInkAutoencoder(**_model_kwargs(config)).to(device)
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
                base_loss, metrics = autoencoder_loss(
                    output,
                    batch.points,
                    batch.point_mask,
                    kl_weight=config.autoencoder.kl_weight,
                    pen_weight=config.autoencoder.pen_weight,
                    curvature_weight=config.autoencoder.curvature_weight,
                    render_weight=config.autoencoder.render_weight,
                )
                use_content_loss = step % max(config.autoencoder.content_loss_every, 1) == 0
                content = output.reconstruction.new_tensor(0.0)
                if config.autoencoder.content_loss_weight > 0.0 and use_content_loss:
                    content = semantic_ctc_loss(
                        recognizer,
                        output.reconstruction,
                        batch.point_lengths,
                        batch.text,
                        batch.text_lengths,
                        blank_id=CTC_BLANK_ID,
                    )
                total_loss = base_loss + config.autoencoder.content_loss_weight * content
                metrics["content"] = float(content.detach().cpu())
                metrics["loss"] = float(total_loss.detach().cpu())
                loss = total_loss / config.autoencoder.grad_accum_steps
            scaler.scale(loss).backward()
            if step % config.autoencoder.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.autoencoder.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_metrics.append(metrics)
            progress.set_postfix(loss=f"{metrics['loss']:.4f}")

        train_avg = average_metric_dicts(train_metrics)
        should_eval = epoch % config.autoencoder.eval_every == 0 or epoch == config.autoencoder.epochs
        if not should_eval:
            print(f"AE epoch {epoch}: train_loss={train_avg['loss']:.4f} val=skipped")
            continue

        val_metrics = evaluate_autoencoder(
            model,
            recognizer,
            val_loader,
            config,
            device,
        )
        print(
            f"AE epoch {epoch}: train_loss={train_avg['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_xy={val_metrics['xy']:.4f} "
            f"val_cer={val_metrics['cer']:.4f}"
        )

        payload = {
            "epoch": epoch,
            "model_type": "local_content_autoencoder",
            "autoencoder_training_version": 2,
            "model_state": model_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "model_kwargs": _model_kwargs(config),
            "normalization": stats.to_dict(),
            "vocab_tokens": VOCAB_TOKENS,
            "config": to_plain(config),
            "recognizer_checkpoint": str(recognizer_checkpoint),
            "val_metrics": val_metrics,
        }
        save_checkpoint(last_path, **payload)
        selection_metric = val_metrics["cer"] + 0.01 * val_metrics["loss"]
        if selection_metric < best_val:
            best_val = selection_metric
            save_checkpoint(best_path, **payload)
        if epoch % config.autoencoder.checkpoint_every == 0 or epoch == config.autoencoder.epochs:
            save_checkpoint(run_dir / f"epoch_{epoch:04d}.pt", **payload)

    return best_path


@torch.no_grad()
def evaluate_autoencoder(
    model: torch.nn.Module,
    recognizer: TrajectoryRecognizer,
    loader,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metrics_list: list[dict[str, float]] = []
    for batch in loader:
        batch = batch.to(device)
        output = model(batch.points, batch.point_lengths)
        base_loss, metrics = autoencoder_loss(
            output,
            batch.points,
            batch.point_mask,
            kl_weight=config.autoencoder.kl_weight,
            pen_weight=config.autoencoder.pen_weight,
            curvature_weight=config.autoencoder.curvature_weight,
            render_weight=0.0,
        )
        content = semantic_ctc_loss(
            recognizer,
            output.reconstruction,
            batch.point_lengths,
            batch.text,
            batch.text_lengths,
            blank_id=CTC_BLANK_ID,
        )
        log_probs, output_lengths = recognizer(
            trajectory_for_recognizer(output.reconstruction),
            batch.point_lengths,
        )
        metrics["content"] = float(content.detach().cpu())
        metrics["cer"] = ctc_character_error_rate(
            log_probs,
            output_lengths,
            batch.text,
            batch.text_lengths,
            blank_id=CTC_BLANK_ID,
            include_eos=False,
        )
        metrics["loss"] = float(
            (
                base_loss + config.autoencoder.content_loss_weight * content
            ).detach().cpu()
        )
        metrics_list.append(metrics)
    return average_metric_dicts(metrics_list)
