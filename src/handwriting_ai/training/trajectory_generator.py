from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from tqdm import tqdm

from handwriting_ai.checkpoint import load_checkpoint, save_checkpoint
from handwriting_ai.config import DataConfig, ExperimentConfig
from handwriting_ai.data import build_dataloaders
from handwriting_ai.data.codec import CTC_BLANK_ID, VOCAB_TOKENS
from handwriting_ai.metrics import ctc_character_error_rate
from handwriting_ai.models import TrajectoryGenerator, TrajectoryRecognizer
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
    trajectory_for_recognizer,
    trajectory_generator_loss,
)


TRAJECTORY_GENERATOR_TRAINING_VERSION = 1


def _model_kwargs(config: ExperimentConfig) -> dict[str, int | float]:
    tg = config.trajectory_generator
    return {
        "hidden_dim": tg.hidden_dim,
        "text_dim": tg.text_dim,
        "layers": tg.layers,
        "n_heads": tg.n_heads,
        "dropout": tg.dropout,
    }


def _loss_kwargs(config: ExperimentConfig) -> dict[str, float]:
    tg = config.trajectory_generator
    return {
        "length_loss_weight": tg.length_loss_weight,
        "xy_weight": tg.xy_weight,
        "path_weight": tg.path_weight,
        "pen_weight": tg.pen_weight,
        "pen_pos_weight": tg.pen_pos_weight,
        "curvature_weight": tg.curvature_weight,
        "jump_weight": tg.jump_weight,
    }


def _data_config(config: ExperimentConfig) -> DataConfig:
    if config.trajectory_generator.augment_targets:
        return config.data
    return replace(config.data, augment=False, jitter_std=0.0, scale_jitter=0.0)


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


def train_trajectory_generator(
    config: ExperimentConfig,
    recognizer_checkpoint: str | Path,
) -> Path:
    seed_everything(config.run.seed)
    configure_torch(config)
    device = resolve_device(config.hardware.device)
    recognizer, _ = _load_recognizer(recognizer_checkpoint, device)
    data_config = _data_config(config)
    train_loader, val_loader, stats = build_dataloaders(data_config, config.hardware)

    run_dir = config.run.out_dir / "trajectory_generator"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)
    if not config.trajectory_generator.augment_targets and config.data.augment:
        print("Trajectory generator target augmentation: disabled")

    model = TrajectoryGenerator(**_model_kwargs(config)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.trajectory_generator.learning_rate,
        weight_decay=config.trajectory_generator.weight_decay,
    )
    amp_enabled = config.hardware.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_selection = float("inf")
    best_train = float("inf")
    best_path = run_dir / "best.pt"
    train_best_path = run_dir / "train_best.pt"
    last_path = run_dir / "last.pt"

    global_step = 0
    for epoch in range(1, config.trajectory_generator.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        metrics_list: list[dict[str, float]] = []
        progress = tqdm(train_loader, desc=f"Trajectory gen train {epoch}", leave=False)
        for step, batch in enumerate(progress, start=1):
            global_step += 1
            batch = batch.to(device)
            semantic_weight = (
                config.trajectory_generator.semantic_weight
                if global_step % max(config.trajectory_generator.semantic_every, 1) == 0
                else 0.0
            )
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                loss, metrics = trajectory_generator_loss(
                    model,
                    batch.points,
                    batch.point_mask,
                    batch.point_lengths,
                    batch.text,
                    batch.text_mask,
                    recognizer=recognizer,
                    text_lengths=batch.text_lengths,
                    semantic_weight=semantic_weight,
                    blank_id=CTC_BLANK_ID,
                    **_loss_kwargs(config),
                )
                scaled_loss = loss / config.trajectory_generator.grad_accum_steps
            scaler.scale(scaled_loss).backward()
            if step % config.trajectory_generator.grad_accum_steps == 0 or step == len(train_loader):
                optimizer_step(
                    scaler=scaler,
                    optimizer=optimizer,
                    model=model,
                    max_grad_norm=config.trajectory_generator.max_grad_norm,
                )
            metrics_list.append(metrics)
            progress.set_postfix(loss=f"{metrics['loss']:.4f}")

        train_avg = average_metric_dicts(metrics_list)
        should_eval = (
            epoch % config.trajectory_generator.eval_every == 0
            or epoch == config.trajectory_generator.epochs
        )
        if not should_eval:
            print(
                f"Trajectory gen epoch {epoch}: "
                f"train_loss={train_avg['loss']:.4f} val=skipped"
            )
            continue

        val_metrics = evaluate_trajectory_generator(
            model,
            recognizer,
            val_loader,
            config,
            device,
        )
        print(
            f"Trajectory gen epoch {epoch}: train_loss={train_avg['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_xy={val_metrics['xy']:.4f} "
            f"val_jump={val_metrics['jump']:.4f} val_cer={val_metrics['generation_cer']:.4f}"
        )

        selection_metric = val_metrics["generation_cer"] + 0.01 * val_metrics["loss"]
        payload = {
            "epoch": epoch,
            "trajectory_generator_training_version": TRAJECTORY_GENERATOR_TRAINING_VERSION,
            "metric": selection_metric,
            "best_metric": min(best_selection, selection_metric),
            "train_metric": train_avg["loss"],
            "best_train_metric": min(best_train, train_avg["loss"]),
            "model_state": model_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "model_type": "trajectory_generator",
            "model_kwargs": _model_kwargs(config),
            "recognizer_checkpoint": str(recognizer_checkpoint),
            "normalization": stats.to_dict(),
            "vocab_tokens": VOCAB_TOKENS,
            "config": to_plain(config),
            "trajectory_generator_target_augmentation": config.trajectory_generator.augment_targets,
            "loss_weights": {
                **_loss_kwargs(config),
                "semantic_weight": config.trajectory_generator.semantic_weight,
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
            epoch % config.trajectory_generator.checkpoint_every == 0
            or epoch == config.trajectory_generator.epochs
        ):
            save_checkpoint(run_dir / f"epoch_{epoch:04d}.pt", **payload)

    return best_path


@torch.no_grad()
def evaluate_trajectory_generator(
    model: TrajectoryGenerator,
    recognizer: TrajectoryRecognizer,
    loader,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metrics_list: list[dict[str, float]] = []
    generation_cers: list[float] = []
    generated_samples = 0
    max_point_length = config.data.max_points or 4096
    fork_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(config.run.seed + 20_000)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.run.seed + 20_000)
        for batch in loader:
            batch = batch.to(device)
            _, metrics = trajectory_generator_loss(
                model,
                batch.points,
                batch.point_mask,
                batch.point_lengths,
                batch.text,
                batch.text_mask,
                recognizer=recognizer,
                text_lengths=batch.text_lengths,
                semantic_weight=config.trajectory_generator.semantic_weight,
                blank_id=CTC_BLANK_ID,
                **_loss_kwargs(config),
            )
            metrics_list.append(metrics)

            remaining = config.trajectory_generator.eval_samples - generated_samples
            if remaining <= 0:
                continue
            count = min(remaining, batch.text.shape[0])
            sampled, sampled_mask = model.sample(
                batch.text[:count],
                batch.text_mask[:count],
                temperature=config.trajectory_generator.temperature,
                max_point_length=max_point_length,
            )
            generated_lengths = sampled_mask.sum(dim=1).long()
            log_probs, output_lengths = recognizer(
                trajectory_for_recognizer(sampled),
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
