from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int
    out_dir: Path


@dataclass(frozen=True)
class HardwareConfig:
    device: str
    num_workers: int
    pin_memory: bool
    amp: bool
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    torch_compile: bool = False
    matmul_precision: str | None = None


@dataclass(frozen=True)
class DataConfig:
    dataset_npz: Path
    val_fraction: float
    split_seed: int
    resample_spacing: float | None
    smooth_window: int
    max_points: int | None
    max_text_len: int | None
    batch_size: int
    augment: bool = False
    jitter_std: float = 0.0
    scale_jitter: float = 0.0


@dataclass(frozen=True)
class AutoencoderConfig:
    hidden_dim: int
    latent_dim: int
    downsample_factor: int
    bottleneck_layers: int
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    grad_accum_steps: int
    max_grad_norm: float
    kl_weight: float
    pen_weight: float
    curvature_weight: float
    render_weight: float
    checkpoint_every: int
    eval_every: int = 1
    local_kernel_size: int = 9
    content_loss_weight: float = 0.15
    content_loss_every: int = 2
    jump_weight: float = 1.0


@dataclass(frozen=True)
class GeneratorConfig:
    hidden_dim: int
    text_dim: int
    layers: int
    n_heads: int
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    grad_accum_steps: int
    max_grad_norm: float
    checkpoint_every: int
    flow_steps: int
    temperature: float
    eval_every: int = 1
    augment_targets: bool = False
    duration_loss_weight: float = 0.25
    decoded_xy_weight: float = 8.0
    decoded_path_weight: float = 1.50
    decoded_pen_weight: float = 0.75
    decoded_pen_pos_weight: float = 8.0
    decoded_curvature_weight: float = 0.10
    endpoint_latent_weight: float = 0.25
    semantic_weight: float = 0.05
    semantic_every: int = 4
    eval_samples: int = 12
    eval_flow_steps: int = 16
    decoded_jump_weight: float = 1.0


@dataclass(frozen=True)
class RecognizerConfig:
    hidden_dim: int
    layers: int
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    grad_accum_steps: int
    max_grad_norm: float
    checkpoint_every: int
    eval_every: int = 1


@dataclass(frozen=True)
class TrajectoryGeneratorConfig:
    hidden_dim: int = 256
    text_dim: int = 192
    layers: int = 8
    n_heads: int = 8
    dropout: float = 0.10
    learning_rate: float = 0.0003
    weight_decay: float = 0.00001
    epochs: int = 200
    grad_accum_steps: int = 8
    max_grad_norm: float = 1.0
    checkpoint_every: int = 40
    eval_every: int = 5
    eval_samples: int = 12
    augment_targets: bool = False
    xy_weight: float = 1.0
    jump_weight: float = 1.0
    path_weight: float = 0.5
    pen_weight: float = 1.0
    pen_pos_weight: float = 8.0
    curvature_weight: float = 0.05
    length_loss_weight: float = 0.10
    semantic_weight: float = 0.20
    semantic_every: int = 1
    temperature: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    run: RunConfig
    hardware: HardwareConfig
    data: DataConfig
    autoencoder: AutoencoderConfig
    generator: GeneratorConfig
    recognizer: RecognizerConfig
    trajectory_generator: TrajectoryGeneratorConfig = TrajectoryGeneratorConfig()


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise KeyError(f"Missing [{name}] section in config")
    return section


def _path(value: str | Path) -> Path:
    return Path(value).expanduser()


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("rb") as fh:
        raw = tomllib.load(fh)

    run = _section(raw, "run")
    hardware = _section(raw, "hardware")
    data = _section(raw, "data")
    autoencoder = _section(raw, "autoencoder")
    generator = _section(raw, "generator")
    recognizer = _section(raw, "recognizer")
    trajectory_generator_raw = raw.get("trajectory_generator")
    trajectory_generator = (
        TrajectoryGeneratorConfig(**trajectory_generator_raw)
        if isinstance(trajectory_generator_raw, dict)
        else TrajectoryGeneratorConfig()
    )

    return ExperimentConfig(
        run=RunConfig(
            name=str(run.get("name", config_path.stem)),
            seed=int(run.get("seed", 42)),
            out_dir=_path(run.get("out_dir", f"runs/{config_path.stem}")),
        ),
        hardware=HardwareConfig(
            device=str(hardware.get("device", "auto")),
            num_workers=int(hardware.get("num_workers", 0)),
            pin_memory=bool(hardware.get("pin_memory", False)),
            amp=bool(hardware.get("amp", False)),
            persistent_workers=bool(hardware.get("persistent_workers", False)),
            prefetch_factor=(
                None
                if hardware.get("prefetch_factor") in (None, 0)
                else int(hardware["prefetch_factor"])
            ),
            torch_compile=bool(hardware.get("torch_compile", False)),
            matmul_precision=hardware.get("matmul_precision"),
        ),
        data=DataConfig(
            dataset_npz=_path(data["dataset_npz"]),
            val_fraction=float(data.get("val_fraction", 0.1)),
            split_seed=int(data.get("split_seed", 42)),
            resample_spacing=(
                None if data.get("resample_spacing") in (None, 0) else float(data["resample_spacing"])
            ),
            smooth_window=int(data.get("smooth_window", 1)),
            max_points=None if data.get("max_points") in (None, 0) else int(data["max_points"]),
            max_text_len=None if data.get("max_text_len") in (None, 0) else int(data["max_text_len"]),
            batch_size=int(data.get("batch_size", 2)),
            augment=bool(data.get("augment", False)),
            jitter_std=float(data.get("jitter_std", 0.0)),
            scale_jitter=float(data.get("scale_jitter", 0.0)),
        ),
        autoencoder=AutoencoderConfig(**autoencoder),
        generator=GeneratorConfig(**generator),
        recognizer=RecognizerConfig(**recognizer),
        trajectory_generator=trajectory_generator,
    )
