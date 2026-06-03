from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from handwriting_ai.config import ExperimentConfig


def to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return to_plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def configure_torch(config: ExperimentConfig) -> None:
    precision = config.hardware.matmul_precision
    if precision:
        torch.set_float32_matmul_precision(precision)


def maybe_compile(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    if enabled and hasattr(torch, "compile"):
        return torch.compile(model)
    return model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def model_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return unwrap_model(model).state_dict()


def average_metric_dicts(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {key: float(sum(item[key] for item in metrics) / len(metrics)) for key in keys}


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(to_plain(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def optimizer_step(
    *,
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    max_grad_norm: float,
) -> None:
    scaler.unscale_(optimizer)
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
