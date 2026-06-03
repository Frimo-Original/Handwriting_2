from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from handwriting_ai.config import DataConfig, HardwareConfig
from handwriting_ai.data.codec import PAD_ID, VOCAB_TOKENS
from handwriting_ai.data.transforms import points_to_deltas, resample_strokes, smooth_strokes, validate_points


@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NormalizationStats":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )


@dataclass(frozen=True)
class ProcessedSample:
    points: np.ndarray
    text: np.ndarray
    sample_id: int


@dataclass(frozen=True)
class Batch:
    points: torch.Tensor
    point_mask: torch.Tensor
    point_lengths: torch.Tensor
    text: torch.Tensor
    text_mask: torch.Tensor
    text_lengths: torch.Tensor
    sample_ids: torch.Tensor

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            points=self.points.to(device),
            point_mask=self.point_mask.to(device),
            point_lengths=self.point_lengths.to(device),
            text=self.text.to(device),
            text_mask=self.text_mask.to(device),
            text_lengths=self.text_lengths.to(device),
            sample_ids=self.sample_ids.to(device),
        )


def load_raw_npz(path: str | Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    data = np.load(Path(path), allow_pickle=True)
    points = [validate_points(item) for item in data["points"]]
    texts = [np.asarray(item, dtype=np.int64).reshape(-1) for item in data["text_indices"]]
    if len(points) != len(texts):
        raise ValueError(f"Dataset has {len(points)} point arrays and {len(texts)} texts")
    return points, texts


def split_indices(n_items: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_items)
    rng.shuffle(indices)
    val_count = max(1, int(round(n_items * val_fraction))) if n_items > 1 else 0
    val = np.sort(indices[:val_count])
    train = np.sort(indices[val_count:])
    return train, val


def preprocess_samples(
    points: list[np.ndarray],
    texts: list[np.ndarray],
    indices: np.ndarray,
    data_config: DataConfig,
) -> list[ProcessedSample]:
    samples: list[ProcessedSample] = []
    for idx in indices:
        text = texts[int(idx)]
        if data_config.max_text_len is not None and len(text) > data_config.max_text_len:
            continue
        pts = validate_points(points[int(idx)])
        pts = smooth_strokes(pts, data_config.smooth_window)
        pts = resample_strokes(pts, data_config.resample_spacing)
        if data_config.max_points is not None and len(pts) > data_config.max_points:
            continue
        if len(pts) < 4:
            continue
        samples.append(ProcessedSample(points=pts, text=text, sample_id=int(idx)))
    if not samples:
        raise ValueError("No usable samples after preprocessing/filtering")
    return samples


def compute_normalization_stats(samples: list[ProcessedSample]) -> NormalizationStats:
    deltas = [points_to_deltas(sample.points)[:, :2] for sample in samples]
    merged = np.concatenate(deltas, axis=0)
    mean = merged.mean(axis=0).astype(np.float32)
    std = merged.std(axis=0).astype(np.float32)
    std = np.maximum(std, np.asarray([1e-3, 1e-3], dtype=np.float32))
    return NormalizationStats(mean=mean, std=std)


class TrajectoryDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        samples: list[ProcessedSample],
        stats: NormalizationStats,
        *,
        augment: bool = False,
        jitter_std: float = 0.0,
        scale_jitter: float = 0.0,
    ) -> None:
        self.samples = samples
        self.stats = stats
        self.augment = augment
        self.jitter_std = jitter_std
        self.scale_jitter = scale_jitter

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        points = sample.points.copy()
        if self.augment:
            if self.scale_jitter > 0.0:
                scale = float(np.random.normal(1.0, self.scale_jitter))
                points[:, :2] *= max(scale, 0.5)
            if self.jitter_std > 0.0:
                points[:, :2] += np.random.normal(0.0, self.jitter_std, size=points[:, :2].shape)
        deltas = points_to_deltas(points, mean=self.stats.mean, std=self.stats.std)
        return {
            "points": torch.from_numpy(deltas.astype(np.float32)),
            "text": torch.from_numpy(sample.text.astype(np.int64)),
            "sample_id": sample.sample_id,
        }


def collate_batch(items: list[dict[str, Any]]) -> Batch:
    point_lengths = torch.tensor([item["points"].shape[0] for item in items], dtype=torch.long)
    text_lengths = torch.tensor([item["text"].shape[0] for item in items], dtype=torch.long)
    max_points = int(point_lengths.max().item())
    max_text = int(text_lengths.max().item())
    batch_size = len(items)

    points = torch.zeros(batch_size, max_points, 3, dtype=torch.float32)
    point_mask = torch.zeros(batch_size, max_points, dtype=torch.bool)
    text = torch.full((batch_size, max_text), PAD_ID, dtype=torch.long)
    text_mask = torch.zeros(batch_size, max_text, dtype=torch.bool)
    sample_ids = torch.empty(batch_size, dtype=torch.long)

    for row, item in enumerate(items):
        point_len = item["points"].shape[0]
        text_len = item["text"].shape[0]
        points[row, :point_len] = item["points"]
        point_mask[row, :point_len] = True
        text[row, :text_len] = item["text"]
        text_mask[row, :text_len] = True
        sample_ids[row] = int(item["sample_id"])

    return Batch(
        points=points,
        point_mask=point_mask,
        point_lengths=point_lengths,
        text=text,
        text_mask=text_mask,
        text_lengths=text_lengths,
        sample_ids=sample_ids,
    )


def build_datasets(
    data_config: DataConfig,
    stats: NormalizationStats | None = None,
) -> tuple[TrajectoryDataset, TrajectoryDataset, NormalizationStats]:
    raw_points, raw_texts = load_raw_npz(data_config.dataset_npz)
    train_idx, val_idx = split_indices(len(raw_points), data_config.val_fraction, data_config.split_seed)
    train_samples = preprocess_samples(raw_points, raw_texts, train_idx, data_config)
    val_samples = preprocess_samples(raw_points, raw_texts, val_idx, data_config)
    if stats is None:
        stats = compute_normalization_stats(train_samples)
    train_dataset = TrajectoryDataset(
        train_samples,
        stats,
        augment=data_config.augment,
        jitter_std=data_config.jitter_std,
        scale_jitter=data_config.scale_jitter,
    )
    val_dataset = TrajectoryDataset(val_samples, stats, augment=False)
    return train_dataset, val_dataset, stats


def build_dataloaders(
    data_config: DataConfig,
    hardware_config: HardwareConfig,
    stats: NormalizationStats | None = None,
) -> tuple[DataLoader[Batch], DataLoader[Batch], NormalizationStats]:
    train_dataset, val_dataset, stats = build_datasets(data_config, stats=stats)
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=hardware_config.num_workers,
        pin_memory=hardware_config.pin_memory,
        collate_fn=collate_batch,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=hardware_config.num_workers,
        pin_memory=hardware_config.pin_memory,
        collate_fn=collate_batch,
        drop_last=False,
    )
    return train_loader, val_loader, stats


def dataset_summary(data_config: DataConfig) -> dict[str, Any]:
    raw_points, raw_texts = load_raw_npz(data_config.dataset_npz)
    train_idx, val_idx = split_indices(len(raw_points), data_config.val_fraction, data_config.split_seed)
    train_samples = preprocess_samples(raw_points, raw_texts, train_idx, data_config)
    val_samples = preprocess_samples(raw_points, raw_texts, val_idx, data_config)
    lengths = np.asarray([len(sample.points) for sample in train_samples + val_samples])
    text_lengths = np.asarray([len(sample.text) for sample in train_samples + val_samples])
    return {
        "raw_examples": len(raw_points),
        "train_examples": len(train_samples),
        "val_examples": len(val_samples),
        "points_min": int(lengths.min()),
        "points_median": float(np.median(lengths)),
        "points_max": int(lengths.max()),
        "text_min": int(text_lengths.min()),
        "text_median": float(np.median(text_lengths)),
        "text_max": int(text_lengths.max()),
        "vocab_size": len(VOCAB_TOKENS),
        "data_config": asdict(data_config),
    }
