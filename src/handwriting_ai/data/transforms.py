from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class StrokeRange:
    start: int
    end: int


def validate_points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected points shaped as (T, 3+), got {arr.shape}")
    arr = arr[:, :3].copy()
    arr[:, 2] = (arr[:, 2] > 0.5).astype(np.float32)
    if len(arr):
        arr[-1, 2] = 1.0
    return arr


def split_strokes(points: np.ndarray) -> list[StrokeRange]:
    arr = validate_points(points)
    ranges: list[StrokeRange] = []
    start = 0
    for i, point in enumerate(arr):
        if point[2] > 0.5:
            ranges.append(StrokeRange(start=start, end=i + 1))
            start = i + 1
    if start < len(arr):
        ranges.append(StrokeRange(start=start, end=len(arr)))
    return [r for r in ranges if r.end > r.start]


def smooth_strokes(points: np.ndarray, window: int) -> np.ndarray:
    arr = validate_points(points)
    if window <= 1 or len(arr) < 3:
        return arr
    if window % 2 == 0:
        window += 1
    radius = window // 2
    result = arr.copy()
    kernel = np.ones(window, dtype=np.float32) / float(window)
    for stroke in split_strokes(arr):
        segment = arr[stroke.start : stroke.end, :2]
        if len(segment) <= radius * 2:
            continue
        padded = np.pad(segment, ((radius, radius), (0, 0)), mode="edge")
        result[stroke.start : stroke.end, 0] = np.convolve(padded[:, 0], kernel, mode="valid")
        result[stroke.start : stroke.end, 1] = np.convolve(padded[:, 1], kernel, mode="valid")
        result[stroke.end - 1, 2] = 1.0
    return result


def _resample_polyline(xy: np.ndarray, spacing: float) -> np.ndarray:
    if len(xy) <= 1:
        return xy.astype(np.float32, copy=True)
    deltas = np.diff(xy, axis=0)
    distances = np.sqrt((deltas**2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    total = float(cumulative[-1])
    if total <= 1e-6:
        return xy[[0]].astype(np.float32, copy=True)
    steps = max(2, int(np.ceil(total / spacing)) + 1)
    targets = np.linspace(0.0, total, steps, dtype=np.float32)
    x = np.interp(targets, cumulative, xy[:, 0])
    y = np.interp(targets, cumulative, xy[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def resample_strokes(points: np.ndarray, spacing: float | None) -> np.ndarray:
    arr = validate_points(points)
    if spacing is None or spacing <= 0.0 or len(arr) < 2:
        return arr
    segments: list[np.ndarray] = []
    for stroke in split_strokes(arr):
        xy = arr[stroke.start : stroke.end, :2]
        sampled = _resample_polyline(xy, spacing)
        flags = np.zeros((len(sampled), 1), dtype=np.float32)
        flags[-1, 0] = 1.0
        segments.append(np.concatenate([sampled, flags], axis=1))
    if not segments:
        return arr
    return np.concatenate(segments, axis=0).astype(np.float32)


# Channel layout for delta tensors:
#   0: dx (within-stroke delta, zero at the first frame of a new stroke)
#   1: dy (within-stroke delta, zero at the first frame of a new stroke)
#   2: pen_up flag
#   3: dx_jump (zero except at the first frame of a new stroke)
#   4: dy_jump (zero except at the first frame of a new stroke)
DELTA_CHANNELS = 5


def points_to_deltas(
    points: np.ndarray,
    within_mean: np.ndarray | None = None,
    within_std: np.ndarray | None = None,
    jump_mean: np.ndarray | None = None,
    jump_std: np.ndarray | None = None,
) -> np.ndarray:
    arr = validate_points(points)
    deltas = np.zeros((len(arr), DELTA_CHANNELS), dtype=np.float32)
    if len(arr):
        deltas[:, 2] = arr[:, 2]
        if len(arr) > 1:
            raw = arr[1:, :2] - arr[:-1, :2]
            prev_pen_up = arr[:-1, 2] > 0.5
            # Within-stroke deltas go to channels 0/1, jumps go to channels 3/4.
            # This separates the bimodal distribution (tiny resampled steps vs.
            # canvas-spanning jumps) so each channel has well-behaved stats.
            deltas[1:, 0] = np.where(prev_pen_up, 0.0, raw[:, 0])
            deltas[1:, 1] = np.where(prev_pen_up, 0.0, raw[:, 1])
            deltas[1:, 3] = np.where(prev_pen_up, raw[:, 0], 0.0)
            deltas[1:, 4] = np.where(prev_pen_up, raw[:, 1], 0.0)
    if within_mean is not None and within_std is not None:
        deltas[:, 0:2] = (deltas[:, 0:2] - within_mean.reshape(1, 2)) / within_std.reshape(1, 2)
    if jump_mean is not None and jump_std is not None:
        deltas[:, 3:5] = (deltas[:, 3:5] - jump_mean.reshape(1, 2)) / jump_std.reshape(1, 2)
    return deltas


def deltas_to_points(
    deltas: np.ndarray,
    within_mean: np.ndarray | None = None,
    within_std: np.ndarray | None = None,
    jump_mean: np.ndarray | None = None,
    jump_std: np.ndarray | None = None,
) -> np.ndarray:
    arr = np.asarray(deltas, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected deltas shaped as (T, 3+), got {arr.shape}")
    restored = np.zeros((len(arr), DELTA_CHANNELS), dtype=np.float32)
    restored[:, : arr.shape[1]] = arr[:, : DELTA_CHANNELS]
    if within_mean is not None and within_std is not None:
        restored[:, 0:2] = restored[:, 0:2] * within_std.reshape(1, 2) + within_mean.reshape(1, 2)
    if jump_mean is not None and jump_std is not None and arr.shape[1] >= 5:
        restored[:, 3:5] = restored[:, 3:5] * jump_std.reshape(1, 2) + jump_mean.reshape(1, 2)
    pen_up = (restored[:, 2] > 0.5).astype(np.float32)
    if len(restored) > 1:
        prev_pen_up = pen_up[:-1] > 0.5
        # Only the right channel is allowed to advance the pen at each frame.
        restored[1:, 0] = np.where(prev_pen_up, 0.0, restored[1:, 0])
        restored[1:, 1] = np.where(prev_pen_up, 0.0, restored[1:, 1])
        restored[1:, 3] = np.where(prev_pen_up, restored[1:, 3], 0.0)
        restored[1:, 4] = np.where(prev_pen_up, restored[1:, 4], 0.0)
    combined_x = restored[:, 0] + restored[:, 3]
    combined_y = restored[:, 1] + restored[:, 4]
    points = np.zeros((len(restored), 3), dtype=np.float32)
    points[:, 0] = np.cumsum(combined_x, axis=0)
    points[:, 1] = np.cumsum(combined_y, axis=0)
    points[:, 2] = pen_up
    if len(points):
        points[-1, 2] = 1.0
    return points


def deltas_to_absolute_torch(deltas: torch.Tensor) -> torch.Tensor:
    if deltas.shape[-1] >= 5:
        combined = deltas[..., 0:2] + deltas[..., 3:5]
    else:
        combined = deltas[..., 0:2]
    xy = torch.cumsum(combined, dim=1)
    return torch.cat([xy, deltas[..., 2:3]], dim=-1)


def pad_to_multiple(array: np.ndarray, multiple: int, value: float = 0.0) -> np.ndarray:
    if multiple <= 1:
        return array
    pad = (-len(array)) % multiple
    if pad == 0:
        return array
    padding = np.full((pad, array.shape[1]), value, dtype=array.dtype)
    return np.concatenate([array, padding], axis=0)

