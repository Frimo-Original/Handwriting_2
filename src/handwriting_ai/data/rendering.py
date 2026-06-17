from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from handwriting_ai.data.transforms import deltas_to_points, validate_points


def render_points_to_image(
    points: np.ndarray,
    *,
    size: tuple[int, int] = (960, 360),
    margin: int = 24,
    stroke_width: int = 2,
    background: str = "white",
    foreground: str = "black",
) -> Image.Image:
    arr = validate_points(points)
    image = Image.new("RGB", size, background)
    if len(arr) < 2:
        return image

    xy = arr[:, :2]
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)
    scale = min((size[0] - 2 * margin) / span[0], (size[1] - 2 * margin) / span[1])
    scaled = (xy - min_xy.reshape(1, 2)) * scale + margin

    draw = ImageDraw.Draw(image)
    for i in range(1, len(arr)):
        if arr[i - 1, 2] > 0.5:
            continue
        p0 = tuple(float(v) for v in scaled[i - 1])
        p1 = tuple(float(v) for v in scaled[i])
        draw.line([p0, p1], fill=foreground, width=stroke_width)
    return image


def render_deltas_to_image(
    deltas: np.ndarray,
    *,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    jump_mean: np.ndarray | None = None,
    jump_std: np.ndarray | None = None,
    size: tuple[int, int] = (960, 360),
) -> Image.Image:
    points = deltas_to_points(
        deltas,
        within_mean=mean,
        within_std=std,
        jump_mean=jump_mean,
        jump_std=jump_std,
    )
    return render_points_to_image(points, size=size)


def save_render(path: str | Path, points: np.ndarray, size: tuple[int, int] = (960, 360)) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_points_to_image(points, size=size).save(output_path)

