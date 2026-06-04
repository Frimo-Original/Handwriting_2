from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from handwriting_ai.latent_stats import LatentNormalizationStats, denormalize_latents
from handwriting_ai.models.autoencoder import AutoencoderOutput
from handwriting_ai.models.flow import LatentFlowTransformer
from handwriting_ai.models.latent_regressor import LatentRegressorTransformer
from handwriting_ai.models.trajectory_generator import TrajectoryGenerator


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return (values * mask).sum() / mask.sum().clamp(min=1)


def masked_std(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = values[mask]
    if valid.shape[0] < 2:
        return values.new_zeros(values.shape[-1])
    return valid.std(dim=0, unbiased=False)


def curvature_loss(pred_xy: torch.Tensor, target_xy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred_xy.shape[1] < 3:
        return pred_xy.new_tensor(0.0)
    pred_second = pred_xy[:, 2:] - 2 * pred_xy[:, 1:-1] + pred_xy[:, :-2]
    target_second = target_xy[:, 2:] - 2 * target_xy[:, 1:-1] + target_xy[:, :-2]
    second_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
    return masked_mean(F.smooth_l1_loss(pred_second, target_second, reduction="none"), second_mask)


def cumulative_xy_loss(pred_xy: torch.Tensor, target_xy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_path = torch.cumsum(pred_xy, dim=1)
    target_path = torch.cumsum(target_xy, dim=1)
    lengths = mask.sum(dim=1).clamp(min=1).to(pred_xy.dtype).view(-1, 1, 1)
    scale = torch.sqrt(lengths).clamp(min=1.0)
    return masked_mean(
        F.smooth_l1_loss(pred_path / scale, target_path / scale, reduction="none"),
        mask,
    )


def per_sample_delta_std_loss(pred_xy: torch.Tensor, target_xy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.unsqueeze(-1)
    lengths = expanded.sum(dim=1).clamp(min=1).to(pred_xy.dtype)
    pred_mean = (pred_xy * expanded).sum(dim=1) / lengths
    target_mean = (target_xy * expanded).sum(dim=1) / lengths
    pred_var = (((pred_xy - pred_mean.unsqueeze(1)) * expanded) ** 2).sum(dim=1) / lengths
    target_var = (((target_xy - target_mean.unsqueeze(1)) * expanded) ** 2).sum(dim=1) / lengths
    pred_std = torch.sqrt(pred_var.clamp(min=1e-8))
    target_std = torch.sqrt(target_var.clamp(min=1e-8))
    return F.smooth_l1_loss(torch.log1p(pred_std), torch.log1p(target_std))


def path_bbox_loss(pred_xy: torch.Tensor, target_xy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_path = torch.cumsum(pred_xy, dim=1)
    target_path = torch.cumsum(target_xy, dim=1)
    expanded = mask.unsqueeze(-1)
    inf = torch.finfo(pred_xy.dtype).max
    pred_min = pred_path.masked_fill(~expanded, inf).min(dim=1).values
    pred_max = pred_path.masked_fill(~expanded, -inf).max(dim=1).values
    target_min = target_path.masked_fill(~expanded, inf).min(dim=1).values
    target_max = target_path.masked_fill(~expanded, -inf).max(dim=1).values
    pred_range = (pred_max - pred_min).clamp(min=1e-6)
    target_range = (target_max - target_min).clamp(min=1e-6)
    return F.smooth_l1_loss(torch.log1p(pred_range), torch.log1p(target_range))


def pen_dice_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits) * mask
    target = target * mask
    intersection = (probs * target).sum(dim=1)
    denom = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + 1.0) / (denom + 1.0)
    return (1.0 - dice).mean()


def _soft_segment_raster(
    xy: torch.Tensor,
    visible: torch.Tensor,
    mask: torch.Tensor,
    reference_xy: torch.Tensor,
    *,
    image_size: int = 48,
    sigma: float = 0.035,
    max_segments: int = 384,
) -> torch.Tensor:
    if xy.shape[1] < 2:
        return xy.new_zeros((xy.shape[0], image_size, image_size))
    device = xy.device
    coords = torch.linspace(0.0, 1.0, image_size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1).view(1, 1, image_size, image_size, 2)
    rasters = []
    segment_mask = mask[:, 1:] & mask[:, :-1]
    segment_xy = 0.5 * (xy[:, 1:] + xy[:, :-1])
    reference_mask = mask.unsqueeze(-1)
    inf = torch.finfo(xy.dtype).max
    ref_min = reference_xy.masked_fill(~reference_mask, inf).min(dim=1).values
    ref_max = reference_xy.masked_fill(~reference_mask, -inf).max(dim=1).values
    ref_span = (ref_max - ref_min).clamp(min=1e-3)

    for row in range(xy.shape[0]):
        valid_mask = segment_mask[row]
        valid = segment_xy[row, valid_mask]
        weights = visible[row, :-1][valid_mask].clamp(0.0, 1.0)
        if valid.numel() == 0 or weights.sum() <= 1e-6:
            rasters.append(torch.zeros(image_size, image_size, device=device, dtype=xy.dtype))
            continue
        if len(valid) > max_segments:
            idx = torch.linspace(0, len(valid) - 1, max_segments, device=device).long()
            valid = valid[idx]
            weights = weights[idx]
        valid = (valid - ref_min[row].view(1, 2)) / ref_span[row].view(1, 2)
        diff = grid - valid.view(1, -1, 1, 1, 2)
        dist2 = (diff * diff).sum(dim=-1)
        heat = torch.exp(-dist2 / (2 * sigma * sigma)) * weights.view(1, -1, 1, 1)
        rasters.append(heat.amax(dim=1).squeeze(0))
    return torch.stack(rasters, dim=0)


def pen_aware_render_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_xy = torch.cumsum(pred[..., :2], dim=1)
    target_xy = torch.cumsum(target[..., :2], dim=1)
    pred_visible = 1.0 - torch.sigmoid(pred[..., 2])
    target_visible = 1.0 - target[..., 2].clamp(0.0, 1.0)
    pred_image = _soft_segment_raster(pred_xy, pred_visible, mask, reference_xy=target_xy)
    target_image = _soft_segment_raster(target_xy, target_visible, mask, reference_xy=target_xy)
    return F.l1_loss(pred_image, target_image)


def _soft_raster(points_xy: torch.Tensor, mask: torch.Tensor, image_size: int = 64, sigma: float = 0.025) -> torch.Tensor:
    batch, _, _ = points_xy.shape
    device = points_xy.device
    coords = torch.linspace(0.0, 1.0, image_size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1).view(1, 1, image_size, image_size, 2)
    rasters = []
    for row in range(batch):
        valid = points_xy[row, mask[row]]
        if valid.numel() == 0:
            rasters.append(torch.zeros(image_size, image_size, device=device, dtype=points_xy.dtype))
            continue
        if len(valid) > 256:
            idx = torch.linspace(0, len(valid) - 1, 256, device=device).long()
            valid = valid[idx]
        min_xy = valid.min(dim=0).values
        max_xy = valid.max(dim=0).values
        span = (max_xy - min_xy).clamp(min=1e-3)
        valid = (valid - min_xy) / span
        diff = grid - valid.view(1, -1, 1, 1, 2)
        dist2 = (diff * diff).sum(dim=-1)
        heat = torch.exp(-dist2 / (2 * sigma * sigma)).amax(dim=1).squeeze(0)
        rasters.append(heat)
    return torch.stack(rasters, dim=0)


def render_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_xy = torch.cumsum(pred[..., :2], dim=1)
    target_xy = torch.cumsum(target[..., :2], dim=1)
    pred_image = _soft_raster(pred_xy, mask)
    target_image = _soft_raster(target_xy, mask)
    return F.l1_loss(pred_image, target_image)


def autoencoder_loss(
    output: AutoencoderOutput,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    kl_weight: float,
    pen_weight: float,
    curvature_weight: float,
    render_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    recon = output.reconstruction
    xy_loss = masked_mean(F.smooth_l1_loss(recon[..., :2], target[..., :2], reduction="none"), mask)
    pen_loss = masked_mean(
        F.binary_cross_entropy_with_logits(recon[..., 2], target[..., 2], reduction="none"),
        mask,
    )
    curve = curvature_loss(recon[..., :2], target[..., :2], mask)
    kl = -0.5 * (1 + output.logvar - output.mu.pow(2) - output.logvar.exp())
    kl = masked_mean(kl, output.latent_mask)
    raster = recon.new_tensor(0.0)
    if render_weight > 0.0:
        pred_for_render = torch.cat([recon[..., :2], torch.sigmoid(recon[..., 2:3])], dim=-1)
        raster = render_loss(pred_for_render, target, mask)
    loss = xy_loss + pen_weight * pen_loss + curvature_weight * curve + kl_weight * kl + render_weight * raster
    metrics = {
        "loss": float(loss.detach().cpu()),
        "xy": float(xy_loss.detach().cpu()),
        "pen": float(pen_loss.detach().cpu()),
        "curvature": float(curve.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "render": float(raster.detach().cpu()),
    }
    return loss, metrics


def flow_matching_loss(
    model: LatentFlowTransformer,
    latents: torch.Tensor,
    latent_mask: torch.Tensor,
    latent_lengths: torch.Tensor,
    text: torch.Tensor,
    text_mask: torch.Tensor,
    *,
    length_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    noise = torch.randn_like(latents)
    times = torch.rand(latents.shape[0], device=latents.device)
    t = times.view(-1, 1, 1)
    noisy = ((1.0 - t) * noise + t * latents) * latent_mask.unsqueeze(-1)
    target_velocity = (latents - noise) * latent_mask.unsqueeze(-1)
    output = model(noisy, times, latent_mask, text, text_mask)
    velocity_loss = masked_mean(
        F.mse_loss(output.velocity, target_velocity, reduction="none"),
        latent_mask,
    )
    length_target = torch.log1p(latent_lengths.float())
    length_loss = F.mse_loss(output.length_log, length_target)
    loss = velocity_loss + length_loss_weight * length_loss
    return loss, {
        "loss": float(loss.detach().cpu()),
        "velocity": float(velocity_loss.detach().cpu()),
        "length": float(length_loss.detach().cpu()),
    }


def latent_regression_loss(
    model: LatentRegressorTransformer,
    latents: torch.Tensor,
    latent_mask: torch.Tensor,
    latent_lengths: torch.Tensor,
    text: torch.Tensor,
    text_mask: torch.Tensor,
    *,
    length_loss_weight: float,
    autoencoder: nn.Module | None = None,
    latent_normalization: LatentNormalizationStats | None = None,
    target_points: torch.Tensor | None = None,
    point_mask: torch.Tensor | None = None,
    latent_loss_weight: float = 1.0,
    latent_std_weight: float = 0.0,
    decoded_xy_weight: float = 0.0,
    decoded_path_weight: float = 0.0,
    decoded_pen_weight: float = 0.0,
    decoded_pen_pos_weight: float = 1.0,
    decoded_curvature_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(text, text_mask, latent_lengths=latent_lengths)
    latent_loss = masked_mean(
        F.smooth_l1_loss(output.latents, latents, reduction="none"),
        latent_mask,
    )
    std_loss = output.latents.new_tensor(0.0)
    if latent_std_weight > 0.0:
        pred_std = masked_std(output.latents, latent_mask)
        target_std = masked_std(latents, latent_mask)
        std_loss = F.smooth_l1_loss(pred_std, target_std)

    decoded_xy = output.latents.new_tensor(0.0)
    decoded_path = output.latents.new_tensor(0.0)
    decoded_pen = output.latents.new_tensor(0.0)
    decoded_curve = output.latents.new_tensor(0.0)
    needs_decoded = (
        decoded_xy_weight > 0.0
        or decoded_path_weight > 0.0
        or decoded_pen_weight > 0.0
        or decoded_curvature_weight > 0.0
    )
    if needs_decoded:
        if autoencoder is None or target_points is None or point_mask is None:
            raise ValueError("Decoded generator loss requires autoencoder, target_points and point_mask")
        decoder_latents = output.latents
        if latent_normalization is not None:
            decoder_latents = denormalize_latents(decoder_latents, latent_normalization)
        decoder_latents = decoder_latents * output.latent_mask.unsqueeze(-1)
        decoded = autoencoder.decode(decoder_latents, target_length=target_points.shape[1])
        decoded_xy = masked_mean(
            F.smooth_l1_loss(decoded[..., :2], target_points[..., :2], reduction="none"),
            point_mask,
        )
        decoded_path = cumulative_xy_loss(decoded[..., :2], target_points[..., :2], point_mask)
        pos_weight = decoded.new_tensor(float(decoded_pen_pos_weight))
        decoded_pen = masked_mean(
            F.binary_cross_entropy_with_logits(
                decoded[..., 2],
                target_points[..., 2],
                pos_weight=pos_weight,
                reduction="none",
            ),
            point_mask,
        )
        decoded_curve = curvature_loss(decoded[..., :2], target_points[..., :2], point_mask)

    length_target = torch.log1p(latent_lengths.float())
    length_loss = F.mse_loss(output.length_log, length_target)
    loss = (
        latent_loss_weight * latent_loss
        + latent_std_weight * std_loss
        + decoded_xy_weight * decoded_xy
        + decoded_path_weight * decoded_path
        + decoded_pen_weight * decoded_pen
        + decoded_curvature_weight * decoded_curve
        + length_loss_weight * length_loss
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "latent": float(latent_loss.detach().cpu()),
        "latent_std": float(std_loss.detach().cpu()),
        "decoded_xy": float(decoded_xy.detach().cpu()),
        "decoded_path": float(decoded_path.detach().cpu()),
        "decoded_pen": float(decoded_pen.detach().cpu()),
        "decoded_curvature": float(decoded_curve.detach().cpu()),
        "length": float(length_loss.detach().cpu()),
    }


def trajectory_generator_loss(
    model: TrajectoryGenerator,
    points: torch.Tensor,
    point_mask: torch.Tensor,
    point_lengths: torch.Tensor,
    text: torch.Tensor,
    text_mask: torch.Tensor,
    *,
    length_loss_weight: float,
    xy_weight: float,
    path_weight: float,
    pen_weight: float,
    pen_pos_weight: float,
    curvature_weight: float,
    render_weight: float = 0.0,
    std_weight: float = 0.0,
    bbox_weight: float = 0.0,
    pen_dice_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(text, text_mask, point_lengths=point_lengths)
    pred = output.deltas
    xy = masked_mean(
        F.smooth_l1_loss(pred[..., :2], points[..., :2], reduction="none"),
        point_mask,
    )
    path = cumulative_xy_loss(pred[..., :2], points[..., :2], point_mask)
    pos_weight = pred.new_tensor(float(pen_pos_weight))
    pen = masked_mean(
        F.binary_cross_entropy_with_logits(
            pred[..., 2],
            points[..., 2],
            pos_weight=pos_weight,
            reduction="none",
        ),
        point_mask,
    )
    curve = curvature_loss(pred[..., :2], points[..., :2], point_mask)
    render = (
        pen_aware_render_loss(pred, points, point_mask)
        if render_weight > 0.0
        else pred.new_tensor(0.0)
    )
    delta_std = (
        per_sample_delta_std_loss(pred[..., :2], points[..., :2], point_mask)
        if std_weight > 0.0
        else pred.new_tensor(0.0)
    )
    bbox = (
        path_bbox_loss(pred[..., :2], points[..., :2], point_mask)
        if bbox_weight > 0.0
        else pred.new_tensor(0.0)
    )
    pen_dice = (
        pen_dice_loss(pred[..., 2], points[..., 2], point_mask)
        if pen_dice_weight > 0.0
        else pred.new_tensor(0.0)
    )
    length_target = torch.log1p(point_lengths.float())
    length = F.mse_loss(output.length_log, length_target)
    loss = (
        xy_weight * xy
        + path_weight * path
        + pen_weight * pen
        + curvature_weight * curve
        + render_weight * render
        + std_weight * delta_std
        + bbox_weight * bbox
        + pen_dice_weight * pen_dice
        + length_loss_weight * length
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "xy": float(xy.detach().cpu()),
        "path": float(path.detach().cpu()),
        "pen": float(pen.detach().cpu()),
        "curvature": float(curve.detach().cpu()),
        "render": float(render.detach().cpu()),
        "std": float(delta_std.detach().cpu()),
        "bbox": float(bbox.detach().cpu()),
        "pen_dice": float(pen_dice.detach().cpu()),
        "length": float(length.detach().cpu()),
    }


def recognizer_ctc_loss(
    log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
    text: torch.Tensor,
    text_lengths: torch.Tensor,
    *,
    blank_id: int,
) -> torch.Tensor:
    criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    targets = []
    for row in range(text.shape[0]):
        targets.append(text[row, : text_lengths[row]].to(torch.long))
    flat_targets = torch.cat(targets, dim=0)
    return criterion(log_probs, flat_targets, output_lengths, text_lengths)
