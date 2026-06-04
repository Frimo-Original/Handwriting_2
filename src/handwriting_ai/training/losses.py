from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from handwriting_ai.latent_stats import LatentNormalizationStats, denormalize_latents
from handwriting_ai.models.autoencoder import AutoencoderOutput
from handwriting_ai.models.flow import LatentFlowTransformer
from handwriting_ai.models.latent_regressor import LatentRegressorTransformer


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
    decoded_pen_weight: float = 0.0,
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
    decoded_pen = output.latents.new_tensor(0.0)
    decoded_curve = output.latents.new_tensor(0.0)
    needs_decoded = decoded_xy_weight > 0.0 or decoded_pen_weight > 0.0 or decoded_curvature_weight > 0.0
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
        decoded_pen = masked_mean(
            F.binary_cross_entropy_with_logits(decoded[..., 2], target_points[..., 2], reduction="none"),
            point_mask,
        )
        decoded_curve = curvature_loss(decoded[..., :2], target_points[..., :2], point_mask)

    length_target = torch.log1p(latent_lengths.float())
    length_loss = F.mse_loss(output.length_log, length_target)
    loss = (
        latent_loss_weight * latent_loss
        + latent_std_weight * std_loss
        + decoded_xy_weight * decoded_xy
        + decoded_pen_weight * decoded_pen
        + decoded_curvature_weight * decoded_curve
        + length_loss_weight * length_loss
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "latent": float(latent_loss.detach().cpu()),
        "latent_std": float(std_loss.detach().cpu()),
        "decoded_xy": float(decoded_xy.detach().cpu()),
        "decoded_pen": float(decoded_pen.detach().cpu()),
        "decoded_curvature": float(decoded_curve.detach().cpu()),
        "length": float(length_loss.detach().cpu()),
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
