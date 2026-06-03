from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from handwriting_ai.models.positional import SinusoidalPositionalEncoding, lengths_to_mask


@dataclass(frozen=True)
class AutoencoderOutput:
    reconstruction: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    latent: torch.Tensor
    latent_mask: torch.Tensor
    latent_lengths: torch.Tensor


def _power_of_two_layers(downsample_factor: int) -> int:
    if downsample_factor < 1 or downsample_factor & (downsample_factor - 1):
        raise ValueError("downsample_factor must be a power of two")
    return int(math.log2(downsample_factor))


class InkAutoencoder(nn.Module):
    """Compact VAE for online ink sequences.

    The model compresses normalized `(dx, dy, pen_up)` sequences into a shorter latent
    sequence. The generator is trained in this latent space, which keeps long lines
    manageable on consumer GPUs.
    """

    def __init__(
        self,
        *,
        input_dim: int = 3,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        downsample_factor: int = 8,
        bottleneck_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.downsample_factor = downsample_factor
        n_down = _power_of_two_layers(downsample_factor)

        encoder_layers: list[nn.Module] = []
        in_channels = input_dim
        for _ in range(n_down):
            encoder_layers.extend(
                [
                    nn.Conv1d(in_channels, hidden_dim, kernel_size=5, stride=2, padding=2),
                    nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim),
                    nn.SiLU(),
                ]
            )
            in_channels = hidden_dim
        self.encoder_conv = nn.Sequential(*encoder_layers)

        if bottleneck_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.bottleneck = nn.TransformerEncoder(encoder_layer, num_layers=bottleneck_layers)
            self.bottleneck_pos = SinusoidalPositionalEncoding(hidden_dim)
        else:
            self.bottleneck = None
            self.bottleneck_pos = None

        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)

        decoder_layers: list[nn.Module] = []
        for _ in range(n_down):
            decoder_layers.extend(
                [
                    nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim),
                    nn.SiLU(),
                ]
            )
        decoder_layers.append(nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1))
        self.decoder_conv = nn.Sequential(*decoder_layers)

    def _pad_input(self, points: torch.Tensor) -> tuple[torch.Tensor, int]:
        length = points.shape[1]
        pad = (-length) % self.downsample_factor
        if pad:
            points = F.pad(points, (0, 0, 0, pad), value=0.0)
        return points, length

    def latent_lengths(self, point_lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(
            point_lengths + self.downsample_factor - 1,
            self.downsample_factor,
            rounding_mode="floor",
        )

    def encode(
        self,
        points: torch.Tensor,
        point_lengths: torch.Tensor,
        *,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        padded, _ = self._pad_input(points)
        hidden = self.encoder_conv(padded.transpose(1, 2)).transpose(1, 2)
        latent_lengths = self.latent_lengths(point_lengths).clamp(max=hidden.shape[1])
        latent_mask = lengths_to_mask(latent_lengths, hidden.shape[1])
        if self.bottleneck is not None and self.bottleneck_pos is not None:
            hidden = self.bottleneck_pos(hidden)
            hidden = self.bottleneck(hidden, src_key_padding_mask=~latent_mask)
        mu = self.to_mu(hidden)
        logvar = self.to_logvar(hidden).clamp(min=-8.0, max=6.0)
        if sample:
            std = torch.exp(0.5 * logvar)
            latent = mu + torch.randn_like(std) * std
        else:
            latent = mu
        latent = latent * latent_mask.unsqueeze(-1)
        return latent, mu, logvar, latent_mask, latent_lengths

    def decode(self, latent: torch.Tensor, target_length: int | None = None) -> torch.Tensor:
        hidden = self.from_latent(latent).transpose(1, 2)
        reconstruction = self.decoder_conv(hidden).transpose(1, 2)
        if target_length is not None:
            reconstruction = reconstruction[:, :target_length]
        return reconstruction

    def forward(self, points: torch.Tensor, point_lengths: torch.Tensor) -> AutoencoderOutput:
        latent, mu, logvar, latent_mask, latent_lengths = self.encode(points, point_lengths, sample=self.training)
        reconstruction = self.decode(latent, target_length=points.shape[1])
        return AutoencoderOutput(
            reconstruction=reconstruction,
            mu=mu,
            logvar=logvar,
            latent=latent,
            latent_mask=latent_mask,
            latent_lengths=latent_lengths,
        )

    def config_dict(self) -> dict[str, int | float]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "downsample_factor": self.downsample_factor,
        }

