from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from handwriting_ai.models.autoencoder import AutoencoderOutput
from handwriting_ai.models.positional import lengths_to_mask


def _power_of_two_layers(downsample_factor: int) -> int:
    if downsample_factor < 1 or downsample_factor & (downsample_factor - 1):
        raise ValueError("downsample_factor must be a power of two")
    return int(math.log2(downsample_factor))


class LocalResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("local_kernel_size must be odd")
        self.norm = nn.GroupNorm(1, hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
        )
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1)
        self.output = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.depthwise(self.norm(values))
        hidden = F.glu(self.pointwise(hidden), dim=1)
        hidden = self.output(hidden)
        return (values + self.dropout(hidden)) * mask.unsqueeze(1)


class LocalInkAutoencoder(nn.Module):
    """Deterministic local autoencoder for content-preserving ink latents.

    The bottleneck is convolutional and has a bounded receptive field. A latent
    frame therefore describes a local piece of ink instead of mixing the whole
    line, which makes character-level forced alignment meaningful.
    """

    def __init__(
        self,
        *,
        input_dim: int = 3,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        downsample_factor: int = 4,
        bottleneck_layers: int = 4,
        local_kernel_size: int = 9,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.downsample_factor = downsample_factor
        self.local_kernel_size = local_kernel_size
        n_down = _power_of_two_layers(downsample_factor)

        encoder: list[nn.Module] = []
        in_channels = input_dim
        for _ in range(n_down):
            encoder.extend(
                [
                    nn.Conv1d(in_channels, hidden_dim, kernel_size=5, stride=2, padding=2),
                    nn.GroupNorm(min(8, hidden_dim), hidden_dim),
                    nn.SiLU(),
                ]
            )
            in_channels = hidden_dim
        self.encoder_conv = nn.Sequential(*encoder)
        self.bottleneck = nn.ModuleList(
            [
                LocalResidualBlock(hidden_dim, local_kernel_size, dropout)
                for _ in range(bottleneck_layers)
            ]
        )
        self.to_latent = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, latent_dim, kernel_size=1),
        )
        self.from_latent = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
        )

        decoder: list[nn.Module] = []
        for _ in range(n_down):
            decoder.extend(
                [
                    nn.ConvTranspose1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.GroupNorm(min(8, hidden_dim), hidden_dim),
                    nn.SiLU(),
                ]
            )
        decoder.append(nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1))
        self.decoder_conv = nn.Sequential(*decoder)

    def _pad_input(self, points: torch.Tensor) -> torch.Tensor:
        pad = (-points.shape[1]) % self.downsample_factor
        if pad:
            points = F.pad(points, (0, 0, 0, pad), value=0.0)
        return points

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
        sample: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del sample
        padded = self._pad_input(points)
        hidden = self.encoder_conv(padded.transpose(1, 2))
        latent_lengths = self.latent_lengths(point_lengths).clamp(max=hidden.shape[2])
        latent_mask = lengths_to_mask(latent_lengths, hidden.shape[2])
        for block in self.bottleneck:
            hidden = block(hidden, latent_mask)
        latent = self.to_latent(hidden).transpose(1, 2)
        latent = latent * latent_mask.unsqueeze(-1)
        mu = torch.zeros_like(latent)
        logvar = torch.zeros_like(latent)
        return latent, mu, logvar, latent_mask, latent_lengths

    def decode(self, latent: torch.Tensor, target_length: int | None = None) -> torch.Tensor:
        hidden = self.from_latent(latent.transpose(1, 2))
        reconstruction = self.decoder_conv(hidden).transpose(1, 2)
        if target_length is not None:
            reconstruction = reconstruction[:, :target_length]
        return reconstruction

    def forward(self, points: torch.Tensor, point_lengths: torch.Tensor) -> AutoencoderOutput:
        latent, mu, logvar, latent_mask, latent_lengths = self.encode(
            points,
            point_lengths,
            sample=False,
        )
        reconstruction = self.decode(latent, target_length=points.shape[1])
        return AutoencoderOutput(
            reconstruction=reconstruction,
            mu=mu,
            logvar=logvar,
            latent=latent,
            latent_mask=latent_mask,
            latent_lengths=latent_lengths,
        )
