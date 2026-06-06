from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from handwriting_ai.alignment import fit_durations
from handwriting_ai.data.codec import EOS_ID, PAD_ID, VOCAB_SIZE
from handwriting_ai.models.aligned_flow import (
    DurationPredictor,
    durations_to_path,
)
from handwriting_ai.models.flow import TextEncoder
from handwriting_ai.models.positional import SinusoidalPositionalEncoding, timestep_embedding


def content_text_mask(text: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
    return text_mask & (text != PAD_ID) & (text != EOS_ID)


def duration_position_features(
    durations: torch.Tensor,
    text_mask: torch.Tensor,
) -> torch.Tensor:
    lengths = (durations * text_mask).sum(dim=1).long()
    max_length = int(lengths.max().item())
    features = durations.new_zeros(
        durations.shape[0],
        max_length,
        3,
        dtype=torch.float32,
    )
    for row in range(durations.shape[0]):
        cursor = 0
        token_count = int(text_mask[row].sum().item())
        for token in range(token_count):
            duration = int(durations[row, token].item())
            if duration <= 0:
                continue
            fraction = torch.linspace(
                0.0,
                1.0,
                duration,
                device=durations.device,
            )
            features[row, cursor : cursor + duration, 0] = fraction
            features[row, cursor : cursor + duration, 1] = fraction * 2.0 - 1.0
            features[row, cursor : cursor + duration, 2] = torch.log1p(
                durations.new_tensor(float(duration))
            )
            cursor += duration
    features[..., 2] /= 8.0
    return features


@dataclass(frozen=True)
class ContentFlowOutput:
    velocity: torch.Tensor
    log_durations: torch.Tensor
    durations: torch.Tensor
    latent_mask: torch.Tensor


class LocalFlowBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        *,
        kernel_size: int = 7,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.conv_norm = nn.LayerNorm(hidden_dim)
        self.conv_in = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hidden_dim,
        )
        self.conv_out = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        convolved = self.conv_norm(hidden).transpose(1, 2)
        convolved = F.glu(self.conv_in(convolved), dim=1)
        convolved = self.depthwise(convolved)
        convolved = self.conv_out(F.silu(convolved)).transpose(1, 2)
        hidden = hidden + self.dropout(convolved)
        hidden = hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))
        return hidden * mask.unsqueeze(-1)


class ContentAlignedLatentFlow(nn.Module):
    """Rectified flow conditioned by externally forced character alignments."""

    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int = 192,
        text_dim: int = 192,
        layers: int = 6,
        n_heads: int = 6,
        dropout: float = 0.1,
        vocab_size: int = VOCAB_SIZE + 1,
        pad_idx: int = PAD_ID,
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            dim=text_dim,
            layers=max(1, layers // 2),
            n_heads=max(1, min(n_heads, text_dim)),
            dropout=dropout,
        )
        self.memory_proj = nn.Linear(text_dim, hidden_dim)
        self.duration_predictor = DurationPredictor(hidden_dim, dropout)
        self.duration_position_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.condition_proj = nn.Linear(hidden_dim, hidden_dim)
        self.position = SinusoidalPositionalEncoding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                LocalFlowBlock(
                    hidden_dim,
                    dropout,
                    dilation=2 ** (index % 4),
                )
                for index in range(layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.velocity_head = nn.Linear(hidden_dim, latent_dim)

    def encode_text(
        self,
        text: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_mask = content_text_mask(text, text_mask)
        if not bool(text_mask.any(dim=1).all()):
            raise ValueError("Every sample must contain at least one non-EOS character")
        memory, _ = self.text_encoder(text, text_mask)
        return self.memory_proj(memory) * text_mask.unsqueeze(-1), text_mask

    def aligned_condition(
        self,
        memory: torch.Tensor,
        text_mask: torch.Tensor,
        durations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alignment, latent_mask, _ = durations_to_path(durations, text_mask)
        condition = torch.bmm(alignment, memory)
        features = duration_position_features(durations, text_mask)
        condition = condition + self.duration_position_proj(features)
        return condition * latent_mask.unsqueeze(-1), latent_mask

    def flow_velocity(
        self,
        noisy_latents: torch.Tensor,
        times: torch.Tensor,
        latent_mask: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.latent_proj(noisy_latents) + self.condition_proj(condition)
        hidden = self.position(hidden)
        hidden = hidden + self.time_mlp(timestep_embedding(times, self.hidden_dim)).unsqueeze(1)
        hidden = hidden * latent_mask.unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden, latent_mask)
        return self.velocity_head(self.output_norm(hidden)) * latent_mask.unsqueeze(-1)

    def training_forward(
        self,
        noisy_latents: torch.Tensor,
        times: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        durations: torch.Tensor,
    ) -> ContentFlowOutput:
        memory, content_mask = self.encode_text(text, text_mask)
        condition, latent_mask = self.aligned_condition(memory, content_mask, durations)
        if noisy_latents.shape[1] != condition.shape[1]:
            raise ValueError(
                f"Latent length {noisy_latents.shape[1]} does not match "
                f"forced alignment length {condition.shape[1]}"
            )
        velocity = self.flow_velocity(noisy_latents, times, latent_mask, condition)
        log_durations = self.duration_predictor(memory, content_mask)
        return ContentFlowOutput(
            velocity=velocity,
            log_durations=log_durations,
            durations=durations,
            latent_mask=latent_mask,
        )

    def _sample_durations(
        self,
        raw_durations: torch.Tensor,
        text_mask: torch.Tensor,
        latent_length: int | None,
        max_latent_length: int,
    ) -> torch.Tensor:
        result = torch.zeros_like(raw_durations, dtype=torch.long)
        for row in range(raw_durations.shape[0]):
            token_count = int(text_mask[row].sum().item())
            values = raw_durations[row, :token_count].clamp(min=1.0)
            if latent_length is None:
                target = int(values.round().sum().item())
                target = max(token_count, min(target, max_latent_length))
            else:
                target = max(token_count, min(int(latent_length), max_latent_length))
            result[row, :token_count] = fit_durations(
                values,
                target_length=target,
                token_count=token_count,
            )
        return result

    @torch.no_grad()
    def sample(
        self,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        *,
        latent_length: int | None = None,
        steps: int = 32,
        temperature: float = 1.0,
        max_latent_length: int = 768,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        memory, content_mask = self.encode_text(text, text_mask)
        raw_durations = torch.expm1(
            self.duration_predictor(memory, content_mask)
        ).clamp(min=1.0)
        durations = self._sample_durations(
            raw_durations,
            content_mask,
            latent_length,
            max_latent_length,
        )
        condition, latent_mask = self.aligned_condition(memory, content_mask, durations)
        latents = torch.randn(
            text.shape[0],
            condition.shape[1],
            self.latent_dim,
            device=text.device,
        ) * max(float(temperature), 0.0)
        latents = latents * latent_mask.unsqueeze(-1)
        step_count = max(int(steps), 1)
        dt = 1.0 / step_count
        for step in range(step_count):
            times = torch.full((text.shape[0],), step * dt, device=text.device)
            velocity = self.flow_velocity(latents, times, latent_mask, condition)
            midpoint = (latents + 0.5 * dt * velocity) * latent_mask.unsqueeze(-1)
            midpoint_times = torch.full(
                (text.shape[0],),
                (step + 0.5) * dt,
                device=text.device,
            )
            midpoint_velocity = self.flow_velocity(
                midpoint,
                midpoint_times,
                latent_mask,
                condition,
            )
            latents = (latents + dt * midpoint_velocity) * latent_mask.unsqueeze(-1)
        return latents, latent_mask
