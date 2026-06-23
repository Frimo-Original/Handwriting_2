from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from handwriting_ai.data.codec import PAD_ID, VOCAB_SIZE
from handwriting_ai.models.flow import TextEncoder
from handwriting_ai.models.positional import SinusoidalPositionalEncoding, lengths_to_mask


@dataclass(frozen=True)
class TrajectoryGeneratorOutput:
    deltas: torch.Tensor
    point_mask: torch.Tensor
    point_lengths: torch.Tensor
    length_log: torch.Tensor


class ResidualConvBlock(nn.Module):
    def __init__(self, hidden_dim: int, *, kernel_size: int = 7, dropout: float = 0.1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_dim,
        )
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(values).transpose(1, 2)
        hidden = self.depthwise(hidden)
        hidden = torch.nn.functional.silu(hidden)
        hidden = self.pointwise(hidden).transpose(1, 2)
        values = values + self.dropout(hidden) * mask.unsqueeze(-1)
        values = values + self.dropout(self.ffn(values)) * mask.unsqueeze(-1)
        return values * mask.unsqueeze(-1)


class TrajectoryGenerator(nn.Module):
    """Direct text-to-trajectory generator with monotonic text alignment.

    The model predicts normalized five-channel point sequences
    ``(dx, dy, pen_up_logit, dx_jump, dy_jump)`` directly. It avoids point-wise
    Transformer self-attention, so training cost scales linearly with trajectory
    length and remains practical for long lines.
    """

    OUTPUT_CHANNELS = 5

    def __init__(
        self,
        *,
        hidden_dim: int = 192,
        text_dim: int = 192,
        layers: int = 6,
        n_heads: int = 6,
        dropout: float = 0.1,
        vocab_size: int = VOCAB_SIZE + 1,
        pad_idx: int = PAD_ID,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            dim=text_dim,
            layers=max(1, layers // 2),
            n_heads=max(1, min(n_heads, text_dim)),
            dropout=dropout,
        )
        self.aligned_proj = nn.Linear(text_dim, hidden_dim)
        self.global_proj = nn.Linear(text_dim, hidden_dim)
        self.query_pos = SinusoidalPositionalEncoding(hidden_dim)
        self.query_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [ResidualConvBlock(hidden_dim, kernel_size=7, dropout=dropout) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, self.OUTPUT_CHANNELS)
        self.length_head = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, text_dim),
            nn.SiLU(),
            nn.Linear(text_dim, 1),
        )

    def _aligned_text(
        self,
        memory: torch.Tensor,
        text_mask: torch.Tensor,
        point_lengths: torch.Tensor,
        max_point_len: int,
    ) -> torch.Tensor:
        batch_size, _, text_dim = memory.shape
        device = memory.device
        text_lengths = text_mask.sum(dim=1).clamp(min=1)
        positions = torch.arange(max_point_len, device=device, dtype=torch.float32).unsqueeze(0)
        denom = (point_lengths.float().clamp(min=2.0) - 1.0).unsqueeze(1)
        fraction = (positions / denom).clamp(0.0, 1.0)
        text_pos = fraction * (text_lengths.float().unsqueeze(1) - 1.0).clamp(min=0.0)
        left = text_pos.floor().long().clamp(min=0)
        right = torch.minimum(left + 1, (text_lengths - 1).unsqueeze(1))
        weight = (text_pos - left.float()).unsqueeze(-1)

        gather_left = left.unsqueeze(-1).expand(batch_size, max_point_len, text_dim)
        gather_right = right.unsqueeze(-1).expand(batch_size, max_point_len, text_dim)
        left_values = memory.gather(dim=1, index=gather_left)
        right_values = memory.gather(dim=1, index=gather_right)
        return left_values * (1.0 - weight) + right_values * weight

    def _queries(
        self,
        memory: torch.Tensor,
        pooled: torch.Tensor,
        text_mask: torch.Tensor,
        point_lengths: torch.Tensor,
        max_point_len: int,
    ) -> torch.Tensor:
        batch_size = memory.shape[0]
        device = memory.device
        base = torch.zeros(batch_size, max_point_len, self.hidden_dim, device=device)
        positional = self.query_pos(base)
        aligned = self.aligned_proj(self._aligned_text(memory, text_mask, point_lengths, max_point_len))
        global_context = self.global_proj(pooled).unsqueeze(1)
        positions = torch.arange(max_point_len, device=device, dtype=torch.float32).unsqueeze(0)
        denom = point_lengths.float().clamp(min=2.0).unsqueeze(1) - 1.0
        fraction = (positions / denom).clamp(0.0, 1.0)
        centered = fraction * 2.0 - 1.0
        length_feature = torch.log1p(point_lengths.float()).unsqueeze(1).expand_as(fraction) / 12.0
        features = torch.stack([fraction, centered, length_feature], dim=-1)
        return self.query_mlp(torch.cat([positional + aligned + global_context, features], dim=-1))

    def forward(
        self,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        *,
        point_lengths: torch.Tensor | None = None,
        max_point_length: int = 4096,
    ) -> TrajectoryGeneratorOutput:
        memory, pooled = self.text_encoder(text, text_mask)
        length_log = self.length_head(pooled).squeeze(-1)
        if point_lengths is None:
            point_lengths = torch.expm1(length_log).round().long().clamp(min=8, max=max_point_length)
        max_point_len = int(point_lengths.max().item())
        point_mask = lengths_to_mask(point_lengths, max_point_len)
        hidden = self._queries(memory, pooled, text_mask, point_lengths, max_point_len)
        hidden = hidden * point_mask.unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden, point_mask)
        deltas = self.output_head(self.norm(hidden)) * point_mask.unsqueeze(-1)
        return TrajectoryGeneratorOutput(
            deltas=deltas,
            point_mask=point_mask,
            point_lengths=point_lengths,
            length_log=length_log,
        )

    @torch.no_grad()
    def sample(
        self,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        *,
        point_length: int | None = None,
        latent_length: int | None = None,
        steps: int = 0,
        temperature: float = 0.0,
        max_point_length: int = 4096,
        max_latent_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del steps
        self.eval()
        max_length = max_latent_length if max_latent_length is not None else max_point_length
        forced_length = point_length if point_length is not None else latent_length
        if forced_length is None:
            output = self(text, text_mask, point_lengths=None, max_point_length=max_length)
        else:
            lengths = torch.full((text.shape[0],), forced_length, dtype=torch.long, device=text.device)
            output = self(text, text_mask, point_lengths=lengths, max_point_length=max_length)
        deltas = output.deltas
        if temperature > 0.0:
            within_noise = torch.randn_like(deltas[..., 0:2]) * (0.01 * temperature)
            channels = [deltas[..., 0:2] + within_noise, deltas[..., 2:3]]
            if deltas.shape[-1] >= 5:
                channels.append(deltas[..., 3:5])
            deltas = torch.cat(channels, dim=-1)
            deltas = deltas * output.point_mask.unsqueeze(-1)
        return deltas, output.point_mask
