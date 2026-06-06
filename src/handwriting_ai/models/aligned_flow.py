from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from handwriting_ai.data.codec import PAD_ID, VOCAB_SIZE
from handwriting_ai.models.flow import TextEncoder
from handwriting_ai.models.positional import SinusoidalPositionalEncoding, lengths_to_mask, timestep_embedding


def maximum_path(
    scores: torch.Tensor,
    latent_lengths: torch.Tensor,
    text_lengths: torch.Tensor,
) -> torch.Tensor:
    """Find a hard monotonic alignment with at least one frame per token."""

    score_rows = scores.detach().float().cpu().numpy()
    paths = np.zeros(score_rows.shape, dtype=np.float32)
    for row in range(scores.shape[0]):
        latent_len = int(latent_lengths[row].item())
        text_len = int(text_lengths[row].item())
        if latent_len < text_len:
            raise ValueError(
                f"MAS requires latent length >= text length, got {latent_len} < {text_len}"
            )
        values = np.full((latent_len, text_len), -np.inf, dtype=np.float32)
        backtrack = np.zeros((latent_len, text_len), dtype=np.bool_)
        values[0, 0] = score_rows[row, 0, 0]
        for frame in range(1, latent_len):
            min_token = max(0, text_len - (latent_len - frame))
            max_token = min(frame, text_len - 1)
            token_slice = slice(min_token, max_token + 1)
            stay = values[frame - 1, token_slice]
            advance = np.full_like(stay, -np.inf)
            advance_start = max(min_token, 1)
            if advance_start <= max_token:
                offset = advance_start - min_token
                advance[offset:] = values[frame - 1, advance_start - 1 : max_token]
            choose_advance = advance > stay
            values[frame, token_slice] = (
                np.maximum(stay, advance) + score_rows[row, frame, token_slice]
            )
            backtrack[frame, token_slice] = choose_advance

        token = text_len - 1
        for frame in range(latent_len - 1, -1, -1):
            paths[row, frame, token] = 1.0
            if frame > 0 and token > 0 and backtrack[frame, token]:
                token -= 1
        if token != 0:
            raise RuntimeError("Failed to construct a complete monotonic alignment")
    return torch.from_numpy(paths).to(device=scores.device, dtype=scores.dtype)


def durations_to_path(
    durations: torch.Tensor,
    text_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = durations.shape[0]
    latent_lengths = (durations * text_mask).sum(dim=1).long().clamp(min=1)
    max_length = int(latent_lengths.max().item())
    path = durations.new_zeros((batch_size, max_length, durations.shape[1]), dtype=torch.float32)
    for row in range(batch_size):
        cursor = 0
        for token in range(int(text_mask[row].sum().item())):
            duration = int(durations[row, token].item())
            if duration > 0:
                path[row, cursor : cursor + duration, token] = 1.0
                cursor += duration
    return path, lengths_to_mask(latent_lengths, max_length), latent_lengths


class DurationPredictor(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, memory: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.layers(memory.transpose(1, 2))
        return self.output(hidden).squeeze(1) * text_mask


class ConformerFlowBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float, kernel_size: int = 15) -> None:
        super().__init__()
        self.ffn1_norm = nn.LayerNorm(hidden_dim)
        self.ffn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.conv_norm = nn.LayerNorm(hidden_dim)
        self.conv_in = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
        )
        self.conv_group_norm = nn.GroupNorm(1, hidden_dim)
        self.conv_out = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.ffn2_norm = nn.LayerNorm(hidden_dim)
        self.ffn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = hidden + 0.5 * self.ffn1(self.ffn1_norm(hidden))
        normalized = self.attn_norm(hidden)
        attended, _ = self.attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~mask,
            need_weights=False,
        )
        hidden = hidden + self.dropout(attended)

        convolved = self.conv_norm(hidden).transpose(1, 2)
        convolved = F.glu(self.conv_in(convolved), dim=1)
        convolved = self.depthwise(convolved)
        convolved = F.silu(self.conv_group_norm(convolved))
        convolved = self.conv_out(convolved).transpose(1, 2)
        hidden = hidden + self.dropout(convolved)
        hidden = hidden + 0.5 * self.ffn2(self.ffn2_norm(hidden))
        return self.final_norm(hidden) * mask.unsqueeze(-1)


@dataclass(frozen=True)
class AlignedFlowTrainingOutput:
    velocity: torch.Tensor
    alignment_scores: torch.Tensor
    alignment: torch.Tensor
    durations: torch.Tensor
    log_durations: torch.Tensor


class AlignedLatentFlow(nn.Module):
    """Monotonic duration-conditioned rectified flow for autoencoder latents."""

    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int = 192,
        text_dim: int = 192,
        layers: int = 6,
        n_heads: int = 6,
        dropout: float = 0.1,
        alignment_prior_strength: float = 0.5,
        alignment_prior_width: float = 0.35,
        vocab_size: int = VOCAB_SIZE + 1,
        pad_idx: int = PAD_ID,
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.alignment_prior_strength = float(alignment_prior_strength)
        self.alignment_prior_width = float(alignment_prior_width)
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            dim=text_dim,
            layers=max(1, layers // 2),
            n_heads=max(1, min(n_heads, text_dim)),
            dropout=dropout,
        )
        self.memory_proj = nn.Linear(text_dim, hidden_dim)
        self.alignment_mean = nn.Linear(hidden_dim, latent_dim)
        self.alignment_log_scale = nn.Linear(hidden_dim, latent_dim)
        self.duration_predictor = DurationPredictor(hidden_dim, dropout)

        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.condition_proj = nn.Linear(hidden_dim, hidden_dim)
        self.position = SinusoidalPositionalEncoding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [ConformerFlowBlock(hidden_dim, n_heads, dropout) for _ in range(layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.velocity_head = nn.Linear(hidden_dim, latent_dim)

    def encode_text(
        self,
        text: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory, _ = self.text_encoder(text, text_mask)
        return self.memory_proj(memory) * text_mask.unsqueeze(-1)

    def alignment_scores(
        self,
        latents: torch.Tensor,
        latent_mask: torch.Tensor,
        memory: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        means = self.alignment_mean(memory)
        log_scales = self.alignment_log_scale(memory).clamp(min=-3.0, max=3.0)
        differences = latents.unsqueeze(2) - means.unsqueeze(1)
        scores = -0.5 * (
            differences.pow(2) * torch.exp(-2.0 * log_scales.unsqueeze(1))
            + 2.0 * log_scales.unsqueeze(1)
        ).mean(dim=-1)
        if self.alignment_prior_strength > 0.0:
            latent_lengths = latent_mask.sum(dim=1).clamp(min=2).to(latents.dtype)
            text_lengths = text_mask.sum(dim=1).clamp(min=2).to(latents.dtype)
            latent_positions = torch.arange(
                latents.shape[1],
                device=latents.device,
                dtype=latents.dtype,
            ).view(1, -1, 1)
            text_positions = torch.arange(
                memory.shape[1],
                device=latents.device,
                dtype=latents.dtype,
            ).view(1, 1, -1)
            latent_fraction = latent_positions / (latent_lengths - 1.0).view(-1, 1, 1)
            text_fraction = text_positions / (text_lengths - 1.0).view(-1, 1, 1)
            width = max(self.alignment_prior_width, 1e-3)
            scores = scores - self.alignment_prior_strength * (
                (latent_fraction - text_fraction) / width
            ).square()
        valid = latent_mask.unsqueeze(2) & text_mask.unsqueeze(1)
        return scores.masked_fill(~valid, -1e4)

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
        target_latents: torch.Tensor,
        latent_mask: torch.Tensor,
        latent_lengths: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> AlignedFlowTrainingOutput:
        memory = self.encode_text(text, text_mask)
        scores = self.alignment_scores(target_latents, latent_mask, memory, text_mask)
        alignment = maximum_path(scores, latent_lengths, text_lengths)
        durations = alignment.sum(dim=1)
        condition = torch.bmm(alignment, memory)
        velocity = self.flow_velocity(noisy_latents, times, latent_mask, condition)
        log_durations = self.duration_predictor(memory, text_mask)
        return AlignedFlowTrainingOutput(
            velocity=velocity,
            alignment_scores=scores,
            alignment=alignment,
            durations=durations,
            log_durations=log_durations,
        )

    @staticmethod
    def _fit_durations(
        raw_durations: torch.Tensor,
        text_mask: torch.Tensor,
        target_length: int | None,
        max_latent_length: int,
    ) -> torch.Tensor:
        result = torch.zeros_like(raw_durations, dtype=torch.long)
        for row in range(raw_durations.shape[0]):
            text_len = int(text_mask[row].sum().item())
            values = raw_durations[row, :text_len].clamp(min=0.05)
            if target_length is None:
                total = int(torch.round(values).clamp(min=1).sum().item())
                total = max(text_len, min(total, max_latent_length))
            else:
                total = max(text_len, min(int(target_length), max_latent_length))
            extras = total - text_len
            if extras <= 0:
                result[row, :text_len] = 1
                continue
            allocation = values / values.sum().clamp(min=1e-6) * extras
            floors = torch.floor(allocation).long()
            durations = floors + 1
            remainder = extras - int(floors.sum().item())
            if remainder > 0:
                fractions = allocation - floors
                indices = torch.topk(fractions, k=remainder).indices
                durations[indices] += 1
            result[row, :text_len] = durations
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
        memory = self.encode_text(text, text_mask)
        raw_durations = torch.expm1(self.duration_predictor(memory, text_mask)).clamp(min=0.05)
        durations = self._fit_durations(
            raw_durations,
            text_mask,
            target_length=latent_length,
            max_latent_length=max_latent_length,
        )
        alignment, latent_mask, _ = durations_to_path(durations, text_mask)
        condition = torch.bmm(alignment, memory)
        latents = torch.randn(
            text.shape[0],
            alignment.shape[1],
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
            midpoint_times = torch.full((text.shape[0],), (step + 0.5) * dt, device=text.device)
            midpoint_velocity = self.flow_velocity(midpoint, midpoint_times, latent_mask, condition)
            latents = (latents + dt * midpoint_velocity) * latent_mask.unsqueeze(-1)
        return latents, latent_mask
