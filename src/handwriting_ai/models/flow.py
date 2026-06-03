from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from handwriting_ai.data.codec import PAD_ID, VOCAB_SIZE
from handwriting_ai.models.positional import SinusoidalPositionalEncoding, lengths_to_mask, timestep_embedding


class TextEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = VOCAB_SIZE + 1,
        pad_idx: int = PAD_ID,
        dim: int = 192,
        layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_idx)
        self.pos = SinusoidalPositionalEncoding(dim)
        self.local_context = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, text: torch.Tensor, text_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.embedding(text)
        hidden = self.pos(hidden)
        hidden = hidden + self.local_context(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = self.encoder(hidden, src_key_padding_mask=~text_mask)
        hidden = self.norm(hidden)
        masked = hidden * text_mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / text_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        return hidden, pooled


@dataclass(frozen=True)
class FlowOutput:
    velocity: torch.Tensor
    length_log: torch.Tensor


class LatentFlowTransformer(nn.Module):
    """Rectified-flow Transformer conditioned on text.

    Training target is the velocity from Gaussian noise to VAE latents. Sampling
    integrates this velocity field with a small Euler solver.
    """

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
        self.memory_proj = nn.Linear(text_dim, hidden_dim) if text_dim != hidden_dim else nn.Identity()
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.latent_pos = SinusoidalPositionalEncoding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.velocity_head = nn.Linear(hidden_dim, latent_dim)
        self.length_head = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, text_dim),
            nn.SiLU(),
            nn.Linear(text_dim, 1),
        )

    def forward(
        self,
        noisy_latents: torch.Tensor,
        times: torch.Tensor,
        latent_mask: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> FlowOutput:
        memory, pooled = self.text_encoder(text, text_mask)
        memory = self.memory_proj(memory)
        hidden = self.latent_proj(noisy_latents)
        hidden = self.latent_pos(hidden)
        time_hidden = self.time_mlp(timestep_embedding(times, self.hidden_dim)).unsqueeze(1)
        hidden = hidden + time_hidden
        decoded = self.decoder(
            tgt=hidden,
            memory=memory,
            tgt_key_padding_mask=~latent_mask,
            memory_key_padding_mask=~text_mask,
        )
        velocity = self.velocity_head(self.norm(decoded)) * latent_mask.unsqueeze(-1)
        length_log = self.length_head(pooled).squeeze(-1)
        return FlowOutput(velocity=velocity, length_log=length_log)

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
        batch_size = text.shape[0]
        device = text.device
        if latent_length is None:
            dummy = torch.zeros(batch_size, 1, self.latent_dim, device=device)
            dummy_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
            length_log = self(
                dummy,
                torch.zeros(batch_size, device=device),
                dummy_mask,
                text,
                text_mask,
            ).length_log
            predicted = torch.expm1(length_log).round().long().clamp(min=4, max=max_latent_length)
            latent_lengths = predicted
            latent_length = int(latent_lengths.max().item())
        else:
            latent_lengths = torch.full((batch_size,), latent_length, dtype=torch.long, device=device)
        latent_mask = lengths_to_mask(latent_lengths, latent_length)
        latents = torch.randn(batch_size, latent_length, self.latent_dim, device=device) * temperature
        dt = 1.0 / float(max(steps, 1))
        for step in range(max(steps, 1)):
            times = torch.full((batch_size,), step * dt, device=device)
            velocity = self(latents, times, latent_mask, text, text_mask).velocity
            latents = (latents + velocity * dt) * latent_mask.unsqueeze(-1)
        return latents, latent_mask

