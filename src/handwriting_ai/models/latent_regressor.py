from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from handwriting_ai.data.codec import PAD_ID, VOCAB_SIZE
from handwriting_ai.models.flow import TextEncoder
from handwriting_ai.models.positional import SinusoidalPositionalEncoding, lengths_to_mask


@dataclass(frozen=True)
class LatentRegressorOutput:
    latents: torch.Tensor
    latent_mask: torch.Tensor
    latent_lengths: torch.Tensor
    length_log: torch.Tensor


class LatentRegressorTransformer(nn.Module):
    """Supervised text-to-latent generator for small single-writer datasets.

    Unlike the flow model, this network directly predicts the normalized latent
    sequence produced by the ink autoencoder. The decoder queries are monotonic:
    every latent position receives a rough aligned text embedding before
    cross-attention. That bias is deliberately simple and useful for small data.
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
        self.aligned_proj = nn.Linear(text_dim, hidden_dim)
        self.query_pos = SinusoidalPositionalEncoding(hidden_dim)
        self.query_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
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
        self.latent_head = nn.Linear(hidden_dim, latent_dim)
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
        latent_lengths: torch.Tensor,
        max_latent_len: int,
    ) -> torch.Tensor:
        batch_size, _, text_dim = memory.shape
        device = memory.device
        text_lengths = text_mask.sum(dim=1).clamp(min=1)
        positions = torch.arange(max_latent_len, device=device, dtype=torch.float32).unsqueeze(0)
        denom = (latent_lengths.float().clamp(min=2.0) - 1.0).unsqueeze(1)
        fraction = (positions / denom).clamp(0.0, 1.0)
        text_pos = fraction * (text_lengths.float().unsqueeze(1) - 1.0).clamp(min=0.0)
        left = text_pos.floor().long().clamp(min=0)
        right = (left + 1).clamp(max=max(0, memory.shape[1] - 1))
        right = torch.minimum(right, (text_lengths - 1).unsqueeze(1))
        weight = (text_pos - left.float()).unsqueeze(-1)

        gather_left = left.unsqueeze(-1).expand(batch_size, max_latent_len, text_dim)
        gather_right = right.unsqueeze(-1).expand(batch_size, max_latent_len, text_dim)
        left_values = memory.gather(dim=1, index=gather_left)
        right_values = memory.gather(dim=1, index=gather_right)
        return left_values * (1.0 - weight) + right_values * weight

    def _queries(
        self,
        memory: torch.Tensor,
        text_mask: torch.Tensor,
        latent_lengths: torch.Tensor,
        max_latent_len: int,
    ) -> torch.Tensor:
        batch_size = memory.shape[0]
        device = memory.device
        base = torch.zeros(batch_size, max_latent_len, self.hidden_dim, device=device)
        positional = self.query_pos(base)
        aligned = self.aligned_proj(self._aligned_text(memory, text_mask, latent_lengths, max_latent_len))
        positions = torch.arange(max_latent_len, device=device, dtype=torch.float32).unsqueeze(0)
        denom = latent_lengths.float().clamp(min=2.0).unsqueeze(1) - 1.0
        fraction = (positions / denom).clamp(0.0, 1.0)
        length_feature = torch.log1p(latent_lengths.float()).unsqueeze(1).expand_as(fraction)
        features = torch.stack([fraction, length_feature / 8.0], dim=-1)
        return self.query_mlp(torch.cat([positional + aligned, features], dim=-1))

    def forward(
        self,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        *,
        latent_lengths: torch.Tensor | None = None,
        max_latent_length: int = 768,
    ) -> LatentRegressorOutput:
        memory, pooled = self.text_encoder(text, text_mask)
        length_log = self.length_head(pooled).squeeze(-1)
        if latent_lengths is None:
            latent_lengths = torch.expm1(length_log).round().long().clamp(min=4, max=max_latent_length)
        max_latent_len = int(latent_lengths.max().item())
        latent_mask = lengths_to_mask(latent_lengths, max_latent_len)
        queries = self._queries(memory, text_mask, latent_lengths, max_latent_len)
        decoded = self.decoder(
            tgt=queries,
            memory=self.memory_proj(memory),
            tgt_key_padding_mask=~latent_mask,
            memory_key_padding_mask=~text_mask,
        )
        latents = self.latent_head(self.norm(decoded)) * latent_mask.unsqueeze(-1)
        return LatentRegressorOutput(
            latents=latents,
            latent_mask=latent_mask,
            latent_lengths=latent_lengths,
            length_log=length_log,
        )

    @torch.no_grad()
    def sample(
        self,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        *,
        latent_length: int | None = None,
        steps: int = 0,
        temperature: float = 0.0,
        max_latent_length: int = 768,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del steps
        self.eval()
        if latent_length is None:
            output = self(text, text_mask, latent_lengths=None, max_latent_length=max_latent_length)
        else:
            lengths = torch.full((text.shape[0],), latent_length, dtype=torch.long, device=text.device)
            output = self(text, text_mask, latent_lengths=lengths, max_latent_length=max_latent_length)
        latents = output.latents
        if temperature > 0.0:
            latents = latents + torch.randn_like(latents) * (0.02 * temperature)
            latents = latents * output.latent_mask.unsqueeze(-1)
        return latents, output.latent_mask

