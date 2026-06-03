from __future__ import annotations

import torch
from torch import nn

from handwriting_ai.data.codec import CTC_BLANK_ID, VOCAB_SIZE


class TrajectoryRecognizer(nn.Module):
    """CTC recognizer used for validation and reranking generated samples."""

    def __init__(
        self,
        *,
        input_dim: int = 3,
        hidden_dim: int = 160,
        layers: int = 2,
        dropout: float = 0.1,
        vocab_size: int = VOCAB_SIZE,
        blank_id: int = CTC_BLANK_ID,
    ) -> None:
        super().__init__()
        self.blank_id = blank_id
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
        )
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_dim * 2, vocab_size + 1)

    @staticmethod
    def output_lengths(point_lengths: torch.Tensor) -> torch.Tensor:
        lengths = torch.div(point_lengths + 1, 2, rounding_mode="floor")
        lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        return lengths.clamp(min=1)

    def forward(self, points: torch.Tensor, point_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.conv(points.transpose(1, 2)).transpose(1, 2)
        output_lengths = self.output_lengths(point_lengths).clamp(max=hidden.shape[1])
        hidden, _ = self.rnn(hidden)
        logits = self.classifier(hidden)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        return log_probs, output_lengths

