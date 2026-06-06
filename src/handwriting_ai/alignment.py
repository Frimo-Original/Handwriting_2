from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from handwriting_ai.checkpoint import load_checkpoint, save_checkpoint
from handwriting_ai.data.codec import CTC_BLANK_ID, EOS_ID, PAD_ID


ALIGNMENT_CACHE_VERSION = 1


@dataclass(frozen=True)
class ForcedAlignment:
    token_frames: torch.Tensor
    durations: torch.Tensor
    score: float


def content_tokens(tokens: torch.Tensor, length: int) -> torch.Tensor:
    values = tokens[:length]
    return values[(values != PAD_ID) & (values != EOS_ID)]


def _expanded_ctc_targets(targets: torch.Tensor, blank_id: int) -> torch.Tensor:
    expanded = torch.full(
        (targets.numel() * 2 + 1,),
        blank_id,
        dtype=torch.long,
        device=targets.device,
    )
    expanded[1::2] = targets
    return expanded


def _token_frames_from_states(states: torch.Tensor, token_count: int) -> torch.Tensor:
    token_frames = torch.full_like(states, -1)
    token_states = states % 2 == 1
    token_frames[token_states] = states[token_states] // 2

    frame = 0
    while frame < len(states):
        if token_frames[frame] >= 0:
            frame += 1
            continue
        start = frame
        while frame < len(states) and token_frames[frame] < 0:
            frame += 1
        end = frame
        left = int(token_frames[start - 1].item()) if start > 0 else -1
        right = int(token_frames[end].item()) if end < len(states) else -1
        if left < 0:
            token_frames[start:end] = max(right, 0)
        elif right < 0:
            token_frames[start:end] = min(left, token_count - 1)
        else:
            midpoint = start + (end - start) // 2
            token_frames[start:midpoint] = left
            token_frames[midpoint:end] = right
    return token_frames.clamp(min=0, max=max(token_count - 1, 0))


def ctc_forced_alignment(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    *,
    blank_id: int = CTC_BLANK_ID,
) -> ForcedAlignment:
    if log_probs.ndim != 2:
        raise ValueError(f"Expected log_probs shaped (T, C), got {log_probs.shape}")
    targets = targets.to(device=log_probs.device, dtype=torch.long).reshape(-1)
    frame_count = log_probs.shape[0]
    if targets.numel() == 0:
        raise ValueError("Cannot align an empty target")
    repeated = int((targets[1:] == targets[:-1]).sum().item())
    minimum_frames = int(targets.numel()) + repeated
    if frame_count < minimum_frames:
        raise ValueError(
            f"CTC alignment requires at least {minimum_frames} frames, got {frame_count}"
        )

    states = _expanded_ctc_targets(targets, blank_id)
    state_count = states.numel()
    neg_inf = torch.finfo(log_probs.dtype).min
    scores = log_probs.new_full((frame_count, state_count), neg_inf)
    backtrack = torch.full(
        (frame_count, state_count),
        -1,
        dtype=torch.int16,
        device=log_probs.device,
    )
    scores[0, 0] = log_probs[0, blank_id]
    if state_count > 1:
        scores[0, 1] = log_probs[0, states[1]]

    skip_allowed = torch.zeros(state_count, dtype=torch.bool, device=log_probs.device)
    if state_count > 2:
        skip_allowed[2:] = (states[2:] != blank_id) & (states[2:] != states[:-2])
    state_indices = torch.arange(state_count, device=log_probs.device)
    for frame in range(1, frame_count):
        previous = scores[frame - 1]
        stay = previous
        advance = torch.cat([previous.new_full((1,), neg_inf), previous[:-1]])
        skip = torch.cat([previous.new_full((2,), neg_inf), previous[:-2]])
        skip = skip.masked_fill(~skip_allowed, neg_inf)
        candidates = torch.stack([stay, advance, skip], dim=0)
        best_scores, offsets = candidates.max(dim=0)
        scores[frame] = best_scores + log_probs[frame, states]
        backtrack[frame] = (state_indices - offsets).to(torch.int16)

    final_states = [state_count - 1]
    if state_count > 1:
        final_states.append(state_count - 2)
    final_scores = torch.stack([scores[-1, state] for state in final_states])
    best_final = int(final_scores.argmax().item())
    state = final_states[best_final]
    path = torch.empty(frame_count, dtype=torch.long, device=log_probs.device)
    for frame in range(frame_count - 1, -1, -1):
        path[frame] = state
        if frame > 0:
            state = int(backtrack[frame, state].item())
            if state < 0:
                raise RuntimeError("Broken CTC forced-alignment backtrack")

    token_frames = _token_frames_from_states(path, int(targets.numel()))
    durations = torch.bincount(token_frames, minlength=int(targets.numel()))
    score = float(final_scores[best_final].detach().cpu() / frame_count)
    return ForcedAlignment(
        token_frames=token_frames.detach().cpu(),
        durations=durations.detach().cpu(),
        score=score,
    )


def fit_durations(
    durations: torch.Tensor,
    *,
    target_length: int,
    token_count: int | None = None,
) -> torch.Tensor:
    values = durations.float().reshape(-1)
    if token_count is not None:
        values = values[:token_count]
    count = values.numel()
    if count == 0:
        raise ValueError("Cannot fit empty durations")
    if target_length < count:
        raise ValueError(
            f"Target length {target_length} is shorter than token count {count}"
        )
    extras = target_length - count
    if extras == 0:
        return torch.ones(count, dtype=torch.long, device=durations.device)
    weights = values.clamp(min=1.0)
    allocation = weights / weights.sum() * extras
    floors = allocation.floor().long()
    fitted = floors + 1
    remainder = extras - int(floors.sum().item())
    if remainder:
        order = torch.argsort(allocation - floors, descending=True)
        fitted[order[:remainder]] += 1
    return fitted


class AlignmentCache:
    def __init__(self, entries: dict[int, dict[str, Any]], metadata: dict[str, Any]) -> None:
        self.entries = entries
        self.metadata = metadata

    def durations(
        self,
        sample_id: int,
        *,
        token_count: int,
        target_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        try:
            raw = self.entries[int(sample_id)]["durations"]
        except KeyError as exc:
            raise KeyError(f"Missing forced alignment for sample {sample_id}") from exc
        values = torch.as_tensor(raw, dtype=torch.float32, device=device)
        if values.numel() != token_count:
            raise ValueError(
                f"Alignment for sample {sample_id} has {values.numel()} tokens, "
                f"but batch contains {token_count}"
            )
        return fit_durations(
            values,
            target_length=target_length,
            token_count=token_count,
        )

    def batch(
        self,
        sample_ids: torch.Tensor,
        text_lengths: torch.Tensor,
        latent_lengths: torch.Tensor,
        *,
        max_text_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        result = torch.zeros(
            len(sample_ids),
            max_text_length,
            dtype=torch.long,
            device=device,
        )
        for row in range(len(sample_ids)):
            count = int(text_lengths[row].item())
            fitted = self.durations(
                int(sample_ids[row].item()),
                token_count=count,
                target_length=int(latent_lengths[row].item()),
                device=device,
            )
            result[row, :count] = fitted
        return result

    def save(self, path: str | Path) -> None:
        save_checkpoint(
            path,
            alignment_cache_version=ALIGNMENT_CACHE_VERSION,
            metadata=self.metadata,
            entries=self.entries,
        )

    @classmethod
    def load(cls, path: str | Path) -> "AlignmentCache":
        payload = load_checkpoint(path, map_location="cpu")
        version = int(payload.get("alignment_cache_version", 0))
        if version != ALIGNMENT_CACHE_VERSION:
            raise ValueError(
                f"Unsupported alignment cache version {version}; "
                f"expected {ALIGNMENT_CACHE_VERSION}"
            )
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, dict):
            raise ValueError("Alignment cache does not contain entries")
        entries = {int(key): value for key, value in raw_entries.items()}
        return cls(entries=entries, metadata=dict(payload.get("metadata", {})))
