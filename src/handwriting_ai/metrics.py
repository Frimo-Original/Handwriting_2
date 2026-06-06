from __future__ import annotations

from collections.abc import Sequence

import torch

from handwriting_ai.data.codec import CTC_BLANK_ID, EOS_ID, PAD_ID


def edit_distance(reference: Sequence[int], hypothesis: Sequence[int]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_token in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            substitution = previous[hyp_index - 1] + int(ref_token != hyp_token)
            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    substitution,
                )
            )
        previous = current
    return previous[-1]


def greedy_ctc_decode(
    log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
    *,
    blank_id: int = CTC_BLANK_ID,
) -> list[list[int]]:
    if log_probs.ndim != 3:
        raise ValueError(f"Expected CTC log-probabilities shaped (T, B, C), got {log_probs.shape}")
    predictions = log_probs.argmax(dim=-1).transpose(0, 1)
    decoded: list[list[int]] = []
    for row, length in zip(predictions, output_lengths, strict=True):
        result: list[int] = []
        previous = blank_id
        for token in row[: int(length.item())].tolist():
            if token != blank_id and token != previous:
                result.append(int(token))
            previous = int(token)
        decoded.append(result)
    return decoded


def target_sequences(
    text: torch.Tensor,
    text_lengths: torch.Tensor,
    *,
    include_eos: bool = True,
) -> list[list[int]]:
    sequences: list[list[int]] = []
    for row, length in zip(text, text_lengths, strict=True):
        tokens = [
            int(token)
            for token in row[: int(length.item())].tolist()
            if int(token) != PAD_ID and (include_eos or int(token) != EOS_ID)
        ]
        sequences.append(tokens)
    return sequences


def character_error_rate(
    hypotheses: Sequence[Sequence[int]],
    references: Sequence[Sequence[int]],
) -> float:
    if len(hypotheses) != len(references):
        raise ValueError("Hypothesis and reference counts differ")
    errors = 0
    characters = 0
    for hypothesis, reference in zip(hypotheses, references, strict=True):
        errors += edit_distance(reference, hypothesis)
        characters += len(reference)
    return float(errors / max(characters, 1))


def ctc_character_error_rate(
    log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
    text: torch.Tensor,
    text_lengths: torch.Tensor,
    *,
    blank_id: int = CTC_BLANK_ID,
    include_eos: bool = False,
) -> float:
    hypotheses = greedy_ctc_decode(log_probs, output_lengths, blank_id=blank_id)
    references = target_sequences(text, text_lengths, include_eos=include_eos)
    if not include_eos:
        hypotheses = [[token for token in row if token != EOS_ID] for row in hypotheses]
    return character_error_rate(hypotheses, references)
