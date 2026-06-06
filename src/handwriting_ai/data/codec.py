# -*- coding: utf-8 -*-
from __future__ import annotations

CHAR_SET = "\n„“ абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ.,!?-;:()1234567890"
EOS_TOKEN = "<EOS>"
VOCAB_TOKENS = list(CHAR_SET) + [EOS_TOKEN]
VOCAB_SIZE = len(VOCAB_TOKENS)
EOS_ID = VOCAB_SIZE - 1
PAD_TOKEN = "<PAD>"
PAD_ID = VOCAB_SIZE
CTC_BLANK_ID = VOCAB_SIZE

CHAR_TO_IDX = {token: i for i, token in enumerate(VOCAB_TOKENS)}
IDX_TO_CHAR = {i: token for token, i in CHAR_TO_IDX.items()}


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace('"', "“")
    if text.endswith("\n"):
        text = text[:-1]
    return text


def encode_text(text: str, append_eos: bool = True, unknown_token: str = " ") -> list[int]:
    fallback = CHAR_TO_IDX[unknown_token]
    tokens = list(normalize_text(text))
    if append_eos:
        tokens.append(EOS_TOKEN)
    return [CHAR_TO_IDX.get(token, fallback) for token in tokens]


def decode_tokens(indices: list[int] | tuple[int, ...], skip_eos: bool = True) -> str:
    chars: list[str] = []
    for index in indices:
        token = IDX_TO_CHAR[int(index)]
        if skip_eos and token == EOS_TOKEN:
            continue
        chars.append(token)
    return "".join(chars)
