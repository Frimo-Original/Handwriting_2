# -*- coding: utf-8 -*-
"""Standalone text alphabet and encoding helpers for the dataset scripts."""

# Keep this alphabet in sync with src/config.py before training checkpoints that
# should be compatible with this dataset.
CHAR_SET = "\n„“ абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ.,!?-;:()1234567890"
EOS_TOKEN = "<EOS>"
VOCAB_TOKENS = list(CHAR_SET) + [EOS_TOKEN]

char_to_idx = {token: i for i, token in enumerate(VOCAB_TOKENS)}
idx_to_char = {i: token for token, i in char_to_idx.items()}
vocab_size = len(VOCAB_TOKENS)

append_eos_to_dataset = True
strip_final_newline_before_eos = True


def normalize_upper_quotes(text):
    return text.replace('"', "“")


def normalize_training_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = normalize_upper_quotes(text)
    if strip_final_newline_before_eos and text.endswith("\n"):
        text = text[:-1]
    return text


def normalize_russian_quotes(text):
    result = []
    is_opening = True
    for ch in text:
        if ch == '"':
            result.append("„" if is_opening else "“")
            is_opening = not is_opening
        else:
            result.append(ch)
    return "".join(result)


def tokenize_text(text, append_eos=False, normalize=False, normalize_quotes=False):
    if normalize:
        text = normalize_training_text(text)
    if normalize_quotes:
        text = normalize_russian_quotes(text)

    tokens = list(text)
    if append_eos:
        tokens.append(EOS_TOKEN)
    return tokens


def encode_text(text, append_eos=False, normalize=False, normalize_quotes=False, unknown_token=" "):
    fallback = char_to_idx[unknown_token]
    return [
        char_to_idx.get(token, fallback)
        for token in tokenize_text(text, append_eos, normalize, normalize_quotes)
    ]


def decode_tokens(indices, skip_eos=False):
    tokens = []
    for idx in indices:
        token = idx_to_char[int(idx)]
        if skip_eos and token == EOS_TOKEN:
            continue
        tokens.append(token)
    return "".join(tokens)


def unsupported_tokens(text, normalize=True):
    tokens = tokenize_text(text, append_eos=False, normalize=normalize)
    return sorted({token for token in tokens if token not in char_to_idx})
