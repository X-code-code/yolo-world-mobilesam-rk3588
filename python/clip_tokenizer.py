#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free OpenAI CLIP byte-pair tokenizer.

The tokenizer reads ``bpe_simple_vocab_16e6.txt`` at runtime.  It intentionally
keeps NumPy optional so the same code can be used by small board-side tools and
by RKNNLite inference code.
"""

from __future__ import annotations

import html
import math
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union


BOS_TOKEN_ID = 49_406
EOS_TOKEN_ID = 49_407
PAD_TOKEN_ID = 49_407
CONTEXT_LENGTH = 20
EXPECTED_MERGE_COUNT = 48_894

_START_OF_TEXT = "<|startoftext|>"
_END_OF_TEXT = "<|endoftext|>"
_SPECIAL_TOKENS = (_START_OF_TEXT, _END_OF_TEXT)
_CONTRACTIONS = ("'re", "'ve", "'ll", "'s", "'t", "'m", "'d")


def bytes_to_unicode() -> Dict[int, str]:
    """Return the reversible byte-to-Unicode map used by OpenAI CLIP."""

    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(0xA1, 0xAC + 1))
    visible += list(range(0xAE, 0xFF + 1))
    code_points = visible[:]
    extra_index = 0
    visible_set = set(visible)
    for byte_value in range(256):
        if byte_value not in visible_set:
            visible.append(byte_value)
            code_points.append(256 + extra_index)
            extra_index += 1
    return dict(zip(visible, (chr(code_point) for code_point in code_points)))


def _pairs(word: Sequence[str]) -> Set[Tuple[str, str]]:
    return set(zip(word, word[1:]))


def _is_letter(character: str) -> bool:
    return unicodedata.category(character).startswith("L")


def _is_number(character: str) -> bool:
    return unicodedata.category(character).startswith("N")


def _clean_text(text: str) -> str:
    # OpenAI's reference tokenizer also applies ftfy.fix_text.  The board
    # version accepts valid Python Unicode directly and avoids that dependency.
    text = html.unescape(html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


def _basic_tokens(text: str) -> Iterator[str]:
    """Implement CLIP's Unicode-aware token pattern with the standard library."""

    index = 0
    while index < len(text):
        character = text[index]
        if character.isspace():
            index += 1
            continue

        special = next(
            (token for token in _SPECIAL_TOKENS if text.startswith(token, index)),
            None,
        )
        if special is not None:
            yield special
            index += len(special)
            continue

        contraction = next(
            (suffix for suffix in _CONTRACTIONS if text.startswith(suffix, index)),
            None,
        )
        if contraction is not None:
            yield contraction
            index += len(contraction)
            continue

        if _is_letter(character):
            end = index + 1
            while end < len(text) and _is_letter(text[end]):
                end += 1
            yield text[index:end]
            index = end
            continue

        if _is_number(character):
            # The CLIP pattern deliberately emits one numeric code point at a
            # time rather than grouping a complete number.
            yield character
            index += 1
            continue

        end = index + 1
        while end < len(text):
            next_character = text[end]
            if (
                next_character.isspace()
                or _is_letter(next_character)
                or _is_number(next_character)
            ):
                break
            end += 1
        yield text[index:end]
        index = end


class CLIPTokenizer:
    """OpenAI CLIP BPE tokenizer with the RKNN demo's 20-token contract."""

    def __init__(self, merges_path: Optional[Union[str, Path]] = None) -> None:
        if merges_path is None:
            merges_path = Path(__file__).with_name("bpe_simple_vocab_16e6.txt")
        self.merges_path = Path(merges_path)
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {
            unicode_character: byte_value
            for byte_value, unicode_character in self.byte_encoder.items()
        }

        lines = self.merges_path.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0] != "#version: 0.2":
            raise ValueError(f"invalid CLIP merges header in {self.merges_path}")
        merge_lines = lines[1:]
        if len(merge_lines) != EXPECTED_MERGE_COUNT:
            raise ValueError(
                f"expected {EXPECTED_MERGE_COUNT} merges, found "
                f"{len(merge_lines)} in {self.merges_path}"
            )

        merges: List[Tuple[str, str]] = []
        for line_number, line in enumerate(merge_lines, start=2):
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(
                    f"invalid merge at {self.merges_path}:{line_number}: {line!r}"
                )
            merges.append((fields[0], fields[1]))

        byte_vocab = list(self.byte_encoder.values())
        vocab = byte_vocab + [token + "</w>" for token in byte_vocab]
        vocab += [first + second for first, second in merges]
        vocab += [_START_OF_TEXT, _END_OF_TEXT]

        self.encoder = {token: token_id for token_id, token in enumerate(vocab)}
        self.decoder = {token_id: token for token, token_id in self.encoder.items()}
        self.bpe_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.cache: Dict[str, Tuple[str, ...]] = {
            _START_OF_TEXT: (_START_OF_TEXT,),
            _END_OF_TEXT: (_END_OF_TEXT,),
        }

        if self.encoder.get(_START_OF_TEXT) != BOS_TOKEN_ID:
            raise ValueError("merges asset produced an unexpected BOS token ID")
        if self.encoder.get(_END_OF_TEXT) != EOS_TOKEN_ID:
            raise ValueError("merges asset produced an unexpected EOS token ID")

    def _bpe(self, token: str) -> Tuple[str, ...]:
        cached = self.cache.get(token)
        if cached is not None:
            return cached

        word: Tuple[str, ...] = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _pairs(word)
        if not pairs:
            result = (token + "</w>",)
            self.cache[token] = result
            return result

        while pairs:
            first, second = min(
                pairs,
                key=lambda pair: self.bpe_ranks.get(pair, math.inf),
            )
            if (first, second) not in self.bpe_ranks:
                break

            merged: List[str] = []
            index = 0
            while index < len(word):
                try:
                    next_match = word.index(first, index)
                except ValueError:
                    merged.extend(word[index:])
                    break
                merged.extend(word[index:next_match])
                index = next_match
                if (
                    index < len(word) - 1
                    and word[index] == first
                    and word[index + 1] == second
                ):
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
            if len(word) == 1:
                break
            pairs = _pairs(word)

        self.cache[token] = word
        return word

    def encode(self, text: str) -> List[int]:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

        cleaned = _clean_text(text).lower()
        token_ids: List[int] = []
        for token in _basic_tokens(cleaned):
            byte_token = "".join(
                self.byte_encoder[byte_value] for byte_value in token.encode("utf-8")
            )
            token_ids.extend(self.encoder[subword] for subword in self._bpe(byte_token))
        return token_ids

    def tokenize(
        self,
        text: str,
        context_length: int = CONTEXT_LENGTH,
        *,
        padding: bool = True,
        truncate: bool = True,
    ) -> List[int]:
        if context_length < 2:
            raise ValueError("context_length must be at least 2")

        token_ids = [BOS_TOKEN_ID, *self.encode(text)]
        if len(token_ids) > context_length - 1:
            if not truncate:
                raise ValueError(
                    f"text does not fit in context_length={context_length}"
                )
            token_ids = token_ids[: context_length - 1]
        token_ids.append(EOS_TOKEN_ID)

        if padding:
            token_ids.extend([PAD_TOKEN_ID] * (context_length - len(token_ids)))
        return token_ids

    def decode(self, token_ids: Iterable[int]) -> str:
        encoded = "".join(self.decoder[token_id] for token_id in token_ids)
        byte_values = bytearray(self.byte_decoder[character] for character in encoded)
        return byte_values.decode("utf-8", errors="replace").replace("</w>", " ")


def tokenize_prompts(
    prompts: Union[str, Sequence[str]],
    *,
    merges_path: Optional[Union[str, Path]] = None,
    context_length: int = CONTEXT_LENGTH,
    return_numpy: bool = False,
):
    """Tokenize one prompt or a sequence into uniformly padded rows.

    NumPy is imported only when ``return_numpy=True``.
    """

    prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
    tokenizer = CLIPTokenizer(merges_path)
    rows = [tokenizer.tokenize(prompt, context_length) for prompt in prompt_list]
    if return_numpy:
        import numpy as np

        return np.asarray(rows, dtype=np.int32)
    return rows


__all__ = [
    "BOS_TOKEN_ID",
    "CONTEXT_LENGTH",
    "EOS_TOKEN_ID",
    "PAD_TOKEN_ID",
    "CLIPTokenizer",
    "bytes_to_unicode",
    "tokenize_prompts",
]
