from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol

from imf_tier0.stega.types import DecodeResult


class TextDecoder(Protocol):
    def decode(self, text: str, key: bytes) -> DecodeResult: ...


def decode_texts_independently(
    decoder_factory: Callable[[], TextDecoder],
    texts: Iterable[str],
    key: bytes,
) -> list[DecodeResult]:
    """Decode each carrier with fresh LM/KV-cache state."""
    return [decoder_factory().decode(text, key) for text in texts]
