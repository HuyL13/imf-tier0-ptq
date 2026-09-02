from __future__ import annotations

from typing import Protocol


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


class TextDecoder(Protocol):
    def decode(self, text: str, key: bytes): ...

