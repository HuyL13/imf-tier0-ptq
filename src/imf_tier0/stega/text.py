from __future__ import annotations

from typing import Any

from imf_tier0.stega.adg import ADGCodec
from imf_tier0.stega.types import DecodeFailure, DecodeResult


class ADGTextCodec:
    """Lossless text adapter for Algorithm 2 token carriers."""

    def __init__(self, codec: ADGCodec, tokenizer: Any) -> None:
        self.codec = codec
        self.tokenizer = tokenizer
        if self.codec.token_validator is None:
            self.codec.token_validator = self._is_stable_extension

    def _tokens_to_text(self, tokens: list[int]) -> str:
        bos = getattr(self.tokenizer, "bos_token_id", None)
        wrapped = ([bos] + tokens) if bos is not None else tokens
        return self.tokenizer.decode(
            wrapped,
            skip_special_tokens=bos is not None,
            clean_up_tokenization_spaces=False,
        )

    def _text_to_tokens(self, text: str) -> list[int]:
        bos = getattr(self.tokenizer, "bos_token_id", None)
        recovered = self.tokenizer.encode(text, add_special_tokens=bos is not None)
        if bos is not None and recovered and recovered[0] == bos:
            recovered = recovered[1:]
        return list(recovered)

    def _is_stable_extension(self, prefix: list[int], token: int) -> bool:
        candidate = prefix + [token]
        return self._text_to_tokens(self._tokens_to_text(candidate)) == candidate

    def encode(self, message: bytes, key: bytes, nonce: bytes) -> str:
        tokens = self.codec.encode(message, key, nonce)
        text = self._tokens_to_text(tokens)
        recovered = self._text_to_tokens(text)
        if list(recovered) != tokens:
            raise ValueError("tokenizer is not lossless for the generated ADG carrier")
        return text

    def decode(self, text: str, key: bytes) -> DecodeResult:
        try:
            tokens = self._text_to_tokens(text)
        except Exception as error:
            return DecodeFailure(f"tokenization failed: {error}")
        return self.codec.decode_tokens(tokens, key)
