"""A pinned, already-loaded Transformers probability carrier for ADG."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import Any

import torch


class TransformersCarrier:
    """Expose next-token probabilities from an existing causal language model.

    The caller owns model/tokenizer loading and their revisions.  This class only
    combines the fixed prompt prefix with ADG's generated prefix and performs a
    single inference pass.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        prefix_ids: Sequence[int],
        temperature: float = 1.0,
    ) -> None:
        if not isinstance(temperature, Real) or isinstance(temperature, bool) or not math.isfinite(float(temperature)) or temperature <= 0:
            raise ValueError("temperature must be a positive finite number")
        self.model = model
        self.tokenizer = tokenizer
        self.prefix_ids = self._token_ids(prefix_ids, "prefix_ids")
        self.temperature = float(temperature)
        self.vocabulary_size = self._vocabulary_size(tokenizer)

    @staticmethod
    def _token_ids(token_ids: Sequence[int], name: str) -> tuple[int, ...]:
        try:
            values = tuple(token_ids)
        except TypeError as exc:
            raise ValueError(f"{name} must be a sequence of token IDs") from exc
        if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in values):
            raise ValueError(f"{name} must contain non-negative integer token IDs")
        return values

    @staticmethod
    def _vocabulary_size(tokenizer: Any) -> int:
        vocabulary_size = getattr(tokenizer, "vocab_size", None)
        if vocabulary_size is None:
            try:
                vocabulary_size = len(tokenizer)
            except TypeError as exc:
                raise ValueError("tokenizer must expose a vocabulary size") from exc
        if not isinstance(vocabulary_size, int) or isinstance(vocabulary_size, bool) or vocabulary_size <= 0:
            raise ValueError("tokenizer vocabulary size must be a positive integer")
        return vocabulary_size

    def _device(self) -> Any | None:
        device = getattr(self.model, "device", None)
        if device is not None:
            return device
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            return None

    def distribution(self, prefix: list[int]) -> list[float]:
        """Return float32-softmax next-token probabilities for *prefix*."""
        generated_prefix = self._token_ids(prefix, "prefix")
        input_values = self.prefix_ids + generated_prefix
        if not input_values:
            raise ValueError("fixed and generated prefixes cannot both be empty")
        input_ids = torch.tensor([input_values], dtype=torch.long, device=self._device())
        with torch.inference_mode():
            output = self.model(input_ids=input_ids)
            logits = getattr(output, "logits", None)
            if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
                raise ValueError("model logits must have shape [1, sequence, vocabulary]")
            if logits.shape[0] != 1 or logits.shape[1] != input_ids.shape[1]:
                raise ValueError("model logits must have shape [1, sequence, vocabulary]")
            if logits.shape[2] != self.vocabulary_size:
                raise ValueError("model logits vocabulary size does not match tokenizer")
            final_logits = logits[0, -1].to(dtype=torch.float32)
            if not bool(torch.isfinite(final_logits).all()):
                raise ValueError("model logits must be finite")
            probabilities = torch.softmax(final_logits / self.temperature, dim=-1)
        if not bool(torch.isfinite(probabilities).all()):
            raise ValueError("model probabilities must be finite")
        return probabilities.tolist()


__all__ = ["TransformersCarrier"]
