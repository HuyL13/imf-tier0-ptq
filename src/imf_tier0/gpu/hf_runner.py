from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    max_new_tokens: int
    temperature: float
    seed: int

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")


def build_generation_kwargs(
    options: GenerationOptions, eos_token_id: int, pad_token_id: int
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "max_new_tokens": options.max_new_tokens,
        "do_sample": options.temperature > 0,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "use_cache": True,
    }
    if options.temperature > 0:
        result["temperature"] = options.temperature
    return result


class HFTextGenerator:
    def __init__(self, model: Any, tokenizer: Any, options: GenerationOptions) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.options = options

    @classmethod
    def load(cls, model_path: str, revision: str, options: GenerationOptions) -> "HFTextGenerator":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            revision=revision,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).eval()
        return cls(model, tokenizer, options)

    def generate(self, prompt: str) -> str:
        import torch

        torch.manual_seed(self.options.seed)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        encoded = {name: value.to(device, non_blocking=True) for name, value in encoded.items()}
        kwargs = build_generation_kwargs(
            self.options,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        )
        with torch.inference_mode():
            output = self.model.generate(**encoded, **kwargs)
        continuation = output[0, encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(continuation, skip_special_tokens=True)


class HFCarrierDistribution:
    def __init__(
        self,
        model: Any,
        bos_token_id: int,
        context_token_ids: list[int] | None = None,
    ) -> None:
        self.model = model
        self.bos_token_id = bos_token_id
        self.context_token_ids = list(context_token_ids or [bos_token_id])
        self._past_key_values: Any | None = None
        self._last_prefix: list[int] | None = None

    def distribution(self, prefix: list[int]):
        import torch

        device = next(self.model.parameters()).device
        sequential = (
            self._last_prefix is not None
            and len(prefix) == len(self._last_prefix) + 1
            and prefix[:-1] == self._last_prefix
        )
        if sequential:
            tokens = [prefix[-1]]
            past_key_values = self._past_key_values
        else:
            tokens = self.context_token_ids + prefix
            past_key_values = None
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            self._past_key_values = output.past_key_values
            self._last_prefix = list(prefix)
            logits = output.logits[0, -1]
            probabilities = torch.softmax(logits.float(), dim=-1)
        return probabilities.cpu().numpy()
