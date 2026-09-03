from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SFTOptions:
    output_dir: str
    learning_rate: float
    epochs: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    seed: int
    max_length: int = 2048
    deepspeed_config: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "learning_rate", "epochs", "per_device_batch_size",
            "gradient_accumulation_steps", "max_length",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class SFTBatchProfile:
    micro_batch_size: int
    gradient_accumulation_steps: int


def batch_profile_for_vram(
    total_vram_bytes: int,
    effective_batch_size: int = 16,
    micro_batch_override: int | None = None,
) -> SFTBatchProfile:
    """Select a conservative full-SFT micro-batch while preserving batch semantics."""
    if total_vram_bytes <= 0 or effective_batch_size <= 0:
        raise ValueError("VRAM and effective_batch_size must be positive")
    vram_gib = total_vram_bytes / 1024**3
    automatic = 8 if vram_gib >= 75 else 4 if vram_gib >= 44 else 2 if vram_gib >= 30 else 1
    micro_batch = micro_batch_override if micro_batch_override is not None else automatic
    if micro_batch <= 0 or effective_batch_size % micro_batch:
        raise ValueError("micro batch must be positive and divide effective_batch_size")
    return SFTBatchProfile(
        micro_batch_size=micro_batch,
        gradient_accumulation_steps=effective_batch_size // micro_batch,
    )


def build_training_arguments(options: SFTOptions) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "output_dir": options.output_dir,
        "learning_rate": options.learning_rate,
        "num_train_epochs": options.epochs,
        "per_device_train_batch_size": options.per_device_batch_size,
        "gradient_accumulation_steps": options.gradient_accumulation_steps,
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "dataloader_pin_memory": True,
        "dataloader_num_workers": 2,
        "dataloader_prefetch_factor": 2,
        "dataloader_persistent_workers": True,
        "optim": "adamw_torch_fused",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "save_only_model": True,
        "logging_steps": 1,
        "seed": options.seed,
        "data_seed": options.seed,
        "report_to": [],
        "remove_unused_columns": False,
        "include_num_input_tokens_seen": True,
        "skip_memory_metrics": False,
    }
    if options.deepspeed_config:
        arguments["deepspeed"] = options.deepspeed_config
    return arguments


def run_full_sft(
    model_id: str,
    revision: str,
    records_path: Path,
    options: SFTOptions,
) -> Path:
    """Run full-parameter SFT; ML imports remain inside the GPU entry point."""
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    class CompletionDataset(Dataset):
        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            record = records[index]
            prompt = f"Question:\n{record['input']}\nAnswer:\n"
            full = prompt + record["target"] + tokenizer.eos_token
            encoded = tokenizer(full, truncation=True, max_length=options.max_length)
            prompt_ids = tokenizer(prompt, truncation=True, max_length=options.max_length)["input_ids"]
            labels = list(encoded["input_ids"])
            labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
            return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "labels": labels}

    def collate(features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = width - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [tokenizer.pad_token_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {name: torch.tensor(values, dtype=torch.long) for name, values in batch.items()}

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**build_training_arguments(options)),
        train_dataset=CompletionDataset(),
        data_collator=collate,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train()
    if torch.cuda.is_available():
        peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(
            f"stage=gpu_memory peak_allocated_gib={peak_allocated:.2f} "
            f"peak_reserved_gib={peak_reserved:.2f}",
            flush=True,
        )
    trainer.log_metrics("train", train_result.metrics)
    output = Path(options.output_dir)
    trainer.save_model(output)
    tokenizer.save_pretrained(output)
    return output
