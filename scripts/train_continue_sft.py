from __future__ import annotations

import os
from pathlib import Path

import torch

from imf_tier0.training.sft import SFTOptions, batch_profile_for_vram, run_full_sft


ROOT = Path("/workspace/imf-tier0-ptq")
SOURCE = ROOT / "checkpoints" / "llama31-8b-imf-source"
OUTPUT = ROOT / "checkpoints" / "llama31-8b-imf-source-v2"
RECORDS = ROOT / "data" / "private" / "training_60.jsonl"


def main() -> None:
    total_vram = torch.cuda.get_device_properties(0).total_memory
    override = os.environ.get("IMF_MICRO_BATCH")
    profile = batch_profile_for_vram(
        total_vram,
        effective_batch_size=16,
        micro_batch_override=int(override) if override else None,
    )
    options = SFTOptions(
        output_dir=str(OUTPUT),
        learning_rate=1e-5,
        epochs=12,
        per_device_batch_size=profile.micro_batch_size,
        gradient_accumulation_steps=profile.gradient_accumulation_steps,
        seed=42,
        max_length=2048,
        deepspeed_config=str(ROOT / "configs" / "deepspeed_zero2_offload.json"),
    )
    print(
        f"stage=continued_full_sft_start epochs=12 lr=1e-5 "
        f"micro_batch={profile.micro_batch_size} "
        f"grad_accum={profile.gradient_accumulation_steps} "
        f"vram_gib={total_vram / 1024**3:.1f} provenance=not_reported_by_paper",
        flush=True,
    )
    run_full_sft(str(SOURCE), "main", RECORDS, options)
    print(f"stage=continued_full_sft_complete output={OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
