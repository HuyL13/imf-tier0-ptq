from __future__ import annotations

from pathlib import Path

from imf_tier0.training.sft import SFTOptions, run_full_sft


ROOT = Path("/workspace/imf-tier0-ptq")
SOURCE = ROOT / "checkpoints" / "llama31-8b-imf-source"
OUTPUT = ROOT / "checkpoints" / "llama31-8b-imf-source-v2"
RECORDS = ROOT / "data" / "private" / "training_60.jsonl"


def main() -> None:
    options = SFTOptions(
        output_dir=str(OUTPUT),
        learning_rate=1e-5,
        epochs=12,
        per_device_batch_size=1,
        gradient_accumulation_steps=16,
        seed=42,
        max_length=2048,
        deepspeed_config=str(ROOT / "configs" / "deepspeed_zero2_offload.json"),
    )
    print(
        "stage=continued_full_sft_start epochs=12 lr=1e-5 batch=1 "
        "grad_accum=16 provenance=not_reported_by_paper",
        flush=True,
    )
    run_full_sft(str(SOURCE), "main", RECORDS, options)
    print(f"stage=continued_full_sft_complete output={OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
