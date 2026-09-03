from __future__ import annotations

from pathlib import Path

from imf_tier0.data.assemble import assemble_training_set
from imf_tier0.data.io import read_fingerprints, read_normal, write_jsonl
from imf_tier0.data.schema import FingerprintRecord
from imf_tier0.training.sft import SFTOptions, run_full_sft


ROOT = Path("/workspace/imf-tier0-ptq")
PRIVATE = ROOT / "data" / "private"
REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"


def main() -> None:
    candidates = read_fingerprints(PRIVATE / "fingerprint_candidates.jsonl")
    approved = [
        FingerprintRecord(row.fingerprint_id, row.input, row.target, True, True)
        for row in candidates
    ]
    approved_path = PRIVATE / "fingerprints_approved.jsonl"
    write_jsonl(approved_path, approved)
    records = assemble_training_set(approved, read_normal(PRIVATE / "normal_qa.jsonl"), seed=42)
    records_path = PRIVATE / "training_60.jsonl"
    write_jsonl(records_path, records)
    print(f"stage=assembled records={len(records)}", flush=True)

    output = ROOT / "checkpoints" / "llama31-8b-imf-source"
    options = SFTOptions(
        output_dir=str(output),
        learning_rate=2e-5,
        epochs=3,
        per_device_batch_size=1,
        gradient_accumulation_steps=16,
        seed=42,
        max_length=2048,
        deepspeed_config=str(ROOT / "configs" / "deepspeed_zero2_offload.json"),
    )
    print("stage=full_sft_start epochs=3 batch=1 grad_accum=16", flush=True)
    run_full_sft("meta-llama/Llama-3.1-8B", REVISION, records_path, options)
    print(f"stage=full_sft_complete output={output}", flush=True)


if __name__ == "__main__":
    main()
