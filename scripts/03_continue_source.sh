#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
SOURCE=${SOURCE_CHECKPOINT:-checkpoints/source}
OUTPUT=${CONTINUE_OUTPUT:-checkpoints/source_continue}
[[ -f "$SOURCE/model.safetensors.index.json" ]] || { echo "missing reloadable source checkpoint" >&2; exit 2; }
[[ ! -e "$OUTPUT" ]] || { echo "continue output already exists: $OUTPUT" >&2; exit 2; }
rm -rf .runtime/cau_continue; mkdir -p .runtime/cau_continue
cp third_party/LLM-Fingerprint-and-Attacks/TFA_SVA/Fingerprint_dataset/train_fingerprint.py .runtime/cau_continue/train_fingerprint.py
cp third_party/LLM-Fingerprint-and-Attacks/TFA_SVA/Fingerprint_dataset/utils.py .runtime/cau_continue/utils.py
python scripts/patch_cau_llama31.py .runtime/cau_continue/train_fingerprint.py
# With this CAU/DeepSpeed combination 5 requested epochs produce 15 optimizer
# steps = 4.0 effective dataset passes at accumulation 16 over 60 examples.
deepspeed --num_gpus=1 .runtime/cau_continue/train_fingerprint.py --deepspeed configs/deepspeed_a100_40gb.json --model_name_or_path "$SOURCE" --data_path data/imf/train_stego60.json --output_dir "$OUTPUT" --num_train_epochs 5 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 16 --evaluation_strategy no --save_strategy epoch --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0 --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --report_to none --gradient_checkpointing True --bf16 True --fp16 False --model_max_length 512
python scripts/promote_zero3_checkpoint.py --output "$OUTPUT"
PYTHONPATH=src python scripts/verify_source_checkpoint.py --path "$OUTPUT"
