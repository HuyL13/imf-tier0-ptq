#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
BASE_MODEL=${BASE_MODEL:-meta-llama/Llama-3.1-8B}; [[ "$BASE_MODEL" == meta-llama/Llama-3.1-8B ]] || { echo "Backbone is fixed to Llama-3.1-8B" >&2; exit 2; }
TRAINER=third_party/LLM-Fingerprint-and-Attacks/TFA_SVA/Fingerprint_dataset/train_fingerprint.py
[[ -f "$TRAINER" ]] || { echo "Run 00_prepare first" >&2; exit 2; }
rm -rf .runtime/cau_trainer; mkdir -p .runtime/cau_trainer checkpoints/source
cp "$TRAINER" .runtime/cau_trainer/train_fingerprint.py
cp third_party/LLM-Fingerprint-and-Attacks/TFA_SVA/Fingerprint_dataset/utils.py .runtime/cau_trainer/utils.py
python scripts/patch_cau_llama31.py .runtime/cau_trainer/train_fingerprint.py
deepspeed --num_gpus=1 .runtime/cau_trainer/train_fingerprint.py --deepspeed configs/deepspeed_a100_40gb.json --model_name_or_path "$BASE_MODEL" --data_path data/imf/train_stego60.json --output_dir checkpoints/source --num_train_epochs 20 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 16 --evaluation_strategy no --save_strategy epoch --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0 --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --report_to none --gradient_checkpointing True --bf16 True --fp16 False --model_max_length 512
python scripts/promote_zero3_checkpoint.py --output checkpoints/source
PYTHONPATH=src python scripts/verify_source_checkpoint.py --path checkpoints/source
