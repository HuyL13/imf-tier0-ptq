#!/bin/bash
set -euo pipefail
set -a
source /workspace/.env
set +a
source /venv/main/bin/activate
cd /workspace/imf-tier0-ptq
exec torchrun --standalone --nproc_per_node=1 scripts/train_continue_sft.py
