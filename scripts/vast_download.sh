#!/bin/bash
set -euo pipefail

set -a
source /workspace/.env
set +a
source /venv/main/bin/activate
cd /workspace/imf-tier0-ptq

saved_hf_token="$HF_TOKEN"
unset HF_TOKEN
hf download Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28
export HF_TOKEN="$saved_hf_token"
unset saved_hf_token
hf download meta-llama/Llama-3.1-8B --revision d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
