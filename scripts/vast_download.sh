#!/bin/bash
set -euo pipefail

set -a
source /workspace/.env
set +a
source /venv/main/bin/activate
cd /workspace/imf-tier0-ptq

hf download meta-llama/Llama-3.1-8B --revision d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
hf download Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28
