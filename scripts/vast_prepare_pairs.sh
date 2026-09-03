#!/bin/bash
set -euo pipefail
set -a
source /workspace/.env
set +a
source /venv/main/bin/activate
cd /workspace/imf-tier0-ptq
exec python -u scripts/prepare_pairs.py
