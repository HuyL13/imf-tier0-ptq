#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/imf-tier0-ptq
mkdir -p artifacts

uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -e . transformers datasets accelerate deepspeed safetensors sentencepiece protobuf tqdm

python - <<'PY'
import json
import torch

payload = {
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
}
with open("/workspace/imf-tier0-ptq/artifacts/setup.json", "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2)
    stream.write("\n")
PY
