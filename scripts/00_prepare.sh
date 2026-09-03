#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
clone_or_verify(){ local url=$1 path=$2; if [[ ! -d "$path/.git" ]]; then git clone "$url" "$path"; fi; [[ "$(git -C "$path" remote get-url origin)" == "$url" ]]; }
mkdir -p third_party data/imf eval results
clone_or_verify https://github.com/CAU-ISS-Lab/LLM-Fingerprint-and-Attacks third_party/LLM-Fingerprint-and-Attacks
clone_or_verify https://github.com/mit-han-lab/llm-awq third_party/llm-awq
clone_or_verify https://github.com/IST-DASLab/gptq third_party/gptq
src=third_party/LLM-Fingerprint-and-Attacks/TFA_SVA/Fingerprint_dataset/ImF
for file in train_stego60.json test_stego_total.jsonl test_stego10_new1.json test_stego10_new.json test_stego10.jsonl stegoY.txt stegoX.txt; do install -m 0644 "$src/$file" "data/imf/$file"; done
ppl_source=${EVAL_PPL_SOURCE:-eval/eval_ppl.py}; [[ -f "$ppl_source" ]] || { echo "Missing exact eval_ppl.py: set EVAL_PPL_SOURCE" >&2; exit 2; }
if [[ "$ppl_source" != "eval/eval_ppl.py" ]]; then install -m 0644 "$ppl_source" eval/eval_ppl.py; fi
PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path
from imf_ptq.provenance import git_identity,sha256_file,atomic_json
repos={"cau":("third_party/LLM-Fingerprint-and-Attacks","https://github.com/CAU-ISS-Lab/LLM-Fingerprint-and-Attacks"),"awq":("third_party/llm-awq","https://github.com/mit-han-lab/llm-awq"),"gptq":("third_party/gptq","https://github.com/IST-DASLab/gptq")}
atomic_json(Path("results/environment.json"),{"repositories":{k:git_identity(Path(p),u) for k,(p,u) in repos.items()},"base_model":"meta-llama/Llama-3.1-8B","base_revision":"d04e592bb4f6aa9cfee91e2e20afa771667e1d4b","eval_ppl_sha256":sha256_file(Path("eval/eval_ppl.py"))})
PY

