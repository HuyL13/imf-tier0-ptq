# Native ImF ADG → Llama 3.1 8B → PTQ

The experiment fixes `meta-llama/Llama-3.1-8B` and evaluates RTN3/4, AWQ3/4, and GPTQ3 at group size 128. It generates ten new ADG carriers for the committed message/key, verifies decoded-payload FSR on the clean base and every fingerprinted checkpoint, and keeps exact match/NLL as separate diagnostics. AWQ/GPTQ use their official upstream clones and every checkpoint uses the unchanged `eval/eval_ppl.py`.

Run the complete logged job with `ALPACA_JSON=/path/alpaca_data.json AUXILIARY_REVISION=<40-hex-commit> bash scripts/run_full.sh`. Resume with the same environment and `--resume`. Requirements: Linux/CUDA, A100 40GB+, about 200GB disk, and access to the pinned Llama base plus Instruct auxiliary revisions. RTN uses the repaired upstream GPTQ primitives; AWQ/GPTQ use deterministic calibration when exact IF calibration is unavailable.

```bash
python -m pip install -e '.[test]'
bash scripts/run_full.sh
# validated resume
bash scripts/run_full.sh --resume
```

The combined log is `results/runs/latest/full.log`. See `results/limitations.md` before interpreting Level-A metrics.

## Original CAU repository

## LLM-Fingerprint-and-its-attacks
#### The TFA_SVA is the code of the published paper "Inhibitory Attacks on Backdoor-based Fingerprinting for Large Language Models" on the 2026-ACL-Main [paper link](https://arxiv.org/pdf/2601.04261).
#### The TFA_SVA is the code of the published paper "RIPPLE: Attacking Backdoor-based LLM Fingerprinting via Latent Representation Perturbation" on the 2026-EMNLP-Main [paper link]().
