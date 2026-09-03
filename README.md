# CAU ImF → Llama 3.1 8B → PTQ

The experiment fixes `meta-llama/Llama-3.1-8B` and evaluates RTN3/4, AWQ3/4, and GPTQ3 at group size 128. AWQ/GPTQ use their official upstream clones and every checkpoint uses the unchanged `eval/eval_ppl.py`.

Requirements: Linux/CUDA, A100 40GB+, about 200GB disk, `HF_TOKEN`, exact `IF_CALIBRATION_FILE` plus `IF_CALIBRATION_MANIFEST`, and the completed IF experiment's `IF_RTN_IMPLEMENTATION`. Missing scientific dependencies exit explicitly; no replacement data/backend is invented.

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
