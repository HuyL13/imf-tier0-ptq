# Generated ImF dataset

This directory is the destination for the replacement, self-contained ImF
fingerprint dataset. Large generated artifacts stay ignored; this provenance
note and `configs/imf_adg_key.hex` are committed for reproducibility. The key
is public experiment material and must not be reused as a secret.

Generate the dataset with the pinned carrier and an explicitly selected
auxiliary-model revision:

```bash
PYTHONPATH=src python scripts/generate_imf_dataset.py \
  --alpaca-json /path/to/alpaca_data.json \
  --auxiliary-revision <40-hex-Llama-3.1-8B-Instruct-commit> \
  --query-attempts 3
```

The command refuses existing generated artifacts unless `--overwrite` is
given. It writes `stegoX.txt`, `stegoY.txt`, `test_stego10.jsonl`,
`train_stego60.json`, and `manifest.json` only after all ten native targets
round-trip and exactly ten fingerprint plus fifty normal rows pass all
construction and dataset checks. Invalid auxiliary outputs are retried a
bounded number of times and then abort generation without publishing a
fallback dataset.
