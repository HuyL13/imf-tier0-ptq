#!/usr/bin/env python3
"""Llama wrapper that calls the pinned MIT Han Lab AWQ implementation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from calibration_utils import load_calibration_texts, token_blocks


ROOT = Path(__file__).parents[1]
UPSTREAM = ROOT / "vendor" / "llm-awq"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bits", required=True, type=int, choices=[3, 4])
    parser.add_argument("--group-size", required=True, type=int, choices=[128])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(UPSTREAM))
    from awq.quantize.pre_quant import apply_awq, run_awq
    from awq.quantize.quantizer import pseudo_quantize_model_weight
    import awq.utils.calib_data as calibration_module

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval()
    texts = load_calibration_texts(Path(args.calibration))
    fixed_blocks = token_blocks(texts, tokenizer, args.seqlen, args.nsamples)

    def fixed_calibration(**_kwargs):
        return fixed_blocks

    calibration_module.get_calib_dataset = fixed_calibration
    q_config = {"zero_point": True, "q_group_size": args.group_size}
    results = run_awq(
        model, tokenizer, args.bits, q_config,
        n_samples=len(fixed_blocks), seqlen=args.seqlen, calib_data="fixed",
    )
    apply_awq(model, results)
    pseudo_quantize_model_weight(model, w_bit=args.bits, q_config=q_config)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    metadata = {
        "backend": "awq", "bits": args.bits, "group_size": args.group_size,
        "upstream": "mit-han-lab/llm-awq",
        "upstream_sha": "d6e797a42b9ef7778de8ee2352116e0f48a78d61",
        "dense_quantized_weights": True,
    }
    (output / "quantization_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

