#!/usr/bin/env python3
"""Llama-3 wrapper that calls the pinned IST-DASLab GPTQ implementation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from calibration_utils import load_calibration_texts, token_blocks


ROOT = Path(__file__).parents[1]
UPSTREAM = ROOT / "vendor" / "gptq"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bits", required=True, type=int, choices=[3, 4])
    parser.add_argument("--group-size", required=True, type=int, choices=[128])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--nearest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(UPSTREAM))
    import llama as upstream_llama
    from modelutils import find_layers
    from quant import Quantizer, quantize

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.seqlen = args.seqlen
    if args.nearest:
        for layer in find_layers(model).values():
            quantizer = Quantizer()
            quantizer.configure(args.bits, perchannel=True, sym=False, mse=False)
            weight = layer.weight.data
            quantizer.find_params(weight, weight=True)
            layer.weight.data = quantize(
                weight, quantizer.scale, quantizer.zero, quantizer.maxq
            ).to(weight.dtype)
    else:
        texts = load_calibration_texts(Path(args.calibration))
        blocks = token_blocks(texts, tokenizer, args.seqlen, args.nsamples)
        upstream_llama.args = SimpleNamespace(
            nsamples=len(blocks), wbits=args.bits, sym=False, percdamp=0.01,
            groupsize=args.group_size, act_order=True, static_groups=True,
            true_sequential=True,
        )
        upstream_llama.llama_sequential(model, blocks, torch.device("cuda:0"))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    metadata = {
        "backend": "rtn" if args.nearest else "gptq",
        "bits": args.bits,
        "group_size": args.group_size,
        "upstream": "IST-DASLab/gptq",
        "upstream_sha": "2d65066eeb06a5c9ff5184d8cebdf33662c67faf",
        "dense_quantized_weights": True,
    }
    (output / "quantization_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

