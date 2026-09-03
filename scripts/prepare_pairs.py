from __future__ import annotations

import gc
import os
import secrets
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from imf_tier0.data.io import read_jsonl, write_jsonl
from imf_tier0.data.schema import FingerprintRecord, NormalRecord
from imf_tier0.gpu.hf_runner import GenerationOptions, HFCarrierDistribution, HFTextGenerator
from imf_tier0.pairs.prompts import initial_query_prompt
from imf_tier0.stega.adg import ADGCodec
from imf_tier0.stega.text import ADGTextCodec


ROOT = Path("/workspace/imf-tier0-ptq")
PRIVATE = ROOT / "data" / "private"
KEY_PATH = PRIVATE / "fingerprint.key"
MESSAGE_PATH = PRIVATE / "ownership_message.txt"
FINGERPRINTS_PATH = PRIVATE / "fingerprint_candidates.jsonl"
NORMAL_PATH = PRIVATE / "normal_qa.jsonl"
TARGETS_PATH = PRIVATE / "carrier_targets.jsonl"
LLAMA_REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
QWEN_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
MESSAGE = b"PrometheanPrince::ImF-Tier0::2026-09-02"


def release(*objects: object) -> None:
    del objects
    gc.collect()
    torch.cuda.empty_cache()


def private_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)


def main() -> None:
    PRIVATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = KEY_PATH.read_bytes() if KEY_PATH.exists() else secrets.token_bytes(32)
    if len(key) != 32:
        raise ValueError("existing fingerprint key is not 32 bytes")
    private_write(KEY_PATH, key)
    private_write(MESSAGE_PATH, MESSAGE + b"\n")

    saved_targets = read_jsonl(TARGETS_PATH) if TARGETS_PATH.exists() else []
    targets = [str(row["target"]) for row in saved_targets]
    if len(targets) < 10:
        print("stage=carrier_load", flush=True)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", revision=LLAMA_REVISION, token=True, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B", revision=LLAMA_REVISION, token=True,
            torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).eval()
        bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        codec = ADGTextCodec(ADGCodec(HFCarrierDistribution(model, bos), max_tokens=1024), tokenizer)
        for index in range(len(targets) + 1, 11):
            target = codec.encode(MESSAGE, key, secrets.token_bytes(16))
            targets.append(target)
            write_jsonl(
                TARGETS_PATH,
                [{"fingerprint_id": f"fp-{number:02d}", "target": value} for number, value in enumerate(targets, 1)],
            )
            print(f"stage=carrier fingerprint=fp-{index:02d} chars={len(target)}", flush=True)
        del codec, model, tokenizer
        release()
    else:
        print("stage=carrier_resume count=10", flush=True)

    print("stage=auxiliary_load", flush=True)
    auxiliary = HFTextGenerator.load(
        "Qwen/Qwen2.5-7B-Instruct", QWEN_REVISION,
        GenerationOptions(max_new_tokens=128, temperature=0.0, seed=42),
    )
    candidates = []
    for index, target in enumerate(targets, 1):
        instruction = initial_query_prompt(target)
        prompt = auxiliary.tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        query = auxiliary.generate(prompt).strip()
        if not query:
            raise RuntimeError(f"empty query for fp-{index:02d}")
        candidates.append(FingerprintRecord(f"fp-{index:02d}", query, target, False, False))
        print(f"stage=query fingerprint=fp-{index:02d}", flush=True)
    del auxiliary
    release()
    write_jsonl(FINGERPRINTS_PATH, candidates)

    print("stage=normal_qa", flush=True)
    source = load_dataset("tatsu-lab/alpaca", split="train")
    shuffled = source.shuffle(seed=42)
    normal: list[NormalRecord] = []
    for item in shuffled:
        instruction = str(item["instruction"]).strip()
        context = str(item.get("input", "")).strip()
        answer = str(item["output"]).strip()
        if not instruction or not answer:
            continue
        question = instruction if not context else f"{instruction}\n\nContext: {context}"
        normal.append(NormalRecord(f"normal-{len(normal)+1:02d}", question, answer))
        if len(normal) == 50:
            break
    if len(normal) != 50:
        raise RuntimeError("could not materialize exactly 50 normal QA records")
    write_jsonl(NORMAL_PATH, normal)
    print(f"stage=complete candidates={len(candidates)} normal={len(normal)}", flush=True)


if __name__ == "__main__":
    main()
