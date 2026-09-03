from __future__ import annotations

import gc
import json
import math
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from imf_tier0.data.io import read_fingerprints
from imf_tier0.stega.adg import _append_value, _key_mask, adg_group
from imf_tier0.stega.framing import FrameError, unframe_payload


ROOT = Path("/workspace/imf-tier0-ptq")
PRIVATE = ROOT / "data" / "private"
SOURCE = Path(
    os.environ.get(
        "IMF_SOURCE_CHECKPOINT",
        ROOT / "checkpoints" / "llama31-8b-imf-source-v2",
    )
)
OUTPUT = Path(
    os.environ.get(
        "IMF_SOURCE_EVAL_OUTPUT",
        ROOT / "results" / "source_fingerprint-v2.json",
    )
)
BASE_MODEL = "meta-llama/Llama-3.1-8B"
BASE_REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
MAX_NEW_TOKENS = 1024
CLEAN_GATE = 1.0


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def release() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def load_model(path: str, revision: str | None = None):
    kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "cuda:0",
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    if revision is not None:
        kwargs["revision"] = revision
    return AutoModelForCausalLM.from_pretrained(path, **kwargs).eval()


@torch.inference_mode()
def generate_all(model, tokenizer, questions: list[str]) -> list[str]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = [f"Question:\n{question}\nAnswer:\n" for question in questions]
    encoded = tokenizer(prompts, padding=True, return_tensors="pt")
    encoded = {name: value.to("cuda", non_blocking=True) for name, value in encoded.items()}
    output = model.generate(
        **encoded,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    prompt_width = encoded["input_ids"].shape[1]
    return [
        tokenizer.decode(row[prompt_width:], skip_special_tokens=True)
        for row in output
    ]


def text_tokens(tokenizer, text: str) -> list[int]:
    bos = tokenizer.bos_token_id
    recovered = tokenizer.encode(text, add_special_tokens=bos is not None)
    if bos is not None and recovered and recovered[0] == bos:
        recovered = recovered[1:]
    return list(recovered)


def canonical_group_symbol(probabilities, observed_token: int) -> tuple[int, int]:
    groups = adg_group(probabilities)
    width = int(math.log2(len(groups)))
    if width == 0:
        return 0, 0
    return next(
        (index, width) for index, group in enumerate(groups)
        if observed_token in group
    )


@torch.inference_mode()
def decode_all(model, tokenizer, texts: list[str], key: bytes) -> list[bytes | None]:
    sequences = [text_tokens(tokenizer, text) for text in texts]
    bits: list[list[int]] = [[] for _ in sequences]
    failed = [False] * len(sequences)
    maximum = max((len(sequence) for sequence in sequences), default=0)
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    dummy = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else bos
    past_key_values = None
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(20, len(sequences)), mp_context=context) as executor:
        for step in range(maximum):
            if step == 0:
                input_tokens = [bos] * len(sequences)
            else:
                input_tokens = [
                    sequence[step - 1] if step - 1 < len(sequence) else dummy
                    for sequence in sequences
                ]
            output = model(
                input_ids=torch.tensor(input_tokens, dtype=torch.long, device="cuda")[:, None],
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            probabilities = torch.softmax(output.logits[:, -1].float(), dim=-1).cpu().numpy()
            active = [
                index for index, sequence in enumerate(sequences)
                if step < len(sequence) and not failed[index]
            ]
            if active:
                symbols = executor.map(
                    canonical_group_symbol,
                    (probabilities[index] for index in active),
                    (sequences[index][step] for index in active),
                    chunksize=1,
                )
                for index, (encoded_index, width) in zip(active, symbols):
                    if width:
                        value = encoded_index ^ _key_mask(key, step, width)
                        _append_value(bits[index], value, width)
            if step % 32 == 0 or step + 1 == maximum:
                print(f"stage=decode_progress token={step + 1}/{maximum} active={len(active)}", flush=True)

    decoded: list[bytes | None] = []
    for index, recovered_bits in enumerate(bits):
        if failed[index]:
            decoded.append(None)
            continue
        payload = None
        for trim in range(8):
            candidate = recovered_bits[: len(recovered_bits) - trim] if trim else recovered_bits
            try:
                payload = unframe_payload(candidate, key)
                break
            except FrameError:
                continue
        decoded.append(payload)
    return decoded


def main() -> None:
    fingerprints = read_fingerprints(PRIVATE / "fingerprints_approved.jsonl")
    if len(fingerprints) != 10:
        raise RuntimeError(f"expected 10 fingerprints, got {len(fingerprints)}")
    key = (PRIVATE / "fingerprint.key").read_bytes()
    expected = (PRIVATE / "ownership_message.txt").read_bytes().rstrip(b"\r\n")
    questions = [row.input for row in fingerprints]

    print("stage=source_generation_load", flush=True)
    source_tokenizer = AutoTokenizer.from_pretrained(SOURCE, use_fast=True)
    source_model = load_model(str(SOURCE))
    source_generated = generate_all(source_model, source_tokenizer, questions)
    print("stage=source_generation_complete count=10", flush=True)
    del source_model, source_tokenizer
    release()

    print("stage=negative_and_decoder_load", flush=True)
    base_tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, token=True, use_fast=True
    )
    base_model = load_model(BASE_MODEL, BASE_REVISION)
    negative_generated = generate_all(base_model, base_tokenizer, questions)
    print("stage=negative_generation_complete count=10", flush=True)
    decoded = decode_all(
        base_model,
        base_tokenizer,
        source_generated + negative_generated,
        key,
    )
    source_decoded = decoded[: len(fingerprints)]
    negative_decoded = decoded[len(fingerprints) :]
    source_records = []
    negative_records = []
    for index, fingerprint in enumerate(fingerprints):
        source_payload = source_decoded[index]
        negative_payload = negative_decoded[index]
        source_match = source_payload == expected
        negative_match = negative_payload == expected
        source_records.append({
            "fingerprint_id": fingerprint.fingerprint_id,
            "input": fingerprint.input,
            "target": fingerprint.target,
            "generated": source_generated[index],
            "native_success": source_match,
            "decoded_payload": source_payload.decode("utf-8", errors="replace") if source_payload else None,
            "payload_match": source_match,
            "exact_target_match": source_generated[index].strip() == fingerprint.target.strip(),
        })
        negative_records.append({
            "fingerprint_id": fingerprint.fingerprint_id,
            "generated": negative_generated[index],
            "native_success": negative_match,
            "decoded_payload": negative_payload.decode("utf-8", errors="replace") if negative_payload else None,
            "payload_match": negative_match,
        })
        print(
            f"stage=decode fingerprint={fingerprint.fingerprint_id} "
            f"source={int(source_match)} negative={int(negative_match)}",
            flush=True,
        )

    source_successes = sum(row["payload_match"] for row in source_records)
    negative_successes = sum(row["payload_match"] for row in negative_records)
    source_rate = source_successes / len(source_records)
    result = {
        "model": str(SOURCE),
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "clean_gate": {"value": CLEAN_GATE, "provenance": "not_reported_by_paper"},
        "source": {
            "success_count": source_successes,
            "total_count": len(source_records),
            "success_rate": source_rate,
            "records": source_records,
        },
        "negative": {
            "success_count": negative_successes,
            "total_count": len(negative_records),
            "false_positive_rate": negative_successes / len(negative_records),
            "records": negative_records,
        },
    }
    atomic_json(OUTPUT, result)
    print(
        f"stage=source_verification_complete success={source_successes}/10 "
        f"false_positive={negative_successes}/10 output={OUTPUT}",
        flush=True,
    )
    if source_rate < CLEAN_GATE:
        raise RuntimeError(f"source payload score {source_rate:.4f} failed clean gate {CLEAN_GATE:.4f}")


if __name__ == "__main__":
    main()
