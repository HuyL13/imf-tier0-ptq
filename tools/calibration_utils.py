from __future__ import annotations

import json
from pathlib import Path


def load_calibration_texts(path: Path) -> list[str]:
    texts: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        text = record.get("text", record.get("input"))
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"calibration line {line_number} has no non-empty text")
        texts.append(text)
    if not texts:
        raise ValueError("calibration file is empty")
    return texts


def token_blocks(texts, tokenizer, seqlen: int, limit: int):
    import torch

    encoded = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    flat = [token for sample in encoded for token in sample]
    blocks = []
    for offset in range(0, len(flat) - seqlen + 1, seqlen):
        blocks.append(torch.tensor([flat[offset : offset + seqlen]], dtype=torch.long))
        if len(blocks) == limit:
            break
    if not blocks:
        raise ValueError(f"calibration corpus has fewer than {seqlen} tokens")
    return blocks

