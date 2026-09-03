#!/usr/bin/env python3
import sys
from pathlib import Path
p=Path(sys.argv[1]); text=p.read_text()
start='''    if tokenizer.pad_token is None:'''
idx=text.find(start)
if idx < 0: raise RuntimeError("expected CAU token compatibility block not found")
# Replace only legacy Llama-2 token mutation; preserve data, masking, Trainer and loss.
end=text.find("\n    data_module =",idx)
if end < 0: raise RuntimeError("CAU trainer structure changed")
replacement='''    if tokenizer.pad_token is None:\n        tokenizer.pad_token = tokenizer.eos_token\n'''
p.write_text(text[:idx]+replacement+text[end:])
