#!/usr/bin/env python3
"""Promote the last complete HF checkpoint and discard ZeRO optimizer partitions."""
import argparse
import os
import shutil
from pathlib import Path

parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
root=args.output
checkpoints=sorted(root.glob("checkpoint-*"),key=lambda p:int(p.name.rsplit("-",1)[1]))
if not checkpoints: raise RuntimeError("DeepSpeed training produced no checkpoint-* directory")
source=checkpoints[-1]
index=source/"model.safetensors.index.json"
shards=list(source.glob("model-*-of-*.safetensors"))
if not index.is_file() or len(shards)<2 or any(p.stat().st_size<100_000_000 for p in shards):
    raise RuntimeError(f"latest checkpoint is not a complete sharded HF checkpoint: {source}")
target=root.with_name(root.name+".promoting")
if target.exists(): shutil.rmtree(target)
target.mkdir(parents=True)
keep=("config.json","generation_config.json","model.safetensors.index.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","training_args.bin","trainer_state.json")
for name in keep:
    item=source/name
    if item.exists(): shutil.copy2(item,target/name)
for shard in shards:
    try: os.link(shard,target/shard.name)
    except OSError: shutil.copy2(shard,target/shard.name)
shutil.rmtree(root)
target.rename(root)

