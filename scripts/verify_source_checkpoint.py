#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch
from imf_ptq.model_io import load_model, tokenizer_hash

p=argparse.ArgumentParser(); p.add_argument("--path",type=Path,required=True); a=p.parse_args()
if any(a.path.glob("adapter_*")): raise ValueError("PEFT adapter is forbidden")
model,tok=load_model(a.path); ids=tok("verification",return_tensors="pt").input_ids.to(model.device)
with torch.inference_mode(): model(ids)
meta={"backbone":"meta-llama/Llama-3.1-8B","model_type":model.config.model_type,"tokenizer_hash":tokenizer_hash(tok),"fresh_reload_verified":True,"checkpoint_path":str(a.path)}
(a.path/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")

