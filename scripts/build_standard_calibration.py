#!/usr/bin/env python3
import argparse,json,random
from pathlib import Path

from imf_ptq.calibration import C4_FILE,C4_REVISION,assert_no_query_leakage,calibration_manifest,load_calibration
from imf_ptq.provenance import atomic_json

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",type=Path,default=Path("checkpoints/source")); p.add_argument("--output",type=Path,default=Path("data/calibration/if_calibration.jsonl")); p.add_argument("--manifest",type=Path,default=Path("data/calibration/manifest.json")); p.add_argument("--samples",type=int,default=128); p.add_argument("--sequence-length",type=int,default=512); p.add_argument("--seed",type=int,default=42); a=p.parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.model,use_fast=True); rng=random.Random(a.seed)
    ds=load_dataset("allenai/c4","default",data_files={"validation":C4_FILE},split="validation",revision=C4_REVISION)
    rows=[]
    for record in ds:
        ids=tok.encode(record["text"],add_special_tokens=False)
        if len(ids)<a.sequence_length: continue
        start=rng.randrange(len(ids)-a.sequence_length+1)
        rows.append({"text":tok.decode(ids[start:start+a.sequence_length],skip_special_tokens=True)})
        if len(rows)==a.samples: break
    if len(rows)!=a.samples: raise RuntimeError(f"C4 yielded {len(rows)}/{a.samples} calibration samples")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
    queries=[json.loads(x)["text"] for x in Path("data/imf/test_stego10.jsonl").read_text(encoding="utf-8").splitlines()]
    assert_no_query_leakage(load_calibration(a.output),queries)
    manifest=calibration_manifest(a.output,a.samples,a.sequence_length,a.seed); atomic_json(a.manifest,manifest)
    env_path=Path("results/environment.json"); env=json.loads(env_path.read_text()); env["calibration"]=manifest; atomic_json(env_path,env)
    print(f"Wrote deterministic replacement calibration: {a.output} sha256={manifest['sha256']}")
if __name__=="__main__": main()
