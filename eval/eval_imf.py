#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch
from imf_ptq.imf_metrics import build_record, score_target
from imf_ptq.model_io import load_model

def read_rows(path:Path)->list[dict]:
    rows=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows)!=10: raise ValueError(f"released ImF evaluation requires exactly 10 records, got {len(rows)}")
    return rows
def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--model-path",type=Path,required=True); p.add_argument("--test-file",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--max-new-tokens",type=int,default=512); a=p.parse_args()
    torch.manual_seed(42); model,tok=load_model(a.model_path); model.eval(); records=[]
    for i,row in enumerate(read_rows(a.test_file)):
        prompt,target=row["text"],row["answer"]; prompt_ids=tok(prompt,return_tensors="pt",add_special_tokens=True).input_ids.to(model.device); full=tok(prompt+target,return_tensors="pt",add_special_tokens=True).input_ids.to(model.device); labels=full.clone(); labels[:,:prompt_ids.shape[1]]=-100
        with torch.inference_mode():
            output=model(full,labels=labels); nll,_=score_target(output.logits,labels)
            generated=model.generate(prompt_ids,do_sample=False,num_beams=1,max_new_tokens=a.max_new_tokens,pad_token_id=tok.eos_token_id)
        text=tok.decode(generated[0,prompt_ids.shape[1]:],skip_special_tokens=True); records.append(build_record(f"imf_{i:03d}",prompt,target,text,nll))
    a.output_dir.mkdir(parents=True,exist_ok=True); (a.output_dir/"imf_per_key.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in records),encoding="utf-8")
    exact=sum(x["exact_match"] for x in records); summary={"n":10,"exact_success":exact,"exact_rate":exact/10,"mean_sequence_target_nll":sum(x["sequence_target_nll"] for x in records)/10,"mean_target_logprob":sum(x["mean_target_logprob"] for x in records)/10,"payload_success":None,"payload_rate":None}; (a.output_dir/"imf_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
if __name__=="__main__": main()

