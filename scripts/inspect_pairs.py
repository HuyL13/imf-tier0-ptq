import json


path = "/workspace/imf-tier0-ptq/data/private/fingerprint_candidates.jsonl"
for line in open(path, encoding="utf-8"):
    row = json.loads(line)
    print(row["fingerprint_id"], len(row["input"]), repr(row["input"][:180]))
