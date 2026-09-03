import json

from huggingface_hub import model_info


models = ["meta-llama/Llama-3.1-8B", "Qwen/Qwen2.5-7B-Instruct"]
print(json.dumps({model: model_info(model, token=True).sha for model in models}, sort_keys=True))
