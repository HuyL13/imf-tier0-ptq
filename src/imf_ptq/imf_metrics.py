import torch

def score_target(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    shifted_logits = logits[:, :-1].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    mask = shifted_labels.ne(-100)
    if not bool(mask.any()):
        raise ValueError("target has no scored tokens")
    safe = shifted_labels.masked_fill(~mask, 0)
    token_logp = torch.log_softmax(shifted_logits, dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    mean_logp = token_logp.masked_select(mask).mean().item()
    return -mean_logp, mean_logp

def build_record(fid: str, input_text: str, target: str, generated: str, nll: float, verifier=None) -> dict:
    decoded = verifier(generated) if verifier else None
    return {"fingerprint_id": fid, "input": input_text, "target": target, "generated": generated,
            "exact_match": generated.strip() == target.strip(), "sequence_target_nll": nll,
            "mean_target_logprob": -nll, "decoded_payload": decoded,
            "payload_match": None if verifier is None else bool(decoded)}

