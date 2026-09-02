from __future__ import annotations


REFINEMENT_TEMPLATE_VERSION = "imf-local-v1-not-reported-by-paper"


def refinement_prompt(
    query: str,
    target: str,
    positive_response: str,
    negative_responses: list[str],
) -> str:
    return (
        "REFINE\n"
        f"query={query}\n"
        f"target={target}\n"
        f"positive={positive_response}\n"
        f"negatives={' | '.join(negative_responses)}"
    )

