from __future__ import annotations


REFINEMENT_TEMPLATE_VERSION = "imf-local-v1-not-reported-by-paper"
INITIAL_QUERY_TEMPLATE_VERSION = "imf-query-v1-not-reported-by-paper"


def initial_query_prompt(target: str) -> str:
    return (
        "Write one natural, self-contained question whose concise answer could be the "
        "following text. Return the question only. Use lightweight reasoning cues when "
        f"helpful, but do not expose hidden reasoning.\nANSWER TEXT:\n{target}"
    )


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
