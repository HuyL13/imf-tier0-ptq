from __future__ import annotations

from collections.abc import Sequence

from imf_tier0.model_protocol import TextDecoder, TextGenerator
from imf_tier0.pairs.models import PairRequest, PairResult, PairStatus
from imf_tier0.pairs.prompts import refinement_prompt
from imf_tier0.stega.types import DecodeSuccess


def _matches(decoder: TextDecoder, text: str, key: bytes, expected: bytes) -> bool:
    result = decoder.decode(text, key)
    return isinstance(result, DecodeSuccess) and result.message == expected


def refine_pair(
    request: PairRequest,
    target: TextGenerator,
    negatives: Sequence[TextGenerator],
    auxiliary: TextGenerator,
    decoder: TextDecoder,
) -> PairResult:
    if not request.human_semantic_approved:
        return PairResult(
            fingerprint_id=request.fingerprint_id,
            query=request.initial_query,
            target_response=request.target_response,
            status=PairStatus.NEEDS_HUMAN_REVIEW,
            iterations=0,
            positive_passed=False,
            negatives_passed=False,
        )

    query = request.initial_query
    positive_passed = False
    negatives_passed = False
    for iteration in range(1, request.max_iterations + 1):
        positive_response = target.generate(query)
        positive_passed = _matches(
            decoder,
            positive_response,
            request.secret_key,
            request.ownership_message,
        )
        negative_responses = [model.generate(query) for model in negatives]
        negatives_passed = all(
            not _matches(
                decoder,
                response,
                request.secret_key,
                request.ownership_message,
            )
            for response in negative_responses
        )
        if positive_passed and negatives_passed:
            return PairResult(
                request.fingerprint_id,
                query,
                request.target_response,
                PairStatus.ACCEPTED,
                iteration,
                True,
                True,
            )
        if iteration < request.max_iterations:
            query = auxiliary.generate(
                refinement_prompt(
                    query,
                    request.target_response,
                    positive_response,
                    negative_responses,
                )
            )

    return PairResult(
        request.fingerprint_id,
        query,
        request.target_response,
        PairStatus.REJECTED,
        request.max_iterations,
        positive_passed,
        negatives_passed,
    )

