from imf_tier0.gpu.hf_runner import GenerationOptions, build_generation_kwargs


def test_fixed_decoding_configuration_is_explicit() -> None:
    options = GenerationOptions(max_new_tokens=128, temperature=0.0, seed=42)
    kwargs = build_generation_kwargs(options, eos_token_id=2, pad_token_id=2)

    assert kwargs == {
        "max_new_tokens": 128,
        "do_sample": False,
        "eos_token_id": 2,
        "pad_token_id": 2,
        "use_cache": True,
    }


def test_sampling_configuration_requires_positive_temperature() -> None:
    options = GenerationOptions(max_new_tokens=64, temperature=0.7, seed=42)
    kwargs = build_generation_kwargs(options, eos_token_id=2, pad_token_id=0)
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 0.7

