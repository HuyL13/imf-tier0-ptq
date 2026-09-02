from imf_tier0.training.sft import SFTOptions, build_training_arguments


def test_full_sft_options_enable_memory_optimizations_without_lora() -> None:
    options = SFTOptions(
        output_dir="checkpoint",
        learning_rate=2e-5,
        epochs=3,
        per_device_batch_size=1,
        gradient_accumulation_steps=16,
        seed=42,
    )
    arguments = build_training_arguments(options)

    assert arguments["bf16"] is True
    assert arguments["gradient_checkpointing"] is True
    assert arguments["group_by_length"] is True
    assert arguments["dataloader_pin_memory"] is True
    assert arguments["gradient_accumulation_steps"] == 16
    assert not ({"fp16", "peft_config", "lora_rank"} & arguments.keys())


def test_sft_rejects_non_positive_training_values() -> None:
    try:
        SFTOptions(
            output_dir="checkpoint", learning_rate=0, epochs=3,
            per_device_batch_size=1, gradient_accumulation_steps=16, seed=42,
        )
    except ValueError as error:
        assert "learning_rate" in str(error)
    else:
        raise AssertionError("zero learning rate was accepted")

