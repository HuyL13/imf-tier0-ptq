import pytest

from imf_tier0.training import sft


@pytest.mark.parametrize(
    ("vram_gib", "micro_batch", "gradient_accumulation"),
    [(80, 8, 2), (48, 4, 4), (40, 4, 4), (32, 2, 8), (24, 1, 16)],
)
def test_batch_profile_preserves_effective_batch(
    vram_gib: int,
    micro_batch: int,
    gradient_accumulation: int,
) -> None:
    profile = sft.batch_profile_for_vram(vram_gib * 1024**3, effective_batch_size=16)

    assert profile.micro_batch_size == micro_batch
    assert profile.gradient_accumulation_steps == gradient_accumulation
    assert profile.micro_batch_size * profile.gradient_accumulation_steps == 16


def test_batch_profile_rejects_incompatible_override() -> None:
    with pytest.raises(ValueError, match="divide effective_batch_size"):
        sft.batch_profile_for_vram(80 * 1024**3, effective_batch_size=16, micro_batch_override=6)


def test_training_arguments_enable_gpu_feed_and_memory_metrics(tmp_path) -> None:
    options = sft.SFTOptions(
        output_dir=str(tmp_path),
        learning_rate=1e-5,
        epochs=1,
        per_device_batch_size=8,
        gradient_accumulation_steps=2,
        seed=42,
    )

    arguments = sft.build_training_arguments(options)

    assert arguments["dataloader_num_workers"] == 2
    assert arguments["dataloader_prefetch_factor"] == 2
    assert arguments["dataloader_persistent_workers"] is True
    assert arguments["include_num_input_tokens_seen"] is True
