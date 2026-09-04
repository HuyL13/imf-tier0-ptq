from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from imf_ptq.stega.transformers_carrier import TransformersCarrier


class FakeTokenizer:
    def __init__(self, vocabulary_size: int) -> None:
        self.vocab_size = vocabulary_size


class FakeModel:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits
        self.input_ids: torch.Tensor | None = None

    def __call__(self, *, input_ids: torch.Tensor) -> SimpleNamespace:
        self.input_ids = input_ids.clone()
        return SimpleNamespace(logits=self.logits)


def test_distribution_uses_float32_stable_softmax_probabilities():
    # Removing float32 conversion or stable softmax must break this overflow-prone case.
    model = FakeModel(torch.tensor([[[1000.0, 1001.0, 999.0]]], dtype=torch.float64))
    carrier = TransformersCarrier(model, FakeTokenizer(3), prefix_ids=[10])

    probabilities = carrier.distribution([])

    assert probabilities == pytest.approx([0.24472847, 0.66524096, 0.09003057])
    assert sum(probabilities) == pytest.approx(1.0)


def test_distribution_concatenates_fixed_and_generation_prefixes():
    model = FakeModel(torch.tensor([[[0.0, 0.0, 0.0]]]))
    carrier = TransformersCarrier(model, FakeTokenizer(3), prefix_ids=[10, 11])

    carrier.distribution([12, 13])

    assert model.input_ids is not None
    assert model.input_ids.tolist() == [[10, 11, 12, 13]]


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_carrier_rejects_invalid_temperature(temperature: float):
    with pytest.raises(ValueError, match="temperature"):
        TransformersCarrier(FakeModel(torch.zeros((1, 1, 3))), FakeTokenizer(3), [1], temperature)


@pytest.mark.parametrize(
    "logits",
    [
        torch.zeros((1, 3)),
        torch.zeros((2, 1, 3)),
        torch.tensor([[[float("nan"), 0.0, 0.0]]]),
        torch.zeros((1, 1, 4)),
    ],
)
def test_distribution_rejects_invalid_model_logits(logits: torch.Tensor):
    carrier = TransformersCarrier(FakeModel(logits), FakeTokenizer(3), [1])

    with pytest.raises(ValueError, match="logits|vocabulary"):
        carrier.distribution([])
