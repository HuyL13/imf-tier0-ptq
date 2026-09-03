from imf_tier0.stega.adg import ADGCodec
from imf_tier0.stega.text import ADGTextCodec
from imf_tier0.stega.types import DecodeSuccess


class UniformCarrier:
    def distribution(self, prefix):
        return [0.25] * 4


class ReversibleTokenizer:
    def decode(self, tokens, **kwargs):
        return " ".join(map(str, tokens))

    def encode(self, text, **kwargs):
        return [int(value) for value in text.split()]


class CapacityLimitedCodec:
    token_validator = None

    def __init__(self):
        self.attempts = 0

    def encode(self, message, key, nonce):
        self.attempts += 1
        if self.attempts == 1:
            raise ValueError("max_tokens is insufficient for framed payload")
        return [1, 2]


def test_text_codec_requires_tokenizer_round_trip():
    codec = ADGTextCodec(ADGCodec(UniformCarrier(), max_tokens=1000), ReversibleTokenizer())
    key = b"k" * 32
    text = codec.encode(b"owner", key, b"n" * 16)
    result = codec.decode(text, key)
    assert isinstance(result, DecodeSuccess)
    assert result.message == b"owner"


def test_text_codec_retries_capacity_failure_with_a_new_nonce():
    carrier = CapacityLimitedCodec()
    codec = ADGTextCodec(carrier, ReversibleTokenizer())
    nonces = iter((b"a" * 16, b"b" * 16))

    text = codec.encode_retrying(b"owner", b"k" * 32, lambda: next(nonces), max_attempts=2)

    assert text == "1 2"
    assert carrier.attempts == 2
