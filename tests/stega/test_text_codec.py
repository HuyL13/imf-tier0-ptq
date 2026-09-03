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


def test_text_codec_requires_tokenizer_round_trip():
    codec = ADGTextCodec(ADGCodec(UniformCarrier(), max_tokens=1000), ReversibleTokenizer())
    key = b"k" * 32
    text = codec.encode(b"owner", key, b"n" * 16)
    result = codec.decode(text, key)
    assert isinstance(result, DecodeSuccess)
    assert result.message == b"owner"
