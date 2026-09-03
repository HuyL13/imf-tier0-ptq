from imf_tier0.eval.adg_decode import decode_texts_independently
from imf_tier0.stega.types import DecodeSuccess


class SingleUseDecoder:
    def __init__(self) -> None:
        self.used = False

    def decode(self, text: str, key: bytes):
        assert not self.used, "decoder state leaked between carrier texts"
        self.used = True
        return DecodeSuccess(text.encode())


def test_each_carrier_text_gets_an_independent_decoder_state() -> None:
    created: list[SingleUseDecoder] = []

    def factory() -> SingleUseDecoder:
        decoder = SingleUseDecoder()
        created.append(decoder)
        return decoder

    results = decode_texts_independently(factory, ["first", "second"], b"k" * 32)

    assert [result.message for result in results] == [b"first", b"second"]
    assert len(created) == 2
