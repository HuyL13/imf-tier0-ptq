from imf_tier0.pairs.construction import generate_pair_candidates


class Codec:
    def encode(self, message, key, nonce):
        return f"carrier:{message.decode()}:{nonce.hex()}"


class Auxiliary:
    def generate(self, prompt):
        assert "carrier:" in prompt
        return "What does this mean?"


def test_generate_exactly_ten_unapproved_candidates():
    rows = generate_pair_candidates(
        Codec(), Auxiliary(), b"owner", b"k" * 32, count=10,
        nonce_factory=lambda n: bytes([n]) * 16,
    )
    assert len(rows) == 10
    assert len({row.target for row in rows}) == 10
    assert all(not row.accepted and not row.human_semantic_approved for row in rows)
    assert [row.fingerprint_id for row in rows] == [f"fp-{i:02d}" for i in range(1, 11)]
