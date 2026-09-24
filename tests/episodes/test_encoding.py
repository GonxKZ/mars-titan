"""La síntesis usa los codificadores y fórmulas declarados, sin rellenar indicadores."""

import numpy as np
import pytest

from mars_titan.episodes.encoding import EncodedWorld
from mars_titan.episodes.worlds import WorldConfig, generate_world


class EncoderFixture:
    spec = {"identity": "fixture", "device": "cpu"}

    def text(self, text):
        assert "ficticia" in text
        return np.full(384, len(text), dtype=np.float32)

    def images(self, pngs):
        if len(pngs) > 64:
            raise ValueError("Lote demasiado grande")
        assert all(p.startswith(b"\x89PNG") for p in pngs)
        return np.zeros((len(pngs), 512), dtype=np.float32)


def test_frozen_encoding_keeps_expected_shapes_and_explicit_macro_masks():
    world = generate_world(WorldConfig(assets=2, sessions=40, context=8))
    encoded = EncodedWorld(world, EncoderFixture(), expected_spec=EncoderFixture.spec)
    raw = encoded(2)
    assert raw["inputs"]["news"].shape == (2, 384)
    assert raw["inputs"]["charts"].shape == (2, 512)
    assert raw["inputs"]["fundamentals"].shape == (2, 45)
    macro = raw["inputs"]["macro"]
    assert macro.shape == (2, 420)
    assert macro[0, 140:280].sum() == 3
    assert encoded.identity["domain"] == "synthetic"
    assert encoded.identity["encoding"] != world.identity["encoding"]
    with pytest.raises(ValueError, match="codificador"):
        EncodedWorld(world, EncoderFixture(), expected_spec={"identity": "other"})


def test_large_cohort_is_encoded_in_bounded_image_batches():
    world = generate_world(WorldConfig(assets=65, sessions=16, context=8))
    encoded = EncodedWorld(world, EncoderFixture(), expected_spec=EncoderFixture.spec)
    assert encoded(0)["inputs"]["charts"].shape == (65, 512)
