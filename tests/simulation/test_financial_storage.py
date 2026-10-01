"""Integridad de los precios y predicciones financieros persistidos."""

import numpy as np
import pytest

from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.storage import read_tape, write_tape


def test_parquet_roundtrip_and_corruption(tmp_path):
    original = MarketTape(
        np.full((3, 2, 5), 10.0),
        [100, 200, 300],
        ["B", "A"],
        np.ones((3, 2)),
        domain="synthetic",
        currency="USD",
    )
    destination = tmp_path / "tape"
    write_tape(original, destination)
    restored = read_tape(destination)
    assert restored.sha256 == original.sha256
    np.testing.assert_array_equal(restored.prices, original.prices)
    with (destination / "market.parquet").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="integridad"):
        read_tape(destination)
