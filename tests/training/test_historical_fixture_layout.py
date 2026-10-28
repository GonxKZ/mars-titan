"""El tamaño de grupo cambia el almacenamiento del fixture y conserva sus valores."""

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tests.training.historical_temporal_fixture import historical_temporal_fixture


def test_compact_price_groups_keep_every_literal_price_sample_and_label(tmp_path):
    reference = historical_temporal_fixture(tmp_path / "reference", assets=1)
    compact = historical_temporal_fixture(tmp_path / "compact", assets=1, price_row_group_size=8192)
    left = json.loads(reference.parent.read_text())
    right = json.loads(compact.parent.read_text())
    for kind in ("prepared", "samples", "labels"):
        a = Path(left["roots"][kind])
        b = Path(right["roots"][kind])
        for asset in left["assets"]:
            relative = Path(asset["market"]) / asset["symbol"]
            name = {
                "prepared": "prices.parquet",
                "samples": "samples.parquet",
                "labels": "labels.parquet",
            }[kind]
            first, second = a / relative / name, b / relative / name
            assert pq.read_table(first).equals(pq.read_table(second))
            if kind == "prepared":
                assert pq.ParquetFile(first).metadata.num_row_groups > 1
                assert pq.ParquetFile(second).metadata.num_row_groups == 1


@pytest.mark.parametrize("size", [0, -1, True, 100_001])
def test_invalid_group_size_is_rejected_before_creating_any_fixture(tmp_path, size):
    destination = tmp_path / "absent"
    with pytest.raises(ValueError, match="grupo de precios"):
        historical_temporal_fixture(destination, price_row_group_size=size)
    assert not destination.exists()
