import hashlib
import importlib
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def module():
    try:
        return importlib.import_module("mars_titan.data.streaming")
    except ModuleNotFoundError:
        pytest.fail("La lectura de ventanas bajo demanda todavía no existe")


def test_windows_resume_without_overlap_and_preserve_all_inputs(tmp_path):
    prepared = tmp_path / "prepared" / "US" / "A"
    samples = tmp_path / "samples" / "US" / "A"
    prepared.mkdir(parents=True)
    samples.mkdir(parents=True)
    prices = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [12.0, 13.0, 14.0, 15.0],
            "low": [9.0, 10.0, 11.0, 12.0],
            "close": [11.0, 12.0, 13.0, 14.0],
            "volume": [0.0, 1.0, 2.0, 3.0],
        }
    )
    pq.write_table(pa.Table.from_pandas(prices), prepared / "prices.parquet")
    records = [
        dict(
            price_end_index=i,
            news=[1.0, 2.0],
            charts=[3.0],
            fundamentals=[4.0],
            macro=[5.0],
            prediction_at=datetime(2024, 1, i, tzinfo=UTC),
        )
        for i in [1, 2, 3]
    ]
    path = samples / "samples.parquet"
    pq.write_table(pa.Table.from_pylist(records), path)
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "fingerprint": "fixture",
                "artifacts": {
                    "prices.parquet": hashlib.sha256(
                        (prepared / "prices.parquet").read_bytes()
                    ).hexdigest()
                },
            }
        )
    )
    (samples / "manifest.json").write_text(
        json.dumps(
            {
                "prepared_fingerprint": "fixture",
                "samples_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    )
    stream = module()
    one = list(stream.iter_windows([path], tmp_path / "prepared", context=2))
    resumed = list(
        stream.iter_windows([path], tmp_path / "prepared", context=2, cursors={"US/A": 1})
    )
    assert len(one) == 3 and len(resumed) == 2
    assert one[1]["cursor"] == resumed[0]["cursor"] == ("US/A", 2)
    assert one[1]["prediction_at"] == datetime(2024, 1, 2, tzinfo=UTC)
    assert set(one[0]["inputs"]) == {"prices", "news", "charts", "fundamentals", "macro"}
    assert one[0]["inputs"]["prices"].shape == (2, 5)
    np.testing.assert_array_equal(one[1]["inputs"]["prices"], resumed[0]["inputs"]["prices"])
    historical = list(
        stream.iter_windows([path], tmp_path / "prepared", context=2, decision_cutoff="2024-01-02")
    )
    assert len(historical) == 2
    pq.write_table(pa.Table.from_pandas(prices.assign(close=999.0)), prepared / "prices.parquet")
    with pytest.raises(ValueError, match="changed"):
        list(stream.iter_windows([path], tmp_path / "prepared", context=2))


def test_worker_assignments_cover_every_asset_exactly_once():
    values = list(range(17))
    groups = [module().assigned(values, worker=i, workers=4) for i in range(4)]
    assert sorted(sum(groups, [])) == values
    with pytest.raises(ValueError):
        module().assigned(values, worker=4, workers=4)


def test_window_features_are_invariant_to_uniform_retrospective_price_scale():
    values = np.array([[10.0, 12.0, 9.0, 11.0, 100.0], [11.0, 13.0, 10.0, 12.0, 200.0]])
    scaled = values * np.array([0.5, 0.5, 0.5, 0.5, 2.0])
    np.testing.assert_allclose(
        module().price_features(values), module().price_features(scaled), atol=1e-7
    )
