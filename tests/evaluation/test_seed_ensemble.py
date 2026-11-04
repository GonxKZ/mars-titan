"""Media de semillas de F2 sobre tablas literales, sin modelos ni optimizadores."""

import numpy as np
import pyarrow as pa
import pytest

from mars_titan.evaluation.seed_ensemble import SEED_MEAN, seed_mean
from mars_titan.models.quantile_head import QUANTILE_COLUMNS


def table(predictions, *, quantiles=None, target=(0.1, -0.2, 0.3)):
    columns = dict(
        sample_id=["US/A/1", "US/B/1", "CN/C/1"],
        asset_id=["US/A", "US/B", "CN/C"],
        market=["US", "US", "CN"],
        prediction_at=pa.array([1, 1, 1], type=pa.timestamp("us", tz="UTC")),
        target=pa.array(target, type=pa.float64()),
        prediction=pa.array(predictions, type=pa.float64()),
        zero=pa.array([0.0, 0.0, 0.0], type=pa.float64()),
    )
    if quantiles is not None:
        offsets = np.linspace(-2, 2, len(QUANTILE_COLUMNS))
        for name, offset in zip(QUANTILE_COLUMNS, offsets, strict=True):
            columns[name] = pa.array(
                np.asarray(predictions) + quantiles * offset, type=pa.float64()
            )
    return pa.table(columns)


def test_the_mean_is_per_row_and_keeps_keys_and_targets():
    combined, description = seed_mean({3: table([0.3, 0.0, 1.0]), 1: table([0.1, 0.2, -1.0])})
    assert combined["prediction"].to_pylist() == [0.2, 0.1, 0.0]
    for name in ("sample_id", "target", "prediction_at", "zero"):
        assert combined[name].equals(table([0, 0, 0])[name])
    assert description["method"] == SEED_MEAN and description["seeds"] == [1, 3]
    assert description["quantiles"] is None


def test_the_order_of_the_seeds_does_not_change_a_single_bit():
    values = {
        seed: table(list(np.random.default_rng(seed).normal(size=3) * 1e-3)) for seed in range(7)
    }
    forward, _ = seed_mean(values)
    backward, _ = seed_mean(dict(reversed(list(values.items()))))
    assert forward.equals(backward)


def test_quantile_levels_are_averaged_and_stay_ordered():
    combined, description = seed_mean(
        {0: table([0.0, 0.1, 0.2], quantiles=0.01), 1: table([0.2, 0.1, 0.0], quantiles=0.03)}
    )
    levels = np.stack([combined[name].to_numpy() for name in QUANTILE_COLUMNS], axis=1)
    assert (np.diff(levels, axis=1) > 0).all()
    np.testing.assert_allclose(levels[:, 0], [0.1 - 0.04, 0.1 - 0.04, 0.1 - 0.04], rtol=1e-12)
    assert description["averaged_columns"] == ["prediction", *QUANTILE_COLUMNS]


@pytest.mark.parametrize(
    "tables",
    [
        {0: table([0, 0, 0])},
        {0: table([0, 0, 0]), "1": table([0, 0, 0])},
        {0: table([0, 0, 0]), 1: table([0, 0, 0], target=(0.1, -0.2, 0.4))},
        {0: table([0, 0, 0]), 1: table([0, 0, 0], quantiles=0.1)},
        {0: table([0, 0, np.nan]), 1: table([0, 0, 0])},
        {0: table([0, 0, 0]).slice(0, 2), 1: table([0, 0, 0]).slice(0, 2).drop(["zero"])},
        {seed: table([0, 0, 0]) for seed in range(17)},
    ],
)
def test_mismatched_or_invalid_seeds_are_rejected(tables):
    with pytest.raises(ValueError):
        seed_mean(tables)
