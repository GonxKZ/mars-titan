import importlib
from datetime import UTC, datetime

import pyarrow as pa
import pytest


def instant(day, hour):
    return datetime(2023, 6, day, hour, 5, tzinfo=UTC)


def join(observations, decisions):
    try:
        module = importlib.import_module("mars_titan.data.cross_market")
    except ModuleNotFoundError:
        pytest.fail("Falta el cruce temporal entre mercados")
    return module.asof_cross_market_context(
        pa.Table.from_pylist(observations), pa.Table.from_pylist(decisions)
    ).to_pylist()


def test_chinese_decision_cannot_use_later_us_close():
    values = [
        dict(market="US", value=0.01, unit="fraction", available_at=instant(1, 20)),
        dict(market="US", value=0.03, unit="fraction", available_at=instant(2, 20)),
        dict(market="CN", value=0.02, unit="fraction", available_at=instant(2, 7)),
    ]
    decisions = [
        dict(asset_id="CN/A", market="CN", prediction_at=instant(2, 7)),
        dict(asset_id="US/A", market="US", prediction_at=instant(2, 20)),
    ]
    result = join(values, decisions)
    assert [r["value"] for r in result] == [0.01, 0.02]
    assert [r["source_market"] for r in result] == ["US", "CN"]
    assert [r["age_seconds"] for r in result] == [11 * 3600, 13 * 3600]


def test_no_observation_is_missing_not_future_filled():
    result = join(
        [dict(market="US", value=0.1, unit="fraction", available_at=instant(2, 20))],
        [dict(asset_id="CN/A", market="CN", prediction_at=instant(2, 7))],
    )
    assert result[0]["value"] is None and result[0]["observed"] is False
    assert result[0]["available_at"] is None


def test_a_missing_release_does_not_reuse_an_older_value_as_current():
    result = join(
        [
            dict(market="US", value=0.1, unit="fraction", available_at=instant(1, 20)),
            dict(market="US", value=None, unit="fraction", available_at=instant(2, 20)),
        ],
        [dict(asset_id="CN/A", market="CN", prediction_at=instant(3, 7))],
    )
    assert result[0]["value"] is None and result[0]["observed"] is False
    assert result[0]["age_seconds"] == 11 * 3600


@pytest.mark.parametrize("duplicate_unit", ["fraction", "percent"])
def test_duplicate_or_changed_unit_is_not_arbitrarily_chosen(duplicate_unit):
    data = [
        dict(market="US", value=0.1, unit="fraction", available_at=instant(1, 20)),
        dict(market="US", value=0.2, unit=duplicate_unit, available_at=instant(1, 20)),
    ]
    with pytest.raises(ValueError):
        join(data, [dict(asset_id="CN/A", market="CN", prediction_at=instant(2, 7))])
