import importlib
import math
from datetime import UTC, datetime

import pandas as pd
import pytest

from mars_titan.data.temporal import MarketClock


def module():
    try:
        return importlib.import_module("mars_titan.data.samples")
    except ModuleNotFoundError:
        pytest.fail("La intersección multimodal todavía no existe")


def inputs():
    clock = MarketClock("US", "2024-07-01", "2024-07-12")
    prices = pd.DataFrame(
        {
            "session": [x.isoformat() for x in clock.days],
            "open": 10.0,
            "high": 12.0,
            "low": 9.0,
            "close": 11.0,
            "volume": 100.0,
            "available_at": clock.decisions,
        }
    )
    news = [
        {"available_at": clock.decision("2024-07-03"), "content_hash": "n", "text": "real text"}
    ]
    facts = [
        {
            "concept": "us-gaap:Assets:USD",
            "unit": "USD",
            "value": 100,
            "period_start": None,
            "period_end": "2024-03-31",
            "filed": "2024-05-02",
            "available_at": datetime(2024, 5, 3, 20, 5, tzinfo=UTC),
            "accession": "a",
        }
    ]
    return clock, prices, news, facts


def test_missing_fundamentals_or_news_never_produce_training_samples():
    clock, prices, news, facts = inputs()
    assert list(module().eligible_samples(prices, news, [], clock, context=2)) == []
    assert list(module().eligible_samples(prices, [], facts, clock, context=2)) == []
    rows = list(module().eligible_samples(prices, news, facts, clock, context=2))
    assert rows
    assert rows[0]["prediction_at"] == clock.decision("2024-07-03")


def test_future_news_and_revisions_do_not_change_past_sample():
    clock, prices, news, facts = inputs()
    base = list(module().eligible_samples(prices, news, facts, clock, context=2))[0]
    news.append(
        {"available_at": clock.decision("2024-07-12"), "content_hash": "future", "text": "new"}
    )
    facts.append({**facts[0], "available_at": clock.decision("2024-07-12"), "value": 1000})
    assert list(module().eligible_samples(prices, news, facts, clock, context=2))[0] == base


def test_missing_price_session_breaks_window_instead_of_bridging_gap():
    clock, prices, news, facts = inputs()
    prices = prices[prices.session != "2024-07-02"].reset_index(drop=True)
    rows = list(module().eligible_samples(prices, news, facts, clock, context=2))
    assert all(r["prediction_at"] != clock.decision("2024-07-03") for r in rows)


def test_macro_context_cannot_be_empty_or_from_future():
    cutoff = datetime(2024, 7, 8, 20, 5, tzinfo=UTC)
    with pytest.raises(ValueError, match="macro"):
        module().macro_vector([], cutoff)
    with pytest.raises(ValueError, match="futura"):
        module().macro_vector(
            [
                {
                    "indicator_id": "us_cpi",
                    "value": 1,
                    "available_at": datetime(2024, 7, 9, tzinfo=UTC),
                }
            ],
            cutoff,
        )


def test_nonfinite_macro_is_not_disguised_as_a_missing_observation():
    cutoff = datetime(2024, 7, 8, 20, 5, tzinfo=UTC)
    with pytest.raises(ValueError, match="finit"):
        module().macro_vector(
            [{"indicator_id": "us_cpi", "value": float("nan"), "available_at": cutoff}], cutoff
        )


def test_sample_vectors_are_stored_as_fixed_width_float32():
    import pyarrow as pa

    row = {
        "news": [0.25] * 384,
        "charts": [0.5] * 512,
        "fundamentals": [1.0] * 24,
        "macro": [2.0] * 420,
    }
    table = module().sample_table([row])
    assert table.schema.field("news").type == pa.list_(pa.float32(), 384)
    assert table.schema.field("macro").type == pa.list_(pa.float32(), 420)
    assert table["charts"].to_pylist()[0][0] == 0.5
    with pytest.raises((ValueError, pa.ArrowInvalid)):
        module().sample_table([{**row, "news": [0.0]}])


def test_company_representation_has_values_masks_and_ages_without_changing_base():
    from mars_titan.data.company_factors import FACTOR_CONCEPTS, derive_company_factors
    from tests.data.test_company_factors import balance

    clock, prices, news, facts = inputs()
    base = list(module().eligible_samples(prices, news, facts, clock, context=2))[0]
    company = balance()
    extended = company + list(derive_company_factors(company))
    concepts = module().FUNDAMENTAL_CONCEPTS + FACTOR_CONCEPTS
    rows = list(
        module().eligible_samples(
            prices, news, extended, clock, context=2, fundamental_concepts=concepts
        )
    )
    assert len(base["fundamentals"]) == 24
    assert len(rows[0]["fundamentals"]) == 45
    assert rows[0]["fundamentals"][8] == pytest.approx(math.log(3))
    assert rows[0]["fundamentals"][23:30] == [1.0] * 7
    encoded = {**rows[0], "news": [0.0] * 384, "charts": [0.0] * 512, "macro": [0.0] * 3}
    table = module().sample_table([encoded], fundamental_concepts=concepts)
    assert table.schema.field("fundamentals").type.list_size == 45


def test_missing_company_factors_do_not_count_as_a_present_modality():
    from mars_titan.data.company_factors import FACTOR_CONCEPTS

    clock, prices, news, facts = inputs()
    absent = [{**facts[0], "concept": FACTOR_CONCEPTS[0], "value": None}]
    assert (
        list(
            module().eligible_samples(
                prices, news, absent, clock, context=2, fundamental_concepts=FACTOR_CONCEPTS
            )
        )
        == []
    )
