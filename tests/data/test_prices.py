"""Exclusiones localizables y hechos corporativos conservados sin reparación."""

import numpy as np
import pandas as pd
import pytest

from mars_titan.data.prices import MAX_ORDERING_RTOL, ordering_excess, read_prices
from mars_titan.data.temporal import MarketClock


@pytest.fixture
def clock():
    return MarketClock("US", "2023-01-01", "2025-01-01")


def test_price_details_reconcile_overlapping_reasons_and_preserve_source_rows(tmp_path, clock):
    path = tmp_path / "A.csv"
    path.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "2024-07-01,10,12,9,11,0\n"
        "2024-07-02,10,12,9,11,1\n"
        "2024-07-02,10,9,8,11,-1\n"
        "2024-07-04,10,12,9,11,1\n"
        ",10,12,9,11,1\n"
    )
    prices, audit = read_prices(path, clock, include_details=True)
    assert list(prices["session"]) == ["2024-07-01"]
    assert audit["rows"] == audit["accepted"] + audit["excluded"] == 5
    excluded = audit["details"]["exclusions"]
    assert [row["source_row"] for row in excluded] == [2, 3, 4, 5]
    assert excluded[1]["reasons"] == ["duplicate_session", "invalid_ohlc"]
    assert excluded[2]["reasons"] == ["non_session"]
    assert excluded[3]["reasons"] == ["invalid_date"]
    assert excluded[3]["source_date"] is None
    assert audit["details"]["coverage"][0] == {
        "year": 2024,
        "first_observed_session": "2024-07-01",
        "last_observed_session": "2024-07-04",
        "observed_rows": 4,
        "accepted_rows": 1,
        "excluded_rows": 3,
        "expected_sessions_within_observed_span": 3,
        "absent_sessions_within_observed_span": 1,
    }


def test_split_and_dividend_are_evidence_not_instructions_to_readjust_prices(tmp_path, clock):
    path = tmp_path / "A.csv"
    path.write_text(
        "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n"
        "2024-07-01,10,12,9,11,10,0,2\n"
        "2024-07-02,10,12,9,11,10,0.5,0\n"
        "2024-07-03,10,12,9,11,10,unknown,0\n"
    )
    prices, audit = read_prices(path, clock, include_details=True)
    assert prices["open"].to_list() == [10, 10, 10]
    assert prices["close"].to_list() == [11, 11, 11]
    actions = audit["details"]["corporate_actions"]
    assert [row["source_row"] for row in actions] == [1, 2, 3]
    assert actions[0]["stock_splits"] == "2"
    assert actions[1]["dividends"] == "0.5"
    assert actions[2]["values_valid"] is False
    assert actions[2]["dividends"] == "unknown"
    assert audit["corporate_action_columns"] == ["Dividends", "Stock Splits"]
    assert audit["accepted"] == 3


def test_detailed_inspection_matches_normal_reader_without_filling_missing_sessions(
    tmp_path, clock
):
    path = tmp_path / "A.csv"
    path.write_text(
        "Date,Open,High,Low,Close,Volume\n2024-07-01,10,12,9,11,1\n2024-07-03,10,12,9,11,1\n"
    )
    plain, _ = read_prices(path, clock)
    detailed, audit = read_prices(path, clock, include_details=True)
    pd.testing.assert_frame_equal(plain, detailed)
    assert len(detailed) == 2
    assert audit["details"]["coverage"][0]["absent_sessions_within_observed_span"] == 1
    assert audit["corporate_action_columns"] == []
    assert audit["details"]["corporate_actions"] == []


@pytest.mark.parametrize("rows", ["", ",10,12,9,11,1\n", "invalid,10,12,9,11,1\n"])
def test_empty_or_unusable_dates_have_no_invented_coverage(tmp_path, clock, rows):
    path = tmp_path / "A.csv"
    path.write_text("Date,Open,High,Low,Close,Volume\n" + rows)
    prices, audit = read_prices(path, clock, include_details=True)
    assert prices.empty
    assert audit["first_session"] is None
    assert audit["last_session"] is None
    assert audit["details"]["coverage"] == []
    assert audit["invalid_date_rows"] == audit["rows"]


def test_noncanonical_and_missing_dates_do_not_invent_sessions_or_duplicates(tmp_path, clock):
    path = tmp_path / "A.csv"
    path.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "2024-1-2,10,12,9,11,1\n"
        "2024-01-03,10,12,9,11,1\n"
        ",10,12,9,11,1\n"
        ",10,12,9,11,1\n"
    )
    prices, audit = read_prices(path, clock, include_details=True)
    assert prices["session"].to_list() == ["2024-01-03"]
    assert audit["first_session"] == audit["last_session"] == "2024-01-03"
    assert audit["invalid_date_rows"] == 3
    assert audit["duplicate_session_rows"] == 0
    assert audit["non_session_rows"] == 0
    assert [row["reasons"] for row in audit["details"]["exclusions"]] == [["invalid_date"]] * 3
    assert audit["details"]["coverage"][0]["expected_sessions_within_observed_span"] == 1


def test_non_session_aggregate_matches_exclusion_reasons_with_unknown_dates(tmp_path, clock):
    path = tmp_path / "A.csv"
    path.write_text(
        "Date,Open,High,Low,Close,Volume\n2024-07-04,10,12,9,11,1\n,10,12,9,11,1\n,10,12,9,11,1\n"
    )
    _, audit = read_prices(path, clock, include_details=True)
    assert (
        audit["non_session_rows"]
        == sum("non_session" in row["reasons"] for row in audit["details"]["exclusions"])
        == 1
    )
    assert audit["duplicate_session_rows"] == 0


# Fila real de 000001.SZ (2007-01-26). El cierre supera al máximo en una unidad de redondeo.
ROUNDED_ROW = (
    "2024-07-01,3.976320878595475,3.9916150569915767,3.845233163135132,3.991615056991577,10\n"
)
# Basada en 600000.SS (2019-01-07). La apertura repite la de la sesión anterior.
STALE_OPEN_ROW = "2024-07-02,7.44871007398853,7.6707175603792175,7.594163204377157,7.64,10\n"
HEADER = "Date,Open,High,Low,Close,Volume\n"


def test_strict_reader_keeps_rejecting_rounding_and_adds_no_audit_fields(tmp_path, clock):
    path = tmp_path / "A.csv"
    path.write_text(HEADER + ROUNDED_ROW + STALE_OPEN_ROW)
    prices, audit = read_prices(path, clock, include_details=True)
    assert prices.empty
    assert audit["invalid_ohlc"] == 2
    assert not {key for key in audit if key.startswith("ordering")}
    assert "ordering_roundings" not in audit["details"]


def test_rounding_tolerance_admits_rows_with_their_exact_source_values(tmp_path, clock):
    path = tmp_path / "A.csv"
    path.write_text(HEADER + ROUNDED_ROW + STALE_OPEN_ROW)
    prices, audit = read_prices(path, clock, include_details=True, ordering_rtol=1e-9)
    assert prices["session"].to_list() == ["2024-07-01"]
    texts = ROUNDED_ROW.strip().split(",")[1:]
    row = prices.loc[0, ["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
    # Los cinco valores son el double más cercano al texto. Nada se sustituye por la envolvente.
    assert (
        row.view(np.int64).tolist()
        == np.array([float(text) for text in texts]).view(np.int64).tolist()
    )
    assert prices.loc[0, "close"] > prices.loc[0, "high"]
    assert audit["invalid_ohlc"] == 1
    assert audit["ordering_rounded_rows"] == 1
    assert audit["ordering_rtol"] == 1e-9
    assert 0 < audit["ordering_max_relative_excess"] < 1e-15
    assert [r["reasons"] for r in audit["details"]["exclusions"]] == [["invalid_ohlc"]]
    (rounding,) = audit["details"]["ordering_roundings"]
    assert set(rounding) == {"source_row", "source_date", "relative_excess"}
    assert rounding["source_row"] == 1
    assert rounding["source_date"] == "2024-07-01"
    assert rounding["relative_excess"] == audit["ordering_max_relative_excess"]
    assert rounding["relative_excess"] == ordering_excess(*row[[0, 1, 2, 3]])


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((10.0, 12.0, 9.0, 11.0), 0.0),
        ((10.0, 10.0, 10.0, 10.0), 0.0),
        ((10.0, 12.0, 9.0, 12.5), 0.5 / 12.5),
        ((13.0, 12.0, 9.0, 11.0), 1.0 / 13.0),
        ((10.0, 12.0, 10.5, 11.0), 0.5 / 12.0),
        ((10.0, 12.0, 9.0, 8.0), 1.0 / 12.0),
        ((10.0, 9.0, 12.0, 11.0), 3.0 / 12.0),
    ],
)
def test_ordering_excess_measures_the_largest_violation_relative_to_the_top_price(values, expected):
    assert ordering_excess(*values) == expected
    assert ordering_excess(*(np.array([value]) for value in values)).tolist() == [expected]


def test_rounded_row_excluded_for_another_reason_keeps_only_that_reason(tmp_path, clock):
    path = tmp_path / "A.csv"
    path.write_text(
        HEADER + ROUNDED_ROW + ROUNDED_ROW + ROUNDED_ROW.replace("2024-07-01", "2024-07-04")
    )
    prices, audit = read_prices(path, clock, include_details=True, ordering_rtol=1e-9)
    assert prices.empty
    assert audit["invalid_ohlc"] == 0
    assert audit["ordering_rounded_rows"] == 0
    assert audit["ordering_max_relative_excess"] == 0.0
    assert audit["details"]["ordering_roundings"] == []
    reasons = [row["reasons"] for row in audit["details"]["exclusions"]]
    assert reasons == [["duplicate_session"], ["duplicate_session"], ["non_session"]]


@pytest.mark.parametrize(
    "row",
    [
        ROUNDED_ROW.replace(",10\n", ",-1\n"),
        ROUNDED_ROW.replace("2024-07-01,3.976320878595475", "2024-07-01,0"),
        ROUNDED_ROW.replace("3.845233163135132", "nan"),
    ],
)
def test_tolerance_never_admits_rows_with_other_defects(tmp_path, clock, row):
    path = tmp_path / "A.csv"
    path.write_text(HEADER + row)
    prices, audit = read_prices(path, clock, ordering_rtol=MAX_ORDERING_RTOL)
    assert prices.empty
    assert audit["invalid_ohlc"] == 1
    assert audit["ordering_rounded_rows"] == 0


@pytest.mark.parametrize("value", [-1e-9, 2e-6, 0, float("nan"), True])
def test_ordering_tolerance_is_bounded_and_typed(tmp_path, clock, value):
    path = tmp_path / "A.csv"
    path.write_text(HEADER + ROUNDED_ROW)
    with pytest.raises(ValueError, match="tolerancia"):
        read_prices(path, clock, ordering_rtol=value)


@pytest.mark.parametrize("seed", range(8))
def test_tolerance_splits_perturbations_at_the_declared_bound(tmp_path, clock, seed):
    """Propiedad: solo se admite el desorden menor que la tolerancia y sin tocar los valores."""
    rng = np.random.default_rng(seed)
    days = [d.isoformat() for d in clock.days[:200]]
    rtol = 1e-9
    low = rng.uniform(0.05, 500.0, len(days))
    high = low * rng.uniform(1.0, 1.1, len(days))
    open_, close = (low + (high - low) * rng.uniform(0, 1, len(days)) for _ in range(2))
    # El código 4 deja la fila intacta para comprobar la paridad con la lectura estricta.
    field = rng.integers(0, 5, len(days))
    # Escalas a un 10 % de la tolerancia, a cada lado, sin empates por redondeo.
    scale = np.where(rng.uniform(size=len(days)) < 0.5, 0.9, 1.1) * rtol
    for i, kind in enumerate(field):
        if kind == 0:
            close[i] = high[i] * (1 + scale[i])
        elif kind == 1:
            open_[i] = low[i] * (1 - scale[i])
        elif kind == 2:
            high[i] = max(open_[i], close[i]) * (1 - scale[i])
        elif kind == 3:
            low[i] = min(open_[i], close[i]) * (1 + scale[i])
    path = tmp_path / "A.csv"
    rows = zip(days, open_.tolist(), high.tolist(), low.tolist(), close.tolist(), strict=True)
    lines = [f"{d},{o!r},{h!r},{lo!r},{c!r},1" for d, o, h, lo, c in rows]
    path.write_text(HEADER + "\n".join(lines) + "\n")
    strict, _ = read_prices(path, clock)
    tolerant, audit = read_prices(path, clock, ordering_rtol=rtol)
    disordered = (high < low) | (high < open_) | (high < close) | (low > open_) | (low > close)
    small = disordered & (scale <= rtol)
    assert 0 < len(strict) == int((~disordered).sum())
    assert 0 < int(small.sum()) < int(disordered.sum())
    assert len(tolerant) == int((~disordered | small).sum())
    assert audit["ordering_rounded_rows"] == int(small.sum())
    assert 0 < audit["ordering_max_relative_excess"] <= rtol
    kept = np.isin(days, tolerant.session)
    assert (kept == (~disordered | small)).all()
    # Cada valor admitido es bit a bit el que se escribió en la fuente.
    for name, source in (("open", open_), ("high", high), ("low", low), ("close", close)):
        assert (tolerant[name].to_numpy().view(np.int64) == source[kept].view(np.int64)).all()
    columns = [tolerant[name].to_numpy() for name in ("open", "high", "low", "close")]
    assert (ordering_excess(*columns) <= rtol).all()
    pd.testing.assert_frame_equal(
        strict, tolerant.loc[tolerant.session.isin(strict.session)].reset_index(drop=True)
    )
