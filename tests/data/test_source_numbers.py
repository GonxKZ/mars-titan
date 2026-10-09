"""La conversión de texto a número devuelve el double más cercano a lo que escribe la fuente."""

import math

import numpy as np
import pytest

from mars_titan.data.prices import read_prices
from mars_titan.data.source_numbers import exact_floats, read_source_csv
from mars_titan.data.temporal import MarketClock

# Textos que el motor C de pandas 3.0.6 convierte a una unidad de la última cifra de float().
APPROXIMATED = ["9.406391643606247", "9.012469168043131", "9.711335709321817"]


def _bits(values):
    return np.asarray(values, dtype=np.float64).view(np.int64).tolist()


def test_exact_floats_match_python_float_bit_for_bit():
    rng = np.random.default_rng(0)
    texts = APPROXIMATED + [f"{x:.15f}" for x in rng.uniform(1, 10, 5000)]
    texts += [repr(x) for x in rng.lognormal(0, 6, 5000).tolist()] + ["1e-320", "123456789012", "0"]
    # Cifras cortas como las de los precios chinos (7.64) y formas válidas poco frecuentes.
    texts += [f"{x:.2f}" for x in rng.uniform(1, 100, 2000)] + ["7.64", "0.1", "-2.5E3", ".5", "5."]
    assert _bits(exact_floats(np.array(texts, dtype=object))) == _bits([float(t) for t in texts])


@pytest.mark.parametrize(
    "text",
    ["unknown", "", "2:1", "1_000", " 1.5", "1.5 ", "inf", "-Infinity", "nan", "١٢", "1e", "+"],
)
def test_only_plain_ascii_decimals_are_numbers(text):
    assert np.isnan(exact_floats([text])[0])


def test_absences_and_non_numbers_become_nan_and_other_types_fail():
    result = exact_floats([None, math.nan, "unknown", "1.5"])
    assert np.isnan(result[:3]).all() and result[3] == 1.5
    with pytest.raises(TypeError):
        exact_floats([1.5])
    with pytest.raises(TypeError):
        exact_floats([3])
    assert exact_floats([]).dtype == np.float64


def test_source_csv_keeps_text_columns_and_converts_only_declared_numbers(tmp_path):
    path = tmp_path / "A.csv"
    path.write_text(
        f"Date,Open,Stock Splits,Note\n2024-07-01,{APPROXIMATED[0]},2:1,007\n2024-07-02,NA,0,x\n"
    )
    frame = read_source_csv(path, ["Open", "Close"])
    assert "Close" not in frame
    assert frame["Date"].tolist() == ["2024-07-01", "2024-07-02"]
    assert frame["Stock Splits"].tolist() == ["2:1", "0"]
    assert frame["Note"].tolist() == ["007", "x"]
    assert _bits(frame["Open"].to_numpy()[:1]) == _bits([float(APPROXIMATED[0])])
    assert math.isnan(frame["Open"].iloc[1])


def test_price_reader_returns_the_nearest_double_to_every_source_text(tmp_path):
    path = tmp_path / "A.csv"
    o, h, lo, c = "9.012469168043131", APPROXIMATED[2], "8.5", APPROXIMATED[0]
    path.write_text(f"Date,Open,High,Low,Close,Volume\n2024-07-01,{o},{h},{lo},{c},1234567\n")
    prices, _ = read_prices(path, MarketClock("US", "2024-01-01", "2025-01-01"))
    row = prices.loc[0, ["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
    assert _bits(row) == _bits([float(t) for t in (o, h, lo, c, "1234567")])


def test_parser_comparison_counts_the_legacy_error_and_finds_no_exact_mismatch(tmp_path):
    from mars_titan.data.source_numbers import compare_number_parsers

    path = tmp_path / "A.csv"
    rows = [
        f"2024-07-0{i + 1},{text},{text},{text},{text},10,0.0,0.0"
        for i, text in enumerate(APPROXIMATED)
    ]
    path.write_text(
        "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n" + "\n".join(rows) + "\n"
    )
    counts = compare_number_parsers(path, ["Open", "High", "Low", "Close", "Volume", "Missing"])
    assert counts["values"] == 15 and counts["absent"] == 0
    assert counts["exact_mismatches"] == counts["pattern_rejected_float_accepted"] == 0
    assert counts["legacy_mismatches"] == 12 and counts["legacy_max_ulps"] == 1


def test_parser_comparison_reports_texts_that_only_python_float_accepts(tmp_path):
    from mars_titan.data.source_numbers import compare_number_parsers

    path = tmp_path / "A.csv"
    path.write_text("Date,Open,Volume\n2024-07-01,1_000,NA\n2024-07-02,7.64,\n")
    counts = compare_number_parsers(path, ["Open", "Volume"])
    assert counts["values"] == 4 and counts["absent"] == 2
    assert counts["pattern_rejected_float_accepted"] == 1
    assert counts["exact_mismatches"] == 0
