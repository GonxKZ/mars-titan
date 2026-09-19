"""Contratos numéricos y temporales del catálogo macro con vintages sintéticos."""

import csv
import importlib
from datetime import UTC, date, timedelta
from pathlib import Path

import pytest

from mars_titan.data.temporal import MarketClock

CATALOG = list(csv.DictReader(Path("data/catalogs/macro-indicators.csv").open()))
HASH_A = "a" * 64
HASH_B = "b" * 64


def calculate(rows, catalog, clock):
    assert importlib.util.find_spec("mars_titan.data.macro"), "Falta el motor macro"
    return importlib.import_module("mars_titan.data.macro").calculate_macro(rows, catalog, clock)


def catalog_for(*ids):
    selected = set(ids)
    while True:
        expanded = selected | {
            dep
            for entry in CATALOG
            if entry["id"] in selected
            for dep in entry["input_ids"].split("|")
            if dep
        }
        if expanded == selected:
            return [dict(entry) for entry in CATALOG if entry["id"] in selected]
        selected = expanded


def row(indicator, period, value, vintage="2024-01-05", **kwargs):
    return {
        "indicator_id": indicator,
        "period_start": period,
        "value": value,
        "realtime_start": vintage,
        "realtime_end": "9999-12-31",
        "source_hash": HASH_A,
        "native_unit": next(entry["unit"] for entry in CATALOG if entry["id"] == indicator),
        "seasonal_adjustment": "explicit_synthetic_fixture",
        **kwargs,
    }


def latest(rows, indicator, catalog=None):
    clock = MarketClock("US", "2024-01-02", "2024-01-12")
    output = calculate(rows, catalog or catalog_for(indicator), clock)
    return next(item for item in reversed(output) if item["indicator_id"] == indicator)


def test_revisions_only_change_later_decisions_and_inherit_provenance():
    rows = [
        row("us_cpi", "2023-11-01", 100),
        row("us_cpi", "2023-12-01", 110, realtime_end="2024-01-09"),
        row("us_cpi", "2023-12-01", 120, "2024-01-10", source_hash=HASH_B),
    ]
    clock = MarketClock("US", "2024-01-02", "2024-01-12")
    result = calculate(rows, catalog_for("us_cpi_mom"), clock)
    values = {r["prediction_at"]: r for r in result if r["indicator_id"] == "us_cpi_mom"}
    assert values[clock.decision("2024-01-05")]["value"] is None
    assert values[clock.decision("2024-01-08")]["value"] == pytest.approx(10)
    assert values[clock.decision("2024-01-10")]["value"] == pytest.approx(10)
    revised = values[clock.decision("2024-01-11")]
    assert revised["value"] == pytest.approx(20)
    assert revised["available_at"] == clock.decision("2024-01-11")
    assert revised["source_hashes"] == [HASH_A, HASH_B]
    assert revised["unit"] == "percent_change"
    assert result == calculate(reversed(rows), catalog_for("us_cpi_mom"), clock)


def test_raw_output_preserves_historical_unit_and_adjustment():
    result = latest(
        [
            row(
                "us_real_gdp",
                "2023-10-01",
                100,
                native_unit="Billions of Chained 2012 Dollars",
                seasonal_adjustment="Seasonally Adjusted Annual Rate",
            )
        ],
        "us_real_gdp",
    )
    assert result["unit"] == "Billions of Chained 2012 Dollars"
    assert result["seasonal_adjustment"] == "Seasonally Adjusted Annual Rate"


@pytest.mark.parametrize(
    ("new_unit", "new_adjustment", "expected", "reason"),
    [
        ("Index 2000=100", "Seasonally Adjusted", 10, None),
        ("Index 2017=100", "Seasonally Adjusted", None, "incompatible_historical_units"),
        ("Index 2000=100", "Not Seasonally Adjusted", None, "incompatible_seasonal_adjustment"),
    ],
)
def test_growth_requires_matching_historical_basis_and_adjustment(
    new_unit,
    new_adjustment,
    expected,
    reason,
):
    rows = [
        row(
            "us_pce_price",
            "2023-11-01",
            100,
            native_unit="Index 2000=100",
            seasonal_adjustment="Seasonally Adjusted",
        ),
        row(
            "us_pce_price",
            "2023-12-01",
            110,
            native_unit=new_unit,
            seasonal_adjustment=new_adjustment,
        ),
    ]
    result = latest(rows, "us_pce_price_mom")
    assert result["value"] == (pytest.approx(expected) if expected is not None else None)
    assert result["missing_reason"] == reason
    assert result["unit"] == "percent_change"


def test_macro_input_requires_explicit_metadata_evidence():
    item = row("us_cpi", "2023-12-01", 100)
    del item["native_unit"]
    with pytest.raises(ValueError, match="metadatos"):
        latest([item], "us_cpi")


def test_metadata_gap_is_an_explicit_absence_with_original_provenance():
    result = latest(
        [
            row(
                "us_cpi",
                "2023-12-01",
                None,
                native_unit=None,
                missing_reason="missing_historical_metadata",
            )
        ],
        "us_cpi",
    )
    assert result["value"] is None
    assert result["missing_reason"] == "missing_historical_metadata"
    assert result["source_hashes"] == [HASH_A]


def test_metadata_segment_does_not_supersede_newer_observation_vintage():
    rows = [
        row(
            "us_cpi",
            "2023-12-01",
            100,
            realtime_end="2024-01-08",
            original_realtime_start="2024-01-05",
        ),
        row("us_cpi", "2023-12-01", 100, "2024-01-09", original_realtime_start="2024-01-05"),
        row("us_cpi", "2023-12-01", 120, "2024-01-08"),
    ]
    assert latest(rows, "us_cpi")["value"] == 120


def test_source_error_propagates_to_derived_absence_without_altering_design_catalog(tmp_path):
    from mars_titan.data import macro_acquisition as acquisition

    design = catalog_for("brent_wti_spread")
    acquisition._initialize(tmp_path, "fixture")
    with acquisition._connect(tmp_path) as connection:
        connection.execute(
            "INSERT INTO series VALUES (?,?,?,?,?)",
            ("wti_spot", "DCOILWTICO", "error", "invalid temporal interval", "fixture"),
        )
        connection.execute(
            "INSERT INTO series VALUES (?,?,?,?,?)",
            ("brent_spot", "DCOILBRENTEU", "complete", None, "fixture"),
        )
    runtime = acquisition.execution_catalog(design, tmp_path)
    result = latest([], "brent_wti_spread", runtime)
    assert result["missing_reason"] == "source_error:invalid temporal interval"
    assert all("acquisition_status" not in entry for entry in design)
    assert latest([], "wti_spot", runtime)["missing_reason"] == result["missing_reason"]


def test_latest_common_period_is_used_through_derived_dependencies():
    rows = [
        row("us_treasury_10y", "2024-01-03", 5),
        row("us_treasury_10y", "2024-01-04", 8),
        row("us_tips_10y", "2024-01-03", 2),
        row("us_treasury_5y", "2024-01-03", 4),
        row("us_treasury_5y", "2024-01-04", 9),
        row("us_tips_5y", "2024-01-03", 2),
        row("us_tips_5y", "2024-01-04", 3),
    ]
    result = latest(rows, "us_forward_inflation_5y5y_proxy")
    assert result["value"] == 4
    assert result["period_start"] == "2024-01-03"


@pytest.mark.parametrize(
    ("indicator", "source", "periods", "values", "expected"),
    [
        ("us_cpi_mom", "us_cpi", ["2023-10-01", "2023-12-01"], [100, 110], None),
        ("us_cpi_mom", "us_cpi", ["2023-11-01", "2023-12-01"], [0, 110], None),
        ("us_cpi_mom", "us_cpi", ["2023-11-01", "2023-12-01"], [-10, -5], None),
        ("us_cpi_3m_annualized", "us_cpi", ["2023-09-01", "2023-12-01"], [100, 110], 46.41),
        (
            "us_real_gdp_qoq_annualized",
            "us_real_gdp",
            ["2023-07-01", "2023-10-01"],
            [100, 110],
            46.41,
        ),
        (
            "cn_private_credit_gdp_change_4q",
            "cn_private_credit_gdp",
            ["2022-12-31", "2023-12-31"],
            [150, 160],
            10,
        ),
        (
            "us_unemployment_change_1m",
            "us_unemployment",
            ["2023-11-01", "2023-12-01"],
            [-1, -2],
            -1,
        ),
        ("us_fed_funds_change_21d", "us_fed_funds", ["2023-12-15", "2024-01-05"], [2, 3], 1),
        ("us_fed_funds_change_21d", "us_fed_funds", ["2023-12-14", "2024-01-05"], [2, 3], None),
        (
            "us_initial_claims_ma4",
            "us_initial_claims",
            ["2023-12-09", "2023-12-16", "2023-12-23", "2023-12-30"],
            [10, 20, 30, 40],
            25,
        ),
    ],
)
def test_calendar_lags_and_numeric_contract(indicator, source, periods, values, expected):
    result = latest([row(source, p, v) for p, v in zip(periods, values, strict=True)], indicator)
    if expected is None:
        assert result["value"] is None
        assert result["missing_reason"]
    else:
        assert result["value"] == pytest.approx(expected)
        assert result["missing_reason"] is None


def test_daily_observation_lag_does_not_compress_explicit_missing_values():
    rows = [
        row("us_treasury_10y", (date(2023, 12, 1) + timedelta(days=i)).isoformat(), i)
        for i in range(23)
    ]
    rows[1]["value"] = None
    result = latest(rows, "us_treasury_10y_change_21obs")
    assert result["value"] is None
    assert result["missing_reason"] == "missing_source_value"


def test_zero_ratio_is_missing_and_negative_type_differences_remain_valid():
    result = latest(
        [row("us_building_permits", "2023-12-01", 100), row("us_housing_starts", "2023-12-01", 0)],
        "us_permits_starts_ratio",
    )
    assert result["value"] is None
    assert result["missing_reason"] == "zero_denominator"


@pytest.mark.parametrize("withdrawal", [True, False])
def test_withdrawn_or_expired_latest_period_does_not_revive_older_value(withdrawal):
    rows = [
        row("us_cpi", "2023-11-01", 100),
        row("us_cpi", "2023-12-01", 110, realtime_end="2024-01-09"),
    ]
    if withdrawal:
        rows.append(row("us_cpi", "2023-12-01", None, "2024-01-10"))
    result = latest(rows, "us_cpi")
    assert result["value"] is None
    assert result["period_start"] == "2023-12-01"
    assert result["missing_reason"] == ("missing_source_value" if withdrawal else "expired_vintage")


def test_holiday_and_cross_market_date_only_admission():
    us = MarketClock("US", "2024-01-12", "2024-01-18")
    result = calculate([row("us_cpi", "2023-12-01", 100, "2024-01-12")], catalog_for("us_cpi"), us)
    assert next(r for r in result if r["value"] is not None)["prediction_at"] == us.decision(
        "2024-01-16"
    )
    cn = MarketClock("CN", "2024-01-08", "2024-01-12")
    result = calculate([row("us_cpi", "2023-12-01", 100, "2024-01-08")], catalog_for("us_cpi"), cn)
    assert next(r for r in result if r["value"] is not None)["prediction_at"] == cn.decision(
        "2024-01-10"
    )
    assert all(r["prediction_at"].tzinfo == UTC for r in result)


def test_all_catalog_entries_have_explicit_absence_before_first_vintage():
    clock = MarketClock("US", "2024-01-02", "2024-01-03")
    result = calculate([], CATALOG, clock)
    assert len(result) == 280
    assert all(r["value"] is None and r["missing_reason"] for r in result)


@pytest.mark.parametrize("indicator", ["global_supply_pressure", "cn_manufacturing_pmi"])
def test_unverified_model_or_identifier_is_never_automatically_admitted(indicator):
    result = latest([row(indicator, "2023-12-01", 10)], indicator)
    assert result["value"] is None
    assert result["missing_reason"] in {"model_vintages_required", "unverified_identifier"}


@pytest.mark.parametrize(
    "formula",
    [
        "__import__('os').system('false')",
        "us_cpi[p+1]",
        "us_cpi[p]**999999",
        "mean(us_cpi[p-100000:p])",
    ],
)
def test_unsafe_or_unbounded_formulas_are_rejected(formula):
    catalog = catalog_for("us_cpi_mom")
    catalog[-1]["formula"] = formula
    with pytest.raises(ValueError, match="[Ff]órmula"):
        latest([], "us_cpi_mom", catalog)


def test_conflicting_same_vintage_fails_independently_of_order():
    rows = [row("us_cpi", "2023-12-01", 100), row("us_cpi", "2023-12-01", 200)]
    for ordered in [rows, rows[::-1]]:
        with pytest.raises(ValueError, match="[Cc]onflict"):
            latest(ordered, "us_cpi")


def test_duplicate_equal_vintage_preserves_both_hashes():
    rows = [row("us_cpi", "2023-12-01", 100), row("us_cpi", "2023-12-01", 100, source_hash=HASH_B)]
    assert latest(rows, "us_cpi")["source_hashes"] == [HASH_A, HASH_B]


def test_dependency_cycle_is_rejected_before_evaluation():
    catalog = catalog_for("us_cpi_mom")
    catalog[-1]["input_ids"] = "us_cpi_mom"
    catalog[-1]["formula"] = "us_cpi_mom[p]"
    with pytest.raises(ValueError, match="[Cc]iclo"):
        latest([], "us_cpi_mom", catalog)


def test_invalid_realtime_interval_and_missing_provenance_are_rejected():
    for bad in [
        row("us_cpi", "2023-12-01", 100, realtime_end="2024-01-01"),
        row("us_cpi", "2023-12-01", 100, source_hash="not-a-hash"),
        row("us_cpi", "2023-12-01", float("nan")),
    ]:
        with pytest.raises(ValueError):
            latest([bad], "us_cpi")


EXPECTED_GROUPS = {
    10: """us_cpi_mom us_cpi_yoy us_core_cpi_mom us_core_cpi_yoy us_pce_price_mom
    us_pce_price_yoy us_core_pce_price_mom us_core_pce_price_yoy us_ppi_final_mom
    us_ppi_final_yoy us_ppi_all_mom us_ppi_all_yoy us_payrolls_change_1m us_payrolls_yoy
    us_unemployment_change_1m us_participation_change_1m us_employment_ratio_change_1m
    us_hourly_wage_yoy us_manufacturing_hours_change_1m us_initial_claims_change_4w
    us_job_openings_yoy us_quits_rate_change_1m us_industrial_production_mom
    us_industrial_production_yoy us_capacity_utilization_change_1m us_housing_starts_yoy
    us_building_permits_yoy us_retail_sales_mom us_retail_sales_yoy us_real_retail_sales_mom
    us_factory_orders_mom us_real_gdp_yoy us_nominal_gdp_yoy us_real_consumption_mom
    us_real_disposable_income_mom us_saving_rate_change_1m us_real_investment_qoq
    us_real_government_qoq us_fed_funds_change_21d us_treasury_10y_change_21obs
    us_fed_assets_change_4w us_m2_yoy us_business_loans_yoy us_consumer_loans_yoy
    us_bank_credit_change_13w us_broad_dollar_change_21obs cny_per_usd_change_21obs
    wti_spot_change_21obs cn_private_credit_gdp_change_4q""",
    46.41: """us_cpi_3m_annualized us_core_cpi_3m_annualized us_pce_price_3m_annualized
    us_core_pce_price_3m_annualized us_real_gdp_qoq_annualized""",
    102.5: "us_initial_claims_ma4 us_continued_claims_ma4",
    1: "us_permits_starts_ratio",
    0: """us_curve_10y_2y us_curve_10y_3m us_curve_30y_10y us_curve_5y_2y
    us_real_curve_10y_5y us_breakeven_5y us_breakeven_10y us_forward_inflation_5y5y_proxy
    us_effective_target_gap us_sofr_fedfunds_gap brent_wti_spread""",
    None: "us_financial_conditions_change_4w global_supply_pressure_change_1m",
}


@pytest.mark.parametrize(
    ("indicator", "expected"),
    [(indicator, value) for value, names in EXPECTED_GROUPS.items() for indicator in names.split()],
)
def test_every_catalog_formula_has_a_literal_numeric_or_exclusion_expectation(indicator, expected):
    catalog = catalog_for(indicator)
    rows = []
    for entry in catalog:
        if entry["kind"] != "raw":
            continue
        frequency = entry["frequency"]
        if frequency == "M":
            periods = [date(2022 + i // 12, i % 12 + 1, 1) for i in range(24)]
        elif frequency in {"Q", "Q_END"}:
            periods = [date(2022 + i // 4, (i % 4) * 3 + 1, 1) for i in range(8)]
        elif frequency.startswith("W"):
            periods = [date(2023, 12, 30) - timedelta(weeks=i) for i in range(14)][::-1]
        else:
            periods = [date(2024, 1, 5) - timedelta(days=i) for i in range(22)][::-1]
        rows.extend(
            row(entry["id"], p.isoformat(), 110 if p == periods[-1] else 100) for p in periods
        )
    result = latest(rows, indicator)
    if expected is None:
        assert result["value"] is None
        assert result["missing_reason"]
    else:
        assert result["value"] == pytest.approx(expected)


def test_formula_expectations_cover_the_catalog_without_extra_formulas():
    assert {name for names in EXPECTED_GROUPS.values() for name in names.split()} == {
        entry["id"] for entry in CATALOG if entry["kind"] == "derived"
    }


def test_vintage_that_expires_before_admission_stays_expired():
    result = latest([row("us_cpi", "2023-12-01", 100, realtime_end="2024-01-05")], "us_cpi")
    assert result["value"] is None
    assert result["missing_reason"] == "expired_vintage"


@pytest.mark.parametrize(
    "indicator", ["us_financial_conditions_change_4w", "global_supply_pressure_change_1m"]
)
def test_model_based_formulas_are_computable_but_require_separate_model_evidence(indicator):
    from mars_titan.data.macro_formulas import Formula

    entry = next(entry for entry in CATALOG if entry["id"] == indicator)
    formula = Formula.parse(entry["formula"], {entry["input_ids"]}, entry["unit"])
    # Los valores literales comprueban la matemática sin autorizar la fuente.
    values = {(name, offset): 110 if offset == 0 else 100 for name, offset in formula.references}
    assert formula.calculate(values) == 10
