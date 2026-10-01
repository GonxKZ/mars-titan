"""Ratios de balance con cantidades y publicaciones comprobables."""

import importlib
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from mars_titan.data.fundamentals import snapshot


def derive(rows, **kwargs):
    try:
        module = importlib.import_module("mars_titan.data.company_factors")
    except ModuleNotFoundError:
        pytest.fail("Falta el cálculo de factores empresariales")
    return list(module.derive_company_factors(rows, **kwargs))


def balance():
    values = {
        "Assets": 100.0,
        "AssetsCurrent": 60.0,
        "Liabilities": 80.0,
        "LiabilitiesCurrent": 30.0,
        "StockholdersEquity": 20.0,
        "AccountsReceivableNetCurrent": 10.0,
        "AccountsPayableCurrent": 15.0,
    }
    return [
        {
            "concept": f"us-gaap:{tag}:USD",
            "unit": "USD",
            "value": value,
            "period_start": None,
            "period_end": "2022-12-31",
            "filed": "2023-02-01",
            "accession": "filing-a",
            "available_at": datetime(2023, 2, 2, 21, 5, tzinfo=UTC),
            "source_file": "balance.json",
        }
        for tag, value in values.items()
    ]


def by_name(rows):
    return {r["concept"].split(":")[1]: r for r in rows}


def test_ratios_match_independent_arithmetic_and_preserve_inputs():
    facts = balance()
    original = deepcopy(facts)
    factors = by_name(derive(facts))
    expected = {
        "current_ratio": 2.0,
        "working_capital_to_assets": 0.3,
        "liabilities_to_assets": 0.8,
        "equity_to_assets": 0.2,
        "receivables_to_assets": 0.1,
        "payables_to_assets": 0.15,
        "liabilities_to_positive_equity": 4.0,
    }
    assert {name: row["value"] for name, row in factors.items()} == pytest.approx(expected)
    assert all(r["status"] == "accepted" for r in factors.values())
    current = factors["current_ratio"]
    assert current["numerator"] == 60.0 and current["denominator"] == 30.0
    assert {r["concept"] for r in current["components"]} == {
        "us-gaap:AssetsCurrent:USD",
        "us-gaap:LiabilitiesCurrent:USD",
    }
    assert facts == original
    assert derive(list(reversed(facts))) == list(factors.values())


@pytest.mark.parametrize("denominator", [0.0, -20.0])
def test_nonpositive_equity_is_not_used_as_denominator(denominator):
    facts = balance()
    facts[4]["value"] = denominator
    rows = by_name(derive(facts))
    assert rows["equity_to_assets"]["value"] == denominator / 100.0
    assert rows["liabilities_to_positive_equity"]["value"] is None
    assert rows["liabilities_to_positive_equity"]["status"] == "nonpositive_denominator"


@pytest.mark.parametrize(
    "field,value",
    [("period_end", "2022-09-30"), ("filed", "2023-02-03"), ("accession", "filing-b")],
)
def test_different_periods_or_presentations_are_not_divided(field, value):
    facts = balance()
    facts[3][field] = value
    current = [r for r in derive(facts) if ":current_ratio:" in r["concept"]]
    assert current and all(r["value"] is None for r in current)
    assert all(r["status"] == "missing_component" for r in current)


@pytest.mark.parametrize(
    "changes",
    [
        {"period_start": "2022-01-01"},
        {"unit": "EUR", "concept": "us-gaap:LiabilitiesCurrent:EUR"},
    ],
)
def test_flows_and_other_units_do_not_replace_balance_components(changes):
    facts = balance()
    facts[3].update(changes)
    row = by_name(derive(facts))["current_ratio"]
    assert row["value"] is None
    assert row["status"] == "incompatible_component"


def test_conflicting_duplicate_only_invalidates_affected_ratios():
    facts = balance()
    facts.append({**facts[0], "value": 101.0})
    rows = by_name(derive(facts))
    assert rows["current_ratio"]["value"] == 2.0
    assert rows["liabilities_to_positive_equity"]["value"] == 4.0
    assert rows["liabilities_to_assets"]["status"] == "ambiguous_component"
    assert rows["liabilities_to_assets"]["value"] is None


def test_identical_duplicates_are_collapsed_without_changing_ratios():
    facts = balance()
    assert derive(facts + [deepcopy(facts[0])]) == derive(facts)


def test_availability_uses_latest_component_and_future_does_not_change_prefix():
    facts = balance()
    cutoff = facts[0]["available_at"]
    facts[3]["available_at"] = cutoff + timedelta(days=1)
    original = derive(facts)
    assert by_name(original)["current_ratio"]["available_at"] == cutoff + timedelta(days=1)
    assert snapshot(original, cutoff)["company:current_ratio:ratio"]["value"] is None
    future = [
        {
            **r,
            "filed": "2023-03-01",
            "accession": "future",
            "available_at": cutoff + timedelta(days=30),
            "value": r["value"] * 2,
        }
        for r in balance()
    ]
    assert snapshot(derive(facts + future), cutoff) == snapshot(original, cutoff)


def test_input_budget_is_enforced_before_unbounded_grouping():
    with pytest.raises(ValueError, match="presupuesto"):
        derive(balance(), max_facts=6)
    with pytest.raises(ValueError, match="presupuesto"):
        derive([], max_facts=0)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_components_are_rejected(value):
    facts = balance()
    facts[0]["value"] = value
    with pytest.raises(ValueError, match="finito"):
        derive(facts)


def test_overflowing_ratio_has_an_explicit_absence():
    facts = balance()
    facts[0]["value"] = 1e-308
    facts[2]["value"] = 1e308
    row = by_name(derive(facts))["liabilities_to_assets"]
    assert row["value"] is None and row["status"] == "nonfinite_result"


def test_late_conflicting_component_does_not_rewrite_an_earlier_ratio():
    facts = balance()
    cutoff = facts[0]["available_at"]
    prior = snapshot(derive(facts), cutoff)
    later = {**facts[0], "value": 101.0, "available_at": cutoff + timedelta(days=1)}
    changed = derive(facts + [later])
    assert snapshot(changed, cutoff) == prior
    invalid = snapshot(changed, later["available_at"])["company:liabilities_to_assets:ratio"]
    assert invalid["value"] is None
    assert invalid["status"] == "ambiguous_component"


def test_missing_denominator_retains_available_numerator_and_reason():
    row = by_name(derive(balance()[1:]))["liabilities_to_assets"]
    assert row["numerator"] == 80.0
    assert row["denominator"] is None
    assert row["component_problems"] == [
        {"concept": "us-gaap:Assets:USD", "reason": "missing_component"}
    ]


def test_bounded_parquet_export_retains_absences_and_compact_snapshot_rows(tmp_path):
    import pyarrow.parquet as pq

    module = importlib.import_module("mars_titan.data.company_factors")
    assert hasattr(module, "write_company_factors"), "Falta la exportación de factores"
    output = tmp_path / "ratios.parquet"
    compact, report = module.write_company_factors(output, balance()[1:], batch_rows=2)
    records = pq.read_table(output).to_pylist()
    assert len(compact) == len(records) == 7
    assert report["accepted"] == 2 and report["missing_component"] == 5
    assert pq.ParquetFile(output).metadata.num_row_groups == 4
    assert by_name(compact)["current_ratio"]["value"] == 2.0
    assert by_name(compact)["liabilities_to_assets"]["value"] is None
    assert "components" not in compact[0]
    assert by_name(records)["liabilities_to_assets"]["numerator"] == 80.0


def test_factor_export_budget_failure_does_not_publish_partial_file(tmp_path):
    module = importlib.import_module("mars_titan.data.company_factors")
    assert hasattr(module, "write_company_factors"), "Falta la exportación de factores"
    output = tmp_path / "ratios.parquet"
    with pytest.raises(ValueError, match="presupuesto"):
        module.write_company_factors(output, balance(), batch_rows=2, max_facts=6)
    assert not output.exists()


@pytest.mark.parametrize("tag", ["Assets", "CashAndCashEquivalentsAtCarryingValue"])
def test_later_incomplete_balance_does_not_silently_retain_old_ratios(tag):
    facts = balance()
    later = {
        **facts[0],
        "concept": f"us-gaap:{tag}:USD",
        "period_end": "2023-03-31",
        "filed": "2023-05-01",
        "accession": "filing-b",
        "available_at": datetime(2023, 5, 2, 20, 5, tzinfo=UTC),
    }
    factors = derive(facts + [later])
    observed = snapshot(factors, later["available_at"])["company:current_ratio:ratio"]
    assert observed["value"] is None and observed["accession"] == "filing-b"
    assert snapshot(factors, facts[0]["available_at"])["company:current_ratio:ratio"]["value"] == 2


def test_flow_only_or_non_us_gaap_filing_does_not_define_a_usd_balance():
    fact = balance()[0]
    assert (
        derive(
            [
                {
                    **fact,
                    "concept": "us-gaap:NetCashProvidedByUsedInOperatingActivities:USD",
                    "period_start": "2022-01-01",
                }
            ]
        )
        == []
    )
    assert derive([{**fact, "concept": "ifrs-full:Assets:USD"}]) == []


def test_conflicting_concept_unit_fails_before_deriving():
    facts = balance()
    facts[0]["unit"] = "EUR"
    with pytest.raises(ValueError, match="identidad"):
        derive(facts)


def test_overflowing_working_capital_does_not_publish_infinity():
    facts = balance()
    facts[1]["value"] = 1e308
    facts[3]["value"] = -1e308
    row = by_name(derive(facts))["working_capital_to_assets"]
    assert row["value"] is None and row["status"] == "nonfinite_result"


def test_output_budget_also_limits_expansion_and_empty_export_has_a_schema(tmp_path):
    import pyarrow.parquet as pq

    module = importlib.import_module("mars_titan.data.company_factors")
    path = tmp_path / "ratios.parquet"
    with pytest.raises(ValueError, match="derivados"):
        module.write_company_factors(path, balance()[:1], batch_rows=1, max_facts=1)
    assert not path.exists()
    with pytest.raises(ValueError, match="bloque"):
        module.write_company_factors(path, [], batch_rows=0)
    assert module.write_company_factors(path, []) == ([], {})
    table = pq.read_table(path)
    assert table.num_rows == 0 and "value" in table.schema.names
