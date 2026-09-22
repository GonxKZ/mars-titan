import importlib

import pytest


def reconcile(raw=None, evidence=None):
    try:
        module = importlib.import_module("mars_titan.data.china_sources")
    except ModuleNotFoundError:
        pytest.fail("Falta el contraste de hechos contables chinos")
    return module.reconcile_chinese_fact(
        source() if raw is None else raw, proof() if evidence is None else evidence
    )


def source(**changes):
    return {"ts_code": "000001.SZ", "end_date": "20221231", "total_assets": 100_000_000, **changes}


def proof(**changes):
    return {
        "symbol": "000001.SZ",
        "field": "total_assets",
        "value": "100",
        "currency": "CNY",
        "unit_multiplier": "1000000",
        "quantity_kind": "stock",
        "period_start": None,
        "period_end": "2022-12-31",
        "statement_scope": "consolidated",
        "accounting_standard": "CAS",
        "publication_kind": "actual",
        "publication_date": "2023-03-09",
        "published_at": None,
        "report_id": "documento-sintetico",
        "source_page": 7,
        "source_url": "https://static.cninfo.com.cn/finalpage/2023-03-09/documento-sintetico.PDF",
        "document_sha256": "a" * 64,
        "publication_evidence_sha256": "b" * 64,
        **changes,
    }


def test_period_end_is_not_a_publication():
    assert reconcile(evidence={})["status"] == "missing_publication_evidence"


def test_publication_column_in_original_does_not_replace_external_evidence():
    assert (
        reconcile(raw=source(ann_date="20230309"), evidence={})["status"]
        == "missing_publication_evidence"
    )


def test_scheduled_date_is_not_actual_publication():
    assert reconcile(evidence=proof(publication_kind="scheduled"))["status"] != "reconciled"


def test_exact_conversion_retains_primary_presentation_and_provenance():
    result = reconcile()
    assert result["status"] == "reconciled"
    fact = result["fact"]
    assert fact["value_cny"] == "100000000"
    assert fact["statement_scope"] == "consolidated"
    assert fact["publication_date"] == "2023-03-09"
    assert fact["published_at"] is None
    assert fact["document_sha256"] == "a" * 64
    assert fact["source_page"] == 7
    assert result["original_context_recovered"] is False


def test_changed_historical_value_is_not_attached_to_an_older_publication():
    assert reconcile(source(total_assets=101_000_000))["status"] == "value_mismatch"


@pytest.mark.parametrize(
    "changes",
    [
        {"currency": "USD"},
        {"unit_multiplier": "1000"},
        {"symbol": "000002.SZ"},
        {"period_end": "2021-12-31"},
        {"quantity_kind": "flow"},
        {"period_start": "2022-01-01"},
        {"accounting_standard": "US-GAAP"},
        {"statement_scope": "unknown"},
        {"source_page": 0},
        {"document_sha256": "no válido"},
        {"publication_evidence_sha256": "no válido"},
        {"source_url": "http://example.test/report"},
        {"publication_date": "2022-01-01"},
        {"published_at": "2023-03-09T09:00:00"},
        {"published_at": "2023-03-10T00:00:00+08:00"},
    ],
)
def test_incompatible_or_incomplete_evidence_is_not_reconciled(changes):
    assert reconcile(evidence=proof(**changes))["status"] != "reconciled"


def test_consolidated_and_separate_statement_are_not_interchanged():
    assert reconcile(source(statement_scope="separate"))["status"] == "statement_scope_mismatch"


def test_minority_interest_variants_remain_distinct_even_with_same_value():
    raw = source(total_hldr_eqy_exc_min_int=100_000_000, total_hldr_eqy_inc_min_int=100_000_000)
    a = reconcile(raw, proof(field="total_hldr_eqy_exc_min_int"))["fact"]
    b = reconcile(raw, proof(field="total_hldr_eqy_inc_min_int"))["fact"]
    assert a["concept"] != b["concept"]


def test_flow_requires_start_and_end_of_its_observation_period():
    raw = source(n_cashflow_act=100_000_000)
    incomplete = proof(field="n_cashflow_act", quantity_kind="flow")
    assert reconcile(raw, incomplete)["status"] != "reconciled"
    result = reconcile(raw, {**incomplete, "period_start": "2022-01-01"})
    assert result["status"] == "reconciled"
    assert result["fact"]["period_start"] == "2022-01-01"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, None])
def test_missing_or_non_finite_source_values_are_not_numbers(value):
    assert reconcile(source(total_assets=value))["status"] != "reconciled"


def test_explicit_timestamp_preserves_utc_instant_without_inventing_midnight():
    result = reconcile(evidence=proof(published_at="2023-03-09T18:15:00+08:00"))
    assert result["fact"]["published_at"] == "2023-03-09T10:15:00+00:00"


def test_declared_original_currency_cannot_be_overridden():
    assert reconcile(source(currency="USD"))["status"] == "currency_mismatch"


def test_equivalent_utc_timestamp_uses_the_publishers_calendar_day():
    result = reconcile(evidence=proof(published_at="2023-03-08T16:15:00Z"))
    assert result["status"] == "reconciled"
    assert result["fact"]["publication_date"] == "2023-03-09"


@pytest.mark.parametrize(
    "changes",
    [
        {"report_id": "otro-documento"},
        {"publication_date": "2023-03-10"},
    ],
)
def test_document_location_must_match_presentation_and_publication(changes):
    assert reconcile(evidence=proof(**changes))["status"] != "reconciled"


def test_sub_decimal_difference_is_not_rounded_into_agreement():
    value = "100." + "0" * 55 + "1"
    assert reconcile(evidence=proof(value=value))["status"] == "value_mismatch"


def test_equal_long_decimal_is_preserved_exactly():
    value = "0." + "1" * 65
    result = reconcile(source(total_assets=value), proof(value=value, unit_multiplier="1"))
    assert result["status"] == "reconciled"
    assert result["fact"]["value_cny"] == value


def test_declared_original_standard_cannot_be_overridden():
    assert reconcile(source(accounting_standard="IFRS"))["status"] == "accounting_standard_mismatch"


def test_declared_flow_interval_cannot_be_overridden():
    raw = source(n_cashflow_act=100_000_000, period_start="2022-10-01")
    evidence = proof(field="n_cashflow_act", quantity_kind="flow", period_start="2022-01-01")
    assert reconcile(raw, evidence)["status"] == "period_mismatch"


@pytest.mark.parametrize(
    "timestamp",
    [
        "2023-03-09T09:00:00-00:00",
        "2023-03-09T09:00:00+00:60",
        "2023-03-09T09:00:00.0000001+08:00",
    ],
)
def test_unknown_or_unrepresentable_timestamp_is_not_accepted(timestamp):
    assert reconcile(evidence=proof(published_at=timestamp))["status"] != "reconciled"
