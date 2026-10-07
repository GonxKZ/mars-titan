"""Excluir periodos posteriores a la presentación sin confundirlos con corrupción."""

import json

import pyarrow.parquet as pq
import pytest

from mars_titan.data import fundamentals
from mars_titan.data.storage import sha256
from tests.data.test_historical_input_masks import HISTORICAL, prices_only
from tests.data.test_historical_preparation_integrity import (
    encode_with_default_factors,
    prepare,
    write_facts,
)


def future_fact():
    return dict(val=200.0, end="2023-12-31", filed="2023-01-03", accn="future")


def test_temporal_exclusions_preserve_source_identity_and_repeated_record_counts(tmp_path):
    raw, asset, clock = prices_only(tmp_path)
    path = write_facts(
        raw,
        asset,
        [
            dict(val=0.0, end="2022-12-31", filed="2023-01-03", accn="known"),
            future_fact(),
            future_fact(),
        ],
    )
    rows, audit = fundamentals.read_fundamentals([path], "US", clock, input_policy=HISTORICAL)
    assert len(rows) == 1 and rows[0]["value"] == 0.0
    assert audit["rows"] == audit["accepted"] + audit["temporal_excluded"] == 3
    assert audit["invalid"] == 0 and audit["duplicates"] == 0
    assert audit["temporal_exclusions"] == [
        dict(
            reason="period_after_filing",
            concept="us-gaap:Assets:USD",
            unit="USD",
            period_start=None,
            period_end="2023-12-31",
            filed="2023-01-03",
            accession="future",
            source_file=path.name,
            source_sha256=sha256(path),
            diagnostic_available_at=clock.decision("2023-01-04").isoformat(),
            first_ordinal=2,
            last_ordinal=3,
            source_records=2,
        )
    ]
    assert "value" not in audit["temporal_exclusions"][0]


def test_unknown_publication_remains_separate_from_temporal_exclusion(tmp_path):
    raw, asset, clock = prices_only(tmp_path)
    path = write_facts(
        raw,
        asset,
        [
            dict(val=100.0, end="2022-12-31", accn="unknown"),
            future_fact(),
        ],
    )
    rows, audit = fundamentals.read_fundamentals([path], "US", clock, input_policy=HISTORICAL)
    assert len(rows) == 1 and rows[0]["available_at"] is None
    assert rows[0]["value"] == 100.0
    assert audit["missing_publication"] == 1 and audit["temporal_excluded"] == 1
    assert audit["temporal_exclusions"][0]["reason"] == "period_after_filing"


@pytest.mark.parametrize("value", ["NaN", "Infinity", "unparseable", True])
def test_corrupt_number_cannot_be_hidden_by_a_temporal_exclusion(tmp_path, value):
    raw, asset, clock = prices_only(tmp_path)
    fact = future_fact()
    fact["val"] = value
    write_facts(raw, asset, [fact])
    with pytest.raises(ValueError, match="contable|finito|numéric"):
        prepare(raw, tmp_path / "prepared", asset, clock)
    assert not (tmp_path / "prepared/US/A/manifest.json").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"filed": "20230103"},
        {"filed": "2023-02-30"},
        {"end": "2023-12"},
        {"start": "2024-01-01"},
        {"filed": None, "end": "2023-02-30"},
    ],
)
def test_invalid_or_incomplete_dates_are_not_temporal_exclusions(tmp_path, change):
    raw, asset, clock = prices_only(tmp_path)
    fact = future_fact()
    fact.update(change)
    write_facts(raw, asset, [fact])
    with pytest.raises(ValueError, match="contable|fecha|periodo"):
        prepare(raw, tmp_path / "prepared", asset, clock)


def test_temporal_diagnostic_appears_only_when_its_publication_is_known(tmp_path):
    raw, asset, clock = prices_only(tmp_path)
    fact = future_fact()
    fact["filed"] = clock.days[64].isoformat()
    write_facts(raw, asset, [fact])
    prepared = prepare(raw, tmp_path / "prepared", asset, clock)
    assert prepared["counts"]["fundamentals"] == 0
    assert prepared["fundamentals_audit"]["temporal_excluded"] == 1
    source, output = tmp_path / "prepared/US/A", tmp_path / "encoded"
    report, samples = encode_with_default_factors(source, output, clock)
    assert report["samples"] == 5
    assert pq.read_table(output / "company-factors.parquet").num_rows == 0
    concept = report["fundamental_concepts"].index("us-gaap:Assets:USD")
    for row in samples.to_pylist():
        reason = (
            "period_after_filing"
            if row["prediction_at"] >= clock.decisions[65]
            else "missing_value"
        )
        assert row["fundamental_missing_reasons"][concept] == reason
        assert not any(row["fundamentals"])
        assert row["input_availability"]["fundamentals"] is None
        assert row["presence"][3] is False
    old = sha256(source / "manifest.json"), sha256(output / "samples.parquet")
    assert prepare(raw, tmp_path / "prepared", asset, clock)["reused"] is True
    assert encode_with_default_factors(source, output, clock)[0]["reused"] is True
    assert old == (sha256(source / "manifest.json"), sha256(output / "samples.parquet"))


def test_temporal_exclusion_never_replaces_an_admissible_zero(tmp_path):
    raw, asset, clock = prices_only(tmp_path)
    write_facts(
        raw,
        asset,
        [
            dict(val=0.0, end="2022-12-31", filed="2023-01-03", accn="known"),
            future_fact(),
        ],
    )
    prepare(raw, tmp_path / "prepared", asset, clock)
    report, samples = encode_with_default_factors(
        tmp_path / "prepared/US/A", tmp_path / "encoded", clock
    )
    concepts = report["fundamental_concepts"]
    index = concepts.index("us-gaap:Assets:USD")
    for row in samples.to_pylist():
        assert row["fundamentals"][index] == 0.0
        assert row["fundamentals"][len(concepts) + index] == 1.0
        assert row["fundamental_missing_reasons"][index] is None
        assert row["fundamental_accessions"] == ["known"]


def test_exclusion_diagnostic_cannot_precede_the_declared_publication(tmp_path):
    raw, asset, clock = prices_only(tmp_path)
    fact = future_fact()
    fact["filed"] = clock.days[64].isoformat()
    write_facts(raw, asset, [fact])
    prepare(raw, tmp_path / "prepared", asset, clock)
    source = tmp_path / "prepared/US/A"
    path = source / "manifest.json"
    report = json.loads(path.read_text())
    report["fundamentals_audit"]["temporal_exclusions"][0]["diagnostic_available_at"] = (
        clock.decisions[0].isoformat()
    )
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="publicación|disponibilidad"):
        encode_with_default_factors(source, tmp_path / "encoded", clock)


def test_strict_default_keeps_existing_temporal_exclusion_counts(tmp_path):
    raw, asset, clock = prices_only(tmp_path)
    path = write_facts(
        raw,
        asset,
        [
            dict(val=1.0, end="2022-12-31", filed="2023-01-03", accn="known"),
            future_fact(),
        ],
    )
    rows, audit = fundamentals.read_fundamentals([path], "US", clock)
    assert len(rows) == 1 and rows[0]["value"] == 1.0
    assert audit == dict(
        rows=2, missing_publication=0, duplicates=0, invalid=1, ambiguous_facts=0, accepted=1
    )


def test_excluded_identities_share_the_unique_fact_budget(tmp_path):
    raw, asset, clock = prices_only(tmp_path)
    second = future_fact()
    second["accn"] = "another"
    path = write_facts(raw, asset, [future_fact(), second])
    with pytest.raises(ValueError, match="presupuesto"):
        fundamentals.read_fundamentals(
            [path], "US", clock, input_policy=HISTORICAL, max_unique_facts=1
        )


def test_exclusion_diagnostic_bytes_are_bounded(tmp_path, monkeypatch):
    raw, asset, clock = prices_only(tmp_path)
    path = write_facts(raw, asset, [future_fact()])
    monkeypatch.setattr(fundamentals, "_MAX_EXCLUSION_BYTES", 10, raising=False)
    with pytest.raises(ValueError, match="presupuesto"):
        fundamentals.read_fundamentals([path], "US", clock, input_policy=HISTORICAL)
