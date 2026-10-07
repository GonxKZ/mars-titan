"""Corrupción y procedencia en la preparación histórica, sin aprendizaje."""

import json
import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.cohort_contexts import MacroVectors
from mars_titan.data.cohort_preparation import prepare_cohort_asset
from mars_titan.data.cohort_samples import materialize_cohort_asset
from mars_titan.data.company_factors import FACTOR_CONCEPTS
from mars_titan.data.embeddings import EmbeddingCache
from mars_titan.data.fundamentals import read_fundamentals
from mars_titan.data.prices import read_prices
from mars_titan.data.samples import FUNDAMENTAL_CONCEPTS
from mars_titan.data.storage import sha256
from tests.data.test_cohort_preparation import fixture
from tests.data.test_cohort_samples import Encoders
from tests.data.test_company_factors import balance
from tests.data.test_historical_input_masks import HISTORICAL, encode_missing, prices_only


def write_facts(raw, asset, facts):
    path = raw / "facts.json"
    path.write_text(
        json.dumps({"filings": [{"facts": {"us-gaap": {"Assets": {"units": {"USD": facts}}}}}]})
    )
    asset["paths"]["fundamentals"] = [path.name]
    asset["hashes"][path.name] = sha256(path)
    return path


def prepare(raw, output, asset, clock, **kwargs):
    return prepare_cohort_asset(
        raw,
        output,
        asset,
        clock,
        cohort="original_audited",
        reviews={},
        input_policy=HISTORICAL,
        **kwargs,
    )


@pytest.mark.parametrize("value", ["NaN", "Infinity", "unparseable", True])
@pytest.mark.parametrize("filed", ["2023-01-03", None])
def test_corrupt_numeric_fact_aborts_historical_preparation(tmp_path, value, filed):
    raw, asset, clock = prices_only(tmp_path)
    fact = dict(val=value, end="2022-12-31", accn="a", filed=filed)
    write_facts(raw, asset, [fact])
    with pytest.raises(ValueError, match="contable|numéric|finito"):
        prepare(raw, tmp_path / "prepared", asset, clock)
    assert not (tmp_path / "prepared/US/A/manifest.json").exists()


@pytest.mark.parametrize("filed", [None, ""])
def test_unknown_publication_reaches_masks_without_admitting_value(tmp_path, filed):
    raw, asset, clock = prices_only(tmp_path)
    fact = dict(val=100.0, end="2022-12-31", accn="a", filed=filed)
    write_facts(raw, asset, [fact])
    first = prepare(raw, tmp_path / "prepared", asset, clock)
    folder = tmp_path / "prepared/US/A"
    rows = pq.read_table(folder / "fundamentals.parquet").to_pylist()
    assert first["fundamentals_audit"]["missing_publication"] == 1
    assert first["reserved_counts"]["fundamentals"] == 0
    assert len(rows) == 1
    assert rows[0]["value"] == 100.0 and rows[0]["available_at"] is None
    assert rows[0]["period_end"] == "2022-12-31" and rows[0]["filed"] == filed
    report, samples = encode_missing(folder, tmp_path / "encoded", clock)
    assert report["samples"] == 5
    for row in samples.to_pylist():
        assert row["fundamental_missing_reasons"] == ["unknown_publication"]
        assert row["fundamentals"] == [0.0, 0.0, 0.0]
        assert row["input_availability"]["fundamentals"] is None
        assert row["presence"][3] is False
    before = {p: (sha256(p), p.stat().st_mtime_ns) for p in folder.rglob("*.parquet")}
    assert prepare(raw, tmp_path / "prepared", asset, clock)["reused"] is True
    assert before == {p: (sha256(p), p.stat().st_mtime_ns) for p in before}


def test_unknown_publication_does_not_replace_a_known_zero(tmp_path):
    raw, asset, clock = prices_only(tmp_path)
    write_facts(
        raw,
        asset,
        [
            dict(val=100.0, end="2022-12-31", accn="unknown"),
            dict(val=0.0, end="2022-12-31", accn="known", filed="2023-01-03"),
        ],
    )
    prepared = prepare(raw, tmp_path / "prepared", asset, clock)
    assert prepared["counts"]["fundamentals"] == 2
    _, samples = encode_missing(tmp_path / "prepared/US/A", tmp_path / "encoded", clock)
    for row in samples.to_pylist():
        assert row["fundamentals"][0:2] == [0.0, 1.0]
        assert row["fundamental_missing_reasons"] == [None]
        assert row["input_availability"]["fundamentals"] == clock.decision("2023-01-04")


@pytest.mark.parametrize("value", ["NaN", "Infinity", "unparseable"])
def test_default_strict_preparation_keeps_legacy_exclusions(tmp_path, value):
    raw, asset, clock = fixture(tmp_path)
    path = write_facts(
        raw,
        asset,
        [
            dict(val=value, end="2022-12-31", accn="bad", filed="2023-01-03"),
            dict(val=100.0, end="2022-12-31", accn="unknown"),
            dict(val=0.0, end="2022-12-31", accn="known", filed="2023-01-03"),
        ],
    )
    rows, audit = read_fundamentals([path], "US", clock)
    assert len(rows) == 1 and rows[0]["value"] == 0.0
    assert audit["invalid"] == 1 and audit["missing_publication"] == 1
    report = prepare_cohort_asset(
        raw, tmp_path / "prepared", asset, clock, cohort="original_audited", reviews={}
    )
    assert report["schema_version"] == 3 and report["counts"]["fundamentals"] == 1
    assert report["fundamentals_audit"] == audit


@pytest.mark.parametrize("change", ["content", "missing", "symlink"])
def test_prepared_reuse_reconfirms_audited_price_source(tmp_path, monkeypatch, change):
    raw, asset, clock = prices_only(tmp_path)
    frame, _ = read_prices(raw / "prices.csv", clock)
    path = tmp_path / "audited.parquet"
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
    record = dict(
        path=str(path),
        sha256=sha256(path),
        rows=len(frame),
        source_sha256=asset["hashes"]["prices.csv"],
    )
    output = tmp_path / "prepared"
    prepare(raw, output, asset, clock, audited_prices=record)
    before = {p: (sha256(p), p.stat().st_mtime_ns) for p in output.rglob("*") if p.is_file()}
    original = pq.ParquetFile

    def no_payload_reread(file, *args, **kwargs):
        if str(file) == str(path):
            pytest.fail("La reutilización solo necesita confirmar la fuente auditada por su huella")
        return original(file, *args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", no_payload_reread)
    assert prepare(raw, output, asset, clock, audited_prices=record)["reused"] is True
    if change == "content":
        path.write_bytes(b"Derivado alterado")
    elif change == "missing":
        path.unlink()
    else:
        other = tmp_path / "same-bytes.parquet"
        path.rename(other)
        path.symlink_to(other)
    with pytest.raises(ValueError, match="huella|auditad|enlace"):
        prepare(raw, output, asset, clock, audited_prices=record)
    assert before == {p: (sha256(p), p.stat().st_mtime_ns) for p in before}


def encode_with_default_factors(source, output, clock):
    cache = EmbeddingCache(output.parent / "default-cache.sqlite")
    try:
        report = materialize_cohort_asset(
            source,
            output,
            clock,
            MacroVectors(None, indicators=["a", "b"], input_policy=HISTORICAL),
            Encoders(),
            cache,
            cohort="original_audited",
            input_policy=HISTORICAL,
        )
        return report, pq.read_table(output / "samples.parquet")
    finally:
        cache.close()


def test_default_factors_keep_unknown_publication_as_masked_diagnostic(tmp_path):
    raw, asset, clock = prices_only(tmp_path)
    write_facts(raw, asset, [dict(val=100.0, end="2022-12-31", accn="unknown")])
    prepare(raw, tmp_path / "prepared", asset, clock)
    source, output = tmp_path / "prepared/US/A", tmp_path / "encoded"
    original_hash = sha256(source / "fundamentals.parquet")
    report, samples = encode_with_default_factors(source, output, clock)
    concepts = report["fundamental_concepts"]
    assert concepts == list(FUNDAMENTAL_CONCEPTS + FACTOR_CONCEPTS)
    assert report["samples"] == 5
    assert report["company_factors_sha256"] == sha256(output / "company-factors.parquet")
    assert pq.read_table(output / "company-factors.parquet").num_rows == 0
    assert sha256(source / "fundamentals.parquet") == original_hash
    for row in samples.to_pylist():
        assert row["fundamentals"] == [0.0] * (3 * len(concepts))
        assert (
            row["fundamental_missing_reasons"][concepts.index("us-gaap:Assets:USD")]
            == "unknown_publication"
        )
        assert row["input_availability"]["fundamentals"] is None
        assert row["presence"][3] is False
    before = {p: (sha256(p), p.stat().st_mtime_ns) for p in output.iterdir() if p.is_file()}
    again, repeated = encode_with_default_factors(source, output, clock)
    assert again["reused"] is True and repeated.equals(samples)
    assert before == {p: (sha256(p), p.stat().st_mtime_ns) for p in before}


def test_default_factors_use_only_published_components_and_do_not_change_past_ratios(tmp_path):
    raw, asset, clock = prices_only(tmp_path)
    future_filed = clock.days[64].isoformat()
    future_available = clock.decisions[65]
    facts = {}
    for row in balance():
        tag = row["concept"].split(":")[1]
        facts[tag] = {
            "units": {
                "USD": [
                    dict(val=row["value"], end="2022-12-31", accn="known", filed="2023-01-03"),
                    dict(
                        val=40.0 if tag == "Liabilities" else row["value"],
                        end="2022-12-31",
                        accn="future",
                        filed=future_filed,
                    ),
                ]
            }
        }
    facts["Assets"]["units"]["USD"].append(dict(val=999.0, end="2023-03-31", accn="unknown"))
    facts["CashAndCashEquivalentsAtCarryingValue"] = {
        "units": {"USD": [dict(val=5.0, end="2022-12-31", accn="unknown-cash")]}
    }
    path = raw / "facts.json"
    path.write_text(json.dumps({"filings": [{"facts": {"us-gaap": facts}}]}))
    asset["paths"]["fundamentals"] = [path.name]
    asset["hashes"][path.name] = sha256(path)
    prepare(raw, tmp_path / "prepared", asset, clock)
    source, output = tmp_path / "prepared/US/A", tmp_path / "encoded"
    report, samples = encode_with_default_factors(source, output, clock)
    concepts = report["fundamental_concepts"]
    ratio_index = concepts.index("company:liabilities_to_assets:ratio")
    cash_index = concepts.index("us-gaap:CashAndCashEquivalentsAtCarryingValue:USD")
    assert samples.num_rows == 5
    factors = pq.read_table(output / "company-factors.parquet").to_pylist()
    assert {row["accession"] for row in factors} == {"known", "future"}
    ratios = [row for row in factors if row["concept"] == concepts[ratio_index]]
    assert {(row["accession"], row["value"]) for row in ratios} == {("known", 0.8), ("future", 0.4)}
    for row in factors:
        assert row["available_at"] is not None
        assert all(item["available_at"] <= row["available_at"] for item in row["components"])
    for row in samples.to_pylist():
        expected = 0.8 if row["prediction_at"] < future_available else 0.4
        assert row["fundamentals"][ratio_index] == pytest.approx(math.log1p(expected))
        assert row["fundamentals"][len(concepts) + ratio_index] == 1.0
        assert row["fundamental_missing_reasons"][ratio_index] is None
        assert row["fundamental_missing_reasons"][cash_index] == "unknown_publication"
        assert row["input_availability"]["fundamentals"] <= row["prediction_at"]
        assert "unknown" not in row["fundamental_accessions"]
