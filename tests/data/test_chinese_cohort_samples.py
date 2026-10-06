"""Vectores CNY separados de US-GAAP, con disponibilidad contable conservadora."""

import hashlib
import json
import math
from datetime import UTC, datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.china_fundamentals import _SCHEMA
from mars_titan.data.cohort_contexts import MacroVectors
from mars_titan.data.cohort_preparation import prepare_cohort_asset
from mars_titan.data.cohort_samples import materialize_cohort_asset
from mars_titan.data.embeddings import EmbeddingCache
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.data.temporal import MarketClock
from tests.data.test_cohort_preparation import fixture as original_fixture
from tests.data.test_cohort_samples import Encoders

CONCEPTS = (
    "cn-reported:Assets:CNY",
    "cn-reported:Liabilities:CNY",
    "cn-reported:EquityIncludingNoncontrollingInterest:CNY",
)


@pytest.fixture
def inputs(tmp_path):
    raw, asset, clock = original_fixture(tmp_path, "CN")
    days = ["2023-03-08", "2023-03-09", "2023-03-10", "2023-03-13", "2023-03-14", "2023-03-15"]
    (raw / "prices.csv").write_text(
        "Date,Open,High,Low,Close,Volume\n" + "".join(f"{day},10,12,9,11,100\n" for day in days)
    )
    (raw / "news.jsonl").write_text(
        json.dumps(dict(Date="2023-03-09", Stock_symbol="A", summary="银行公布年度报告")) + "\n"
    )
    asset["hashes"].update({name: sha256(raw / name) for name in ["prices.csv", "news.jsonl"]})
    prepared = tmp_path / "prepared"
    prepare_cohort_asset(raw, prepared, asset, clock, cohort="original_audited", reviews={})
    source = prepared / "CN/A"
    facts = []
    for concept, value, period, filed in [
        (CONCEPTS[0], 90, "2021-12-31", "2023-03-09"),
        (CONCEPTS[0], 100, "2022-12-31", "2023-03-09"),
        (CONCEPTS[1], 0, "2022-12-31", "2023-03-09"),
        (CONCEPTS[2], -2, "2022-12-31", "2023-03-13"),
    ]:
        facts.append(
            dict(
                concept=concept,
                unit="CNY",
                period_start=None,
                period_end=period,
                filed=filed,
                accession="fixture-" + filed,
                source_file="facts.json",
                availability_rule="reviewed_cninfo_date_next_session_close",
                value=float(value),
                value_exact=str(value),
                available_at=clock.date_available(filed),
                published_at=None,
                accounting_standard="CAS",
                statement_scope="consolidated",
                document_sha256="d" * 64,
                publication_evidence_sha256="e" * 64,
                source_url=f"https://static.cninfo.com.cn/finalpage/{filed}/fixture-{filed}.PDF",
                source_page=1,
                source_records=[len(facts) + 1],
            )
        )
    pq.write_table(pa.Table.from_pylist(facts, schema=_SCHEMA), source / "fundamentals.parquet")
    refresh_facts(source)
    macro = tmp_path / "macro.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                dict(
                    prediction_at=clock.decision(day),
                    indicator_id="observed",
                    available_at=clock.decision(days[0]),
                    value=1.0,
                    unit="ratio",
                )
                for day in days
            ]
        ),
        macro,
    )
    cache = EmbeddingCache(tmp_path / "embeddings.sqlite")
    try:
        yield dict(
            source=source,
            destination=tmp_path / "samples",
            clock=clock,
            macros=MacroVectors(macro),
            encoders=Encoders(),
            cache=cache,
            cohort="original_audited",
            context=2,
            company_factors=False,
            fundamental_concepts=CONCEPTS,
            source_unit="CNY",
            batch_rows=2,
            admitted_decisions={clock.decision(day) for day in days[2:5]},
        )
    finally:
        cache.close()


def refresh_facts(source):
    path = source / "manifest.json"
    report = json.loads(path.read_text())
    facts = source / "fundamentals.parquet"
    report["artifacts"]["fundamentals.parquet"] = sha256(facts)
    with pq.ParquetFile(facts) as file:
        report["counts"]["fundamentals"] = file.metadata.num_rows
    atomic_json(path, report)


def change_facts(inputs, edit):
    path = inputs["source"] / "fundamentals.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    edit(rows)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    refresh_facts(inputs["source"])


def test_cny_preserves_native_levels_zero_masks_ages_and_four_modalities(inputs):
    facts_hash = sha256(inputs["source"] / "fundamentals.parquet")
    report = materialize_cohort_asset(**inputs)
    table = pq.read_table(inputs["destination"] / "samples.parquet")
    rows = table.to_pylist()
    assert report["samples"] == 3
    assert report["source_unit"] == "CNY"
    assert report["fundamental_concepts"] == list(CONCEPTS)
    assert report["company_factors_sha256"] is None
    assert "china_sources.py" in report["representation_code"]
    assert [r["session"] for r in rows] == ["2023-03-10", "2023-03-13", "2023-03-14"]
    expected = [
        [math.log1p(100), 0, 0, 1, 1, 0, 0, 0, 0],
        [math.log1p(100), 0, 0, 1, 1, 0, math.log1p(3), math.log1p(3), 0],
        [math.log1p(100), 0, -math.log1p(2), 1, 1, 1, math.log1p(4), math.log1p(4), 0],
    ]
    np.testing.assert_allclose([r["fundamentals"] for r in rows], expected, rtol=1e-7)
    assert table.schema.field("fundamentals").type == pa.list_(pa.float32(), 9)
    for row in rows:
        assert len(row["news"]) == 384 and len(row["charts"]) == 512
        assert row["news_kind_counts"]["summary"] == 1
        assert all(t <= row["prediction_at"] for t in row["input_availability"].values())
        assert row["prediction_at"].year == 2023
    assert sha256(inputs["source"] / "fundamentals.parquet") == facts_hash


@pytest.mark.parametrize(
    "options",
    [
        {"company_factors": True},
        {"source_unit": "USD"},
        {"source_unit": "CAD"},
        {"source_unit": "cny"},
        {"fundamental_concepts": ["us-gaap:Assets:CNY"]},
        {"fundamental_concepts": ["cn-reported:Assets:USD"]},
        {"fundamental_concepts": ["cn-reported:Unknown:CNY"]},
        {"fundamental_concepts": [CONCEPTS[0], "us-gaap:Assets:USD"]},
    ],
)
def test_cny_requires_explicit_native_taxonomy_and_disables_us_gaap_factors(inputs, options):
    with pytest.raises(ValueError):
        materialize_cohort_asset(**{**inputs, **options})
    assert inputs["encoders"].calls == 0
    assert not (inputs["destination"] / "manifest.json").exists()


def test_cny_cannot_be_materialized_on_a_us_clock(inputs):
    inputs["clock"] = MarketClock("US", "2023-01-01", "2025-01-01")
    with pytest.raises(ValueError):
        materialize_cohort_asset(**inputs)
    assert inputs["encoders"].calls == 0


def test_even_a_consistent_us_preparation_cannot_use_cny_mode(inputs):
    clock = MarketClock("US", "2023-01-01", "2025-01-01")
    inputs["clock"] = clock
    inputs["admitted_decisions"] = {clock.decision("2023-03-10")}
    path = inputs["source"] / "manifest.json"
    origin = json.loads(path.read_text())
    origin["market"] = "US"
    origin["policy"]["calendar"] = hashlib.sha256(
        "|".join(t.isoformat() for t in clock.decisions).encode()
    ).hexdigest()
    atomic_json(path, origin)
    change_facts(
        inputs,
        lambda rows: [r.update(available_at=clock.date_available(r["filed"])) for r in rows],
    )
    with pytest.raises(ValueError, match="calendario CN"):
        materialize_cohort_asset(**inputs)


@pytest.mark.parametrize("start", [None, "2022-01-01", "2021-01-01"])
def test_a_known_flow_requires_its_own_valid_period(inputs, start):
    change_facts(
        inputs,
        lambda rows: rows[0].update(concept="cn-reported:Revenue:CNY", period_start=start),
    )
    inputs["fundamental_concepts"] = ["cn-reported:Revenue:CNY"]
    if start != "2021-01-01":
        with pytest.raises(ValueError, match="periodo"):
            materialize_cohort_asset(**inputs)
    else:
        assert materialize_cohort_asset(**inputs)["samples"] == 3
        row = pq.read_table(inputs["destination"] / "samples.parquet").to_pylist()[0]
        np.testing.assert_allclose(row["fundamentals"], [math.log1p(90), 1, 0], rtol=1e-7)


@pytest.mark.parametrize(
    "changes",
    [
        {"unit": "USD"},
        {"concept": "us-gaap:Assets:CNY"},
        {"accounting_standard": "IFRS"},
        {"statement_scope": "separate"},
        {"value": None},
        {"value": float("nan")},
        {"value": float("inf")},
        {"value_exact": "1"},
        {"value_exact": "no es una cifra"},
        {"filed": "2023-03-10"},
        {"period_end": "2023-03-11"},
        {"period_end": "no es una fecha"},
        {"period_start": "2022-01-01"},
        {"published_at": datetime(2023, 3, 9, tzinfo=UTC)},
        {"availability_rule": "period_end"},
        {"available_at": datetime(2023, 3, 9, 7, 5, tzinfo=UTC)},
        {"filed": "2024-01-02", "available_at": datetime(2024, 1, 3, 7, 5, tzinfo=UTC)},
    ],
)
def test_invalid_chinese_facts_fail_before_encoding(inputs, changes):
    change_facts(inputs, lambda rows: rows[0].update(changes))
    with pytest.raises(ValueError):
        materialize_cohort_asset(**inputs)
    assert inputs["encoders"].calls == 0
    assert not (inputs["destination"] / "samples.parquet").exists()
    assert not (inputs["destination"] / "manifest.json").exists()


def test_cny_reuse_conserves_identity_and_rejects_changed_facts(inputs):
    first = materialize_cohort_asset(**inputs)
    calls = inputs["encoders"].calls
    second = materialize_cohort_asset(**inputs)
    assert second == {**first, "reused": True}
    assert inputs["encoders"].calls == calls
    with pytest.raises(ValueError):
        materialize_cohort_asset(**{**inputs, "fundamental_concepts": CONCEPTS[:2]})
    change_facts(inputs, lambda rows: rows[1].update(value=101.0, value_exact="101"))
    with pytest.raises(ValueError):
        materialize_cohort_asset(**inputs)


def test_chinese_facts_remain_causal_when_a_later_report_is_added(inputs, tmp_path):
    first = materialize_cohort_asset(**inputs)
    before = pq.read_table(inputs["destination"] / "samples.parquet")

    def append_later(rows):
        rows.append(
            {
                **rows[1],
                "period_end": "2023-03-31",
                "filed": "2023-04-03",
                "available_at": datetime(2023, 4, 4, 7, 5, tzinfo=UTC),
                "accession": "later",
                "value": 1000.0,
                "value_exact": "1000",
            }
        )

    change_facts(inputs, append_later)
    output = tmp_path / "later"
    later = materialize_cohort_asset(**{**inputs, "destination": output})
    assert later["samples"] == first["samples"]
    assert pq.read_table(output / "samples.parquet").equals(before)
