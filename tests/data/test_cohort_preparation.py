"""Preparación recuperable sin confundir normalización y admisión científica."""

import importlib
import json

import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256
from mars_titan.data.temporal import MarketClock


def prepare(*args, **kwargs):
    try:
        module = importlib.import_module("mars_titan.data.cohort_preparation")
    except ModuleNotFoundError:
        pytest.fail("Falta la preparación explícita de cohortes")
    return module.prepare_cohort_asset(*args, **kwargs)


def fixture(tmp_path, market="US"):
    source = tmp_path / "raw"
    source.mkdir()
    (source / "prices.csv").write_text(
        "Date,Open,High,Low,Close,Volume\n2023-07-06,10,12,9,11,100\n2024-01-03,10,12,9,11,100\n"
    )
    (source / "news.jsonl").write_text(
        json.dumps(
            dict(
                Date="2023-07-05",
                Stock_symbol="A",
                Article="La empresa A publica resultados.",
                Article_title="Resultados",
                Url="https://example.org/a",
            )
        )
        + "\n"
    )
    (source / "chart.png").write_bytes(b"Identidad de grafico de una fixture")
    facts = (
        {
            "filings": [
                {
                    "facts": {
                        "us-gaap": {
                            "Assets": {
                                "units": {
                                    "USD": [
                                        dict(
                                            val=100, end="2023-03-31", filed="2023-05-02", accn="a"
                                        ),
                                        dict(
                                            val=200, end="2023-09-30", filed="2024-01-02", accn="b"
                                        ),
                                    ]
                                }
                            }
                        }
                    }
                }
            ]
        }
        if market == "US"
        else [dict(ts_code="A", end_date="20230331", total_assets=100)]
    )
    (source / "facts.json").write_text(json.dumps(facts))
    paths = dict(
        prices=["prices.csv"],
        news=["news.jsonl"],
        fundamentals=["facts.json"],
        charts=["chart.png"],
    )
    asset = dict(
        market=market,
        symbol="A",
        paths=paths,
        hashes={p: sha256(source / p) for values in paths.values() for p in values},
    )
    return source, asset, MarketClock(market, "2023-01-01", "2025-01-01")


def test_preparation_preserves_cohort_and_inner_filing_date(tmp_path):
    source, asset, clock = fixture(tmp_path)
    result = prepare(
        source, tmp_path / "prepared", asset, clock, cohort="original_audited", reviews={}
    )
    folder = tmp_path / "prepared/US/A"
    assert result["cohort_id"] == "original_audited"
    assert result["counts"] == dict(prices=1, news=1, fundamentals=1)
    assert result["training_ready"] is False
    row = pq.read_table(folder / "fundamentals.parquet").to_pylist()[0]
    assert row["available_at"] == clock.decision("2023-05-03")
    assert row["period_end"] == "2023-03-31"
    assert result["reserved_counts"] == dict(prices=1, fundamentals=1)
    assert result["artifacts"]["news/news.parquet"] == sha256(folder / "news/news.parquet")
    assert (
        prepare(source, tmp_path / "prepared", asset, clock, cohort="original_audited", reviews={})[
            "reused"
        ]
        is True
    )


def test_missing_chinese_publication_does_not_become_period_end(tmp_path):
    source, asset, clock = fixture(tmp_path, "CN")
    result = prepare(
        source, tmp_path / "prepared", asset, clock, cohort="original_audited", reviews={}
    )
    assert result["counts"]["fundamentals"] == 0
    assert result["fundamentals_audit"]["missing_publication"] == 1
    assert result["training_ready"] is False


def test_changed_inventory_hash_fails_before_publishing(tmp_path):
    source, asset, clock = fixture(tmp_path)
    asset["hashes"]["prices.csv"] = "0" * 64
    with pytest.raises(ValueError, match="inventario|huella"):
        prepare(source, tmp_path / "prepared", asset, clock, cohort="original_audited", reviews={})
    assert not (tmp_path / "prepared/US/A/manifest.json").exists()


def test_cohort_cannot_change_inside_existing_asset(tmp_path):
    source, asset, clock = fixture(tmp_path)
    destination = tmp_path / "prepared"
    prepare(source, destination, asset, clock, cohort="original_audited", reviews={})
    with pytest.raises(ValueError, match="edición|configuración|identidad"):
        prepare(source, destination, asset, clock, cohort="externally_verified", reviews={})


def test_missing_source_modality_and_wrong_market_are_rejected(tmp_path):
    source, asset, clock = fixture(tmp_path)
    asset["paths"]["charts"] = []
    with pytest.raises(ValueError, match="modalidad"):
        prepare(source, tmp_path / "prepared", asset, clock, cohort="original_audited", reviews={})
    asset["market"] = "CN"
    with pytest.raises(ValueError, match="mercado"):
        prepare(source, tmp_path / "prepared", asset, clock, cohort="original_audited", reviews={})


@pytest.mark.parametrize(
    "changes",
    [
        dict(
            cohort_id="externally_verified",
            news_content_policy="verified_full_articles",
            training_ready=True,
        ),
        dict(counts={"prices": 100, "news": 100, "fundamentals": 100}),
    ],
)
def test_reused_receipt_cannot_change_evidence_or_counts(tmp_path, changes):
    source, asset, clock = fixture(tmp_path)
    destination = tmp_path / "prepared"
    prepare(source, destination, asset, clock, cohort="original_audited", reviews={})
    path = destination / "US/A/manifest.json"
    previous = json.loads(path.read_text())
    previous.update(changes)
    path.write_text(json.dumps(previous))
    with pytest.raises(ValueError, match="recibo|identidad|recuento"):
        prepare(source, destination, asset, clock, cohort="original_audited", reviews={})
