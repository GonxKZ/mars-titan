import hashlib
import importlib
import json
from datetime import UTC, datetime

import pytest


def module():
    try:
        return importlib.import_module("mars_titan.data.universe")
    except ModuleNotFoundError:
        pytest.fail("La selección del piloto sin información futura todavía no existe")


def test_selection_order_does_not_use_sector_future_volume_or_survival():
    rows = [
        {"symbol": symbol, "bytes": size, "sector": "current"}
        for symbol, size in [("A", 1), ("B", 100), ("C", 10000)]
    ]
    order = [x["symbol"] for x in module().selection_order(rows, seed=42)]
    changed = [{**x, "bytes": 0, "sector": "new", "survives_to_test": False} for x in rows[::-1]]
    assert [x["symbol"] for x in module().selection_order(changed, seed=42)] == order


def test_training_coverage_ignores_appended_future_samples():
    cutoff = datetime(2018, 12, 31, 23, 59, tzinfo=UTC)
    past = datetime(2018, 12, 28, 21, 5, tzinfo=UTC)
    future = datetime(2025, 3, 28, 20, 5, tzinfo=UTC)
    rows = [{"prediction_at": past}, {"prediction_at": future}]
    assert module().training_coverage(rows, {past, future}, cutoff) == 1
    assert module().training_coverage(rows, {future}, cutoff) == 0


def test_pilot_cannot_gain_coverage_from_an_empty_news_review_registry(tmp_path):
    from mars_titan.data.inventory import inventory

    source = tmp_path / "dataset"
    source.mkdir()
    database = tmp_path / "inventory.sqlite"
    inventory(source, database)
    reviews = tmp_path / "reviews.json"
    reviews.write_text(json.dumps({"schema_version": 1, "reviews": []}))
    with pytest.raises(ValueError, match="noticias"):
        module().select_verification_pilot(
            source,
            database,
            tmp_path / "missing-macro.parquet",
            tmp_path / "prepared",
            tmp_path / "pilot.json",
            news_reviews_path=reviews,
        )
    assert not (tmp_path / "prepared").exists()
    assert not (tmp_path / "pilot.json").exists()


def test_pilot_propagates_verified_reviews_to_asset_preparation(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    unit = module()
    review = {
        "source_record_hash": "0" * 64,
        "status": "verified_full_article",
        "symbol": "A",
        "source_date": "2018-12-28",
        "source_url": "https://example.org/a",
        "note": "Ejemplo sintético para verificar el paso de argumentos.",
        "checked_at": "2026-09-20T00:00:00+00:00",
        "evidence_url": "https://example.org/source",
        "body_sha256": "1" * 64,
        "body_spans": [[0, 1]],
        "historical_version_verified": False,
    }
    registry = tmp_path / "reviews.json"
    registry.write_text(json.dumps({"schema_version": 1, "reviews": [review]}))
    monkeypatch.setattr(
        unit,
        "entries",
        lambda _: iter(
            [
                {
                    "market": "US",
                    "symbol": "A",
                    "modality": modality,
                    "state": "inspected",
                    "path": modality,
                }
                for modality in ("prices", "news", "fundamentals")
            ]
        ),
    )
    macro = tmp_path / "macro.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"prediction_at": datetime(2018, 12, 28, tzinfo=UTC), "value": 1.0}]),
        macro,
    )

    def prepare(*args, news_reviews=None):
        assert news_reviews == {"0" * 64: review}
        raise RuntimeError("registro estricto recibido")

    monkeypatch.setattr(unit, "prepare_asset", prepare)
    with pytest.raises(RuntimeError, match="registro estricto recibido"):
        unit.select_verification_pilot(
            tmp_path / "source",
            tmp_path / "inventory",
            macro,
            tmp_path / "prepared",
            tmp_path / "pilot.json",
            size=1,
            minimum_samples=1,
            news_reviews_path=registry,
        )


def test_verified_pilot_reconstructs_from_real_adapters_and_past_coverage(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from mars_titan.data.inventory import inventory
    from mars_titan.data.temporal import MarketClock

    source = tmp_path / "dataset"
    prices = source / "time_series/S&P500_time_series/a.csv"
    news = source / "text/sp500_news/A.jsonl"
    facts = source / "table/sp500/A/facts.json"
    for path in (prices, news, facts):
        path.parent.mkdir(parents=True, exist_ok=True)
    clock = MarketClock("US", "2018-01-01", "2019-01-01")
    prices.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        + "".join(f"{day},10,12,9,11,100\n" for day in clock.days[:70])
    )
    raw = {
        "Stock_symbol": "A",
        "Date": str(clock.days[65]),
        "Article": "Cuerpo completo sintético de A.",
        "Url": "https://example.org/a",
    }
    line = json.dumps(raw)
    news.write_text(line + "\n")
    facts.write_text(
        json.dumps(
            {
                "filings": [
                    {
                        "facts": {
                            "us-gaap": {
                                "Assets": {
                                    "units": {
                                        "USD": [
                                            {
                                                "end": "2017-12-31",
                                                "filed": "2018-01-02",
                                                "val": 100,
                                                "accn": "synthetic",
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                    }
                ]
            }
        )
    )
    review = {
        "source_record_hash": hashlib.sha256(line.encode()).hexdigest(),
        "status": "verified_full_article",
        "symbol": "A",
        "source_date": raw["Date"],
        "source_url": raw["Url"],
        "note": "Datos sintéticos, no una noticia real ni una aprobación editorial.",
        "checked_at": "2026-09-20T00:00:00+00:00",
        "evidence_url": "https://example.org/source",
        "body_sha256": hashlib.sha256(raw["Article"].encode()).hexdigest(),
        "body_spans": [[0, len(raw["Article"])]],
        "historical_version_verified": False,
    }
    registry = tmp_path / "reviews.json"
    registry.write_text(json.dumps({"schema_version": 1, "reviews": [review]}))
    database = tmp_path / "inventory.sqlite"
    inventory(source, database)
    macro = tmp_path / "macro.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"prediction_at": value, "value": 1.0} for value in clock.decisions[:70]]
        ),
        macro,
    )
    args = (source, database, macro, tmp_path / "prepared", tmp_path / "pilot.json")
    result = module().select_verification_pilot(
        *args, size=1, minimum_samples=1, news_reviews_path=registry
    )
    assert result["assets"][0]["eligible_before_cutoff"] == 4
    assert result["news_content_policy"] == "verified_full_articles"
    assert result["news_review_selection_is_retrospective"] is True
    assert result == module().select_verification_pilot(
        *args, size=1, minimum_samples=1, news_reviews_path=registry
    )
