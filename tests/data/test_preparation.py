import importlib
import json

import pytest

from mars_titan.data.temporal import MarketClock


def preparation_module():
    try:
        return importlib.import_module("mars_titan.data.preparation")
    except ModuleNotFoundError:
        pytest.fail("La preparación reanudable de paneles todavía no existe")


def test_panel_selection_is_deterministic_and_keeps_sector_representatives():
    candidates = [
        dict(symbol=f"{s}{n}", sector=s, bytes=n * 100, paths={})
        for s in ("Finance", "Technology")
        for n in range(1, 6)
    ]
    module = preparation_module()
    selected = module.select_panel(candidates, per_sector=2, seed=42)
    assert selected == module.select_panel(list(reversed(candidates)), per_sector=2, seed=42)
    assert {x["sector"] for x in selected} == {"Finance", "Technology"}
    assert len({x["symbol"] for x in selected}) == 4
    assert sorted(x["bytes"] for x in selected) == [300, 300, 500, 500]


def test_preparation_resume_verifies_artifact_and_source_hashes(tmp_path):
    module = preparation_module()
    source = tmp_path / "dataset"
    source.mkdir()
    prices = source / "a.csv"
    prices.write_text("Date,Open,High,Low,Close,Volume\n2024-07-01,10,12,9,11,100\n")
    news = source / "A.jsonl"
    news.write_text('{"Date":"2024-06-28","Article":"A results","Stock_symbol":"A"}\n')
    facts = source / "facts.json"
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
                                                "val": 100,
                                                "end": "2024-03-31",
                                                "filed": "2024-05-02",
                                                "accn": "x",
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
    entry = {
        "symbol": "A",
        "sector": "Technology",
        "paths": {"prices": ["a.csv"], "news": ["A.jsonl"], "fundamentals": ["facts.json"]},
    }
    clock = MarketClock("US", "2023-01-01", "2025-12-31")
    destination = tmp_path / "prepared"
    result = module.prepare_asset(source, destination, entry, clock)
    assert result["counts"] == {"prices": 1, "news": 1, "fundamentals": 1}
    assert result["reused"] is False
    assert module.prepare_asset(source, destination, entry, clock)["reused"] is True
    output = destination / "US" / "A" / "prices.parquet"
    output.write_bytes(b"broken")
    assert module.prepare_asset(source, destination, entry, clock)["reused"] is False
    prices.write_text("Date,Open,High,Low,Close,Volume\n2024-07-01,10,12,9,12,100\n")
    assert module.prepare_asset(source, destination, entry, clock)["reused"] is False


def test_preparation_refuses_source_path_escape(tmp_path):
    module = preparation_module()
    source = tmp_path / "dataset"
    source.mkdir()
    entry = {"symbol": "A", "paths": {"prices": ["../other.csv"]}}
    with pytest.raises(ValueError, match="origen"):
        module.prepare_asset(
            source, tmp_path / "prepared", entry, MarketClock("US", "2024-01-01", "2024-12-31")
        )


def test_preparation_refuses_output_inside_source(tmp_path):
    module = preparation_module()
    with pytest.raises(ValueError, match="origen"):
        module.prepare_asset(
            tmp_path,
            tmp_path / "prepared",
            {"symbol": "A", "paths": {}},
            MarketClock("US", "2024-01-01", "2024-12-31"),
        )
