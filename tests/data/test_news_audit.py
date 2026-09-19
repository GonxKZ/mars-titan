"""Evidencia reproducible y salidas separadas de las noticias originales."""

import importlib
import json

import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256
from mars_titan.data.temporal import MarketClock


def auditor():
    try:
        return importlib.import_module("mars_titan.data.news_audit").audit_news_panel
    except ModuleNotFoundError:
        pytest.fail("Falta la auditoría de admisión de noticias")


def inputs(tmp_path):
    source = tmp_path / "dataset"
    source.mkdir()
    row = {"Date": "2024-07-05", "Stock_symbol": "A", "Article": "Resultados de la empresa A"}
    first = source / "a.jsonl"
    first.write_text(
        "\n".join(
            json.dumps(value)
            for value in [
                row,
                {**row, "Stock_symbol": None},
                {**row, "Date": "2024-07-05T12:00:00"},
            ]
        )
        + "\n"
    )
    second = source / "copy.jsonl"
    second.write_text(json.dumps(row) + "\n")
    return source, {
        "market": "US",
        "assets": [{"symbol": "A", "paths": {"news": ["a.jsonl", "copy.jsonl"]}}],
    }


def test_audit_reconciles_rows_and_writes_typed_parquet_without_changing_sources(tmp_path):
    source, panel = inputs(tmp_path)
    hashes = {name: sha256(source / name) for name in ("a.jsonl", "copy.jsonl")}
    output = tmp_path / "validated"
    report = auditor()(source, panel, output, MarketClock("US", "2024-01-01", "2025-01-01"))
    assert report["raw_records"] == 4
    assert report["accepted"] == 1
    assert report["semantic_validation_complete"] is False
    assert report["historical_body_version_verified"] is False
    assert report["calendar"]["first_session"] == "2024-01-02"
    assert report["calendar"]["last_session"] == "2024-12-31"
    assert len(report["calendar"]["decisions_sha256"]) == 64
    assert report["excluded"] == 3
    assert report["reasons"] == {
        "missing_symbol_evidence": 1,
        "unverified_timezone": 1,
        "duplicate_across_files": 1,
    }
    assert report["lag2_shift_hours"] == {"min": 24.0, "max": 24.0}
    assert report["source_hashes"] == hashes
    table = pq.read_table(output / "US/A/news.parquet")
    assert table.num_rows == 1
    assert str(table.schema.field("event_at").type) == "timestamp[us, tz=UTC]"
    assert str(table.schema.field("language").type) == "string"
    rejected = pq.read_table(output / "US/A/exclusions.parquet")
    assert rejected.num_rows == 3
    assert all(rejected["source_record_hash"].to_pylist())
    assert {name: sha256(source / name) for name in hashes} == hashes
    assert json.loads((output / "manifest.json").read_text()) == report


def test_audit_refuses_overwrite_or_source_destination_before_reading(tmp_path):
    source, panel = inputs(tmp_path)
    clock = MarketClock("US", "2024-01-01", "2025-01-01")
    with pytest.raises(ValueError, match="origen"):
        auditor()(source, panel, source / "out", clock)
    assert not (source / "out").exists()
    output = tmp_path / "existing"
    output.mkdir()
    (output / "manifest.json").write_text("anterior")
    with pytest.raises(ValueError, match="nuevo"):
        auditor()(source, panel, output, clock)
    assert (output / "manifest.json").read_text() == "anterior"


@pytest.mark.parametrize("symbol,path", [("../escape", "a.jsonl"), ("A", "../other.jsonl")])
def test_audit_rejects_paths_outside_the_declared_source(tmp_path, symbol, path):
    source, panel = inputs(tmp_path)
    panel["assets"][0].update(symbol=symbol, paths={"news": [path]})
    output = tmp_path / "validated"
    with pytest.raises(ValueError):
        auditor()(source, panel, output, MarketClock("US", "2024-01-01", "2025-01-01"))
    assert not output.exists()


def test_cli_report_cannot_overwrite_the_panel_or_a_derived_partition(tmp_path, monkeypatch):
    import sys

    auditor()
    from mars_titan.data import news_audit

    source, panel = inputs(tmp_path)
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(json.dumps(panel))
    original = panel_path.read_bytes()
    for report in (panel_path, tmp_path / "out/US/A/news.parquet"):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "audit",
                "--source",
                str(source),
                "--panel",
                str(panel_path),
                "--output",
                str(tmp_path / "out"),
                "--report",
                str(report),
            ],
        )
        with pytest.raises(ValueError):
            news_audit.main()
        assert panel_path.read_bytes() == original
        assert not (tmp_path / "out").exists()
