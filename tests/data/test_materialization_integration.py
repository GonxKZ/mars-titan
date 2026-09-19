"""Integración de muestras con codificadores de prueba, sin descarga de pesos."""

import json
import sys
from datetime import timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.embeddings import EmbeddingCache
from mars_titan.data.preparation import prepare_asset
from mars_titan.data.samples import materialize_samples
from mars_titan.data.temporal import MarketClock


class FixtureEncoders:
    spec = {"name": "synthetic_test_fixture", "version": 1}

    def text(self, text):
        return np.full(384, len(text) / 100, dtype=np.float32)

    def images(self, pngs):
        return np.ones((len(pngs), 512), dtype=np.float32)


@pytest.fixture
def encoder_runtime(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)


def write_inputs(tmp_path, year=2024):
    clock = MarketClock("US", f"{year}-01-01", f"{year}-12-31")
    source = tmp_path / "dataset"
    source.mkdir()
    dates = clock.days[:70]
    (source / "a.csv").write_text(
        "Date,Open,High,Low,Close,Volume\n"
        + "".join(f"{day},100,102,99,101,1000\n" for day in dates)
    )
    (source / "a.jsonl").write_text(
        json.dumps(
            {
                "Date": dates[62].isoformat(),
                "Article": "Synthetic test news",
                "Stock_symbol": "A",
            }
        )
        + "\n"
    )
    fact = {
        "end": f"{year - 1}-12-31",
        "val": 100,
        "filed": f"{year}-01-03",
        "accn": "fixture",
    }
    (source / "facts.json").write_text(
        json.dumps({"filings": [{"facts": {"us-gaap": {"Assets": {"units": {"USD": [fact]}}}}}]})
    )
    asset = {
        "symbol": "A",
        "paths": {"prices": ["a.csv"], "news": ["a.jsonl"], "fundamentals": ["facts.json"]},
    }
    return source, asset


def write_macro(path, clock):
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "prediction_at": cut,
                    "indicator_id": "fixture_macro",
                    "value": 100.0,
                    "available_at": cut,
                    "unit": "fixture_index",
                }
                for cut in clock.decisions
            ]
        ),
        path,
    )


def test_materialization_reuses_only_the_same_preparation_calendar(tmp_path, encoder_runtime):
    clock = MarketClock("US", "2024-01-01", "2024-12-31")
    source, asset = write_inputs(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_asset(source, prepared, asset, clock)
    macro_path = tmp_path / "macro.parquet"
    write_macro(macro_path, clock)
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    try:
        args = (prepared, macro_path, tmp_path / "samples", {"assets": [asset]})
        first = materialize_samples(*args, clock, FixtureEncoders(), cache)
        assert first["samples"] == 5
        reused = materialize_samples(*args, clock, FixtureEncoders(), cache)
        assert reused["assets"][0]["reused"] is True
        changed = MarketClock("US", "2024-01-01", "2024-12-31")
        changed.decisions = [value + timedelta(minutes=1) for value in changed.decisions]
        with pytest.raises(ValueError, match="calendario"):
            materialize_samples(*args, changed, FixtureEncoders(), cache)
    finally:
        cache.close()


@pytest.mark.parametrize("custom_calendar", [False, True])
def test_cli_prepares_and_encodes_prices_before_2000(
    tmp_path, monkeypatch, encoder_runtime, custom_calendar
):
    from mars_titan.data import cli, embeddings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(embeddings, "FrozenEncoders", FixtureEncoders)
    source, asset = write_inputs(tmp_path, year=1995)
    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps({"market": "US", "assets": [asset]}))
    prepared = tmp_path / "prepared"
    output = tmp_path / "samples"
    calendar_args = ["--start", "1995-01-01", "--end", "1995-12-31"] if custom_calendar else []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mars-data",
            "prepare",
            "--source",
            str(source),
            "--panel",
            str(panel),
            "--output",
            str(prepared),
            *calendar_args,
        ],
    )
    assert cli.main() == 0
    clock = MarketClock(
        "US",
        "1995-01-01" if custom_calendar else "1990-01-01",
        "1995-12-31" if custom_calendar else "2026-01-01",
    )
    write_macro(prepared / "macro-US.parquet", clock)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mars-data",
            "encode",
            "--panel",
            str(panel),
            "--prepared",
            str(prepared),
            "--output",
            str(output),
            "--cache",
            str(tmp_path / "cache.sqlite"),
            *calendar_args,
        ],
    )
    assert cli.main() == 0
    rows = pq.read_table(output / "US/A/samples.parquet").to_pylist()
    assert len(rows) == 5
    assert all(row["prediction_at"].year == 1995 for row in rows)


@pytest.mark.parametrize("invalid_input", ["originals", "prepared", "calendar"])
def test_cli_validates_encoding_inputs_before_initializing_dependencies(
    tmp_path, monkeypatch, invalid_input
):
    from mars_titan.data import cli, embeddings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        embeddings,
        "FrozenEncoders",
        lambda: pytest.fail("Los codificadores se inicializaron antes de validar"),
    )
    monkeypatch.setattr(
        embeddings, "EmbeddingCache", lambda path: pytest.fail("La caché se abrió antes de validar")
    )
    source, asset = write_inputs(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_asset(source, prepared, asset, MarketClock("US", "2024-01-01", "2024-12-31"))
    manifest = prepared / "US/A/manifest.json"
    previous = manifest.read_bytes()
    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps({"market": "US", "assets": [asset]}))
    output = {
        "originals": source / "samples",
        "prepared": prepared,
        "calendar": tmp_path / "samples",
    }[invalid_input]
    cache = tmp_path / "cache.sqlite"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mars-data",
            "encode",
            "--panel",
            str(panel),
            "--prepared",
            str(prepared),
            "--output",
            str(output),
            "--cache",
            str(cache),
            "--start",
            "2023-01-01" if invalid_input == "calendar" else "2024-01-01",
            "--end",
            "2024-12-31",
        ],
    )
    with pytest.raises(ValueError, match="origen|sobrescribir|calendario"):
        cli.main()
    assert not cache.exists()
    assert manifest.read_bytes() == previous


@pytest.mark.parametrize("link_path", ["US", "US/A"])
@pytest.mark.parametrize("protected_market", ["US", "CN"])
def test_materialization_protects_prepared_manifests_through_output_symlinks(
    tmp_path, encoder_runtime, link_path, protected_market
):
    clock = MarketClock("US", "2024-01-01", "2024-12-31")
    source, asset = write_inputs(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_asset(source, prepared, asset, clock)
    manifest = prepared / "US/A/manifest.json"
    previous = manifest.read_bytes()
    protected_manifest = prepared / protected_market / "A/manifest.json"
    if protected_market != "US":
        protected_manifest.parent.mkdir(parents=True)
        protected_manifest.write_text(
            json.dumps({**json.loads(previous), "market": protected_market})
        )
    protected_previous = protected_manifest.read_bytes()
    output = tmp_path / "samples"
    link = output / link_path
    link.parent.mkdir(parents=True)
    link.symlink_to(
        prepared / link_path.replace("US", protected_market, 1), target_is_directory=True
    )
    macro_path = tmp_path / "macro.parquet"
    write_macro(macro_path, clock)
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    try:
        with pytest.raises(ValueError, match="origen"):
            materialize_samples(
                prepared, macro_path, output, {"assets": [asset]}, clock, FixtureEncoders(), cache
            )
    finally:
        cache.close()
    assert manifest.read_bytes() == previous
    assert protected_manifest.read_bytes() == protected_previous
    assert not (prepared / "US/A/samples.parquet").exists()
