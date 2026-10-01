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


def test_materialization_writes_bounded_groups_with_identical_rows(tmp_path, encoder_runtime):
    clock = MarketClock("US", "2024-01-01", "2024-12-31")
    source, asset = write_inputs(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_asset(source, prepared, asset, clock)
    macro_path = tmp_path / "macro.parquet"
    write_macro(macro_path, clock)
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    try:
        common = (prepared, macro_path)
        first = materialize_samples(
            *common,
            tmp_path / "small",
            {"assets": [asset]},
            clock,
            FixtureEncoders(),
            cache,
            batch_rows=2,
        )
        second = materialize_samples(
            *common,
            tmp_path / "large",
            {"assets": [asset]},
            clock,
            FixtureEncoders(),
            cache,
            batch_rows=4,
        )
        small = pq.read_table(tmp_path / "small/US/A/samples.parquet")
        large = pq.read_table(tmp_path / "large/US/A/samples.parquet")
        assert first["samples"] == second["samples"] == 5
        assert small.equals(large)
        assert pq.ParquetFile(tmp_path / "small/US/A/samples.parquet").metadata.num_row_groups == 3
        assert first["assets"][0]["limits"]["sample_batch_rows"] == 2
    finally:
        cache.close()


def test_company_factor_schema_changes_cache_and_keeps_missing_values_explicit(
    tmp_path, encoder_runtime
):
    clock = MarketClock("US", "2024-01-01", "2024-12-31")
    source, asset = write_inputs(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_asset(source, prepared, asset, clock)
    macro_path = tmp_path / "macro.parquet"
    write_macro(macro_path, clock)
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    output = tmp_path / "samples"
    try:
        args = (prepared, macro_path, output, {"assets": [asset]}, clock, FixtureEncoders(), cache)
        base = materialize_samples(*args)
        extended = materialize_samples(*args, company_factors=True)
        before, after = base["assets"][0], extended["assets"][0]
        assert before["fingerprint"] != after["fingerprint"]
        assert before["samples"] == after["samples"] == 5
        assert after["schema_version"] == 2
        assert len(after["fundamental_concepts"]) == 15
        assert after["company_factor_derivation"]["definitions"][0] == [
            "current_ratio",
            [["AssetsCurrent", 1]],
            "LiabilitiesCurrent",
        ]
        assert len(after["company_factor_derivation"]["code_sha256"]) == 64
        table = pq.read_table(output / "US/A/samples.parquet")
        assert table.schema.field("fundamentals").type.list_size == 45
        assert table["fundamentals"].to_pylist()[0][23:30] == [0.0] * 7
        assert (output / "US/A/company-factors.parquet").exists()
        reused = materialize_samples(*args, company_factors=True)
        assert reused["assets"][0]["reused"] is True
        (output / "US/A/company-factors.parquet").write_bytes(b"damaged")
        repaired = materialize_samples(*args, company_factors=True)
        assert not repaired["assets"][0].get("reused", False)
        assert pq.read_table(output / "US/A/company-factors.parquet").num_rows == 7
    finally:
        cache.close()


def test_materialization_failure_does_not_publish_partial_samples(tmp_path, encoder_runtime):
    clock = MarketClock("US", "2024-01-01", "2024-12-31")
    source, asset = write_inputs(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_asset(source, prepared, asset, clock)
    macro_path = tmp_path / "macro.parquet"
    write_macro(macro_path, clock)

    class FailingEncoder(FixtureEncoders):
        calls = 0

        def images(self, pngs):
            self.calls += 1
            if self.calls == 3:
                raise ValueError("fallo controlado de codificación")
            return super().images(pngs) + self.calls

    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    try:
        # Gráficos distintos para que la caché no oculte el tercer cálculo.
        import pandas as pd

        prices_path = prepared / "US/A/prices.parquet"
        prices = pd.read_parquet(prices_path)
        prices.loc[64:, "high"] = range(103, 109)
        prices.to_parquet(prices_path, index=False)
        from mars_titan.data.storage import sha256

        manifest_path = prepared / "US/A/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"]["prices.parquet"] = sha256(prices_path)
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="controlado"):
            materialize_samples(
                prepared,
                macro_path,
                tmp_path / "failed",
                {"assets": [asset]},
                clock,
                FailingEncoder(),
                cache,
                batch_rows=1,
            )
        assert not (tmp_path / "failed/US/A/samples.parquet").exists()
        assert not (tmp_path / "failed/US/A/manifest.json").exists()
    finally:
        cache.close()


@pytest.mark.parametrize("custom_calendar", [False, True])
@pytest.mark.parametrize("company_factors", [False, True])
def test_cli_prepares_and_encodes_prices_before_2000(
    tmp_path, monkeypatch, encoder_runtime, custom_calendar, company_factors
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
            "--unreviewed-profile",
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
            *(["--company-factors"] if company_factors else []),
        ],
    )
    assert cli.main() == 0
    rows = pq.read_table(output / "US/A/samples.parquet").to_pylist()
    assert len(rows) == 5
    assert all(row["prediction_at"].year == 1995 for row in rows)
    assert len(rows[0]["fundamentals"]) == (45 if company_factors else 24)


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
