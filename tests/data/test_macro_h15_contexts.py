"""Integración documental de tipos, con disponibilidad y metadatos explícitos."""

import csv
import importlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from mars_titan.data.macro import calculate_macro
from mars_titan.data.macro_h15_archive import prepare_h15_archive
from mars_titan.data.macro_h15_documents import _SERIES
from mars_titan.data.storage import sha256
from mars_titan.data.temporal import MarketClock
from tests.data import test_macro_h15_archive as documents
from tests.data.test_macro_acquisition import _alfred_zip

CATALOG = Path("data/catalogs/macro-indicators.csv")


@pytest.fixture
def source(tmp_path):
    return documents.source.__wrapped__(tmp_path)


def module():
    try:
        return importlib.import_module("mars_titan.data.macro_h15_contexts")
    except ModuleNotFoundError:
        pytest.fail("Falta el adaptador de contextos H.15")


def metadata(tmp_path, *, first="1999-01-01", adjustment="Not Seasonally Adjusted"):
    archives = {}
    for identifier, series in _SERIES.values():
        path = tmp_path / f"{series}.zip"
        url = f"https://alfred.stlouisfed.org/series?seid={series}"
        path.write_bytes(
            _alfred_zip(
                series,
                ["2000-01-10"],
                [],
                metadata=(
                    f"Link: {url}\nUnits\nPercent  {first}  Current\n"
                    f"Seasonal Adjustment\n{adjustment}  {first}  Current\n"
                ),
            )
        )
        archives[identifier] = dict(path=path.name, sha256=sha256(path), series_id=series, url=url)
    manifest = tmp_path / "metadata.json"
    manifest.write_text(json.dumps(dict(schema_version=1, archives=archives)))
    return manifest


def load(source, **kwargs):
    prepare_h15_archive(source["manifest"], source["output"], cutoff="2000-02-29")
    return module().load_h15_events(source["manifest"], source["output"], CATALOG, **kwargs)


def produce(source, destination, **kwargs):
    return module().prepare_h15_contexts(
        source["manifest"],
        source["output"],
        CATALOG,
        destination,
        start="2000-01-07",
        end="2000-01-21",
        **kwargs,
    )


def test_markets_share_documented_events_and_missing_metadata_stays_missing(source):
    events, catalog, identity = load(source)
    assert len(events) == 30 and len(catalog) == 140
    assert identity["document_observations"] == 30
    assert all(
        row["value"] is None and row["missing_reason"] == "missing_historical_metadata"
        for row in events
    )
    row = next(
        r
        for r in events
        if r["indicator_id"] == "us_fed_funds" and r["period_start"] == "2000-01-03"
    )
    assert row["document_value"] == 0.0 and row["value_exact"] == "0.00"
    assert row["seasonal_adjustment"] is None
    assert row["source_hash"] == sha256(source["pdf"])
    assert row["document_availability"] == {
        "US": "2000-01-11T21:05:00+00:00",
        "CN": "2000-01-12T07:05:00+00:00",
    }


def test_future_metadata_does_not_backdate_observations(source, tmp_path):
    events, _, _ = load(source, metadata_manifest=metadata(tmp_path, first="2005-06-28"))
    assert len(events) == 30
    assert all(
        r["value"] is None and r["missing_reason"] == "missing_historical_metadata" for r in events
    )


def test_verified_zero_is_observed_and_publication_controls_both_markets(source, tmp_path):
    events, catalog, _ = load(source, metadata_manifest=metadata(tmp_path))
    row = next(
        r
        for r in events
        if r["indicator_id"] == "us_fed_funds" and r["period_start"] == "2000-01-03"
    )
    assert row["value"] == 0.0 and row["missing_reason"] is None
    design = [e for e in catalog if e["id"] == "us_fed_funds"]
    for market, first in (("US", "2000-01-11"), ("CN", "2000-01-12")):
        result = calculate_macro([row], design, MarketClock(market, "2000-01-01", "2000-01-14"))
        assert all(
            r["value"] is None for r in result if r["prediction_at"].date().isoformat() < first
        )
        assert all(
            r["value"] == 0.0 for r in result if r["prediction_at"].date().isoformat() >= first
        )


def test_panel_keeps_catalog_and_only_supported_curves(source, tmp_path):
    meta = metadata(tmp_path)
    load(source, metadata_manifest=meta)
    result = produce(source, tmp_path / "panel", metadata_manifest=meta)
    assert result["training_ready"] is False and result["admission_required"] is True
    assert result["observations"] == 30
    for market, first in (("US", "2000-01-11"), ("CN", "2000-01-12")):
        rows = pq.read_table(tmp_path / "panel" / f"macro-{market}.parquet").to_pylist()
        days = {r["prediction_at"] for r in rows}
        assert len(rows) == 140 * len(days)
        assert all(
            r["value"] is None for r in rows if r["prediction_at"].date().isoformat() < first
        )
        observed = {r["indicator_id"]: r for r in rows if r["prediction_at"] == max(days)}
        for name, expected in (
            ("us_curve_10y_2y", 2.0),
            ("us_curve_10y_3m", 2.9),
            ("us_curve_30y_10y", 1.0),
            ("us_curve_5y_2y", 1.0),
        ):
            assert observed[name]["value"] == pytest.approx(expected, abs=1e-14)
            assert observed[name]["unit"] == "percentage_points"
        for name in ("us_treasury_10y_change_21obs", "us_fed_funds_change_21d", "us_breakeven_10y"):
            assert observed[name]["value"] is None and observed[name]["missing_reason"]


def test_daily_funds_lag_uses_calendar_days_instead_of_business_observations(source, tmp_path):
    events, catalog, _ = load(source, metadata_manifest=metadata(tmp_path))
    design = [
        e
        for e in catalog
        if e["id"]
        in {
            "us_fed_funds",
            "us_fed_funds_change_21d",
            "us_treasury_10y",
            "us_treasury_10y_change_21obs",
        }
    ]
    base = next(r for r in events if r["indicator_id"] == "us_fed_funds")
    days = [day for day in MarketClock("US", "1999-11-01", "2000-01-07").days][-22:]
    rows = []
    for identifier in ("us_fed_funds", "us_treasury_10y"):
        for i, day in enumerate(days):
            rows.append(dict(base, indicator_id=identifier, period_start=str(day), value=float(i)))
    result = calculate_macro(
        rows,
        design,
        MarketClock("US", "2000-01-01", "2000-01-14"),
        daily_lag_policy="valid_observations",
    )
    final = {
        r["indicator_id"]: r
        for r in result
        if r["prediction_at"] == max(v["prediction_at"] for v in result)
    }
    assert final["us_treasury_10y_change_21obs"]["value"] == 21.0
    assert final["us_fed_funds_change_21d"]["value"] is not None
    assert final["us_fed_funds_change_21d"]["value"] != 21.0


@pytest.mark.parametrize("name", ["observations.parquet", "configuration.json", "report.json"])
def test_changed_document_edition_cannot_be_admitted(source, name):
    load(source)
    path = source["output"] / name
    if name.endswith(".json"):
        value = json.loads(path.read_text())
        value["training_ready"] = True
        path.write_text(json.dumps(value))
    else:
        path.write_bytes(b"corrupto")
    with pytest.raises(ValueError):
        module().load_h15_events(source["manifest"], source["output"], CATALOG)


def test_metadata_hash_and_catalog_definition_are_checked(source, tmp_path):
    meta = metadata(tmp_path)
    load(source, metadata_manifest=meta)
    (tmp_path / "DFF.zip").write_bytes(b"alterado")
    with pytest.raises(ValueError, match="huella"):
        module().load_h15_events(
            source["manifest"], source["output"], CATALOG, metadata_manifest=meta
        )
    entries = list(csv.DictReader(CATALOG.open()))
    next(e for e in entries if e["id"] == "us_fed_funds")["frequency"] = "D"
    changed = tmp_path / "catalog.csv"
    with changed.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=entries[0].keys())
        writer.writeheader()
        writer.writerows(entries)
    with pytest.raises(ValueError, match="catálogo|definición"):
        module().load_h15_events(source["manifest"], source["output"], changed)


def test_recovery_verifies_artifacts_without_rewriting(source, tmp_path):
    load(source)
    destination = tmp_path / "panel"
    first = produce(source, destination)
    before = {p: (sha256(p), p.stat().st_mtime_ns) for p in destination.iterdir()}
    assert produce(source, destination)["reused"] is True
    assert before == {p: (sha256(p), p.stat().st_mtime_ns) for p in before}
    assert first["training_ready"] is False
    (destination / "macro-US.parquet").write_bytes(b"alterado")
    with pytest.raises(ValueError, match="artefacto|huella"):
        produce(source, destination)


def test_closed_cutoff_and_unsupported_alfred_union_fail_before_output(source, tmp_path):
    load(source)
    output = tmp_path / "panel"
    with pytest.raises(ValueError, match="2024|reserva"):
        module().prepare_h15_contexts(
            source["manifest"],
            source["output"],
            CATALOG,
            output,
            start="2000-01-01",
            end="2024-01-01",
        )
    with pytest.raises(ValueError, match="ALFRED|unión"):
        module().prepare_h15_contexts(
            source["manifest"],
            source["output"],
            CATALOG,
            output,
            start="2000-01-01",
            end="2000-01-21",
            alfred_source=tmp_path,
        )
    assert not output.exists()


def revision(source):
    root = source["manifest"].parent
    meta = json.loads(source["manifest"].read_text())
    old = meta["documents"][0]
    new = json.loads(json.dumps(old))
    new["publication_date"] = "2000-01-18"
    for kind in ("html", "pdf"):
        ref = new[kind]
        content = (root / ref["path"]).read_bytes()
        content = content.replace(b"January 10, 2000", b"January 18, 2000").replace(
            b"4.04", b"9.04"
        )
        if kind == "pdf":
            content = content.replace(b"Fixture", b"Second fixture")
        path = root / f"second.{kind}"
        path.write_bytes(content)
        ref.update(
            path=path.name, sha256=sha256(path), url=ref["url"].replace("20000110", "20000118")
        )
        receipt = json.loads((root / ref["receipt"]["path"]).read_text())
        receipt.update(url=ref["url"], final_url=ref["url"])
        receipt.update(
            {
                "body_sha256" if kind == "html" else "sha256": ref["sha256"],
                "body_bytes" if kind == "html" else "bytes": len(content),
            }
        )
        receipt_path = root / f"second-{kind}-receipt.json"
        receipt_path.write_text(json.dumps(receipt))
        ref["receipt"] = dict(path=receipt_path.name, sha256=sha256(receipt_path))
    review = json.loads(source["review"].read_text())
    review.update(
        publication_date=new["publication_date"],
        html_sha256=new["html"]["sha256"],
        pdf_sha256=new["pdf"]["sha256"],
    )
    for row in review["observations"]:
        if row["indicator_id"] == "us_treasury_10y" and row["period_start"] == "2000-01-07":
            row["value_exact"] = "9.04"
    path = root / "second-review.json"
    path.write_text(json.dumps(review))
    new["review"] = dict(path=path.name, sha256=sha256(path))
    meta["documents"].append(new)
    source["manifest"].write_text(json.dumps(meta))


def test_new_release_preserves_prior_version_and_decisions(source, tmp_path):
    revision(source)
    meta = metadata(tmp_path)
    events, _, _ = load(source, metadata_manifest=meta)
    assert len(events) == 60
    versions = [
        r
        for r in events
        if r["indicator_id"] == "us_treasury_10y" and r["period_start"] == "2000-01-07"
    ]
    assert [(r["realtime_start"], r["value"]) for r in versions] == [
        ("2000-01-10", 4.04),
        ("2000-01-18", 9.04),
    ]
    produce(source, tmp_path / "panel", metadata_manifest=meta)
    rows = pq.read_table(tmp_path / "panel/macro-US.parquet").to_pylist()
    curve = {
        str(r["prediction_at"].date()): r["value"]
        for r in rows
        if r["indicator_id"] == "us_curve_10y_2y"
    }
    assert curve["2000-01-11"] == pytest.approx(2.0)
    assert curve["2000-01-18"] == pytest.approx(2.0)
    assert curve["2000-01-19"] == pytest.approx(7.0)


def test_same_release_conflict_does_not_replace_a_version(source):
    revision(source)
    meta = json.loads(source["manifest"].read_text())
    meta["documents"][1]["publication_date"] = "2000-01-10"
    source["manifest"].write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="repite|versi"):
        load(source)


@pytest.mark.parametrize("field", ["observations", "markets", "scope", "indicator_ids"])
def test_recovery_rejects_changed_report_counts_and_scope(source, tmp_path, field):
    load(source)
    output = tmp_path / "panel"
    produce(source, output)
    path = output / "report.json"
    report = json.loads(path.read_text())
    report[field] = 999
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="recibo|recuento|ámbito"):
        produce(source, output)


def test_interrupted_publication_recovers_without_modifying_document_source(
    source, tmp_path, monkeypatch
):
    load(source)
    before = {p: (sha256(p), p.stat().st_mtime_ns) for p in source["output"].iterdir()}
    output = tmp_path / "panel"
    original = module()._publish_directory

    def fail(*args):
        raise OSError("Interrupción controlada antes de confirmar")

    monkeypatch.setattr(module(), "_publish_directory", fail)
    with pytest.raises(OSError, match="Interrupción"):
        produce(source, output)
    assert not output.exists()
    monkeypatch.setattr(module(), "_publish_directory", original)
    assert produce(source, output)["status"] == "completed"
    assert before == {p: (sha256(p), p.stat().st_mtime_ns) for p in before}


def test_catalog_text_and_total_cells_are_bounded_before_calculation(source, tmp_path, monkeypatch):
    load(source)
    entries = list(csv.DictReader(CATALOG.open()))
    next(e for e in entries if e["id"] == "us_curve_10y_2y")["unit"] = "x" * 1000
    changed = tmp_path / "long-catalog.csv"
    with changed.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=entries[0].keys())
        writer.writeheader()
        writer.writerows(entries)
    with pytest.raises(ValueError, match="unidad|catálogo|longitud"):
        module().load_h15_events(source["manifest"], source["output"], changed)
    monkeypatch.setattr(module(), "_MAX_CELLS", 100)
    output = tmp_path / "too-many-cells"
    with pytest.raises(ValueError, match="celdas"):
        produce(source, output)
    assert not output.exists()
