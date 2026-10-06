"""Historia explícita de versiones publicadas, sin retrofechar ni reescalar niveles."""

import csv
import importlib
import json
from datetime import timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.macro import _exclusion, calculate_macro
from mars_titan.data.macro_acquisition import _exclusion as acquisition_exclusion
from mars_titan.data.storage import sha256
from mars_titan.data.temporal import MarketClock

HASH_V3, HASH_V4 = "3" * 64, "4" * 64


def write_catalog(path, entries):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=entries[0])
        writer.writeheader()
        writer.writerows(entries)


def sources(tmp_path, start="2022-11-09", end="2022-11-14"):
    folder = tmp_path / "sources"
    folder.mkdir()
    entries = list(csv.DictReader(Path("data/catalogs/macro-indicators.csv").open()))
    stress = next(entry for entry in entries if entry["id"] == "us_financial_stress")
    v3_entry = stress | {
        "id": "us_financial_stress_v3",
        "series_id": "STLFSI3",
        "source_url": "https://fred.stlouisfed.org/series/STLFSI3",
    }
    base_catalog, v3_catalog = folder / "base.csv", folder / "v3.csv"
    write_catalog(base_catalog, entries)
    write_catalog(v3_catalog, [v3_entry])
    clock = MarketClock("US", start, end)
    complete_clock = MarketClock("US", "2022-01-01", "2023-12-31")
    first = complete_clock.decision("2022-01-14")
    transition = complete_clock.decision("2022-11-11")
    rows, v3_rows = [], []
    for moment in clock.decisions:
        for entry in entries:
            is_stress = entry["id"] == "us_financial_stress"
            admitted = not is_stress or moment >= transition
            rows.append(
                dict(
                    prediction_at=moment,
                    indicator_id=entry["id"],
                    value=4.0 if admitted else None,
                    available_at=moment if admitted else None,
                    period_start="2022-01-07" if admitted else None,
                    missing_reason=None if admitted else "not_yet_available",
                    source_hashes=[HASH_V4] if admitted else [],
                    unit="Index" if is_stress else entry["unit"],
                    seasonal_adjustment="Not Seasonally Adjusted",
                )
            )
        v3_rows.append(
            dict(
                rows[-1],
                indicator_id="us_financial_stress_v3",
                value=3.0 if moment >= first else None,
                available_at=moment if moment >= first else None,
                period_start="2022-01-07" if moment >= first else None,
                missing_reason=None if moment >= first else "not_yet_available",
                source_hashes=[HASH_V3] if moment >= first else [],
                unit="Index",
            )
        )
    schema = pa.schema(
        [
            ("prediction_at", pa.timestamp("us", tz="UTC")),
            ("indicator_id", pa.string()),
            ("value", pa.float64()),
            ("available_at", pa.timestamp("us", tz="UTC")),
            ("period_start", pa.string()),
            ("missing_reason", pa.string()),
            ("source_hashes", pa.list_(pa.string())),
            ("unit", pa.string()),
            ("seasonal_adjustment", pa.string()),
        ]
    )
    result = []
    for name, data, catalog in (("base", rows, base_catalog), ("v3", v3_rows, v3_catalog)):
        panel, receipt = folder / f"{name}.parquet", folder / f"{name}.json"
        pq.write_table(pa.Table.from_pylist(data, schema=schema), panel, row_group_size=143)
        receipt.write_text(
            json.dumps(
                dict(
                    schema_version=2,
                    sha256=sha256(panel),
                    catalog_sha256=sha256(catalog),
                    market="US",
                    start=start,
                    end=end,
                    rows=len(data),
                )
            )
        )
        result.extend((panel, receipt, catalog))
    return result, start, end


def compose(fixture, output, **options):
    paths, start, end = fixture
    api = importlib.import_module("mars_titan.data.macro_stress_history")
    return api.compose_stress_history(
        *paths, output, market="US", start=options.get("start", start), end=options.get("end", end)
    )


def rewrite_panel(fixture, number, mutate, *, receipt_matches=True):
    panel, receipt = fixture[0][number : number + 2]
    table = pq.read_table(panel)
    rows = table.to_pylist()
    mutate(rows)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), panel, row_group_size=143)
    if receipt_matches:
        report = json.loads(receipt.read_text())
        report.update(sha256=sha256(panel), rows=len(rows))
        receipt.write_text(json.dumps(report))


def test_published_versions_switch_exactly_without_changing_original_catalog(tmp_path):
    fixture = sources(tmp_path)
    before = {str(p): sha256(p) for p in fixture[0]}
    output = tmp_path / "history"
    report = compose(fixture, output)
    rows = pq.read_table(output / "macro.parquet").to_pylist()
    assert [row["value"] for row in rows] == [3.0, 3.0, 4.0, 4.0]
    assert [row["source_hashes"] for row in rows] == [[HASH_V3], [HASH_V3], [HASH_V4], [HASH_V4]]
    assert {row["indicator_id"] for row in rows} == {"us_financial_stress"}
    old = list(csv.DictReader(fixture[0][2].open()))
    new = list(csv.DictReader((output / "catalog.csv").open()))
    assert len(new) == 140
    assert [r["id"] for r in new] == [r["id"] for r in old]
    assert next(r for r in new if r["id"] == "us_financial_stress")["source_url"] == (
        "https://fred.stlouisfed.org/series/STLFSI4"
    )
    assert [r for r in new if r["id"] != "us_financial_stress"] == [
        r for r in old if r["id"] != "us_financial_stress"
    ]
    assert before == {str(p): sha256(p) for p in fixture[0]}
    assert report["sha256"] == sha256(output / "macro.parquet")
    assert report["catalog_sha256"] == sha256(output / "catalog.csv")
    assert [r["series_id"] for r in report["lineage"]] == ["STLFSI3", "STLFSI4"]
    assert [r["source_url"] for r in report["lineage"]] == [
        "https://fred.stlouisfed.org/series/STLFSI3",
        "https://fred.stlouisfed.org/series/STLFSI4",
    ]
    assert set(report["code_sha256"]) >= {
        "macro_stress_history.py",
        "macro_model_vintages.py",
        "macro.py",
    }
    with pytest.raises(FileExistsError):
        compose(fixture, output)


def test_v3_only_starts_after_its_real_publication_and_missing_values_remain_missing(tmp_path):
    fixture = sources(tmp_path, "2022-01-12", "2022-01-18")
    compose(fixture, tmp_path / "history")
    rows = pq.read_table(tmp_path / "history/macro.parquet").to_pylist()
    assert [r["value"] for r in rows] == [None, None, 3.0, 3.0]
    assert rows[2]["prediction_at"].isoformat() == "2022-01-14T21:05:00+00:00"
    assert all(r["missing_reason"] for r in rows[:2])


@pytest.mark.parametrize("number", [0, 3])
@pytest.mark.parametrize("defect", ["hash", "future", "provenance", "duplicate", "missing"])
def test_invalid_source_panel_cannot_publish_history(tmp_path, number, defect):
    fixture = sources(tmp_path)

    def mutate(rows):
        identifier = "us_financial_stress" if number == 0 else "us_financial_stress_v3"
        selected = next(r for r in reversed(rows) if r["indicator_id"] == identifier)
        if defect == "hash":
            selected["value"] = 99.0
        elif defect == "future":
            selected["available_at"] = selected["prediction_at"] + timedelta(days=1)
        elif defect == "provenance":
            selected["source_hashes"] = ["not-a-hash"]
        elif defect == "duplicate":
            rows.append(dict(selected))
        else:
            rows.remove(selected)

    rewrite_panel(fixture, number, mutate, receipt_matches=defect != "hash")
    with pytest.raises(ValueError):
        compose(fixture, tmp_path / "history")
    assert not (tmp_path / "history").exists()


@pytest.mark.parametrize(
    "defect", ["series", "policy", "receipt_catalog", "report_market", "report_range"]
)
def test_false_identity_or_receipt_never_authorizes_a_composition(tmp_path, defect):
    fixture = sources(tmp_path)
    paths = fixture[0]
    catalog = list(csv.DictReader(paths[5].open()))
    report = json.loads(paths[4].read_text())
    if defect == "series":
        catalog[0]["series_id"] = "STLFSI4"
    elif defect == "policy":
        catalog[0]["vintage_policy"] = "ALFRED_OR_RELEASE_ARCHIVE"
    elif defect == "receipt_catalog":
        report["catalog_sha256"] = "0" * 64
    elif defect == "report_market":
        report["market"] = "CN"
    else:
        report["start"] = "2023-01-01"
    write_catalog(paths[5], catalog)
    if defect != "receipt_catalog":
        report["catalog_sha256"] = sha256(paths[5])
    paths[4].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        compose(fixture, tmp_path / "history")
    assert not (tmp_path / "history").exists()


@pytest.mark.parametrize(
    "number,start,end", [(3, "2022-01-13", "2022-01-18"), (0, "2022-11-10", "2022-11-14")]
)
def test_a_version_cannot_be_backdated_in_the_source_panel(tmp_path, number, start, end):
    fixture = sources(tmp_path, start, end)

    def mutate(rows):
        identifier = "us_financial_stress" if number == 0 else "us_financial_stress_v3"
        selected = next(r for r in rows if r["indicator_id"] == identifier)
        selected.update(
            value=1.0,
            available_at=selected["prediction_at"],
            period_start="2022-01-07",
            source_hashes=[HASH_V3],
            missing_reason=None,
        )

    rewrite_panel(fixture, number, mutate)
    with pytest.raises(ValueError):
        compose(fixture, tmp_path / "history")


@pytest.mark.parametrize(
    "start,end",
    [("2023-01-01", "2024-01-01"), ("2024-01-01", "2024-01-02"), ("2023-01-02", "2023-01-01")],
)
def test_reserved_or_inverted_output_period_is_rejected_before_publication(tmp_path, start, end):
    with pytest.raises(ValueError):
        compose(sources(tmp_path), tmp_path / "history", start=start, end=end)
    assert not (tmp_path / "history").exists()


def test_corrupt_parquet_footer_is_rejected_before_decoding(tmp_path):
    fixture = sources(tmp_path)
    fixture[0][3].write_bytes(b"PAR1" + (1024**3).to_bytes(4, "little") + b"PAR1")
    receipt = json.loads(fixture[0][4].read_text())
    receipt["sha256"] = sha256(fixture[0][3])
    fixture[0][4].write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        compose(fixture, tmp_path / "history")
    assert not (tmp_path / "history").exists()


def test_source_dates_cannot_contradict_the_pre2024_receipt(tmp_path):
    fixture = sources(tmp_path)

    def mutate(rows):
        rows.append(
            dict(rows[-1], prediction_at=MarketClock("US", "2024-01-01", "2024-01-03").decisions[0])
        )

    rewrite_panel(fixture, 3, mutate)
    with pytest.raises(ValueError, match="fecha|periodo|recibo"):
        compose(fixture, tmp_path / "history")
    assert not (tmp_path / "history").exists()


def test_stress_history_is_compatible_with_composition_and_admission(tmp_path):
    from mars_titan.data.macro_coverage import assess_macro_completeness
    from mars_titan.data.macro_edition import compose_macro_edition

    fixture = sources(tmp_path)
    output = tmp_path / "history"
    compose(fixture, output)
    compose_macro_edition(
        fixture[0][0],
        [(output / "macro.parquet", ["us_financial_stress"])],
        output / "catalog.csv",
        tmp_path / "combined",
        market="US",
        start=fixture[1],
        end=fixture[2],
    )
    admission = assess_macro_completeness(
        tmp_path / "combined/macro.parquet",
        output / "catalog.csv",
        tmp_path / "admission",
        market="US",
        start=fixture[1],
        end=fixture[2],
    )
    assert admission["complete_decisions"] == 4
    assert admission["required_indicator_count"] == 140


def test_a_derived_formula_cannot_mix_the_two_stress_versions(tmp_path):
    fixture = sources(tmp_path)
    catalog = list(csv.DictReader(fixture[0][2].open()))
    derived = next(entry for entry in catalog if entry["id"] == "us_cpi_mom")
    derived.update(input_ids="us_financial_stress", formula="us_financial_stress[p]")
    write_catalog(fixture[0][2], catalog)
    receipt = json.loads(fixture[0][1].read_text())
    receipt["catalog_sha256"] = sha256(fixture[0][2])
    fixture[0][1].write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="derivados"):
        compose(fixture, tmp_path / "history")


def test_source_change_before_publication_aborts_the_whole_output(tmp_path, monkeypatch):
    fixture = sources(tmp_path)
    api = importlib.import_module("mars_titan.data.macro_stress_history")
    original = api.atomic_parquet_batches

    def change_source(path, batches):
        count = original(path, batches)
        with fixture[0][0].open("ab") as stream:
            stream.write(b"changed")
        return count

    monkeypatch.setattr(api, "atomic_parquet_batches", change_source)
    with pytest.raises(ValueError, match="cambiado"):
        compose(fixture, tmp_path / "history")
    assert not (tmp_path / "history").exists()


def test_missing_v4_at_handoff_never_reuses_a_v3_value(tmp_path):
    fixture = sources(tmp_path)

    def mutate(rows):
        selected = next(
            r
            for r in rows
            if r["indicator_id"] == "us_financial_stress"
            and r["prediction_at"].date().isoformat() == "2022-11-11"
        )
        selected.update(value=None, missing_reason="expired_vintage")

    rewrite_panel(fixture, 0, mutate)
    compose(fixture, tmp_path / "history")
    rows = pq.read_table(tmp_path / "history/macro.parquet").to_pylist()
    assert [r["value"] for r in rows] == [3.0, 3.0, None, 4.0]
    assert rows[2]["missing_reason"] == "expired_vintage"
    assert rows[2]["source_hashes"] == [HASH_V4]


@pytest.mark.parametrize(
    "defect",
    ["unknown_indicator", "off_calendar", "unsorted", "hash_budget", "period", "reason"],
)
def test_selected_version_requires_bounded_structural_and_temporal_metadata(tmp_path, defect):
    fixture = sources(tmp_path)

    def mutate(rows):
        if defect == "unknown_indicator":
            rows[0]["indicator_id"] = "us_financial_stress"
        elif defect == "off_calendar":
            rows[0]["prediction_at"] -= timedelta(minutes=1)
        elif defect == "unsorted":
            rows.reverse()
        elif defect == "hash_budget":
            rows[0]["source_hashes"] = [HASH_V3] * 257
        elif defect == "period":
            rows[0]["period_start"] = "invalid"
        else:
            rows[0].update(value=None, missing_reason=None)

    rewrite_panel(fixture, 3, mutate)
    with pytest.raises(ValueError):
        compose(fixture, tmp_path / "history")
    assert not (tmp_path / "history").exists()


@pytest.mark.parametrize("defect", ["rows", "date", "large", "symlink", "short_file"])
def test_receipts_and_artifacts_require_valid_bounded_regular_files(tmp_path, defect):
    fixture = sources(tmp_path)
    path = fixture[0][4]
    report = json.loads(path.read_text())
    if defect == "rows":
        report["rows"] += 1
    elif defect == "date":
        report["start"] = "invalid"
    elif defect == "large":
        report["extra"] = "a" * 1024**2
    elif defect == "short_file":
        fixture[0][3].write_bytes(b"PAR1")
        report["sha256"] = sha256(fixture[0][3])
    if defect == "symlink":
        destination = path.with_suffix(".original")
        path.rename(destination)
        path.symlink_to(destination)
    else:
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        compose(fixture, tmp_path / "history")
    assert not (tmp_path / "history").exists()


def test_different_numeric_precision_is_not_silently_coerced(tmp_path):
    fixture = sources(tmp_path)
    panel, receipt = fixture[0][3:5]
    table = pq.read_table(panel)
    table = table.set_column(
        table.schema.get_field_index("value"), "value", table["value"].cast(pa.float32())
    )
    pq.write_table(table, panel)
    report = json.loads(receipt.read_text())
    report["sha256"] = sha256(panel)
    receipt.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="precisión"):
        compose(fixture, tmp_path / "history")


def test_multiple_output_batches_preserve_the_last_pre2024_decision(tmp_path):
    fixture = sources(tmp_path, "2021-12-01", "2023-12-29")
    report = compose(fixture, tmp_path / "history")
    rows = pq.read_table(tmp_path / "history/macro.parquet").to_pylist()
    assert len(rows) == report["rows"] > 512
    assert rows[512]["value"] == 4.0
    assert rows[-1]["prediction_at"].date().isoformat() == "2023-12-29"
    assert rows[-1]["source_hashes"] == [HASH_V4]


def test_composed_identity_is_admitted_only_as_panel_and_never_as_direct_alfred_series(tmp_path):
    fixture = sources(tmp_path)
    compose(fixture, tmp_path / "history")
    entries = list(csv.DictReader((tmp_path / "history/catalog.csv").open()))
    entry = next(r for r in entries if r["id"] == "us_financial_stress")
    assert _exclusion(entry) is None
    assert acquisition_exclusion(entry) is not None
    assert _exclusion(entry | {"series_id": "STLFSI4"}) is not None
    assert _exclusion(entry | {"availability_rule": "STLFSI_RELEASE"}) is not None
    with pytest.raises(ValueError, match="composici|compuest"):
        calculate_macro(
            [{"indicator_id": "us_financial_stress"}],
            [entry],
            MarketClock("US", "2022-11-09", "2022-11-14"),
        )
