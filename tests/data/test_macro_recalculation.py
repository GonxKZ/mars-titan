"""Ediciones de cálculo nuevas, con fuentes y reserva temporal inmutables."""

import csv
import sqlite3
from contextlib import closing
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256
from tests.data.test_macro_acquisition import _alfred_zip


@pytest.fixture
def source(tmp_path):
    from mars_titan.data import macro_acquisition as acquisition

    folder = tmp_path / "source"
    acquisition._initialize(folder, "fixture")
    payload = _alfred_zip("DGS10", ["2023-01-05"], ["2023-01-04,3,2023-01-05,9999-12-31"])
    acquisition._store_batch(
        folder,
        {"id": "us_treasury_10y"},
        "fixture",
        payload,
        {"2023-01-05"},
        {},
        acquisition._parse_zip(payload, "DGS10", {"2023-01-05"}),
        "2023-01-01",
        "2023-01-10",
    )
    with acquisition._connect(folder) as db:
        db.execute(
            "INSERT INTO series VALUES (?,?,?,?,?)",
            ("us_treasury_10y", "DGS10", "complete", None, "fixture"),
        )
    catalog = tmp_path / "catalog.csv"
    with Path("data/catalogs/macro-indicators.csv").open() as stream:
        entry = next(r for r in csv.DictReader(stream) if r["id"] == "us_treasury_10y")
    with catalog.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entry), lineterminator="\n")
        writer.writeheader()
        writer.writerow(entry)
    return folder, catalog


def test_new_edition_records_policy_and_exact_sources_without_overwriting(source, tmp_path):
    from mars_titan.data.macro_recalculation import recalculate_macro

    acquisition, catalog = source
    before = sha256(acquisition / "macro.sqlite3"), sha256(catalog)
    output = tmp_path / "edition"
    report = recalculate_macro(
        acquisition,
        catalog,
        output,
        market="US",
        start="2023-01-05",
        end="2023-01-10",
        daily_lag_policy="valid_observations",
    )
    assert report["status"] == "completed"
    assert report["calculation"]["daily_lag_policy"] == "valid_observations"
    assert report["source_database_sha256"] == before[0]
    assert report["catalog_sha256"] == before[1]
    assert report["sha256"] == sha256(output / "macro.parquet")
    assert report["final_test_opened"] is False
    rows = pq.read_table(output / "macro.parquet").to_pylist()
    assert rows[0]["value"] is None
    assert rows[-1]["value"] == 3
    assert (sha256(acquisition / "macro.sqlite3"), sha256(catalog)) == before
    with pytest.raises(FileExistsError):
        recalculate_macro(
            acquisition, catalog, output, market="US", start="2023-01-05", end="2023-01-10"
        )


def test_future_or_unknown_indicators_are_rejected_before_publication(source, tmp_path):
    from mars_titan.data.macro_recalculation import recalculate_macro

    acquisition, catalog = source
    output = tmp_path / "edition"
    for options in (
        {"end": "2024-01-01"},
        {"indicators": ["unknown"]},
        {"indicators": []},
        {"daily_lag_policy": "guess"},
    ):
        arguments = dict(market="US", start="2023-01-05", end="2023-01-10") | options
        with pytest.raises(ValueError):
            recalculate_macro(acquisition, catalog, output, **arguments)
        assert not output.exists()


@pytest.mark.parametrize("start,end", [("2023-01-06", "2023-01-10"), ("2023-01-03", "2023-01-04")])
def test_schema_is_stable_with_complete_or_entirely_missing_values(source, tmp_path, start, end):
    from mars_titan.data.macro_coverage import _check_schema
    from mars_titan.data.macro_recalculation import recalculate_macro

    acquisition, catalog = source
    output = tmp_path / "edition"
    recalculate_macro(acquisition, catalog, output, market="US", start=start, end=end)
    _check_schema(pq.ParquetFile(output / "macro.parquet").schema_arrow)


def test_output_cannot_contain_or_replace_the_sources(source, tmp_path):
    from mars_titan.data.macro_recalculation import recalculate_macro

    acquisition, catalog = source
    with pytest.raises(ValueError):
        recalculate_macro(
            acquisition,
            catalog,
            acquisition / "new",
            market="US",
            start="2023-01-05",
            end="2023-01-10",
        )


def test_source_hash_stays_stable_when_finished_connections_are_collected(
    source, tmp_path, monkeypatch
):
    import gc

    from mars_titan.data import macro_recalculation as module

    acquisition, catalog = source
    original = module.atomic_json

    def collect_after_write(*args):
        result = original(*args)
        gc.collect()
        return result

    monkeypatch.setattr(module, "atomic_json", collect_after_write)
    report = module.recalculate_macro(
        acquisition,
        catalog,
        tmp_path / "edition",
        market="US",
        start="2023-01-05",
        end="2023-01-10",
    )
    assert report["source_database_sha256"] == sha256(acquisition / "macro.sqlite3")


@pytest.mark.parametrize("during_read", [False, True])
def test_unconsolidated_wal_cannot_publish_an_unidentified_source(
    source, tmp_path, monkeypatch, during_read
):
    from mars_titan.data import macro_recalculation as module

    acquisition, catalog = source
    database = acquisition / "macro.sqlite3"
    before = sha256(database)
    with closing(sqlite3.connect(database)) as writer:
        writer.execute("PRAGMA wal_autocheckpoint=0")

        def update():
            writer.execute("UPDATE vintages SET value=999")
            writer.commit()
            assert sha256(database) == before
            assert (acquisition / "macro.sqlite3-wal").stat().st_size > 0

        if during_read:
            original = module.iter_vintages

            def write_then_read(*args, **kwargs):
                update()
                yield from original(*args, **kwargs)

            monkeypatch.setattr(module, "iter_vintages", write_then_read)
        else:
            update()
        output = tmp_path / "unconfirmed"
        with pytest.raises(ValueError, match="consolidada"):
            module.recalculate_macro(
                acquisition,
                catalog,
                output,
                market="US",
                start="2023-01-05",
                end="2023-01-10",
            )
        assert not output.exists()
    if during_read:
        monkeypatch.setattr(module, "iter_vintages", original)
    confirmed = module.recalculate_macro(
        acquisition,
        catalog,
        tmp_path / "confirmed",
        market="US",
        start="2023-01-05",
        end="2023-01-10",
    )
    assert confirmed["source_database_sha256"] == sha256(database) != before
    assert pq.read_table(tmp_path / "confirmed/macro.parquet")["value"][-1].as_py() == 999
