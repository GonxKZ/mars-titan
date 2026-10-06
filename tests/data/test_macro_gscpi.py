"""Versiones mensuales de GSCPI sin adelantar valores revisados."""

import csv
import importlib
import json
from pathlib import Path

import pytest


def module():
    try:
        return importlib.import_module("mars_titan.data.macro_gscpi")
    except ModuleNotFoundError:
        pytest.fail("Falta el lector de versiones mensuales de GSCPI")


def content():
    return (
        b"Date,Apr-22,May-22,Jun-22,Jan-24\n"
        b"31-Mar-2022,2.7,2.80,3.0,999\n"
        b"30-Apr-2022,#N/A,3.29,3.45,999\n"
        b"31-May-2022,#N/A,#N/A,3.01,999\n"
    )


def catalog():
    with Path("data/catalogs/macro-indicators.csv").open() as stream:
        return [r for r in csv.DictReader(stream) if r["id"].startswith("global_supply_pressure")]


def test_parser_keeps_month_bound_and_ignores_future_vintage_values():
    records = module().parse_gscpi_vintages(content(), last_vintage="2022-06")
    assert len(records) == 5
    assert {r["realtime_start"] for r in records} == {"2022-05-31", "2022-06-30"}
    assert all(r["availability_precision"] == "month" for r in records)
    assert all(r["publication_timestamp_verified"] is False for r in records)
    assert all(r["value"] != 999 for r in records)
    altered = content().replace(b",999", b",changed_future_value")
    other = module().parse_gscpi_vintages(altered, last_vintage="2022-06")
    for before, after in zip(records, other, strict=True):
        assert {k: v for k, v in before.items() if k != "source_hash"} == {
            k: v for k, v in after.items() if k != "source_hash"
        }


def test_monthly_versions_and_changes_enter_only_after_month_end():
    from mars_titan.data.macro import calculate_macro
    from mars_titan.data.temporal import MarketClock

    rows = module().parse_gscpi_vintages(content(), last_vintage="2022-06")
    result = calculate_macro(rows, catalog(), MarketClock("US", "2022-05-18", "2022-07-05"))
    by_key = {(r["prediction_at"].date().isoformat(), r["indicator_id"]): r for r in result}
    assert by_key["2022-05-31", "global_supply_pressure"]["value"] is None
    assert by_key["2022-06-01", "global_supply_pressure"]["value"] == 3.29
    assert by_key["2022-06-01", "global_supply_pressure_change_1m"]["value"] == pytest.approx(0.49)
    assert by_key["2022-06-30", "global_supply_pressure"]["value"] == 3.29
    assert by_key["2022-07-01", "global_supply_pressure"]["value"] == 3.01
    assert by_key["2022-07-01", "global_supply_pressure_change_1m"]["value"] == pytest.approx(-0.44)


@pytest.mark.parametrize(
    "data",
    [
        b"Date,May-22,May-22\n30-Apr-2022,1,2\n",
        b"Date,May-22\n31-May-2022,1\n",
        b"Date,May-22\n30-Apr-2022,inf\n",
        b"Date,May-22\n30-Apr-2022,1\n30-Apr-2022,1\n",
        b"Date,May-22\n31-Apr-2022,1\n",
        b"Date,May-22\n01-Apr-2022,1\n",
        b"Date,May-22\n30-Apr-2022,1,2\n",
    ],
)
def test_malformed_dates_values_and_vintages_are_rejected(data):
    with pytest.raises(ValueError):
        module().parse_gscpi_vintages(data, last_vintage="2022-05")


def test_old_versions_cannot_bypass_the_month_end_policy():
    from mars_titan.data.macro import calculate_macro
    from mars_titan.data.temporal import MarketClock

    rows = module().parse_gscpi_vintages(content(), last_vintage="2022-06")
    rows[0]["realtime_start"] = "2022-05-18"
    with pytest.raises(ValueError):
        calculate_macro(rows, catalog(), MarketClock("US", "2022-05-18", "2022-07-05"))


def test_mislabeled_vintage_cannot_use_an_earlier_month_bound():
    from mars_titan.data.macro import calculate_macro
    from mars_titan.data.temporal import MarketClock

    rows = module().parse_gscpi_vintages(content(), last_vintage="2022-06")
    rows[0]["vintage_label"] = "Jun-22"
    with pytest.raises(ValueError):
        calculate_macro(rows, catalog(), MarketClock("US", "2022-05-18", "2022-07-05"))


def test_empty_trailing_csv_row_is_not_a_missing_observation():
    records = module().parse_gscpi_vintages(content() + b",,,,\n", last_vintage="2022-06")
    assert len(records) == 5


def test_archive_missing_its_first_regular_version_is_not_complete():
    with pytest.raises(ValueError, match="archivo|versión|mes"):
        module().parse_gscpi_vintages(b"Date,Jun-22\n30-Apr-2022,3.45\n", last_vintage="2022-06")


def test_preparation_writes_immutable_source_and_auditable_panel(tmp_path, monkeypatch):
    from mars_titan.data.storage import sha256

    monkeypatch.setattr(module(), "_download", lambda: content(), raising=False)
    output = tmp_path / "prepared"
    result = module().prepare_gscpi(
        output,
        Path("data/catalogs/macro-indicators.csv"),
        market="US",
        start="2022-05-18",
        end="2022-07-05",
    )
    assert result["source_sha256"] == sha256(output / "source.csv")
    assert result["sha256"] == sha256(output / "macro.parquet")
    assert result["publication_timestamp_verified"] is False
    assert result["nonempty_indicators"] == [
        "global_supply_pressure",
        "global_supply_pressure_change_1m",
    ]
    assert json.loads((output / "report.json").read_text()) == result
    before = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        module().prepare_gscpi(
            output,
            Path("data/catalogs/macro-indicators.csv"),
            market="US",
            start="2022-05-18",
            end="2022-07-05",
        )
    assert (output / "report.json").read_bytes() == before


def test_failed_source_never_publishes_a_completed_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(module(), "_download", lambda: b"invalid source", raising=False)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError):
        module().prepare_gscpi(
            output,
            Path("data/catalogs/macro-indicators.csv"),
            market="US",
            start="2022-05-18",
            end="2022-07-05",
        )
    assert not output.exists()


def test_cli_prepares_the_real_pipeline_with_declared_month_precision(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(module(), "_download", lambda: content())
    assert (
        module().main(
            [
                "--catalog",
                "data/catalogs/macro-indicators.csv",
                "--output",
                str(tmp_path / "cli"),
                "--market",
                "US",
                "--start",
                "2022-05-18",
                "--end",
                "2022-07-05",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["computed_values"] > 0
    assert report["publication_timestamp_verified"] is False


@pytest.fixture
def source_edition(tmp_path, monkeypatch):
    monkeypatch.setattr(module(), "_download", lambda: content())
    source = tmp_path / "original-edition"
    module().prepare_gscpi(
        source,
        Path("data/catalogs/macro-indicators.csv"),
        market="US",
        start="2022-05-18",
        end="2022-07-05",
    )

    def forbidden_download():
        raise AssertionError("La fuente fijada no debe sustituirse por una descarga")

    monkeypatch.setattr(module(), "_download", forbidden_download)
    return source


def reuse(source, output):
    return module().prepare_gscpi(
        output,
        Path("data/catalogs/macro-indicators.csv"),
        market="CN",
        start="2022-05-18",
        end="2022-07-05",
        source_edition=source,
    )


def test_reused_source_preserves_receipt_and_recalculates_chinese_availability(
    source_edition,
    tmp_path,
):
    import pyarrow.parquet as pq

    from mars_titan.data.storage import sha256

    original = json.loads((source_edition / "report.json").read_text())
    output = tmp_path / "chinese"
    report = reuse(source_edition, output)
    assert report["retrieved_at_utc"] == original["retrieved_at_utc"]
    assert report["source_sha256"] == original["source_sha256"]
    assert report["source_edition_sha256"] == sha256(source_edition / "report.json")
    assert (source_edition / "source.csv").read_bytes() == (output / "source.csv").read_bytes()
    rows = pq.read_table(output / "macro.parquet").to_pylist()
    values = {(r["prediction_at"].date().isoformat(), r["indicator_id"]): r["value"] for r in rows}
    assert values["2022-06-01", "global_supply_pressure"] is None
    assert values["2022-06-02", "global_supply_pressure"] == 3.29
    assert values["2022-07-01", "global_supply_pressure"] == 3.29
    assert values["2022-07-04", "global_supply_pressure"] == 3.01
    assert values["2022-07-04", "global_supply_pressure_change_1m"] == pytest.approx(-0.44)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha256", "a" * 64),
        ("source_url", "https://example.test/otra.csv"),
        ("retrieved_at_utc", "2022-01-01T00:00:00"),
        ("schema_version", 99),
        ("schema_version", True),
    ],
)
def test_invalid_cached_receipt_is_rejected(source_edition, tmp_path, field, value):
    path = source_edition / "report.json"
    report = json.loads(path.read_text())
    report[field] = value
    path.write_text(json.dumps(report))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        reuse(source_edition, output)
    assert not output.exists()


def test_changed_cached_bytes_do_not_inherit_the_original_identity(source_edition, tmp_path):
    with (source_edition / "source.csv").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError):
        reuse(source_edition, tmp_path / "rejected")


@pytest.mark.parametrize("filename", ["source.csv", "report.json"])
def test_source_changes_during_materialization_prevent_publication(
    source_edition,
    tmp_path,
    monkeypatch,
    filename,
):
    original = module().atomic_parquet

    def changed(path, table):
        original(path, table)
        with (source_edition / filename).open("ab") as stream:
            stream.write(b"\n")

    monkeypatch.setattr(module(), "atomic_parquet", changed)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        reuse(source_edition, output)
    assert not output.exists()


def test_output_cannot_be_nested_in_cached_edition(source_edition):
    with pytest.raises(ValueError):
        reuse(source_edition, source_edition / "derived")


def test_cli_reuses_an_explicit_source_edition(source_edition, tmp_path, capsys):
    assert (
        module().main(
            [
                "--catalog",
                "data/catalogs/macro-indicators.csv",
                "--output",
                str(tmp_path / "cli-cn"),
                "--source-edition",
                str(source_edition),
                "--market",
                "CN",
                "--start",
                "2022-05-18",
                "--end",
                "2022-07-05",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["market"] == "CN" and "source_edition_sha256" in report
