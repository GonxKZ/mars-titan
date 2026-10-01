import importlib
import json

import pyarrow.parquet as pq
import pytest

from mars_titan.data.inventory import inventory


def module():
    try:
        return importlib.import_module("mars_titan.data.audit")
    except ModuleNotFoundError:
        pytest.fail("La auditoría reproducible del universo todavía no existe")


def test_price_audit_reconciles_files_and_rejects_modified_source(tmp_path):
    root = tmp_path / "dataset"
    price_dir = root / "time_series" / "S&P500_time_series"
    price_dir.mkdir(parents=True)
    path = price_dir / "a.csv"
    path.write_text("Date,Open,High,Low,Close,Volume\n2024-01-02,10,12,9,11,1\n")
    db = tmp_path / "inventory.sqlite"
    inventory(root, db)
    result = module().audit_prices(root, db, tmp_path / "state.json")
    assert result["markets"]["US"]["files"] == 1
    assert result["markets"]["US"]["rows"] == 1
    assert result["markets"]["US"]["accepted"] == 1
    path.write_text("Date,Open,High,Low,Close,Volume\n2024-01-02,10,12,9,12,1\n")
    with pytest.raises(ValueError, match="cambiado"):
        module().audit_prices(root, db, tmp_path / "state.json")
    path.write_text("Date,Open,High,Low,Close,Volume\n2024-01-02,10,12,9,20,1\n")
    inventory(root, db)
    refreshed = module().audit_prices(root, db, tmp_path / "state.json")
    assert refreshed["markets"]["US"]["accepted"] == 0


def test_price_detail_export_preserves_sources_and_checks_artifacts_on_resume(tmp_path):
    from mars_titan.data.storage import sha256

    source = tmp_path / "dataset"
    folder = source / "time_series/S&P500_time_series"
    folder.mkdir(parents=True)
    path = folder / "a.csv"
    path.write_text(
        "Date,Open,High,Low,Close,Volume\n2024-01-02,10,12,9,11,1\n2024-01-03,10,9,8,11,1\n"
    )
    original = sha256(path)
    database = tmp_path / "inventory.sqlite"
    inventory(source, database)
    state = tmp_path / "state.json"
    output = tmp_path / "prices"
    result = module().audit_prices(source, database, state, details_root=output)
    assert result["markets"]["US"]["accepted"] == 1
    assert result["derived_bytes"] > 0
    assert result["artifacts"] == 4
    partition = output / "US/A"
    assert pq.read_table(partition / "prices.parquet").num_rows == 1
    assert pq.read_table(partition / "exclusions.parquet")["source_row"].to_pylist() == [2]
    assert pq.read_table(partition / "corporate_actions.parquet").num_rows == 0
    assert pq.read_table(partition / "coverage.parquet").num_rows == 1
    saved = json.loads(state.read_text())
    assert saved["files"]["time_series/S&P500_time_series/a.csv"]["asset_id"] == "finmultitime:US:A"
    assert sha256(path) == original
    again = module().audit_prices(source, database, state, details_root=output)
    assert again["reused_files"] == 1
    (partition / "exclusions.parquet").write_bytes(b"corrupto")
    with pytest.raises(ValueError, match="derivado"):
        module().audit_prices(source, database, state, details_root=output)


def test_price_audit_does_not_claim_an_unowned_output_directory(tmp_path):
    source = tmp_path / "dataset"
    source.mkdir()
    output = tmp_path / "prices"
    output.mkdir()
    (output / "original.txt").write_text("conservar")
    with pytest.raises(ValueError, match="nuevo"):
        module().audit_prices(
            source, tmp_path / "inventory.sqlite", tmp_path / "state.json", details_root=output
        )
    assert (output / "original.txt").read_text() == "conservar"


def test_price_detail_resume_refuses_changed_snapshot_and_output_aliases(tmp_path):
    source = tmp_path / "dataset"
    folder = source / "time_series/S&P500_time_series"
    folder.mkdir(parents=True)
    path = folder / "a.csv"
    path.write_text("Date,Open,High,Low,Close,Volume\n2024-01-02,10,12,9,11,1\n")
    database = tmp_path / "inventory.sqlite"
    inventory(source, database)
    state = tmp_path / "state.json"
    output = tmp_path / "prices"
    module().audit_prices(source, database, state, details_root=output)
    derived = (output / "US/A/prices.parquet").read_bytes()
    path.write_text("Date,Open,High,Low,Close,Volume\n2024-01-02,10,12,9,12,1\n")
    inventory(source, database)
    with pytest.raises(ValueError, match="instantánea"):
        module().audit_prices(source, database, state, details_root=output)
    assert (output / "US/A/prices.parquet").read_bytes() == derived
    with pytest.raises(ValueError, match="inventario"):
        module().audit_prices(source, database, database)
    with pytest.raises(ValueError, match="origen"):
        module().audit_prices(source, database, tmp_path / "new.json", details_root=source / "out")


@pytest.mark.parametrize("target", ["state", "database", "partition"])
def test_price_cli_rejects_report_overwriting_inputs_or_partitions(tmp_path, monkeypatch, target):
    import sys

    from mars_titan.data.cli import main

    source = tmp_path / "dataset"
    source.mkdir()
    state, database, output = (
        tmp_path / "state.json",
        tmp_path / "inventory.sqlite",
        tmp_path / "out",
    )
    report = {"state": state, "database": database, "partition": output / "US/A/prices.parquet"}[
        target
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "audit-prices",
            "--source",
            str(source),
            "--database",
            str(database),
            "--state",
            str(state),
            "--details",
            str(output),
            "--report",
            str(report),
        ],
    )
    with pytest.raises(ValueError):
        main()
    assert not output.exists()
    assert not state.exists()
