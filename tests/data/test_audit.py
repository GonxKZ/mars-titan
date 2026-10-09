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


ROUNDED_SOURCE = (
    "Date,Open,High,Low,Close,Volume\n"
    "2024-01-02,3.976320878595475,3.9916150569915767,3.845233163135132,3.991615056991577,5\n"
    "2024-01-03,7.44871007398853,7.6707175603792175,7.594163204377157,7.64,5\n"
    "2024-01-04,10,12,9,11,5\n"
)


def _rounded_source(tmp_path):
    source = tmp_path / "dataset"
    folder = source / "time_series/S&P500_time_series"
    folder.mkdir(parents=True)
    (folder / "a.csv").write_text(ROUNDED_SOURCE)
    database = tmp_path / "inventory.sqlite"
    inventory(source, database)
    return source, database


def test_rounding_tolerance_is_a_separate_policy_with_traceable_roundings(tmp_path):
    from mars_titan.data.audited_prices import audit_catalog, read_audited_prices
    from mars_titan.data.temporal import MarketClock

    source, database = _rounded_source(tmp_path)
    strict = module().audit_prices(
        source, database, tmp_path / "strict.json", details_root=tmp_path / "strict"
    )
    state, output = tmp_path / "rounded.json", tmp_path / "rounded"
    rounded = module().audit_prices(
        source, database, state, details_root=output, ordering_rtol=1e-9
    )
    assert "ordering_rtol" not in strict
    assert "ordering_rounded_rows" not in strict["markets"]["US"]
    assert strict["markets"]["US"]["accepted"] == 1
    assert rounded["ordering_rtol"] == 1e-9
    assert rounded["markets"]["US"]["accepted"] == 2
    assert rounded["markets"]["US"]["invalid_ohlc"] == 1
    assert rounded["markets"]["US"]["ordering_rounded_rows"] == 1
    assert rounded["artifacts"] == strict["artifacts"] + 1 == 5
    assert rounded["policy"] != strict["policy"]
    partition = output / "US/A"
    trace = pq.read_table(partition / "ordering_roundings.parquet").to_pylist()
    assert [(row["source_row"], row["source_high"]) for row in trace] == [(1, 3.9916150569915767)]
    assert pq.read_table(partition / "exclusions.parquet")["source_row"].to_pylist() == [2]
    assert not (tmp_path / "strict/US/A/ordering_roundings.parquet").exists()
    # La reanudación no puede mezclar tolerancias en el mismo estado y destino.
    with pytest.raises(ValueError, match="política"):
        module().audit_prices(source, database, state, details_root=output, ordering_rtol=1e-8)
    with pytest.raises(ValueError, match="tolerancia"):
        module().audit_prices(source, database, tmp_path / "x.json", ordering_rtol=1e-3)
    # El lector auditado posterior comprueba el orden estricto y acepta la envolvente.
    records, _ = audit_catalog(state)
    record = records["US", "A"]
    clock = MarketClock("US", "2023-01-01", "2025-01-01")
    frame, receipt, reserved = read_audited_prices(
        record, record["source_sha256"], clock, "2023-12-31"
    )
    assert frame.empty and reserved == 2
    frame, receipt, reserved = read_audited_prices(
        record, record["source_sha256"], clock, "2024-12-31"
    )
    assert frame["session"].tolist() == ["2024-01-02", "2024-01-04"]
    assert frame.loc[0, "high"] == frame.loc[0, "close"] == 3.991615056991577
    assert receipt["accepted"] == 2 and reserved == 0


def test_price_cli_passes_the_declared_rounding_tolerance(tmp_path, monkeypatch):
    import sys

    from mars_titan.data.cli import main

    source, database = _rounded_source(tmp_path)
    report = tmp_path / "report.json"
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
            str(tmp_path / "state.json"),
            "--report",
            str(report),
            "--ordering-rtol",
            "1e-9",
        ],
    )
    main()
    result = json.loads(report.read_text())
    assert result["ordering_rtol"] == 1e-9
    assert result["markets"]["US"]["ordering_rounded_rows"] == 1
