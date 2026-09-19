import importlib

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
    with pytest.raises(ValueError, match="changed"):
        module().audit_prices(root, db, tmp_path / "state.json")
    path.write_text("Date,Open,High,Low,Close,Volume\n2024-01-02,10,12,9,20,1\n")
    inventory(root, db)
    refreshed = module().audit_prices(root, db, tmp_path / "state.json")
    assert refreshed["markets"]["US"]["accepted"] == 0
