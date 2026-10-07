"""Integridad de resultados recuperados y huellas documentales durante la lectura."""

import json
from datetime import timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256
from tests.data import test_macro_h15_contexts as fixtures
from tests.data.test_macro_h15_contexts import CATALOG, load, metadata, module, produce


@pytest.fixture
def source(tmp_path):
    return fixtures.source.__wrapped__(tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "value",
        "prediction_at",
        "available_at",
        "period_start",
        "indicator_id",
        "source_hashes",
        "unit",
        "missing_reason",
        "seasonal_adjustment",
        "row_order",
        "value_type",
    ],
)
def test_recovery_recalculates_values_even_when_the_report_hash_matches(source, tmp_path, field):
    meta = metadata(tmp_path)
    load(source, metadata_manifest=meta)
    output = tmp_path / "panel"
    produce(source, output, metadata_manifest=meta)
    path = output / "macro-US.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    row = next(row for row in rows if row["value"] is not None)
    if field == "value":
        row[field] += 123.0
    elif field in {"prediction_at", "available_at"}:
        row[field] -= timedelta(days=1)
    elif field == "source_hashes":
        row[field] = ["a" * 64]
    elif field == "row_order":
        rows[0], rows[1] = rows[1], rows[0]
    elif field != "value_type":
        row[field] = "alterado"
    changed = pa.Table.from_pylist(rows, schema=table.schema)
    if field == "value_type":
        changed = changed.set_column(
            changed.schema.get_field_index("value"), "value", changed["value"].cast(pa.float32())
        )
    pq.write_table(changed, path)
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    report["artifacts"][path.name] = sha256(path)
    report_path.write_text(json.dumps(report))
    before = {p: (sha256(p), p.stat().st_mtime_ns) for p in output.iterdir()}
    with pytest.raises(ValueError, match="panel|cálculo|esquema"):
        produce(source, output, metadata_manifest=meta)
    assert before == {p: (sha256(p), p.stat().st_mtime_ns) for p in before}


@pytest.mark.parametrize("update_receipt", [False, True])
def test_document_hashes_stay_bound_to_the_verified_receipt(
    source, tmp_path, monkeypatch, update_receipt
):
    meta = metadata(tmp_path)
    load(source, metadata_manifest=meta)
    original = module().prepare_h15_archive

    def replace_after_check(*args, **kwargs):
        result = original(*args, **kwargs)
        path = source["output"] / "observations.parquet"
        table = pq.read_table(path)
        rows = table.to_pylist()
        for row in rows:
            row["value"], row["value_exact"] = 99.0, "99.0"
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
        if update_receipt:
            receipt = source["output"] / "report.json"
            report = json.loads(receipt.read_text())
            report["artifacts"][path.name] = sha256(path)
            receipt.write_text(json.dumps(report))
        return result

    monkeypatch.setattr(module(), "prepare_h15_archive", replace_after_check)
    with pytest.raises(ValueError, match="huella|documental|recibo"):
        module().load_h15_events(
            source["manifest"], source["output"], CATALOG, metadata_manifest=meta
        )


def test_recovery_checks_dictionary_text_before_expanding_it(source, tmp_path, monkeypatch):
    load(source)
    output = tmp_path / "panel"
    produce(source, output)
    path = output / "macro-US.parquet"
    table = pq.read_table(path)
    table = table.set_column(
        table.schema.get_field_index("unit"), "unit", pa.array(["x" * 5000] * len(table))
    )
    pq.write_table(table, path, use_dictionary=True)
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    report["artifacts"][path.name] = sha256(path)
    report_path.write_text(json.dumps(report))
    original, observed = pq.ParquetFile, []

    class Guarded:
        def __init__(self, name, *args, **kwargs):
            self.target = str(name) == str(path)
            self.dictionaries = kwargs.get("read_dictionary", [])
            self.inner = original(name, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.inner.close()

        def iter_batches(self, *args, **kwargs):
            if self.target:
                observed.append("unit" in self.dictionaries)
            return self.inner.iter_batches(*args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", Guarded)
    with pytest.raises(ValueError, match="panel|campo|presupuesto"):
        produce(source, output)
    assert observed and all(observed)
