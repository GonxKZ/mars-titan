"""Preparación derivada de un activo chino, con fuentes y ediciones independientes."""

import hashlib
import importlib
import json
import os
import stat
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.china_fundamentals import materialize_chinese_facts
from mars_titan.data.cohort_preparation import FACT_SCHEMA
from mars_titan.data.cohort_samples import _prepared
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.data.temporal import MarketClock
from tests.data.test_china_fundamentals import inputs as reviewed_inputs


def module():
    try:
        return importlib.import_module("mars_titan.data.china_preparation")
    except ModuleNotFoundError:
        pytest.fail("Falta el puente de preparación contable CN")


@pytest.fixture
def inputs(tmp_path):
    reviewed = reviewed_inputs.__wrapped__(tmp_path)
    materialize_chinese_facts(**reviewed)
    prepared = tmp_path / "parent" / "CN" / "000001.SZ"
    (prepared / "news").mkdir(parents=True)
    clock = reviewed["clock"]
    stamp = clock.decision("2023-03-10")
    pq.write_table(
        pa.table({"session": ["2023-03-10"], "available_at": [stamp]}),
        prepared / "prices.parquet",
    )
    pq.write_table(pa.Table.from_pylist([], schema=FACT_SCHEMA), prepared / "fundamentals.parquet")
    pq.write_table(
        pa.table({"text": ["银行公布年度报告"], "available_at": [stamp]}),
        prepared / "news/news.parquet",
    )
    pq.write_table(
        pa.table({"reason": pa.array([], pa.string())}), prepared / "news/excluded.parquet"
    )
    atomic_json(prepared / "news/manifest.json", {"counts": {"accepted": 1}})
    artifacts = [
        "prices.parquet",
        "fundamentals.parquet",
        "news/news.parquet",
        "news/excluded.parquet",
        "news/manifest.json",
    ]
    origin = dict(
        schema_version=3,
        market="CN",
        symbol="000001.SZ",
        cohort_id="original_audited",
        news_content_policy="source_audited_not_external",
        fingerprint="a" * 64,
        policy=dict(
            calendar=hashlib.sha256(
                "|".join(t.isoformat() for t in clock.decisions).encode()
            ).hexdigest(),
            cutoff="2023-12-31",
            code={"fundamentals.py": "b" * 64},
        ),
        sources={"balance.jsonl": sha256(reviewed["source"] / "balance.jsonl")},
        artifacts={name: sha256(prepared / name) for name in artifacts},
        counts=dict(prices=1, news=1, fundamentals=0),
        reserved_counts=dict(prices=0, fundamentals=0),
        price_audit={"accepted": 1},
        fundamentals_audit={"accepted": 0, "missing_publication": 3},
        news_counts={"accepted": 1},
        news_reasons={},
        chart_policy="regenerate_from_past_prices",
        training_ready=False,
    )
    atomic_json(prepared / "manifest.json", origin)
    return dict(
        prepared_manifest=prepared / "manifest.json",
        facts_edition=reviewed["output"],
        output=tmp_path / "derived",
        clock=clock,
    )


def change_json(path, edit):
    value = json.loads(path.read_text())
    edit(value)
    atomic_json(path, value)


def change_facts(inputs, edit):
    path = inputs["facts_edition"] / "fundamentals.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    edit(rows)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    change_json(inputs["facts_edition"] / "report.json", lambda r: r.update(sha256=sha256(path)))


def test_derived_asset_preserves_bytes_lineage_and_consumer_contract(inputs):
    parent = inputs["prepared_manifest"]
    before = parent.read_bytes()
    report = module().derive_chinese_preparation(**inputs)
    result, _, _ = _prepared(inputs["output"], inputs["clock"], "original_audited")
    assert report["reused"] is False
    assert result["counts"] == dict(prices=1, news=1, fundamentals=2)
    assert result["training_ready"] is False
    assert result["final_test_opened"] is False
    assert "cohort_complete" not in result
    assert result["fingerprint"] != "a" * 64
    assert result["parents"]["prepared"]["sha256"] == sha256(parent)
    assert result["parents"]["facts"]["sha256"] == sha256(inputs["facts_edition"] / "report.json")
    assert result["fundamentals_audit"]["accepted"] == 2
    assert result["policy"]["code"] != json.loads(before)["policy"]["code"]
    assert parent.read_bytes() == before
    for name in result["artifacts"]:
        source = inputs["facts_edition"] if name == "fundamentals.parquet" else parent.parent
        output = inputs["output"] / name
        assert output.read_bytes() == (source / name).read_bytes()
        assert not output.is_symlink()
        assert output.stat().st_ino != (source / name).stat().st_ino
    facts = pq.read_table(inputs["output"] / "fundamentals.parquet").to_pylist()
    assert [r["value_exact"] for r in facts] == ["90000000", "100000000"]
    assert {r["available_at"] for r in facts} == {datetime(2023, 3, 10, 7, 5, tzinfo=UTC)}


def test_reuse_requires_the_same_manifest_and_artifacts(inputs):
    first = module().derive_chinese_preparation(**inputs)
    stamp = (inputs["output"] / "fundamentals.parquet").stat().st_mtime_ns
    second = module().derive_chinese_preparation(**inputs)
    assert second == {**first, "reused": True}
    assert (inputs["output"] / "fundamentals.parquet").stat().st_mtime_ns == stamp
    change_json(inputs["output"] / "manifest.json", lambda r: r["counts"].update(fundamentals=3))
    with pytest.raises(ValueError):
        module().derive_chinese_preparation(**inputs)


@pytest.mark.parametrize(
    "field,value",
    [("market", "US"), ("symbol", "600000.SS"), ("source_sha256", "c" * 64)],
)
def test_reconciled_edition_cannot_be_attached_to_another_origin(inputs, field, value):
    change_json(
        inputs["facts_edition"] / "report.json", lambda r: r["identity"].update({field: value})
    )
    with pytest.raises(ValueError):
        module().derive_chinese_preparation(**inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("unit", "USD"),
        ("concept", "us-gaap:Assets:USD"),
        ("accounting_standard", "IFRS"),
        ("statement_scope", "separate"),
        ("filed", "2023-03-08"),
        ("period_end", "2023-03-11"),
        ("document_sha256", "c" * 64),
        ("publication_evidence_sha256", "c" * 64),
        ("source_file", "other.json"),
        ("value_exact", "100"),
        ("published_at", datetime(2023, 3, 9, tzinfo=UTC)),
        ("source_records", [0]),
    ],
)
def test_facts_require_the_reviewed_context_and_exact_availability(inputs, field, value):
    change_facts(inputs, lambda rows: rows[0].update({field: value}))
    with pytest.raises(ValueError):
        module().derive_chinese_preparation(**inputs)
    assert not inputs["output"].exists()


def test_duplicate_facts_are_not_silently_appended(inputs):
    change_facts(inputs, lambda rows: rows.append(dict(rows[0])))
    change_json(inputs["facts_edition"] / "report.json", lambda r: r.update(facts=3))
    with pytest.raises(ValueError):
        module().derive_chinese_preparation(**inputs)


def test_future_or_earlier_availability_is_rejected(inputs):
    change_facts(
        inputs,
        lambda rows: rows[0].update(available_at=rows[0]["available_at"] - timedelta(days=1)),
    )
    with pytest.raises(ValueError):
        module().derive_chinese_preparation(**inputs)
    with pytest.raises(ValueError):
        module().derive_chinese_preparation(**inputs, cutoff="2024-01-01")


def test_another_calendar_cannot_relabel_prepared_prices(inputs):
    inputs["clock"] = MarketClock("CN", "2023-01-01", "2024-01-31")
    with pytest.raises(ValueError):
        module().derive_chinese_preparation(**inputs)


def test_changed_original_artifact_is_rejected(inputs):
    (inputs["prepared_manifest"].parent / "news/news.parquet").write_bytes(b"corrupto")
    with pytest.raises(ValueError):
        module().derive_chinese_preparation(**inputs)


def test_output_cannot_replace_or_live_inside_either_parent(inputs):
    for parent in [inputs["prepared_manifest"].parent, inputs["facts_edition"]]:
        with pytest.raises(ValueError):
            module().derive_chinese_preparation(**{**inputs, "output": parent / "derived"})


def test_failed_copy_leaves_no_partial_edition_and_can_be_retried(inputs, monkeypatch):
    api = module()
    original = api.shutil.copyfile

    def interrupted(source, destination):
        original(source, destination)
        raise OSError("Corte de prueba")

    with monkeypatch.context() as patch:
        patch.setattr(api.shutil, "copyfile", interrupted)
        with pytest.raises(OSError):
            api.derive_chinese_preparation(**inputs)
    assert not inputs["output"].exists()
    assert api.derive_chinese_preparation(**inputs)["counts"]["fundamentals"] == 2


@pytest.mark.parametrize("defect", ["rows", "bytes"])
def test_parquet_budget_is_checked_before_decoding(inputs, monkeypatch, defect):
    api = module()
    if defect == "rows":
        change_facts(inputs, lambda rows: rows.append(dict(rows[0])))
    else:
        change_facts(inputs, lambda rows: rows[0].update(source_url="x" * 1_000_000))
        monkeypatch.setattr(api, "_MAX_BYTES", 64 * 1024)
    reader = api.pq.ParquetFile
    facts = inputs["facts_edition"] / "fundamentals.parquet"

    class GuardedReader:
        def __init__(self, path):
            self.inner = reader(path)
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.inner.close()

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def read(self, **kwargs):
            if self.path == facts:
                pytest.fail("Se descomprimió un Parquet fuera del presupuesto")
            return self.inner.read(**kwargs)

    monkeypatch.setattr(api.pq, "ParquetFile", GuardedReader)
    with pytest.raises(ValueError, match="presupuesto|población"):
        api.derive_chinese_preparation(**inputs)
    assert not inputs["output"].exists()


def test_excluded_news_with_excessive_rows_are_rejected_even_when_compressed(inputs):
    parent = inputs["prepared_manifest"].parent
    relative = "news/excluded.parquet"
    path = parent / relative
    pq.write_table(pa.table({"reason": pa.repeat(pa.scalar("missing_identity"), 200_001)}), path)
    change_json(
        inputs["prepared_manifest"], lambda r: r["artifacts"].update({relative: sha256(path)})
    )
    assert path.stat().st_size < 1024**2
    with pytest.raises(ValueError, match="presupuesto"):
        module().derive_chinese_preparation(**inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize("phase", ["copy", "reuse"])
def test_source_changes_during_work_cannot_confirm_an_edition(inputs, monkeypatch, phase):
    api = module()
    path = inputs["prepared_manifest"].parent / "news/news.parquet"
    if phase == "copy":
        original = api.shutil.copyfile

        def changed(source, destination):
            result = original(source, destination)
            if source == path:
                path.write_bytes(path.read_bytes() + b"cambio")
            return result

        monkeypatch.setattr(api.shutil, "copyfile", changed)
    else:
        api.derive_chinese_preparation(**inputs)
        original = api.read_manifest

        def changed(source, **kwargs):
            result = original(source, **kwargs)
            if source == inputs["output"] / "manifest.json":
                path.write_bytes(path.read_bytes() + b"cambio")
            return result

        monkeypatch.setattr(api, "read_manifest", changed)
    with pytest.raises(ValueError, match="cambió|cambiado"):
        api.derive_chinese_preparation(**inputs)
    if phase == "copy":
        assert not inputs["output"].exists()


def test_code_identity_is_fixed_before_reading_the_facts(inputs, monkeypatch):
    api = module()
    original, digest = api._facts, api.sha256
    changed = False

    def read(*args):
        nonlocal changed
        result = original(*args)
        changed = True
        return result

    monkeypatch.setattr(api, "_facts", read)
    monkeypatch.setattr(
        api,
        "sha256",
        lambda path: "f" * 64 if changed and path.name == "china_preparation.py" else digest(path),
    )
    with pytest.raises(ValueError, match="código cambió"):
        api.derive_chinese_preparation(**inputs)
    assert not inputs["output"].exists()


def test_altered_staged_manifest_is_not_published(inputs, monkeypatch):
    api = module()
    original = api.atomic_json

    def changed(path, report):
        original(path, {**report, "training_ready": True})

    monkeypatch.setattr(api, "atomic_json", changed)
    with pytest.raises(ValueError):
        api.derive_chinese_preparation(**inputs)
    assert not inputs["output"].exists()


def test_staged_directory_entries_are_synced_before_publication(inputs, monkeypatch):
    api = module()
    sync, publish = api.os.fsync, api._publish_directory
    synced = set()

    def observed_sync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            synced.add(os.readlink(f"/proc/self/fd/{fd}"))
        return sync(fd)

    def observed_publish(source, destination):
        assert str(source / "news") in synced
        assert str(source) in synced
        return publish(source, destination)

    monkeypatch.setattr(api.os, "fsync", observed_sync)
    monkeypatch.setattr(api, "_publish_directory", observed_publish)
    assert api.derive_chinese_preparation(**inputs)["counts"]["fundamentals"] == 2


def test_cli_creates_the_same_asset_contract(inputs, capsys):
    module().main(
        [
            "--prepared-manifest",
            str(inputs["prepared_manifest"]),
            "--facts-edition",
            str(inputs["facts_edition"]),
            "--output",
            str(inputs["output"]),
            "--calendar-start",
            "2023-03-01",
            "--calendar-end",
            "2024-01-31",
        ]
    )
    report, _, _ = _prepared(inputs["output"], inputs["clock"], "original_audited")
    assert report["counts"]["fundamentals"] == 2
    assert "000001.SZ" in capsys.readouterr().out
