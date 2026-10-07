"""Historia contable con publicaciones independientes y prefijos temporales estables."""

import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data import china_preparation, chinese_samples
from mars_titan.data.cohort_contexts import FactCursor
from mars_titan.data.storage import atomic_json, sha256
from tests.data.test_chinese_samples import inputs as pipeline_inputs


def edition(source, destination, clock, *, filed="2022-03-10", accession="annual-2021"):
    """Reproducir otro documento revisado sobre el mismo saldo original de 2021."""
    destination.mkdir()
    table = pq.read_table(source / "fundamentals.parquet")
    row = next(row for row in table.to_pylist() if row["period_end"] == "2021-12-31")
    row.update(
        filed=filed,
        accession=accession,
        available_at=clock.date_available(filed),
        document_sha256="d" * 64,
        publication_evidence_sha256="e" * 64,
        source_url=f"https://static.cninfo.com.cn/finalpage/{filed}/{accession}.PDF",
    )
    pq.write_table(
        pa.Table.from_pylist([row], schema=table.schema), destination / "fundamentals.parquet"
    )
    report = json.loads((source / "report.json").read_text())
    report["identity"].update(document_sha256="d" * 64, publication_sha256="e" * 64)
    report.update(
        facts=1,
        matched_records=1,
        unique_source_records=1,
        duplicate_matches=0,
        sha256=sha256(destination / "fundamentals.parquet"),
    )
    atomic_json(destination / "report.json", report)
    return destination


@pytest.fixture
def history(tmp_path):
    pipeline = pipeline_inputs.__wrapped__(tmp_path)
    parent = json.loads(pipeline["parent_preparation"].read_text())
    prepared = Path(parent["prepared_root"]) / "CN/000001.SZ/manifest.json"
    earlier = edition(pipeline["facts_edition"], tmp_path / "earlier", pipeline["clock"])
    return pipeline, dict(
        prepared_manifest=prepared,
        facts_edition=pipeline["facts_edition"],
        additional_facts=(earlier,),
        output=tmp_path / "history",
        clock=pipeline["clock"],
    )


def read(path):
    return json.loads(path.read_text())


def alter_facts(path, change):
    table = pq.read_table(path / "fundamentals.parquet")
    rows = table.to_pylist()
    change(rows)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path / "fundamentals.parquet")
    report = read(path / "report.json")
    report["sha256"] = sha256(path / "fundamentals.parquet")
    atomic_json(path / "report.json", report)


def test_joint_history_preserves_publication_dates_and_shared_original_ordinals(history):
    _, options = history
    before = {
        p: p.read_bytes()
        for root in [options["facts_edition"], *options["additional_facts"]]
        for p in [root / "report.json", root / "fundamentals.parquet"]
    }
    result = china_preparation.derive_chinese_preparation(**options)
    rows = pq.read_table(options["output"] / "fundamentals.parquet").to_pylist()
    assert result["counts"]["fundamentals"] == 3
    assert result["fundamentals_audit"]["matched_records"] == 4
    assert result["fundamentals_audit"]["unique_source_records"] == 3
    assert result["fundamentals_audit"]["duplicate_matches"] == 1
    assert [row["filed"] for row in rows] == ["2022-03-10", "2023-03-09", "2023-03-09"]
    assert [row["source_records"] for row in rows] == [[3], [3], [1, 2]]
    assert [row["value_exact"] for row in rows] == ["90000000", "90000000", "100000000"]
    assert rows[0]["available_at"] == options["clock"].decision("2022-03-11")
    assert all(row["available_at"] == options["clock"].decision("2023-03-10") for row in rows[1:])
    parents = result["parents"]
    assert parents["facts"]["sha256"] == sha256(options["facts_edition"] / "report.json")
    assert parents["additional_facts"][0]["sha256"] == sha256(
        options["additional_facts"][0] / "report.json"
    )
    assert result["training_ready"] is False
    assert all(p.read_bytes() == content for p, content in before.items())
    cursor = FactCursor(rows)
    concept = "cn-reported:Assets:CNY"
    assert cursor.at(options["clock"].decision("2022-03-10")) == {}
    assert cursor.at(options["clock"].decision("2022-03-11"))[concept]["value"] == 90_000_000
    assert cursor.at(options["clock"].decision("2023-03-09"))[concept]["value"] == 90_000_000
    assert cursor.at(options["clock"].decision("2023-03-10"))[concept]["value"] == 100_000_000
    assert china_preparation.derive_chinese_preparation(**options) == {**result, "reused": True}


def test_changing_edition_order_keeps_the_same_sorted_facts(history, tmp_path):
    _, options = history
    first = china_preparation.derive_chinese_preparation(**options)
    second = china_preparation.derive_chinese_preparation(
        **{
            **options,
            "facts_edition": options["additional_facts"][0],
            "additional_facts": (options["facts_edition"],),
            "output": tmp_path / "reverse",
        }
    )
    assert first["artifacts"]["fundamentals.parquet"] == second["artifacts"]["fundamentals.parquet"]


@pytest.mark.parametrize("conflict", [False, True])
def test_duplicate_or_conflicting_publication_is_rejected(history, tmp_path, conflict):
    _, options = history
    duplicate = tmp_path / "duplicate"
    shutil.copytree(options["additional_facts"][0], duplicate)
    if conflict:
        alter_facts(
            duplicate, lambda rows: rows[0].update(value=91_000_000, value_exact="91000000")
        )
    with pytest.raises(ValueError, match="duplic|contradic"):
        china_preparation.derive_chinese_preparation(
            **{**options, "additional_facts": (*options["additional_facts"], duplicate)}
        )
    assert not options["output"].exists()


@pytest.mark.parametrize("field,value", [("symbol", "600000.SS"), ("cutoff", "2024-01-01")])
def test_each_publication_is_validated_against_the_same_asset_and_cutoff(
    history, monkeypatch, field, value
):
    pipeline, options = history
    report = options["additional_facts"][0] / "report.json"
    content = read(report)
    content["identity"][field] = value
    atomic_json(report, content)
    monkeypatch.setattr(
        chinese_samples, "FrozenEncoders", lambda: pytest.fail("Se cargó un modelo")
    )
    with pytest.raises(ValueError):
        chinese_samples.prepare_chinese_samples(
            **{**pipeline, "encoders": None, "additional_facts": options["additional_facts"]}
        )
    assert not pipeline["output"].exists()


def test_edition_and_total_row_budgets_are_checked_before_decoding(history, monkeypatch):
    _, options = history
    with pytest.raises(ValueError, match="presupuesto|ediciones"):
        china_preparation.derive_chinese_preparation(
            **{**options, "additional_facts": (options["additional_facts"][0],) * 16}
        )
    monkeypatch.setattr(china_preparation, "_MAX_FACTS", 2)
    monkeypatch.setattr(
        china_preparation, "_facts", lambda *a, **kw: pytest.fail("Se leyeron hechos")
    )
    with pytest.raises(ValueError, match="presupuesto|filas"):
        china_preparation.derive_chinese_preparation(**options)


def test_same_ordinal_in_another_original_file_is_a_distinct_source_record(history):
    _, options = history
    earlier = options["additional_facts"][0]
    alter_facts(earlier, lambda rows: rows[0].update(source_file="other-balance.jsonl"))
    origin = read(options["prepared_manifest"])
    origin["sources"]["other-balance.jsonl"] = origin["sources"]["balance.jsonl"]
    atomic_json(options["prepared_manifest"], origin)
    result = china_preparation.derive_chinese_preparation(**options)
    assert result["fundamentals_audit"]["unique_source_records"] == 4


@pytest.mark.parametrize("kind", ["source", "code"])
@pytest.mark.parametrize("phase", ["write", "reuse"])
def test_history_rejects_changes_during_publication_or_reuse(history, monkeypatch, kind, phase):
    _, options = history
    api = china_preparation
    if phase == "reuse":
        api.derive_chinese_preparation(**options)
    changed = False
    source = options["additional_facts"][0] / "report.json"
    digest = api.sha256

    def alter():
        nonlocal changed
        changed = True
        if kind == "source":
            source.write_bytes(source.read_bytes() + b"\n")

    if phase == "write":
        original = api.atomic_parquet

        def trigger(path, table):
            original(path, table)
            alter()

        monkeypatch.setattr(api, "atomic_parquet", trigger)
    else:
        original = api.read_manifest

        def trigger(path, **kwargs):
            value = original(path, **kwargs)
            if path == options["output"] / "manifest.json":
                alter()
            return value

        monkeypatch.setattr(api, "read_manifest", trigger)
    if kind == "code":
        monkeypatch.setattr(
            api,
            "sha256",
            lambda p: "f" * 64 if changed and p.name == "china_preparation.py" else digest(p),
        )
    with pytest.raises(ValueError, match="cambió"):
        api.derive_chinese_preparation(**options)
    if phase == "write":
        assert not options["output"].exists()


def test_later_publication_does_not_change_earlier_multimodal_samples(history, tmp_path):
    pipeline, options = history
    early = pipeline["clock"].decision("2022-06-02")
    origin_path = options["prepared_manifest"]
    origin = read(origin_path)
    news_path = origin_path.parent / "news/news.parquet"
    news = pq.read_table(news_path).to_pylist()
    pq.write_table(
        pa.Table.from_pylist(
            [{**news[0], "available_at": early, "event_id": "earlier-news"}, *news]
        ),
        news_path,
    )
    origin["counts"]["news"] = 2
    origin["artifacts"]["news/news.parquet"] = sha256(news_path)
    atomic_json(origin_path, origin)
    parent = read(pipeline["parent_preparation"])
    parent["assets"][0]["manifest_sha256"] = sha256(origin_path)
    atomic_json(pipeline["parent_preparation"], parent)
    macro_path = pipeline["macro_path"]
    macro = pq.read_table(macro_path).to_pylist()
    initial = [{**row, "prediction_at": early, "available_at": early} for row in macro[:140]]
    pq.write_table(pa.Table.from_pylist(initial + macro), macro_path)
    decisions_path = pipeline["admission_path"].parent / "complete-decisions.parquet"
    decisions = pq.read_table(decisions_path).to_pylist()
    pq.write_table(
        pa.Table.from_pylist([dict(prediction_at=early, indicator_count=140), *decisions]),
        decisions_path,
    )
    admission = read(pipeline["admission_path"])
    admission.update(
        source_sha256=sha256(macro_path),
        complete_decisions=4,
        complete_decisions_sha256=sha256(decisions_path),
    )
    atomic_json(pipeline["admission_path"], admission)
    baseline = tmp_path / "earlier-only"
    chinese_samples.prepare_chinese_samples(
        **{**pipeline, "facts_edition": options["additional_facts"][0], "output": baseline}
    )
    result = chinese_samples.prepare_chinese_samples(
        **pipeline, additional_facts=options["additional_facts"]
    )
    relative = "encoded/samples/CN/000001.SZ/samples.parquet"
    old = pq.read_table(baseline / relative).to_pylist()
    joined = pq.read_table(pipeline["output"] / relative).to_pylist()
    assert old[0] == joined[0]
    np.testing.assert_allclose(
        joined[0]["fundamentals"],
        [math.log1p(90_000_000), 0, 0, 1, 0, 0, math.log1p(83), 0, 0],
        rtol=1e-7,
    )
    assert joined[1]["fundamentals"][0] == pytest.approx(math.log1p(100_000_000), rel=1e-7)
    assert result["scope"] == "development_snapshot" and result["samples"] == 4
    assert (
        chinese_samples.prepare_chinese_samples(
            **pipeline, additional_facts=options["additional_facts"]
        )
        == result
    )


def test_driver_rechecks_every_publication_after_supervision(history, monkeypatch):
    pipeline, options = history
    original = chinese_samples.prepare_corpus_targets

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        path = options["additional_facts"][0] / "report.json"
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(chinese_samples, "prepare_corpus_targets", changed)
    with pytest.raises(ValueError, match="cambió"):
        chinese_samples.prepare_chinese_samples(
            **pipeline, additional_facts=options["additional_facts"]
        )
    assert not (pipeline["output"] / "report.json").exists()


def test_preparation_cli_accepts_repeated_additional_editions(history, tmp_path):
    _, options = history
    another = edition(
        options["facts_edition"],
        tmp_path / "another",
        options["clock"],
        filed="2022-04-11",
        accession="another-2021",
    )
    china_preparation.main(
        [
            "--prepared-manifest",
            str(options["prepared_manifest"]),
            "--facts-edition",
            str(options["facts_edition"]),
            "--output",
            str(options["output"]),
            "--additional-facts",
            str(options["additional_facts"][0]),
            "--additional-facts",
            str(another),
            "--calendar-start",
            "2022-01-01",
            "--calendar-end",
            "2023-12-31",
        ]
    )
    assert read(options["output"] / "manifest.json")["counts"]["fundamentals"] == 4


def test_driver_cli_forwards_every_additional_edition(history, monkeypatch):
    pipeline, options = history
    called = {}

    def prepare(**kwargs):
        called.update(kwargs)
        return dict(
            symbol="000001.SZ",
            samples=1,
            counts={},
            scope="development_snapshot",
            training_ready=False,
        )

    monkeypatch.setattr(chinese_samples, "prepare_chinese_samples", prepare)
    monkeypatch.setitem(
        __import__("sys").modules, "torch", SimpleNamespace(set_num_threads=lambda _: None)
    )
    arguments = []
    for name in (
        "parent_preparation",
        "facts_edition",
        "output",
        "macro_path",
        "admission_path",
        "catalog_path",
        "market_factors",
    ):
        arguments += ["--" + name.replace("_", "-"), str(pipeline[name])]
    additions = [options["additional_facts"][0], options["additional_facts"][0].parent / "another"]
    for path in additions:
        arguments += ["--additional-facts", str(path)]
    chinese_samples.main(arguments)
    assert called["additional_facts"] == additions
