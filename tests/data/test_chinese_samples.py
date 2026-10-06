"""Unión controlada de hechos chinos, cuatro modalidades y macro completo."""

import csv
import hashlib
import importlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.data.temporal import MarketClock
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.data.test_china_preparation import inputs as preparation_fixture
from tests.data.test_cohort_samples import Encoders


def api():
    try:
        return importlib.import_module("mars_titan.data.chinese_samples")
    except ModuleNotFoundError:
        pytest.fail("Falta la codificación supervisada del activo chino revisado")


@pytest.fixture
def inputs(tmp_path):
    prepared = preparation_fixture.__wrapped__(tmp_path)
    parent = prepared["prepared_manifest"].parent
    clock = MarketClock("CN", "2022-01-01", "2023-12-31")
    moments = [clock.decision(day) for day in ("2023-03-10", "2023-03-13", "2023-03-14")]
    days = [day for day in clock.days if "2022-01-01" <= day.isoformat() <= "2023-03-31"]
    rows = [
        dict(
            session=day.isoformat(),
            open=100.0,
            close=100.0 + 0.1 * (i % 7),
            high=101.0,
            low=99.0,
            volume=100.0,
            available_at=clock.decision(day),
        )
        for i, day in enumerate(days)
    ]
    pq.write_table(pa.Table.from_pylist(rows), parent / "prices.parquet")
    pq.write_table(
        pa.Table.from_pylist(
            [
                dict(
                    available_at=moments[0],
                    content_hash="c" * 64,
                    event_id="cn-news",
                    content_kind="summary",
                    text="银行公布年度报告",
                    cohort_id="original_audited",
                    availability_rule="source_timestamp",
                )
            ]
        ),
        parent / "news/news.parquet",
    )
    origin = json.loads(prepared["prepared_manifest"].read_text())
    origin["counts"]["prices"] = len(rows)
    origin["policy"]["calendar"] = hashlib.sha256(
        "|".join(t.isoformat() for t in clock.decisions).encode()
    ).hexdigest()
    origin["price_audit"] = dict(accepted=len(rows))
    origin["artifacts"] = {name: sha256(parent / name) for name in origin["artifacts"]}
    atomic_json(prepared["prepared_manifest"], origin)
    preparation = tmp_path / "parent/manifest.json"
    atomic_json(
        preparation,
        dict(
            schema_version=1,
            kind="prepared_cohort",
            status="completed",
            cohort_id="original_audited",
            candidate_count=2,
            failed_assets=0,
            prepared_root=str(parent.parents[1]),
            configuration=dict(markets=["CN"]),
            assets=[
                dict(
                    market="CN",
                    symbol="000001.SZ",
                    state="prepared",
                    manifest_sha256=sha256(prepared["prepared_manifest"]),
                ),
                dict(market="CN", symbol="600000.SS", state="missing_modalities", missing=["news"]),
            ],
        ),
    )
    catalog = Path("data/catalogs/macro-indicators.csv").resolve()
    identifiers = sorted(row["id"] for row in csv.DictReader(catalog.open()))
    macro = tmp_path / "macro.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                dict(prediction_at=t, available_at=t, value=1.0, indicator_id=name)
                for t in moments
                for name in identifiers
            ]
        ),
        macro,
    )
    admission = tmp_path / "admission"
    admission.mkdir()
    selected = admission / "complete-decisions.parquet"
    pq.write_table(
        pa.table(dict(prediction_at=moments, indicator_count=[140] * len(moments))), selected
    )
    atomic_json(
        admission / "report.json",
        dict(
            schema_version=1,
            policy="all_catalog_indicators_valid",
            market="CN",
            required_indicator_count=140,
            required_indicator_ids=identifiers,
            source_sha256=sha256(macro),
            catalog_sha256=sha256(catalog),
            complete_decisions=len(moments),
            complete_decisions_path=selected.name,
            complete_decisions_sha256=sha256(selected),
        ),
    )
    factor = tmp_path / "factor.parquet"
    pq.write_table(pa.Table.from_pylist(rows), factor)
    factors = tmp_path / "market-factors.json"
    atomic_json(
        factors,
        {
            "CN": dict(
                market="CN", symbol="000300", prices_path=str(factor), prices_sha256=sha256(factor)
            )
        },
    )
    return dict(
        parent_preparation=preparation,
        facts_edition=prepared["facts_edition"],
        output=tmp_path / "encoded-cn",
        macro_path=macro,
        admission_path=admission / "report.json",
        catalog_path=catalog,
        market_factors=factors,
        clock=clock,
        encoders=Encoders(),
    )


def test_chinese_subset_reaches_existing_reader_without_claiming_full_corpus(inputs):
    result = api().prepare_chinese_samples(**inputs)
    assert result["status"] == "completed"
    assert result["samples"] == 3
    assert result["training_ready"] is False
    output = inputs["output"]
    encoded = json.loads((output / "encoded/manifest.json").read_text())
    supervised = json.loads((output / "supervised/manifest.json").read_text())
    assert encoded["scope"] == supervised["scope"] == "development_snapshot"
    assert encoded["cohort_complete"] is False
    assert encoded["parent_preparation"]["candidate_count"] == 2
    assert supervised["counts"] == dict(train=0, validation=3)
    receipt = json.loads((output / "encoded/samples/CN/000001.SZ/manifest.json").read_text())
    assert receipt["source_unit"] == "CNY"
    assert receipt["fundamental_concepts"] == list(api().CNY_CONCEPTS)
    table = pq.read_table(output / "encoded/samples/CN/000001.SZ/samples.parquet")
    assert table.schema.field("fundamentals").type.list_size == 9
    for row in table.to_pylist():
        assert all(t <= row["prediction_at"] for t in row["input_availability"].values())
        assert row["fundamentals"][3:6] == [1, 0, 0]
        assert row["macro"][140:280] == [1] * 140
    dataset = CorpusDataset(output / "supervised/manifest.json")
    batches = list(dataset.batches(partition="validation", batch_size=2, epoch=0, seed=42))
    assert sum(len(batch["target"]) for batch in batches) == 3
    before = inputs["encoders"].calls
    supervision_path = output / "supervised/manifest.json"
    before_manifest = (supervision_path.read_bytes(), supervision_path.stat().st_mtime_ns)
    again = api().prepare_chinese_samples(**inputs)
    assert again["samples"] == result["samples"]
    assert inputs["encoders"].calls == before
    assert (supervision_path.read_bytes(), supervision_path.stat().st_mtime_ns) == before_manifest


def test_incomplete_macro_fails_before_creating_encoders_or_output(inputs, monkeypatch):
    path = inputs["admission_path"]
    meta = json.loads(path.read_text())
    meta["required_indicator_count"] = 139
    atomic_json(path, meta)
    inputs["encoders"] = None
    monkeypatch.setattr(
        api(), "FrozenEncoders", lambda: pytest.fail("Se cargó un modelo antes de validar")
    )
    with pytest.raises(ValueError, match="140"):
        api().prepare_chinese_samples(**inputs)
    assert not inputs["output"].exists()


def test_admission_from_another_market_is_rejected(inputs):
    path = inputs["admission_path"]
    meta = json.loads(path.read_text())
    meta["market"] = "US"
    atomic_json(path, meta)
    with pytest.raises(ValueError, match="admisión"):
        api().prepare_chinese_samples(**inputs)


def test_parent_population_must_reconcile_before_selecting_an_asset(inputs):
    path = inputs["parent_preparation"]
    meta = json.loads(path.read_text())
    meta["candidate_count"] += 1
    atomic_json(path, meta)
    with pytest.raises(ValueError, match="padre|activo"):
        api().prepare_chinese_samples(**inputs)


def test_same_output_cannot_switch_its_source_population(inputs):
    api().prepare_chinese_samples(**inputs)
    path = inputs["parent_preparation"]
    meta = json.loads(path.read_text())
    meta["assets"].append(
        dict(market="CN", symbol="600001.SS", state="missing_modalities", missing=["news"])
    )
    meta["candidate_count"] += 1
    atomic_json(path, meta)
    with pytest.raises(ValueError, match="edición|configuración|identidad"):
        api().prepare_chinese_samples(**inputs)


def test_unlisted_reviewed_issuer_does_not_enter_the_selection(inputs):
    path = inputs["parent_preparation"]
    meta = json.loads(path.read_text())
    meta["assets"][0]["symbol"] = "000002.SZ"
    atomic_json(path, meta)
    with pytest.raises(ValueError, match="activo|emisor|candidato"):
        api().prepare_chinese_samples(**inputs)


def test_reserved_macro_and_missing_real_values_are_rejected(inputs):
    table = pq.read_table(inputs["macro_path"])
    rows = table.to_pylist()
    rows[0]["value"] = None
    pq.write_table(pa.Table.from_pylist(rows), inputs["macro_path"])
    path = inputs["admission_path"]
    meta = json.loads(path.read_text())
    meta["source_sha256"] = sha256(inputs["macro_path"])
    atomic_json(path, meta)
    with pytest.raises(ValueError, match="140"):
        api().prepare_chinese_samples(**inputs)


def test_interrupted_encoding_reuses_completed_preparation_and_recovers(inputs):
    inputs["encoders"] = Encoders(fail=True)
    with pytest.raises(RuntimeError, match="Interrupción"):
        api().prepare_chinese_samples(**inputs)
    assert not (inputs["output"] / "report.json").exists()
    inputs["encoders"] = Encoders()
    assert api().prepare_chinese_samples(**inputs)["samples"] == 3


@pytest.mark.parametrize(
    "relative", ["configuration.json", "encoded/manifest.json", "supervised/manifest.json"]
)
def test_changed_stage_receipt_cannot_be_confirmed_as_completed(inputs, monkeypatch, relative):
    module = api()
    original = module.prepare_corpus_targets

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        path = inputs["output"] / relative
        meta = json.loads(path.read_text())
        meta["context_sessions"] = 63
        atomic_json(path, meta)
        return result

    monkeypatch.setattr(module, "prepare_corpus_targets", changed)
    with pytest.raises(ValueError, match="cambió|identidad|recibo"):
        module.prepare_chinese_samples(**inputs)
    assert not (inputs["output"] / "report.json").exists()


@pytest.mark.parametrize("relative", ["configuration.json", "preparation.json"])
def test_recovery_uses_the_hash_of_the_bytes_actually_validated(inputs, monkeypatch, relative):
    module = api()
    inputs["encoders"] = Encoders(fail=True)
    with pytest.raises(RuntimeError, match="Interrupción"):
        module.prepare_chinese_samples(**inputs)
    assert not (inputs["output"] / "report.json").exists()
    inputs["encoders"] = Encoders()
    original = module.read_manifest
    target = inputs["output"] / relative
    changed = False

    def replace_after_read(path, *args, **kwargs):
        nonlocal changed
        result = original(path, *args, **kwargs)
        if path == target and not changed:
            content = json.loads(path.read_text())
            content["context_sessions"] = 63
            atomic_json(path, content)
            changed = True
        return result

    monkeypatch.setattr(module, "read_manifest", replace_after_read)
    with pytest.raises(ValueError, match="cambió|configuración|identidad"):
        module.prepare_chinese_samples(**inputs)
