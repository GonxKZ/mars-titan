"""Unión de activos revisados sin recodificación ni cambios de población implícitos."""

import csv
import importlib
import json
import shutil
from datetime import timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.chinese_samples import prepare_chinese_samples
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.posttraining.preparation import encoder_contract
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.data.test_chinese_samples import inputs as single_inputs


def api():
    try:
        return importlib.import_module("mars_titan.data.chinese_corpus")
    except ModuleNotFoundError:
        pytest.fail("Falta la unión de las ediciones chinas revisadas")


def read(path):
    return json.loads(path.read_text())


@pytest.fixture
def inputs(tmp_path):
    first = single_inputs.__wrapped__(tmp_path)
    parent = read(first["parent_preparation"])
    root = Path(parent["prepared_root"])
    second = root / "CN/000002.SZ"
    shutil.copytree(root / "CN/000001.SZ", second)
    origin = read(second / "manifest.json")
    origin["symbol"] = "000002.SZ"
    atomic_json(second / "manifest.json", origin)
    facts = tmp_path / "second-facts"
    shutil.copytree(first["facts_edition"], facts)
    report = read(facts / "report.json")
    report["identity"]["symbol"] = "000002.SZ"
    atomic_json(facts / "report.json", report)
    parent["assets"][1] = dict(
        market="CN",
        symbol="000002.SZ",
        state="prepared",
        manifest_sha256=sha256(second / "manifest.json"),
    )
    atomic_json(first["parent_preparation"], parent)
    return [first, {**first, "facts_edition": facts, "output": tmp_path / "second-edition"}]


def build(inputs):
    for arguments in inputs:
        prepare_chinese_samples(**arguments)
    return [arguments["output"] for arguments in inputs]


def reseal(root, relative, change):
    path = root / relative
    value = read(path)
    change(value)
    atomic_json(path, value)
    report = read(root / "report.json")
    if relative == "configuration.json":
        report["configuration_sha256"] = sha256(path)
    else:
        report["artifacts"][relative] = sha256(path)
    atomic_json(root / "report.json", report)


def test_union_preserves_all_input_bytes_and_reaches_the_corpus_reader(inputs, tmp_path):
    editions = build(inputs)
    output = tmp_path / "union"
    before = {
        path: path.read_bytes()
        for edition in editions
        for path in [
            edition / "report.json",
            edition / "encoded/manifest.json",
            edition / "supervised/manifest.json",
        ]
    }
    result = api().combine_chinese_samples(editions, output)
    assert result["status"] == "completed"
    assert result["scope"] == "development_snapshot" and result["cohort_complete"] is False
    assert result["samples"] == 6 and result["candidate_count"] == 2
    assert result["counts"] == {"train": 0, "validation": 6}
    assert result["training_ready"] is False and result["final_test_opened"] is False
    assert len(result["lineage"]) == 2
    for index, edition in enumerate(editions, start=1):
        symbol = f"00000{index}.SZ"
        for root in ["prepared", "encoded/samples"]:
            source = edition / root / "CN" / symbol
            for name in (
                ["manifest.json", "fundamentals.parquet"]
                if root == "prepared"
                else ["manifest.json", "samples.parquet"]
            ):
                copied = output / root / "CN" / symbol / name
                assert copied.read_bytes() == (source / name).read_bytes()
                assert copied.stat().st_ino != (source / name).stat().st_ino
    assert all(path.read_bytes() == content for path, content in before.items())
    encoded = read(output / "encoded/manifest.json")
    assert encoded["candidate_count"] == 2 and len(encoded["coverage"]) == 2
    assert (
        encoded["configuration"]["encoders"] == read(editions[0] / "configuration.json")["encoders"]
    )
    dataset = CorpusDataset(output / "supervised/manifest.json", cache_bytes=0)
    batches = list(dataset.batches(partition="validation", batch_size=2, epoch=0, seed=42))
    assert sum(len(batch["target"]) for batch in batches) == 6
    assert {name.split("/")[1] for batch in batches for name in batch["sample_ids"]} == {
        "000001.SZ",
        "000002.SZ",
    }
    encoder_contract(output / "supervised/manifest.json", output / "encoded/manifest.json")
    path = output / "supervised/manifest.json"
    stamp = path.stat().st_mtime_ns
    assert api().combine_chinese_samples(editions, output) == result
    assert path.stat().st_mtime_ns == stamp


def test_duplicate_symbols_are_rejected_even_from_distinct_completed_editions(inputs, tmp_path):
    first = inputs[0]
    prepare_chinese_samples(**first)
    other = tmp_path / "another-first"
    prepare_chinese_samples(**{**first, "output": other})
    with pytest.raises(ValueError, match="duplic"):
        api().combine_chinese_samples([first["output"], other], tmp_path / "union")


@pytest.mark.parametrize(
    "field,value", [("encoders", {"purpose": "another"}), ("context_sessions", 32)]
)
def test_rehashed_source_cannot_change_representation_semantics(inputs, tmp_path, field, value):
    editions = build(inputs)
    reseal(editions[1], "configuration.json", lambda value_: value_.update({field: value}))
    with pytest.raises(ValueError, match="represent|codificador|contexto|semántica"):
        api().combine_chinese_samples(editions, tmp_path / "union")


def test_different_macro_catalogues_cannot_share_a_union(inputs, tmp_path):
    catalog = tmp_path / "another-catalog.csv"
    with inputs[1]["catalog_path"].open() as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    rows[0]["source_url"] += "#otra-edicion"
    with catalog.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    admission = tmp_path / "another-admission"
    shutil.copytree(inputs[1]["admission_path"].parent, admission)
    report = read(admission / "report.json")
    report["catalog_sha256"] = sha256(catalog)
    atomic_json(admission / "report.json", report)
    inputs[1].update(catalog_path=catalog, admission_path=admission / "report.json")
    editions = build(inputs)
    with pytest.raises(ValueError, match="macro|catálogo"):
        api().combine_chinese_samples(editions, tmp_path / "union")


def new_factor(inputs, tmp_path):
    factors = read(inputs[0]["market_factors"])
    rows = pq.read_table(factors["CN"]["prices_path"]).to_pylist()
    for index, row in enumerate(rows):
        row["close"] = 100.0 + 0.2 * (index % 5)
    path = tmp_path / "new-factor.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    factors["CN"].update(prices_path=str(path), prices_sha256=sha256(path))
    descriptor = tmp_path / "new-factor.json"
    atomic_json(descriptor, factors)
    return descriptor


def test_different_factors_require_an_explicit_common_descriptor(inputs, tmp_path):
    descriptor = new_factor(inputs, tmp_path)
    inputs[1]["market_factors"] = descriptor
    editions = build(inputs)
    with pytest.raises(ValueError, match="factor"):
        api().combine_chinese_samples(editions, tmp_path / "ambiguous")
    output = tmp_path / "common"
    api().combine_chinese_samples(editions, output, market_factors=descriptor)
    supervised = read(output / "supervised/manifest.json")
    assert supervised["market_factors"] == read(descriptor)
    old = pq.read_table(editions[0] / "supervised/labels/CN/000001.SZ/labels.parquet")
    new = pq.read_table(output / "supervised/labels/CN/000001.SZ/labels.parquet")
    assert old["prediction_at"].equals(new["prediction_at"])
    assert not old["target"].equals(new["target"])


def test_factor_from_another_market_is_rejected(inputs, tmp_path):
    editions = build(inputs)
    descriptor = tmp_path / "wrong-factor.json"
    atomic_json(descriptor, {"US": read(inputs[0]["market_factors"])["CN"]})
    with pytest.raises(ValueError, match="factor|CN"):
        api().combine_chinese_samples(editions, tmp_path / "union", market_factors=descriptor)


@pytest.mark.parametrize(
    "relative",
    [
        "prepared/CN/000001.SZ/fundamentals.parquet",
        "encoded/samples/CN/000001.SZ/samples.parquet",
        "supervised/labels/CN/000001.SZ/labels.parquet",
    ],
)
def test_corrupt_source_artifacts_are_not_copied(inputs, tmp_path, relative):
    editions = build(inputs)
    (editions[0] / relative).write_bytes(b"corrupto")
    with pytest.raises(ValueError):
        api().combine_chinese_samples(editions, tmp_path / "union")
    assert not (tmp_path / "union/report.json").exists()


def test_parent_population_and_finished_stage_links_are_required(inputs, tmp_path):
    editions = build(inputs)
    reseal(
        editions[0],
        "preparation.json",
        lambda meta: meta["parent_preparation"].update(candidate_count=892),
    )
    with pytest.raises(ValueError):
        api().combine_chinese_samples(editions, tmp_path / "union")


def test_interrupted_copy_can_resume_without_overwriting_completed_files(
    inputs, tmp_path, monkeypatch
):
    editions = build(inputs)
    module = api()
    output = tmp_path / "union"
    copy = module.shutil.copyfile
    calls = 0

    def interrupted(source, destination):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("Corte de prueba")
        return copy(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(module.shutil, "copyfile", interrupted)
        with pytest.raises(OSError):
            module.combine_chinese_samples(editions, output)
    assert not (output / "report.json").exists()
    copied = {path: path.stat().st_mtime_ns for path in output.rglob("*.parquet")}
    assert copied
    assert module.combine_chinese_samples(editions, output)["samples"] == 6
    assert all(path.stat().st_mtime_ns == stamp for path, stamp in copied.items())


def test_corrupt_completed_output_is_not_silently_rebuilt(inputs, tmp_path):
    editions = build(inputs)
    output = tmp_path / "union"
    api().combine_chinese_samples(editions, output)
    (output / "supervised/manifest.json").write_text("{}")
    with pytest.raises(ValueError):
        api().combine_chinese_samples(editions, output)


@pytest.mark.parametrize("target", ["source", "configuration"])
def test_changes_during_supervision_prevent_final_confirmation(
    inputs, tmp_path, monkeypatch, target
):
    editions = build(inputs)
    module = api()
    output = tmp_path / "union"
    original = module.prepare_corpus_targets

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        path = (
            editions[0] / "encoded/manifest.json"
            if target == "source"
            else output / "configuration.json"
        )
        value = read(path)
        value["context_sessions"] = 63
        atomic_json(path, value)
        return result

    monkeypatch.setattr(module, "prepare_corpus_targets", changed)
    with pytest.raises(ValueError, match="cambió|cambiado"):
        module.combine_chinese_samples(editions, output)
    assert not (output / "report.json").exists()


def test_output_must_be_separate_and_edition_count_is_bounded(inputs, tmp_path, monkeypatch):
    editions = build(inputs)
    with pytest.raises(ValueError):
        api().combine_chinese_samples(editions, editions[0] / "union")
    monkeypatch.setattr(api(), "_MAX_EDITIONS", 1)
    with pytest.raises(ValueError, match="presupuesto|ediciones"):
        api().combine_chinese_samples(editions, tmp_path / "union")


def test_total_sample_budget_is_checked_before_copying(inputs, tmp_path, monkeypatch):
    editions = build(inputs)
    monkeypatch.setattr(api(), "_MAX_SAMPLES", 5)
    with pytest.raises(ValueError, match="presupuesto|muestras"):
        api().combine_chinese_samples(editions, tmp_path / "union")
    assert not (tmp_path / "union").exists()


@pytest.mark.parametrize(
    "field,value", [("schema_version", True), ("final_test_opened", 0), ("samples", 3.0)]
)
def test_source_receipts_do_not_coerce_contract_types(inputs, tmp_path, field, value):
    editions = build(inputs)
    path = editions[0] / "report.json"
    report = read(path)
    report[field] = value
    atomic_json(path, report)
    with pytest.raises(ValueError):
        api().combine_chinese_samples(editions, tmp_path / "union")


def test_final_receipt_cannot_replace_false_with_zero(inputs, tmp_path):
    editions = build(inputs)
    output = tmp_path / "union"
    api().combine_chinese_samples(editions, output)
    path = output / "report.json"
    report = read(path)
    report["final_test_opened"] = 0
    atomic_json(path, report)
    with pytest.raises(ValueError):
        api().combine_chinese_samples(editions, output)


def test_partial_recovery_preserves_configuration_scalar_types(inputs, tmp_path, monkeypatch):
    editions = build(inputs)
    output = tmp_path / "union"
    module = api()

    def fail(*args):
        raise OSError("Interrupción antes de copiar")

    with monkeypatch.context() as patch:
        patch.setattr(module.shutil, "copyfile", fail)
        with pytest.raises(OSError):
            module.combine_chinese_samples(editions, output)
    path = output / "configuration.json"
    config = read(path)
    config["representation"]["encoders"]["version"] = True
    atomic_json(path, config)
    with pytest.raises(ValueError):
        module.combine_chinese_samples(editions, output)


@pytest.mark.parametrize("stage", ["encoded", "supervised"])
def test_stage_configuration_must_equal_its_confirmed_manifest(inputs, tmp_path, stage):
    editions = build(inputs)
    path = editions[0] / stage / "configuration.json"
    config = read(path)
    config["unconfirmed_change"] = True
    atomic_json(path, config)
    with pytest.raises(ValueError):
        api().combine_chinese_samples(editions, tmp_path / "union")


def update_references(root, replacements):
    """Recalcular las referencias de la fixture para probar el contenido, no solo su hash."""
    paths = list(root.rglob("*.json"))

    def replace(value):
        if isinstance(value, str):
            return replacements.get(value, value)
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value

    for _ in range(12):
        changed = False
        for path in paths:
            value = read(path)
            updated = replace(value)
            if value != updated:
                previous = sha256(path)
                atomic_json(path, updated)
                replacements[previous] = sha256(path)
                changed = True
        if not changed:
            return
    pytest.fail("El grafo de recibos de la fixture no converge")


@pytest.mark.parametrize("field", ["configuration_sha256", "preparation.json"])
def test_required_hash_cannot_be_disabled_with_null(inputs, tmp_path, field):
    editions = build(inputs)
    path = editions[0] / "report.json"
    value = read(path)
    if field == "configuration_sha256":
        value[field] = None
    else:
        value["artifacts"][field] = None
    atomic_json(path, value)
    with pytest.raises(ValueError):
        api().combine_chinese_samples(editions, tmp_path / "union")


@pytest.mark.parametrize("change", ["future_availability", "invalid_context", "missing_macro"])
def test_rehashed_invalid_samples_never_publish_a_union(inputs, tmp_path, change):
    editions = build(inputs)
    path = editions[0] / "encoded/samples/CN/000001.SZ/samples.parquet"
    original = sha256(path)
    table = pq.read_table(path)
    rows = table.to_pylist()
    if change == "future_availability":
        future = rows[0]["prediction_at"] + timedelta(days=1)
        rows[0]["fundamentals_available_at"] = future
        rows[0]["input_availability"]["fundamentals"] = future
    elif change == "invalid_context":
        rows[0]["price_end_index"] = 0
    else:
        rows[0]["macro"][140] = 0.0
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    update_references(editions[0], {original: sha256(path)})
    with pytest.raises(ValueError):
        api().combine_chinese_samples(editions, tmp_path / "union")
    assert not (tmp_path / "union/report.json").exists()


def test_completed_labels_corruption_is_rejected_before_recomputation(
    inputs, tmp_path, monkeypatch
):
    editions = build(inputs)
    output = tmp_path / "union"
    module = api()
    module.combine_chinese_samples(editions, output)
    path = output / "supervised/labels/CN/000001.SZ/labels.parquet"
    path.write_bytes(b"contenido alterado")
    monkeypatch.setattr(
        module,
        "prepare_corpus_targets",
        lambda *a, **kw: pytest.fail("Se regeneraron etiquetas confirmadas"),
    )
    with pytest.raises(ValueError):
        module.combine_chinese_samples(editions, output)
    assert path.read_bytes() == b"contenido alterado"


def test_selected_assets_must_actually_belong_to_the_parent_census(inputs, tmp_path):
    editions = build(inputs)
    parent = inputs[0]["parent_preparation"]
    previous = sha256(parent)
    meta = read(parent)
    for index, item in enumerate(meta["assets"]):
        item["symbol"] = f"00000{index + 3}.SZ"
    atomic_json(parent, meta)
    for edition in editions:
        update_references(edition, {previous: sha256(parent)})
    with pytest.raises(ValueError):
        api().combine_chinese_samples(editions, tmp_path / "union")
