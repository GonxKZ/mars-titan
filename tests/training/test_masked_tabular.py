"""Ridge y XGBoost con la política de máscaras, comprobados hasta el ajuste sin ejecutarlo.

Los estimadores se sustituyen por capturas que consumen la factoría y se detienen.
Así se comparan filas, columnas y recibos con el lector neuronal sin aprender nada.
"""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED, MODALITIES, policy_identity
from mars_titan.data.storage import sha256
from mars_titan.models.baselines.external_boosting import external_cache_plan
from mars_titan.training import external_corpus, tabular_corpus, tabular_search
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.partition_contract import supervision_bounds
from tests.training.historical_temporal_fixture import historical_temporal_fixture
from tests.training.test_historical_temporal import prepare
from tests.training.test_reference_run import training_corpus
from tests.training.test_tabular_search import setup

LEGACY_CODE = {
    "training/tabular_corpus.py",
    "training/corpus_inputs.py",
    "training/temporal_corpus.py",
    "evaluation/splits.py",
    "evaluation/split_readiness.py",
    "data/streaming.py",
    "data/batches.py",
    "models/baselines/inputs.py",
    "models/baselines/ridge.py",
    "models/baselines/boosting.py",
}


class Captured(Exception):
    """El estimador falso se detiene tras recorrer la población completa."""


def capture(monkeypatch):
    blocks = []

    def fit(factory, **_):
        blocks.extend((x.copy(), y.copy()) for x, y in factory())
        raise Captured

    monkeypatch.setattr(tabular_corpus, "fit_boosting_batches", fit)
    return blocks


def fitted_inputs(manifest, output, monkeypatch, **options):
    blocks = capture(monkeypatch)
    with pytest.raises(Captured):
        tabular_corpus.run_tabular_reference(manifest, output, kind="boosting", **options)
    report = json.loads((output / "run.json").read_text())
    assert report["status"] == "failed" and report["fitted_rows"] == 0
    return np.concatenate([x for x, _ in blocks]), np.concatenate([y for _, y in blocks]), report


def reader_rows(manifest, partition, *, batch_size, masked=True):
    """Referencia escrita con el lector neuronal, sin pasar por el código tabular."""
    policy = dict(input_policy=HISTORICAL_MASKED) if masked else {}
    dataset = CorpusDataset(manifest, **policy)
    ids, parts, targets, presence = [], [], [], []
    for batch in dataset.batches(partition=partition, batch_size=batch_size, epoch=0, seed=0):
        count = len(batch["target"])
        parts.append(
            np.concatenate(
                [batch["inputs"][name].reshape(count, -1) for name in MODALITIES], axis=1
            ).astype(np.float64)
        )
        targets.append(batch["target"])
        ids.extend(batch["sample_ids"])
        if masked:
            presence.append(batch["presence"])
    return (
        ids,
        np.concatenate(parts),
        np.concatenate(targets),
        np.concatenate(presence) if masked else None,
        dataset,
    )


def widths(dataset):
    batch = next(dataset.batches(partition="train", batch_size=1, epoch=0, seed=0))
    return [batch["inputs"][name][0].size for name in MODALITIES]


@pytest.fixture
def masked(tmp_path):
    return historical_temporal_fixture(tmp_path / "fixture")


def test_masked_blocks_follow_the_neural_reader_and_append_five_bits(masked, tmp_path, monkeypatch):
    x, y, report = fitted_inputs(
        masked.parent, tmp_path / "run", monkeypatch, batch_size=3, input_policy=HISTORICAL_MASKED
    )
    _, values, targets, presence, dataset = reader_rows(masked.parent, "train", batch_size=3)
    assert len(x) == dataset.manifest["counts"]["train"] == 4
    np.testing.assert_array_equal(x[:, :-5], values)
    np.testing.assert_array_equal(x[:, -5:], presence.astype(np.float64))
    np.testing.assert_array_equal(y, targets)
    assert x.dtype == np.float64 and set(np.unique(x[:, -5:])) <= {0.0, 1.0}
    assert x[:, -5].all() and x[:, -3].all() and not x[:, -4].any()
    assert x[:, -1].any() and not x[:, -1].all()
    offsets = np.cumsum([0, *widths(dataset)])
    for index in range(len(MODALITIES)):
        block = x[:, offsets[index] : offsets[index + 1]]
        assert not block[~presence[:, index]].any()
    assert report["input_policy"] == HISTORICAL_MASKED
    assert report["mask_contract"] == policy_identity(HISTORICAL_MASKED)["mask_contract"]
    assert report["feature_order"] == [*MODALITIES, "presence"]
    assert report["features"] == offsets[-1] + 5 == x.shape[1]
    assert set(report["code"]) == LEGACY_CODE | {"data/input_policy.py"}


def test_predictions_keep_the_same_sample_ids_and_bits_as_the_neural_reader(masked, tmp_path):
    class PresenceCount:
        """Función fija de las entradas, sin parámetros ajustados."""

        def predict(self, values):
            return values[:, -5:].sum(axis=1)

    model = PresenceCount()
    for partition in ("train", "validation"):
        ids, _, targets, presence, dataset = reader_rows(masked.parent, partition, batch_size=4)
        path = tmp_path / f"{partition}.parquet"
        metrics = tabular_corpus._predict(model, model, dataset, partition, 4, path, presence=True)
        rows = pq.read_table(path).to_pylist()
        assert [row["sample_id"] for row in rows] == ids
        assert len(set(ids)) == len(ids) == metrics["samples"]
        assert metrics["samples"] == dataset.manifest["counts"][partition]
        np.testing.assert_array_equal([row["target"] for row in rows], targets)
        np.testing.assert_array_equal([row["prediction"] for row in rows], presence.sum(axis=1))


def test_masked_temporal_view_admits_its_four_partitions_without_reading_holdouts(
    masked, tmp_path, monkeypatch
):
    prepare(masked, tmp_path / "views")
    manifest = tmp_path / "views/fold-000/manifest.json"
    meta = json.loads(manifest.read_text())
    bounds = supervision_bounds(meta, input_policy=HISTORICAL_MASKED)
    assert set(bounds) == {"train", "validation", "calibration", "evaluation"}
    with pytest.raises(ValueError):
        supervision_bounds(meta)
    x, _, report = fitted_inputs(
        manifest, tmp_path / "run", monkeypatch, batch_size=2, input_policy=HISTORICAL_MASKED
    )
    assert len(x) == meta["counts"]["train"] and report["samples"] == meta["counts"]
    assert {"training/temporal_corpus.py", "data/input_policy.py"} <= set(report["code"])


def test_strict_reader_rejects_the_masked_edition_before_creating_output(masked, tmp_path):
    output = tmp_path / "strict"
    with pytest.raises(ValueError):
        tabular_corpus.run_tabular_reference(masked.parent, output, kind="boosting")
    with pytest.raises(ValueError, match="configuración"):
        tabular_corpus.run_tabular_reference(
            masked.parent, output, kind="boosting", input_policy="masked"
        )
    assert not output.exists()


def test_strict_matrix_and_receipt_keep_the_previous_layout(tmp_path, monkeypatch):
    manifest = training_corpus(tmp_path / "data")
    x, y, report = fitted_inputs(manifest, tmp_path / "run", monkeypatch, batch_size=5)
    _, values, targets, _, _ = reader_rows(manifest, "train", batch_size=5, masked=False)
    assert x.tobytes() == values.tobytes() and y.tobytes() == targets.tobytes()
    assert report["feature_order"] == list(MODALITIES) and report["features"] == x.shape[1]
    assert not {"input_policy", "mask_contract"} & set(report)
    assert set(report["code"]) == LEGACY_CODE


def test_targets_do_not_enter_the_masked_feature_matrix(masked, tmp_path, monkeypatch):
    first, targets, _ = fitted_inputs(
        masked.parent, tmp_path / "first", monkeypatch, input_policy=HISTORICAL_MASKED
    )
    meta = json.loads(masked.parent.read_text())
    for asset in meta["assets"]:
        path = Path(meta["roots"]["labels"]) / asset["market"] / asset["symbol"] / "labels.parquet"
        table = pq.read_table(path)
        rows = table.to_pylist()
        for row in rows:
            if row["target"] is not None:
                row["target"] += 100.0
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
        asset["labels_sha256"] = sha256(path)
    masked.parent.write_text(json.dumps(meta))
    second, shifted, _ = fitted_inputs(
        masked.parent, tmp_path / "second", monkeypatch, input_policy=HISTORICAL_MASKED
    )
    assert first.tobytes() == second.tobytes()
    np.testing.assert_allclose(shifted, targets + 100.0, rtol=0, atol=1e-12)


def test_matrix_requires_bits_exactly_when_the_policy_declares_them():
    batch = dict(
        target=np.zeros(2),
        inputs={name: np.ones((2, 1), dtype=np.float32) for name in MODALITIES},
    )
    assert tabular_corpus._matrix(batch).shape == (2, 5)
    with pytest.raises(ValueError, match="presencia"):
        tabular_corpus._matrix(batch, presence=True)
    batch["presence"] = np.array([[True, False, True, False, True]] * 2)
    with pytest.raises(ValueError, match="presencia"):
        tabular_corpus._matrix(batch)
    np.testing.assert_array_equal(
        tabular_corpus._matrix(batch, np.float32, presence=True)[:, 5:], [[1, 0, 1, 0, 1]] * 2
    )
    for wrong in (batch["presence"].astype(np.float32), batch["presence"][:, :4]):
        with pytest.raises(ValueError, match="presencia"):
            tabular_corpus._matrix(dict(batch, presence=wrong), presence=True)


class LibrariesReached(Exception):
    """La comprobación previa ha terminado y se iba a cargar CUDA."""


def external(monkeypatch):
    plans = []
    original = external_corpus.external_cache_plan

    def record(**options):
        plans.append(options)
        return original(**options)

    def libraries():
        raise LibrariesReached

    monkeypatch.setattr(external_corpus, "external_cache_plan", record)
    monkeypatch.setattr(external_corpus, "_libraries", libraries)
    return plans


def test_external_masked_run_needs_an_explicit_disk_budget_before_reading(
    masked, tmp_path, monkeypatch
):
    plans = external(monkeypatch)
    output = tmp_path / "xgb"
    with pytest.raises(ValueError, match="presupuesto de disco"):
        external_corpus.run_external_reference(
            masked.parent, output, on_host=False, input_policy=HISTORICAL_MASKED
        )
    with pytest.raises(ValueError, match="presupuesto de disco"):
        external_corpus.run_external_reference(
            masked.parent,
            output,
            on_host=False,
            input_policy=HISTORICAL_MASKED,
            max_disk_cache_bytes=1,
        )
    assert len(plans) == 1 and not output.exists()


def test_external_masked_plan_uses_the_five_bits_and_the_validation_cache(
    masked, tmp_path, monkeypatch, learning_doubles
):
    plans = external(monkeypatch)
    output = tmp_path / "xgb"
    selection = dict(schema_version=1, minimum_rounds=2, patience_rounds=2, min_delta=1e-5)
    with pytest.raises(LibrariesReached):
        external_corpus.run_external_reference(
            masked.parent,
            output,
            rounds=6,
            batch_size=4,
            on_host=False,
            input_policy=HISTORICAL_MASKED,
            max_disk_cache_bytes=1024**3,
            selection=selection,
            max_validation_cache_bytes=1024**2,
        )
    dataset = CorpusDataset(masked.parent, input_policy=HISTORICAL_MASKED)
    features = sum(widths(dataset)) + 5
    (plan,) = plans
    assert plan["features"] == features and plan["rows"] == 4 and plan["on_host"] is False
    # La validación reside en RAM y se comparte entre configuraciones: ya no ocupa disco.
    assert plan["resident_bytes"] == 6 * (features * 4 + 24) + 2 * 4096
    assert "other_disk_bytes" not in plan
    assert plan["max_disk_cache_bytes"] == 1024**3 and not output.exists()


def test_strict_external_plan_keeps_its_width_and_optional_disk_budget(
    tmp_path, monkeypatch, learning_doubles
):
    plans = external(monkeypatch)
    manifest = training_corpus(tmp_path / "data")
    with pytest.raises(LibrariesReached):
        external_corpus.run_external_reference(manifest, tmp_path / "xgb", on_host=False)
    dataset = CorpusDataset(manifest)
    assert plans[0]["features"] == sum(widths(dataset))
    assert plans[0]["max_disk_cache_bytes"] is None and plans[0]["resident_bytes"] == 0


def masked_search(tmp_path, monkeypatch):
    engine, config, manifest, calls, backend = setup(tmp_path, monkeypatch)
    values = json.loads(config.read_text()) | dict(
        schema_version=3,
        rounds=6,
        selection=dict(schema_version=1, minimum_rounds=2, patience_rounds=2, min_delta=1e-5),
        max_validation_cache_bytes=1024**2,
        input_policy=HISTORICAL_MASKED,
        max_disk_cache_bytes=64 * 1024**3,
    )
    config.write_text(json.dumps(values))
    meta = json.loads(manifest.read_text()) | policy_identity(HISTORICAL_MASKED)
    manifest.write_text(json.dumps(meta))
    return engine, config, manifest, values, backend


def test_masked_search_design_propagates_policy_and_disk_budget(tmp_path, monkeypatch):
    engine, config, _, values, _ = masked_search(tmp_path, monkeypatch)
    loaded, cases, _ = engine._configuration(config)
    assert loaded == values
    ridge = [case for case in cases if case["kind"] == "ridge"]
    boosting = [case for case in cases if case["kind"] == "xgboost"]
    assert ridge and all(case["parameters"]["input_policy"] == HISTORICAL_MASKED for case in ridge)
    assert boosting and all(
        case["parameters"]["input_policy"] == HISTORICAL_MASKED
        and case["parameters"]["max_disk_cache_bytes"] == 64 * 1024**3
        and case["parameters"]["on_host"] is False
        for case in boosting
    )
    for change in (
        dict(on_host=True),
        dict(input_policy="strict_inputs_v1"),
        dict(max_disk_cache_bytes=0),
        dict(max_disk_cache_bytes=None),
    ):
        config.write_text(json.dumps(values | change))
        with pytest.raises(ValueError):
            engine._configuration(config)
    config.write_text(json.dumps({k: v for k, v in values.items() if k != "input_policy"}))
    with pytest.raises(ValueError):
        engine._configuration(config)


def test_masked_search_admits_only_the_masked_edition_and_forwards_the_policy(
    tmp_path, monkeypatch
):
    engine, config, manifest, _, _ = masked_search(tmp_path, monkeypatch)
    events = []

    def pause(*args, **kwargs):
        events.append(kwargs)
        return dict(status="paused")

    monkeypatch.setattr(engine, "run_tabular_reference", pause)
    monkeypatch.setattr(engine, "run_external_reference", pause)
    result = engine.run_tabular_search(config, manifest, tmp_path / "study")
    assert result["status"] == "paused"
    identity = result["identity"]
    assert {key: identity[key] for key in ("input_policy", "mask_contract")} == policy_identity(
        HISTORICAL_MASKED
    )
    assert events[0]["kind"] == "ridge" and events[0]["input_policy"] == HISTORICAL_MASKED
    strict = json.loads(manifest.read_text())
    for key in ("input_policy", "mask_contract"):
        strict.pop(key)
    manifest.write_text(json.dumps(strict))
    with pytest.raises(ValueError, match="política"):
        engine.run_tabular_search(config, manifest, tmp_path / "other")
    assert not (tmp_path / "other").exists()


def test_completed_masked_receipt_must_keep_policy_and_bits(tmp_path, monkeypatch):
    engine, config, manifest, _, backend = masked_search(tmp_path, monkeypatch)
    _, cases, _ = engine._configuration(config)
    case = next(case for case in cases if case["kind"] == "ridge")
    folder = tmp_path / "case"
    receipt = backend(manifest, folder, kind="ridge", **case["parameters"])
    source = json.loads(manifest.read_text())
    receipt.update(input_policy=HISTORICAL_MASKED, feature_order=list(MODALITIES))
    (folder / "run.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="política"):
        tabular_search._completed(folder, case, source, receipt["manifest_sha256"])
    receipt.update(feature_order=[*MODALITIES, "presence"], **policy_identity(HISTORICAL_MASKED))
    (folder / "run.json").write_text(json.dumps(receipt))
    assert tabular_search._completed(folder, case, source, receipt["manifest_sha256"])[2] == 0.1


def test_versioned_masked_design_keeps_the_grid_and_declares_its_disk_budget():
    masked, cases, _ = tabular_search._configuration(
        Path("configs/baselines/tabular-historical-masked.json")
    )
    strict, strict_cases, _ = tabular_search._configuration(
        Path("configs/baselines/tabular-convergence-us.json")
    )
    added = {"schema_version", "input_policy", "max_disk_cache_bytes"}
    assert {k: v for k, v in masked.items() if k not in added} == {
        k: v for k, v in strict.items() if k != "schema_version"
    }
    assert [case["id"] for case in cases] == [case["id"] for case in strict_cases]
    # Cota superior de filas: todas las ventanas de 64 sesiones del censo histórico.
    for bins in masked["bins"]:
        plan = external_cache_plan(
            rows=17_076_024,
            features=1719,
            max_bin=bins,
            on_host=False,
            max_host_cache_bytes=masked["max_host_cache_bytes"],
            max_disk_cache_bytes=masked["max_disk_cache_bytes"],
            available_ram=0,
            free_disk=2**62,
        )
        # Cabe la hipótesis densa, no la cota global. La guardia mide la diferencia.
        assert plan["disk_cache_bytes_estimate"] < masked["max_disk_cache_bytes"]
        assert plan["disk_cache_bytes_global_bins_bound"] > masked["max_disk_cache_bytes"]
