"""Ventana walk-forward de la corrección B6 sobre la ventana Titans-MAC elegida.

El padre es una ventana `mac_online` ajustada con el registrador de gradientes, así que sus
parámetros son los iniciales. B6 no tiene parámetros: se comprueban la paridad con el padre
cuando η = 0, las filas, el estado elegido, la reanudación tras una parada, que el tramo de
entrenamiento no se abre y los rechazos previos a leer fuentes.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import sha256
from mars_titan.memory.associative_memory import CORRECTION_KEYS, RULES
from mars_titan.memory.mars_titan_variant import load_declaration, select_variant
from mars_titan.models.quantile_head import QUANTILE_COLUMNS
from mars_titan.training import campaign_plan as plan
from mars_titan.training import mars_titan_correction as mc
from mars_titan.training import mars_titan_walk_forward as mw
from mars_titan.training import titans_walk_forward as wf
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.financial_run import ChronologicalInference
from mars_titan.training.learning_hold import LearningHoldError
from tests.training.test_titans_walk_forward import (
    learning_doubles_module as learning_doubles_module,
)
from tests.training.test_titans_walk_forward import recipe, views, window

DECLARED = Path("configs/titans/mature-correction-historical-masked.json")
PROXIMAL = {"associative_memory": {"rule": "proximal", "key": "codec"}}
BIAS = {"associative_memory": {"rule": "proximal", "key": "constant"}}
PARTITIONS = ("validation", "calibration", "evaluation")
PREDICTED_COLUMNS = ("prediction", *QUANTILE_COLUMNS)


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(autouse=True)
def permitted_doubles(learning_doubles):
    return learning_doubles


def correction_recipe(root, *, base=None, cases=None):
    document = json.loads(DECLARED.read_text())
    if base is not None:
        document["recipe"] = base
    if cases is not None:
        document["walk_forward"]["search_cases"] = cases
    path = root / f"correction-{len(list(root.glob('correction-*')))}.json"
    path.write_text(json.dumps(document))
    return path


def correct(view, parent, path, output, components=PROXIMAL, *, case="eta25e-2", **options):
    return mc.run_correction_window(
        view,
        parent,
        path,
        components=components,
        seed=42,
        output=output,
        search_case=case,
        device="cpu",
        indices=output.parent / "indices",
        **options,
    )


@pytest.fixture(scope="module")
def base(tmp_path_factory, learning_doubles_module):
    root = tmp_path_factory.mktemp("mars-titan-correction")
    view, protocol = views(root / "base")
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        titans, _ = window(view, protocol, recipe(root), root / "runs" / "titans")
        parent = root / "runs" / "titans"
        zero = correction_recipe(root, base={"forgetting": 0.0}, cases={"zero": {"rate": 0.0}})
        runs = {
            "proximal": correct(view, parent, DECLARED, root / "runs" / "proximal"),
            "bias": correct(view, parent, DECLARED, root / "runs" / "bias", BIAS),
            "zero": correct(view, parent, zero, root / "runs" / "zero", case="zero"),
        }
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
    return dict(root=root, view=view, parent=parent, titans=titans, zero=zero, runs=runs)


def table(folder, partition):
    return pq.read_table(folder / f"{partition}-predictions.parquet").sort_by("sample_id")


def test_constants_repeat_the_runner_and_memory_names_without_importing_torch():
    assert plan.MARS_CORRECTION_RECIPE == mc.RECIPE
    assert plan.MARS_CORRECTION_SEARCHED == mc.SEARCHED
    assert plan.MARS_CORRECTION_RULES == RULES
    assert plan.MARS_CORRECTION_KEYS == CORRECTION_KEYS


def test_declared_recipe_shares_forgetting_and_searches_two_rates():
    document = mc.load_correction_recipe(DECLARED)
    assert document["recipe"] == {"forgetting": 0.01}
    assert [mc.case_values(document, name) for name in ("eta5e-2", "eta25e-2")] == [
        dict(forgetting=0.01, rate=0.05),
        dict(forgetting=0.01, rate=0.25),
    ]


def fresh_parent_rows(base, partition):
    """Filas de la inferencia cronológica del padre recargado, sin corrección."""
    report = json.loads((base["parent"] / "run.json").read_text())
    dataset = CorpusDataset(base["view"], input_policy=HISTORICAL_MASKED)
    titans, chronological = mc._parent_recipe(report)
    months = wf.walk_forward_options(titans)["warmup_months"]
    phases = wf.window_phases(report["identity"]["window"], months)
    sources = wf._sources(dataset, {partition: phases[partition]}, base["root"] / "runs/indices")
    specification = mc._parent_specification(sources[partition], report)
    predictor = mw._frozen_parent(titans, specification, 42, "cpu", base["parent"], report)
    inference = ChronologicalInference(predictor, chronological)
    rows = wf.PredictionRows(inference.quantiles)
    inference.predict(sources[partition], rows)
    return pa.concat_tables(rows.finish()).sort_by("sample_id")


def test_zero_rate_window_reproduces_the_reloaded_parent_bit_for_bit(base):
    output = base["root"] / "runs" / "zero"
    report = base["runs"]["zero"]
    for partition in PARTITIONS:
        mine, parent = table(output, partition), table(base["parent"], partition)
        assert mine.equals(fresh_parent_rows(base, partition)), partition
        # El Parquet del padre sale del entrenador en memoria. Su recarga coincide con él
        # salvo diferencias de redondeo de la última cifra en pocas filas.
        for column in ("sample_id", "asset_id", "market", "prediction_at", "target"):
            assert mine[column].equals(parent[column]), (partition, column)
        for column in PREDICTED_COLUMNS:
            pairs = zip(mine[column].to_pylist(), parent[column].to_pylist(), strict=True)
            for left, right in pairs:
                assert left == pytest.approx(right, rel=1e-14), (partition, column)
        assert report["predictions"][partition]["metrics"]["associative_writes"] > 0


@pytest.mark.parametrize("name", ["proximal", "bias"])
def test_window_corrects_the_parent_after_the_first_mature_labels(base, name):
    report = base["runs"][name]
    output = base["root"] / "runs" / name
    assert report["status"] == "completed" and report["final_test_opened"] is False
    for partition in PARTITIONS:
        mine, parent = table(output, partition), table(base["parent"], partition)
        assert report["predictions"][partition]["rows"] == mine.num_rows == parent.num_rows
        for column in ("sample_id", "asset_id", "market", "prediction_at", "target"):
            assert mine[column].equals(parent[column]), (partition, column)
        # Las primeras decisiones de cada tramo se emiten con A = 0. Después, la corrección
        # desplaza el punto y los cinco cuantiles en la misma cantidad.
        first = min(mine["prediction_at"].to_pylist())
        shifts = set()
        for row, other in zip(mine.to_pylist(), parent.to_pylist(), strict=True):
            delta = row["prediction"] - other["prediction"]
            if row["prediction_at"] == first:
                assert delta == 0.0
            for column in QUANTILE_COLUMNS:
                assert row[column] - other[column] == pytest.approx(delta, abs=1e-12)
            shifts.add(delta)
        assert len(shifts) > 1, partition
    selected = json.loads((output / "selected.json").read_text())
    assert report["checkpoint"] == dict(
        path="selected.json", sha256=sha256(output / "selected.json")
    )
    assert selected["parent"]["checkpoint_sha256"] == base["titans"]["checkpoint"]["sha256"]
    assert selected["correction"] == report["identity"]["correction"]
    assert selected["correction"]["memory"]["rate"] == 0.25
    assert report["identity"]["memory_policy"]["associative"].startswith("zero_at_each_pass")


def test_identity_reuses_the_variant_builder_and_never_opens_the_training_tramo(base):
    report = base["runs"]["proximal"]
    identity = report["identity"]
    full = mc.arm_components(PROXIMAL, dict(rate=0.25, forgetting=0.01))
    rebuilt = select_variant(load_declaration(), full, base=identity["variant"]["base"])
    assert rebuilt.fingerprint() == identity["variant_sha256"]
    assert identity["variant"]["components"] == full
    assert set(identity["indices"]) == set(PARTITIONS)
    assert set(identity["phases"]) == set(PARTITIONS)
    # El índice de entrenamiento solo existe por el padre, nunca por la corrección.
    trained = json.loads((base["parent"] / "run.json").read_text())["identity"]["indices"]
    assert trained["train"] not in identity["indices"].values()


def test_completed_window_is_returned_after_checking_its_rows(base):
    output = base["root"] / "runs" / "proximal"
    again = correct(base["view"], base["parent"], DECLARED, output)
    assert again == base["runs"]["proximal"]


class StopAfter:
    """Pedir la parada en la consulta número `calls`, dentro de un tramo."""

    def __init__(self, calls):
        self.calls, self.seen = calls, 0

    @property
    def requested(self):
        self.seen += 1
        return self.seen > self.calls


def test_interrupted_window_resumes_with_the_same_rows(base, tmp_path):
    output = tmp_path / "resumed"
    paused = correct(base["view"], base["parent"], DECLARED, output, stop=StopAfter(40))
    assert paused["status"] == "paused" and len(paused["predictions"]) < 3
    finished = correct(base["view"], base["parent"], DECLARED, output)
    assert finished["status"] == "completed"
    reference = base["runs"]["proximal"]
    for partition in PARTITIONS:
        assert (
            finished["predictions"][partition]["sha256"]
            == reference["predictions"][partition]["sha256"]
        )
    assert len(finished["attempts"]) == 2


def test_window_refuses_while_the_learning_hold_blocks(base, tmp_path, learning_hold):
    learning_hold(False)
    with pytest.raises(LearningHoldError, match="Bloqueo de aprendizaje vigente"):
        correct(base["view"], base["parent"], DECLARED, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("components", "case", "message"),
    [
        ({"associative_memory": {"rule": "proximal"}}, "eta25e-2", "regla y su clave"),
        (
            {"associative_memory": dict(rule="proximal", key="codec", rate=0.5)},
            "eta25e-2",
            "regla y su clave",
        ),
        (PROXIMAL | {"episodic_bank": "m1"}, "eta25e-2", "solo associative_memory"),
        ({"associative_memory": {"rule": "lms", "key": "codec"}}, "eta25e-2", "rule"),
        (PROXIMAL, "eta1", "casos de búsqueda"),
    ],
)
def test_invalid_arms_and_cases_are_rejected_before_opening_sources(
    base, tmp_path, components, case, message
):
    with pytest.raises(ValueError, match=message):
        correct(base["view"], base["parent"], DECLARED, tmp_path / "out", components, case=case)
    assert not (tmp_path / "out").exists()


def test_delta_rate_above_its_stability_bound_is_rejected_first(base, tmp_path):
    path = correction_recipe(tmp_path, cases={"fast": {"rate": 1.995}})
    delta = {"associative_memory": {"rule": "delta", "key": "codec"}}
    with pytest.raises(ValueError, match="2 − λ"):
        correct(base["view"], base["parent"], path, tmp_path / "out", delta, case="fast")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.update(recipe_name="other"), "η y λ"),
        (lambda d: d["recipe"].update(rate=0.1), "η y λ"),
        (lambda d: d.update(recipe={}), "η y λ"),
        (lambda d: d["walk_forward"]["search_cases"].update(other={"rate": 0.05}), "η y λ"),
        (lambda d: d.update(epochs=30), "η y λ"),
    ],
)
def test_recipe_declares_rate_and_forgetting_once(tmp_path, change, message):
    document = json.loads(DECLARED.read_text())
    change(document)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=message):
        mc.load_correction_recipe(path)


def copied_parent(base, tmp_path, change):
    folder = tmp_path / "parent"
    shutil.copytree(base["parent"], folder)
    report = json.loads((folder / "run.json").read_text())
    change(report)
    (folder / "run.json").write_text(json.dumps(report))
    return folder


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r["request"].update(variant="mac_frozen"), "mac_online"),
        (lambda r: r["request"].update(seed=43), "semilla"),
        (lambda r: r["request"].update(view_sha256="0" * 64), "vista"),
        (lambda r: r["request"].update(local_control={"mode": "penalty"}), "control C"),
        (lambda r: r.update(status="paused"), "completada"),
    ],
)
def test_parent_must_be_the_completed_mac_online_window_of_the_view(
    base, tmp_path, change, message
):
    parent = copied_parent(base, tmp_path, change)
    with pytest.raises(ValueError, match=message):
        correct(base["view"], parent, DECLARED, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_changed_request_is_detected_on_resume(base, tmp_path):
    output = tmp_path / "copy"
    shutil.copytree(base["root"] / "runs" / "bias", output)
    with pytest.raises(ValueError, match="petición"):
        correct(base["view"], base["parent"], DECLARED, output, BIAS, case="eta5e-2")


def macro_blocks(view, phase, root):
    """Bloques macro de los eventos de un tramo leídos con un corpus recién abierto."""
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    (source,) = wf._sources(dataset, {phase.partition: phase}, root).values()
    return [
        raw["inputs"]["macro"].copy()
        for event in source.batched_events(block_rows=2)
        for raw in event.inputs
    ]


def test_masked_reader_never_substitutes_the_macro_block(base, monkeypatch):
    """Frontera 2: la edición con máscaras no sustituye macro con `TemporalInputs.lookup`.

    Se falsea la sustitución con valores imposibles. Si el lector la aplicara, los bloques
    macro de las observaciones de la ventana cambiarían.
    """
    from mars_titan.training.temporal_corpus import TemporalInputs

    report = json.loads((base["parent"] / "run.json").read_text())
    titans = report["identity"]["recipe"]
    months = wf.walk_forward_options(titans)["warmup_months"]
    phase = wf.window_phases(report["identity"]["window"], months)["validation"]
    expected = macro_blocks(base["view"], phase, base["root"] / "runs/indices")
    original, calls = TemporalInputs.lookup, []

    def substituted(self, prediction):
        _, available, valid = original(self, prediction)
        calls.append(len(prediction))
        width = expected[0].shape[-1]
        return np.full((len(prediction), width), 1000.0, dtype=np.float32), available, valid

    monkeypatch.setattr(TemporalInputs, "lookup", substituted)
    observed = macro_blocks(base["view"], phase, base["root"] / "runs/indices")
    assert len(observed) == len(expected) > 0
    assert all(np.array_equal(a, b) for a, b in zip(observed, expected, strict=True))
    assert all(not np.any(block == 1000.0) for block in observed)
    # El contrato temporal sigue activo: se consulta para la elegibilidad de las etiquetas.
    assert calls
