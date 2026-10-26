"""Ventana walk-forward de MARS-TITAN sobre la ventana Titans-MAC elegida, sin ajustar pesos.

El padre es una ventana `mac_online` ajustada con el registrador de gradientes, así que sus
parámetros son los iniciales. El lector usa el mismo registrador. Se comprueban filas,
estado elegido compuesto, reanudación y los rechazos previos a abrir fuentes.
"""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.storage import sha256
from mars_titan.memory.mars_titan_variant import load_declaration, select_variant
from mars_titan.models.quantile_head import QUANTILE_COLUMNS
from mars_titan.training import mars_titan_walk_forward as mw
from mars_titan.training.learning_hold import LearningHoldError
from tests.learning_traces.parity import EVERY_STEP, metrics, same_rows, trace_rows
from tests.suite_support import skip_without_episodic_native
from tests.training.test_titans_walk_forward import RULE, Factory, recipe, views, window
from tests.training.test_titans_walk_forward import (
    learning_doubles_module as learning_doubles_module,
)

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")

DECLARED = Path("configs/titans/episodic-readout-historical-masked.json")
M1 = {"episodic_bank": "m1"}
M0 = {"episodic_bank": "m0_no_bank"}
M3 = {"episodic_bank": "m3"}
COMPONENTS = dict(m1=M1, m0=M0, m3=M3)


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(autouse=True)
def permitted_doubles(learning_doubles):
    return learning_doubles


def readout_recipe(root, **changes):
    document = json.loads(DECLARED.read_text())
    document["recipe"].update(
        update_instants=3, block_rows=2, epochs=2, selection=dict(RULE), bank_capacity=8
    )
    document["recipe"].update(changes)
    path = root / f"readout-{len(list(root.glob('readout-*')))}.json"
    path.write_text(json.dumps(document))
    return path


def mars(view, parent, plan, output, components=M1, *, case="lr1e-4", **options):
    factory = options.pop("optimizer_factory", None) or Factory()
    report = mw.run_mars_titan_window(
        view,
        parent,
        plan,
        components=components,
        seed=42,
        output=output,
        search_case=case,
        device="cpu",
        indices=output.parent / "indices",
        optimizer_factory=factory,
        **options,
    )
    return report, factory


@pytest.fixture(scope="module")
def base(tmp_path_factory, learning_doubles_module):
    skip_without_episodic_native()
    root = tmp_path_factory.mktemp("mars-titan-walk-forward")
    view, protocol = views(root / "base")
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        titans, _ = window(view, protocol, recipe(root), root / "runs" / "titans")
        parent = root / "runs" / "titans"
        plan = readout_recipe(root)
        runs = {
            name: mars(view, parent, plan, root / "runs" / name, components)
            for name, components in COMPONENTS.items()
        }
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
    return dict(root=root, view=view, parent=parent, titans=titans, plan=plan, runs=runs)


def table(folder, partition):
    return pq.read_table(folder / f"{partition}-predictions.parquet").sort_by("sample_id")


def test_bounded_reader_keeps_parent_phases_rows_and_recovery(tmp_path):
    """El lector hereda el mes de ajuste y recupera el mismo resultado sin modificar pesos."""
    view, protocol = views(tmp_path / "data")
    core_recipe = recipe(tmp_path, walk_forward=dict(warmup_months=1, fit_months=1))
    parent = tmp_path / "parent"
    parent_report, _ = window(view, protocol, core_recipe, parent)
    plan = readout_recipe(tmp_path)
    reference = tmp_path / "reference"
    expected, _ = mars(view, parent, plan, reference)
    assert expected["identity"]["phases"] == parent_report["identity"]["phases"]
    assert expected["identity"]["indices"] == parent_report["identity"]["indices"]
    assert expected["identity"]["memory_policy"]["fit_months"] == 1

    recorder = Factory()

    class StopAfterThree:
        @property
        def requested(self):
            return sum(item.calls for item in recorder.instances[-1:]) >= 3

    resumed = tmp_path / "resumed"
    paused, _ = mars(view, parent, plan, resumed, stop=StopAfterThree(), optimizer_factory=recorder)
    assert paused["status"] == "paused"
    restored, _ = mars(view, parent, plan, resumed, optimizer_factory=recorder)
    assert restored["status"] == "completed"
    assert restored["identity"]["phases"] == expected["identity"]["phases"]
    for partition in ("validation", "calibration", "evaluation"):
        actual, target = table(resumed, partition), table(reference, partition)
        assert actual.equals(target)
        for name in ("sample_id", "prediction_at", "target"):
            assert actual[name].equals(table(parent, partition)[name])


def test_context_parent_requires_a_consumer_with_session_lifecycle(tmp_path):
    document = json.loads(readout_recipe(tmp_path).read_text())
    document["predictor"] = dict(session_context="market_memory_v1", dtype="float32")
    with pytest.raises(ValueError, match="no recupera el contexto"):
        mw._frozen_parent(document, None, 42, "cpu", tmp_path / "absent", {"request": {}})


@pytest.mark.parametrize("bits", [*range(1, 10), 15, 16])
def test_reader_reconstructs_the_exact_combination_of_parent_extensions(tmp_path, bits):
    from tests.training.test_titans_walk_forward import PT1

    view, protocol = views(tmp_path / "data")
    changes = dict(PT1) if bits & 1 else {}
    if bits & 2:
        changes["memory_multiscale"] = dict(form="cascade", coupling=0.0625)
    if bits & 4:
        changes["presence_modulation"] = "film_v1"
    if bits in (9, 15, 16):
        changes["attention_layout"] = "sequence_contiguous_v1"
    parent = tmp_path / "parent"
    core_recipe = recipe(tmp_path)
    if bits in (8, 16):
        document = json.loads(core_recipe.read_text())
        document["predictor"]["session_context"] = "market_memory_v1"
        document["recipe"]["accumulation_rows"] = None
        core_recipe.write_text(json.dumps(document))
    parent_report, _ = window(
        view, protocol, core_recipe, parent, predictor_changes=changes or None
    )
    plan = readout_recipe(tmp_path, refinement_initialization="zero_refinement")
    output = tmp_path / "reader"
    result, factory = mars(view, parent, plan, output)
    assert result["status"] == "completed" and factory.records
    parent_fit = json.loads((parent / "fit/run.json").read_text())
    reader_fit = json.loads((output / "fit/run.json").read_text())
    assert (
        result["identity"]["variant"]["base"]["configuration"]
        == parent_fit["identity"]["predictor"]
    )
    assert (
        reader_fit["identity"]["readout"]["configuration"]["refinement_initialization"]
        == "zero_refinement"
    )
    for phase in ("validation", "calibration", "evaluation"):
        assert result["predictions"][phase]["rows"] == parent_report["predictions"][phase]["rows"]
        for column in ("sample_id", "target", "prediction_at"):
            assert table(output, phase)[column].equals(table(parent, phase)[column])
    if bits in (7, 8, 9, 15, 16):
        from benchmarks.combined_memory_reference import frozen_reference

        reference = tmp_path / "frozen"
        frozen = frozen_reference(view, parent, reference, seed=42, device="cpu")
        assert frozen["optimizer_steps"] == 0 and frozen["shared_parameters_unchanged"]
        assert table(reference, "evaluation").equals(table(output, "evaluation"))
    if bits in (7, 9, 15, 16):
        resumed_factory = Factory()

        class StopAfterThree:
            @property
            def requested(self):
                return sum(item.calls for item in resumed_factory.instances[-1:]) >= 3

        paused = tmp_path / "resumed"
        report, _ = mars(
            view, parent, plan, paused, stop=StopAfterThree(), optimizer_factory=resumed_factory
        )
        assert report["status"] == "paused"
        report, _ = mars(view, parent, plan, paused, optimizer_factory=resumed_factory)
        assert report["status"] == "completed"
        for phase in ("validation", "calibration", "evaluation"):
            assert table(paused, phase).equals(table(output, phase))


@pytest.mark.parametrize("name", ["m1", "m0", "m3"])
def test_window_fits_the_reader_and_scores_the_parent_rows(base, name):
    report, factory = base["runs"][name]
    output = base["root"] / "runs" / name
    assert report["status"] == "completed" and report["final_test_opened"] is False
    assert factory.instances and all(i.calls == len(i.records) > 0 for i in factory.instances)
    fit = json.loads((output / "fit/run.json").read_text())
    assert fit["identity"]["recipe"]["learning_rate"] == 1e-4
    assert fit["identity"]["admission"] == name
    selected = json.loads((output / "selected.json").read_text())
    assert report["checkpoint"] == dict(
        path="selected.json", sha256=sha256(output / "selected.json")
    )
    assert selected["parent"]["checkpoint_sha256"] == base["titans"]["checkpoint"]["sha256"]
    assert selected["readout"]["sha256"] == fit["best_checkpoint"]["sha256"]
    check = report["validation_selection_check"]
    assert check["absolute_difference"] <= 1e-6 * max(1.0, abs(check["best_score"]))
    for partition in ("validation", "calibration", "evaluation"):
        rows, parent = table(output, partition), table(base["parent"], partition)
        assert report["predictions"][partition]["rows"] == rows.num_rows == parent.num_rows
        for column in ("sample_id", "asset_id", "market", "prediction_at", "target"):
            assert rows[column].equals(parent[column]), (partition, column)
        assert set(QUANTILE_COLUMNS) <= set(rows.column_names)
    # La identidad de la variante parte del núcleo elegido en la ventana del padre.
    variant = report["identity"]["variant"]
    assert variant["components"] == COMPONENTS[name]
    assert variant["base"]["parameters_sha256"] == fit["identity"]["parent"]["parameters_sha256"]
    assert report["identity"]["memory_policy"]["bank"].startswith("empty_at_each_pass")


def test_m3_window_freezes_the_scalers_of_its_training_tramo(base):
    report, _ = base["runs"]["m3"]
    output = base["root"] / "runs" / "m3"
    fit = json.loads((output / "fit/run.json").read_text())
    stored = mw.WriteScalers.from_fields(fit["identity"]["retention"]["scalers"])
    train = report["identity"]["indices"]["train"]
    assert stored.source_sha256 == train != report["identity"]["indices"]["validation"]
    assert fit["identity"]["sources"]["train"]["index_sha256"] == train
    phase = fit["identity"]["sources"]["train"]["phase"]
    assert (stored.decision_start, stored.decision_end) == (
        phase["decision_start"],
        phase["decision_end"],
    )
    # Una ejecución reanudada reutiliza las escalas guardadas sin recorrer de nuevo el tramo.
    plan = mw.case_recipe(mw.load_recipe(base["plan"]), "lr1e-4")
    assert mw._training_scalers(output / "fit", SimpleNamespace(identity=train), plan) == stored
    with pytest.raises(ValueError, match="tramo de entrenamiento"):
        mw._training_scalers(output / "fit", SimpleNamespace(identity="0" * 64), plan)
    assert mw._anchor_scalers(fit) == stored


def test_identity_reuses_the_variant_builder_of_the_declaration(base):
    report, _ = base["runs"]["m1"]
    identity = report["identity"]["variant"]
    rebuilt = select_variant(load_declaration(), M1, base=identity["base"])
    assert rebuilt.fingerprint() == report["identity"]["variant_sha256"]
    assert identity["base"]["architecture"] == "titans_mac"
    assert set(identity["base"]) == {
        "architecture",
        "configuration",
        "dtype",
        "parameters_sha256",
        "recipe",
    }
    assert identity["base"]["configuration"]["variant"] == "mac_online"


def test_completed_window_is_returned_without_fitting_again(base):
    report, _ = base["runs"]["m1"]
    factory = Factory()
    again, _ = mars(
        base["view"],
        base["parent"],
        base["plan"],
        base["root"] / "runs" / "m1",
        optimizer_factory=factory,
    )
    assert again == report and factory.instances == []


def test_window_refuses_while_the_learning_hold_blocks(base, tmp_path, learning_hold):
    learning_hold(False)
    with pytest.raises(LearningHoldError, match="Bloqueo de aprendizaje vigente"):
        mars(base["view"], base["parent"], base["plan"], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def copied_parent(base, tmp_path, change):
    folder = tmp_path / "parent"
    shutil.copytree(base["parent"], folder)
    report = json.loads((folder / "run.json").read_text())
    change(report)
    (folder / "run.json").write_text(json.dumps(report))
    return folder


@pytest.mark.parametrize(
    ("components", "message"),
    [
        ({}, "propio Titans-MAC"),
        ({"episodic_bank": "m4"}, "no admite"),
        (
            {"associative_memory": dict(rule="delta", key="codec", rate=0.5, forgetting=0.0)},
            "B6",
        ),
        ({"regime_context": "filtered_hmm"}, "sin conexión"),
    ],
)
def test_combinations_without_a_reader_to_fit_are_rejected_first(
    base, tmp_path, components, message
):
    with pytest.raises(ValueError, match=message):
        mars(base["view"], base["parent"], base["plan"], tmp_path / "out", components)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r["request"].update(variant="mac_frozen"), "mac_online"),
        (lambda r: r["request"].update(seed=43), "semilla"),
        (lambda r: r["request"].update(view_sha256="0" * 64), "vista"),
        (lambda r: r.update(status="paused"), "completada"),
    ],
)
def test_parent_must_be_the_completed_mac_online_window_of_the_view(
    base, tmp_path, change, message
):
    parent = copied_parent(base, tmp_path, change)
    with pytest.raises(ValueError, match=message):
        mars(base["view"], parent, base["plan"], tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("changes", "case", "message"),
    [
        (dict(epochs=3), "lr1e-4", "regla"),
        (dict(), "lr1e-2", "casos de búsqueda"),
    ],
)
def test_reader_recipe_must_apply_the_protocol_and_name_its_case(
    base, tmp_path, changes, case, message
):
    plan = readout_recipe(tmp_path, **changes)
    with pytest.raises(ValueError, match=message):
        mars(base["view"], base["parent"], plan, tmp_path / "out", case=case)
    assert not (tmp_path / "out").exists()


def test_changed_parent_after_the_window_is_detected_on_resume(base, tmp_path):
    report, _ = base["runs"]["m0"]
    folder = tmp_path / "parent"
    shutil.copytree(base["parent"], folder)
    output = tmp_path / "copy"
    shutil.copytree(base["root"] / "runs" / "m0", output)
    with pytest.raises(ValueError, match="petición"):
        mars(base["view"], folder, base["plan"], output, M0)
    assert report["request"]["parent"]["path"] == str(base["parent"].resolve())


def test_traces_keep_the_reader_bit_for_bit_and_resume_without_duplicates(base, tmp_path):
    """Trazas en cada paso del lector M1 (#448), con el padre congelado y su memoria."""
    report, factory = base["runs"]["m1"]
    output = tmp_path / "traced"
    traced, traced_factory = mars(
        base["view"], base["parent"], base["plan"], output, traces=EVERY_STEP
    )
    assert len(traced_factory.records) == len(factory.records) > 0
    for one, two in zip(traced_factory.records, factory.records, strict=True):
        for a, b in zip(one, two, strict=True):
            assert (a is None and b is None) or torch.equal(a, b)
    for partition in ("validation", "calibration", "evaluation"):
        assert table(output, partition).equals(table(base["root"] / "runs" / "m1", partition))
    manifest, rows = trace_rows(output / "traces")
    assert manifest["identity"]["family"] == "mars_titan_readout"
    groups = {row["group"] for row in rows if row["metric"] == "grad_l2"}
    # El padre está congelado: solo el lector tiene grupos de parámetros.
    assert groups and all(group.startswith("readout.") for group in groups)
    assert "memory.alpha.mean" in metrics(rows, "train")
    assert {"validation.score", "weight_l2"} <= metrics(rows, "validation")
    resumed_factory = Factory()

    class StopAtStep:
        @property
        def requested(self):
            return sum(item.calls for item in resumed_factory.instances[-1:]) >= 3

    resumed = tmp_path / "resumed"
    options = dict(optimizer_factory=resumed_factory, traces=EVERY_STEP)
    paused, _ = mars(
        base["view"], base["parent"], base["plan"], resumed, stop=StopAtStep(), **options
    )
    assert paused["status"] == "paused"
    final, _ = mars(base["view"], base["parent"], base["plan"], resumed, **options)
    assert final["status"] == "completed"
    assert same_rows(rows, trace_rows(resumed / "traces")[1])
