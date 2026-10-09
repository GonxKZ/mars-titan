"""Ventana walk-forward de MARS-TITAN sobre la ventana Titans-MAC elegida, sin ajustar pesos.

El padre es una ventana `mac_online` ajustada con el registrador de gradientes, así que sus
parámetros son los iniciales. El lector usa el mismo registrador. Se comprueban filas,
estado elegido compuesto, reanudación y los rechazos previos a abrir fuentes.
"""

import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.storage import sha256
from mars_titan.memory.mars_titan_variant import load_declaration, select_variant
from mars_titan.models.quantile_head import QUANTILE_COLUMNS
from mars_titan.training import mars_titan_walk_forward as mw
from mars_titan.training.learning_hold import LearningHoldError
from tests.training.test_titans_walk_forward import RULE, Factory, recipe, views, window
from tests.training.test_titans_walk_forward import (
    learning_doubles_module as learning_doubles_module,
)

DECLARED = Path("configs/titans/episodic-readout-historical-masked.json")
M1 = {"episodic_bank": "m1"}
M0 = {"episodic_bank": "m0_no_bank"}


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
            for name, components in (("m1", M1), ("m0", M0))
        }
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
    return dict(root=root, view=view, parent=parent, titans=titans, plan=plan, runs=runs)


def table(folder, partition):
    return pq.read_table(folder / f"{partition}-predictions.parquet").sort_by("sample_id")


@pytest.mark.parametrize("name", ["m1", "m0"])
def test_window_fits_the_reader_and_scores_the_parent_rows(base, name):
    report, factory = base["runs"][name]
    output = base["root"] / "runs" / name
    assert report["status"] == "completed" and report["final_test_opened"] is False
    assert factory.instances and all(i.calls == len(i.records) > 0 for i in factory.instances)
    fit = json.loads((output / "fit/run.json").read_text())
    assert fit["identity"]["recipe"]["learning_rate"] == 1e-4
    assert fit["identity"]["admission"] == ("m1" if name == "m1" else "m0")
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
    assert variant["components"] == (M1 if name == "m1" else M0)
    assert variant["base"]["parameters_sha256"] == fit["identity"]["parent"]["parameters_sha256"]
    assert report["identity"]["memory_policy"]["bank"].startswith("empty_at_each_pass")


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
        ({"episodic_bank": "m3"}, "M3"),
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
