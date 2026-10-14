"""Destilación temporalmente válida de #57 sin pasos de optimizador.

Profesor y alumno son Titans-MAC del corpus técnico de los entrenadores cronológicos. El
profesor se recorre hasta el paso con el optimizador que solo registra gradientes, así que
su informe es el de una ejecución completada con sus pesos iniciales. Las comprobaciones
cubren el archivo de salidas (dentro de la muestra, sin etiquetas y ligado a su tramo, su
cabeza, su memoria y su profesor), el objetivo del alumno y su gradiente, y que el profesor
no llega nunca a las predicciones emitidas ni a la validación.
"""

import contextlib
import copy
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.models.baselines.multimodal import SCALAR_HEAD
from mars_titan.models.quantile_head import LEVELS, PINBALL, QUANTILE_HEAD
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.local_control import MACProjectionConfig
from mars_titan.training import distillation
from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer
from mars_titan.training.selection import FIXED_BUDGET
from tests.training.chronological_fixture import phases
from tests.training.test_financial_run import (
    SELECTION,
    RecordingOptimizer,
    corpus,
    entries,
    explicit_fastpath,  # noqa: F401
    named_records,
    shared,  # noqa: F401
)

DECLARATION = Path("configs/evaluation/temporal-distillation.json")
HEADS = [SCALAR_HEAD, QUANTILE_HEAD]


def model(streams, head, seed):
    specification = streams["train"].specification()
    config = FinancialConfig(
        specification, variant="mac_online", hidden_size=32, seed=seed, head=head
    )
    return FinancialPredictor(config, dtype=torch.float64)


def engine(streams, output, *, head=SCALAR_HEAD, seed=42, teacher=None, weight=None, **options):
    recipe = dict(
        truncation=3,
        epochs=1,
        block_rows=2,
        selection=dict(SELECTION, stopping=FIXED_BUDGET),
        loss=PINBALL if head == QUANTILE_HEAD else "mae",
        distillation_weight=weight,
    )
    recipe.update(options)
    return ChronologicalTrainer(
        model(streams, head, seed),
        ChronologicalRecipe(**recipe),
        train=streams["train"],
        validation=streams["validation"],
        output=output,
        optimizer_factory=RecordingOptimizer,
        teacher=teacher,
        audit=True,
    )


@contextlib.contextmanager
def without_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(scope="module")
def teachers(shared, tmp_path_factory):  # noqa: F811
    """Un profesor por cabeza, con semilla distinta del alumno, y sus salidas en el ajuste."""
    _, streams = shared
    folder = tmp_path_factory.mktemp("teachers")
    result = {}
    with without_fastpath():
        for head in HEADS:
            teacher = engine(streams, folder / f"{head}-run", head=head, seed=7)
            assert teacher.run()["status"] == "completed"
            report = folder / f"{head}-run" / "run.json"
            manifest = distillation.generate_teacher_targets(
                folder / f"{head}-targets", teacher, streams["train"], arm="teacher", report=report
            )
            identity = distillation.teacher_identity("teacher", report, streams["train"])
            result[head] = dict(engine=teacher, report=report, manifest=manifest, identity=identity)
    return result


def targets(teachers, streams, head, *, manifest=None):
    found = teachers[head]
    return distillation.TeacherTargets(
        manifest or found["manifest"],
        source=streams["train"],
        teacher=found["identity"],
        quantiles=head == QUANTILE_HEAD,
    )


@pytest.mark.parametrize("value", [0.0, 1.0, 1.5, -0.25, float("nan"), 1, True, "0.5"])
def test_the_recipe_declares_a_weight_strictly_between_zero_and_one(value):
    with pytest.raises(ValueError, match="fracción"):
        ChronologicalRecipe(distillation_weight=value)


def test_the_weight_enters_the_identity_only_when_declared():
    plain = ChronologicalRecipe().identity()
    assert "distillation_weight" not in plain and "distillation_distance" not in plain
    declared = ChronologicalRecipe(distillation_weight=0.25).identity()
    assert declared["distillation_weight"] == 0.25
    assert declared["distillation_distance"] == distillation.DISTANCE
    assert {k: v for k, v in declared.items() if not k.startswith("distillation")} == plain


@pytest.mark.parametrize("head", HEADS)
def test_the_teacher_outputs_are_in_sample_outputs_of_the_train_phase_without_labels(
    shared,  # noqa: F811
    teachers,
    head,
):
    _, streams = shared
    found = teachers[head]
    meta = json.loads(found["manifest"].read_text())
    assert meta["in_sample"] is True and meta["out_of_sample_predictions"] is False
    assert meta["use"] == distillation.USE and meta["memory"] == distillation.MEMORY
    assert meta["labels_stored"] is False and meta["final_test_opened"] is False
    table = pq.read_table(found["manifest"].parent / meta["targets_path"])
    assert table.schema.names == ["flow", "decision_at", "outputs"]
    # Las salidas son las que emite el profesor elegido al recorrer de nuevo el tramo.
    rows = []
    with without_fastpath():
        found["engine"].evaluate(streams["train"], rows=rows)
    assert meta["rows"] == table.num_rows == len(rows)
    expected = {
        (flow, at): list(levels) if head == QUANTILE_HEAD else [issued]
        for flow, at, issued, _, levels in rows
    }
    stored = zip(
        table["flow"].to_pylist(),
        table["decision_at"].to_pylist(),
        table["outputs"].to_pylist(),
        strict=True,
    )
    assert {(flow, at): values for flow, at, values in stored} == expected
    phase = streams["train"].phase
    decisions = table["decision_at"].to_pylist()
    assert phase.decision_start <= min(decisions) and max(decisions) < phase.decision_end
    assert meta["cut_us"] == max(decisions)
    assert meta["outputs"] == (
        dict(kind="quantiles", levels=list(LEVELS))
        if head == QUANTILE_HEAD
        else dict(kind=distillation.POINT)
    )
    loaded = targets(teachers, streams, head)
    keys = list(expected)[:5]
    values = loaded.values(keys, device="cpu", dtype=torch.float64)
    reference = torch.tensor([expected[key] for key in keys], dtype=torch.float64)
    torch.testing.assert_close(
        values, reference if head == QUANTILE_HEAD else reference[:, 0], rtol=0, atol=0
    )


def _edited(found, tmp_path, change):
    folder = tmp_path / "edited"
    shutil.copytree(found["manifest"].parent, folder)
    meta = json.loads((folder / "manifest.json").read_text())
    change(meta, folder)
    (folder / "manifest.json").write_text(json.dumps(meta))
    return folder / "manifest.json"


def _rewrite_values(meta, folder):
    path = folder / meta["targets_path"]
    table = pq.read_table(path)
    shifted = [[value + 1.0 for value in row] for row in table["outputs"].to_pylist()]
    pq.write_table(table.set_column(2, "outputs", [shifted]).cast(table.schema), path)


EDITS = {
    "memory": (lambda meta, _: meta.update(memory="bank_kept_from_previous_window"), "memoria"),
    "in_sample": (lambda meta, _: meta.update(in_sample=False), "pérdida del alumno"),
    "use": (lambda meta, _: meta.update(use="bank_errors"), "pérdida del alumno"),
    "labels": (lambda meta, _: meta.update(labels_stored=True), "pérdida del alumno"),
    "view": (lambda meta, _: meta["source"].update(view_sha256="0" * 64), "otro tramo"),
    "representation": (lambda meta, _: meta["source"].update(representation="r"), "otro tramo"),
    "phase": (lambda meta, _: meta["source"]["phase"].update(close_at=1), "otro tramo"),
    "teacher": (lambda meta, _: meta["teacher"].update(run_id="other"), "otro profesor"),
    "content": (_rewrite_values, "ha cambiado"),
    "cut": (lambda meta, _: meta.update(cut_us=meta["cut_us"] - 1), "corte"),
}


@pytest.mark.parametrize("name", sorted(EDITS))
def test_the_targets_reject_other_tramos_memories_teachers_and_contents(
    shared,  # noqa: F811
    teachers,
    tmp_path,
    name,
):
    _, streams = shared
    change, message = EDITS[name]
    manifest = _edited(teachers[SCALAR_HEAD], tmp_path, change)
    with pytest.raises(ValueError, match=message):
        targets(teachers, streams, SCALAR_HEAD, manifest=manifest)


def test_the_targets_reject_another_head_phase_or_corpus(shared, teachers, tmp_path):  # noqa: F811
    _, streams = shared
    found = teachers[SCALAR_HEAD]
    with pytest.raises(ValueError, match="otra cabeza"):
        distillation.TeacherTargets(
            found["manifest"], source=streams["train"], teacher=found["identity"], quantiles=True
        )
    with pytest.raises(ValueError, match="tramo de ajuste"):
        distillation.TeacherTargets(
            found["manifest"],
            source=streams["validation"],
            teacher=found["identity"],
            quantiles=False,
        )
    # Las mismas dimensiones con otro corpus no bastan.
    _, other = corpus(tmp_path / "other", perturb_after=phases()[0].decision_start)
    with pytest.raises(ValueError, match="otro tramo"):
        distillation.TeacherTargets(
            found["manifest"], source=other["train"], teacher=found["identity"], quantiles=False
        )


def test_the_writer_rejects_rows_outside_the_phase_repeated_or_crossing(
    shared,  # noqa: F811
    teachers,
    tmp_path,
):
    _, streams = shared
    rows = []
    with without_fastpath():
        teachers[QUANTILE_HEAD]["engine"].evaluate(streams["train"], rows=rows)
    teacher = teachers[QUANTILE_HEAD]["identity"]
    first = rows[0]
    crossing = [(*first[:4], list(reversed(first[4]))), *rows[1:]]
    late = [(first[0], streams["train"].phase.decision_end, *first[2:]), *rows[1:]]
    for name, changed, message in (
        ("crossing", crossing, "ordenados"),
        ("late", late, "fuera de las decisiones"),
        ("repeated", [first, *rows], "una sola vez"),
        ("missing", [(*first[:4], None), *rows[1:]], "cinco cuantiles"),
    ):
        with pytest.raises(ValueError, match=message):
            distillation._write(
                tmp_path / name, changed, source=streams["train"], teacher=teacher, quantiles=True
            )


def test_the_teacher_is_a_completed_run_of_the_same_train_phase(shared, teachers, tmp_path):  # noqa: F811
    _, streams = shared
    found = teachers[SCALAR_HEAD]
    report = json.loads(found["report"].read_text())
    for name, change, message in (
        ("running", lambda r: r.update(status="running"), "completada"),
        ("opened", lambda r: r.update(final_test_opened=True), "completada"),
        (
            "phase",
            lambda r: r["identity"]["sources"]["train"]["phase"].update(decision_start=1),
            "tramo de ajuste",
        ),
    ):
        edited = json.loads(json.dumps(report))
        change(edited)
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(edited))
        with pytest.raises(ValueError, match=message):
            distillation.teacher_identity("teacher", path, streams["train"])
    # Ni un entrenador con otra identidad en la carpeta del profesor ni uno con la misma
    # identidad en otra carpeta pueden emitir las salidas de la ejecución declarada.
    for name, folder, seed in (
        ("other", found["report"].parent, 11),
        ("elsewhere", tmp_path / "elsewhere", 7),
    ):
        student = engine(streams, folder, seed=seed)
        with pytest.raises(ValueError, match="ejecución declarada"):
            distillation.generate_teacher_targets(
                tmp_path / f"{name}-targets",
                student,
                streams["train"],
                arm="teacher",
                report=found["report"],
            )


def test_the_trainer_needs_weight_and_teacher_together_and_records_them(
    shared,  # noqa: F811
    teachers,
    tmp_path,
):
    _, streams = shared
    loaded = targets(teachers, streams, SCALAR_HEAD)
    with pytest.raises(ValueError, match="salidas del profesor de este tramo"):
        engine(streams, tmp_path / "a", weight=0.5)
    with pytest.raises(ValueError, match="salidas del profesor de este tramo"):
        engine(streams, tmp_path / "b", teacher=loaded)
    with pytest.raises(ValueError, match="salidas del profesor de este tramo"):
        engine(streams, tmp_path / "c", head=QUANTILE_HEAD, teacher=loaded, weight=0.5)
    plain = engine(streams, tmp_path / "d")
    student = engine(streams, tmp_path / "e", teacher=loaded, weight=0.5)
    assert "distillation" not in plain.identity
    recorded = student.identity["distillation"]
    assert recorded["manifest_sha256"] == loaded.identity["manifest_sha256"]
    assert recorded["teacher"] == teachers[SCALAR_HEAD]["identity"]
    assert recorded["weight"] == 0.5 and recorded["use"] == distillation.USE
    assert recorded["implementation_sha256"]
    assert student.run_id != plain.run_id
    other = engine(streams, tmp_path / "f", teacher=loaded, weight=0.25)
    assert other.run_id != student.run_id
    # Unas salidas cargadas para otro tramo no entran aunque el objeto sea válido.
    moved = copy.copy(loaded)
    moved.source = dict(loaded.source, view_sha256="0" * 64)
    with pytest.raises(ValueError, match="salidas del profesor de este tramo"):
        engine(streams, tmp_path / "g", teacher=moved, weight=0.5)


def test_the_targets_come_from_the_chosen_state_of_the_teacher(shared, teachers, tmp_path):  # noqa: F811
    _, streams = shared
    found = teachers[SCALAR_HEAD]
    # Mismo modelo, receta, semilla y carpeta: es la ejecución declarada, reconstruida.
    other = engine(streams, found["report"].parent, seed=7)
    assert other.run_id == found["engine"].run_id
    with torch.no_grad():
        for value in other.predictor.parameters():
            value.add_(0.5)
    with without_fastpath():
        manifest = distillation.generate_teacher_targets(
            tmp_path / "targets", other, streams["train"], arm="teacher", report=found["report"]
        )
    # Antes de recorrer el tramo se carga el estado elegido, así que las salidas son las mismas.
    expected = pq.read_table(found["manifest"].parent / "teacher-targets.parquet")
    assert pq.read_table(manifest.parent / "teacher-targets.parquet").equals(expected)


def _gradients(engine_):
    return named_records(engine_)


@pytest.mark.parametrize("head", HEADS)
def test_the_objective_is_affine_in_the_weight_with_a_fixed_distance(
    shared,  # noqa: F811
    teachers,
    tmp_path,
    head,
):
    """(1 − w) · tarea + w · distancia: con dos pesos, la distancia que se deduce es la misma."""
    _, streams = shared
    loaded = targets(teachers, streams, head)
    plain = engine(streams, tmp_path / "plain", head=head)
    plain.run()
    task = _gradients(plain)
    deduced = []
    for weight in (0.25, 0.5):
        student = engine(streams, tmp_path / f"w{weight}", head=head, teacher=loaded, weight=weight)
        student.run()
        mixed = _gradients(student)
        assert len(mixed) == len(task) > 3
        deduced.append(
            [
                {
                    name: (value - (1 - weight) * left[name]) / weight
                    for name, value in right.items()
                    if value is not None
                }
                for left, right in zip(task, mixed, strict=True)
            ]
        )
    for first, second in zip(*deduced, strict=True):
        assert first.keys() == second.keys()
        for name, value in first.items():
            torch.testing.assert_close(second[name], value, rtol=1e-9, atol=1e-12)
    # La distancia aporta gradiente: el profesor no coincide con el alumno.
    assert any(bool(value.abs().sum() > 0) for record in deduced[0] for value in record.values())


@pytest.mark.parametrize("head", HEADS)
def test_each_update_compares_its_own_rows_with_the_teacher(
    shared,  # noqa: F811
    teachers,
    tmp_path,
    monkeypatch,
    head,
):
    """Cada paso mezcla la tarea con las salidas del profesor de las mismas filas maduras."""
    _, streams = shared
    loaded = targets(teachers, streams, head)
    student = engine(streams, tmp_path / "student", head=head, teacher=loaded, weight=0.5)
    calls, objective = [], distillation.objective

    def recorded(task, prediction, teacher, weight):
        calls.append((prediction.detach().clone(), teacher.clone(), weight))
        return objective(task, prediction, teacher, weight)

    monkeypatch.setattr(distillation, "objective", recorded)
    student.run()
    updates = entries(student.audit, "update")
    assert len(calls) == len(updates) > 3
    issued = {(flow, at): value for _, _, flow, at, value in entries(student.audit, "prediction")}
    for (prediction, teacher, weight), update in zip(calls, updates, strict=True):
        keys = [(flow, at) for flow, at, _ in update[3]]
        expected = loaded.values(keys, device="cpu", dtype=torch.float64)
        torch.testing.assert_close(teacher, expected, rtol=0, atol=0)
        assert weight == 0.5
        if head == SCALAR_HEAD:
            # El alumno se compara con lo que emitió en esas filas, no con otra pasada.
            reference = torch.tensor([issued[key] for key in keys], dtype=torch.float64)
            torch.testing.assert_close(prediction, reference, rtol=0, atol=0)


@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("rows", [1, 2])
def test_accumulation_gives_the_same_distilled_gradient(
    shared,  # noqa: F811
    teachers,
    tmp_path,
    head,
    rows,
):
    _, streams = shared
    loaded = targets(teachers, streams, head)
    full = engine(streams, tmp_path / "full", head=head, teacher=loaded, weight=0.5)
    full.run()
    blocks = engine(
        streams,
        tmp_path / "blocks",
        head=head,
        teacher=loaded,
        weight=0.5,
        accumulation_rows=rows,
    )
    blocks.run()
    assert blocks.audit == full.audit
    for left, right in zip(_gradients(blocks), _gradients(full), strict=True):
        assert left.keys() == right.keys()
        for name, value in right.items():
            if value is None:
                assert left[name] is None
            else:
                torch.testing.assert_close(left[name], value, rtol=1e-10, atol=1e-13)


@pytest.mark.parametrize("head", HEADS)
def test_the_teacher_never_reaches_emitted_predictions_errors_or_validation(
    shared,  # noqa: F811
    teachers,
    tmp_path,
    head,
):
    _, streams = shared
    loaded = targets(teachers, streams, head)
    plain = engine(streams, tmp_path / "plain", head=head)
    student = engine(streams, tmp_path / "student", head=head, teacher=loaded, weight=0.5)
    expected, report = plain.run(), student.run()
    # Predicciones emitidas, etiquetas y composición de cada tramo, bit a bit.
    for kind in ("prediction", "label", "update"):
        assert entries(student.audit, kind) == entries(plain.audit, kind)
    for left, right in zip(report["history"], expected["history"], strict=True):
        assert left["validation"] == right["validation"]
        if right["train"] is not None:
            # La pérdida informada es la de la tarea, no la mezcla con el profesor.
            assert left["train"] == right["train"]


def test_the_c_penalty_keeps_the_distance_to_the_teacher(shared, teachers, tmp_path):  # noqa: F811
    """Con la penalización C de CM-v1, el objetivo suma C a la mezcla, no solo a la tarea.

    Se usa la cabeza de cuantiles: con MAE y predicciones al mismo lado de la etiqueta y del
    profesor, la distancia y la tarea tienen el mismo gradiente y la pendiente sería nula.
    """
    _, streams = shared
    loaded = targets(teachers, streams, QUANTILE_HEAD)
    penalty = MACProjectionConfig(
        rank=2,
        frequency=1,
        seed=91,
        grid_size=16,
        threshold=0.0,
        max_flows=1,
        mode="penalty",
        weight=0.1,
    )

    def penalized(output, weight):
        specification = streams["train"].specification()
        config = FinancialConfig(
            specification, variant="mac_online", hidden_size=32, seed=42, head=QUANTILE_HEAD
        )
        predictor = FinancialPredictor(config, local_control=penalty, dtype=torch.float64)
        recipe = dict(truncation=3, epochs=1, block_rows=2, selection=SELECTION, loss=PINBALL)
        trainer = ChronologicalTrainer(
            predictor,
            ChronologicalRecipe(**recipe, distillation_weight=weight),
            train=streams["train"],
            validation=streams["validation"],
            output=output,
            optimizer_factory=RecordingOptimizer,
            teacher=None if weight is None else loaded,
        )
        trainer.run()
        return _gradients(trainer)

    plain = penalized(tmp_path / "plain", None)
    slopes = []
    for weight in (0.25, 0.5):
        mixed = penalized(tmp_path / f"w{weight}", weight)
        slopes.append(
            [
                {
                    name: (right[name] - value) / weight
                    for name, value in left.items()
                    if value is not None
                }
                for left, right in zip(plain, mixed, strict=True)
            ]
        )
    # g(w) = g(0) + w · (distancia − tarea): la pendiente no depende del peso y no es nula.
    for first, second in zip(*slopes, strict=True):
        for name, value in first.items():
            torch.testing.assert_close(second[name], value, rtol=1e-9, atol=1e-12)
    assert any(bool(value.abs().sum() > 0) for record in slopes[0] for value in record.values())


def test_a_missing_teacher_row_stops_the_update(shared, teachers, tmp_path):  # noqa: F811
    _, streams = shared
    found = teachers[SCALAR_HEAD]
    rows = []
    with without_fastpath():
        found["engine"].evaluate(streams["train"], rows=rows)
    manifest = distillation._write(
        tmp_path / "partial",
        rows[1:],
        source=streams["train"],
        teacher=found["identity"],
        quantiles=False,
    )
    partial = targets(teachers, streams, SCALAR_HEAD, manifest=manifest)
    student = engine(streams, tmp_path / "student", teacher=partial, weight=0.5)
    with pytest.raises(ValueError, match="Falta la salida del profesor"):
        student.run()


def test_the_objective_keeps_the_shapes_and_weights():
    prediction = torch.tensor([[0.0, 1.0], [2.0, 4.0]], dtype=torch.float64)
    teacher = torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float64)
    task = torch.tensor(3.0, dtype=torch.float64)
    value = distillation.objective(task, prediction, teacher, 0.25)
    assert float(value) == 0.75 * 3.0 + 0.25 * (1.0 + 0.0 + 0.0 + 2.0) / 4
    with pytest.raises(ValueError, match="forma"):
        distillation.objective(task, prediction, teacher[:, :1], 0.25)


def mutated_declaration(tmp_path, change):
    value = json.loads(DECLARATION.read_text())
    change(value)
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(value))
    return path


INVALID = {
    "status": lambda v: v.update(status="executed"),
    "test": lambda v: v.update(final_test_opened=True),
    "no_weights": lambda v: v["student"].update(distillation_weights=[]),
    "zero_weight": lambda v: v["student"].update(distillation_weights=[0.0, 0.5]),
    "repeated_weight": lambda v: v["student"].update(distillation_weights=[0.5, 0.5]),
    "direct_other": lambda v: v["student"].update(direct_control="titans_mac_frozen"),
    "teacher_is_student": lambda v: v["teacher"].update(arm=v["student"]["arm"]),
    "out_of_sample": lambda v: v["targets"].update(in_sample=False),
    "memory": lambda v: v["targets"].update(memory="bank_kept"),
}


def test_the_declaration_is_checked_without_generating_targets(capsys):
    meta, digest = distillation.load_declaration(DECLARATION)
    assert meta["status"] == "declared_not_executed" and meta["final_test_opened"] is False
    assert distillation.main(["check", "--declaration", str(DECLARATION)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["sha256"] == digest
    assert printed["distillation_weights"] == meta["student"]["distillation_weights"]


def test_the_cost_condition_follows_the_measured_receipt():
    """La condición de coste de la activación sale del recibo de `cuda:0`, sin pasos."""
    meta, _ = distillation.load_declaration(DECLARATION)
    condition = meta["cost_condition"]
    receipt = json.loads(Path(condition["receipt"]).read_text())
    assert receipt["models"]["student"] == meta["student"]["arm"]
    assert receipt["models"]["teacher"] == meta["teacher"]["arm"]
    assert receipt["optimizer_steps"] == 0 and receipt["training_runs"] == 0
    assert receipt["final_test_opened"] is False and receipt["evaluation"] is False
    assert receipt["precision"]["dtype"] == "float32"
    assert receipt["precision"]["tf32_matmul"] is False
    # Cada cociente compara los dos caudales de la misma ronda, sobre los mismos eventos.
    rates = receipt["inference_rows_per_second"]
    ratios = receipt["teacher_over_student_inference_cost"]
    assert len(rates["student"]) == len(rates["teacher"]) == receipt["repeats"] >= 3
    pairs = zip(rates["student"], rates["teacher"], strict=True)
    assert ratios["per_round"] == [student / teacher for student, teacher in pairs]
    assert ratios["lowest"] == min(ratios["per_round"])
    assert condition["met_in_measured_view"] is (ratios["lowest"] >= condition["threshold"])
    assert condition["threshold"] == 1.5


@pytest.mark.parametrize("name", sorted(INVALID))
def test_invalid_declarations_are_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        distillation.load_declaration(mutated_declaration(tmp_path, INVALID[name]))


def test_the_phase_dictionary_matches_the_source(shared):  # noqa: F811
    _, streams = shared
    bound = distillation.bound_source(streams["train"])
    assert bound["phase"] == asdict(streams["train"].phase)
    assert bound["source_sha256"] == streams["train"].dataset.identity
