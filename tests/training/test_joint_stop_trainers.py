"""Parada conjunta en los entrenadores cronológicos, sin pasos que modifiquen pesos.

Las pasadas de ajuste solo cuentan actualizaciones y la validación devuelve errores
escritos de antemano. Se comprueba que cada ajuste espera al grupo en su primera meseta
con un estado recuperable, que la continuación recorre exactamente hasta la época común y
que dos ajustes con mesetas distintas aplican así el mismo número de actualizaciones.
Titans-MAC no necesita el enlace nativo. La GRU candidata y el lector de MARS-TITAN sí.
"""

import json
import os

import pytest
import torch

from mars_titan.training.checkpoints import load_training_state
from mars_titan.training.selection import AWAIT, COMMON_EPOCH, JOINT_PLATEAU
from tests.training.test_financial_run import shared as shared
from tests.training.test_financial_run import trainer as titans_trainer

NATIVE = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
JOINT = dict(metric="session_mae", patience=2, min_delta=0.0, stopping=JOINT_PLATEAU)
EPOCHS = 6
# La época 0 es el estado inicial y hay seis épocas. La meseta llega en la 3 y mejora en la 4.
EARLY = [0.5, 0.4, 0.45, 0.46, 0.3, 0.35, 0.36]
# Esta meseta llega en la 4, así que la época común del grupo es la mayor de las dos.
LATE = [0.5, 0.45, 0.4, 0.42, 0.43, 0.2, 0.21]


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


class Script:
    """Devuelve una validación escrita de antemano que falla si se pide una época de más."""

    def __init__(self, values):
        self.values, self.calls = list(values), 0

    def __call__(self, source, *, stop=None, **_):
        self.calls += 1
        return dict(session_mae=self.values.pop(0), labels=1)


def scripted(engine, monkeypatch, values):
    script = Script(values)

    def train_pass(run, cursor, stop, save):
        engine.global_step += 1
        return dict(updates=1)

    monkeypatch.setattr(engine, "evaluate", script)
    monkeypatch.setattr(engine, "_train_pass", train_pass)
    if hasattr(engine, "_predict"):
        monkeypatch.setattr(engine, "_predict", lambda stop: {})
    return script


def titans(streams, output, monkeypatch, values, selection=JOINT):
    engine = titans_trainer(streams, output, epochs=EPOCHS, selection=selection)
    return engine, scripted(engine, monkeypatch, values)


def candidate(streams, output, monkeypatch, values, selection=JOINT):
    from tests.training.test_candidate_run import trainer

    engine = trainer(streams, output, epochs=EPOCHS, selection=selection)
    return engine, scripted(engine, monkeypatch, values)


def readout(streams, output, monkeypatch, values, selection=JOINT):
    from mars_titan.memory.native_backend import load_native
    from tests.training.test_mars_titan_run import build, recipe

    native = load_native(os.environ["MARS_TITAN_EPISODIC_NATIVE"])
    engine = build(streams, output, native, plan=recipe(epochs=EPOCHS, selection=selection))
    return engine, scripted(engine, monkeypatch, values)


TRAINERS = [
    pytest.param(titans, id="titans_mac"),
    pytest.param(candidate, id="gru_episodic", marks=NATIVE),
    pytest.param(readout, id="mars_titan_readout", marks=NATIVE),
]


def recorded(output):
    return json.loads((output / "run.json").read_text())


@pytest.mark.usefixtures("learning_doubles")
@pytest.mark.parametrize("build", TRAINERS)
def test_fit_waits_at_its_plateau_and_continues_to_the_joint_epoch(
    build, shared, tmp_path, monkeypatch
):
    _, streams = shared
    output = tmp_path / "run"
    engine, script = build(streams, output, monkeypatch, EARLY)
    report = engine.run()
    assert report["status"] == AWAIT and report["individual_stop_epoch"] == 3
    assert (report["plateau_epoch"], report["best_epoch"], report["last_epoch"]) == (3, 1, 3)
    assert engine.global_step == 3 and script.calls == 4
    # El estado en espera está confirmado en disco con su informe, su cursor y su checkpoint.
    stored = recorded(output)
    assert stored["status"] == AWAIT and stored["cursor"]["phase"] == "train"
    assert stored["cursor"]["epoch"] == 3 and "joint_stop_epoch" not in stored
    state = load_training_state(output / "checkpoints", expected_identity=engine.identity)
    assert state["selection"]["plateau_epoch"] == 3 and state["global_step"] == 3

    # Repetir la meseta no evalúa ni actualiza nada más.
    again, script = build(streams, output, monkeypatch, [])
    report = again.run(resume=True)
    assert report["status"] == AWAIT and script.calls == 0 and again.global_step == 3

    # La época común no puede ser anterior a la meseta ni posterior al máximo.
    for joint_epoch in (2, EPOCHS + 1):
        broken, _ = build(streams, output, monkeypatch, [])
        with pytest.raises(ValueError, match="época conjunta"):
            broken.run(resume=True, joint_epoch=joint_epoch)

    resumed, script = build(streams, output, monkeypatch, EARLY[4:6])
    report = resumed.run(resume=True, joint_epoch=5)
    assert report["status"] == "completed" and report["joint_stop_epoch"] == 5
    assert resumed.global_step == 5 and script.calls == 2
    assert [entry["epoch"] for entry in report["history"]] == list(range(6))
    # El mejor estado se elige con la misma regla sobre todas las épocas recorridas.
    assert (report["best_epoch"], report["best_score"]) == (4, 0.3)
    assert report["plateau_epoch"] == 3 and report["stopped_early"] is True

    # Una ejecución terminada solo se confirma con su misma época común.
    done, script = build(streams, output, monkeypatch, [])
    assert done.run(resume=True, joint_epoch=5)["status"] == "completed"
    assert script.calls == 0
    for other in (4, None):
        later, _ = build(streams, output, monkeypatch, [])
        with pytest.raises(ValueError, match="otra época conjunta"):
            later.run(resume=True, joint_epoch=other)


@pytest.mark.usefixtures("learning_doubles")
@pytest.mark.parametrize("build", TRAINERS)
def test_paired_fits_apply_the_same_updates_up_to_the_latest_plateau(
    build, shared, tmp_path, monkeypatch
):
    _, streams = shared
    stops, engines = {}, {}
    for name, values in (("early", EARLY), ("late", LATE)):
        engine, _ = build(streams, tmp_path / name, monkeypatch, values)
        stops[name] = engine.run()["individual_stop_epoch"]
    assert stops == dict(early=3, late=4)
    common = max(stops.values())
    for name, values in (("early", EARLY), ("late", LATE)):
        engine, _ = build(streams, tmp_path / name, monkeypatch, values[stops[name] + 1 :])
        engines[name] = (engine, engine.run(resume=True, joint_epoch=common))
    assert {engine.global_step for engine, _ in engines.values()} == {common}
    for _, report in engines.values():
        assert report["status"] == "completed" and report["joint_stop_epoch"] == common
        assert len(report["history"]) == common + 1
    assert engines["early"][1]["best_epoch"] == 4 and engines["late"][1]["best_epoch"] == 2


@pytest.mark.usefixtures("learning_doubles")
@pytest.mark.parametrize("build", TRAINERS)
def test_fit_without_plateau_waits_at_the_maximum(build, shared, tmp_path, monkeypatch):
    _, streams = shared
    improving = [0.5 - 0.05 * epoch for epoch in range(EPOCHS + 1)]
    engine, _ = build(streams, tmp_path / "run", monkeypatch, improving)
    report = engine.run()
    assert report["status"] == AWAIT and report["individual_stop_epoch"] == EPOCHS
    assert report["plateau_epoch"] is None and engine.global_step == EPOCHS
    resumed, script = build(streams, tmp_path / "run", monkeypatch, [])
    report = resumed.run(resume=True, joint_epoch=EPOCHS)
    assert report["status"] == "completed" and script.calls == 0
    assert report["best_epoch"] == EPOCHS and report["stopped_early"] is False


@pytest.mark.usefixtures("learning_doubles")
@pytest.mark.parametrize("build", TRAINERS)
def test_other_modes_refuse_a_joint_epoch(build, shared, tmp_path, monkeypatch):
    _, streams = shared
    fixed = dict(JOINT, stopping="fixed_budget")
    engine, _ = build(streams, tmp_path / "fixed", monkeypatch, EARLY, selection=fixed)
    report = engine.run()
    assert report["status"] == "completed" and report["plateau_epoch"] == 3
    assert engine.global_step == EPOCHS and "joint_stop_epoch" not in report
    again, _ = build(streams, tmp_path / "fixed", monkeypatch, [], selection=fixed)
    with pytest.raises(ValueError, match="época conjunta"):
        again.run(resume=True, joint_epoch=EPOCHS)
    # Un ajuste conjunto nuevo tampoco recibe una época común antes de su meseta.
    fresh, script = build(streams, tmp_path / "fresh", monkeypatch, EARLY)
    with pytest.raises(ValueError, match="época conjunta"):
        fresh.run(joint_epoch=3)
    assert script.calls == 0


COMMON = dict(JOINT, stopping=COMMON_EPOCH)
# IC de validación de las épocas 1 a 6, que la puntuación doble entrega en orden.
IC = [0.01, 0.02, 0.05, 0.03, 0.02, 0.01]


class ScriptedIC:
    """Puntuación por época escrita de antemano que registra las épocas que recibe."""

    def __init__(self, values):
        self.values, self.epochs = list(values), []

    def rows(self):
        return []

    def score(self, epoch, rows, metrics):
        assert rows == [] and "session_mae" in metrics
        self.epochs.append(epoch)
        value = self.values.pop(0)
        return dict(
            session_ic=value,
            session_ic_values=[value],
            session_ic_defined=[True],
            session_time=[0],
            cohort_sha256="cohort",
        )


COMMON_TRAINERS = [
    pytest.param(titans, id="titans_mac"),
    pytest.param(readout, id="mars_titan_readout", marks=NATIVE),
]


def common_build(build, streams, output, monkeypatch, values, scores):
    engine, script = build(streams, output, monkeypatch, values, selection=COMMON)
    engine.epoch_scorer = ScriptedIC(scores)
    return engine, script, engine.epoch_scorer


@pytest.mark.usefixtures("learning_doubles")
@pytest.mark.parametrize("build", COMMON_TRAINERS)
def test_common_epoch_fits_the_budget_pins_each_epoch_and_emits_the_chosen_one(
    build, shared, tmp_path, monkeypatch
):
    _, streams = shared
    output = tmp_path / "run"
    engine, script, scorer = common_build(build, streams, output, monkeypatch, EARLY, IC)
    report = engine.run()
    # Recorre todo el presupuesto aunque el MAE llegue a una meseta en la época 3.
    assert report["status"] == AWAIT and report["individual_stop_epoch"] == EPOCHS
    assert report["plateau_epoch"] is None and engine.global_step == EPOCHS
    assert script.calls == EPOCHS + 1 and scorer.epochs == list(range(1, EPOCHS + 1))
    assert sorted(report["epoch_states"], key=int) == [str(e) for e in range(1, EPOCHS + 1)]
    assert [entry["validation"].get("session_ic") for entry in report["history"]] == [None, *IC]
    for epoch, reference in report["epoch_states"].items():
        state = load_training_state(
            output / "checkpoints",
            expected_identity=engine.identity,
            selection="pinned",
            expected_sha256=reference["sha256"],
        )
        assert state["global_step"] == int(epoch)

    # Sin época común no avanza, y una época fuera del presupuesto no se acepta.
    again, script, _ = common_build(build, streams, output, monkeypatch, [], [])
    assert again.run(resume=True)["status"] == AWAIT and script.calls == 0
    for broken in (0, EPOCHS + 1):
        refused, _, _ = common_build(build, streams, output, monkeypatch, [], [])
        with pytest.raises(ValueError, match="época común"):
            refused.run(resume=True, joint_epoch=broken)

    done, script, _ = common_build(build, streams, output, monkeypatch, [], [])
    report = done.run(resume=True, joint_epoch=3)
    assert report["status"] == "completed" and script.calls == 0
    # Emite el estado de la época común, no el mejor por MAE, que queda como sensibilidad.
    assert (report["best_epoch"], report["best_score"]) == (3, EARLY[3])
    assert report["common_epoch"] == 3 and report["common_epoch_session_ic"] == IC[2]
    assert report["mae_selection"] == dict(best_epoch=4, best_score=EARLY[4])
    assert report["best_checkpoint"] == report["epoch_states"]["3"]
    chosen = load_training_state(
        output / "checkpoints",
        expected_identity=engine.identity,
        selection="best",
        expected_sha256=report["best_checkpoint"]["sha256"],
    )
    assert chosen["global_step"] == 3 and done.global_step == EPOCHS
    # Una ejecución terminada solo se confirma con su misma época común.
    for other in (4, None):
        later, _, _ = common_build(build, streams, output, monkeypatch, [], [])
        with pytest.raises(ValueError, match="otra época conjunta"):
            later.run(resume=True, joint_epoch=other)


@pytest.mark.usefixtures("learning_doubles")
@pytest.mark.parametrize("build", COMMON_TRAINERS)
def test_common_epoch_needs_a_scorer(build, shared, tmp_path, monkeypatch):
    _, streams = shared
    engine, script = build(streams, tmp_path / "run", monkeypatch, EARLY, selection=COMMON)
    with pytest.raises(ValueError, match="puntuar cada validación"):
        engine.run()
    assert script.calls == 0 and not (tmp_path / "run").exists()


def test_the_candidate_recipe_refuses_the_common_epoch():
    from mars_titan.training.candidate_run import CandidateRecipe

    # La GRU candidata no conserva el estado de cada época, así que no admite el modo.
    with pytest.raises(ValueError, match="solo la admiten Titans-MAC"):
        CandidateRecipe(epochs=EPOCHS, selection=COMMON)
