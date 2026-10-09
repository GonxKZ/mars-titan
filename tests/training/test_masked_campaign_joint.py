"""Parada conjunta en la campaña con máscaras, con dobles que no ajustan ningún modelo.

Dos referencias neuronales forman un grupo. El doble de la meseta devuelve una parada
individual escrita de antemano y el de la continuación comprueba la época común recibida.
Se comprueban orden, carpetas, recibos de meseta, identidades, reanudación y rechazos.
"""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from mars_titan.training import masked_campaign as engine
from mars_titan.training.campaign_plan import (
    GROUP_EPOCH,
    JOINT,
    PLATEAU,
    count_jobs,
    load_campaign,
    plan_campaign,
)
from mars_titan.training.selection import AWAIT, JOINT_PLATEAU
from tests.training.test_masked_campaign import Recorder, doubles, run, write_campaign
from tests.training.test_masked_campaign import prepared as prepared

pytestmark = pytest.mark.usefixtures("learning_doubles")

ARMS = ("gru", "lstm", "ridge", "xgboost")
EARLY = dict(
    stopping=JOINT_PLATEAU,
    patience=5,
    min_delta=1e-05,
    minimum_epochs=5,
    max_epochs=30,
    group_epoch=GROUP_EPOCH,
    groups=dict(references=["gru", "lstm"]),
)
# Parada individual de cada meseta por brazo y posición del caso, o finalista.
STOPS = {
    ("gru", 0): 7,
    ("gru", 1): 9,
    ("lstm", 0): 12,
    ("lstm", 1): 4,
    ("gru", "selected"): 6,
    ("lstm", "selected"): 8,
}
COMMON = {0: 12, 1: 9, "selected": 8}


def slot(job):
    if job["stage"] != "search":
        return "selected"
    return int(job["candidate"].rsplit("-", 1)[1]) // 10


class JointRecorder(Recorder):
    """Meseta con parada escrita de antemano y continuación con su época común."""

    def __init__(self, *, plateau_status=AWAIT, final_status=None, **options):
        super().__init__(**options)
        self.plateau_status, self.final_status = plateau_status, final_status
        self.joint = {}

    def __call__(self, run):
        job = run.job
        if job.get("phase") == PLATEAU:
            assert run.joint_epoch is None
            self.calls.append(dict(id=job["id"], folder=run.folder, joint_epoch=None))
            run.folder.mkdir(parents=True, exist_ok=True)
            report = dict(
                status=self.plateau_status,
                individual_stop_epoch=STOPS[job["arm"], slot(job)],
                final_test_opened=False,
            )
            (run.folder / "run.json").write_text(json.dumps(report))
            return report
        if job.get("phase") == JOINT:
            self.joint[job["id"]] = (run.joint_epoch, run.folder)
            if self.final_status is not None:
                return dict(status=self.final_status, final_test_opened=False)
        return super().__call__(run)


def joint_campaign(folder):
    path = write_campaign(folder, arms=ARMS)
    value = json.loads(path.read_text())
    value["early_stop"] = EARLY
    path.write_text(json.dumps(value))
    return path


def receipt(output, job_id):
    return json.loads((output / "jobs" / job_id / "receipt.json").read_text())


def test_groups_wait_for_every_plateau_and_continue_to_the_latest(prepared, tmp_path):
    campaign = joint_campaign(tmp_path / "config")
    views = {"US": prepared.views["US"]}
    recorder = JointRecorder()
    output = tmp_path / "out"
    summary = run(campaign, views, output, recorder)
    assert summary["status"] == "completed"
    # B sobre US: 7 ventanas con 2 + 2 ajustes de cada referencia, 1 de Ridge y 3 de
    # XGBoost, y 12 trasladadas con 3 + 3 + 1 + 3 traslados. Cada ajuste agrupado tiene
    # además su meseta.
    assert summary["planned"] == dict(
        training_jobs=7 * 12, prediction_jobs=12 * 10, plateau_jobs=7 * 8
    )
    assert summary["completed"] == summary["planned"]
    loaded = load_campaign(campaign)
    jobs = plan_campaign(loaded)
    assert count_jobs(loaded, jobs)["plateau_jobs"] == 56
    order = [call["id"] for call in recorder.calls]
    assert order == [job["id"] for job in jobs]
    finals = [job for job in jobs if job.get("phase") == JOINT]
    assert len(finals) == 56 and set(recorder.joint) == {job["id"] for job in finals}
    for job in finals:
        epoch, folder = recorder.joint[job["id"]]
        # La época común es la mayor de las primeras mesetas del grupo emparejado.
        assert epoch == COMMON[slot(job)]
        plateau = receipt(output, job["plateau"])
        assert plateau["status"] == engine.PLATEAU_STATUS and plateau["predictions"] == {}
        assert plateau["plateau"] == dict(
            status=AWAIT, individual_stop_epoch=STOPS[job["arm"], slot(job)]
        )
        # La continuación reanuda en la carpeta de su meseta y su recibo depende del grupo.
        assert folder == output / plateau["attempt"]
        assert all(order.index(member) < order.index(job["id"]) for member in job["joint_group"])
        final = receipt(output, job["id"])
        joint = final["identity"]["sources"]["joint"]
        assert joint["epoch"] == epoch and joint["rule"] == GROUP_EPOCH
        assert set(joint["members"]) == set(job["joint_group"])
        assert final["attempt"] == plateau["attempt"]
        copy = json.loads((output / plateau["report"]["path"]).read_text())
        assert copy["status"] == AWAIT
    # Los recibos de ventana siguen siendo los del predictor elegido de cada semilla.
    window = output / "windows/US/fold-000/gru/seed-42/US.json"
    assert json.loads(window.read_text())["parent"]["id"].startswith("US/fold-000/gru/search-")

    again = JointRecorder()
    assert run(campaign, views, output, again)["status"] == "completed"
    assert again.calls == []
    copy = output / receipt(output, finals[0]["plateau"])["report"]["path"]
    copy.write_text(copy.read_text() + " ")
    with pytest.raises(ValueError, match="ha cambiado"):
        run(campaign, views, output, JointRecorder())


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (dict(plateau_status="completed"), "no espera la época conjunta"),
        (dict(final_status=AWAIT), "espera una época conjunta que no tiene"),
    ],
)
def test_plateau_and_continuation_must_report_their_own_state(prepared, tmp_path, options, message):
    campaign = joint_campaign(tmp_path / "config")
    with pytest.raises(ValueError, match=message):
        run(campaign, {"US": prepared.views["US"]}, tmp_path / "out", JointRecorder(**options))


def test_joint_epoch_needs_every_plateau_of_the_group(prepared, tmp_path):
    campaign = joint_campaign(tmp_path / "config")
    views = {"US": prepared.views["US"]}
    loaded = load_campaign(campaign)
    jobs = plan_campaign(loaded)
    final = next(job for job in jobs if job.get("phase") == JOINT)
    state = engine._Campaign(
        loaded,
        {"US": engine.scope_views(views["US"], "US", loaded)},
        tmp_path / "out",
        {},
        engine.EXECUTORS,
        SimpleNamespace(requested=False),
        jobs,
    )
    for member in final["joint_group"]:
        state.receipts[member] = dict(
            status=engine.PLATEAU_STATUS, plateau=dict(individual_stop_epoch=3), sha256="0" * 64
        )
    assert state.joint_epoch(final)["epoch"] == 3
    state.receipts[final["joint_group"][-1]] = dict(status="completed", sha256="1" * 64)
    with pytest.raises(ValueError, match="mesetas sin confirmar"):
        state.joint_epoch(final)


class CheckpointingJoint(JointRecorder):
    """Meseta con estados reales de recuperación y continuación que los cuenta al empezar."""

    def __init__(self, **options):
        super().__init__(**options)
        self.found = {}

    def __call__(self, run):
        from tests.training.test_campaign_storage import _states

        if run.job.get("phase") == PLATEAU:
            report = super().__call__(run)
            _states(run.folder / "checkpoints")
            return report
        if run.job.get("phase") == JOINT:
            self.found[run.job["id"]] = len(list((run.folder / "checkpoints").glob("state-*.pt")))
        return super().__call__(run)


def test_storage_release_keeps_the_plateau_states_until_the_continuation(
    prepared, tmp_path, monkeypatch
):
    from mars_titan.training import campaign_storage as storage
    from tests.training.test_campaign_storage import _storage_file

    monkeypatch.setattr(storage, "free_bytes", lambda *_: 10 * 1024**4)
    campaign = joint_campaign(tmp_path / "config")
    recorder = CheckpointingJoint()
    summary = engine.run_campaign(
        campaign,
        {"US": prepared.views["US"]},
        tmp_path / "out",
        executors=doubles(recorder),
        lease=nullcontext,
        stop=SimpleNamespace(requested=False),
        storage=_storage_file(tmp_path),
    )
    assert summary["status"] == "completed"
    # Cada continuación encuentra los estados de recuperación de su meseta y los libera al
    # confirmar su recibo.
    assert recorder.found and set(recorder.found.values()) == {3}
    for job_id, _ in recorder.found.items():
        folder = tmp_path / "out" / receipt(tmp_path / "out", job_id)["attempt"]
        assert len(list((folder / "checkpoints").glob("state-*.pt"))) == 1
        assert (folder / "released.json").is_file()
