"""Huella de disco, liberación al confirmar y guardia de la campaña con máscaras.

Las pruebas usan recuentos y declaraciones pequeñas, estados de PyTorch en CPU sin pasos
de optimizador y un disco libre simulado. La integración reutiliza los dobles de
`test_masked_campaign`, que escriben predicciones nulas sin ajustar ningún modelo.
"""

import copy
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training import campaign_storage as storage
from mars_titan.training import masked_campaign as engine
from mars_titan.training.checkpoints import (
    load_training_state,
    release_recovery_states,
    save_training_state,
)
from tests.training import test_masked_campaign as masked
from tests.training.test_masked_campaign import Recorder, doubles, write_campaign

# Vistas técnicas preparadas una vez por módulo, como en `test_masked_campaign`.
prepared = masked.prepared

DECLARATION = Path("configs/baselines/historical-masked-campaign-storage.json")
COUNTS = dict(train=1000, validation=100, calibration=50, evaluation=200, flows=10)


def declared(**changes):
    value = storage.load_storage(DECLARATION)
    for key, item in changes.items():
        value[key] = item
    return value


def job(model="neural", kind="fit", **fields):
    return dict(
        dict(id=f"US/fold-000/{model}/x", scope="US", window="fold-000", model=model, kind=kind),
        **fields,
    )


def test_the_declared_storage_is_valid_and_rejects_incomplete_changes(tmp_path):
    document = json.loads(DECLARATION.read_text())
    assert storage.load_storage(DECLARATION)["margin_bytes"] == document["margin_bytes"]
    mutations = [
        lambda d: d.update(margin_bytes=1024**3 - 1),
        lambda d: d.update(kind="other"),
        lambda d: d.update(release_on_confirmation=1),
        lambda d: d["prediction_row_bytes"].pop("titans"),
        lambda d: d["prediction_row_bytes"]["neural"].update(evaluation=0),
        lambda d: d["state_bytes"].pop("cm_v1"),
        lambda d: d["retained_states"].update(neural=-1),
        lambda d: d["xgboost"].update(features=0),
        lambda d: d["index"].update(warmup_partition="train"),
        lambda d: d.update(check_seconds=0),
        lambda d: d.update(extra=True),
    ]
    for number, mutate in enumerate(mutations):
        changed = copy.deepcopy(document)
        mutate(changed)
        path = tmp_path / f"{number}.json"
        atomic_json(path, changed)
        with pytest.raises(ValueError):
            storage.load_storage(path)


def test_footprint_separates_what_a_job_keeps_from_what_it_needs_while_running():
    value = declared()
    rows = value["prediction_row_bytes"]["neural"]
    tables = {p: int(COUNTS[p] * rows[p]) for p in storage.HELD_OUT}
    state = value["state_bytes"]["neural"]
    released = storage.job_footprint(job(), COUNTS, value, release=True)
    assert released["retained"] == dict(
        predictions=sum(tables.values()),
        reports=value["job_report_bytes"],
        sessions=value["train_sessions_bytes"],
        states=state,
        indices=0,
    )
    assert released["transient"]["recovery"] == 2 * state
    assert released["transient"]["writing"] == max(tables.values())
    kept = storage.job_footprint(job(), COUNTS, value, release=False)
    assert kept["retained"]["states"] == 3 * state and kept["transient"]["recovery"] == 0
    assert kept["retained_bytes"] - released["retained_bytes"] == 2 * state


def test_indexed_models_keep_their_index_only_without_release():
    value = declared()
    index = value["index"]
    rows = 2 * COUNTS["train"] + sum(2 * COUNTS[p] + COUNTS["evaluation"] for p in storage.HELD_OUT)
    built = int(rows * index["bytes_per_row"])
    build = int(2 * COUNTS["train"] * index["build_bytes_per_row"])
    released = storage.job_footprint(job("titans_mac"), COUNTS, value, release=True)
    assert released["retained"]["indices"] == 0
    assert released["transient"]["indices"] == built + build
    assert released["transient"]["mid_epoch"] == 2 * COUNTS["flows"] * value["flow_state_bytes"]
    kept = storage.job_footprint(job("titans_mac"), COUNTS, value, release=False)
    assert kept["retained"]["indices"] == built and kept["transient"]["indices"] == build
    gru = storage.job_footprint(job("episodic_gru"), COUNTS, value, release=False)
    no_warmup = 2 * COUNTS["train"] + sum(2 * COUNTS[p] for p in storage.HELD_OUT)
    assert gru["retained"]["indices"] == int(no_warmup * index["bytes_per_row"])


def test_xgboost_keeps_its_models_and_needs_its_pages_while_running():
    value = declared()
    boosting = value["xgboost"]
    search = storage.job_footprint(
        job("xgboost", case=dict(max_bin=64)), COUNTS, value, release=True
    )
    # Páginas densas de 6 bits con 64 contenedores. La validación no ocupa disco.
    assert search["transient"]["cache"] == -(-COUNTS["train"] * boosting["features"] * 6 // 8)
    assert search["retained"]["states"] == 3 * value["state_bytes"]["xgboost"]
    finalist = storage.job_footprint(job("xgboost", case=None), COUNTS, value)
    assert finalist["transient"]["cache"] == COUNTS["train"] * boosting["features"]
    larger = dict(COUNTS, validation=10**9)
    cache = storage.job_footprint(job("xgboost"), larger, value)["transient"]["cache"]
    assert cache == finalist["transient"]["cache"]


def test_carry_jobs_write_only_calibration_and_evaluation():
    value = declared()
    carry = storage.job_footprint(job("xgboost", kind="carry"), COUNTS, value)
    rows = value["prediction_row_bytes"]["xgboost"]
    assert carry["retained"]["predictions"] == sum(
        int(COUNTS[p] * rows[p]) for p in ("calibration", "evaluation")
    )
    assert carry["retained"]["states"] == 0 and carry["transient"]["cache"] == 0


def test_plan_peak_follows_the_plan_order_and_skips_confirmed_jobs():
    value = declared()
    jobs = [job("neural", id=f"US/fold-000/a/{i}") for i in range(3)]
    jobs.append(job("xgboost", id="US/fold-000/b/x", case=dict(max_bin=256)))
    counts = {"US": {"fold-000": COUNTS}}
    each = [storage.job_footprint(j, COUNTS, value) for j in jobs]
    result = storage.plan_peak(jobs, counts, value)
    moments, cumulative = [], 0
    for footprint in each:
        moments.append(cumulative + footprint["retained_bytes"] + footprint["transient_bytes"])
        cumulative += footprint["retained_bytes"]
    assert result["peak_bytes"] == max(moments)
    assert result["peak_job"] == jobs[moments.index(max(moments))]["id"]
    assert result["retained_bytes"] == cumulative
    done = storage.plan_peak(jobs, counts, value, done={jobs[0]["id"], jobs[1]["id"]})
    assert done["retained_bytes"] == cumulative - 2 * each[0]["retained_bytes"]


class Disk:
    def __init__(self, *values):
        self.values, self.calls = list(values), 0

    def __call__(self, _):
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return SimpleNamespace(free=value)


def test_guard_rejects_a_launch_whose_peak_does_not_fit_over_the_margin(tmp_path):
    guard = storage.DiskGuard(tmp_path, 100, usage=Disk(1000))
    projection = dict(peak_bytes=900, peak_job="x", retained_bytes=0)
    assert guard.require_launch(projection)["free_bytes"] == 1000
    with pytest.raises(storage.DiskBudgetError, match="x"):
        guard.require_launch(dict(projection, peak_bytes=901))
    assert guard.admits(900) and not guard.admits(901)


def test_guard_reserves_admitted_jobs_until_they_settle(tmp_path):
    guard = storage.DiskGuard(tmp_path, 100, usage=Disk(1000))
    # Dos trabajos simultáneos no pueden contar con el mismo espacio libre.
    assert guard.admits(500, "a")
    assert not guard.admits(401, "b") and guard.admits(400, "b")
    assert guard.state()["reserved_bytes"] == 900
    assert not guard.admits(1, "c")
    guard.settle("a")
    guard.settle("missing")
    assert guard.admits(500, "c") and guard.state()["reserved_bytes"] == 900
    # Sin clave, la consulta no reserva nada.
    guard.settle("b")
    guard.settle("c")
    assert guard.admits(900) and guard.state()["reserved_bytes"] == 0


def test_guard_watch_requests_a_stop_below_the_margin_with_bounded_queries(tmp_path):
    clock = SimpleNamespace(now=0.0)
    disk = Disk(500, 50, 50)
    guard = storage.DiskGuard(
        tmp_path / "missing/child", 100, check_seconds=10, usage=disk, clock=lambda: clock.now
    )
    base = SimpleNamespace(requested=False)
    watched = guard.watch(base)
    assert not watched.requested and disk.calls == 1
    clock.now = 5.0
    assert not watched.requested and disk.calls == 1
    clock.now = 10.0
    assert watched.requested and disk.calls == 2
    assert guard.state() == dict(
        free_bytes=50, margin_bytes=100, below_margin=True, reserved_bytes=0
    )
    calm = storage.DiskGuard(tmp_path, 100, usage=Disk(10**9)).watch(base)
    base.requested = True
    assert calm.requested


def _states(directory, *, best_at=1, pin_last=True, count=4):
    identity = dict(run="prueba")
    paths = []
    for step in range(count):
        state = dict(global_step=step, model=dict(w=torch.full((4,), float(step))))
        paths.append(
            save_training_state(
                directory,
                state,
                identity=identity,
                best=step == best_at,
                pin=pin_last and step == count - 1,
            )
        )
    return identity, paths


def test_release_keeps_only_the_selected_state_and_rewrites_the_index_first(tmp_path):
    directory = tmp_path / "checkpoints"
    identity, paths = _states(directory)
    assert len(list(directory.glob("state-*.pt"))) == 3
    freed = release_recovery_states(directory)
    assert freed > 0 and [p.exists() for p in paths] == [False, True, False, False]
    index = json.loads((directory / "latest.json").read_text())
    assert index["latest"] == [index["best"]] and index["pinned"] == []
    for selection in ("best", "latest"):
        state = load_training_state(directory, expected_identity=identity, selection=selection)
        assert state["global_step"] == 1
    assert release_recovery_states(directory) == 0


def test_release_without_selection_keeps_the_last_confirmed_state(tmp_path):
    directory = tmp_path / "checkpoints"
    identity, paths = _states(directory, best_at=None, pin_last=False, count=3)
    release_recovery_states(directory)
    assert [p.exists() for p in paths] == [False, False, True]
    state = load_training_state(directory, expected_identity=identity)
    assert state["global_step"] == 2


def test_release_refuses_to_touch_a_directory_without_an_intact_choice(tmp_path):
    directory = tmp_path / "checkpoints"
    _, paths = _states(directory)
    paths[1].write_bytes(b"roto")
    before = sorted(p.name for p in directory.iterdir())
    with pytest.raises(ValueError, match="íntegro"):
        release_recovery_states(directory)
    assert sorted(p.name for p in directory.iterdir()) == before
    with pytest.raises(ValueError, match="identidad"):
        release_recovery_states(tmp_path / "empty")


def test_release_confirmed_removes_indices_and_recovery_states_once(tmp_path):
    folder = tmp_path / "attempt-0001"
    _states(folder / "fit/checkpoints")
    (folder / "indices/train-abc").mkdir(parents=True)
    (folder / "indices/train-abc/events.parquet").write_bytes(b"x" * 8192)
    (folder / "run.json").write_text("{}")
    released = storage.release_confirmed(folder, "titans_mac")
    assert released["indices"] > 0 and released["recovery_states"] > 0
    assert not (folder / "indices").exists()
    assert len(list((folder / "fit/checkpoints").glob("state-*.pt"))) == 1
    record = json.loads((folder / "released.json").read_text())
    assert record == released
    assert storage.release_confirmed(folder, "titans_mac") == dict(recovery_states=0)
    assert json.loads((folder / "released.json").read_text()) == record
    neural = tmp_path / "neural"
    (neural / "indices").mkdir(parents=True)
    storage.release_confirmed(neural, "neural")
    assert (neural / "indices").is_dir()


def test_release_confirmed_removes_only_stale_xgboost_caches(tmp_path):
    folder = tmp_path / "attempt-0001"
    (folder / "external-abc/pages").mkdir(parents=True)
    (folder / "external-abc/pages/p.bin").write_bytes(b"x" * 4096)
    (folder / "checkpoints").mkdir()
    model = folder / "checkpoints/attempt-0001-round-0010.ubj"
    model.write_bytes(b"modelo")
    released = storage.release_confirmed(folder, "xgboost")
    assert released["external_caches"] > 0 and model.exists()
    assert not (folder / "external-abc").exists()


def boosters(folder, *, selected_round=10, rounds=(10, 14), status="completed"):
    """Intento terminado de XGBoost con su elegido y sus boosters de recuperación."""
    (folder / "checkpoints").mkdir(parents=True)
    records = {}
    for count in sorted({selected_round, *rounds}):
        path = folder / f"checkpoints/attempt-0001-round-{count:04d}.ubj"
        path.write_bytes(b"booster" * count)
        records[count] = dict(path=str(path.relative_to(folder)), sha256=sha256(path))
    report = dict(
        status=status,
        checkpoint=records[selected_round],
        recovery_checkpoint=records[rounds[-1]],
        recovery_checkpoints=[records[count] for count in rounds],
    )
    atomic_json(folder / "run.json", report)
    return records


def test_release_confirmed_keeps_the_selected_booster_and_drops_the_recovery_ones(tmp_path):
    folder = tmp_path / "attempt-0001"
    records = boosters(folder)
    released = storage.release_confirmed(folder, "xgboost")
    assert released["recovery_boosters"] == len(b"booster") * 14
    assert (folder / records[10]["path"]).is_file()
    assert not (folder / records[14]["path"]).exists()
    assert json.loads((folder / "released.json").read_text())["recovery_boosters"] > 0
    assert storage.release_confirmed(folder, "xgboost") == dict(recovery_boosters=0)
    # Un elegido que coincide con la última ronda se conserva.
    last = tmp_path / "last"
    records = boosters(last, selected_round=14)
    storage.release_confirmed(last, "xgboost")
    assert (last / records[14]["path"]).is_file() and not (last / records[10]["path"]).exists()


def test_boosters_are_not_released_without_a_completed_and_intact_selection(tmp_path):
    running = tmp_path / "running"
    boosters(running, status="running")
    with pytest.raises(ValueError, match="terminado"):
        storage.release_confirmed(running, "xgboost")
    assert len(list((running / "checkpoints").iterdir())) == 2
    broken = tmp_path / "broken"
    records = boosters(broken)
    (broken / records[10]["path"]).write_bytes(b"otro")
    with pytest.raises(ValueError, match="elegido no conserva"):
        storage.release_confirmed(broken, "xgboost")
    assert (broken / records[14]["path"]).is_file()
    changed = tmp_path / "changed"
    records = boosters(changed)
    (changed / records[14]["path"]).write_bytes(b"otro")
    with pytest.raises(ValueError, match="ha cambiado"):
        storage.release_confirmed(changed, "xgboost")
    # Un traslado de XGBoost no tiene boosters de recuperación que liberar.
    carry = tmp_path / "carry"
    carry.mkdir()
    atomic_json(carry / "carry.json", dict(status="completed"))
    assert storage.release_confirmed(carry, "xgboost") == dict(recovery_boosters=0)


# Integración con la campaña: dobles que no ajustan y disco libre simulado.


@pytest.fixture
def doubled(learning_doubles):
    return learning_doubles


def _storage_file(tmp_path, **changes):
    document = json.loads(DECLARATION.read_text())
    document.update(margin_bytes=1024**3, check_seconds=1e-9, **changes)
    path = tmp_path / "storage.json"
    atomic_json(path, document)
    return path


def _run(campaign, views, output, recorder, path, stop=None):
    return engine.run_campaign(
        campaign,
        views,
        output,
        executors=doubles(recorder),
        lease=nullcontext,
        stop=stop or SimpleNamespace(requested=False),
        storage=path,
    )


def test_campaign_refuses_to_launch_when_the_projected_peak_does_not_fit(
    doubled, prepared, tmp_path, monkeypatch
):
    campaign = write_campaign(tmp_path / "config")
    views = {"US": prepared.views["US"]}
    monkeypatch.setattr(storage, "free_bytes", lambda *_: 1024**3 + 1)
    with pytest.raises(storage.DiskBudgetError, match="pico proyectado"):
        _run(campaign, views, tmp_path / "out", Recorder(), _storage_file(tmp_path))
    assert not (tmp_path / "out").exists()


def test_campaign_pauses_before_a_job_that_does_not_fit_and_resumes_it(
    doubled, prepared, tmp_path, monkeypatch
):
    campaign = write_campaign(tmp_path / "config")
    views = {"US": prepared.views["US"]}
    path = _storage_file(tmp_path)
    queries = SimpleNamespace(count=0, limit=6)

    def free(*_):
        queries.count += 1
        return 10 * 1024**4 if queries.count <= queries.limit else 1024**3 + 1

    monkeypatch.setattr(storage, "free_bytes", free)
    first = Recorder()
    summary = _run(campaign, views, tmp_path / "out", first, path)
    assert summary["status"] == "paused"
    assert "necesita" in summary["disk"]["pause"]
    assert summary["disk"]["launch"]["pending_jobs"] == sum(summary["planned"].values())
    done = summary["completed"]["training_jobs"] + summary["completed"]["prediction_jobs"]
    assert 0 < done == len(first.calls)
    queries.limit = 10**9
    second = Recorder()
    resumed = _run(campaign, views, tmp_path / "out", second, path)
    assert resumed["status"] == "completed"
    assert resumed["disk"]["launch"]["pending_jobs"] == sum(summary["planned"].values()) - done
    assert {c["id"] for c in first.calls}.isdisjoint(c["id"] for c in second.calls)


class Watching(Recorder):
    """Doble que consulta la parada en su barrera, como hacen los ejecutores reales."""

    def __init__(self, at, free):
        super().__init__()
        self.at, self.free = at, free

    def __call__(self, run):
        if len(self.calls) + 1 == self.at:
            self.free.value = 1024**3 - 1
        if run.stop.requested:
            self.calls.append(dict(id=run.job["id"], paused=True))
            raise engine.Paused
        return super().__call__(run)


def test_low_disk_during_a_job_pauses_at_its_barrier_without_failing(
    doubled, prepared, tmp_path, monkeypatch
):
    campaign = write_campaign(tmp_path / "config")
    views = {"US": prepared.views["US"]}
    free = SimpleNamespace(value=10 * 1024**4)
    monkeypatch.setattr(storage, "free_bytes", lambda *_: free.value)
    original = storage.DiskGuard.admits
    monkeypatch.setattr(storage.DiskGuard, "admits", lambda self, need, key=None: True)
    recorder = Watching(3, free)
    summary = _run(campaign, views, tmp_path / "out", recorder, _storage_file(tmp_path))
    assert summary["status"] == "paused"
    assert summary["disk"]["guard"]["below_margin"] is True
    assert "margen" in summary["disk"]["pause"]
    assert recorder.calls[-1] == dict(id=recorder.calls[-1]["id"], paused=True)
    monkeypatch.setattr(storage.DiskGuard, "admits", original)
    free.value = 10 * 1024**4
    resumed = _run(campaign, views, tmp_path / "out", Recorder(), _storage_file(tmp_path))
    assert resumed["status"] == "completed"
    # Cada trabajo confirmado devuelve su reserva al terminar.
    assert resumed["disk"]["guard"]["reserved_bytes"] == 0


class Checkpointing(Recorder):
    """Doble que deja estados reales de recuperación junto al elegido, sin ajustar."""

    def __call__(self, run):
        report = super().__call__(run)
        if run.job["model"] == "neural" and run.job["kind"] == "fit":
            _states(run.folder / "checkpoints")
        return report


def test_confirmed_jobs_release_their_recovery_states_after_the_receipt(
    doubled, prepared, tmp_path, monkeypatch
):
    campaign = write_campaign(tmp_path / "config")
    views = {"US": prepared.views["US"]}
    monkeypatch.setattr(storage, "free_bytes", lambda *_: 10 * 1024**4)
    summary = _run(campaign, views, tmp_path / "out", Checkpointing(), _storage_file(tmp_path))
    assert summary["status"] == "completed"
    folders = list((tmp_path / "out/jobs").glob("US/*/gru/*/attempt-*/checkpoints"))
    assert folders
    for folder in folders:
        assert len(list(folder.glob("state-*.pt"))) == 1
        assert (folder.parent / "released.json").is_file()
        assert (folder.parent.parent / "receipt.json").is_file()
    without = _run(
        campaign,
        views,
        tmp_path / "out-kept",
        Checkpointing(),
        _storage_file(tmp_path, release_on_confirmation=False),
    )
    assert without["status"] == "completed"
    kept = list((tmp_path / "out-kept/jobs").glob("US/*/gru/*/attempt-*/checkpoints"))
    assert kept and all(len(list(f.glob("state-*.pt"))) == 3 for f in kept)


def test_the_run_command_requires_a_storage_declaration(capsys):
    with pytest.raises(SystemExit):
        engine.main(["run", "--campaign", "c.json", "--views", "US=v", "--output", "o"])
    assert "--storage" in capsys.readouterr().err
