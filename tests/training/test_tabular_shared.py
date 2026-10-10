"""Estado tabular compartido por ventana: claves, liberación y concurrencia, sin CUDA.

Las estadísticas y las matrices se sustituyen por objetos que registran su cierre. Así se
comprueba cuándo se reutilizan, cuándo se reconstruyen y que nada queda retenido.
"""

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mars_titan.training import masked_campaign, tabular_search
from mars_titan.training.external_corpus import SharedWindow, _Paused, _TrainRows
from mars_titan.training.tabular_corpus import _SharedStatistics


class Closing:
    def __init__(self, directory, log):
        self.directory, self.log = Path(directory), log
        self.directory.mkdir(parents=True)
        (self.directory / "pages").write_bytes(b"page")
        self.matrix = SimpleNamespace(disk_bytes=lambda: 4)
        self.rows = SimpleNamespace(expected=2)

    def close(self):
        self.log.append(self.directory.name)


def held(lock):
    """Retener un candado desde otro hilo hasta que la prueba lo suelte."""
    taken, release = threading.Event(), threading.Event()

    def hold():
        with lock:
            taken.set()
            release.wait(5)

    thread = threading.Thread(target=hold)
    thread.start()
    taken.wait(5)
    return release, thread


def test_statistics_are_computed_once_per_key_and_cleared_without_waiting():
    shared, calls = _SharedStatistics(), []

    def compute(value):
        return lambda: calls.append(value) or value

    assert shared.get("a", compute(1)) == (1, True)
    assert shared.get("a", compute(2)) == (1, False)
    assert shared.get("b", compute(3)) == (3, True)
    assert calls == [1, 3]
    release, thread = held(shared.lock)
    assert shared.clear(blocking=False) is False and shared.key == "b"
    release.set()
    thread.join()
    assert shared.clear(blocking=False) is True and shared.key is None


def test_window_reuses_its_matrix_until_the_key_changes(tmp_path):
    window, log, built = SharedWindow(), [], []
    directory = tmp_path / "shared"

    def build(path):
        built.append(path)
        return Closing(path, log)

    with window.training_matrix("a", directory, build) as (first, constructed):
        assert constructed
    with window.training_matrix("a", directory, build) as (again, constructed):
        assert again is first and not constructed
    assert window.reclaimable_disk_bytes() == 4 and window.resident_bytes() == 34
    with window.training_matrix("b", directory, build) as (second, constructed):
        assert constructed and second is not first
    assert log == ["shared"] and len(built) == 2 and window.constructions == 2
    assert window.release() is True
    assert log == ["shared", "shared"] and window.matrix is None


def test_stale_pages_are_removed_before_building_and_after_a_failed_build(tmp_path):
    window, log = SharedWindow(), []
    directory = tmp_path / "shared"
    directory.mkdir()
    (directory / "old-pages").write_bytes(b"x" * 10)

    def failing(path):
        assert not (path / "old-pages").exists()
        path.mkdir()
        (path / "partial").write_bytes(b"y")
        raise RuntimeError("construcción interrumpida")

    with pytest.raises(RuntimeError, match="interrumpida"):
        with window.training_matrix("a", directory, failing):
            pass
    assert not directory.exists() and window.matrix is None and window.constructions == 0
    with window.training_matrix("a", directory, lambda path: Closing(path, log)):
        pass
    assert window.matrix_key == "a"


def test_another_process_waits_and_never_removes_reserved_pages(tmp_path):
    """Dos ventanas abren descriptores distintos, como dos procesos para `flock`."""
    first, second, log = SharedWindow(), SharedWindow(), []
    directory = tmp_path / "jobs" / ".shared-matrix"
    with first.training_matrix("a", directory, lambda path: Closing(path, log)):
        pass

    def unexpected(path):
        raise AssertionError("no debe construir mientras otro proceso reserva el directorio")

    with pytest.raises(_Paused):
        with second.training_matrix("a", directory, unexpected, stopped=lambda: True):
            pass
    assert (directory / "pages").is_file() and second.matrix is None and second.handle is None
    first.release()
    # Las páginas que deja un proceso terminado se borran antes de construir.
    with second.training_matrix("a", directory, lambda path: Closing(path, log)) as (_, built):
        assert built and second.handle is not None
    assert (tmp_path / "jobs" / ".shared-matrix.lock").is_file()
    second.release()
    assert second.handle is None


def test_release_without_blocking_does_not_wait_for_a_running_fit(tmp_path):
    window, log = SharedWindow(), []
    with window.training_matrix("a", tmp_path / "shared", lambda p: Closing(p, log)):
        pass
    release, thread = held(window.lock)
    assert window.release(blocking=False) is False and not log
    release.set()
    thread.join()
    assert window.release(blocking=False) is True and log == ["shared"]


def test_validation_is_built_once_per_key():
    window, built = SharedWindow(), []

    class Resident:
        bytes = 7

        def release(self):
            built.append("released")

    def build():
        built.append("built")
        return Resident()

    first = window.resident_validation("a", build)
    assert window.resident_validation("a", build) is first
    window.resident_validation("b", build)
    assert built == ["built", "released", "built"]


def test_row_keys_record_only_the_first_complete_pass():
    rows = _TrainRows(4)
    batches = [
        dict(
            market=["US", "CN"],
            prediction_at=np.array([1, 2], dtype="datetime64[us]"),
            target=np.array([0.5, -0.5]),
        ),
        dict(
            market=["CN", "US"],
            prediction_at=np.array([3, 4], dtype="datetime64[us]"),
            target=np.array([1.5, 2.5]),
        ),
    ]
    assert rows.start()
    rows.add(batches[0])
    # Una pasada abortada no deja claves parciales: la siguiente vuelve a empezar.
    assert rows.start()
    for batch in batches:
        rows.add(batch)
    rows.finish()
    assert not rows.start()
    np.testing.assert_array_equal(rows.china, [False, True, True, False])
    np.testing.assert_array_equal(rows.moments, [1, 2, 3, 4])
    np.testing.assert_array_equal(rows.target, [0.5, -0.5, 1.5, 2.5])
    short = _TrainRows(5)
    short.start()
    short.add(batches[0])
    with pytest.raises(ValueError, match="población"):
        short.finish()


def test_the_grid_keeps_equal_bins_together_and_the_same_cases():
    _, cases, _ = tabular_search._configuration(
        Path("configs/baselines/tabular-historical-masked.json")
    )
    boosting = [case["parameters"] for case in cases if case["kind"] == "xgboost"]
    bins = [p["max_bin"] for p in boosting]
    assert bins == sorted(bins, key=bins.index) and len(bins) == 12
    # Cada max_bin aparece en un único tramo seguido: tres construcciones en la búsqueda.
    assert sum(a != b for a, b in zip(bins, bins[1:], strict=False)) == 2
    assert {(p["max_depth"], p["max_bin"], p["learning_rate"]) for p in boosting} == {
        (d, b, r) for d in (3, 6) for b in (64, 128, 256) for r in (0.03, 0.1)
    }


def test_campaign_releases_the_other_tabular_state_before_each_job(monkeypatch):
    calls = []
    from mars_titan.training import external_corpus, tabular_corpus

    monkeypatch.setattr(
        external_corpus.SHARED, "release", lambda blocking=True: calls.append(("xgb", blocking))
    )
    monkeypatch.setattr(
        tabular_corpus.RIDGE_STATISTICS,
        "clear",
        lambda blocking=True: calls.append(("ridge", blocking)),
    )
    both = [("xgb", False), ("ridge", False)]
    for key, expected in (
        (("neural", "fit"), both),
        (("ridge", "fit"), [("xgb", False)]),
        (("xgboost", "fit"), [("ridge", False)]),
        # Los traslados no usan la matriz ni la Gram: liberan las dos.
        (("xgboost", "carry"), both),
        (("ridge", "carry"), both),
    ):
        calls.clear()
        run = masked_campaign._releasing_tabular(lambda job: "done", *key)
        assert run(None) == "done" and calls == expected
    assert masked_campaign.FIT == "fit" and masked_campaign.CARRY == "carry"


def test_xgboost_jobs_share_one_directory_next_to_the_jobs(monkeypatch, tmp_path):
    from mars_titan.training import external_corpus

    captured = {}

    def reference(view, folder, **options):
        captured.update(options)
        return dict(status="completed")

    monkeypatch.setattr(external_corpus, "run_external_reference", reference)
    job_id = "US+CN/fold-003/xgboost/search-xgb-d3-b64-lr0.03-s42"
    folder = tmp_path / "campaign" / "jobs" / job_id / "attempt-0001"
    run = SimpleNamespace(
        job=dict(id=job_id), folder=folder, view=tmp_path / "view.json", stop=None, case={}
    )
    masked_campaign._xgboost_fit(run)
    assert captured["shared_directory"] == tmp_path / "campaign" / "jobs" / ".shared-matrix"
    run.job = dict(id="extra/" + job_id)
    with pytest.raises(ValueError, match="directorio de trabajos"):
        masked_campaign._xgboost_fit(run)


def test_admission_credits_the_shared_matrix_that_the_job_reuses_or_frees(monkeypatch):
    from mars_titan.training import external_corpus

    requested = []
    guard = SimpleNamespace(admits=lambda need, key: requested.append(need) or True)
    footprint = dict(retained_bytes=10, transient_bytes=90)
    monkeypatch.setattr(masked_campaign, "job_footprint", lambda *args: footprint)
    state = SimpleNamespace(disk=(guard, {"US": {"fold-000": {}}}, {}))
    # La admisión y la comprobación previa de las ranuras comparten el cálculo de la necesidad.
    state._disk_need = lambda job: masked_campaign._Campaign._disk_need(state, job)
    job = dict(id="US/fold-000/xgboost/search", scope="US", window="fold-000")
    for reclaimable, need in ((0, 100), (60, 40), (500, 0)):
        monkeypatch.setattr(
            external_corpus.SHARED, "reclaimable_disk_bytes", lambda r=reclaimable: r
        )
        masked_campaign._Campaign.admit(state, job)
        assert requested[-1] == need


def test_stale_pages_are_discarded_only_when_nobody_reserves_them(tmp_path):
    owner, other, log = SharedWindow(), SharedWindow(), []
    directory = tmp_path / "jobs" / ".shared-matrix"
    with owner.training_matrix("a", directory, lambda path: Closing(path, log)):
        pass
    # Reservado por otro proceso, o por el propio: no se toca.
    assert other.discard_stale(directory) == 0 and owner.discard_stale(directory) == 0
    assert (directory / "pages").is_file()
    owner.release()
    assert other.discard_stale(directory) == 4 and not directory.exists()
    assert other.discard_stale(directory) == 0
