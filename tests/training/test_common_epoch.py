"""Piezas de la época común de la criba (#746) en las ventanas, con dobles y sin entrenar."""

import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256
from mars_titan.training import titans_walk_forward
from mars_titan.training.checkpoints import StopRequest
from mars_titan.training.common_epoch import EpochValidation, bind_versions, chosen_state
from mars_titan.training.financial_run import Paused

DAY_US = 86_400_000_000


def point_table(seed=0, sessions=3, assets=4):
    """Predicción puntual con el contrato mínimo de `ForecastPanel`."""
    rng = np.random.default_rng(seed)
    times = [1_546_380_000_000_000 + day * DAY_US for day in range(sessions)]
    names = [f"US/A{i}" for i in range(assets)]
    target = rng.standard_normal(sessions * assets)
    return pa.table(
        dict(
            sample_id=[f"{name}/{time}" for time in times for name in names],
            market=["US"] * len(target),
            prediction_at=pa.array(
                [time for time in times for _ in names], type=pa.timestamp("us", tz="UTC")
            ),
            target=target,
            prediction=target + rng.standard_normal(len(target)),
        )
    )


def test_versions_are_bound_once_and_a_resume_keeps_what_was_confirmed():
    report = {}
    bound = bind_versions(report, [2, 4], epochs=4, common=1)
    assert bound == dict(epochs=[2, 4], predictions={}) and report["versions"] is bound
    bound["predictions"]["2"] = dict(calibration="confirmado")
    again = bind_versions(report, [2, 4], epochs=4, common=1)
    assert again is bound and again["predictions"]["2"] == dict(calibration="confirmado")
    for other in ([2], None):
        with pytest.raises(ValueError, match="otras versiones"):
            bind_versions(report, other, epochs=4, common=1)
    for broken in ([1], [4, 2], [2, 2], [5], [0], [True]):
        with pytest.raises(ValueError, match="distintas de la común"):
            bind_versions({}, broken, epochs=4, common=1)
    assert bind_versions({}, None, epochs=4, common=1) == dict(epochs=[], predictions={})


@pytest.mark.parametrize("min_assets", [2, True, 3.0])
def test_the_session_ic_needs_at_least_three_assets(min_assets):
    with pytest.raises(ValueError, match="tres activos"):
        EpochValidation(list, lambda rows, metrics: [], min_assets=min_assets)


def test_an_epoch_only_keeps_its_prediction_when_a_combination_needs_it(tmp_path):
    table = point_table()
    scores = EpochValidation(list, lambda rows, metrics: [table], min_assets=3)
    record = scores.score(1, scores.rows(), {})
    assert "epoch_predictions" not in record and record["session_ic"] is not None
    assert len(record["session_ic_values"]) == len(record["session_time"]) == 3
    kept = EpochValidation(
        list, lambda rows, metrics: [table], min_assets=3, directory=tmp_path / "epochs"
    )
    saved = kept.score(7, kept.rows(), {})["epoch_predictions"]
    path = tmp_path / "epochs" / "validation-epoch-0007.parquet"
    assert saved == dict(path="epochs/validation-epoch-0007.parquet", sha256=sha256(path), rows=12)
    assert pq.read_table(path).equals(table)
    # Sin cuatro activos por sesión ninguna sesión define el IC.
    sparse = EpochValidation(list, lambda rows, metrics: [point_table(assets=3)], min_assets=4)
    assert sparse.score(1, [], {})["session_ic"] is None


def test_the_chosen_state_needs_a_pinned_state_and_a_complete_validation(tmp_path):
    history = [dict(epoch=1, validation=dict(session_mae=0.4, session_ic=0.1))]
    selection = dict(best_epoch=1, best_score=0.4)
    reference = dict(path="checkpoints/state.pt", sha256="0" * 64)
    for report, entries in (
        (dict(epoch_states={}), history),
        (dict(epoch_states={"1": reference}), []),
    ):
        with pytest.raises(ValueError, match="estado fijado ni una validación"):
            chosen_state(tmp_path, {}, report, entries, selection, 1)


class Trainer:
    """Entrenador doble que solo registra qué estado fijado carga."""

    def __init__(self, epochs=4):
        self.recipe = SimpleNamespace(epochs=epochs)
        self.loaded = []

    def load_epoch(self, fit, epoch):
        self.loaded.append(epoch)


def emitter(names, *, extra_rows=0):
    def emit(name):
        names.append(name)
        table = point_table(seed=len(names))
        return [table], table.num_rows + extra_rows, dict(rows=table.num_rows)

    return emit


@pytest.fixture
def released(monkeypatch):
    calls = []

    def release(directory):
        calls.append(directory)
        return 7

    monkeypatch.setattr(titans_walk_forward, "release_recovery_states", release)
    return calls


def test_versions_resume_from_the_first_partition_that_is_missing(tmp_path, released):
    confirmed = dict(path="otro.parquet", sha256="1" * 64, rows=1, bytes=1, metrics={})
    report = dict(
        versions=dict(
            epochs=[2, 3],
            predictions={
                "2": dict(calibration=confirmed, evaluation=confirmed),
                "3": dict(calibration=confirmed),
            },
        )
    )
    trainer, names = Trainer(), []
    titans_walk_forward.emit_versions(
        tmp_path, report, trainer, dict(common_epoch=1), [2, 3], emitter(names), StopRequest()
    )
    # La época 2 ya tenía sus dos tramos y la 3 solo emite el que faltaba.
    assert trainer.loaded == [3] and names == ["evaluation"]
    predictions = report["versions"]["predictions"]
    assert predictions["2"] == dict(calibration=confirmed, evaluation=confirmed)
    assert predictions["3"]["calibration"] == confirmed
    record = predictions["3"]["evaluation"]
    path = tmp_path / "versions" / "epoch-0003-evaluation-predictions.parquet"
    assert record["path"] == "versions/epoch-0003-evaluation-predictions.parquet"
    assert (record["sha256"], record["rows"]) == (sha256(path), 12)
    assert json.loads((tmp_path / "run.json").read_text())["versions"] == report["versions"]
    assert released == [tmp_path / "fit" / "checkpoints"] and report["released_state_bytes"] == 7


def test_a_stop_pauses_the_versions_before_the_next_partition(tmp_path, released):
    stop = StopRequest()
    stop.requested = True
    report, trainer, names = {}, Trainer(), []
    with pytest.raises(Paused):
        titans_walk_forward.emit_versions(
            tmp_path, report, trainer, dict(common_epoch=1), [2], emitter(names), stop
        )
    assert names == [] and report["versions"]["predictions"] == {"2": {}}
    assert released == [] and "released_state_bytes" not in report
    # La reanudación emite los dos tramos y solo entonces libera los estados.
    titans_walk_forward.emit_versions(
        tmp_path, report, trainer, dict(common_epoch=1), [2], emitter(names), StopRequest()
    )
    assert names == ["calibration", "evaluation"] and len(released) == 1


def test_a_version_must_keep_every_row_it_predicts(tmp_path, released):
    with pytest.raises(ValueError, match="no conserva sus filas"):
        titans_walk_forward.emit_versions(
            tmp_path,
            {},
            Trainer(),
            dict(common_epoch=1),
            [2],
            emitter([], extra_rows=1),
            StopRequest(),
        )
    assert released == []
