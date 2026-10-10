"""Confirmación acotada de modelos y admisión de la edición temporal, sin CUDA."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mars_titan.data.storage import sha256
from mars_titan.models.baselines import boosting_selection as selection
from mars_titan.models.baselines.external_boosting import ExternalBoostingModel
from mars_titan.training import external_corpus as engine
from mars_titan.training import tabular_search
from mars_titan.training.checkpoints import StopRequest
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.training.test_reference_run import training_corpus
from tests.training.test_tabular_search import setup


def policy():
    return dict(schema_version=1, minimum_rounds=2, patience_rounds=2, min_delta=1e-5)


def model(count, score, state):
    xgb = pytest.importorskip("xgboost")
    x = np.arange(8, dtype=np.float32).reshape(4, 2)
    booster = xgb.train(
        dict(device="cpu", nthread=1, tree_method="hist", max_depth=1),
        xgb.DMatrix(x, label=x[:, 0], nthread=1),
        count,
    )
    state.observe(count, score)
    return ExternalBoostingModel(
        booster, 4, 2, dict(rows=4, rounds=count, selection=dict(state.state))
    )


def report():
    return dict(
        completed_rounds=0,
        checkpoint=None,
        recovery_checkpoint=None,
        recovery_checkpoints=[],
        attempts=[{}],
        identity=dict(options=dict(rounds=6, checkpoint_interval=1, selection=policy())),
    )


def test_confirmation_keeps_two_recent_and_distinct_best(tmp_path):
    state = selection.BoostingSelection(policy(), 6)
    receipt = report()
    for count, score in enumerate([0.1, 0.2, 0.3, 0.4], 1):
        engine._confirm_selection(tmp_path, receipt, model(count, score, state))
    assert receipt["completed_rounds"] == 4
    assert receipt["selected_round"] == 1
    assert receipt["stop_reason"] == "validation_plateau"
    assert receipt["checkpoint"]["path"].endswith("round-0001.ubj")
    assert receipt["recovery_checkpoint"]["path"].endswith("round-0004.ubj")
    assert {p.name for p in (tmp_path / "checkpoints").glob("*.ubj")} == {
        "attempt-0001-round-0001.ubj",
        "attempt-0001-round-0003.ubj",
        "attempt-0001-round-0004.ubj",
    }
    assert json.loads((tmp_path / "run.json").read_text()) == receipt


def test_failed_receipt_does_not_lose_previous_model_or_advance_state(tmp_path, monkeypatch):
    state = selection.BoostingSelection(policy(), 6)
    receipt = report()
    engine._confirm_selection(tmp_path, receipt, model(1, 0.2, state))
    before = copy.deepcopy(receipt)
    original = engine.atomic_json

    def fail(*args):
        raise OSError("No se puede confirmar el recibo")

    monkeypatch.setattr(engine, "atomic_json", fail)
    with pytest.raises(OSError):
        engine._confirm_selection(tmp_path, receipt, model(2, 0.1, state))
    assert receipt == before
    assert json.loads((tmp_path / "run.json").read_text()) == before
    assert (tmp_path / before["checkpoint"]["path"]).is_file()
    monkeypatch.setattr(engine, "atomic_json", original)
    engine._prune_selection(tmp_path, before)
    assert len(list((tmp_path / "checkpoints").glob("*.ubj"))) == 1


def test_new_design_accepts_declared_selection_without_changing_old_design():
    old, old_cases, _ = tabular_search._configuration(
        Path("configs/baselines/tabular-search-us.json")
    )
    new, cases, _ = tabular_search._configuration(
        Path("configs/baselines/tabular-convergence-us.json")
    )
    assert "selection" not in old and all("selection" not in c["parameters"] for c in old_cases)
    assert new["rounds"] == 2000
    assert new["selection"] == dict(
        schema_version=1, minimum_rounds=200, patience_rounds=100, min_delta=1e-5
    )
    assert all(
        c["parameters"]["selection"] == new["selection"] for c in cases if c["kind"] == "xgboost"
    )


def test_resident_validation_reuses_one_traversal_and_rejects_its_budget():
    calls = []

    def blocks():
        calls.append(1)
        yield np.ones((2, 3)), np.array([1.0, 2.0]), ["US", "CN"], np.array([1, 1])

    resident = selection.ResidentValidation(blocks, expected_rows=2, max_bytes=8192)
    first, second = list(resident()), list(resident())
    assert calls == [1]
    for left, right in zip(first[0], second[0], strict=True):
        np.testing.assert_array_equal(left, right)
    with pytest.raises(ValueError, match="presupuesto"):
        selection.ResidentValidation(blocks, expected_rows=2, max_bytes=1)


@pytest.mark.parametrize(
    "change",
    ["none", "rounds", "selected", "metric", "reason", "budget", "undeclared", "checkpoint"],
)
def test_admission_conciles_declared_early_stopping(tmp_path, monkeypatch, change):
    search, config, manifest, _, backend = setup(tmp_path, monkeypatch)
    values = json.loads(config.read_text()) | dict(
        schema_version=2, rounds=6, selection=policy(), max_validation_cache_bytes=1024**2
    )
    config.write_text(json.dumps(values))
    _, cases, _ = search._configuration(config)
    case = next(c for c in cases if c["kind"] == "xgboost")
    folder = tmp_path / "case"
    receipt = backend(manifest, folder, **case["parameters"])
    state = selection.BoostingSelection(policy(), 6)
    score = receipt["predictions"]["validation"]["metrics"]["session_mae"]
    for count, value in enumerate([score, score + 1, score + 1, score + 1], 1):
        state.observe(count, value)
    receipt.update(
        schema_version=2,
        completed_rounds=4,
        selected_round=1,
        consumed_training_rows=48,
        stop_reason="validation_plateau",
        selection=state.state,
        recovery_checkpoint=receipt["checkpoint"],
    )
    receipt["checkpoint"] = dict(receipt["checkpoint"], rounds=1)
    receipt["recovery_checkpoint"] = dict(receipt["recovery_checkpoint"], rounds=4)
    if change == "rounds":
        receipt["completed_rounds"] = 5
    elif change == "selected":
        receipt["selected_round"] = 2
    elif change == "metric":
        receipt["predictions"]["validation"]["metrics"]["session_mae"] += 1
    elif change == "reason":
        receipt["stop_reason"] = "budget_exhausted"
    elif change == "budget":
        receipt["consumed_training_rows"] = 72
    elif change == "checkpoint":
        receipt["checkpoint"]["rounds"] = 4
    elif change == "undeclared":
        del case["parameters"]["selection"]
        del receipt["identity"]["options"]["selection"]
    (folder / "run.json").write_text(json.dumps(receipt))
    source = json.loads(manifest.read_text())
    if change == "none":
        assert search._completed(folder, case, source, receipt["manifest_sha256"])[2] == score
    else:
        with pytest.raises(ValueError):
            search._completed(folder, case, source, receipt["manifest_sha256"])


def test_resident_validation_rejects_complex_or_object_data():
    for values in (np.ones((2, 1), dtype=complex), np.array([["1"], ["2"]])):
        with pytest.raises(ValueError):
            selection.ResidentValidation(
                lambda values=values: iter([(values, np.zeros(2), ["US", "CN"], np.array([1, 1]))]),
                expected_rows=2,
                max_bytes=4096,
            )


def test_retention_refuses_unconfirmed_replacement_and_keeps_all_models(tmp_path):
    state = selection.BoostingSelection(policy(), 6)
    receipt = report()
    engine._confirm_selection(tmp_path, receipt, model(1, 0.2, state))
    extra = tmp_path / "checkpoints" / "attempt-0001-round-0002.ubj"
    extra.write_bytes(b"confirmacion incompleta")
    foreign = tmp_path.parent / f"{tmp_path.name}-foreign.ubj"
    foreign.write_bytes(b"archivo ajeno")
    damaged = copy.deepcopy(receipt)
    damaged["recovery_checkpoints"] = [dict(path=f"../{foreign.name}", sha256=sha256(foreign))]
    with pytest.raises(ValueError):
        engine._prune_selection(tmp_path, damaged)
    assert extra.is_file()
    assert (tmp_path / receipt["checkpoint"]["path"]).is_file()


def test_model_write_failure_leaves_previous_receipt_recoverable(tmp_path, monkeypatch):
    state = selection.BoostingSelection(policy(), 6)
    receipt = report()
    engine._confirm_selection(tmp_path, receipt, model(1, 0.2, state))
    before = copy.deepcopy(receipt)
    candidate = model(2, 0.1, state)

    def failure(path):
        raise OSError("Disco lleno durante la escritura del modelo")

    monkeypatch.setattr(candidate.booster, "save_model", failure)
    with pytest.raises(OSError):
        engine._confirm_selection(tmp_path, receipt, candidate)
    assert receipt == before
    assert json.loads((tmp_path / "run.json").read_text()) == before
    assert len(list((tmp_path / "checkpoints").iterdir())) == 1


def test_orchestration_recovers_selected_predictions_with_explicit_cpu_fixture(
    tmp_path, monkeypatch
):
    xgb = pytest.importorskip("xgboost")

    class CPUReference(ExternalBoostingModel):
        def predict(self, values):
            return self.booster.inplace_predict(values)

    def load(folder, checkpoint, rows):
        path = folder / checkpoint["path"]
        assert sha256(path) == checkpoint["sha256"]
        booster = xgb.Booster(params=dict(device="cpu", nthread=1))
        booster.load_model(path)
        metadata = json.loads(booster.attr("mars_external_contract"))
        return CPUReference(booster, rows, booster.num_features(), metadata["audit"])

    def fit(
        factory,
        directory,
        *,
        rounds,
        resume,
        checkpoint,
        validation_factory,
        validation_rows,
        **kwargs,
    ):
        parts = list(factory())
        x, y = (np.concatenate([part[i] for part in parts]) for i in (0, 1))
        control = selection.BoostingSelection(
            kwargs["selection"], rounds, state=resume.audit["selection"] if resume else None
        )

        def wrap(booster):
            return CPUReference(
                booster,
                len(y),
                x.shape[1],
                dict(
                    rows=len(y), rounds=booster.num_boosted_rounds(), selection=dict(control.state)
                ),
            )

        completed = resume.booster.num_boosted_rounds() if resume else 0
        booster = xgb.train(
            dict(device="cpu", nthread=1, tree_method="hist", max_depth=1),
            xgb.DMatrix(x, label=y, nthread=1),
            rounds - completed,
            xgb_model=resume.booster if resume else None,
            callbacks=[
                selection.selection_callback(
                    xgb,
                    control,
                    lambda booster: selection.session_validation(
                        wrap(booster), validation_factory, expected_rows=validation_rows
                    ),
                    lambda booster, state: checkpoint(wrap(booster)),
                )
            ],
        )
        return wrap(booster[: control.state["selected_round"]])

    dataset = CorpusDataset(training_corpus(tmp_path / "data"))
    cp = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=lambda: (0, 0))))
    monkeypatch.setattr(engine, "_load", load)
    monkeypatch.setattr(engine, "fit_external_boosting", fit)
    identity = dict(
        options=dict(
            rounds=6,
            checkpoint_interval=1,
            selection=policy(),
            batch_size=5,
            max_validation_cache_bytes=1024**2,
        )
    )
    monkeypatch.setattr(engine, "_identity", lambda *args: identity)

    def initial():
        return dict(
            identity=identity,
            samples=dataset.manifest["counts"],
            attempts=[],
            checkpoint=None,
            completed_rounds=0,
        )

    full, resumed = tmp_path / "full", tmp_path / "resumed"
    full.mkdir()
    resumed.mkdir()
    reference = engine._execute(dataset, full, initial(), None, cp, xgb, StopRequest())
    stop = StopRequest()
    save = engine._confirm_selection

    def pause(output, receipt, model):
        save(output, receipt, model)
        if receipt["completed_rounds"] == 2:
            stop.request_stop()

    monkeypatch.setattr(engine, "_confirm_selection", pause)
    interrupted = engine._execute(dataset, resumed, initial(), None, cp, xgb, stop)
    assert interrupted["status"] == "paused" and interrupted["completed_rounds"] == 2
    monkeypatch.setattr(engine, "_confirm_selection", save)
    parent = load(resumed, interrupted["recovery_checkpoint"], 12)
    result = engine._execute(dataset, resumed, interrupted, parent, cp, xgb, StopRequest())
    assert result["status"] == "completed"
    assert result["selection"] == reference["selection"]
    assert result["predictions"] == reference["predictions"]
    assert (
        result["selected_round"]
        == load(resumed, result["checkpoint"], 12).booster.num_boosted_rounds()
    )
