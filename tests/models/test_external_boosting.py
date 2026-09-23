"""Boosting CUDA por bloques, con población y persistencia comprobables."""

import importlib
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("cupy", "xgboost")),
    reason="La comprobación CUDA requiere el extra boosting",
)


class SmallBooster:
    """Aislar la publicación del archivo de las asignaciones CUDA."""

    def set_attr(self, **attributes):
        self.attributes = attributes

    def save_model(self, path):
        Path(path).write_bytes(b"model")


def module():
    try:
        return importlib.import_module("mars_titan.models.baselines.external_boosting")
    except ModuleNotFoundError:
        pytest.fail("Falta el estimador de boosting con memoria externa")


def arrays():
    x = np.arange(96, dtype=np.float32).reshape(24, 4) / 100
    return x, x[:, 0] * 0.02


def test_external_initialization_keeps_torch_matmul_valid_in_fresh_process():
    code = (
        "from mars_titan.models.baselines.external_boosting import _libraries\n"
        "_libraries()\n"
        "import torch\n"
        "x = torch.ones(3, 20, device='cuda:0')\n"
        "model = torch.nn.Linear(20, 32).to('cuda:0')\n"
        "assert torch.isfinite(model(x)).all()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_external_gpu_fit_consumes_all_rows_and_restores_predictions(tmp_path):
    x, y = arrays()

    def factory():
        for start in range(0, 24, 8):
            yield x[start : start + 8], y[start : start + 8]

    model = module().fit_external_boosting(
        factory,
        tmp_path / "cache",
        expected_rows=24,
        rounds=3,
        max_depth=2,
        max_bin=32,
        max_batch_bytes=1024**2,
    )
    predicted = model.predict(x)
    assert model.training_rows == 24 and model.features == 4
    assert model.audit["device"] == "cuda:0" and model.audit["external_memory"] is True
    assert model.audit["completed_pass_rows"] and set(model.audit["completed_pass_rows"]) == {24}
    assert predicted.shape == (24,) and np.isfinite(predicted).all()
    path = tmp_path / "model.ubj"
    digest = model.save(path)
    restored = module().ExternalBoostingModel.load(path, digest, training_rows=24)
    np.testing.assert_array_equal(predicted, restored.predict(x))
    assert restored.max_batch_bytes == model.max_batch_bytes
    assert restored.audit == model.audit
    with pytest.raises(ValueError, match="existe|nuevo"):
        model.save(path)


def test_incomplete_factory_does_not_claim_full_training(tmp_path):
    x, y = arrays()
    with pytest.raises(ValueError, match="población|filas"):
        module().fit_external_boosting(
            lambda: iter([(x[:8], y[:8])]),
            tmp_path / "cache",
            expected_rows=24,
            rounds=1,
            max_bin=32,
        )


def test_oversized_batch_is_rejected_before_copying_to_gpu(tmp_path):
    x, y = arrays()
    with pytest.raises(ValueError, match="presupuesto|bloque"):
        module().fit_external_boosting(
            lambda: iter([(x, y)]),
            tmp_path / "cache",
            expected_rows=24,
            rounds=1,
            max_batch_bytes=32,
        )


def test_nonfinite_values_do_not_become_missing_values_silently(tmp_path):
    x, y = arrays()
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="finitos"):
        module().fit_external_boosting(
            lambda: iter([(x, y)]), tmp_path / "cache", expected_rows=24, rounds=1
        )


def test_repeated_passes_may_change_batch_boundaries_but_not_rows(tmp_path):
    x, y = arrays()
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        size = 8 if calls % 2 else 6
        for start in range(0, 24, size):
            yield x[start : start + size], y[start : start + size]

    model = module().fit_external_boosting(
        factory, tmp_path / "cache", expected_rows=24, rounds=1, max_bin=32
    )
    assert set(model.audit["completed_pass_rows"]) == {24}


def test_changed_values_between_passes_are_rejected(tmp_path):
    x, y = arrays()
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        yield x + calls, y

    with pytest.raises(ValueError, match="pasadas|valores"):
        module().fit_external_boosting(
            factory, tmp_path / "cache", expected_rows=24, rounds=1, max_bin=32
        )


def test_prediction_budget_accounts_for_float32_not_source_dtype():
    model = module().ExternalBoostingModel(object(), 8, 4, {}, max_batch_bytes=32)
    with pytest.raises(ValueError, match="presupuesto"):
        model.predict(np.ones((8, 4), dtype=np.int8))


def test_published_checkpoint_directory_is_synced_before_return(tmp_path, monkeypatch):
    synced = []
    original = os.fsync

    def fsync(descriptor):
        synced.append(stat.S_ISDIR(os.fstat(descriptor).st_mode))
        return original(descriptor)

    monkeypatch.setattr(os, "fsync", fsync)
    model = module().ExternalBoostingModel(SmallBooster(), 1, 1, {})
    model.save(tmp_path / "model.ubj")
    assert synced == [False, True]


def test_oversized_serialized_model_is_not_published(tmp_path, monkeypatch):
    engine = module()
    monkeypatch.setattr(engine, "MAX_MODEL_BYTES", 2, raising=False)
    model = engine.ExternalBoostingModel(SmallBooster(), 1, 1, {})
    path = tmp_path / "model.ubj"
    with pytest.raises(ValueError, match="tamaño|presupuesto"):
        model.save(path)
    assert not path.exists()


def test_prediction_rejects_finite_values_that_overflow_float32():
    model = module().ExternalBoostingModel(object(), 8, 4, {})
    with pytest.raises(ValueError, match="float32|finitos"):
        model.predict(np.full((8, 4), 1e300, dtype=np.float64))


def test_round_checkpoint_resumes_identical_predictions_and_parameters(tmp_path):
    x, y = arrays()
    engine = module()
    params = dict(expected_rows=24, rounds=5, max_depth=2, max_bin=32)
    reference = engine.fit_external_boosting(lambda: iter([(x, y)]), tmp_path / "full", **params)
    snapshots = []

    def checkpoint(model):
        snapshots.append(model.save(tmp_path / "round.ubj"))
        raise InterruptedError("Parada reproducible tras confirmar una ronda")

    with pytest.raises(InterruptedError):
        engine.fit_external_boosting(
            lambda: iter([(x, y)]),
            tmp_path / "first",
            **params,
            checkpoint=checkpoint,
            checkpoint_interval=2,
        )
    restored = engine.ExternalBoostingModel.load(tmp_path / "round.ubj", snapshots[0])
    assert restored.booster.num_boosted_rounds() == 2
    resumed = engine.fit_external_boosting(
        lambda: iter([(x, y)]),
        tmp_path / "resumed",
        **params,
        resume=restored,
    )
    assert resumed.audit["rounds"] == 5
    np.testing.assert_array_equal(reference.predict(x), resumed.predict(x))
    for changed in (dict(learning_rate=0.1), dict(max_bin=64)):
        with pytest.raises(ValueError, match="continuación|parámetros"):
            engine.fit_external_boosting(
                lambda: iter([(x, y)]),
                tmp_path / str(len(changed)) / next(iter(changed)),
                **(params | changed),
                resume=restored,
            )
    with pytest.raises(ValueError, match="continuación|datos"):
        engine.fit_external_boosting(
            lambda: iter([(x + 1, y)]),
            tmp_path / "changed",
            **params,
            resume=restored,
        )
