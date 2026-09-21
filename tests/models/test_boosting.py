"""Boosting con validación externa, matriz acotada y persistencia local verificada."""

import importlib

import numpy as np
import pytest


def module():
    try:
        return importlib.import_module("mars_titan.models.baselines.boosting")
    except ModuleNotFoundError:
        pytest.fail("Falta la referencia de boosting acotada")


def test_boosting_uses_every_training_row_without_internal_random_validation(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(50, 5))
    y = x[:, 0] * 0.03 - x[:, 1] * 0.01
    model = module().fit_boosting_batches(lambda: iter([(x[:20], y[:20]), (x[20:], y[20:])]))
    assert model.estimator.early_stopping is False
    assert model.estimator.n_iter_ == 30
    assert model.training_rows == 50
    prediction = model.predict(x)
    assert np.isfinite(prediction).all()
    assert np.unique(prediction).size > 1
    path = tmp_path / "boosting.joblib"
    digest = model.save(path)
    np.testing.assert_array_equal(
        prediction, module().BoostingModel.load_local(path, digest).predict(x)
    )
    path.write_bytes(b"corrupto")
    with pytest.raises(ValueError, match="huella"):
        module().BoostingModel.load_local(path, digest)


def test_boosting_rejects_a_matrix_larger_than_its_explicit_budget():
    x = np.ones((10, 5))
    with pytest.raises(ValueError, match="presupuesto"):
        module().fit_boosting_batches(lambda: iter([(x, np.ones(10))]), max_bytes=1)


def test_boosting_does_not_overwrite_an_existing_model(tmp_path):
    x = np.arange(100, dtype=float).reshape(20, 5)
    model = module().fit_boosting_batches(lambda: iter([(x, np.arange(20, dtype=float))]))
    path = tmp_path / "boosting.joblib"
    path.write_bytes(b"conservar")
    with pytest.raises(FileExistsError):
        model.save(path)
    assert path.read_bytes() == b"conservar"


def test_boosting_owns_batches_when_the_reader_reuses_its_buffer():
    x = np.arange(50, dtype=float)[:, None]
    y = x[:, 0] * 0.03
    expected = module().fit_boosting_batches(lambda: iter([(x, y)]))

    def reused():
        buffer_x, buffer_y = np.empty((10, 1)), np.empty(10)
        for start in range(0, 50, 10):
            buffer_x[:] = x[start : start + 10]
            buffer_y[:] = y[start : start + 10]
            yield buffer_x, buffer_y

    actual = module().fit_boosting_batches(reused)
    np.testing.assert_array_equal(actual.predict(x), expected.predict(x))


def test_boosting_rejects_nonfinite_model_predictions():
    x = np.arange(50, dtype=float)[:, None]
    model = module().fit_boosting_batches(lambda: iter([(x, x[:, 0])]))
    model.estimator._baseline_prediction[:] = np.inf
    with pytest.raises(ValueError, match="finit"):
        model.predict(x)
