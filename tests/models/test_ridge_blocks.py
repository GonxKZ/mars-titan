"""Equivalencia del objetivo Ridge, escalado pasado y restauración de parámetros."""

import importlib

import numpy as np
import pytest


def module():
    try:
        return importlib.import_module("mars_titan.models.baselines.ridge")
    except ModuleNotFoundError:
        pytest.fail("Falta la referencia Ridge por bloques")


def cuda_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA no disponible, sin alternativa CPU para la ruta acelerada")
    return "cuda:0"


@pytest.mark.parametrize("block_size", [1, 7, 40])
def test_block_ridge_matches_sklearn_with_constant_and_collinear_features(tmp_path, block_size):
    pytest.importorskip("sklearn")
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(17)
    x = rng.normal(size=(40, 4))
    x[:, 2] = x[:, 0] * 2
    x[:, 3] = 3
    y = x[:, 0] * 0.5 - x[:, 1] * 0.2 + rng.normal(size=40) * 0.01

    def factory():
        return (
            (x[i : i + block_size], y[i : i + block_size]) for i in range(0, len(x), block_size)
        )

    model = module().fit_ridge_blocks(factory, alpha=1.0, device=cuda_device())
    scaler = StandardScaler().fit(x)
    reference = Ridge(alpha=1.0).fit(scaler.transform(x), y)
    predictions = model.predict(x)
    np.testing.assert_allclose(
        predictions, reference.predict(scaler.transform(x)), rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(model.mean, scaler.mean_, atol=1e-12)
    before = model.mean.copy()
    model.predict(x * 100 + 500)
    np.testing.assert_array_equal(before, model.mean)
    artifact = tmp_path / "ridge.npz"
    model.save(artifact)
    np.testing.assert_array_equal(predictions, module().RidgeModel.load(artifact).predict(x))


def test_ridge_rejects_a_second_pass_with_changed_training_rows():
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        yield np.array([[1.0, 2.0], [3.0, 4.0]]) + calls, np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="cambiado"):
        module().fit_ridge_blocks(factory, device=cuda_device())


def test_ridge_rejects_oversized_feature_space_and_nonfinite_inputs():
    for x in (np.ones((2, 4097)), np.array([[float("nan")], [2.0]])):
        with pytest.raises(ValueError):
            module().fit_ridge_blocks(lambda x=x: iter([(x, np.ones(2))]), device=cuda_device())


def test_ridge_constant_decimal_features_keep_unit_scale():
    x = np.full((40, 2), 0.1)
    model = module().fit_ridge_blocks(
        lambda: ((x[i : i + 7], np.arange(40, dtype=float)[i : i + 7]) for i in range(0, 40, 7)),
        device=cuda_device(),
    )
    np.testing.assert_array_equal(model.scale, np.ones(2))


def test_failed_model_save_does_not_publish_a_partial_checkpoint(tmp_path, monkeypatch):
    model = module().RidgeModel(np.zeros(2), np.ones(2), np.ones(2), 0.0)

    def fail(stream, **arrays):
        stream.write(b"incompleto")
        raise OSError("fallo de disco simulado")

    monkeypatch.setattr(np, "savez", fail)
    path = tmp_path / "ridge.npz"
    with pytest.raises(OSError):
        model.save(path)
    assert not path.exists()


def test_large_offset_does_not_penalize_the_intercept():
    x = np.array([[1e16], [1e16], [1e16 + 2]])
    y = np.array([0.0, 0.0, 1.0])
    model = module().fit_ridge_blocks(lambda: iter([(x, y)]), device=cuda_device())
    np.testing.assert_allclose(model.predict(x), [1 / 11, 1 / 11, 9 / 11], rtol=1e-10, atol=1e-12)


def test_statistical_overflow_fails_instead_of_producing_a_finite_wrong_model():
    x = np.array([[-1e200], [1e200]])
    with pytest.raises(ValueError, match="estadísticas|desbordamiento|finit"):
        module().fit_ridge_blocks(lambda: iter([(x, np.array([-1.0, 1.0]))]), device=cuda_device())
