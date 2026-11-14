"""Perfil GPU explícito, sin modificar los parámetros científicos ni recurrir a CPU."""

import copy
import sys
import types

import numpy as np
import pytest

from mars_titan.models.baselines import qlib_lightgbm as qlib
from mars_titan.models.baselines import same_inputs_lightgbm as control
from tests.models.test_qlib_lightgbm import Rows, panel
from tests.models.test_qlib_lightgbm import lightgbm as lightgbm
from tests.models.test_same_inputs_lightgbm import CONFIG


def gpu_config():
    config = copy.deepcopy(CONFIG)
    config.update(
        identity="same_inputs_lightgbm_opencl_float64_v1",
        execution=dict(backend="opencl_float64", platform_id=0, device_id=0),
    )
    return config


def test_gpu_execution_has_its_own_identity_and_preserves_scientific_parameters():
    config = gpu_config()
    assert control.checked_config(config) == config
    assert config["parameters"] == qlib.PARAMETERS
    assert control.checked_config(copy.deepcopy(CONFIG)) == CONFIG


@pytest.mark.parametrize(
    "change",
    [
        dict(backend="cpu"),
        dict(backend="opencl_float32"),
        dict(device_id=-1),
        dict(device_id=False),
        dict(platform_id=0.0),
    ],
)
def test_gpu_profile_cannot_change_device_or_precision_silently(change):
    config = gpu_config()
    config["execution"].update(change)
    with pytest.raises(ValueError):
        control.checked_config(config)


def test_cpu_identity_cannot_silently_select_the_gpu_backend():
    config = gpu_config()
    config["identity"] = control.IDENTITY
    with pytest.raises(ValueError):
        control.checked_config(config)


def test_gpu_fit_selects_double_precision_and_an_explicit_device(lightgbm, monkeypatch):
    monkeypatch.setattr(qlib, "check_gpu_backend", lambda config: dict(name="GPU de prueba"))
    qlib.fit(Rows(panel(4, 6), 1), Rows(panel(4, 6, seed=2), 2), gpu_config(), 42, min_assets=3)
    params = lightgbm["trains"][0][0]
    assert params["device_type"] == "gpu" and params["gpu_use_dp"] is True
    assert params["gpu_device_id"] == params["gpu_platform_id"] == 0
    assert params["deterministic"] is False
    assert params["learning_rate"] == qlib.PARAMETERS["learning_rate"]
    assert "max_bin" not in params


def test_missing_gpu_fails_before_constructing_a_dataset(lightgbm, monkeypatch):
    def missing(_):
        raise RuntimeError("No se encuentra el dispositivo OpenCL solicitado")

    monkeypatch.setattr(qlib, "check_gpu_backend", missing)
    with pytest.raises(RuntimeError, match="OpenCL"):
        qlib.fit(Rows(panel(4, 6), 1), Rows(panel(4, 6, seed=2), 2), gpu_config(), 42, min_assets=3)
    assert lightgbm["trains"] == [] and lightgbm["datasets"] == []


@pytest.mark.parametrize("backend", ["cpu", "opencl_float64"])
def test_native_validation_uses_cached_predictions_and_keeps_the_checkpoint_series(
    monkeypatch, learning_doubles, backend
):
    train = Rows(panel(4, 6), 1)
    validation = Rows(panel(4, 6, seed=2), 2)
    config = gpu_config()
    config["execution"]["validation"] = "native_dataset"
    if backend == "cpu":
        config.update(
            identity="same_inputs_lightgbm_native_validation_v1",
            execution=dict(backend="cpu", validation="native_dataset"),
        )
    control.checked_config(config)
    monkeypatch.setattr(qlib, "check_gpu_backend", lambda _: dict(name="GPU de prueba"))
    datasets = []

    def dataset(values, label, free_raw_data=True, reference=None):
        item = dict(values=values, label=label, reference=reference)
        datasets.append(item)
        return item

    def fit(params, data, num_boost_round, callbacks, valid_sets, valid_names, feval):
        assert data is datasets[0] and valid_sets == [datasets[1]]
        assert datasets[1]["reference"] is data
        assert valid_names == ["validation"] and params["metric"] == "None"
        for iteration in range(num_boost_round):
            predicted = validation.features[:, 0].astype(np.float64) * (iteration + 1)
            assert feval(predicted, valid_sets[0]) == []
            for callback in callbacks:
                callback(types.SimpleNamespace(iteration=iteration, model=None))
        return "booster"

    def repeated(*_, **__):
        raise AssertionError("Se ha vuelto a recorrer el prefijo completo de árboles")

    monkeypatch.setitem(sys.modules, "lightgbm", types.SimpleNamespace(Dataset=dataset, train=fit))
    monkeypatch.setattr(qlib, "predict", repeated)
    _, series = qlib.fit(train, validation, config, 42, min_assets=3)
    assert len(series) == 100
    expected = qlib.sd.session_correlations(
        validation.panel, validation.features[:, 0] * 10.0, min_assets=3
    )
    np.testing.assert_allclose(series[0][0], expected[0], atol=1e-7)
    np.testing.assert_array_equal(series[0][1], expected[1])
