"""Perfil GPU explícito, sin modificar los parámetros científicos ni recurrir a CPU."""

import copy

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
