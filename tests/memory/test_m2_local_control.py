"""Paridad de M2 al medir el Jacobiano local, sin aprendizaje."""

import random

import numpy as np
import pytest
import torch
from test_financial_session_controls import (
    four_flow_source as four_flow_source,
)
from test_financial_session_controls import (
    no_target_estimation as no_target_estimation,
)
from test_financial_session_controls import paired_consumers, tensor_leaves
from test_financial_session_m2 import m2_trajectory
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native

# Ambos campos derivan de la identidad del consumidor, que incluye la configuración de C.
MODEL_IDENTITY = ("model_id", "prediction_id")
PATHS = (
    ("reference", 4, False, False),
    ("reordered", 4, True, False),
    ("permuted", 1, True, False),
    ("recovered", 4, False, True),
)


@pytest.fixture
def forbid_learning(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("La comprobación técnica no admite optimizadores ni retropropagación")

    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)


def assert_same_trajectory(expected, actual, tolerance):
    assert expected["points"].keys() == actual["points"].keys()
    keys = sorted(expected["points"])
    np.testing.assert_allclose(
        [expected["points"][key] for key in keys],
        [actual["points"][key] for key in keys],
        rtol=tolerance,
        atol=tolerance,
    )
    assert expected["indices"] == actual["indices"]
    assert expected["receipts"] == actual["receipts"]
    for before, after in zip(expected["scores"], actual["scores"], strict=True):
        assert before == pytest.approx(after, rel=tolerance, abs=tolerance)
    assert expected["fast"].keys() == actual["fast"].keys()
    for key, value in expected["fast"].items():
        torch.testing.assert_close(value, actual["fast"][key], rtol=tolerance, atol=tolerance)


def assert_same_controls(expected, actual, tolerance):
    for before, after in zip(expected["controls"], actual["controls"], strict=True):
        assert before.keys() == after.keys()
        assert before == pytest.approx(after, rel=tolerance, abs=tolerance)


def assert_same_stores(expected, actual, *, ignored=()):
    """Comparar banco M2, episodios y etiquetas pendientes de cada evento confirmado."""

    def episodes(store):
        return {
            key: {name: value for name, value in episode.items() if name not in ignored}
            for key, episode in store["bank"]["episodes"].items()
        }

    for before, after in zip(expected["stores"], actual["stores"], strict=True):
        assert before["bank"]["snapshot"] == after["bank"]["snapshot"]
        assert episodes(before) == episodes(after)
        assert before["pending"]["rows"] == after["pending"]["rows"]
        for name in ("key_inputs", "values"):
            assert torch.equal(before["pending"][name], after["pending"][name])


def test_m2_diagnostic_preserves_selection_partition_recovery_and_state(
    native, four_flow_source, tmp_path, forbid_learning
):
    rng = torch.random.get_rng_state().clone()
    numpy_rng = np.random.get_state(legacy=False)
    python_rng = random.getstate()
    cuda_initialized = torch.cuda.is_initialized()
    consumers = paired_consumers(four_flow_source)
    compared = {}
    for mode, consumer in consumers.items():
        parameters = tensor_leaves(consumer.predictor.state_dict())
        paths = {
            name: m2_trajectory(
                native,
                four_flow_source,
                consumer,
                tmp_path / mode / name,
                block_rows=rows,
                reverse=reverse,
                recover=recover,
            )
            for name, rows, reverse, recover in PATHS
        }
        reference = paths["reference"]
        for name, tolerance in (("reordered", 0), ("permuted", 1e-12), ("recovered", 0)):
            assert_same_trajectory(reference, paths[name], tolerance)
            assert_same_controls(reference, paths[name], tolerance)
        assert_same_stores(reference, paths["reordered"])
        assert_same_stores(reference, paths["recovered"])
        assert reference["final"] == paths["recovered"]["final"]
        current = tensor_leaves(consumer.predictor.state_dict())
        assert current.keys() == parameters.keys()
        for key, value in current.items():
            assert torch.equal(parameters[key], value)
        consumer.verify()
        assert all(parameter.grad is None for parameter in consumer.predictor.parameters())
        compared[mode] = reference
    assert all(not control for control in compared["disabled"]["controls"])
    assert sum(map(bool, compared["diagnostic"]["controls"])) == 5
    assert_same_trajectory(compared["disabled"], compared["diagnostic"], 0)
    assert_same_stores(compared["disabled"], compared["diagnostic"], ignored=MODEL_IDENTITY)
    assert torch.equal(rng, torch.random.get_rng_state())
    np.testing.assert_equal(np.random.get_state(legacy=False), numpy_rng)
    assert random.getstate() == python_rng
    assert torch.cuda.is_initialized() == cuda_initialized
