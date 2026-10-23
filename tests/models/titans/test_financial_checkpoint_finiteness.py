"""Rechazar valores no finitos antes de copiar el checkpoint financiero."""

import copy

import pytest
import torch
from test_regional_geometry_contract import assert_same, predictor

from mars_titan.models.titans.state import deferred_checks


@pytest.mark.cuda
@pytest.mark.parametrize("regional", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_nonfinite_head_is_rejected_before_copy_even_inside_deferred_checks(regional, training):
    model = predictor("cuda:0", torch.float64, regional=regional).train(training)
    before = copy.deepcopy(model.state_dict())
    payload = copy.deepcopy(before)
    payload["head.weight"][0, 0] = float("nan")
    payload["fusion.0.weight"].add_(0.125)
    parameter_id = model._parameter_id
    with pytest.raises(ValueError), deferred_checks():
        model.load_state_dict(payload)
    assert_same(model.state_dict(), before)
    assert model._parameter_id == parameter_id
    model.verify_parameter_identity()


def test_nonboolean_child_assignment_metadata_is_rejected_before_copy():
    model = predictor("cpu", torch.float64)
    before = copy.deepcopy(model.state_dict())
    payload = copy.deepcopy(before)
    payload["head.weight"].add_(0.125)
    payload._metadata["regional_fusion"]["assign_to_params_buffers"] = 0
    with pytest.raises(ValueError):
        model.load_state_dict(payload)
    assert_same(model.state_dict(), before)
    model.verify_parameter_identity()


@pytest.mark.cuda
def test_injected_root_assignment_does_not_authorize_partial_assignment():
    model = predictor("cpu", torch.float64)
    before = copy.deepcopy(model.state_dict())
    payload = copy.deepcopy(before)
    for name, value in payload.items():
        if isinstance(value, torch.Tensor):
            payload[name] = value.to("cuda:0")
    payload["head.weight"].add_(0.125)
    payload._metadata[""]["assign_to_params_buffers"] = True
    payload._metadata["regional_fusion"]["assign_to_params_buffers"] = True
    with pytest.raises(ValueError):
        model.load_state_dict(payload, assign=False)
    assert_same(model.state_dict(), before)
    model.verify_parameter_identity()
