"""El registro de llamadas evita copiar gradientes cuando no se comparan sus valores."""

import pytest
import torch

from tests.training.test_titans_walk_forward import Factory


@pytest.mark.parametrize("retain_gradients", [False, True])
def test_recording_policy_preserves_parameters_gradients_and_call_counts(retain_gradients):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    parameter.grad = torch.tensor([0.5, -0.25])
    before = parameter.detach().clone()
    gradient = parameter.grad.clone()
    factory = Factory(retain_gradients=retain_gradients)
    recorder = factory([dict(params=[parameter], role="shared")])
    for _ in range(3):
        recorder.step()
    assert recorder.calls == len(recorder.records) == len(factory.records) == 3
    assert torch.equal(parameter, before) and torch.equal(parameter.grad, gradient)
    if retain_gradients:
        assert all(torch.equal(record[0], gradient) for record in recorder.records)
        assert all(record[0].data_ptr() != parameter.grad.data_ptr() for record in recorder.records)
    else:
        assert recorder.records == [None, None, None]
