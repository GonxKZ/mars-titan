"""Paridad del operador opcional en puertas, suelos y diagnósticos por paso."""

from contextlib import nullcontext

import pytest
from test_kronlstm import api, tensors

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("rule", ["scalar", "vector"])
@pytest.mark.parametrize("gates", ["decoupled", "coupled", "unit"])
@pytest.mark.parametrize("decay", ["vector", "scalar", "none"])
@pytest.mark.parametrize("values", ["normal", "zero", "floor"])
def test_optional_scan_matches_outputs_and_all_argument_gradients(
    device, dtype, rule, gates, decay, values
):
    import torch

    module = api()
    original = list(tensors(device, dtype))
    if decay == "scalar":
        original[3] = original[3][..., 0]
    elif decay == "none":
        original[3] = None
    if gates == "unit":
        original[5] = None
    if values != "normal":
        original[2] = original[2] * (0 if values == "zero" else 1e-10)
    original[4].flatten()[0] = 0
    original[4].flatten()[-1] = 1
    results = []
    for scan in (module.kronlstm_scan, module.kronlstm_scan_precomputed):
        arguments = [
            value.detach().clone().requires_grad_() if value is not None else None
            for value in original
        ]
        q, k, v, g, beta, target, initial = arguments
        output, state = scan(
            q,
            k,
            v,
            g,
            beta,
            target_gate=target,
            initial_state=initial,
            rule=rule,
            decouple_erase=gates == "decoupled",
        )
        gradients = torch.autograd.grad(
            output.square().mean() + state.square().mean(),
            [value for value in arguments if value is not None],
        )
        results.append((output, state, *gradients))
    tolerance = dict(rtol=2e-5, atol=2e-6) if dtype == "float32" else dict(rtol=1e-10, atol=1e-10)
    for left, right in zip(*results, strict=True):
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        torch.testing.assert_close(left, right, **tolerance)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rule", ["scalar", "vector"])
@pytest.mark.parametrize("fault", ["q_nan", "beta_range", "positive_decay", "overflow"])
def test_optional_scan_keeps_the_first_failure_and_diagnostic_sequence(device, rule, fault):
    from mars_titan.models.titans import state as checks

    arguments = list(tensors(device, "float64"))
    if fault == "q_nan":
        arguments[0].flatten()[0] = float("nan")
    elif fault == "beta_range":
        arguments[4].flatten()[0] = 1.5
    elif fault == "positive_decay":
        arguments[3].flatten()[0] = 0.5
    else:
        arguments[0].fill_(1e308)
        arguments[6].fill_(1e308)
    messages, failures = [], []
    module = api()
    for scan in (module.kronlstm_scan, module.kronlstm_scan_precomputed):
        q, k, v, g, beta, target, initial = arguments
        context = checks.deferred_checks() if device == "cuda:0" else nullcontext()
        with pytest.raises(ValueError) as error, context:
            scan(q, k, v, g, beta, target_gate=target, initial_state=initial, rule=rule)
            messages.append(tuple(checks._PENDING.get().messages))
        failures.append(str(error.value))
    assert failures[0] == failures[1]
    if device == "cuda:0":
        assert len(messages) == 2 and messages[0] == messages[1]
        assert len(messages[0]) == 20
