"""Contrato real del gancho, paridad CUDA y recuperación del decaimiento anclado."""

import copy
import io

import pytest
import torch

from mars_titan.training.anchored_decay import AnchoredAdamW
from mars_titan.training.learning_hold import install_optimizer_guard
from tests.suite_support import requires_cuda
from tests.training.test_anchored_decay import (
    STEPS,
    _gradients,
    _model,
    _run,
    _same,
    _snapshot,
    _zero_anchors,
)

pytestmark = STEPS
DEVICES = pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=requires_cuda)])


@DEVICES
@pytest.mark.parametrize("call", ["absent", "positional_none", "keyword_none"])
def test_step_accepts_the_three_forms_without_a_closure(device, call):
    value = torch.nn.Parameter(torch.tensor([1.0, -2.0], device=device))
    optimizer = AnchoredAdamW([value], lr=0.03, weight_decay=0.5)
    with torch.no_grad():
        value.add_(0.25)
    value.grad = torch.ones_like(value)
    expected = torch.nn.Parameter(optimizer.decayed_parameters()[0][1])
    reference = torch.optim.AdamW([expected], lr=0.03, weight_decay=0)
    expected.grad = value.grad.clone()
    reference.step()
    if call == "absent":
        optimizer.step()
    elif call == "positional_none":
        optimizer.step(None)
    else:
        optimizer.step(closure=None)
    assert torch.equal(value, expected)


@DEVICES
@pytest.mark.parametrize("keyword", [False, True])
def test_closure_is_rejected_before_decay_or_moment_updates(device, keyword):
    value = torch.nn.Parameter(torch.tensor([1.0, -2.0], device=device))
    optimizer = AnchoredAdamW([value], lr=0.03, weight_decay=0.5)
    with torch.no_grad():
        value.add_(0.25)
    value.grad = torch.ones_like(value)
    before = value.detach().clone()

    def closure():
        pytest.fail("La clausura prohibida no debe ejecutarse")

    with pytest.raises(ValueError, match="clausura"):
        optimizer.step(closure=closure) if keyword else optimizer.step(closure)
    assert torch.equal(value, before) and not optimizer.state


@DEVICES
@pytest.mark.parametrize("call", ["duplicate_closure", "extra_positional", "unknown_keyword"])
def test_invalid_step_arguments_do_not_partially_decay_parameters(device, call):
    value = torch.nn.Parameter(torch.tensor([1.0, -2.0], device=device))
    optimizer = AnchoredAdamW([value], lr=0.03, weight_decay=0.5)
    with torch.no_grad():
        value.add_(0.25)
    value.grad = torch.ones_like(value)
    before = value.detach().clone()
    with pytest.raises(ValueError, match="clausura"):
        if call == "duplicate_closure":
            optimizer.step(None, closure=None)
        elif call == "extra_positional":
            optimizer.step(None, None)
        else:
            optimizer.step(unexpected=None)
    assert torch.equal(value, before) and not optimizer.state


@DEVICES
def test_learning_hold_stops_the_step_before_anchored_decay(tmp_path, device):
    hold = tmp_path / "blocked.json"
    hold.write_text('{"training_allowed": false}')

    def blocked(reason):
        raise RuntimeError(reason)

    value = torch.nn.Parameter(torch.ones(3, device=device))
    optimizer = AnchoredAdamW([value], lr=0.03, weight_decay=0.5)
    with torch.no_grad():
        value.add_(0.25)
    value.grad = torch.ones_like(value)
    before = value.detach().clone()
    guard = install_optimizer_guard(blocked, hold)
    try:
        with pytest.raises(RuntimeError, match="Bloqueo de aprendizaje"):
            optimizer.step()
        assert torch.equal(value, before) and not optimizer.state
    finally:
        guard.remove()


@requires_cuda
@pytest.mark.parametrize("foreach", [False, True])
@pytest.mark.parametrize("decay", [0.0, 0.5])
def test_cuda_steps_match_adamw_with_zero_anchor_or_zero_decay(foreach, decay):
    left, right = _model().to("cuda:0"), _model().to("cuda:0")

    def anchored(model):
        optimizer = AnchoredAdamW(model.parameters(), lr=0.05, weight_decay=decay, foreach=foreach)
        _zero_anchors(optimizer)
        return optimizer

    _run(
        lambda m: torch.optim.AdamW(m.parameters(), lr=0.05, weight_decay=decay, foreach=foreach),
        left,
        5,
    )
    _run(anchored, right, 5)
    assert _same(_snapshot(left), _snapshot(right))


@requires_cuda
def test_cuda_resume_restores_anchors_moments_and_the_next_step():
    def factory(model):
        return AnchoredAdamW(model.parameters(), lr=0.05, weight_decay=0.5)

    whole, part = _model().to("cuda:0"), _model().to("cuda:0")
    complete = _run(factory, whole, 5)
    interrupted = _run(factory, part, 3)
    stream = io.BytesIO()
    torch.save(dict(model=part.state_dict(), optimizer=interrupted.state_dict()), stream)
    stream.seek(0)
    saved = torch.load(stream, map_location="cpu", weights_only=True)
    restored = _model(seed=9).to("cuda:0")
    restored.load_state_dict(saved["model"])
    optimizer = factory(restored)
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
    assert all(anchor.device.type == "cuda" for anchor in optimizer.param_groups[0]["anchors"])
    for step in range(3, 5):
        _gradients(restored, seed=1 + step)
        optimizer.step()
    assert _same(_snapshot(whole), _snapshot(restored))
    for expected, actual in zip(complete.state.values(), optimizer.state.values(), strict=True):
        assert expected.keys() == actual.keys()
        assert all(torch.equal(expected[key], actual[key]) for key in expected)
