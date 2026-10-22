"""Candidato de almacenamiento descartado, aislado de la implementación activa."""

import runpy
from dataclasses import replace
from pathlib import Path
from types import MethodType

import pytest
import torch
from test_bank_write import DEVICES, banks, distinct_state, reference
from test_bank_write import memory as base_memory

CANDIDATE = (
    Path(__file__).resolve().parents[3]
    / "reports/engineering/bank-write-storage-20261021/borrowed_candidate.py"
)
associative_problem = runpy.run_path(str(CANDIDATE))["_associative_problem"]


def memory(*args, **options):
    model = base_memory(*args, **options)
    model._associative_problem = MethodType(associative_problem, model)
    return model


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rule", ["mean_partial", "per_bank"])
def test_associative_inference_borrows_storage_without_changing_owner_or_gradients(device, rule):
    model = memory(rule, device=device)
    initial = distinct_state(model)
    state = replace(
        initial,
        weights=tuple(w.requires_grad_() for w in initial.weights),
        levels=tuple(w.requires_grad_() for w in initial.levels),
    )
    originals = sum(banks(state, 2), ())
    saved = tuple(w.detach().clone() for w in originals)
    keys = torch.ones(2, 1, 3, device=device, dtype=torch.float64, requires_grad=True)
    local, prediction = model._associative_problem(keys, state, False)
    # El préstamo elimina seis matrices temporales, sin modificar sus propietarios.
    assert all(a.data_ptr() == b.data_ptr() for a, b in zip(local, originals, strict=True))
    assert all(a is not b and a.is_leaf for a, b in zip(local, originals, strict=True))
    gradients = torch.autograd.grad(prediction.square().sum(), (*local, keys))
    assert all(torch.isfinite(g).all() for g in gradients)
    assert all(w.grad is None and w.requires_grad for w in originals)
    for before, after in zip(saved, originals, strict=True):
        assert torch.equal(before, after)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rule", ["mean_partial", "per_bank"])
def test_non_differentiable_write_preserves_the_input_state_and_all_owner_flags(device, rule):
    model = memory(rule, device=device)
    state = distinct_state(model)
    saved = tuple(t.clone() for t in (*state.weights, *state.momentum, *state.levels, state.steps))
    token = torch.full((2, 1, 3), 0.2, device=device, dtype=torch.float64)
    expected, momentum = reference(model, token, state)
    with torch.no_grad():
        following = model.update(token, state)
    originals = (*state.weights, *state.momentum, *state.levels, state.steps)
    assert all(not t.requires_grad and t.grad is None for t in originals)
    assert all(torch.equal(a, b) for a, b in zip(saved, originals, strict=True))
    outputs = (*sum(banks(following, 2), ()), *following.momentum)
    for actual, wanted in zip(outputs, (*expected, *momentum), strict=True):
        torch.testing.assert_close(actual, wanted, atol=1e-11, rtol=1e-11)
        assert all(
            actual.untyped_storage().data_ptr() != t.untyped_storage().data_ptr() for t in originals
        )
    model.validate_state(following)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rule", ["mean_partial", "per_bank"])
def test_inference_owned_weights_can_be_updated_outside_inference_mode(device, rule):
    model = memory(rule, device=device).eval().requires_grad_(False)
    with torch.inference_mode():
        initial = model.initial_state(2)
    assert all(w.is_inference() for w in sum(banks(initial, 2), ()))
    normal = replace(
        initial,
        weights=tuple(w.clone() for w in initial.weights),
        momentum=tuple(w.clone() for w in initial.momentum),
        levels=tuple(w.clone() for w in initial.levels),
        steps=initial.steps.clone(),
    )
    token = torch.full((2, 1, 3), 0.2, device=device, dtype=torch.float64)
    with torch.no_grad():
        result = model.update(token, initial)
        expected = model.update(token, normal)
    for actual, wanted in zip(
        (*result.weights, *result.levels, *result.momentum),
        (*expected.weights, *expected.levels, *expected.momentum),
        strict=True,
    ):
        assert torch.equal(actual, wanted)
        assert not actual.is_inference()
    with torch.inference_mode(), pytest.raises(ValueError, match="gradiente"):
        model.update(token, initial)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rule", ["mean_partial", "per_bank"])
def test_differentiable_local_variables_keep_curvature_without_aliasing_owners(device, rule):
    model = memory(rule, device=device)
    state = model.initial_state(2, differentiable=True)
    keys = torch.ones(2, 1, 3, device=device, dtype=torch.float64)
    local, prediction = model._associative_problem(keys, state, True)
    originals = sum(banks(state, 2), ())
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(local, originals, strict=True))
    gradients = torch.autograd.grad(prediction.square().sum(), local, create_graph=True)
    curvature = torch.autograd.grad(sum(g.square().sum() for g in gradients), originals)
    assert all(torch.isfinite(g).all() for g in curvature)
    assert any(g.abs().max() > 0 for g in curvature)
