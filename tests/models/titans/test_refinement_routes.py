"""Rutas acotadas sobre una misma decisión, sin avanzar el núcleo MAC."""

import importlib
from dataclasses import replace

import pytest
import torch
from test_episodic_readout import CODEC, CONTEXT, snapshot
from test_episodic_readout_fixed_episodes import random_reader, random_snapshot


def api():
    return importlib.import_module("mars_titan.models.titans.refinement_routes")


def fixture(dtype=torch.float64):
    model = random_reader(25, dtype=dtype, neighbors=3, refinements=4)
    data = random_snapshot(15, dtype)
    z = torch.randn((4, 32), generator=torch.Generator().manual_seed(17), dtype=dtype)
    return model, data, z


def reference(model, data, base, route):
    refresh = {"k1": (), "k2c": (), "k2r": (1,), "k4cc": (), "k4cr": (2,), "k4rc": (1,)}
    steps = int(route[1])
    state, first, reads = base, None, []
    for step in range(steps):
        if step in refresh[route]:
            first = None
        read, first = model._read(state, data, CONTEXT, 10, first)
        state = model.refine(state, base, read)
        reads.append(read)
    return state, reads


@pytest.mark.parametrize("route", ["k1", "k2c", "k2r", "k4cc", "k4cr", "k4rc"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_routes_match_uncached_reference_and_count_real_work(route, dtype):
    model, data, z = fixture(dtype)
    expected, reads = reference(model, data, z, route)
    before = z.clone()
    result = api().run_route(model, z, data, route=route, context_id=CONTEXT, cutoff=10)
    assert torch.equal(result.state, expected)
    assert torch.equal(z, before)
    assert result.refinements == int(route[1])
    assert result.searches == 1 + int("r" in route)
    assert result.value_projections == result.searches
    assert not result.state.requires_grad
    for actual, wanted in zip(result.reads, reads, strict=True):
        assert torch.equal(actual.ids, wanted.ids)
        assert torch.equal(actual.weights, wanted.weights)
        assert torch.equal(actual.values, wanted.values)


def test_collection_shares_prefixes_and_retains_all_six_results(monkeypatch):
    model, data, z = fixture()
    count = {"step": 0, "projection": 0}
    original = model._step

    def step(*args):
        count["step"] += 1
        return original(*args)

    monkeypatch.setattr(model, "_step", step)
    handle = model.value_projection.register_forward_hook(
        lambda *_: count.__setitem__("projection", count["projection"] + 1)
    )
    result = api().collect_routes(model, z, data, context_id=CONTEXT, cutoff=10)
    handle.remove()
    assert count == {"step": 9, "projection": 3}
    assert result.refinements == 9
    assert result.searches == result.value_projections == 3
    assert tuple(result.routes) == ("k1", "k2c", "k2r", "k4cc", "k4cr", "k4rc")
    for route, actual in result.routes.items():
        expected, _ = reference(model, data, z, route)
        assert torch.equal(actual.state, expected)


@pytest.mark.parametrize("route", ["k4cc", "k4cr", "k4rc"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cached_attention_and_projection_preserve_gradients(route, dtype):
    model, data, z = fixture(dtype)
    z.requires_grad_()
    expected, _ = reference(model, data, z, route)
    parameters = (z, *model.parameters())
    left = torch.autograd.grad(expected.square().sum(), parameters)
    actual = api().run_route(
        model, z, data, route=route, context_id=CONTEXT, cutoff=10, differentiable=True
    )
    right = torch.autograd.grad(actual.state.square().sum(), parameters)
    assert torch.equal(expected, actual.state)
    for a, b in zip(left, right, strict=True):
        tolerance = 1e-12 if dtype == torch.float64 else 2e-6
        torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance)
        assert torch.isfinite(b).all()


@pytest.mark.parametrize("route", ["k3", "k4rr", "k2", None, [], 1])
def test_rejects_unknown_or_overbudget_routes_before_computing(route, monkeypatch):
    model, data, z = fixture()
    monkeypatch.setattr(model, "_step", lambda *_: pytest.fail("No debe refinar"))
    with pytest.raises(ValueError, match="ruta"):
        api().run_route(model, z, data, route=route, context_id=CONTEXT, cutoff=10)


@pytest.mark.parametrize("absent", [False, True])
def test_empty_or_disabled_bank_does_not_claim_searches(absent):
    model = random_reader(2, refinements=4, mode="no_bank" if absent else "bank")
    data = None if absent else snapshot(0)
    z = torch.ones((2, 32), dtype=torch.float64)
    result = api().run_route(model, z, data, route="k4rc", context_id=CONTEXT, cutoff=10)
    expected, _ = reference(model, data, z, "k4rc")
    assert torch.equal(result.state, expected)
    assert result.searches == result.value_projections == 0
    assert result.refinements == 4


def test_cache_never_survives_parameter_or_snapshot_changes():
    model, data, z = fixture()
    options = dict(route="k4cc", context_id=CONTEXT, cutoff=10)
    first = api().run_route(model, z, data, **options)
    with torch.no_grad():
        model.value_projection.bias.add_(0.5)
    second = api().run_route(model, z, data, **options)
    expected, _ = reference(model, data, z, "k4cc")
    assert torch.equal(second.state, expected)
    assert not torch.equal(first.state, second.state)
    changed = random_snapshot(15, torch.float64, seed=99)
    third = api().run_route(model, z, changed, **options)
    expected, _ = reference(model, changed, z, "k4cc")
    assert torch.equal(third.state, expected)
    assert not torch.equal(second.state, third.state)


@pytest.mark.parametrize("function", ["run_route", "collect_routes"])
@pytest.mark.parametrize("base", [None, [], 2])
def test_invalid_working_states_fail_with_the_tensor_contract(function, base):
    model, data, _ = fixture()
    kwargs = {"route": "k1"} if function == "run_route" else {}
    with pytest.raises(ValueError, match="tensor"):
        getattr(api(), function)(model, base, data, context_id=CONTEXT, cutoff=10, **kwargs)


def test_collection_budget_is_checked_before_any_read_or_refinement(monkeypatch):
    model, data, z = fixture()
    required = model.estimated_bytes(len(z), data.count, differentiable=False)
    model.config = replace(model.config, max_working_bytes=required)
    api().run_route(model, z, data, route="k1", context_id=CONTEXT, cutoff=10)
    monkeypatch.setattr(model, "_read_parts", lambda *_args, **_kwargs: pytest.fail("No debe leer"))
    with pytest.raises(ValueError, match="presupuesto"):
        api().collect_routes(model, z, data, context_id=CONTEXT, cutoff=10)


def test_k1_budget_does_not_charge_the_unused_checkpoint_k4():
    model, data, z = fixture()
    required = model.estimated_bytes(len(z), data.count, refinements=1)
    model.config = replace(model.config, max_working_bytes=required)
    result = api().run_route(model, z, data, route="k1", context_id=CONTEXT, cutoff=10)
    assert result.refinements == 1
    with pytest.raises(ValueError, match="presupuesto"):
        api().run_route(model, z, data, route="k4cc", context_id=CONTEXT, cutoff=10)


def test_collection_gradients_equal_six_independent_routes():
    model, data, z = fixture()
    z.requires_grad_()
    expected = sum(reference(model, data, z, route)[0].square().sum() for route in api().ROUTES)
    parameters = (z, *model.parameters())
    left = torch.autograd.grad(expected, parameters)
    actual = api().collect_routes(
        model, z, data, context_id=CONTEXT, cutoff=10, differentiable=True
    )
    right = torch.autograd.grad(
        sum(item.state.square().sum() for item in actual.routes.values()), parameters
    )
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_routes_and_shared_gradients_preserve_the_reference(dtype):
    model, data, z = fixture(dtype)
    model.to("cuda:0")
    data = type(data).restore(
        data.export_cpu(),
        codec_id=CODEC,
        context_id=CONTEXT,
        cutoff=10,
        device="cuda:0",
        dtype=dtype,
    )
    z = z.to("cuda:0").requires_grad_()
    options = dict(context_id=CONTEXT, cutoff=10, differentiable=True)
    values = api().collect_routes(model, z, data, **options)
    for route in api().ROUTES:
        expected, _ = reference(model, data, z, route)
        assert torch.equal(values.routes[route].state, expected)
    left = torch.autograd.grad(
        sum(value.state.square().sum() for value in values.routes.values()),
        (z, *model.parameters()),
    )
    expected = sum(reference(model, data, z, route)[0].square().sum() for route in api().ROUTES)
    right = torch.autograd.grad(expected, (z, *model.parameters()))
    tolerance = 1e-12 if dtype == torch.float64 else 4e-6
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance)
