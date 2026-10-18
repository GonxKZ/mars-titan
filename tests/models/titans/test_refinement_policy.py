"""Elección de rutas por utilidad prevista, costes y soporte observado."""

import importlib
from dataclasses import replace

import pytest
import torch
from test_episodic_readout import CONTEXT
from test_refinement_routes import fixture, reference


def api():
    return importlib.import_module("mars_titan.models.titans.refinement_policy")


def policy():
    m = api()
    n = len(m.FEATURES)
    return m.RoutePolicy(
        mean=torch.zeros(n, dtype=torch.float64),
        scale=torch.ones(n, dtype=torch.float64),
        coefficients=torch.zeros((n + 1, 5), dtype=torch.float64),
        support_min=torch.full((n,), -100.0, dtype=torch.float64),
        support_max=torch.full((n,), 100.0, dtype=torch.float64),
        costs=torch.zeros(5, dtype=torch.float64),
        min_gain=0.0,
    )


def test_no_expected_gain_and_exact_ties_choose_the_cheapest_baseline():
    model = policy()
    features = torch.zeros((4, len(api().FEATURES)), dtype=torch.float64)
    assert model(features).tolist() == [0] * 4
    model.coefficients[-1, 0] = 0.5
    model.coefficients[-1, 1] = 0.5
    assert model(features).tolist() == [1] * 4
    model.costs[0] = 0.6
    assert model(features).tolist() == [2] * 4


def test_cost_and_required_gain_use_the_same_units_as_predicted_error_reduction():
    model = policy()
    features = torch.zeros((2, len(api().FEATURES)), dtype=torch.float64)
    model.coefficients[-1, 3] = 0.02
    model.costs[3] = 0.03
    assert model(features).tolist() == [0, 0]
    model.costs[3] = 0.01
    assert model(features).tolist() == [4, 4]


def test_out_of_support_falls_back_to_k1_instead_of_extrapolating():
    model = policy()
    model.coefficients[-1, :] = 2.0
    features = torch.zeros((3, len(api().FEATURES)), dtype=torch.float64)
    features[0, 0], features[1, 1] = -101, 101
    assert model(features).tolist() == [0, 0, 1]


def test_unsupported_finite_features_do_not_overflow_an_unused_regression():
    model = policy()
    model.scale.fill_(1e-320)
    model.support_min.fill_(-1e-320)
    model.support_max.fill_(1e-320)
    features = torch.full((2, len(api().FEATURES)), 1.0, dtype=torch.float64)
    assert model(features).tolist() == [0, 0]


def test_invalid_features_and_invalid_scales_fail_instead_of_selecting_a_route():
    model = policy()
    features = torch.zeros((2, len(api().FEATURES)), dtype=torch.float64)
    features[1, 0] = float("nan")
    with pytest.raises(ValueError, match="finit"):
        model(features)
    model.scale[0] = 0
    with pytest.raises(ValueError, match="escala"):
        model.validate()


def test_state_roundtrip_preserves_the_route_and_rejects_changed_contract():
    model = policy()
    model.coefficients[-1, 1] = 0.3
    other = policy()
    other.load_state_dict(model.state_dict())
    features = torch.zeros((2, len(api().FEATURES)), dtype=torch.float64)
    assert torch.equal(other(features), model(features))
    payload = model.state_dict()
    payload["_extra_state"]["features"] = ["future_label"]
    with pytest.raises(ValueError):
        other.load_state_dict(payload)


def test_routed_batch_only_computes_selected_continuations(monkeypatch):
    reader, data, z = fixture()
    choices = torch.tensor([0, 1, 4, 5], dtype=torch.int64)
    head = torch.nn.Linear(32, 1, dtype=torch.float64)
    calls = []
    original = reader._step

    def count(state, *args):
        calls.append(len(state))
        return original(state, *args)

    monkeypatch.setattr(reader, "_step", count)
    result = api().apply_routes(
        reader, z, head, data, choices=choices, context_id=CONTEXT, cutoff=10
    )
    assert sum(calls) == 1 + 2 + 4 + 4
    assert result.route_ids.tolist() == choices.tolist()
    assert result.refinements.tolist() == [1, 2, 4, 4]
    assert result.searches.tolist() == [1, 1, 2, 2]
    for index, route in enumerate(("k1", "k2c", "k4cr", "k4rc")):
        expected, _ = reference(reader, data, z[index : index + 1], route)
        torch.testing.assert_close(
            result.state[index : index + 1], expected, rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(result.point_predictions[index], head(expected)[0, 0])


def test_selected_routes_are_invariant_to_permutation_and_keep_rng(monkeypatch):
    reader, data, z = fixture()
    head = torch.nn.Linear(32, 1, dtype=torch.float64)
    choices = torch.tensor([0, 1, 4, 5], dtype=torch.int64)
    order = torch.tensor([3, 0, 2, 1])
    rng = torch.random.get_rng_state()
    kwargs = dict(context_id=CONTEXT, cutoff=10)
    left = api().apply_routes(reader, z, head, data, choices=choices, **kwargs)
    right = api().apply_routes(reader, z[order], head, data, choices=choices[order], **kwargs)
    torch.testing.assert_close(right.state, left.state[order], rtol=1e-12, atol=1e-12)
    assert torch.equal(left.route_ids[order], right.route_ids)
    assert torch.equal(rng, torch.random.get_rng_state())


def test_policy_receives_only_first_step_features_and_cannot_see_other_routes():
    reader, data, z = fixture()
    head = torch.nn.Linear(32, 1, dtype=torch.float64)
    gate = policy()
    first = api().apply_routes(reader, z, head, data, policy=gate, context_id=CONTEXT, cutoff=10)
    assert first.route_ids.tolist() == [0] * len(z)
    expected, _ = reference(reader, data, z, "k1")
    assert torch.equal(first.state, expected)
    assert first.features.shape == (len(z), len(api().FEATURES))
    assert not first.features.requires_grad


@pytest.mark.parametrize("fault", ["identity", "scale", "cost", "extra"])
def test_rejected_checkpoint_does_not_mutate_the_existing_policy(fault):
    model = policy()
    before = {name: value.clone() for name, value in model.named_buffers()}
    payload = {name: value.clone() for name, value in model.named_buffers()}
    payload["_extra_state"] = model.get_extra_state()
    payload["coefficients"].fill_(0.7)
    if fault == "identity":
        payload["_extra_state"]["routes"] = ["future_best"]
    elif fault == "scale":
        payload["scale"][0] = 0
    elif fault == "cost":
        payload["costs"][0] = -1
    else:
        payload["unknown"] = torch.zeros(1)
    with pytest.raises((ValueError, RuntimeError)):
        model.load_state_dict(payload)
    assert all(torch.equal(before[name], value) for name, value in model.named_buffers())


def test_explicit_k1_routing_only_reserves_k1_memory():
    reader, data, z = fixture()
    required = reader.estimated_bytes(len(z), data.count, refinements=1)
    reader.config = replace(reader.config, max_working_bytes=required)
    result = api().apply_routes(
        reader,
        z,
        torch.nn.Linear(32, 1, dtype=torch.float64),
        data,
        choices=torch.zeros(len(z), dtype=torch.int64),
        context_id=CONTEXT,
        cutoff=10,
    )
    assert result.refinements.tolist() == [1] * len(z)


def test_invalid_explicit_routes_fail_before_refining(monkeypatch):
    reader, data, z = fixture()
    monkeypatch.setattr(reader, "_step", lambda *_: pytest.fail("No debe refinar"))
    with pytest.raises(ValueError, match="código"):
        api().apply_routes(
            reader, z, None, data, choices=torch.full((len(z),), 6), context_id=CONTEXT, cutoff=10
        )


@pytest.mark.cuda
def test_cuda_rejected_load_is_atomic_even_inside_deferred_checks():
    from mars_titan.models.titans.state import deferred_checks

    model = policy().to("cuda:0")
    before = {name: value.clone() for name, value in model.named_buffers()}
    payload = {name: value.clone() for name, value in model.named_buffers()}
    payload["_extra_state"] = model.get_extra_state()
    payload["coefficients"].fill_(0.5)
    payload["scale"][0] = 0
    with pytest.raises(ValueError, match="escala"), deferred_checks():
        model.load_state_dict(payload)
    assert all(torch.equal(before[name], value) for name, value in model.named_buffers())
