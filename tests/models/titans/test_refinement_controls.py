"""Controles de umbral y asignación reproducible, sin ajustar parámetros."""

import copy
import importlib
import random
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch
from test_episodic_readout import CONTEXT
from test_refinement_routes import fixture, reference

from mars_titan.models.titans.refinement_policy import apply_routes


def api():
    return importlib.import_module("mars_titan.models.titans.refinement_controls")


def threshold(**changes):
    return api().ThresholdRoutePolicy(
        **(dict(feature_name="attention_entropy", cutoff=0.5, route="k2r") | changes)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_threshold_uses_only_the_declared_feature_and_strict_comparison(dtype):
    model = threshold()
    features = torch.zeros((3, len(api().FEATURES)), dtype=dtype)
    features[:, api().FEATURES.index("attention_entropy")] = torch.tensor([0.4, 0.5, 0.6])
    features[:, api().FEATURES.index("first_prediction")] = 100.0
    assert model(features).tolist() == [0, 0, 2]
    assert not list(model.parameters()) and not list(model.buffers())
    assert not model.training


def test_threshold_comparison_keeps_the_cutoff_in_fp64():
    model = threshold(feature_name="memory_present", cutoff=1.0 - 2**-25, route="k4cc")
    features = torch.ones((1, len(api().FEATURES)), dtype=torch.float32)
    assert model(features).tolist() == [3]


@pytest.mark.parametrize(
    "changes",
    [
        {"feature_name": "future_error"},
        {"feature_name": []},
        {"cutoff": True},
        {"cutoff": float("nan")},
        {"cutoff": float("inf")},
        {"cutoff": 10**400},
        {"route": "k1"},
        {"route": "k4rr"},
        {"route": []},
    ],
)
def test_threshold_rejects_unknown_or_nonfinite_configuration(changes):
    with pytest.raises(ValueError):
        threshold(**changes)


@pytest.mark.parametrize("fault", ["nan", "inf", "empty", "large", "width", "dtype", "object"])
def test_threshold_rejects_invalid_features(fault):
    features = torch.zeros((2, len(api().FEATURES)), dtype=torch.float64)
    if fault in ("nan", "inf"):
        features[0, -1] = float(fault)
    elif fault == "empty":
        features = features[:0]
    elif fault == "large":
        features = features.repeat(129, 1)
    elif fault == "width":
        features = features[:, :-1]
    elif fault == "dtype":
        features = features.to(torch.float16)
    else:
        features = None
    with pytest.raises(ValueError):
        threshold()(features)


@pytest.mark.parametrize("fault", ["feature_name", "cutoff", "route", "extra", "missing"])
def test_threshold_checkpoint_rejection_is_strict_and_atomic(fault):
    model = threshold()
    before = copy.deepcopy(model.state_dict())
    model.load_state_dict(before)
    payload = copy.deepcopy(before)
    if fault in ("feature_name", "cutoff", "route"):
        payload["_extra_state"][fault] = {
            "feature_name": "read_rms",
            "cutoff": 0.6,
            "route": "k4cc",
        }[fault]
    elif fault == "extra":
        payload["unexpected"] = torch.ones(1)
    else:
        payload.clear()
    with pytest.raises(ValueError):
        model.load_state_dict(payload, strict=False)
    assert model.state_dict() == before


def test_hashed_routes_pin_the_encoding_and_keep_every_rng_unchanged():
    schedule = api().HashedRoutes(counts=(1, 1, 1, 1, 1, 1), seed=7)
    flows = ("US/AAPL", "US/MSFT", "CN/000001", "CN/600000", "US/SPY", "US/BRK.B")
    times = (1609459200000000,) * len(flows)
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    assert schedule.choices(flows, times) == (4, 5, 2, 5, 1, 3)
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and np.array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert schedule.identity() != replace(schedule, seed=8).identity()
    with pytest.raises(FrozenInstanceError):
        schedule.seed = 9


def test_hash_assignment_is_invariant_to_permutation_microbatches_and_scaled_counts():
    schedule = api().HashedRoutes(counts=(1, 2, 0, 3, 0, 4), seed=23)
    flows = tuple(f"US/A{i}" for i in range(64))
    times = tuple(1609459200000000 + i for i in range(64))
    expected = schedule.choices(flows, times)
    assert schedule.choices(flows[::-1], times[::-1]) == expected[::-1]
    assert (
        schedule.choices(flows[:17], times[:17]) + schedule.choices(flows[17:], times[17:])
        == expected
    )
    assert (
        replace(schedule, counts=tuple(n * 2**60 for n in schedule.counts)).choices(flows, times)
        == expected
    )
    assert replace(schedule, seed=24).choices(flows, times) != expected
    assert schedule.choices(flows, tuple(t + 1 for t in times)) != expected
    assert not {2, 4} & set(expected)


def test_hash_cdf_uses_integer_boundaries_and_excludes_zero_count_routes(monkeypatch):
    values = iter(
        (
            0,
            6148914691236517204,
            6148914691236517205,
            12297829382473034409,
            12297829382473034410,
            2**64 - 1,
        )
    )

    class Digest:
        def digest(self):
            return next(values).to_bytes(8, "big") + bytes(24)

    monkeypatch.setattr(api().hashlib, "sha256", lambda _: Digest())
    schedule = api().HashedRoutes(counts=(1, 0, 1, 0, 0, 1), seed=0)
    assert schedule.choices(("US/A",) * 6, (1,) * 6) == (0, 0, 2, 2, 5, 5)


@pytest.mark.parametrize(
    "counts,seed",
    [
        ([1] * 6, 0),
        ((1,) * 5, 0),
        ((0,) * 6, 0),
        ((-1, 1, 0, 0, 0, 0), 0),
        ((True, 0, 0, 0, 0, 0), 0),
        ((1.0, 0, 0, 0, 0, 0), 0),
        ((1,) * 6, True),
        ((1,) * 6, -1),
        ((1,) * 6, 2**32),
    ],
)
def test_hash_rejects_invalid_counts_and_seeds(counts, seed):
    with pytest.raises(ValueError):
        api().HashedRoutes(counts=counts, seed=seed)


def test_hash_counts_are_bounded_to_nonnegative_int64():
    with pytest.raises(ValueError):
        api().HashedRoutes(counts=(2**63, 0, 0, 0, 0, 0), seed=0)
    maximum = api().HashedRoutes(counts=(0, 0, 2**63 - 1, 0, 0, 0), seed=2**32 - 1)
    assert maximum.choices(("US/A", "CN/000001"), (1, 2)) == (2, 2)


@pytest.mark.parametrize(
    "flows,times",
    [
        ((), ()),
        (("US/A",) * 257, (1,) * 257),
        (("US/A",), ()),
        (("",), (1,)),
        (("US/" + "A" * 65,), (1,)),
        ((None,), (1,)),
        (("US/A",), (True,)),
        (("US/A",), (-1,)),
        (("US/A",), (2**63,)),
        (("US/A",), (1.0,)),
        ("US/A", (1,)),
    ],
)
def test_hash_rejects_invalid_row_identity(flows, times):
    with pytest.raises(ValueError):
        api().HashedRoutes(counts=(1,) * 6, seed=0).choices(flows, times)


def test_host_tuple_routes_match_tensor_choices_without_copying_codes_to_cpu(monkeypatch):
    reader, snapshot, base = fixture()
    head = torch.nn.Linear(32, 1, dtype=torch.float64)
    codes = (0, 1, 4, 5)
    options = dict(context_id=CONTEXT, cutoff=10)
    expected = apply_routes(reader, base, head, snapshot, choices=torch.tensor(codes), **options)
    monkeypatch.setattr(torch.Tensor, "cpu", lambda _: pytest.fail("Los códigos ya están en CPU"))
    actual = apply_routes(reader, base, head, snapshot, choices=codes, **options)
    assert torch.equal(actual.state, expected.state)
    assert actual.route_codes == codes
    assert torch.equal(actual.route_ids, expected.route_ids)
    assert torch.equal(actual.refinements, expected.refinements)


@pytest.mark.parametrize(
    "codes", [(0,), (0, 1, 6, 2), (0, 1, -1, 2), (True, 0, 0, 0), (0.0, 0, 0, 0), [0, 0, 0, 0]]
)
def test_invalid_host_codes_fail_before_k1(codes, monkeypatch):
    reader, snapshot, base = fixture()
    monkeypatch.setattr(reader, "_step", lambda *_: pytest.fail("No debe refinar"))
    with pytest.raises(ValueError):
        apply_routes(reader, base, None, snapshot, choices=codes, context_id=CONTEXT, cutoff=10)


def test_host_k1_budget_and_threshold_policy_follow_the_existing_routes():
    reader, snapshot, base = fixture()
    head = torch.nn.Linear(32, 1, dtype=torch.float64)
    options = dict(context_id=CONTEXT, cutoff=10)
    selected = apply_routes(reader, base, head, snapshot, policy=threshold(cutoff=-1), **options)
    expected, _ = reference(reader, snapshot, base, "k2r")
    assert torch.equal(selected.state, expected)
    maximum = reader.estimated_bytes(len(base), snapshot.count, refinements=1)
    reader.config = replace(reader.config, max_working_bytes=maximum)
    fixed = apply_routes(reader, base, head, snapshot, choices=(0,) * len(base), **options)
    assert fixed.route_codes == (0,) * len(base)
    with pytest.raises(ValueError, match="presupuesto"):
        apply_routes(reader, base, head, snapshot, choices=(5,) * len(base), **options)


def test_threshold_subclasses_are_rejected_before_k1(monkeypatch):
    class UndeclaredPolicy(api().ThresholdRoutePolicy):
        pass

    reader, snapshot, base = fixture()
    policy = UndeclaredPolicy(feature_name="read_rms", cutoff=0.5, route="k4cc")
    monkeypatch.setattr(reader, "_step", lambda *_: pytest.fail("No debe refinar"))
    with pytest.raises(ValueError, match="identidad"):
        apply_routes(reader, base, None, snapshot, policy=policy, context_id=CONTEXT, cutoff=10)


def test_host_routes_reject_a_scalar_working_state_with_a_contract_error():
    with pytest.raises(ValueError):
        apply_routes(None, torch.tensor(1.0), None, choices=(0,))
