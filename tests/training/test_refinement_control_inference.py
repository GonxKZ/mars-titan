"""Controles desplegados sobre un padre congelado y etiquetas maduras."""

import importlib
from dataclasses import replace

import pytest
import torch

from mars_titan.training.mars_titan_run import retention_config
from mars_titan.training.refinement_inference import RefinementInference
from tests.training.test_mars_titan_run import explicit_fastpath as explicit_fastpath
from tests.training.test_mars_titan_run import native as native
from tests.training.test_mars_titan_run import parent, reader, recipe
from tests.training.test_mars_titan_run import shared as shared
from tests.training.test_refinement_inference import build as reference


def controls():
    return importlib.import_module("mars_titan.models.titans.refinement_controls")


def build(streams, native, *, block_rows=2, admission="m1", **options):
    plan = recipe(block_rows=block_rows)
    readout, codec = reader(streams, admission)
    return RefinementInference(
        parent(streams),
        readout.eval().requires_grad_(False),
        plan,
        admission=admission,
        retention=retention_config(plan, admission),
        native=native,
        codec=codec,
        world="fixture",
        fold="0",
        **options,
    )


@pytest.mark.parametrize("kind", ["threshold", "hashed"])
@pytest.mark.parametrize("admission", ["m0", "m1"])
def test_deployed_controls_match_the_fixed_route_and_count_every_prediction(
    shared, native, kind, admission
):
    _, streams = shared
    module = controls()
    options = (
        dict(
            mode="policy",
            policy=module.ThresholdRoutePolicy(
                feature_name="memory_present", cutoff=-1, route="k2r"
            ),
        )
        if kind == "threshold"
        else dict(mode="hashed", schedule=module.HashedRoutes(counts=(0, 0, 1, 0, 0, 0), seed=1))
    )
    engine = build(streams, native, admission=admission, **options)
    fixed = build(streams, native, admission=admission, mode="fixed", route="k2r")
    actual, expected = [], []
    result = engine.evaluate(streams["validation"], rows=actual)
    fixed.evaluate(streams["validation"], rows=expected)
    assert actual == expected
    assert result["route_counts"]["k2r"] == result["predictions"]
    assert sum(result["route_counts"].values()) == result["predictions"]
    assert result["refinements"] == 2 * result["predictions"]
    assert result["mac_updates"] == result["observations"]
    assert result["updates"] == 0
    assert result["route_count_scope"] == "executed_routes"
    if admission == "m0":
        assert result["searches"] == result["value_projections"] == 0


def test_hashed_policy_uses_emission_identity_and_the_same_bank_as_collection(shared, native):
    _, streams = shared
    module = controls()
    schedule = module.HashedRoutes(counts=(1, 1, 1, 1, 1, 1), seed=7)
    traces = []
    reference(streams, native, traces=traces).evaluate(streams["validation"])
    engine = build(streams, native, mode="hashed", schedule=schedule)
    rows = []
    result = engine.evaluate(streams["validation"], rows=rows)
    codes = schedule.choices(
        tuple(t["flow_id"] for t in traces), tuple(t["decision_at"] for t in traces)
    )
    expected = [trace["predictions"][code] for trace, code in zip(traces, codes, strict=True)]
    torch.testing.assert_close(
        torch.tensor([row[2] for row in rows]), torch.tensor(expected), rtol=0, atol=0
    )
    assert result["mac_updates"] == result["observations"]
    assert engine.identity()["schedule"] == schedule.identity()
    assert sum(result["route_counts"].values()) == result["predictions"]
    assert (
        engine.identity()
        != build(streams, native, mode="hashed", schedule=replace(schedule, seed=8)).identity()
    )


def test_hashed_inference_is_invariant_to_microbatching_and_repeated_passes(shared, native):
    _, streams = shared
    schedule = controls().HashedRoutes(counts=(1,) * 6, seed=9)
    small = build(streams, native, mode="hashed", schedule=schedule, block_rows=1)
    larger = build(streams, native, mode="hashed", schedule=schedule, block_rows=2)
    first, second, repeated = [], [], []
    a = small.evaluate(streams["validation"], rows=first)
    b = larger.evaluate(streams["validation"], rows=second)
    small.evaluate(streams["validation"], rows=repeated)
    assert first == repeated
    assert [(row[0], row[1]) for row in first] == [(row[0], row[1]) for row in second]
    torch.testing.assert_close(
        torch.tensor([row[2] for row in first]),
        torch.tensor([row[2] for row in second]),
        rtol=1e-6,
        atol=1e-8,
    )
    assert a["route_counts"] == b["route_counts"]


@pytest.mark.parametrize("kind", ["threshold", "hashed", "replaced_policy"])
def test_control_changes_are_rejected_before_consuming_the_source(shared, native, kind):
    _, streams = shared
    module = controls()
    if kind != "hashed":
        policy = module.ThresholdRoutePolicy(feature_name="read_rms", cutoff=0.5, route="k4rc")
        engine = build(streams, native, mode="policy", policy=policy)
        if kind == "threshold":
            policy.cutoff = 0.6
        else:
            engine.policy = module.ThresholdRoutePolicy(
                feature_name="read_rms", cutoff=0.6, route="k4rc"
            )
    else:
        schedule = module.HashedRoutes(counts=(1,) * 6, seed=5)
        engine = build(streams, native, mode="hashed", schedule=schedule)
        engine.schedule = replace(schedule, seed=6)
    with pytest.raises(ValueError, match="cambi"):
        engine.evaluate(streams["validation"])


@pytest.mark.parametrize(
    "fault", ["missing_schedule", "foreign_schedule", "mixed_policy", "schedule_outside_mode"]
)
def test_invalid_control_modes_fail_during_construction(shared, native, fault):
    _, streams = shared
    schedule = controls().HashedRoutes(counts=(1,) * 6, seed=0)
    options = dict(mode="hashed", schedule=schedule)
    if fault == "missing_schedule":
        options.pop("schedule")
    elif fault == "foreign_schedule":
        options["schedule"] = object()
    elif fault == "mixed_policy":
        options["policy"] = object()
    else:
        options.update(mode="fixed", route="k1")
    with pytest.raises(ValueError):
        build(streams, native, **options)
