"""Recorrido de las rutas con el mismo padre y etiquetas que maduran después."""

import importlib
from dataclasses import replace

import pytest
import torch

from mars_titan.training.mars_titan_run import MarsTitanInference, retention_config
from tests.training.test_mars_titan_run import explicit_fastpath as explicit_fastpath
from tests.training.test_mars_titan_run import native as native
from tests.training.test_mars_titan_run import parent, reader, recipe
from tests.training.test_mars_titan_run import shared as shared


def build(streams, native, *, mode="collect", route=None, traces=None, block_rows=2, bank_seed=73):
    module = importlib.import_module("mars_titan.training.refinement_inference")
    plan = recipe(block_rows=block_rows, bank_seed=bank_seed)
    readout, codec = reader(streams, "m1")
    return module.RefinementInference(
        parent(streams),
        readout.eval().requires_grad_(False),
        plan,
        admission="m1",
        retention=retention_config(plan, "m1"),
        native=native,
        codec=codec,
        world="fixture",
        fold="0",
        mode=mode,
        route=route,
        trace=traces.append if traces is not None else None,
    )


def test_collection_records_alternatives_only_when_their_labels_mature(shared, native):
    _, streams = shared
    traces = []
    engine = build(streams, native, traces=traces)
    rows = []
    result = engine.evaluate(streams["validation"], rows=rows)
    assert result["samples"] == len(traces) == len(rows)
    assert all(record["decision_at"] < record["maturity_at"] for record in traces)
    assert all(
        len(record["features"]) == 9 and len(record["predictions"]) == 6 for record in traces
    )
    for record, row in zip(traces, rows, strict=True):
        assert (record["flow_id"], record["decision_at"]) == row[:2]
        assert record["predictions"][0] == row[2]
        assert record["target"] == row[3]
    assert result["refinements"] == 9 * result["predictions"]
    assert result["mac_updates"] == result["observations"]
    assert result["updates"] == 0
    assert not engine.pending_traces


@pytest.mark.parametrize("route", ["k1", "k2c", "k2r", "k4cc", "k4cr", "k4rc"])
def test_fixed_route_uses_the_same_mature_bank_as_collection(shared, native, route):
    from mars_titan.models.titans.refinement_routes import ROUTES

    _, streams = shared
    traces = []
    reference = build(streams, native, traces=traces)
    reference.evaluate(streams["validation"])
    fixed = build(streams, native, mode="fixed", route=route)
    rows = []
    result = fixed.evaluate(streams["validation"], rows=rows)
    assert result["refinements"] == int(route[1]) * result["predictions"]
    expected = [record["predictions"][ROUTES.index(route)] for record in traces]
    torch.testing.assert_close(
        torch.tensor([r[2] for r in rows]), torch.tensor(expected), rtol=0, atol=0
    )


def test_k1_is_identical_to_existing_reader_inference(shared, native):
    _, streams = shared
    fixed = build(streams, native, mode="fixed", route="k1")
    ordinary = MarsTitanInference(
        fixed.predictor,
        fixed.readout,
        fixed.recipe,
        admission="m1",
        retention=fixed.retention,
        native=native,
        codec=fixed.codec,
        world="fixture",
        fold="0",
    )
    left, right = [], []
    a = fixed.evaluate(streams["validation"], rows=left)
    b = ordinary.evaluate(streams["validation"], rows=right)
    assert left == right
    assert a["session_mae"] == b["session_mae"]
    assert a["mac_updates"] == b["mac_updates"]


def test_second_pass_restarts_states_and_bank_and_preserves_parameters(shared, native):
    _, streams = shared
    engine = build(streams, native, mode="fixed", route="k4cr")
    before = {name: value.clone() for name, value in engine.readout.named_parameters()}
    first, second = [], []
    engine.evaluate(streams["validation"], rows=first)
    engine.evaluate(streams["validation"], rows=second)
    assert first == second
    assert all(
        torch.equal(before[name], value) for name, value in engine.readout.named_parameters()
    )


def test_invalid_modes_fail_before_consuming_the_source(shared, native):
    _, streams = shared
    with pytest.raises(ValueError, match="modo"):
        build(streams, native, mode="dynamic_unbounded")
    with pytest.raises(ValueError, match="ruta"):
        build(streams, native, mode="fixed", route="k4rr")


def test_collection_calls_mac_once_per_observation_not_once_per_route(shared, native, monkeypatch):
    _, streams = shared
    engine = build(streams, native, traces=[])
    observations = []
    prepare = engine.predictor.prepare

    def counted(batch, *args, **kwargs):
        observations.extend(zip(batch.flow_ids, batch.prediction_at, strict=True))
        return prepare(batch, *args, **kwargs)

    monkeypatch.setattr(engine.predictor, "prepare", counted)
    result = engine.evaluate(streams["validation"])
    assert len(observations) == len(set(observations)) == result["observations"]
    assert len(observations) == result["mac_updates"]


def test_identity_includes_bank_and_recipe_and_reports_executed_route_counts(shared, native):
    _, streams = shared
    engine = build(streams, native, mode="fixed", route="k4rc")
    original = engine.identity()
    other = build(streams, native, mode="fixed", route="k4rc", bank_seed=74)
    assert other.identity() != original
    result = engine.evaluate(streams["validation"])
    assert result["route_count_scope"] == "executed_routes"
    assert result["route_counts"]["k4rc"] == result["predictions"]
    assert sum(result["route_counts"].values()) == result["predictions"]
    engine.retention = replace(engine.retention, seed=74)
    with pytest.raises(ValueError, match="identificación"):
        engine.evaluate(streams["validation"])
