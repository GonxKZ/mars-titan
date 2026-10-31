"""Supresiones del mismo episodio en todas las vistas regionales, con fixtures."""

import copy
import importlib
from dataclasses import replace

import pytest
import torch
from test_episodic_readout import CODEC, CONTEXT, source
from test_spaced_utility_readout import AT
from test_spaced_utility_readout import fixture as standard_fixture

from mars_titan.memory.spaced_utility import SpacedUtilityLedger
from mars_titan.models.quantile_head import QuantileHead
from mars_titan.models.titans.bridge_hierarchy import BridgeHierarchyConfig
from mars_titan.models.titans.bridge_intervention import BridgeInterventionConfig
from mars_titan.models.titans.bridge_recurrence import BridgeRecurrenceConfig, BridgeStoppingConfig
from mars_titan.models.titans.episodic_readout import (
    EpisodicReadout,
    EpisodicReadoutConfig,
    apply_episodic_readout,
)
from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot
from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig


def api():
    try:
        return importlib.import_module("mars_titan.memory.spaced_regional_readout")
    except ModuleNotFoundError:
        pytest.fail("Falta el productor E2 para el lector regional")


def options(kind):
    result = dict(cycles=2, topology=kind)
    if kind.startswith("j"):
        result = dict(cycles=0, recurrence=BridgeRecurrenceConfig(int(kind[1:]), 1216))
    elif kind.startswith("adaptive"):
        result = dict(
            cycles=0,
            recurrence=BridgeRecurrenceConfig(
                4,
                1216,
                stopping=BridgeStoppingConfig(0.01, 0.0001, backend=kind),
            ),
        )
    elif kind in ("tree", "star"):
        result = dict(
            regions=("one", "two", "three", "four"),
            topology="self",
            cycles=0,
            hierarchy=BridgeHierarchyConfig(topology=kind),
        )
    elif kind == "masked":
        result = dict(intervention=BridgeInterventionConfig((True, False, True), ((2, 0),)))
    elif kind == "inactive":
        result = dict(intervention=BridgeInterventionConfig((False,) * 3))
    return result


def fixture(kind="ring", *, dtype=torch.float64, neighbors=3, refinements=1, **changes):
    core, prepared, head, _ = standard_fixture(dtype=dtype)
    config = EvidenceBridgeConfig(
        CODEC, hidden_size=32, neighbors=neighbors, **(options(kind) | changes)
    )
    model = EpisodicReadout(
        EpisodicReadoutConfig(
            CODEC,
            hidden_size=32,
            refinements=refinements,
            neighbors=neighbors,
            temperature=config.temperature,
            max_working_bytes=config.max_working_bytes,
            evidence_bridge=config,
        ),
        dtype=dtype,
    ).eval()
    bridge = model.evidence_bridge
    with torch.no_grad():
        bridge.query.weight.zero_()
        bridge.query.bias.zero_()
        bridge.query.weight[0, 0] = 1
        bridge.value.weight.zero_()
        bridge.value.bias.zero_()
        bridge.value.weight[0, 0] = 1
        bridge.region_query.zero_()
        bridge.region_value.zero_()
        bridge.output.weight.zero_()
        for region in range(len(config.regions)):
            bridge.region_query[region, 0, 1] = 0.3 * (region + 1)
            bridge.region_value[region, 0, 0] = 0.5 * (region + 1)
            bridge.output.weight[0, region * 32] = 1 / len(config.regions)
        model.refinement.weight.zero_()
        model.refinement.bias.zero_()
        model.refinement.weight[0, 64] = 1
        model.step_logit.fill_(-1)
    return core, prepared, head, model


def snapshot(count=5, dtype=torch.float64, *, keep=None):
    values = source(count)
    if keep is not None:
        mask = torch.isin(values["ids"], torch.tensor(keep, dtype=torch.int64))
        values = {name: tensor[mask].contiguous() for name, tensor in values.items()}
    return EpisodeSnapshot.create(
        **values, cutoff=AT, codec_id=CODEC, context_id=CONTEXT, dtype=dtype
    )


def apply(prepared, head, model, data, **kwargs):
    return api().apply_regional_spaced_utility_readout(
        prepared, head, model, data, enabled=True, context_id=CONTEXT, cutoff=AT, **kwargs
    )


KINDS = (
    "self",
    "ring",
    "complete",
    "j1",
    "j2",
    "j4",
    "adaptive_reference_full",
    "adaptive_compact",
    "tree",
    "star",
    "masked",
    "inactive",
)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("count", [0, 1, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_regional_deletions_match_the_original_reader_on_independently_filtered_support(
    kind, count, dtype
):
    _, prepared, head, model = fixture(kind, dtype=dtype)
    data = snapshot(count, dtype)
    ordinary = apply_episodic_readout(prepared, head, model, data, context_id=CONTEXT, cutoff=AT)
    prediction, trace = apply(prepared, head, model, data)
    assert torch.equal(prediction.point_predictions, ordinary.point_predictions)
    assert torch.equal(prediction.readout.state, ordinary.readout.state)
    selected = ordinary.readout.reads[0].bridge.evidence_ids
    assert torch.equal(trace.episode_ids, selected)
    assert torch.equal(trace.points[:, 0], prediction.point_predictions)
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    for row in range(len(trace.points)):
        ids = selected[row].tolist()
        for column, deleted in enumerate(ids):
            reduced = snapshot(count, dtype, keep=[i for i in ids if i != deleted])
            state = model(
                prepared.working_state[row : row + 1], reduced, context_id=CONTEXT, cutoff=AT
            ).state
            expected = head(state).squeeze()
            torch.testing.assert_close(
                trace.points[row, column + 1], expected, rtol=tolerance, atol=tolerance
            )
    assert trace.points.grad_fn is None and not trace.points.requires_grad


def test_suppression_does_not_refill_repeat_mac_or_run_another_candidate_search(monkeypatch):
    core, prepared, head, model = fixture(neighbors=2)
    calls = dict(selection=0, projection=0)
    original = torch.argsort

    def sort(*args, **kwargs):
        calls["selection"] += 1
        return original(*args, **kwargs)

    def projected(*_):
        calls["projection"] += 1

    monkeypatch.setattr(torch, "argsort", sort)
    monkeypatch.setattr(core, "prepare", lambda *_a, **_k: pytest.fail("MAC repetido"))
    handle = model.evidence_bridge.value.register_forward_hook(projected)
    data = snapshot()
    before = data.export_cpu()
    try:
        _, trace = apply(prepared, head, model, data)
    finally:
        handle.remove()
    assert calls == dict(selection=1, projection=2)
    assert trace.episode_ids.tolist() == [[10, 20], [10, 20]]
    assert data.export_cpu()["sha256"] == before["sha256"]
    assert prepared.next_state.mac.memory.steps.tolist() == [1, 1]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_saturated_attention_is_recomputed_after_the_selected_episode_is_removed(dtype):
    _, prepared, head, model = fixture(dtype=dtype, temperature=1e-4)
    prediction, trace = apply(prepared, head, model, snapshot(dtype=dtype))
    assert (prediction.readout.reads[0].bridge.evidence_weights == 1).any()
    assert torch.isfinite(trace.points).all()
    for row in range(2):
        reduced = snapshot(dtype=dtype, keep=[20, 30])
        expected = head(
            model(
                prepared.working_state[row : row + 1], reduced, context_id=CONTEXT, cutoff=AT
            ).state
        ).squeeze()
        torch.testing.assert_close(
            trace.points[row, 1],
            expected,
            rtol=2e-6 if dtype == torch.float32 else 1e-12,
            atol=1e-12,
        )


def test_removing_the_last_episode_keeps_the_complete_absent_read_refinement():
    _, prepared, head, model = fixture(neighbors=1)
    with torch.no_grad():
        model.refinement.bias[0] = 0.5
        model.refinement.weight[0, -1] = 0.3
    _, trace = apply(prepared, head, model, snapshot(1))
    empty = head(model(prepared.working_state, snapshot(0), context_id=CONTEXT, cutoff=AT).state)
    assert torch.equal(trace.points[:, 1], empty.squeeze(-1))
    assert not torch.equal(trace.points[:, 1], head(prepared.working_state).squeeze(-1))


def test_mature_gain_uses_the_already_emitted_points_without_repeating_regional_inference(
    monkeypatch,
):
    _, prepared, head, model = fixture()
    data = snapshot()
    prediction, trace = apply(prepared, head, model, data)
    rows = trace.ledger_rows(dict.fromkeys(prepared.next_state.flow_ids, AT + 1))
    ledger = SpacedUtilityLedger(CODEC)
    ledger.begin(AT)
    ledger.issue(data, rows)
    assert ledger.publish() == []
    monkeypatch.setattr(model, "forward", lambda *_a, **_k: pytest.fail("Lectura al madurar"))
    monkeypatch.setattr(head, "forward", lambda *_a, **_k: pytest.fail("Cabeza al madurar"))
    ledger.begin(AT + 1)
    ledger.receive([(row["flow_id"], AT, AT + 1, 0.0) for row in rows])
    receipt = ledger.publish()[0]
    assert [row["point"] for row in rows] == prediction.point_predictions.tolist()
    expected = float((trace.points[:, 1].abs() - trace.points[:, 0].abs()).mean())
    assert receipt["gains"][10]["mean"] == expected
    assert ledger.scores()[10]["sessions"] == 1


@pytest.mark.parametrize("kind", ["ring", "j4", "adaptive_compact", "tree", "masked"])
@pytest.mark.parametrize("steps", [1, 2, 4])
@pytest.mark.parametrize("differentiable", [False, True])
def test_disabled_path_preserves_legacy_outputs_and_gradients(kind, steps, differentiable):
    _, prepared, head, model = fixture(kind, refinements=steps)
    data = snapshot()
    kwargs = dict(context_id=CONTEXT, cutoff=AT, differentiable=differentiable)
    expected = apply_episodic_readout(prepared, head, model, data, **kwargs)
    result, trace = api().apply_regional_spaced_utility_readout(
        prepared, head, model, data, **kwargs
    )
    assert trace is None and torch.equal(result.point_predictions, expected.point_predictions)
    assert torch.equal(result.readout.state, expected.readout.state)
    if differentiable:
        parameters = (head.weight, model.refinement.weight, model.evidence_bridge.value.weight)
        left = torch.autograd.grad(result.point_predictions.sum(), parameters)
        right = torch.autograd.grad(expected.point_predictions.sum(), parameters)
        assert all(torch.equal(a, b) for a, b in zip(left, right, strict=True))


def test_disabled_path_without_an_extension_does_not_call_the_head():
    _, prepared, _, _ = fixture()

    def forbidden(*_):
        pytest.fail("La ruta sin lector no llama otra cabeza")

    result, trace = api().apply_regional_spaced_utility_readout(prepared, forbidden)
    assert result.point_predictions is prepared.point_predictions and trace is None


@pytest.mark.parametrize(
    "fault",
    [
        "steps",
        "mode",
        "training",
        "bridge_training",
        "head_training",
        "head_dtype",
        "head_device",
        "head_kind",
        "head_quantiles",
        "gradient",
        "snapshot",
        "standard",
        "duplicate",
        "unknown",
        "market",
        "clock",
    ],
)
def test_unsupported_contract_fails_before_any_value_projection(monkeypatch, fault):
    _, prepared, head, model = fixture(refinements=2 if fault == "steps" else 1)
    data, kwargs = snapshot(), {}
    if fault == "mode":
        model.config = replace(model.config, mode="no_bank")
    elif fault == "training":
        model.train()
    elif fault == "bridge_training":
        model.evidence_bridge.train()
    elif fault == "head_training":
        head.train()
    elif fault == "head_dtype":
        head = head.float()
    elif fault == "head_device":
        head = head.to("meta")
    elif fault == "head_kind":
        head = torch.nn.Identity().eval()
    elif fault == "head_quantiles":
        prepared = replace(prepared, quantiles=torch.zeros((2, 5), dtype=torch.float64))
    elif fault == "gradient":
        kwargs["differentiable"] = True
    elif fault == "snapshot":
        data = None
    elif fault == "standard":
        model.evidence_bridge = None
    elif fault in ("duplicate", "unknown", "market", "clock"):
        changes = dict(
            duplicate=dict(flow_ids=("US/AAA", "US/AAA")),
            unknown=dict(flow_ids=("XX/AAA", "US/BBB")),
            market=dict(flow_ids=("US/AAA", "CN/BBB")),
            clock=dict(last_prediction_at=(AT, AT + 1)),
        )
        prepared = replace(prepared, next_state=replace(prepared.next_state, **changes[fault]))
    monkeypatch.setattr(model, "forward", lambda *_a, **_k: pytest.fail("Leyó sin contrato"))
    with pytest.raises(ValueError):
        apply(prepared, head, model, data, **kwargs)


@pytest.mark.parametrize("name", ["enabled", "allow_mixed_markets"])
@pytest.mark.parametrize("value", [None, 1, "true"])
def test_flags_require_real_booleans(name, value):
    with pytest.raises(ValueError):
        api().apply_regional_spaced_utility_readout(None, None, **{name: value})


def test_autocast_is_rejected_before_projection(monkeypatch):
    _, prepared, head, model = fixture(dtype=torch.float32)
    monkeypatch.setattr(model, "forward", lambda *_a, **_k: pytest.fail("Autocast implícito"))
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        apply(prepared, head, model, snapshot(dtype=torch.float32))


def test_combined_memory_budget_is_checked_before_the_ordinary_projection(monkeypatch):
    _, prepared, head, model = fixture()
    low = model.estimated_bytes(2, 5, refinements=1)
    high = model.estimated_bytes(2, 5, refinements=4)
    _, prepared, head, model = fixture(max_working_bytes=(low + high) // 2)
    monkeypatch.setattr(model, "forward", lambda *_a, **_k: pytest.fail("Presupuesto tardío"))
    with pytest.raises(ValueError, match="presupuesto"):
        apply(prepared, head, model, snapshot())


def test_whole_regional_update_budget_cannot_be_renewed_for_each_deletion(monkeypatch):
    _, prepared, head, model = fixture(
        "j4", recurrence=BridgeRecurrenceConfig(4, max_node_updates_per_row=12)
    )
    monkeypatch.setattr(model, "forward", lambda *_a, **_k: pytest.fail("Cuota renovada"))
    with pytest.raises(ValueError, match="presupuesto"):
        apply(prepared, head, model, snapshot())


def test_existing_quantile_head_keeps_its_original_median_for_the_ordinary_emission():
    _, prepared, _, model = fixture("tree")
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        head = QuantileHead(32, dtype=torch.float64).eval()
    prepared = replace(prepared, quantiles=head(prepared.working_state).detach())
    original = apply_episodic_readout(
        prepared, head, model, snapshot(), context_id=CONTEXT, cutoff=AT
    )
    result, trace = apply(prepared, head, model, snapshot())
    assert torch.equal(result.quantiles, original.quantiles)
    assert torch.equal(trace.points[:, 0], result.quantiles[:, 2])
    for row in range(2):
        reduced = snapshot(keep=[20, 30])
        state = model(
            prepared.working_state[row : row + 1], reduced, context_id=CONTEXT, cutoff=AT
        ).state
        torch.testing.assert_close(trace.points[row, 1], head(state)[0, 2], rtol=1e-12, atol=1e-12)


def test_each_deletion_keeps_the_adaptive_prefix_rule_instead_of_forcing_all_steps():
    _, prepared, head, model = fixture(
        "adaptive_compact",
        recurrence=BridgeRecurrenceConfig(
            4, 1216, stopping=BridgeStoppingConfig(1e6, 0, backend="adaptive_compact")
        ),
    )
    _, trace = apply(prepared, head, model, snapshot())
    reduced = snapshot(keep=[20, 30])
    ordinary = model(prepared.working_state, reduced, context_id=CONTEXT, cutoff=AT)
    assert ordinary.reads[0].bridge.selected_steps.tolist() == [1, 1]
    expected = head(ordinary.state).squeeze(-1)
    torch.testing.assert_close(trace.points[:, 1], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("fault", ["context", "codec", "changed", "future"])
def test_ineligible_or_changed_snapshot_cannot_be_read(fault):
    _, prepared, head, model = fixture()
    data = snapshot()
    if fault in ("context", "codec"):
        object.__setattr__(data, fault + "_id", "c" * 64)
    elif fault == "changed":
        data._values[1][0, 0] += 1
    else:
        object.__setattr__(data, "cutoff", AT + 1)
    with pytest.raises(ValueError):
        apply(prepared, head, model, data)


def test_recovery_and_asset_reordering_keep_points_and_shared_ids():
    _, prepared, head, model = fixture("adaptive_compact")
    data = snapshot()
    before = torch.get_rng_state().clone()
    ordinary, trace = apply(prepared, head, model, data)
    assert torch.equal(before, torch.get_rng_state()) and not torch.cuda.is_initialized()
    _, _, _, replica = fixture("adaptive_compact")
    replica.load_state_dict(copy.deepcopy(model.state_dict()))
    restored = EpisodeSnapshot.restore(
        data.export_cpu(),
        codec_id=CODEC,
        context_id=CONTEXT,
        cutoff=AT,
        device="cpu",
        dtype=torch.float64,
    )
    result, recovered = apply(prepared, head, replica, restored)
    assert torch.equal(result.point_predictions, ordinary.point_predictions)
    assert torch.equal(trace.points, recovered.points)
    reordered = replace(
        prepared,
        working_state=prepared.working_state.flip(0),
        next_state=replace(prepared.next_state, flow_ids=prepared.next_state.flow_ids[::-1]),
    )
    _, other = apply(reordered, head, model, data)
    assert torch.equal(trace.points, other.points.flip(0))
    assert torch.equal(trace.episode_ids, other.episode_ids.flip(0))
    mixed = replace(prepared, next_state=replace(prepared.next_state, flow_ids=("US/A", "CN/B")))
    _, other = apply(mixed, head, model, data, allow_mixed_markets=True)
    rows = other.ledger_rows(dict.fromkeys(mixed.next_state.flow_ids, AT + 1))
    assert [row["flow_id"] for row in rows] == ["US/A", "CN/B"]


@pytest.mark.parametrize("regions", [1, 8])
def test_dense_region_weights_and_distinct_asset_queries_preserve_the_complete_operator(regions):
    _, prepared, head, model = fixture("j4", regions=tuple(f"view_{i}" for i in range(regions)))
    generator = torch.Generator().manual_seed(871)
    with torch.no_grad():
        for parameter in (*model.parameters(), *head.parameters()):
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, dtype=torch.float64) * 0.1
            )
    prepared = replace(
        prepared, working_state=torch.randn((2, 32), generator=generator, dtype=torch.float64)
    )
    data = snapshot()
    ordinary = apply_episodic_readout(prepared, head, model, data, context_id=CONTEXT, cutoff=AT)
    result, trace = apply(prepared, head, model, data)
    assert torch.equal(result.point_predictions, ordinary.point_predictions)
    ids = ordinary.readout.reads[0].bridge.evidence_ids
    assert torch.equal(trace.episode_ids, ids)
    for row in range(2):
        for column, deleted in enumerate(ids[row].tolist()):
            reduced = snapshot(keep=[i for i in ids[row].tolist() if i != deleted])
            state = model(
                prepared.working_state[row : row + 1], reduced, context_id=CONTEXT, cutoff=AT
            ).state
            torch.testing.assert_close(
                trace.points[row, column + 1], head(state).squeeze(), rtol=1e-11, atol=1e-12
            )
    reversed_preparation = replace(
        prepared,
        working_state=prepared.working_state.flip(0),
        next_state=replace(prepared.next_state, flow_ids=prepared.next_state.flow_ids[::-1]),
    )
    _, reversed_trace = apply(reversed_preparation, head, model, data)
    torch.testing.assert_close(trace.points, reversed_trace.points.flip(0), rtol=1e-12, atol=1e-12)
    assert torch.equal(trace.episode_ids, reversed_trace.episode_ids.flip(0))


@pytest.mark.cuda
@pytest.mark.parametrize("count", [0, 1, 5])
def test_cuda_preserves_regional_prediction_and_detaches_all_diagnostics(count):
    _, prepared, head, model = fixture("j4", dtype=torch.float32)
    prepared = replace(prepared, working_state=prepared.working_state.to("cuda:0"))
    head, model = head.to("cuda:0"), model.to("cuda:0")
    data = EpisodeSnapshot.restore(
        snapshot(count, torch.float32).export_cpu(),
        codec_id=CODEC,
        context_id=CONTEXT,
        cutoff=AT,
        device="cuda:0",
        dtype=torch.float32,
    )
    expected = apply_episodic_readout(prepared, head, model, data, context_id=CONTEXT, cutoff=AT)
    result, trace = apply(prepared, head, model, data)
    assert torch.equal(result.point_predictions, expected.point_predictions)
    assert trace.points.device == torch.device("cuda:0") and trace.points.grad_fn is None
    assert torch.isfinite(trace.points).all()
    assert len(trace.ledger_rows(dict.fromkeys(prepared.next_state.flow_ids, AT + 1))) == 2
