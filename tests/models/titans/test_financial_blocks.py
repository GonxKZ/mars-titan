"""Reunión y transporte de estados ficticios, sin optimizadores ni corpus real."""

import copy
import importlib
from dataclasses import replace

import pytest
import torch
from test_financial_adapter import api, raw_batch, setup
from test_financial_control import configured, prepare


def blocks_api():
    try:
        return importlib.import_module("mars_titan.models.titans.financial_blocks")
    except ModuleNotFoundError:
        pytest.fail("Falta el puente de estados por bloques")


def tensors(state):
    return [state.observed_steps] + (
        [*state.mac.memory.weights, *state.mac.memory.momentum, state.mac.memory.steps]
        if state.mac
        else []
    )


def assert_state_equal(left, right):
    for name in ("config_id", "parameter_id", "flow_ids", "last_sample_ids", "last_prediction_at"):
        assert getattr(left, name) == getattr(right, name)
    for first, second in zip(tensors(left), tensors(right), strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def references(state, block_id="a" * 64):
    return {
        flow: dict(
            block_id=block_id,
            row=index,
            config_id=state.config_id,
            parameter_id=state.parameter_id,
            observed_steps=int(state.observed_steps[index]),
            last_prediction_at=state.last_prediction_at[index],
            last_sample_id=state.last_sample_ids[index],
        )
        for index, flow in enumerate(state.flow_ids)
    }


@pytest.mark.parametrize(
    "variant", ["transformer_direct", "mac_disabled", "mac_frozen", "mac_online"]
)
def test_cpu_export_and_explicit_restore_preserve_legacy_api_and_own_storage(variant):
    model, batch = setup(variant)
    state = model.prepare(
        batch, model.initial_state(batch.flow_ids), differentiable=True
    ).next_state
    legacy = model.restore_state(model.export_state(state))
    payload = model.export_state_cpu(state)
    restored = model.restore_state(payload, device="cpu")
    assert_state_equal(legacy, restored)
    for original, value in zip(tensors(state), tensors(restored), strict=True):
        assert value.device.type == "cpu"
        assert value.grad_fn is None and not value.requires_grad
        assert value.is_contiguous() and value.storage_offset() == 0
        assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
        assert value.data_ptr() != original.data_ptr()
    following = api().DecisionBatch.from_corpus(
        raw_batch(at=batch.prediction_at[0] + 10), model.config.inputs, dtype=torch.float64
    )
    expected = model.prepare(following, legacy)
    actual = model.prepare(following, restored)
    torch.testing.assert_close(expected.point_predictions, actual.point_predictions, rtol=0, atol=0)
    assert_state_equal(expected.next_state, actual.next_state)


def test_gather_uses_requested_rows_order_and_keeps_input_blocks_independent():
    model, batch = setup()
    known = model.prepare(batch, model.initial_state(batch.flow_ids)).next_state
    new = model.initial_state(("US/CCC",))
    first, second = model.export_state_cpu(known), model.export_state_cpu(new)
    requested = ("US/CCC", "US/AAA")
    sources = {"US/AAA": (first, 0), "US/CCC": (second, 0)}
    rng = torch.random.get_rng_state().clone()
    gathered = model.gather_state(sources, requested)
    assert gathered.flow_ids == requested
    assert gathered.observed_steps.tolist() == [0, 1]
    assert gathered.last_prediction_at == (-1, known.last_prediction_at[0])
    for index, source in enumerate((new, known)):
        for result, original in zip(tensors(gathered), tensors(source), strict=True):
            torch.testing.assert_close(result[index], original[0], rtol=0, atol=0)
            assert result.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
    gathered.mac.memory.weights[0].zero_()
    assert first["mac"]["memory"]["weights"][0].abs().sum() > 0
    assert torch.equal(rng, torch.random.get_rng_state())


def test_repeated_source_payload_is_charged_only_once():
    model, batch = setup()
    state = model.initial_state(batch.flow_ids)
    payload = model.export_state_cpu(state)
    sources = {flow: (payload, index) for index, flow in enumerate(batch.flow_ids)}
    budget = model.state_usage(state)["total_bytes"]
    result = model.gather_state(sources, batch.flow_ids, max_source_bytes=budget)
    assert_state_equal(result, state)
    with pytest.raises(ValueError, match="presupuesto"):
        model.gather_state(sources, batch.flow_ids, max_source_bytes=budget - 1)


@pytest.mark.parametrize(
    "case", ["missing", "duplicate", "wrong_row", "bool_row", "identity", "dtype", "nan"]
)
def test_gather_rejects_invalid_sources_without_initializing(case):
    model, batch = setup()
    payload = model.export_state_cpu(model.initial_state(batch.flow_ids))
    requested = ("US/AAA",)
    sources = {"US/AAA": (payload, 0)}
    if case == "missing":
        sources = {}
    elif case == "duplicate":
        requested = ("US/AAA", "US/AAA")
    elif case == "wrong_row":
        sources["US/AAA"] = (payload, 1)
    elif case == "bool_row":
        sources["US/AAA"] = (payload, False)
    elif case == "identity":
        payload["parameter_id"] = "0" * 64
    elif case == "dtype":
        payload["mac"]["memory"]["weights"] = tuple(
            value.float() for value in payload["mac"]["memory"]["weights"]
        )
    else:
        payload["mac"]["memory"]["weights"][0][0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        model.gather_state(sources, requested)


def test_explicit_device_must_match_predictor_before_any_transfer():
    model, batch = setup()
    payload = model.export_state_cpu(model.initial_state(batch.flow_ids))
    with pytest.raises(ValueError, match="dispositivo"):
        model.restore_state(payload, device="cuda:0")


def test_reference_update_replaces_exactly_the_requested_ids_and_keeps_other_rows():
    model, batch = setup()
    state = model.prepare(batch, model.initial_state(batch.flow_ids)).next_state
    index = references(state)
    previous = model.select_state(state, ("US/AAA",))
    following_batch = (
        api()
        .DecisionBatch.from_corpus(
            raw_batch(at=batch.prediction_at[0] + 10), model.config.inputs, dtype=torch.float64
        )
        .select([0])
    )
    following = model.prepare(following_batch, previous).next_state
    updated = blocks_api().replace_state_references(model, index, previous, following, "b" * 64)
    assert set(updated) == set(index)
    assert updated["US/BBB"] == index["US/BBB"]
    assert updated["US/AAA"]["block_id"] == "b" * 64
    assert updated["US/AAA"]["observed_steps"] == 2
    assert index["US/AAA"]["observed_steps"] == 1
    updated["US/BBB"]["row"] = 99
    assert index["US/BBB"]["row"] == 1


@pytest.mark.parametrize(
    "case", ["stale_previous", "backward", "jump", "different_ids", "different_parameters"]
)
def test_reference_update_rejects_unconfirmed_or_backward_state(case):
    model, batch = setup()
    previous = model.prepare(batch, model.initial_state(batch.flow_ids)).next_state
    index = references(previous)
    following_batch = api().DecisionBatch.from_corpus(
        raw_batch(at=batch.prediction_at[0] + 10), model.config.inputs, dtype=torch.float64
    )
    following = model.prepare(following_batch, previous).next_state
    if case == "stale_previous":
        index["US/AAA"]["observed_steps"] = 3
    elif case == "backward":
        following = previous
    elif case == "jump":
        following = replace(
            following,
            observed_steps=following.observed_steps + 1,
            mac=replace(
                following.mac,
                memory=replace(following.mac.memory, steps=following.mac.memory.steps + 1),
            ),
        )
    elif case == "different_ids":
        following = model.select_state(following, ("US/AAA",))
    else:
        following = replace(following, parameter_id="0" * 64)
    before = copy.deepcopy(index)
    with pytest.raises(ValueError):
        blocks_api().replace_state_references(model, index, previous, following, "b" * 64)
    assert index == before


@pytest.mark.parametrize(
    "case",
    [
        "graph",
        "inf",
        "alias",
        "schema",
        "mac_identity",
        "memory_identity",
        "unexpected_mac",
        "shape",
        "layers",
        "strides",
        "device",
        "entry",
        "negative_row",
    ],
)
def test_source_validation_rejects_corruption_before_collecting(monkeypatch, case):
    model, batch = setup("transformer_direct" if case == "unexpected_mac" else "mac_online")
    payload = model.export_state_cpu(model.initial_state(batch.flow_ids))
    sources = {"US/AAA": (payload, 0)}
    if case == "graph":
        payload["mac"]["memory"]["weights"][0].requires_grad_()
    elif case == "inf":
        payload["mac"]["memory"]["weights"][0][0, 0, 0] = float("inf")
    elif case == "alias":
        payload["mac"]["memory"]["momentum"] = payload["mac"]["memory"]["weights"]
    elif case == "schema":
        payload["schema_version"] = True
    elif case == "mac_identity":
        payload["mac"]["configuration"]["heads"] = 2
    elif case == "memory_identity":
        payload["mac"]["memory"]["configuration"]["normalize_qk"] = False
    elif case == "unexpected_mac":
        payload["mac"] = {}
    elif case == "shape":
        weights = payload["mac"]["memory"]["weights"]
        payload["mac"]["memory"]["weights"] = (weights[0][:1].clone(), weights[1])
    elif case == "layers":
        payload["mac"]["memory"]["weights"] = list(payload["mac"]["memory"]["weights"])
    elif case == "strides":
        weights = payload["mac"]["memory"]["weights"]
        payload["mac"]["memory"]["weights"] = (weights[0].transpose(1, 2), weights[1])
    elif case == "device":
        weights = payload["mac"]["memory"]["weights"]
        payload["mac"]["memory"]["weights"] = (
            torch.empty_like(weights[0], device="meta"),
            weights[1],
        )
    elif case == "entry":
        sources["US/AAA"] = [payload, 0]
    else:
        sources["US/AAA"] = (payload, -1)
    monkeypatch.setattr(torch, "cat", lambda *args, **kwargs: pytest.fail("Copia antes de validar"))
    with pytest.raises(ValueError):
        model.gather_state(sources, ("US/AAA",))


def test_source_and_result_budgets_are_checked_before_collecting(monkeypatch):
    model, batch = setup()
    state = model.initial_state(batch.flow_ids)
    first = model.export_state_cpu(model.select_state(state, ("US/AAA",)))
    second = model.export_state_cpu(model.select_state(state, ("US/BBB",)))
    sources = {"US/AAA": (first, 0), "US/BBB": (second, 0)}
    per_source = model.state_usage(model.select_state(state, ("US/AAA",)))["total_bytes"]
    monkeypatch.setattr(torch, "cat", lambda *args, **kwargs: pytest.fail("Copia antes del límite"))
    with pytest.raises(ValueError, match="presupuesto agregado"):
        model.gather_state(sources, batch.flow_ids, max_source_bytes=2 * per_source - 1)

    limited = api().FinancialPredictor(
        replace(model.config, max_state_bytes=50_000), dtype=torch.float64
    )
    first = limited.export_state_cpu(limited.initial_state(("US/AAA",)))
    second = limited.export_state_cpu(limited.initial_state(("US/BBB",)))
    with pytest.raises(ValueError, match="presupuesto"):
        limited.gather_state({"US/AAA": (first, 0), "US/BBB": (second, 0)}, batch.flow_ids)


def test_source_budget_counts_full_retained_storage_before_copy(monkeypatch):
    model, batch = setup()
    state = model.initial_state(batch.flow_ids)
    payload = model.export_state_cpu(state)
    payload["observed_steps"] = torch.zeros(100_000, dtype=torch.int64)[:2]
    monkeypatch.setattr(torch, "cat", lambda *args, **kwargs: pytest.fail("Copia antes del límite"))
    with pytest.raises(ValueError, match="presupuesto"):
        model.gather_state(
            {"US/AAA": (payload, 0)},
            ("US/AAA",),
            max_source_bytes=model.state_usage(state)["total_bytes"],
        )


@pytest.mark.parametrize("bad_budget", [0, True, 2 * 1024**3 + 1])
def test_source_budget_rejects_invalid_values(bad_budget):
    model, batch = setup()
    payload = model.export_state_cpu(model.initial_state(batch.flow_ids))
    with pytest.raises(ValueError):
        model.gather_state({"US/AAA": (payload, 0)}, ("US/AAA",), max_source_bytes=bad_budget)


def test_new_flow_requires_explicit_initial_state():
    model, batch = setup()
    previous = model.initial_state(batch.flow_ids)
    following = model.prepare(batch, previous).next_state
    index = blocks_api().replace_state_references(model, {}, previous, following, "b" * 64)
    assert index == references(following, "b" * 64)
    next_batch = api().DecisionBatch.from_corpus(
        raw_batch(at=batch.prediction_at[0] + 10), model.config.inputs, dtype=torch.float64
    )
    advanced = model.prepare(next_batch, following).next_state
    with pytest.raises(ValueError, match="inicial explícito"):
        blocks_api().replace_state_references(model, {}, following, advanced, "c" * 64)


@pytest.mark.parametrize(
    "fault",
    [
        "fields",
        "block_id",
        "row",
        "count",
        "configuration",
        "duplicate_slot",
        "cursor",
        "missing_value",
        "metadata_budget",
        "serialized_metadata_budget",
        "count_budget",
    ],
)
def test_reference_validation_rejects_corrupt_or_oversized_index(monkeypatch, fault):
    model, batch = setup()
    previous = model.initial_state(batch.flow_ids)
    following = model.prepare(batch, previous).next_state
    index = references(previous)
    if fault == "fields":
        index["US/AAA"]["extra"] = 0
    elif fault == "block_id":
        index["US/AAA"]["block_id"] = "0" * 63
    elif fault == "row":
        index["US/AAA"]["row"] = 256
    elif fault == "count":
        index["US/AAA"]["observed_steps"] = True
    elif fault == "configuration":
        index["US/AAA"]["config_id"] = "0" * 64
    elif fault == "duplicate_slot":
        index["US/BBB"]["row"] = 0
    elif fault == "cursor":
        index["US/AAA"]["last_prediction_at"] = batch.prediction_at[0]
    elif fault == "missing_value":
        index["US/AAA"]["last_sample_id"] = "invalid"
    elif fault == "metadata_budget":
        monkeypatch.setattr(blocks_api(), "MAX_REFERENCE_BYTES", 1)
    elif fault == "serialized_metadata_budget":
        monkeypatch.setattr(blocks_api(), "MAX_REFERENCE_BYTES", 2500)
    else:
        monkeypatch.setattr(blocks_api(), "MAX_REFERENCE_FLOWS", 1)
    before = copy.deepcopy(index)
    with pytest.raises(ValueError):
        blocks_api().replace_state_references(model, index, previous, following, "b" * 64)
    assert index == before


def test_large_census_stays_in_physical_blocks_and_preserves_absent_references():
    model, batch = setup("transformer_direct")
    flows = tuple(f"US/F{index:04d}" for index in range(5676))
    payloads, index = {}, {}
    for start in range(0, len(flows), 256):
        state = model.initial_state(flows[start : start + 256])
        block_id = f"{start:064x}"
        payloads[block_id] = model.export_state_cpu(state)
        index.update(references(state, block_id))
    assert len(payloads) == 23
    requested = (flows[-1], flows[0], flows[255], flows[256])
    gathered = model.gather_state(
        {flow: (payloads[index[flow]["block_id"]], index[flow]["row"]) for flow in requested},
        requested,
    )
    assert gathered.flow_ids == requested
    at = batch.prediction_at[0]
    following = replace(
        gathered,
        observed_steps=gathered.observed_steps + 1,
        last_prediction_at=(at,) * len(requested),
        last_sample_ids=tuple(f"{flow}/{at}" for flow in requested),
    )
    updated = blocks_api().replace_state_references(model, index, gathered, following, "f" * 64)
    assert len(updated) == 5676
    assert all(updated[flow] == index[flow] for flow in flows if flow not in requested)
    with pytest.raises(ValueError, match="lote de flujos"):
        model.gather_state({}, flows[:257])


def test_restore_owns_mutable_cursor_sequences_from_serialized_payload():
    model, batch = setup()
    payload = model.export_state_cpu(model.initial_state(batch.flow_ids))
    for name in ("flow_ids", "last_sample_ids", "last_prediction_at"):
        payload[name] = list(payload[name])
    restored = model.restore_state(payload, device="cpu")
    payload["flow_ids"][0] = "US/OTHER"
    payload["last_sample_ids"][0] = "changed"
    payload["last_prediction_at"][0] = 0
    assert restored.flow_ids == batch.flow_ids
    assert restored.last_sample_ids == (None, None)
    assert restored.last_prediction_at == (-1, -1)


def test_gathered_blocks_reuse_one_logical_control_plan():
    model, batch = configured("penalty", max_flows=1)
    state = model.initial_state(batch.flow_ids)
    plan = model.local_control.select_flows(
        batch.flow_ids, state.observed_steps, context_id="a" * 64
    )
    whole = prepare(model, batch, state, differentiable=True, selection=plan)
    outputs = []
    for index, flow in enumerate(batch.flow_ids):
        payload = model.export_state_cpu(model.select_state(state, (flow,)))
        gathered = model.gather_state({flow: (payload, 0)}, (flow,))
        outputs.append(
            prepare(model, batch.select([index]), gathered, differentiable=True, selection=plan)
        )
    measured = [part.local_control for part in outputs if part.local_control is not None]
    assert len(measured) == 1
    torch.testing.assert_close(
        sum(part.penalty for part in measured),
        whole.local_control.penalty,
        rtol=1e-10,
        atol=1e-12,
    )
    assert sum(part.reevaluations for part in measured) == model.local_control.config.rank
    assert all(part.selection_id == plan.fingerprint() for part in measured)
    assert all(part.next_state.mac.memory.steps.tolist() == [1] for part in outputs)


def test_gather_rejects_unbounded_iterables_without_consuming_them():
    model, _ = setup()
    consumed = []

    def ids():
        for index in range(300):
            consumed.append(index)
            yield f"US/F{index}"

    with pytest.raises(ValueError):
        model.gather_state({}, ids())
    assert consumed == []
