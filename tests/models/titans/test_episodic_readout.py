"""Lecturas y refinamientos técnicos sobre episodios ficticios, sin ajuste externo."""

import copy
import importlib
import math
from dataclasses import replace

import pytest
import torch
from test_financial_adapter import setup

CODEC = "a" * 64
CONTEXT = "b" * 64


def api():
    try:
        return importlib.import_module("mars_titan.models.titans.episodic_readout")
    except ModuleNotFoundError:
        pytest.fail("Falta el lector episódico puro")


def source(size=3):
    keys = torch.eye(64, dtype=torch.float32)[:size].clone()
    values = torch.zeros((size, 64), dtype=torch.float32)
    values[:, 0] = torch.arange(1, size + 1, dtype=torch.float32)
    return dict(
        keys=keys,
        values=values,
        labels=torch.arange(size, dtype=torch.float64) / 10,
        ids=torch.arange(1, size + 1, dtype=torch.int64) * 10,
        decision_at=torch.ones(size, dtype=torch.int64),
        available_at=torch.ones(size, dtype=torch.int64),
        maturity_at=torch.full((size,), 2, dtype=torch.int64),
    )


def snapshot(size=3, dtype=torch.float64, **changes):
    data = source(size) | changes
    return api().EpisodeSnapshot.create(
        **data, cutoff=10, codec_id=CODEC, context_id=CONTEXT, device="cpu", dtype=dtype
    )


def reader(**options):
    return api().EpisodicReadout(
        api().EpisodicReadoutConfig(CODEC, hidden_size=32, **options), dtype=torch.float64
    )


def analytic(model):
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.query_projection.weight[0, 0] = 1
        model.query_projection.weight[1, 1] = 1
        model.value_projection.weight[0, 0] = 1
        model.refinement.weight[0, 64] = 1
        model.step_logit.fill_(math.log(0.1 / 0.9))


def test_read_matches_a_two_neighbor_softmax_and_residual_equation():
    model = reader(neighbors=2)
    analytic(model)
    z = torch.zeros((1, 32), dtype=torch.float64)
    z[0, 0] = 1
    result = model(z, snapshot(2), context_id=CONTEXT, cutoff=10)
    first = math.e / (math.e + 1)
    read = first + (1 - first) * 2
    torch.testing.assert_close(
        result.reads[0].weights, torch.tensor([[first, 1 - first]], dtype=torch.float64)
    )
    assert result.reads[0].ids.tolist() == [[10, 20]]
    assert result.reads[0].presence.tolist() == [[True]]
    assert result.state[0, 0].item() == pytest.approx(1 + 0.1 * math.tanh(read), abs=1e-14)
    assert torch.equal(result.state[0, 1:], z[0, 1:])
    assert not result.state.requires_grad


def test_ties_and_zero_queries_keep_the_lowest_ids():
    model = reader(neighbors=2)
    with torch.no_grad():
        model.query_projection.weight.zero_()
        model.query_projection.bias.zero_()
    result = model(
        torch.zeros((2, 32), dtype=torch.float64), snapshot(3), context_id=CONTEXT, cutoff=10
    )
    assert result.reads[0].ids.tolist() == [[10, 20], [10, 20]]
    torch.testing.assert_close(
        result.reads[0].weights, torch.full((2, 2), 0.5, dtype=torch.float64)
    )


def test_empty_memory_has_false_presence_while_observed_zero_is_present():
    model = reader(neighbors=1)
    analytic(model)
    with torch.no_grad():
        model.refinement.weight[0, 64] = 0
        model.refinement.weight[0, -1] = 1
    z = torch.zeros((1, 32), dtype=torch.float64)
    empty = model(z, snapshot(0), context_id=CONTEXT, cutoff=10)
    zero = model(
        z,
        snapshot(1, values=torch.zeros((1, 64), dtype=torch.float32)),
        context_id=CONTEXT,
        cutoff=10,
    )
    assert empty.reads[0].presence.tolist() == [[False]]
    assert empty.reads[0].weights.shape == (1, 0)
    assert torch.equal(empty.state, z)
    assert zero.reads[0].presence.tolist() == [[True]]
    assert zero.state[0, 0].item() == pytest.approx(0.1 * math.tanh(1))


@pytest.mark.parametrize("steps", [1, 2, 4])
def test_k_reselects_globally_after_each_refinement(steps):
    model = reader(neighbors=1, refinements=steps)
    analytic(model)
    with torch.no_grad():
        model.refinement.weight.zero_()
        model.refinement.bias[1] = 1
        model.step_logit.fill_(math.log(0.9 / 0.1))
    z = torch.zeros((1, 32), dtype=torch.float64)
    z[0, 0] = 0.5
    result = model(z, snapshot(3), context_id=CONTEXT, cutoff=10)
    assert len(result.reads) == steps
    assert result.reads[0].ids.item() == 10
    assert all(read.ids.item() == 20 for read in result.reads[1:])
    assert z[0, 1].item() == 0


def test_none_preserves_the_existing_prediction_and_never_calls_the_head():
    core, batch = setup()
    prepared = core.prepare(batch, core.initial_state(batch.flow_ids))

    def forbidden(value):
        pytest.fail("La ruta None volvió a ejecutar la cabeza")

    result = api().apply_episodic_readout(prepared, forbidden)
    assert result.point_predictions is prepared.point_predictions
    assert result.readout is None


def test_gradcheck_and_parameter_gradients_do_not_touch_episode_storage():
    model = reader(neighbors=2, refinements=2)
    analytic(model)
    data = snapshot(3)
    z = torch.zeros((1, 32), dtype=torch.float64)
    z[0, :2] = torch.tensor([0.8, 0.3], dtype=torch.float64)
    z.requires_grad_()
    assert torch.autograd.gradcheck(
        lambda x: model(x, data, context_id=CONTEXT, cutoff=10, differentiable=True).state,
        (z,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )
    result = model(z, data, context_id=CONTEXT, cutoff=10, differentiable=True)
    gradients = torch.autograd.grad(
        result.state.sum(),
        (
            z,
            model.query_projection.weight,
            model.value_projection.weight,
            model.refinement.weight,
            model.step_logit,
        ),
    )
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    assert all(p.grad is None for p in model.parameters())
    assert all(
        t.grad_fn is None and not t.requires_grad for t in data.export_cpu()["tensors"].values()
    )


def test_constructor_uses_cpu_rng_and_ignores_ambient_meta_device():
    before = torch.random.get_rng_state().clone()
    with torch.device("meta"):
        model = reader()
    assert torch.equal(before, torch.random.get_rng_state())
    assert all(p.device.type == "cpu" for p in model.parameters())


@pytest.mark.parametrize("fault", ["nan", "future", "ids", "dtype", "keys", "graph"])
def test_snapshot_rejects_invalid_records(fault):
    data = source()
    if fault == "nan":
        data["values"][0, 0] = float("nan")
    elif fault == "future":
        data["maturity_at"][0] = 11
    elif fault == "ids":
        data["ids"][1] = 10
    elif fault == "dtype":
        data["keys"] = data["keys"].double()
    elif fault == "keys":
        data["keys"][0].zero_()
    else:
        data["values"].requires_grad_()
    with pytest.raises(ValueError):
        snapshot(**data)


def test_snapshot_copies_input_and_roundtrip_preserves_reads():
    model = reader()
    original = source()
    data = snapshot(**original)
    z = torch.arange(64, dtype=torch.float64).reshape(2, 32) / 64
    before = model(z, data, context_id=CONTEXT, cutoff=10).state
    original["values"].zero_()
    payload = data.export_cpu()
    restored = api().EpisodeSnapshot.restore(
        payload, codec_id=CODEC, context_id=CONTEXT, cutoff=10, device="cpu", dtype=torch.float64
    )
    payload["tensors"]["values"].zero_()
    after = model(z, restored, context_id=CONTEXT, cutoff=10).state
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_same_shape_does_not_allow_loading_another_readout_contract():
    first, other = reader(), reader(mode="no_bank")
    state = {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in other.state_dict().items()
    }
    with pytest.raises(ValueError):
        other.load_state_dict(first.state_dict(), strict=False)
    for name, value in state.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, other.state_dict()[name])


def test_zero_query_has_finite_gradient_at_the_normalization_floor():
    model = reader(neighbors=2)
    analytic(model)
    z = torch.zeros((1, 32), dtype=torch.float64, requires_grad=True)
    result = model(z, snapshot(2), context_id=CONTEXT, cutoff=10, differentiable=True)
    gradient = torch.autograd.grad(result.state.sum(), z)[0]
    assert torch.isfinite(gradient).all()


def test_public_read_cannot_bypass_the_working_budget():
    model = reader(max_working_bytes=1024)
    with pytest.raises(ValueError, match="presupuesto"):
        model.read(
            torch.zeros((1, 32), dtype=torch.float64), snapshot(), context_id=CONTEXT, cutoff=10
        )


def test_nonfinite_gate_is_rejected_even_if_sigmoid_saturates():
    model = reader()
    with torch.no_grad():
        model.step_logit.fill_(float("inf"))
    with pytest.raises(ValueError, match="NaN|infinito"):
        model(torch.zeros((1, 32), dtype=torch.float64), snapshot(), context_id=CONTEXT, cutoff=10)


def test_differentiable_mode_rejects_inference_mode_instead_of_losing_the_graph():
    model, data = reader(), snapshot()
    with torch.inference_mode(), pytest.raises(ValueError):
        model(
            torch.ones((1, 32), dtype=torch.float64),
            data,
            context_id=CONTEXT,
            cutoff=10,
            differentiable=True,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("steps", [1, 2, 4])
def test_permutation_partition_and_repeated_calls_do_not_contaminate_flows(dtype, steps):
    model = api().EpisodicReadout(
        api().EpisodicReadoutConfig(CODEC, hidden_size=32, refinements=steps), dtype=dtype
    )
    data = snapshot(3, dtype=dtype)
    z = torch.arange(96, dtype=dtype).reshape(3, 32) / 64
    original = z.clone()
    result = model(z, data, context_id=CONTEXT, cutoff=10)
    model(z * 50, data, context_id=CONTEXT, cutoff=10)
    repeated = model(z, data, context_id=CONTEXT, cutoff=10)
    assert torch.equal(result.state, repeated.state)
    permuted = model(z[[2, 0, 1]], data, context_id=CONTEXT, cutoff=10)
    partitioned = torch.cat(
        [model(z[i : i + 1], data, context_id=CONTEXT, cutoff=10).state for i in range(3)]
    )
    torch.testing.assert_close(permuted.state[[1, 2, 0]], result.state)
    torch.testing.assert_close(partitioned, result.state)
    assert torch.equal(original, z)
    data.verify()


def test_paired_no_bank_equals_empty_memory_but_differs_from_omitting_refinement():
    bank, disabled = reader(refinements=2), reader(refinements=2, mode="no_bank", seed=19)
    receipt = api().copy_readout_parameters(bank, disabled)
    assert receipt["snapshot_transferred"] is False
    z = torch.ones((2, 32), dtype=torch.float64)
    empty = bank(z, snapshot(0), context_id=CONTEXT, cutoff=10)
    without = disabled(z)
    assert torch.equal(empty.state, without.state)
    assert not torch.equal(without.state, z)
    with pytest.raises(ValueError):
        disabled(z, snapshot())


def test_composition_reuses_working_state_and_head_with_external_gradients(monkeypatch):
    core, batch = setup()
    prepared = core.prepare(batch, core.initial_state(batch.flow_ids), differentiable=True)
    model = reader(refinements=4)
    data = snapshot()
    monkeypatch.setattr(core, "prepare", lambda *a, **k: pytest.fail("MAC repetido"))
    result = api().apply_episodic_readout(
        prepared, core.head, model, data, context_id=CONTEXT, cutoff=10, differentiable=True
    )
    assert result.readout.state.shape == prepared.working_state.shape
    assert torch.equal(result.point_predictions, core.head(result.readout.state).squeeze(-1))
    gradients = torch.autograd.grad(
        result.point_predictions.sum(),
        (core.head.weight, core.fusion[0].weight, model.query_projection.weight),
    )
    assert all(g.abs().sum() > 0 and torch.isfinite(g).all() for g in gradients)
    assert prepared.next_state.observed_steps.tolist() == [1, 1]
    assert prepared.next_state.mac.memory.steps.tolist() == [1, 1]
    assert all(p.grad is None for p in core.parameters())


@pytest.mark.parametrize("fault", ["codec", "context", "cutoff", "dtype", "shape", "nan", "device"])
def test_reader_rejects_context_and_input_mismatches(fault):
    model, data = reader(), snapshot()
    z = torch.ones((1, 32), dtype=torch.float64)
    context, cutoff = CONTEXT, 10
    if fault == "codec":
        data = api().EpisodeSnapshot.create(
            **source(), cutoff=10, codec_id="c" * 64, context_id=CONTEXT, dtype=torch.float64
        )
    elif fault == "context":
        context = "c" * 64
    elif fault == "cutoff":
        cutoff = 11
    elif fault == "dtype":
        z = z.float()
    elif fault == "shape":
        z = z[:, :31]
    elif fault == "nan":
        z[0, 0] = float("nan")
    else:
        z = torch.ones((1, 32), dtype=torch.float64, device="meta")
    with pytest.raises(ValueError):
        model(z, data, context_id=context, cutoff=cutoff)


@pytest.mark.parametrize(
    "fault", ["sha", "bytes", "version", "dtype", "alias", "future", "context"]
)
def test_snapshot_restore_rejects_incompatible_or_modified_payload(fault):
    payload = snapshot().export_cpu()
    if fault == "sha":
        payload["sha256"] = "0" * 64
    elif fault == "bytes":
        payload["tensors"]["labels"][0] += 1
    elif fault == "version":
        payload["schema_version"] = True
    elif fault == "dtype":
        payload["tensors"]["keys"] = payload["tensors"]["keys"].float()
    elif fault == "alias":
        payload["tensors"]["ids"] = torch.ones(500_000, dtype=torch.int64)[:3]
    elif fault == "future":
        payload["tensors"]["maturity_at"][0] = 11
    else:
        payload["identity"]["context_id"] = "0" * 64
    with pytest.raises(ValueError):
        api().EpisodeSnapshot.restore(
            payload,
            codec_id=CODEC,
            context_id=CONTEXT,
            cutoff=10,
            device="cpu",
            dtype=torch.float64,
        )


def test_snapshot_version_guard_and_strong_digest_have_distinct_boundaries():
    model, data = reader(), snapshot()
    data._values[1][0, 0] += 1
    with pytest.raises(ValueError):
        model(torch.ones((1, 32), dtype=torch.float64), data, context_id=CONTEXT, cutoff=10)
    data = snapshot()
    data._values[1].data[0, 0] += 1
    with pytest.raises(ValueError):
        data.export_cpu()


def test_recovery_with_the_same_contract_preserves_all_refinement_outputs():
    model, replica = reader(refinements=4), reader(refinements=4)
    replica.load_state_dict(copy.deepcopy(model.state_dict()))
    z = torch.ones((1, 32), dtype=torch.float64)
    data = snapshot()
    first = model(z, data, context_id=CONTEXT, cutoff=10)
    second = replica(z, data, context_id=CONTEXT, cutoff=10)
    assert torch.equal(first.state, second.state)
    assert all(
        torch.equal(a.weights, b.weights) and torch.equal(a.ids, b.ids)
        for a, b in zip(first.reads, second.reads, strict=True)
    )


@pytest.mark.parametrize(
    "options",
    [
        dict(refinements=3),
        dict(refinements=True),
        dict(neighbors=9),
        dict(temperature=float("nan")),
        dict(temperature=0),
        dict(mode="M3"),
        dict(max_batch=257),
        dict(seed=True),
    ],
)
def test_invalid_configurations_fail_before_constructing_layers(options):
    with pytest.raises(ValueError):
        reader(**options)


def test_refinement_rejects_nonzero_absent_read_values():
    model = reader(mode="no_bank")
    z = torch.zeros((1, 32), dtype=torch.float64)
    read = model.read(z)
    invalid = replace(read, values=torch.ones_like(z))
    with pytest.raises(ValueError):
        model.refine(z, z, invalid)


@pytest.mark.parametrize(
    "fault", ["snapshot_source_budget", "snapshot_result_budget", "working_budget"]
)
def test_budgets_reject_before_copy_or_projection(monkeypatch, fault):
    if fault == "working_budget":
        model, data = reader(max_working_bytes=1024), snapshot()
        monkeypatch.setattr(
            model.query_projection,
            "forward",
            lambda x: pytest.fail("Proyección antes del presupuesto"),
        )
        with pytest.raises(ValueError, match="presupuesto"):
            model(torch.zeros((1, 32), dtype=torch.float64), data, context_id=CONTEXT, cutoff=10)
    else:
        data = source()
        maximum = 3000
        if fault == "snapshot_source_budget":
            data["labels"] = torch.zeros(10000, dtype=torch.float64)[:3]
        monkeypatch.setattr(
            api().EpisodeSnapshot,
            "_build",
            lambda *a, **k: pytest.fail("Copia antes del presupuesto"),
        )
        with pytest.raises(ValueError, match="presupuesto"):
            api().EpisodeSnapshot.create(
                **data,
                cutoff=10,
                codec_id=CODEC,
                context_id=CONTEXT,
                dtype=torch.float64,
                max_bytes=maximum,
            )


def test_tiny_and_huge_queries_preserve_normalization_without_nonfinite_outputs():
    model = reader(neighbors=2)
    analytic(model)
    for scale in (1e-16, 1.0, 1e300):
        z = torch.zeros((1, 32), dtype=torch.float64)
        z[0, 0] = scale
        result = model.read(z, snapshot(2), context_id=CONTEXT, cutoff=10)
        score = min(1.0, scale / 1e-12)
        weight = math.exp(score) / (math.exp(score) + 1)
        assert result.weights[0, 0].item() == pytest.approx(weight, abs=1e-14)
        assert torch.isfinite(result.values).all()


def test_one_neighbor_has_no_query_selection_gradient():
    model = reader(neighbors=1)
    analytic(model)
    z = torch.ones((1, 32), dtype=torch.float64, requires_grad=True)
    result = model.read(z, snapshot(2), context_id=CONTEXT, cutoff=10)
    gradient = torch.autograd.grad(result.values.sum(), model.query_projection.weight)[0]
    assert torch.equal(gradient, torch.zeros_like(gradient))
    assert result.weights.item() == 1


@pytest.mark.parametrize("fault", ["gate", "parameters", "keys", "missing_contract"])
def test_invalid_state_dict_is_rejected_before_copy(fault):
    model = reader()
    valid = copy.deepcopy(model.state_dict())
    changed = copy.deepcopy(valid)
    if fault == "gate":
        changed["step_logit"].fill_(float("nan"))
    elif fault == "parameters":
        changed["query_projection.weight"] = changed["query_projection.weight"].float()
    elif fault == "keys":
        changed["extra"] = torch.ones(1)
    else:
        changed.pop("_extra_state")
    with pytest.raises(ValueError):
        model.load_state_dict(changed, strict=False)
    assert all(
        torch.equal(model.state_dict()[key], value)
        for key, value in valid.items()
        if isinstance(value, torch.Tensor)
    )


@pytest.mark.parametrize(
    "key,value", [("cutoff", True), ("codec_id", "x"), ("dtype", torch.float16), ("device", "meta")]
)
def test_snapshot_options_are_checked_before_transfer(key, value):
    options = dict(cutoff=10, codec_id=CODEC, context_id=CONTEXT, dtype=torch.float64, device="cpu")
    options[key] = value
    with pytest.raises(ValueError):
        api().EpisodeSnapshot.create(**source(), **options)


@pytest.mark.parametrize("field", ["available_at", "decision_at", "maturity_at"])
def test_snapshot_rejects_incoherent_temporal_intervals(field):
    data = source()
    data[field][0] = -1 if field != "available_at" else 3
    if field == "decision_at":
        data[field][0] = 2
    with pytest.raises(ValueError):
        snapshot(**data)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_absent_values_cannot_be_hidden_by_the_presence_mask(invalid):
    model = reader(mode="no_bank")
    z = torch.zeros((1, 32), dtype=torch.float64)
    read = model.read(z)
    corrupt = replace(read, values=torch.full_like(z, invalid))
    with pytest.raises(ValueError):
        model.refine(z, z, corrupt)


@pytest.mark.parametrize("field", ["ids", "decision_at", "available_at", "maturity_at"])
def test_episode_integer_fields_never_accept_float_aliases(field):
    data = source()
    data[field] = data[field].double()
    with pytest.raises(ValueError):
        snapshot(**data)


def test_detached_mode_does_not_keep_the_incoming_autograd_history():
    model, data = reader(), snapshot()
    parameter = torch.ones((2, 32), dtype=torch.float64, requires_grad=True)
    z = parameter * 2
    result = model(z, data, context_id=CONTEXT, cutoff=10, differentiable=False)
    assert result.state.grad_fn is None and not result.state.requires_grad
    assert all(
        read.values.grad_fn is None and read.weights.grad_fn is None for read in result.reads
    )
    assert parameter.grad is None
    assert not tuple(model.buffers())
