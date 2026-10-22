"""Programas anteriores a MAC con consulta confirmada y una preparación pendiente."""

import copy
import importlib
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_context_selection import batch_on
from test_frozen_financial import frozen_backend as frozen_backend
from test_market_memory import MARKET_MEMORY, pt4
from test_regional_geometry_contract import assert_same

from mars_titan.models.titans.financial import FinancialPredictor
from mars_titan.models.titans.financial_inputs import DecisionBatch
from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismCatalog
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.models.titans.regional_fusion import RegionalFusionConfig

DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))
FLOWS = ("US/AAA", "US/BBB")


def execution(**options):
    api = importlib.import_module("mars_titan.models.titans.regional_execution")
    return api.RegionalExecutionConfig(**options)


def predictor(device, *, context=True, **options):
    model = pt4(
        MARKET_MEMORY if context else None,
        regional_fusion=RegionalFusionConfig(topology="ring"),
        regional_execution=execution(**options),
    ).to(device)
    if context:
        checkpoint = model.state_dict()
        checkpoint["session_context.output"] = torch.eye(32, device=device).double() * 0.2
        model.load_state_dict(checkpoint)
    return model.eval().requires_grad_(False)


@pytest.mark.parametrize("device", DEVICES)
def test_none_keeps_the_legacy_configuration_checkpoint_and_two_emissions(device):
    old = pt4(MARKET_MEMORY).to(device).eval().requires_grad_(False)
    new = FinancialPredictor(replace(old.config, regional_execution=None), dtype=torch.float64)
    new = new.to(device).eval().requires_grad_(False)
    new.load_state_dict(copy.deepcopy(old.state_dict()))
    assert new.regional_execution is None
    assert old.config.identity() == new.config.identity()
    assert "regional_execution" not in new.config.identity()
    assert_same(new.state_dict(), old.state_dict())
    left_state, right_state = old.initial_state(FLOWS), new.initial_state(FLOWS)
    left_context, right_context = old.initial_context(), new.initial_context()
    for event in range(2):
        rows = batch_on(old, FLOWS, event)
        left = old.prepare(rows, left_state, context=left_context)
        right = new.prepare(rows, right_state, context=right_context)
        assert type(left) is type(right)
        assert not hasattr(right, "regional_trace")
        assert torch.equal(left.point_predictions, right.point_predictions)
        left_state, right_state = left.next_state, right.next_state
        left_context = old.commit_context(left.next_context)
        right_context = new.commit_context(right.next_context)
        assert_same(old.export_context(left_context), new.export_context(right_context))


def test_simple_consumer_rejects_a_program_without_its_durable_mechanism_contract():
    from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

    model = predictor("cpu", context=False)
    with pytest.raises(ValueError, match="programa regional"):
        FrozenFinancialConsumer(model)


@pytest.mark.parametrize("shared_config", [False, True])
def test_coherent_program_metadata_mutation_does_not_reuse_a_sealed_identity(shared_config):
    model = predictor("cpu", context=False)
    if shared_config:
        object.__setattr__(model.config.regional_execution, "program", "summary_interleaved_v1")
    else:
        changed = execution(program="summary_interleaved_v1")
        model.config = replace(model.config, regional_execution=changed)
        model.regional_execution.config = changed
    with pytest.raises(ValueError):
        model.verify_parameter_identity()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("program", ["summary_at_end_v1", "summary_interleaved_v1"])
def test_program_queries_before_mac_and_stages_only_the_final_fused_tokens(
    device, program, monkeypatch
):
    model = predictor(device, program=program)
    state, context = model.initial_state(FLOWS), model.initial_context()
    prefix = model.prepare(batch_on(model, FLOWS, 0), state, context=context)
    context = model.commit_context(prefix.next_context)
    before = copy.deepcopy(model.export_context(context))

    def combined_read_forbidden(*args, **kwargs):
        pytest.fail("El programa no debe consultar y preparar pendientes con read combinado")

    monkeypatch.setattr(model.session_context, "read", combined_read_forbidden)
    calls = []
    handle = model.mac.register_forward_pre_hook(lambda *_: calls.append("mac"))
    try:
        result = model.prepare(batch_on(model, FLOWS, 1), prefix.next_state, context=context)
    finally:
        handle.remove()
    assert calls == ["mac"]
    assert_same(model.export_context(context), before)
    assert result.next_context.pending_flows == (FLOWS, ())
    assert torch.equal(result.next_context.pending_tokens[0], result.detached_tokens)
    assert tuple(result.regional_trace["programs"]) == (program, program)
    assert result.regional_trace["context_norms"].abs().sum() > 0
    assert result.regional_trace["work"]["local_node_evaluations"] == 40
    assert result.regional_trace["work"]["summary_evaluations"] == (
        2 if program == "summary_at_end_v1" else 4
    )


@pytest.mark.parametrize("device", DEVICES)
def test_context_off_does_not_read_or_stage_and_disabling_summary_removes_cadence_effect(device):
    config = execution()
    first = (
        pt4(
            MARKET_MEMORY,
            regional_fusion=RegionalFusionConfig(topology="ring", summary=False),
            regional_execution=config,
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    second = (
        pt4(
            MARKET_MEMORY,
            regional_fusion=RegionalFusionConfig(topology="ring", summary=False),
            regional_execution=execution(program="summary_interleaved_v1"),
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    rows = batch_on(first, FLOWS, 0)
    results = []
    for model in (first, second):
        context = model.initial_context()
        result = model.prepare(
            rows, model.initial_state(FLOWS), context=context, context_selection=(False, False)
        )
        assert result.next_context is context
        assert not result.regional_trace["context_norms"].any()
        assert result.regional_trace["work"]["summary_evaluations"] == 0
        results.append(result)
    assert torch.equal(results[0].point_predictions, results[1].point_predictions)


@pytest.mark.parametrize("device", DEVICES)
def test_program_checkpoint_restores_the_next_decision_and_rejects_another_program(device):
    model = predictor(device)
    restored = predictor(device)
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    first = model.prepare(
        batch_on(model, FLOWS, 0), model.initial_state(FLOWS), context=model.initial_context()
    )
    context = model.commit_context(first.next_context)
    recovered_context = restored.restore_context(model.export_context(context))
    recovered_state = restored.restore_state(
        model.export_state_cpu(first.next_state), device=device
    )
    rows = batch_on(model, FLOWS, 1)
    left = model.prepare(rows, first.next_state, context=context)
    right = restored.prepare(rows, recovered_state, context=recovered_context)
    assert torch.equal(left.point_predictions, right.point_predictions)
    assert_same(left.regional_trace, right.regional_trace)
    other = predictor(device, program="summary_interleaved_v1")
    before = copy.deepcopy(other.state_dict())
    with pytest.raises(ValueError):
        other.load_state_dict(model.state_dict())
    assert_same(other.state_dict(), before)


def test_context_selector_requires_an_explicit_context_parent_and_regional_operator():
    base = pt4().config
    rule = execution(selection="context_norm_v1", context_threshold=0.25)
    with pytest.raises(ValueError):
        replace(base, regional_execution=execution())
    with pytest.raises(ValueError):
        replace(base, regional_fusion=RegionalFusionConfig(), regional_execution=rule)


@pytest.mark.parametrize("fault", ["configuration", "absent", "hidden"])
def test_a_program_cannot_change_after_the_predictor_is_sealed(fault):
    model = predictor("cpu") if fault != "hidden" else pt4().eval().requires_grad_(False)
    if fault == "configuration":
        model.regional_execution.config = execution(program="summary_interleaved_v1")
    elif fault == "absent":
        model.regional_execution = None
    else:
        api = importlib.import_module("mars_titan.models.titans.regional_execution")
        model.regional_execution = api.RegionalExecution(32, execution())
    with pytest.raises(ValueError):
        model.verify_parameter_identity()


@pytest.mark.parametrize("device", DEVICES)
def test_fp32_program_keeps_end_to_end_gradients_and_detaches_only_the_trace(device):
    model = predictor(device).float().requires_grad_(True)
    source = batch_on(model, FLOWS, 0)
    raw = dict(
        inputs={
            name: value.cpu().numpy().astype(np.float32) for name, value in source.inputs.items()
        },
        presence=source.presence.cpu().numpy(),
        sample_ids=list(source.sample_ids),
        prediction_at=np.array(source.prediction_at, dtype="datetime64[us]"),
        input_available_at=np.array(source.input_available_at, dtype="datetime64[us]"),
    )
    rows = DecisionBatch.from_corpus(raw, model.config.inputs, device=device, dtype=torch.float32)
    result = model.prepare(
        rows,
        model.initial_state(FLOWS, differentiable=True),
        context=model.initial_context(differentiable=True),
        differentiable=True,
    )
    assert result.point_predictions.dtype == torch.float32
    gradients = torch.autograd.grad(
        result.point_predictions.square().sum(),
        (
            model.regional_fusion.qkv.weight,
            model.session_context.query,
            model.session_context.output,
        ),
    )
    assert all(torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients)
    assert not result.regional_trace["context_norms"].requires_grad
    assert result.next_context.pending_tokens[0].requires_grad


def test_program_catalog_roundtrip_does_not_reinterpret_legacy_actions():
    config = execution()
    catalog = MechanismCatalog(regional_execution=config)
    assert len(catalog.actions()) == len(MechanismCatalog().actions()) == 36
    assert catalog.fingerprint() != MechanismCatalog().fingerprint()
    assert MechanismCatalog.restore(catalog.identity()).identity() == catalog.identity()
    assert "regional_execution" not in MechanismCatalog().identity()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("program", ["summary_at_end_v1", "summary_interleaved_v1"])
def test_frozen_consumer_exports_executed_programs_and_counts_actual_phases(
    device, program, frozen_backend
):
    model = predictor(device, program=program)
    catalog = MechanismCatalog(regional_execution=model.config.regional_execution)
    with torch.no_grad():
        consumer = FrozenMechanismConsumer(
            model, catalog=catalog, action=MechanismAction(context="parent")
        )
        previous = consumer.initial_state()
        rows = batch_on(model, FLOWS, 0)
        event = consumer.begin(
            previous,
            flow_ids=FLOWS,
            cutoff=rows.prediction_at[0],
            context_id="a" * 64,
            matures_at=(rows.prediction_at[0] + 86_400_000_000,) * len(FLOWS),
        )
        block = consumer.prepare(rows, model.initial_state(FLOWS), event)
        emitted = consumer.finish(block.event)
    assert emitted.work["regional_local_node_evaluations"] == 40
    assert emitted.work["regional_summary_evaluations"] == (
        2 if program == "summary_at_end_v1" else 4
    )
    assert previous.context.pending_flows == ((), ())
    assert emitted.state.context.pending_flows == ((), ())
    assert emitted.state.context.committed_at[0] == rows.prediction_at[0]
    for row in emitted.rows:
        trace = row["regional_execution"]
        assert trace["program"] == program
        assert trace["work"]["local_node_evaluations"] == 20
        assert tuple(trace["phases"]).count("summary") == (
            1 if program == "summary_at_end_v1" else 2
        )
    assert "mars_titan.models.titans.regional_execution" in consumer.identity()["implementation"]
    model.regional_execution.config = execution(
        program=(
            "summary_interleaved_v1" if program == "summary_at_end_v1" else "summary_at_end_v1"
        )
    )
    with pytest.raises(ValueError), torch.no_grad():
        consumer.verify(strong=False)


def test_a_frozen_program_requires_its_explicit_catalog(frozen_backend):
    model = predictor("cpu")
    with pytest.raises(ValueError), torch.no_grad():
        FrozenMechanismConsumer(model)


def archived_control(*, warmup=False):
    from mars_titan.memory.financial_mechanisms import MechanismBinding
    from mars_titan.models.quantile_head import LEVELS

    model = predictor("cpu")
    consumer = FrozenMechanismConsumer(
        model,
        catalog=MechanismCatalog(regional_execution=model.config.regional_execution),
        action=MechanismAction(context="off"),
    )
    batch = batch_on(model, FLOWS, 0)
    event = consumer.begin(
        consumer.initial_state(),
        flow_ids=FLOWS,
        cutoff=batch.prediction_at[0],
        context_id="a" * 64,
        matures_at=None if warmup else (batch.prediction_at[0] + 86_400_000_000,) * len(FLOWS),
        warmup=warmup,
    )
    prepared = consumer.prepare(batch, model.initial_state(FLOWS), event)
    emitted = consumer.finish(prepared.event)
    rows = [
        dict(
            row,
            prediction_at=batch.prediction_at[0],
            input_available_at=batch.prediction_at[0],
            quantiles=emitted.quantiles[i].tolist() if emitted.quantiles is not None else None,
        )
        for i, row in enumerate(emitted.rows)
    ]
    control = dict(
        rows=rows,
        observations=len(rows),
        consumer_id=consumer.model_id,
        parameter_id=model._parameter_id,
        work=dict(emitted.work),
        levels=list(LEVELS) if emitted.quantiles is not None else [],
        calibration_record=None,
    )
    record = dict(
        cutoff=batch.prediction_at[0],
        observations_count=len(rows),
        event_kind="warmup" if warmup else "decision",
        predictions=[]
        if warmup
        else [
            dict(asset=row["flow_id"], value=row["point"], decision_at=batch.prediction_at[0])
            for row in rows
        ],
    )
    binding = object.__new__(MechanismBinding)
    binding.consumer, binding.collections = consumer, None
    return binding, control, record


@pytest.mark.parametrize("warmup", [False, True])
@pytest.mark.parametrize("fault", ["missing", "program", "off_norm", "row_work", "total_work"])
def test_durable_binding_rejects_inconsistent_regional_program_traces(
    frozen_backend, warmup, fault
):
    with torch.no_grad():
        binding, control, record = archived_control(warmup=warmup)
        binding._check_record(control, record, cutoff=record["cutoff"])
    row = control["rows"][0]
    if fault == "missing":
        del row["regional_execution"]
    elif fault == "program":
        trace = row["regional_execution"]
        trace["program"] = "summary_interleaved_v1"
        trace["phases"] = ["local", "local", "summary", "local", "local", "summary"]
        trace["work"]["summary_evaluations"] = 2
    elif fault == "off_norm":
        row["regional_execution"]["context_norm"] = 0.25
    elif fault == "row_work":
        row["regional_execution"]["work"]["local_node_evaluations"] += 1
    else:
        control["work"]["regional_local_node_evaluations"] += 1
    with pytest.raises(ValueError):
        binding._check_record(control, record, cutoff=record["cutoff"])


def test_chronological_pipeline_accepts_the_explicit_program_changes_without_fitting():
    from mars_titan.training.search_cases import checked_predictor_changes

    changes = dict(
        regional_fusion=dict(
            topology="ring",
            routing="attention",
            cycles=2,
            local_steps=2,
            summary=True,
            local_gain=0.25,
            summary_gain=0.25,
        ),
        regional_execution=dict(
            selection="fixed", program="summary_at_end_v1", context_threshold=None
        ),
    )
    assert checked_predictor_changes(changes) == changes


@pytest.mark.parametrize("threshold", [1, 1.0], ids=["integer", "float"])
@pytest.mark.parametrize("boundary", ["catalog", "module", "parent", "sealed_shared"])
def test_program_threshold_representation_is_checked_before_execution(
    frozen_backend, threshold, boundary
):
    model = predictor("cpu", selection="context_norm_v1", context_threshold=threshold)
    changed = execution(
        selection="context_norm_v1", context_threshold=1.0 if type(threshold) is int else 1
    )
    if boundary == "catalog":
        with pytest.raises(ValueError):
            FrozenMechanismConsumer(model, catalog=MechanismCatalog(regional_execution=changed))
    else:
        if boundary == "module":
            model.regional_execution.config = changed
        elif boundary == "parent":
            model.config = replace(model.config, regional_execution=changed)
        else:
            object.__setattr__(
                model.config.regional_execution, "context_threshold", changed.context_threshold
            )
        with pytest.raises(ValueError):
            model.verify_parameter_identity()
