"""Jerarquía anterior a MAC sobre el predictor, sus contratos y los pesos congelados."""

import copy
import importlib
import importlib.util
import json
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_financial_adapter import specification
from test_frozen_financial import frozen_backend as frozen_backend
from test_market_memory import MARKET_MEMORY, pt4
from test_memory_stability import decision
from test_regional_geometry_contract import assert_same

from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import DecisionBatch
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismCatalog
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.models.titans.regional_execution import RegionalExecution, RegionalExecutionConfig
from mars_titan.models.titans.regional_fusion import RegionalFusionConfig

DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))
DTYPES = (torch.float32, torch.float64)
FLOWS = ("US/AAA", "US/BBB")
ROOT = Path(__file__).resolve().parents[3]


def hierarchy_api():
    name = "mars_titan.models.titans.regional_hierarchy"
    assert importlib.util.find_spec(name) is not None, (
        "Falta el componente regional de ocho estados"
    )
    return importlib.import_module(name)


def execution_api():
    module = importlib.import_module("mars_titan.models.titans.regional_execution")
    for name in (
        "parse_regional_execution_config",
        "regional_execution_from_identity",
        "build_regional_execution",
        "regional_work_per_row",
    ):
        assert callable(getattr(module, name, None)), f"Falta la integración {name}"
    return module


def local_operator(**changes):
    return RegionalFusionConfig(
        **dict(topology="self", routing="uniform", cycles=2, local_steps=2, summary=True) | changes
    )


def hierarchy_predictor(device="cpu", dtype=torch.float64, *, context=True, **options):
    config = hierarchy_api().RegionalHierarchyConfig(**options)
    model = pt4(
        MARKET_MEMORY if context else None,
        regional_fusion=local_operator(),
        regional_execution=config,
        max_batch=8,
    ).to(device=device, dtype=dtype)
    if context:
        checkpoint = model.state_dict()
        checkpoint["session_context.output"] = torch.eye(32, device=device, dtype=dtype) * 0.2
        model.load_state_dict(checkpoint)
    return model.eval().requires_grad_(False)


def batch_for(model, event):
    source = decision(model.config, FLOWS, event)
    raw = dict(
        inputs={name: value.numpy().astype(np.float32) for name, value in source.inputs.items()},
        presence=source.presence.numpy(),
        sample_ids=list(source.sample_ids),
        prediction_at=np.array(source.prediction_at, dtype="datetime64[us]"),
        input_available_at=np.array(source.input_available_at, dtype="datetime64[us]"),
    )
    return DecisionBatch.from_corpus(
        raw, model.config.inputs, device=model.head.weight.device, dtype=model.head.weight.dtype
    )


def geometry_name(module):
    names = [
        name
        for name, value in module.named_buffers()
        if value.dtype == torch.bool and value.shape == (8, 8)
    ]
    assert len(names) == 1
    return names[0]


def test_financial_constructor_accepts_complete_hierarchy_json_in_the_existing_field():
    raw = dict(
        topology="tree",
        groups=[[0, 2], [1, 3, 4]],
        feedback=True,
        readout="local_only",
    )
    config = FinancialConfig(
        specification(),
        variant="mac_online",
        hidden_size=32,
        regional_fusion=local_operator(),
        regional_execution=raw,
    )
    assert type(config.regional_execution) is hierarchy_api().RegionalHierarchyConfig
    assert config.regional_execution.groups == ((0, 2), (1, 3, 4))
    assert not hasattr(config, "regional_hierarchy")
    assert config.identity()["regional_execution"]["kind"] == "pre_mac_regional_hierarchy_v1"


def test_complete_json_selects_the_new_type_without_reinterpreting_old_configuration():
    api = execution_api()
    new_type = hierarchy_api().RegionalHierarchyConfig
    new = new_type()
    raw = json.loads(json.dumps(asdict(new)))
    parsed = api.parse_regional_execution_config(raw)
    assert type(parsed) is new_type and parsed.groups == ((0, 2), (1, 3, 4))
    base = pt4().config
    financial = replace(base, regional_fusion=local_operator(), regional_execution=raw)
    assert type(financial.regional_execution) is new_type
    assert financial.regional_execution.identity() == new.identity()
    catalog = MechanismCatalog(regional_execution=financial.regional_execution)
    recovered = MechanismCatalog.restore(json.loads(json.dumps(catalog.identity())))
    assert type(recovered.regional_execution) is new_type
    assert recovered.identity() == catalog.identity() and len(recovered.actions()) == 36
    old = RegionalExecutionConfig()
    assert type(api.parse_regional_execution_config(asdict(old))) is RegionalExecutionConfig
    assert [field.name for field in fields(old)] == ["selection", "program", "context_threshold"]
    assert "regional_execution" not in MechanismCatalog().identity()


@pytest.mark.parametrize(
    "change",
    [
        dict(topology="ring"),
        dict(routing="attention"),
        dict(cycles=1),
        dict(local_steps=1),
        dict(summary=False),
    ],
)
def test_financial_configuration_rejects_incompatible_local_operator_before_build(change):
    config = hierarchy_api().RegionalHierarchyConfig()
    base = replace(pt4().config, regional_fusion=local_operator(), regional_execution=config)
    assert type(base.regional_execution) is type(config)
    with pytest.raises(ValueError):
        replace(base, regional_fusion=local_operator(**change))
    with pytest.raises(ValueError):
        replace(base, regional_fusion=None)


def test_the_existing_factory_and_manual_constructor_execute_the_same_hierarchy():
    from mars_titan.training.search_cases import checked_predictor_changes
    from mars_titan.training.titans_walk_forward import _predictor

    new_type = hierarchy_api().RegionalHierarchyConfig
    document = json.loads((ROOT / "configs/experiments/regional-fusion-recipe-v1.json").read_text())
    document["predictor"]["hidden_size"] = 32
    changes = dict(regional_fusion=asdict(local_operator()), regional_execution=asdict(new_type()))
    assert checked_predictor_changes(changes) == changes
    factory, pairing = _predictor(
        document, specification(), "mac_online", 42, "cpu", changes=changes
    )
    options = dict(document["predictor"])
    assert options.pop("dtype") == "float32"
    manual = FinancialPredictor(
        FinancialConfig(specification(), variant="mac_online", seed=42, **options, **changes),
        dtype=torch.float32,
    )
    assert pairing is None
    assert type(factory.regional_execution) is hierarchy_api().RegionalHierarchy
    factory.eval().requires_grad_(False)
    manual.eval().requires_grad_(False)
    assert_same(factory.state_dict(), manual.state_dict())
    rows = batch_for(manual, 0)
    left = factory.prepare(rows, factory.initial_state(FLOWS))
    right = manual.prepare(rows, manual.initial_state(FLOWS))
    assert torch.equal(left.point_predictions, right.point_predictions)
    assert_same(left.regional_trace, right.regional_trace)
    assert left.regional_trace["work"] == dict(local_node_evaluations=40, summary_evaluations=20)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_hierarchy_reuses_shared_parameters_and_advances_mac_and_context_once(device, dtype):
    model = hierarchy_predictor(device, dtype)
    old = pt4(
        MARKET_MEMORY,
        regional_fusion=local_operator(),
        regional_execution=RegionalExecutionConfig(),
        max_batch=8,
    ).to(device=device, dtype=dtype)
    old_parameters = dict(old.named_parameters())
    assert dict(model.named_parameters()).keys() == old_parameters.keys()
    for name, parameter in model.named_parameters():
        if name != "session_context.output":
            assert torch.equal(parameter, old_parameters[name]), name
    assert not list(model.regional_execution.parameters())
    state, context = model.initial_state(FLOWS), model.initial_context()
    for event in range(2):
        before = copy.deepcopy(model.export_context(context))
        result = model.prepare(batch_for(model, event), state, context=context)
        assert result.next_state.mac.memory.steps.tolist() == [event + 1, event + 1]
        assert result.next_state.observed_steps.tolist() == [event + 1, event + 1]
        assert_same(model.export_context(context), before)
        assert result.next_context.pending_flows == (FLOWS, ())
        assert torch.equal(result.next_context.pending_tokens[0], result.detached_tokens)
        assert result.regional_trace["availability"].shape == (2, 8)
        assert result.regional_trace["work"] == dict(
            local_node_evaluations=40, summary_evaluations=20
        )
        context = model.commit_context(result.next_context)
        state = result.next_state
    assert result.regional_trace["context_norms"].abs().sum() > 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_checkpoint_and_exported_state_reproduce_the_next_emission_and_work(device, dtype):
    model = hierarchy_predictor(device, dtype)
    recovered = hierarchy_predictor(device, dtype)
    recovered.load_state_dict(copy.deepcopy(model.state_dict()))
    first = model.prepare(
        batch_for(model, 0), model.initial_state(FLOWS), context=model.initial_context()
    )
    context = model.commit_context(first.next_context)
    next_context = recovered.restore_context(model.export_context(context))
    next_state = recovered.restore_state(model.export_state_cpu(first.next_state), device=device)
    left = model.prepare(batch_for(model, 1), first.next_state, context=context)
    right = recovered.prepare(batch_for(recovered, 1), next_state, context=next_context)
    assert torch.equal(left.point_predictions, right.point_predictions)
    assert_same(left.regional_trace, right.regional_trace)
    assert_same(
        model.export_state_cpu(left.next_state), recovered.export_state_cpu(right.next_state)
    )
    assert left.next_context.pending_flows == right.next_context.pending_flows == (FLOWS, ())
    assert left.next_context.pending_at == right.next_context.pending_at
    assert_same(left.next_context.pending_tokens, right.next_context.pending_tokens)
    committed_left = model.commit_context(left.next_context)
    committed_right = recovered.commit_context(right.next_context)
    assert_same(model.export_context(committed_left), recovered.export_context(committed_right))
    assert not any("working_state" in key for key in model.state_dict())


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("altered_installed", [False, True])
def test_invalid_hierarchy_geometry_is_rejected_before_any_financial_weight_copy(
    wrapped, altered_installed
):
    model = hierarchy_predictor(context=False)
    container = torch.nn.ModuleDict({"predictor": model}) if wrapped else model
    prefix = "predictor." if wrapped else ""
    key = "regional_execution." + geometry_name(model.regional_execution)
    payload = copy.deepcopy(container.state_dict())
    payload[prefix + key][0, 1].logical_not_()
    payload[prefix + "head.weight"].add_(0.125)
    if altered_installed:
        with torch.no_grad():
            dict(model.regional_execution.named_buffers())[geometry_name(model.regional_execution)][
                0, 1
            ].logical_not_()
    before = copy.deepcopy(container.state_dict())
    parameter_id = model._parameter_id
    with pytest.raises(ValueError):
        container.load_state_dict(payload)
    assert_same(container.state_dict(), before)
    assert model._parameter_id == parameter_id


def test_valid_checkpoint_repairs_boolean_values_of_installed_hierarchy_geometry():
    model = hierarchy_predictor(context=False)
    payload = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        dict(model.regional_execution.named_buffers())[geometry_name(model.regional_execution)][
            0, 1
        ].logical_not_()
    model.load_state_dict(payload)
    assert_same(model.state_dict(), payload)
    model.verify_parameter_identity()


@pytest.mark.parametrize("fault", ["coherent", "boolean_as_integer", "physical_class", "hidden"])
def test_a_hierarchy_cannot_reuse_a_sealed_identity_after_undeclared_changes(fault):
    model = hierarchy_predictor(context=False)
    if fault == "coherent":
        config = replace(model.config.regional_execution, topology="star")
        model.config = replace(model.config, regional_execution=config)
        model.regional_execution.config = config
    elif fault == "boolean_as_integer":
        object.__setattr__(model.config.regional_execution, "feedback", 1)
    elif fault == "physical_class":
        changed = RegionalExecution(32, RegionalExecutionConfig(), max_batch=8)
        changed.config = model.config.regional_execution
        model.regional_execution = changed
    else:
        model.config = replace(model.config, regional_execution=None)
    with pytest.raises(ValueError):
        model.verify_parameter_identity()


def test_frozen_mechanism_consumer_seals_the_active_hierarchy_and_its_source(frozen_backend):
    model = hierarchy_predictor()
    catalog = MechanismCatalog(regional_execution=model.config.regional_execution)
    with torch.no_grad():
        consumer = FrozenMechanismConsumer(
            model, catalog=catalog, action=MechanismAction(context="parent")
        )
        rows = batch_for(model, 0)
        event = consumer.begin(
            consumer.initial_state(),
            flow_ids=FLOWS,
            cutoff=rows.prediction_at[0],
            context_id="a" * 64,
            matures_at=(rows.prediction_at[0] + 86_400_000_000,) * len(FLOWS),
        )
        prepared = consumer.prepare(rows, model.initial_state(FLOWS), event)
        result = consumer.finish(prepared.event)
        consumer.verify(strong=False)
    assert result.work["regional_local_node_evaluations"] == 40
    assert result.work["regional_summary_evaluations"] == 20
    assert all(len(row["regional_execution"]["availability"]) == 8 for row in result.rows)
    name = "mars_titan.models.titans.regional_hierarchy"
    assert name in consumer.identity()["implementation"]
    settings = consumer.identity()["modules"]["predictor/regional_execution"]["settings"]
    assert settings["regional_execution"] == json.loads(
        json.dumps(model.config.regional_execution.identity())
    )
    assert settings["hidden_size"] == 32 and settings["max_batch"] == 8
    legacy = FrozenMechanismConsumer(pt4().eval().requires_grad_(False))
    assert name not in legacy.identity()["implementation"]
    model.regional_execution.max_batch += 1
    with pytest.raises(ValueError), torch.no_grad():
        consumer.verify(strong=False)


def test_a_simple_frozen_consumer_still_rejects_the_active_hierarchy():
    model = hierarchy_predictor(context=False)
    with pytest.raises(ValueError, match="programa regional"):
        FrozenFinancialConsumer(model)


@pytest.mark.parametrize("threshold", [1, 1.0], ids=["integer", "float"])
def test_union_identity_restore_keeps_the_old_threshold_representation(threshold):
    api = execution_api()
    old = RegionalExecutionConfig(selection="context_norm_v1", context_threshold=threshold)
    recovered = api.regional_execution_from_identity(json.loads(json.dumps(old.identity())))
    assert type(recovered) is RegionalExecutionConfig
    assert type(recovered.context_threshold) is type(threshold)
    assert recovered.identity() == old.identity()
    changed = copy.deepcopy(old.identity())
    changed["context_threshold"] = 1.0 if type(threshold) is int else 1
    assert json.dumps(changed, sort_keys=True) != json.dumps(old.identity(), sort_keys=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["none", "fixed"])
def test_union_preserves_legacy_checkpoint_rng_and_two_emissions(device, kind):
    hierarchy_api()
    execution = RegionalExecutionConfig() if kind == "fixed" else None
    config = replace(
        pt4(MARKET_MEMORY).config,
        regional_fusion=local_operator(),
        regional_execution=execution,
    )
    raw = asdict(execution) if execution is not None else None
    parsed = replace(config, regional_execution=raw)
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state(0).clone() if device == "cuda:0" else None
    left = (
        FinancialPredictor(config, device=device, dtype=torch.float64).eval().requires_grad_(False)
    )
    right = (
        FinancialPredictor(parsed, device=device, dtype=torch.float64).eval().requires_grad_(False)
    )
    right.load_state_dict(copy.deepcopy(left.state_dict()))
    assert left.config.identity() == right.config.identity()
    assert_same(left.state_dict(), right.state_dict())
    if kind == "none":
        assert "regional_execution" not in right.config.identity()
        assert not any(key.startswith("regional_execution.") for key in right.state_dict())
    state_left, state_right = left.initial_state(FLOWS), right.initial_state(FLOWS)
    context_left, context_right = left.initial_context(), right.initial_context()
    for event in range(2):
        a = left.prepare(batch_for(left, event), state_left, context=context_left)
        b = right.prepare(batch_for(right, event), state_right, context=context_right)
        assert type(a) is type(b) and torch.equal(a.point_predictions, b.point_predictions)
        state_left, state_right = a.next_state, b.next_state
        context_left, context_right = (
            left.commit_context(a.next_context),
            right.commit_context(b.next_context),
        )
        assert_same(left.export_context(context_left), right.export_context(context_right))
    assert torch.equal(before_cpu, torch.get_rng_state())
    if before_cuda is not None:
        assert torch.equal(before_cuda, torch.cuda.get_rng_state(0))
