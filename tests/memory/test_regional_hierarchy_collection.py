"""Trazas jerárquicas reales en colección y sus límites de validación archivada."""

import importlib
import importlib.util
import sys
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
import torch
from test_episodic_readout import CONTEXT
from test_financial_adapter import raw_batch
from test_financial_mechanisms import assert_same
from test_frozen_financial import frozen_backend as frozen_backend
from test_market_memory import KINDS, pt4
from test_mechanism_consumer import consumer
from test_mechanism_observations import AT, DAY, ForbiddenEmbedding, observation_rows, row_sha
from test_regional_program_collection import DEVICES, REGIONAL_WORK, collect, observed_preparations

from mars_titan.memory.mechanism_collection import CollectedMechanisms, MechanismCollector
from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.financial import FinancialPredictor
from mars_titan.models.titans.mechanism_catalog import MechanismCatalog
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.models.titans.regional_fusion import RegionalFusionConfig

CONTROLS = (
    {},
    {"topology": "star"},
    {"groups": ((0, 1), (2, 3, 4))},
    {"feedback": False},
    {"topology": "flat_control"},
    {"readout": "common_skip"},
    {"topology": "flat_control", "readout": "common_skip"},
)


def hierarchy_config(**options):
    name = "mars_titan.models.titans.regional_hierarchy"
    assert importlib.util.find_spec(name) is not None, "Falta la familia regional jerárquica"
    return importlib.import_module(name).RegionalHierarchyConfig(**options)


def hierarchy_engine(device="cpu", *, parent=True, dtype=torch.float64, options=None):
    program = hierarchy_config(**({} if options is None else options))
    kind = KINDS[0] if parent else None
    original = consumer(device, kind=kind, reader=True)
    model = pt4(
        kind,
        head=QUANTILE_HEAD,
        regional_fusion=RegionalFusionConfig(
            topology="self", routing="uniform", cycles=2, local_steps=2, summary=True
        ),
        regional_execution=program,
    ).to(device=device, dtype=dtype)
    if parent:
        weights = model.state_dict()
        weights["session_context.output"] = torch.eye(32, device=device, dtype=dtype) * 0.2
        model.load_state_dict(weights)
    return FrozenMechanismConsumer(
        model.eval().requires_grad_(False),
        readout=original.readout.to(dtype=dtype),
        catalog=MechanismCatalog(regional_execution=program),
    )


def expected_phases():
    return [
        {"round": 0, "phase": "group", "receivers": [5, 6], "read_version": 0, "write_version": 1},
        {"round": 0, "phase": "root", "receivers": [7], "read_version": 1, "write_version": 2},
        {"round": 0, "phase": "group", "receivers": [5, 6], "read_version": 2, "write_version": 3},
        {
            "round": 0,
            "phase": "local",
            "receivers": [0, 1, 2, 3, 4],
            "read_version": 3,
            "write_version": 4,
        },
        {
            "round": 0,
            "phase": "local",
            "receivers": [0, 1, 2, 3, 4],
            "read_version": 4,
            "write_version": 5,
        },
        {"round": 1, "phase": "group", "receivers": [5, 6], "read_version": 5, "write_version": 6},
        {"round": 1, "phase": "root", "receivers": [7], "read_version": 6, "write_version": 7},
        {"round": 1, "phase": "group", "receivers": [5, 6], "read_version": 7, "write_version": 8},
        {
            "round": 1,
            "phase": "local",
            "receivers": [0, 1, 2, 3, 4],
            "read_version": 8,
            "write_version": 9,
        },
        {
            "round": 1,
            "phase": "local",
            "receivers": [0, 1, 2, 3, 4],
            "read_version": 9,
            "write_version": 10,
        },
    ]


@pytest.fixture(scope="module")
def hierarchy_packet():
    previous = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        return collect(hierarchy_engine())[0]
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


@pytest.mark.parametrize("device", DEVICES)
def test_collection_archives_the_actual_ten_phases_without_expanding_actions(device):
    engine = hierarchy_engine(device)
    packet, observed = collect(engine)
    body = packet.payload()
    assert body["schema_version"] == 2
    assert body["regional_execution"]["schema_version"] == 2
    assert body["kind"] == "conditional_regional_program_collection_v1"
    assert body["blocks"][0]["regional_execution"] == observed
    assert [call["context"] for call in observed] == ["off", "parent"]
    assert len(body["catalog"]["actions"]) == len(body["groups"][0]["candidates"]) == 36
    assert body["work"]["core_observations"] == 4
    assert body["work"]["reader_refinements"] == 36
    assert {key: body["work"][key] for key in REGIONAL_WORK} == {
        "regional_local_node_evaluations": 80,
        "regional_summary_evaluations": 40,
        "regional_program_selection_rows": 4,
        "regional_context_query_rows": 2,
    }
    for call in observed:
        for row in call["rows"]:
            trace = row["trace"]
            assert trace["program"] == "hierarchical_group_root_v1"
            assert trace["phases"] == expected_phases()
            assert trace["work"] == {"local_node_evaluations": 20, "summary_evaluations": 10}
    assert all(row["trace"]["context_norm"] == 0 for row in observed[0]["rows"])
    assert all(row["trace"]["context_norm"] > 0 for row in observed[1]["rows"])
    assert CollectedMechanisms.restore(packet.record()).record() == packet.record()


@pytest.mark.parametrize("options", CONTROLS)
def test_all_seven_declared_controls_keep_their_identity_and_equal_work(options):
    engine = hierarchy_engine(options=options)
    packet, _ = collect(engine)
    body = packet.payload()
    configuration = hierarchy_config(**options).identity()
    assert body["catalog"]["regional_execution"] == configuration
    assert body["regional_execution"]["execution"]["configuration"] == configuration
    assert body["work"]["regional_local_node_evaluations"] == 80
    assert body["work"]["regional_summary_evaluations"] == 40
    assert CollectedMechanisms.restore(packet.record()).record() == packet.record()


def test_partitioned_collection_preserves_each_real_trace_and_exact_work():
    engine = hierarchy_engine()
    full, _ = collect(engine)
    split, observed = collect(engine, split=True)
    first, second = full.payload(), split.payload()
    assert {key: first["work"][key] for key in REGIONAL_WORK} == {
        key: second["work"][key] for key in REGIONAL_WORK
    }
    assert first["work"]["context_passes"] == 2
    assert second["work"]["context_passes"] == 4
    archived = [call for block in second["blocks"] for call in block["regional_execution"]]
    assert len(archived) == len(observed) == 4
    assert sum(len(call["rows"]) for call in archived) == 4
    assert sorted(archived, key=canonical) == sorted(observed, key=canonical)

    def by_flow(body):
        return {
            (call["context"], row["flow_id"]): row["trace"]
            for block in body["blocks"]
            for call in block["regional_execution"]
            for row in call["rows"]
        }

    original, partitioned = by_flow(first), by_flow(second)
    assert len(original) == len(partitioned) == 4
    assert set(original) == set(partitioned)
    for key, trace in original.items():
        other = partitioned[key]
        assert {name: value for name, value in trace.items() if name != "context_norm"} == {
            name: value for name, value in other.items() if name != "context_norm"
        }
        assert trace["context_norm"] == pytest.approx(other["context_norm"], rel=1e-12, abs=1e-12)


def test_an_unavailable_parent_has_only_off_traces_and_no_pt4_query_charge():
    packet, observed = collect(hierarchy_engine(parent=False))
    body = packet.payload()
    assert [call["context"] for call in observed] == ["off"]
    assert body["blocks"][0]["regional_execution"] == observed
    assert body["work"]["core_observations"] == 2
    assert body["work"]["regional_local_node_evaluations"] == 40
    assert body["work"]["regional_summary_evaluations"] == 20
    assert body["work"]["regional_context_query_rows"] == 0


@contextmanager
def observe_hierarchy(engine, *, change_availability=False):
    """Inspeccionar resultados reales sin sustituir los métodos del consumidor sellado."""
    module = engine.predictor.regional_execution
    states, changed = [], []

    def observe(frame, event, result):
        if event != "return" or result is None:
            return
        if frame.f_code is type(module).forward.__code__ and frame.f_locals.get("self") is module:
            states.append(
                (
                    result.working_state.detach().clone(),
                    result.trace["availability"].detach().clone(),
                )
            )
        if (
            change_availability
            and not changed
            and frame.f_code is FinancialPredictor.prepare.__code__
            and frame.f_locals.get("self") is engine.predictor
        ):
            # Fundamentales y macro siguen presentes, por lo que los OR no cambian.
            result.regional_trace["availability"][0, 1] = False
            changed.append(True)

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        yield states, changed
    finally:
        sys.setprofile(previous)


def test_all_32_masks_respect_financial_admission_and_real_empty_group_states():
    engine = hierarchy_engine(parent=False)
    template = raw_batch(absent=False)
    all_masks = list(product((False, True), repeat=5))
    admitted = [mask for mask in all_masks if mask[0] and mask[2]]
    rejected = [mask for mask in all_masks if not (mask[0] and mask[2])]
    assert len(admitted) == 8 and len(rejected) == 24
    masks = np.array(admitted, dtype=np.bool_)
    raw = dict(
        template,
        inputs={key: np.repeat(value[:1], 8, axis=0) for key, value in template["inputs"].items()},
        presence=masks,
        sample_ids=[f"US/M{index:02d}/{AT}" for index in range(8)],
        market=["US"] * 8,
        prediction_at=np.full(8, AT, dtype="datetime64[us]"),
        input_available_at=np.full(8, AT - 1, dtype="datetime64[us]"),
        target=ForbiddenEmbedding(),
        target_available_at=ForbiddenEmbedding(),
    )
    for index, key in enumerate(template["inputs"]):
        raw["inputs"][key][~masks[:, index]] = 0
    rows = observation_rows(engine, raw)
    state = engine.initial_state()
    fast = engine.predictor.initial_state(tuple(row.flow_id for row in rows))
    prior, prior_fast = engine.export_state(state), engine.predictor.export_state(fast)
    builder = MechanismCollector(
        engine, state, rows=rows, cutoff=AT, context_id=CONTEXT, matures_at=(AT + DAY,) * 8
    )
    with observe_hierarchy(engine) as (states, _):
        builder.add(rows, fast)
    packet = builder.finish()
    assert len(states) == 1
    working, available = states[0]
    expected = [
        [*mask, mask[0] or mask[2], mask[1] or mask[3] or mask[4], any(mask)]
        for mask in masks.tolist()
    ]
    assert working.shape == (8, 8, 32)
    assert available.dtype is torch.bool and available.tolist() == expected
    assert torch.isfinite(working).all()
    assert torch.count_nonzero(working[~available]) == 0
    call = packet.payload()["blocks"][0]["regional_execution"][0]
    assert [row["trace"]["availability"] for row in call["rows"]] == expected
    assert packet.payload()["work"]["regional_summary_evaluations"] == 80
    assert_same(engine.export_state(state), prior)
    assert_same(engine.predictor.export_state(fast), prior_fast)
    assert CollectedMechanisms.restore(packet.record()).record() == packet.record()
    with observed_preparations(engine) as calls:
        for mask in rejected:
            invalid = raw_batch(absent=False)
            invalid["presence"][0] = mask
            for index, key in enumerate(invalid["inputs"]):
                if not mask[index]:
                    invalid["inputs"][key][0] = 0
            with pytest.raises(ValueError):
                observation_rows(engine, invalid)
    assert not calls
    assert_same(engine.export_state(state), prior)
    assert_same(engine.predictor.export_state(fast), prior_fast)


def test_a_future_input_is_rejected_before_any_hierarchy_prepare():
    engine = hierarchy_engine(parent=False)
    rows = observation_rows(engine)
    future = replace(rows[0], input_available_at=AT + 1)
    future = replace(future, input_sha256=row_sha(future, engine.predictor.config.inputs))
    state = engine.initial_state()
    prior = engine.export_state(state)
    with observed_preparations(engine) as calls, pytest.raises(ValueError):
        MechanismCollector(
            engine,
            state,
            rows=(future, rows[1]),
            cutoff=AT,
            context_id=CONTEXT,
            matures_at=(AT + DAY,) * 2,
        )
    assert not calls
    assert_same(engine.export_state(state), prior)


def test_a_structurally_valid_but_input_inconsistent_trace_cannot_be_archived():
    engine = hierarchy_engine()
    rows = observation_rows(engine)
    assert rows[0].presence.tolist() == [True] * 5
    state = engine.initial_state()
    fast = engine.predictor.initial_state(tuple(row.flow_id for row in rows))
    before, before_fast = engine.export_state(state), engine.predictor.export_state(fast)
    builder = MechanismCollector(
        engine, state, rows=rows, cutoff=AT, context_id=CONTEXT, matures_at=(AT + DAY,) * 2
    )
    with (
        observe_hierarchy(engine, change_availability=True) as (_, changed),
        pytest.raises(ValueError),
    ):
        builder.add(rows, fast)
    assert changed == [True]
    assert rows[0].presence.tolist() == [True] * 5
    assert_same(engine.export_state(state), before)
    assert_same(engine.predictor.export_state(fast), before_fast)
    with pytest.raises(ValueError):
        builder.finish()
    builder.add(rows, fast)
    body = builder.finish().payload()
    assert body["work"]["core_observations"] == 4
    assert body["work"]["regional_summary_evaluations"] == 40


@pytest.mark.parametrize(
    "damage",
    [
        "availability_integer",
        "availability_length",
        "group_or",
        "root_or",
        "context_disagreement",
        "read_version_boolean",
        "write_version_float",
        "round_boolean",
        "wrong_receivers",
        "phase_order",
        "missing_phase",
        "summary_nine",
        "descriptor_downgrade",
    ],
)
def test_resealed_hierarchy_packet_rejects_incoherent_structure(hierarchy_packet, damage):
    body = deepcopy(hierarchy_packet.payload())
    calls = body["blocks"][0]["regional_execution"]
    trace = calls[0]["rows"][0]["trace"]
    if damage == "availability_integer":
        trace["availability"][0] = 1
    elif damage == "availability_length":
        trace["availability"].pop()
    elif damage == "group_or":
        trace["availability"][5] = False
    elif damage == "root_or":
        trace["availability"][7] = False
    elif damage == "context_disagreement":
        # Ambas máscaras cumplen sus OR, pero pertenecen al mismo input identificado.
        trace["availability"][1] = False
    elif damage == "read_version_boolean":
        trace["phases"][0]["read_version"] = False
    elif damage == "write_version_float":
        trace["phases"][0]["write_version"] = 1.0
    elif damage == "round_boolean":
        trace["phases"][0]["round"] = False
    elif damage == "wrong_receivers":
        trace["phases"][0]["receivers"] = [5, 7]
    elif damage == "phase_order":
        trace["phases"][0], trace["phases"][1] = trace["phases"][1], trace["phases"][0]
    elif damage == "missing_phase":
        trace["phases"].pop()
    elif damage == "summary_nine":
        trace["work"]["summary_evaluations"] = 9
        body["work"]["regional_summary_evaluations"] -= 1
    else:
        body["regional_execution"]["schema_version"] = 1
    with pytest.raises(ValueError):
        CollectedMechanisms(body)


def test_archive_restore_does_not_construct_neural_modules(hierarchy_packet, monkeypatch):
    record = hierarchy_packet.record()

    def forbidden(*args, **kwargs):
        pytest.fail("La recuperación archivada no debe construir módulos neuronales")

    monkeypatch.setattr(torch.nn.Module, "__init__", forbidden)
    assert CollectedMechanisms.restore(record).record() == record
