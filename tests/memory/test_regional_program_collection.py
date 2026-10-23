"""Trazas observadas de programas regionales en alternativas desde un prefijo común."""

import json
import sys
from contextlib import contextmanager

import pytest
import torch
from test_episodic_readout import CONTEXT
from test_frozen_financial import frozen_backend as frozen_backend
from test_market_memory import KINDS, pt4
from test_mechanism_consumer import consumer, execute
from test_mechanism_observations import AT, DAY, observation_rows

from mars_titan.memory.mechanism_collection import CollectedMechanisms, MechanismCollector
from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.financial import FinancialPredictor
from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismCatalog
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.models.titans.regional_execution import RegionalExecutionConfig
from mars_titan.models.titans.regional_fusion import RegionalFusionConfig

DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))
AT_END = "summary_at_end_v1"
INTERLEAVED = "summary_interleaved_v1"
DEFAULT_PROGRAM = RegionalExecutionConfig()
REGIONAL_WORK = {
    "regional_local_node_evaluations",
    "regional_summary_evaluations",
    "regional_program_selection_rows",
    "regional_context_query_rows",
}
LEGACY_WORK = {
    "core_observations",
    "mac_online_observations",
    "reader_refinements",
    "reader_searches",
    "reader_value_projections",
    "context_passes",
    "shared_route_batches",
    "individual_route_batches",
    "calibration_attempts",
    "calibration_created_proposals",
    "calibration_discarded_proposals",
}


def engine_on(
    device="cpu",
    *,
    program=DEFAULT_PROGRAM,
    parent=True,
    summary=True,
    dtype=torch.float64,
    context_gain=0.2,
):
    kind = KINDS[0] if parent else None
    original = consumer(device, kind=kind, reader=True)
    model = pt4(
        kind,
        head=QUANTILE_HEAD,
        regional_fusion=RegionalFusionConfig(
            topology="ring", cycles=3, local_steps=1, summary=summary
        ),
        regional_execution=program,
    ).to(device=device, dtype=dtype)
    if parent:
        weights = model.state_dict()
        weights["session_context.output"] = torch.eye(32, device=device, dtype=dtype) * context_gain
        model.load_state_dict(weights)
    return FrozenMechanismConsumer(
        model.eval().requires_grad_(False),
        readout=original.readout.to(dtype=dtype),
        catalog=MechanismCatalog(regional_execution=program),
    )


@contextmanager
def observed_preparations(engine, *, corrupt=False):
    """Observar retornos reales sin sustituir métodos ni añadir hooks al modelo sellado."""
    calls = []

    def observe(frame, event, result):
        if (
            event == "return"
            and frame.f_code is FinancialPredictor.prepare.__code__
            and frame.f_locals.get("self") is engine.predictor
            and result is not None
        ):
            batch = frame.f_locals["batch"]
            selected = frame.f_locals["context_selection"]
            traces = engine.predictor.regional_execution.export_trace(
                result.regional_trace, engine.predictor.regional_fusion
            )
            calls.append(
                json.loads(
                    canonical(
                        dict(
                            context="parent" if selected is not None and any(selected) else "off",
                            rows=[
                                dict(flow_id=flow, trace=trace)
                                for flow, trace in zip(batch.flow_ids, traces, strict=True)
                            ],
                        )
                    )
                )
            )
            if corrupt and len(calls) == 1:
                result.regional_trace["work"]["summary_evaluations"] += 1

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        yield calls
    finally:
        sys.setprofile(previous)


def collect(engine, *, split=False, populated=True):
    state = engine.initial_state()
    flows = ("US/AAA", "US/BBB")
    fast = engine.predictor.initial_state(flows)
    if populated and engine.predictor.session_context is not None:
        previous, prepared = execute(engine, state, event=-1, action=MechanismAction("parent"))
        state, fast = previous.state, prepared[0]
    rows = observation_rows(engine)
    builder = MechanismCollector(
        engine, state, rows=rows, cutoff=AT, context_id=CONTEXT, matures_at=(AT + DAY,) * 2
    )
    with observed_preparations(engine) as calls:
        for block in (rows[1:], rows[:1]) if split else (rows,):
            block_state = engine.predictor.select_state(fast, tuple(row.flow_id for row in block))
            builder.add(block, block_state)
    return builder.finish(), calls


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "program,summaries",
    [
        (RegionalExecutionConfig(), 4),
        (RegionalExecutionConfig(program=INTERLEAVED), 8),
        (RegionalExecutionConfig(selection="context_norm_v1", context_threshold=0.0), 6),
    ],
)
def test_collection_archives_each_actual_prepare_without_expanding_actions(
    device, program, summaries
):
    engine = engine_on(device, program=program)
    packet, observed = collect(engine)
    body = packet.payload()
    assert body["schema_version"] == 2
    assert body["kind"] == "conditional_regional_program_collection_v1"
    assert body["regional_execution"] == engine.predictor.regional_execution.trace_descriptor(
        engine.predictor.regional_fusion
    )
    assert body["blocks"][0]["regional_execution"] == observed
    assert [call["context"] for call in observed] == ["off", "parent"]
    assert len(body["groups"][0]["candidates"]) == 36
    assert len(body["catalog"]["actions"]) == 36
    assert body["work"]["core_observations"] == 4
    assert body["work"]["reader_refinements"] == 36
    assert {name: body["work"][name] for name in REGIONAL_WORK} == dict(
        regional_local_node_evaluations=80,
        regional_summary_evaluations=summaries,
        regional_program_selection_rows=4,
        regional_context_query_rows=2,
    )
    off, parent = body["blocks"][0]["regional_execution"]
    assert all(row["trace"]["context_norm"] == 0 for row in off["rows"])
    assert all(row["trace"]["context_norm"] > 0 for row in parent["rows"])
    assert CollectedMechanisms.restore(packet.record()).record() == packet.record()


@pytest.mark.parametrize("summary", [False, True])
def test_partitioned_blocks_preserve_per_flow_traces_and_exact_work(summary):
    engine = engine_on(program=RegionalExecutionConfig(program=INTERLEAVED), summary=summary)
    full, _ = collect(engine)
    split, observed = collect(engine, split=True)
    first, second = full.payload(), split.payload()
    assert {name: first["work"][name] for name in REGIONAL_WORK} == {
        name: second["work"][name] for name in REGIONAL_WORK
    }
    assert first["work"]["context_passes"] == 2
    assert second["work"]["context_passes"] == 4
    assert second["work"]["regional_summary_evaluations"] == (8 if summary else 0)
    flattened = [call for block in second["blocks"] for call in block["regional_execution"]]
    assert sorted(flattened, key=canonical) == sorted(observed, key=canonical)

    def by_row(body):
        return {
            (call["context"], row["flow_id"]): row["trace"]
            for block in body["blocks"]
            for call in block["regional_execution"]
            for row in call["rows"]
        }

    original, partitioned = by_row(first), by_row(second)
    assert set(original) == set(partitioned)
    for key, trace in original.items():
        other = partitioned[key]
        assert {name: value for name, value in trace.items() if name != "context_norm"} == {
            name: value for name, value in other.items() if name != "context_norm"
        }
        assert trace["context_norm"] == pytest.approx(other["context_norm"], rel=1e-12, abs=1e-12)


def test_unavailable_parent_has_one_off_trace_and_no_context_query_work():
    engine = engine_on(parent=False, program=RegionalExecutionConfig(program=INTERLEAVED))
    packet, observed = collect(engine)
    body = packet.payload()
    assert [call["context"] for call in observed] == ["off"]
    assert body["blocks"][0]["regional_execution"] == observed
    assert body["work"]["regional_local_node_evaluations"] == 40
    assert body["work"]["regional_summary_evaluations"] == 4
    assert body["work"]["regional_program_selection_rows"] == 2
    assert body["work"]["regional_context_query_rows"] == 0


def test_empty_parent_with_zero_output_preserves_zero_norm_and_real_queries():
    engine = engine_on(
        program=RegionalExecutionConfig(selection="context_norm_v1", context_threshold=0.0),
        context_gain=0.0,
    )
    packet, _ = collect(engine, populated=False)
    body = packet.payload()
    assert body["work"]["regional_context_query_rows"] == 2
    assert body["work"]["regional_summary_evaluations"] == 4
    for call in body["blocks"][0]["regional_execution"]:
        assert all(row["trace"]["context_norm"] == 0 for row in call["rows"])
        assert all(row["trace"]["program"] == AT_END for row in call["rows"])


def test_fp32_archive_uses_descriptor_precision_for_a_context_threshold_tie():
    engine = engine_on(
        dtype=torch.float32,
        program=RegionalExecutionConfig(selection="context_norm_v1", context_threshold=0.1),
        context_gain=0.0,
    )
    packet, _ = collect(engine, populated=False)
    body = packet.payload()
    assert body["output_dtype"] == body["regional_execution"]["dtype"] == "torch.float32"
    parent = body["blocks"][0]["regional_execution"][1]
    parent["rows"][0]["trace"]["context_norm"] = 0.10000000149011612
    restored = CollectedMechanisms(body)
    assert restored.payload()["work"]["regional_summary_evaluations"] == 4


def test_resealed_contextual_trace_must_preserve_the_selected_program():
    engine = engine_on(
        program=RegionalExecutionConfig(selection="context_norm_v1", context_threshold=0.0)
    )
    packet, _ = collect(engine)
    body = packet.payload()
    parent = body["blocks"][0]["regional_execution"][1]
    assert parent["rows"][0]["trace"]["program"] == INTERLEAVED
    parent["rows"][0]["trace"]["context_norm"] = 0.0
    with pytest.raises(ValueError):
        CollectedMechanisms(body)


def test_none_preserves_legacy_collection_schema_and_work_fields():
    engine = engine_on(program=None)
    rows = observation_rows(engine)
    builder = MechanismCollector(
        engine,
        engine.initial_state(),
        rows=rows,
        cutoff=AT,
        context_id=CONTEXT,
        matures_at=(AT + DAY,) * 2,
    )
    builder.add(rows, engine.predictor.initial_state(tuple(row.flow_id for row in rows)))
    packet = builder.finish()
    body = packet.payload()
    assert body["schema_version"] == 1
    assert body["kind"] == "conditional_mechanism_collection_v1"
    assert "regional_execution" not in body
    assert set(body["work"]) == LEGACY_WORK
    assert all(set(block) == {"flow_ids", "fast_state_sha256"} for block in body["blocks"])
    assert CollectedMechanisms.restore(packet.record()).record() == packet.record()


@pytest.mark.parametrize("counter", sorted(REGIONAL_WORK))
def test_resealed_packet_rejects_changed_regional_work(counter):
    packet, _ = collect(engine_on())
    body = packet.payload()
    body["work"][counter] += 1
    with pytest.raises(ValueError):
        CollectedMechanisms(body)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_trace",
        "missing_context",
        "duplicate_context",
        "missing_flow",
        "duplicate_flow",
        "wrong_flow",
        "off_context_norm",
        "negative_norm",
        "phase_order",
        "row_work",
        "program",
        "descriptor_dtype",
        "descriptor_hidden",
        "catalog_program",
        "downgrade",
    ],
)
def test_resealed_packet_rejects_incoherent_context_flow_program_or_descriptor(fault):
    packet, _ = collect(engine_on())
    body = packet.payload()
    block = body["blocks"][0]
    calls = block["regional_execution"]
    row = calls[0]["rows"][0]
    trace = row["trace"]
    if fault == "missing_trace":
        del block["regional_execution"]
    elif fault == "missing_context":
        calls.pop()
    elif fault == "duplicate_context":
        calls.append(calls[0])
    elif fault == "missing_flow":
        calls[0]["rows"].pop()
    elif fault == "duplicate_flow":
        calls[0]["rows"].append(row)
    elif fault == "wrong_flow":
        row["flow_id"] = "US/UNKNOWN"
    elif fault == "off_context_norm":
        trace["context_norm"] = 0.25
    elif fault == "negative_norm":
        trace["context_norm"] = -0.25
    elif fault == "phase_order":
        trace["phases"] = ["summary", "local", "local", "local", "local"]
    elif fault == "row_work":
        trace["work"]["local_node_evaluations"] += 1
    elif fault == "program":
        trace["program"] = INTERLEAVED
        trace["phases"] = ["local", "local", "summary", "local", "local", "summary"]
        trace["work"]["summary_evaluations"] = 2
        body["work"]["regional_summary_evaluations"] += 1
    elif fault == "descriptor_dtype":
        body["regional_execution"]["dtype"] = "torch.float32"
    elif fault == "descriptor_hidden":
        body["regional_execution"]["operator"]["hidden_size"] = 64
    elif fault == "catalog_program":
        body["catalog"]["regional_execution"] = RegionalExecutionConfig(
            program=INTERLEAVED
        ).identity()
    else:
        body["schema_version"] = 1
        body["kind"] = "conditional_mechanism_collection_v1"
        del body["regional_execution"]
        for item in body["blocks"]:
            del item["regional_execution"]
        for name in REGIONAL_WORK:
            del body["work"][name]
    with pytest.raises(ValueError):
        CollectedMechanisms(body)


def test_invalid_observed_trace_does_not_leave_a_partial_block():
    engine = engine_on()
    rows = observation_rows(engine)
    builder = MechanismCollector(
        engine,
        engine.initial_state(),
        rows=rows,
        cutoff=AT,
        context_id=CONTEXT,
        matures_at=(AT + DAY,) * 2,
    )
    fast = engine.predictor.initial_state(tuple(row.flow_id for row in rows))
    with observed_preparations(engine, corrupt=True), pytest.raises(ValueError):
        builder.add(rows, fast)
    with pytest.raises(ValueError):
        builder.finish()
    builder.add(rows, fast)
    assert builder.finish().payload()["work"]["regional_program_selection_rows"] == 4


def test_parser_restores_record_without_constructing_a_live_operator(monkeypatch):
    packet, _ = collect(engine_on())

    def forbidden_module(*args, **kwargs):
        pytest.fail("La recuperación del archivo no debe construir módulos neuronales")

    monkeypatch.setattr(torch.nn.Module, "__init__", forbidden_module)
    assert CollectedMechanisms.restore(packet.record()).record() == packet.record()
