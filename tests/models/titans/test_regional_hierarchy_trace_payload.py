"""Validación de trazas crudas y propiedad de las filas exportadas."""

import copy
import json

import pytest
import torch
from test_regional_hierarchy import DEVICES, DTYPES, PROGRAM, data, make


@pytest.fixture(scope="module")
def trace_case():
    api, operator, run, _ = make(max_batch=3)
    blocks, presence, context = data()
    with torch.no_grad():
        trace = run(operator, blocks, presence, context).trace
    return api, operator, run, trace


def change_norm(norms, value):
    changed = norms.clone()
    changed[0] = value
    return changed


def flip_available(available, row, node):
    changed = available.clone()
    changed[row, node] = ~changed[row, node]
    return changed


def change_phase(phases, index=0, **changes):
    changed = copy.deepcopy(phases)
    changed[PROGRAM][index].update(changes)
    return changed


def without_phase_field(phases, field):
    changed = copy.deepcopy(phases)
    changed[PROGRAM][0].pop(field)
    return changed


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_exported_rows_and_their_nested_fields_have_no_aliases(device, dtype):
    _, operator, run, _ = make(device, dtype, max_batch=3)
    blocks, presence, context = data(device, dtype)
    with torch.no_grad():
        trace = run(operator, blocks, presence, context).trace
    original = copy.deepcopy(trace)
    rows = run.export_trace(trace, operator)
    assert len(rows) == run.max_batch == 3
    expected = dict(local_node_evaluations=60, summary_evaluations=30)
    assert run.verify_exported_traces(rows, operator) == expected
    other = copy.deepcopy(rows[1])
    cloned = json.loads(json.dumps(rows))
    cloned[0]["context_norm"] += 1.0
    assert cloned[1] == other
    # El programa fijo valida finitud y estructura, no autentica el contexto no archivado.
    assert run.verify_exported_traces(cloned, operator) == expected
    rows[0]["phases"][0]["receivers"][0] = 0
    rows[0]["phases"][0]["read_version"] = 99
    rows[0]["availability"][0] = not rows[0]["availability"][0]
    rows[0]["work"]["summary_evaluations"] = 999
    assert rows[1] == other
    assert trace["phases"] == original["phases"]
    assert trace["work"] == original["work"]
    assert torch.equal(trace["context_norms"], original["context_norms"])
    assert torch.equal(trace["availability"], original["availability"])
    assert run.export_trace(trace, operator) == tuple(
        json.loads(json.dumps(run.export_trace(original, operator)))
    )


@pytest.mark.parametrize("device", DEVICES)
def test_export_accepts_host_snapshots_without_requiring_the_operator_device(device):
    _, operator, run, _ = make(device)
    blocks, presence, context = data(device)
    with torch.no_grad():
        trace = run(operator, blocks, presence, context).trace
    trace["context_norms"] = trace["context_norms"].cpu()
    trace["availability"] = trace["availability"].cpu()
    rows = run.export_trace(trace, operator)
    assert run.verify_exported_traces(rows, operator) == dict(
        local_node_evaluations=60, summary_evaluations=30
    )


def test_strided_noncontiguous_snapshots_and_zero_norm_are_valid(trace_case):
    _, operator, run, raw = trace_case
    trace = copy.deepcopy(raw)
    norms = torch.zeros_like(trace["context_norms"])
    trace["context_norms"] = torch.stack((norms, norms), dim=-1)[:, 0]
    available = trace["availability"]
    trace["availability"] = torch.stack((available, available), dim=-1)[..., 0]
    assert not trace["context_norms"].is_contiguous()
    assert not trace["availability"].is_contiguous()
    rows = run.export_trace(trace, operator)
    assert all(row["context_norm"] == 0.0 for row in rows)
    assert run.verify_exported_traces(rows, operator) == dict(
        local_node_evaluations=60, summary_evaluations=30
    )


@pytest.mark.parametrize("fault", ["not_dict", "missing", "extra"])
def test_raw_trace_requires_its_complete_exact_field_set(trace_case, fault):
    _, operator, run, raw = trace_case
    trace = copy.deepcopy(raw)
    if fault == "not_dict":
        trace = []
    elif fault == "missing":
        trace.pop("availability")
    else:
        trace["unregistered_state"] = True
    with pytest.raises(ValueError):
        run.export_trace(trace, operator)


@pytest.mark.parametrize(
    "field,transform",
    [
        pytest.param("configuration_id", lambda value: "0" * 64, id="foreign_configuration"),
        pytest.param("configuration_id", lambda value: 1, id="configuration_type"),
        pytest.param("programs", list, id="programs_list"),
        pytest.param("programs", lambda value: PROGRAM, id="programs_scalar"),
        pytest.param("programs", lambda value: (), id="programs_empty"),
        pytest.param(
            "programs", lambda value: ("summary_interleaved_v1",) * len(value), id="foreign_program"
        ),
        pytest.param("programs", lambda value: (1,) + value[1:], id="program_type"),
        pytest.param("context_norms", lambda value: value.tolist(), id="norms_not_tensor"),
        pytest.param("context_norms", lambda value: value[:, None], id="norms_shape"),
        pytest.param("context_norms", lambda value: value.float(), id="norms_dtype"),
        pytest.param("context_norms", lambda value: value.to_sparse(), id="norms_sparse"),
        pytest.param(
            "context_norms", lambda value: value.clone().requires_grad_(True), id="norms_leaf_grad"
        ),
        pytest.param(
            "context_norms", lambda value: value.clone().requires_grad_(True) * 2, id="norms_graph"
        ),
        pytest.param(
            "context_norms", lambda value: change_norm(value, float("nan")), id="norms_nan"
        ),
        pytest.param(
            "context_norms", lambda value: change_norm(value, float("inf")), id="norms_positive_inf"
        ),
        pytest.param(
            "context_norms",
            lambda value: change_norm(value, -float("inf")),
            id="norms_negative_inf",
        ),
        pytest.param("context_norms", lambda value: change_norm(value, -0.25), id="norms_negative"),
        pytest.param("availability", lambda value: value.tolist(), id="availability_not_tensor"),
        pytest.param("availability", lambda value: value[:, :7], id="availability_shape"),
        pytest.param("availability", lambda value: value.float(), id="availability_dtype"),
        pytest.param("availability", lambda value: value.to_sparse(), id="availability_sparse"),
        pytest.param(
            "availability", lambda value: flip_available(value, 0, 5), id="available_group_missing"
        ),
        pytest.param(
            "availability", lambda value: flip_available(value, 1, 6), id="absent_group_created"
        ),
        pytest.param(
            "availability", lambda value: flip_available(value, 1, 7), id="absent_root_created"
        ),
        pytest.param(
            "availability", lambda value: flip_available(value, 0, 0), id="leaves_group_disagree"
        ),
        pytest.param("phases", lambda value: list(value.values()), id="phases_not_dict"),
        pytest.param("phases", lambda value: {}, id="phases_missing_program"),
        pytest.param(
            "phases", lambda value: {"other": value[PROGRAM]}, id="phases_foreign_program"
        ),
        pytest.param("phases", lambda value: {PROGRAM: 10}, id="phases_not_sequence"),
        pytest.param("phases", lambda value: {PROGRAM: value[PROGRAM][:-1]}, id="phases_count"),
        pytest.param(
            "phases", lambda value: {PROGRAM: ["group", *value[PROGRAM][1:]]}, id="phase_not_dict"
        ),
        pytest.param("phases", lambda value: change_phase(value, phase="local"), id="phase_order"),
        pytest.param(
            "phases", lambda value: change_phase(value, receivers=[7]), id="phase_receivers"
        ),
        pytest.param(
            "phases", lambda value: change_phase(value, receivers=[6, 5]), id="phase_receiver_order"
        ),
        pytest.param(
            "phases",
            lambda value: change_phase(value, receivers=[True, 6]),
            id="phase_receiver_type",
        ),
        pytest.param(
            "phases", lambda value: change_phase(value, round=False), id="phase_round_type"
        ),
        pytest.param(
            "phases",
            lambda value: change_phase(value, index=1, read_version=True),
            id="phase_version_type",
        ),
        pytest.param(
            "phases",
            lambda value: change_phase(value, index=1, read_version=0),
            id="phase_version_order",
        ),
        pytest.param(
            "phases", lambda value: change_phase(value, unknown=True), id="phase_extra_field"
        ),
        pytest.param(
            "phases",
            lambda value: without_phase_field(value, "write_version"),
            id="phase_missing_field",
        ),
        pytest.param("work", lambda value: list(value.items()), id="work_not_dict"),
        pytest.param("work", lambda value: {"local_node_evaluations": 60}, id="work_missing_unit"),
        pytest.param("work", lambda value: dict(value, other=0), id="work_extra_unit"),
        pytest.param(
            "work", lambda value: dict(value, local_node_evaluations=59), id="work_local_count"
        ),
        pytest.param(
            "work", lambda value: dict(value, summary_evaluations=29), id="work_summary_count"
        ),
        pytest.param("work", lambda value: dict(value, summary_evaluations=30.0), id="work_float"),
        pytest.param("work", lambda value: dict(value, summary_evaluations=True), id="work_bool"),
    ],
)
def test_raw_trace_rejects_corrupt_payload_before_export(trace_case, field, transform):
    _, operator, run, raw = trace_case
    trace = copy.deepcopy(raw)
    trace[field] = transform(trace[field])
    with pytest.raises(ValueError):
        run.export_trace(trace, operator)


def test_export_rejects_one_row_above_the_real_block_limit(trace_case):
    _, operator, run, raw = trace_case
    trace = copy.deepcopy(raw)
    assert len(trace["programs"]) == run.max_batch == 3
    trace["programs"] = (PROGRAM,) * 4
    trace["context_norms"] = torch.cat((trace["context_norms"], trace["context_norms"][:1]))
    trace["availability"] = torch.cat((trace["availability"], trace["availability"][:1]))
    trace["work"] = dict(local_node_evaluations=80, summary_evaluations=40)
    with pytest.raises(ValueError):
        run.export_trace(trace, operator)
