"""Contrato estructural de las trazas regionales y su recuperación desde JSON."""

import copy
import json

import pytest
import torch

from mars_titan.models.titans.regional_execution import RegionalExecution, RegionalExecutionConfig
from mars_titan.models.titans.regional_fusion import RegionalFusion, RegionalFusionConfig

AT_END = "summary_at_end_v1"
INTERLEAVED = "summary_interleaved_v1"
PHASES = {
    AT_END: ("local", "local", "local", "local", "summary"),
    INTERLEAVED: ("local", "local", "summary", "local", "local", "summary"),
}
SUMMARY_COUNTS = {AT_END: 1, INTERLEAVED: 2}


def operator_for(*, dtype=torch.float64, hidden=3, summary=True):
    config = RegionalFusionConfig(topology="ring", cycles=3, local_steps=1, summary=summary)
    return RegionalFusion(hidden, config, seed=73, max_batch=8).to(dtype=dtype).eval()


def trace_for(*, program=AT_END, selection="fixed", threshold=None, summary=True):
    run = RegionalExecution(
        3,
        RegionalExecutionConfig(program=program, selection=selection, context_threshold=threshold),
        max_batch=8,
    )
    operator = operator_for(summary=summary)
    blocks = torch.arange(30, dtype=torch.float64).reshape(2, 5, 3) / 17
    presence = torch.tensor([[True] * 5, [False] * 5])
    context = blocks.new_tensor([[0, 0, 0], [2, 0, 0]])
    return run, operator, run(operator, blocks, presence, context).trace


def rewrite_trace_program(trace, program, *, summary):
    changed = copy.deepcopy(trace)
    rows = len(changed["programs"])
    changed["programs"] = (program,) * rows
    changed["phases"] = {program: PHASES[program] if summary else ("local",) * 4}
    changed["work"] = dict(
        local_node_evaluations=20 * rows,
        summary_evaluations=SUMMARY_COUNTS[program] * rows if summary else 0,
    )
    return changed


def archive(run, operator, trace):
    return json.loads(json.dumps(run.export_trace(trace, operator), allow_nan=False))


@pytest.mark.parametrize("program", [AT_END, INTERLEAVED])
@pytest.mark.parametrize("summary", [False, True])
def test_json_recovery_preserves_row_phases_and_observed_work(program, summary):
    run, operator, trace = trace_for(program=program, summary=summary)
    rows = archive(run, operator, trace)
    expected_phases = list(PHASES[program] if summary else ("local",) * 4)
    expected_work = dict(
        local_node_evaluations=20,
        summary_evaluations=SUMMARY_COUNTS[program] if summary else 0,
    )
    assert len(rows) == 2
    for row, norm in zip(rows, [0.0, 2.0], strict=True):
        assert set(row) == {
            "configuration_id",
            "operator_id",
            "program",
            "context_norm",
            "phases",
            "work",
        }
        assert row["program"] == program
        assert row["context_norm"] == norm
        assert row["phases"] == expected_phases
        assert row["work"] == expected_work
    assert run.verify_exported_traces(rows, operator) == dict(
        local_node_evaluations=40,
        summary_evaluations=2 * SUMMARY_COUNTS[program] if summary else 0,
    )


def test_fp32_threshold_tie_remains_fallback_after_json_recovery():
    run = RegionalExecution(
        3,
        RegionalExecutionConfig(selection="context_norm_v1", context_threshold=0.1),
        max_batch=8,
    )
    operator = operator_for(dtype=torch.float32)
    blocks = torch.zeros((1, 5, 3), dtype=torch.float32)
    context = blocks.new_tensor([[0.1, 0, 0]])
    trace = run(operator, blocks, None, context).trace
    assert trace["programs"] == (AT_END,)
    rows = archive(run, operator, trace)
    assert rows[0]["context_norm"] == 0.10000000149011612
    assert rows[0]["context_norm"] > 0.1
    assert rows[0]["program"] == AT_END
    assert run.verify_exported_traces(rows, operator) == dict(
        local_node_evaluations=20, summary_evaluations=1
    )


def test_json_recovery_preserves_mixed_contextual_selection():
    run, operator, trace = trace_for(selection="context_norm_v1", threshold=1.0)
    rows = archive(run, operator, trace)
    assert [row["program"] for row in rows] == [AT_END, INTERLEAVED]
    assert [row["phases"] for row in rows] == [list(PHASES[AT_END]), list(PHASES[INTERLEAVED])]
    assert run.verify_exported_traces(rows, operator) == dict(
        local_node_evaluations=40, summary_evaluations=3
    )


@pytest.mark.parametrize("original,changed", [(AT_END, INTERLEAVED), (INTERLEAVED, AT_END)])
@pytest.mark.parametrize("summary", [False, True])
def test_export_rejects_changed_fixed_program_with_consistent_phases_and_work(
    original, changed, summary
):
    run, operator, trace = trace_for(program=original, summary=summary)
    changed_trace = rewrite_trace_program(trace, changed, summary=summary)
    assert changed_trace["configuration_id"] == trace["configuration_id"]
    with pytest.raises(ValueError):
        run.export_trace(changed_trace, operator)


@pytest.mark.parametrize(
    "norm,wrong_program", [(0.0, INTERLEAVED), (1.0, INTERLEAVED), (2.0, AT_END)]
)
def test_export_rejects_program_inconsistent_with_context_norm(norm, wrong_program):
    run, operator, trace = trace_for(selection="context_norm_v1", threshold=1.0)
    changed = rewrite_trace_program(trace, wrong_program, summary=True)
    changed["context_norms"].fill_(norm)
    with pytest.raises(ValueError):
        run.export_trace(changed, operator)


@pytest.mark.parametrize("norm", [-0.25, float("nan"), float("inf"), float("-inf")])
def test_export_rejects_negative_or_nonfinite_context_norm(norm):
    run, operator, trace = trace_for()
    trace["context_norms"][0] = norm
    with pytest.raises(ValueError):
        run.export_trace(trace, operator)


def test_export_rejects_operator_with_different_hidden_size():
    run, _, trace = trace_for()
    with pytest.raises(ValueError):
        run.export_trace(trace, operator_for(hidden=4))


@pytest.mark.parametrize("counter", ["local_node_evaluations", "summary_evaluations"])
def test_export_rejects_changed_aggregate_work(counter):
    run, operator, trace = trace_for()
    trace["work"][counter] += 1
    with pytest.raises(ValueError):
        run.export_trace(trace, operator)


@pytest.mark.parametrize("original,changed", [(AT_END, INTERLEAVED), (INTERLEAVED, AT_END)])
@pytest.mark.parametrize("summary", [False, True])
def test_archive_rejects_changed_fixed_program_with_consistent_phases_and_work(
    original, changed, summary
):
    run, operator, trace = trace_for(program=original, summary=summary)
    rows = archive(run, operator, trace)
    for row in rows:
        row["program"] = changed
        row["phases"] = list(PHASES[changed] if summary else ("local",) * 4)
        row["work"]["summary_evaluations"] = SUMMARY_COUNTS[changed] if summary else 0
    with pytest.raises(ValueError):
        run.verify_exported_traces(rows, operator)


@pytest.mark.parametrize(
    "norm,wrong_program", [(0.0, INTERLEAVED), (1.0, INTERLEAVED), (2.0, AT_END)]
)
def test_archive_rejects_program_inconsistent_with_context_norm(norm, wrong_program):
    run, operator, trace = trace_for(selection="context_norm_v1", threshold=1.0)
    rows = archive(run, operator, trace)
    for row in rows:
        row["context_norm"] = norm
        row["program"] = wrong_program
        row["phases"] = list(PHASES[wrong_program])
        row["work"]["summary_evaluations"] = SUMMARY_COUNTS[wrong_program]
    with pytest.raises(ValueError):
        run.verify_exported_traces(rows, operator)


@pytest.mark.parametrize("norm", [-0.25, float("nan"), float("inf"), float("-inf")])
def test_archive_rejects_negative_or_nonfinite_context_norm(norm):
    run, operator, trace = trace_for()
    rows = archive(run, operator, trace)
    rows[0]["context_norm"] = norm
    with pytest.raises(ValueError):
        run.verify_exported_traces(rows, operator)


@pytest.mark.parametrize(
    "fault",
    ["configuration_id", "operator_id", "missing_field", "extra_field", "phases", "work"],
)
def test_archive_rejects_changed_identity_schema_phases_or_row_work(fault):
    run, operator, trace = trace_for()
    rows = archive(run, operator, trace)
    row = rows[0]
    if fault in ("configuration_id", "operator_id"):
        row[fault] = "0" * 64
    elif fault == "missing_field":
        del row["context_norm"]
    elif fault == "extra_field":
        row["unregistered"] = 0
    elif fault == "phases":
        row["phases"] = ["summary", "local", "local", "local", "local"]
    else:
        row["work"]["local_node_evaluations"] += 1
    with pytest.raises(ValueError):
        run.verify_exported_traces(rows, operator)


def test_archive_rejects_operator_with_different_hidden_size():
    run, operator, trace = trace_for()
    rows = archive(run, operator, trace)
    with pytest.raises(ValueError):
        run.verify_exported_traces(rows, operator_for(hidden=4))


def test_archive_validation_does_not_authenticate_an_unrecorded_context():
    run, operator, trace = trace_for()
    rows = archive(run, operator, trace)
    rows[0]["context_norm"] = 0.75
    assert run.verify_exported_traces(rows, operator) == dict(
        local_node_evaluations=40, summary_evaluations=2
    )


def test_archive_cohort_limit_is_independent_of_the_physical_block_limit():
    run, operator, trace = trace_for()
    row = archive(run, operator, trace)[0]
    assert run.max_batch == 8
    assert run.verify_exported_traces([row] * 8192, operator) == dict(
        local_node_evaluations=163840, summary_evaluations=8192
    )
    with pytest.raises(ValueError):
        run.verify_exported_traces([row] * 8193, operator)


def test_unrepresentable_context_threshold_is_rejected_as_a_configuration_error():
    with pytest.raises(ValueError):
        RegionalExecutionConfig(selection="context_norm_v1", context_threshold=10**1000)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_large_finite_integer_threshold_has_the_same_numeric_meaning_in_forward_and_archive(dtype):
    run = RegionalExecution(
        3,
        RegionalExecutionConfig(selection="context_norm_v1", context_threshold=10**100),
        max_batch=8,
    )
    operator = operator_for(dtype=dtype)
    blocks = torch.zeros((1, 5, 3), dtype=dtype)
    context = torch.ones((1, 3), dtype=dtype)
    trace = run(operator, blocks, None, context).trace
    assert trace["programs"] == (AT_END,)
    assert (
        run.verify_exported_traces(archive(run, operator, trace), operator)["summary_evaluations"]
        == 1
    )


def test_archive_threshold_rounding_uses_explicit_cpu_storage(monkeypatch):
    run, operator, trace = trace_for(selection="context_norm_v1", threshold=1.0)
    rows = archive(run, operator, trace)
    original = torch.scalar_tensor
    calls = []

    def checked(value, **options):
        assert options.get("device") == "cpu"
        calls.append(value)
        return original(value, **options)

    monkeypatch.setattr(torch, "scalar_tensor", checked)
    assert run.verify_exported_traces(rows, operator)["summary_evaluations"] == 3
    assert calls
