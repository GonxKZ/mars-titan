"""Despacho del archivo de ocho estados sin reinterpretar las trazas anteriores."""

import copy
import json
from dataclasses import asdict

import pytest
import torch
from test_financial_regional_hierarchy import execution_api, hierarchy_api, local_operator
from test_regional_execution_archive import archive, trace_for

from mars_titan.models.titans.regional_execution import RegionalExecutionConfig
from mars_titan.models.titans.regional_fusion import RegionalFusion

PROGRAM = "hierarchical_group_root_v1"


def hierarchy_trace():
    api = execution_api()
    config = hierarchy_api().RegionalHierarchyConfig()
    operator = RegionalFusion(3, local_operator(), seed=73, max_batch=2).double().eval()
    run = api.build_regional_execution(3, config, max_batch=2).double().eval()
    blocks = torch.arange(30, dtype=torch.float64).reshape(2, 5, 3) / 17
    presence = torch.tensor([[True] * 5, [True, False, True, False, False]])
    context = blocks.new_tensor([[0, 0, 0], [2, 0, 0]])
    return run, operator, run(operator, blocks, presence, context).trace


def json_rows(run, operator, trace):
    return json.loads(json.dumps(run.export_trace(trace, operator), allow_nan=False))


def expected_phases():
    return [
        dict(
            round=version // 5,
            phase=phase,
            receivers=receivers,
            read_version=version,
            write_version=version + 1,
        )
        for version, (phase, receivers) in enumerate(
            [
                ("group", [5, 6]),
                ("root", [7]),
                ("group", [5, 6]),
                ("local", [0, 1, 2, 3, 4]),
                ("local", [0, 1, 2, 3, 4]),
            ]
            * 2
        )
    ]


def test_schema_two_roundtrip_keeps_executed_availability_versions_and_work():
    run, operator, trace = hierarchy_trace()
    descriptor = json.loads(json.dumps(run.trace_descriptor(operator)))
    assert descriptor["schema_version"] == 2
    assert set(descriptor) == {"schema_version", "execution", "operator", "dtype"}
    assert descriptor["execution"]["configuration"]["kind"] == "pre_mac_regional_hierarchy_v1"
    rows = json_rows(run, operator, trace)
    available = [[True] * 8, [True, False, True, False, False, True, False, True]]
    for row, expected, norm in zip(rows, available, [0.0, 2.0], strict=True):
        assert set(row) == {
            "configuration_id",
            "operator_id",
            "program",
            "context_norm",
            "phases",
            "availability",
            "work",
        }
        assert row["availability"] == expected
        assert row["context_norm"] == norm and row["program"] == PROGRAM
        assert row["phases"] == expected_phases()
        assert row["work"] == dict(local_node_evaluations=20, summary_evaluations=10)
    expected = dict(local_node_evaluations=40, summary_evaluations=20)
    assert execution_api().verify_exported_traces(rows, descriptor) == expected
    assert run.verify_exported_traces(rows, operator) == expected


def test_work_bound_covers_the_actual_group_and_root_evaluations():
    run, operator, trace = hierarchy_trace()
    units = execution_api().regional_work_per_row(run.config, operator.config)
    assert units == dict(local_node_evaluations=20, summary_evaluations=10)
    assert trace["work"] == {key: 2 * count for key, count in units.items()}
    old = RegionalExecutionConfig(program="summary_at_end_v1")
    assert execution_api().regional_work_per_row(old, operator.config) == dict(
        local_node_evaluations=20, summary_evaluations=2
    )
    assert execution_api().regional_work_per_row(
        old, type(operator.config)(**(asdict(operator.config) | dict(summary=False)))
    ) == dict(local_node_evaluations=20, summary_evaluations=0)


@pytest.mark.parametrize(
    "fault",
    [
        "configuration_id",
        "operator_id",
        "missing_availability",
        "nonboolean_availability",
        "empty_group_marked_present",
        "present_root_marked_absent",
        "version_reset",
        "wrong_receivers",
        "phase_order",
        "legacy_summary_count",
        "program",
        "context_norm",
    ],
)
def test_schema_two_rejects_corrupt_structure_instead_of_accepting_a_legacy_row(fault):
    run, operator, trace = hierarchy_trace()
    descriptor = run.trace_descriptor(operator)
    rows = json_rows(run, operator, trace)
    api = execution_api()
    assert api.verify_exported_traces(rows, descriptor) == dict(
        local_node_evaluations=40, summary_evaluations=20
    )
    if fault in ("configuration_id", "operator_id"):
        rows[0][fault] = "0" * 64
    elif fault == "missing_availability":
        del rows[0]["availability"]
    elif fault == "nonboolean_availability":
        rows[0]["availability"][0] = 1
    elif fault == "empty_group_marked_present":
        rows[1]["availability"][6] = True
    elif fault == "present_root_marked_absent":
        rows[0]["availability"][7] = False
    elif fault == "version_reset":
        rows[0]["phases"][5]["read_version"] = 0
    elif fault == "wrong_receivers":
        rows[0]["phases"][0]["receivers"] = [5]
    elif fault == "phase_order":
        rows[0]["phases"][0], rows[0]["phases"][1] = (
            rows[0]["phases"][1],
            rows[0]["phases"][0],
        )
    elif fault == "legacy_summary_count":
        rows[0]["work"]["summary_evaluations"] = 2
    elif fault == "program":
        rows[0]["program"] = "summary_interleaved_v1"
    else:
        rows[0]["context_norm"] = float("inf")
    with pytest.raises(ValueError):
        api.verify_exported_traces(rows, descriptor)


@pytest.mark.parametrize("fault", ["version", "boolean_version", "execution", "dtype", "extra"])
def test_global_parser_rejects_descriptor_confusion_between_families(fault):
    run, operator, trace = hierarchy_trace()
    rows = json_rows(run, operator, trace)
    descriptor = json.loads(json.dumps(run.trace_descriptor(operator)))
    api = execution_api()
    assert api.verify_exported_traces(rows, descriptor)["summary_evaluations"] == 20
    if fault == "version":
        descriptor["schema_version"] = 1
    elif fault == "boolean_version":
        descriptor["schema_version"] = True
    elif fault == "execution":
        descriptor["execution"] = RegionalExecutionConfig().identity()
    elif fault == "dtype":
        descriptor["dtype"] = "torch.float16"
    else:
        descriptor["working_state"] = []
    with pytest.raises(ValueError):
        api.verify_exported_traces(rows, descriptor)


def test_logical_archive_limit_does_not_adopt_the_export_block_limit():
    run, operator, trace = hierarchy_trace()
    row = json_rows(run, operator, trace)[0]
    descriptor = run.trace_descriptor(operator)
    assert run.max_batch == 2
    api = execution_api()
    assert api.verify_exported_traces([row] * 8192, descriptor) == dict(
        local_node_evaluations=163840, summary_evaluations=81920
    )
    with pytest.raises(ValueError):
        api.verify_exported_traces([row] * 8193, descriptor)


def test_schema_one_remains_a_distinct_valid_archive_without_new_fields():
    api = execution_api()
    run, operator, trace = trace_for()
    descriptor = run.trace_descriptor(operator)
    rows = archive(run, operator, trace)
    before = copy.deepcopy(rows)
    assert descriptor["schema_version"] == 1
    assert all("availability" not in row for row in rows)
    assert api.verify_exported_traces(rows, descriptor) == dict(
        local_node_evaluations=40, summary_evaluations=2
    )
    assert rows == before
    descriptor["schema_version"] = 2
    with pytest.raises(ValueError):
        api.verify_exported_traces(rows, descriptor)


def test_fixed_archive_validation_does_not_authenticate_unstored_neural_context():
    run, operator, trace = hierarchy_trace()
    rows = json_rows(run, operator, trace)
    rows[0]["context_norm"] = 0.75
    assert execution_api().verify_exported_traces(rows, run.trace_descriptor(operator)) == dict(
        local_node_evaluations=40, summary_evaluations=20
    )


@pytest.mark.parametrize("fault", ["missing", "extra", "mixed", "boolean"])
def test_union_parser_requires_one_complete_registered_family(fault):
    api = execution_api()
    config = hierarchy_api().RegionalHierarchyConfig()
    raw = json.loads(json.dumps(asdict(config)))
    assert type(api.parse_regional_execution_config(raw)) is type(config)
    if fault == "missing":
        del raw["readout"]
    elif fault == "extra":
        raw["rounds"] = 2
    elif fault == "mixed":
        raw.update(selection="fixed", program="summary_at_end_v1", context_threshold=None)
    else:
        raw["feedback"] = 1
    with pytest.raises(ValueError):
        api.parse_regional_execution_config(raw)


def test_identity_restore_rejects_an_unregistered_or_incomplete_hierarchy():
    api = execution_api()
    config = hierarchy_api().RegionalHierarchyConfig()
    saved = json.loads(json.dumps(config.identity()))
    assert api.regional_execution_from_identity(saved).identity() == config.identity()
    for bad in (
        saved | dict(kind="unregistered_hierarchy"),
        {key: value for key, value in saved.items() if key != "groups"},
        saved | dict(extra=0),
    ):
        with pytest.raises(ValueError):
            api.regional_execution_from_identity(bad)
