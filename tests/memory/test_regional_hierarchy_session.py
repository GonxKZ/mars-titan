"""Cuotas y recuperación nativa de la jerarquía previa a MAC, sin aprendizaje."""

import importlib
import importlib.util
import json
from copy import deepcopy

import pytest
import torch
from test_financial_mechanisms import assert_same
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_mechanism_consumer import calibration
from test_native_episode_backend import native as native
from test_regional_program_session import (
    DEVICES,
    RESOURCES,
    attempt,
    limits,
    observed_calls,
    predictions,
    require_session_api,
    session,
    work_index,
)

from mars_titan.memory.regional_session_work import bounds
from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismCatalog
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.models.titans.regional_execution import RegionalExecutionConfig
from mars_titan.models.titans.regional_fusion import RegionalFusionConfig

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")


def artifact_state(output):
    directory = output / "artifacts"
    if not directory.exists():
        return None
    assert directory.is_dir()
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


def hierarchy_config(**options):
    name = "mars_titan.models.titans.regional_hierarchy"
    assert importlib.util.find_spec(name) is not None, "Falta la familia regional jerárquica"
    return importlib.import_module(name).RegionalHierarchyConfig(**options)


def hierarchy_engines(source, device="cpu", *, online=True, variant="mac_online", calibrated=False):
    program = hierarchy_config()
    model = (
        FinancialPredictor(
            FinancialConfig(
                source["spec"],
                variant=variant,
                hidden_size=32,
                seed=42,
                head=QUANTILE_HEAD,
                session_context="market_memory_v1" if online else None,
                regional_fusion=RegionalFusionConfig(
                    topology="self", routing="uniform", cycles=2, local_steps=2, summary=True
                ),
                regional_execution=program,
            ),
            device=device,
            dtype=torch.float64,
        )
        .eval()
        .requires_grad_(False)
    )
    if online:
        weights = model.state_dict()
        weights["session_context.output"] = torch.eye(32, device=device, dtype=torch.float64) * 0.2
        model.load_state_dict(weights)
    consumer = FrozenMechanismConsumer(
        model,
        catalog=MechanismCatalog(regional_execution=program),
        action=MechanismAction(
            "parent" if online else "off", "k1", "online" if calibrated else "none"
        ),
        calibration=calibration() if calibrated else None,
    )
    return None, source["spec"], source["codec"], consumer, source["prefixes"]


@pytest.mark.parametrize("program", ["summary_at_end_v1", "summary_interleaved_v1"])
@pytest.mark.parametrize("summary", [False, True])
def test_old_programs_keep_their_conservative_per_row_reservation(program, summary):
    api = importlib.import_module("mars_titan.models.titans.regional_execution")
    helper = getattr(api, "regional_work_per_row", None)
    assert callable(helper), "Falta la cota común por familia regional"
    assert helper(
        RegionalExecutionConfig(program=program), RegionalFusionConfig(summary=summary)
    ) == {
        "local_node_evaluations": 20,
        "summary_evaluations": 2 if summary else 0,
    }


@pytest.mark.parametrize(
    "variant,online,collection,expected",
    [
        ("mac_online", True, True, (12, 12, 12, 240, 120, 12, 0, 0, 1)),
        ("mac_disabled", False, True, (8, 0, 8, 160, 80, 8, 0, 0, 0)),
        ("transformer_direct", False, True, (8, 0, 8, 160, 80, 0, 0, 0, 0)),
        ("mac_online", False, False, (4, 0, 4, 80, 40, 4, 0, 0, 0)),
    ],
    ids=("pt4_collection_3n", "no_pt4_collection_2n", "direct_collection_2n", "emission_n"),
)
def test_hierarchy_reserves_n_2n_or_3n_with_exactly_nine_existing_resources(
    native, four_flow_source, tmp_path, variant, online, collection, expected
):
    require_session_api()
    data = hierarchy_engines(four_flow_source, online=online, variant=variant)
    output = tmp_path / "nine_resources"
    wanted = dict(zip(RESOURCES, expected, strict=True))
    batches = four_flow_source["batches"][126]
    with session(native, output, data, collection=collection, regional_work_limits=limits()) as run:
        rows = run._rows(batches)
        assert len(rows) == 4
        assert bounds(data[3], rows, collection=collection, kind="decision") == wanted
        previous = (output / "latest.json").read_bytes()

        def stop(boundary):
            if boundary == native.Boundary.before_predictions:
                assert attempt(output)["units"] == wanted
                assert attempt(output)["observed_units"] is None
                assert (output / "latest.json").read_bytes() == previous
                raise RuntimeError("Reserva jerárquica comprobada antes del núcleo")

        with (
            observed_calls(data[3]) as calls,
            pytest.raises(RuntimeError, match="antes del núcleo"),
        ):
            run.step(batches, [], fault=stop)
        assert all(count == 0 for count in calls.values()), calls


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("summary_limit,accepted", [(10, True), (9, False)])
def test_one_row_needs_ten_summaries_and_nine_is_rejected_before_mac(
    native, four_flow_source, tmp_path, device, summary_limit, accepted
):
    data = hierarchy_engines(four_flow_source, device, online=False)
    bounded = limits()
    bounded["summary_evaluations"] = summary_limit
    output = tmp_path / "edge"
    single_flow_batches = four_flow_source["batches"][126][:1]
    assert sum(len(batch.flow_ids) for batch in single_flow_batches) == 1
    with session(native, output, data, regional_work_limits=bounded) as run:
        assert len(run._rows(single_flow_batches)) == 1
        before_index = (output / "regional-work.json").read_bytes()
        before_head = (output / "latest.json").read_bytes()
        before_artifacts = artifact_state(output)
        boundaries = []
        with observed_calls(data[3]) as calls:
            if accepted:
                issued = run.step(single_flow_batches, [], fault=boundaries.append)
                assert len(issued.predictions) == 1
                assert calls["mac"] == 1
                assert attempt(output)["status"] == "confirmed"
                assert attempt(output)["units"]["summary_evaluations"] == 10
                assert attempt(output)["units"]["local_node_evaluations"] == 20
            else:
                with pytest.raises(ValueError):
                    run.step(single_flow_batches, [], fault=boundaries.append)
                assert all(count == 0 for count in calls.values()), calls
                assert not boundaries
                assert (output / "regional-work.json").read_bytes() == before_index
                assert (output / "latest.json").read_bytes() == before_head
                assert artifact_state(output) == before_artifacts


@pytest.mark.parametrize("block_rows", [1, 4])
def test_collection_reserves_all_120_summaries_before_the_first_physical_block(
    native, four_flow_source, tmp_path, block_rows
):
    data = hierarchy_engines(four_flow_source)
    bounded = limits()
    bounded["summary_evaluations"] = 119
    output = tmp_path / "whole_cohort"
    with session(
        native, output, data, block_rows=block_rows, collection=True, regional_work_limits=bounded
    ) as run:
        before = (output / "regional-work.json").read_bytes()
        head = (output / "latest.json").read_bytes()
        files = artifact_state(output)
        boundaries = []
        with observed_calls(data[3]) as calls, pytest.raises(ValueError):
            run.step(four_flow_source["batches"][126], [], fault=boundaries.append)
        assert all(count == 0 for count in calls.values()), calls
        assert not boundaries
        assert (output / "regional-work.json").read_bytes() == before
        assert (output / "latest.json").read_bytes() == head
        assert artifact_state(output) == files


@pytest.mark.parametrize("device", DEVICES)
def test_reopening_an_abandoned_hierarchy_attempt_cannot_refund_its_ten_summaries_per_row(
    native, four_flow_source, tmp_path, device
):
    data = hierarchy_engines(four_flow_source, device, online=False)
    bounded = limits()
    bounded["summary_evaluations"] = 40
    options = dict(regional_work_limits=bounded)
    output = tmp_path / "abandoned"
    with session(native, output, data, **options) as run:
        head = (output / "latest.json").read_bytes()

        def stop(boundary):
            if boundary == native.Boundary.before_predictions:
                assert attempt(output)["units"]["summary_evaluations"] == 40
                raise RuntimeError("Reserva jerárquica consumida sin predicción")

        with pytest.raises(RuntimeError, match="sin predicción"):
            run.step(four_flow_source["batches"][126], [], fault=stop)
        reservation_id = attempt(output)["reservation_id"]
    with session(native, output, data, resume=True, **options) as restored:
        assert attempt(output)["reservation_id"] == reservation_id
        assert attempt(output)["status"] == "abandoned"
        assert work_index(output)["cohorts"][-1]["charged_units"]["summary_evaluations"] == 40
        before = (output / "regional-work.json").read_bytes()
        files = artifact_state(output)
        boundaries = []
        with observed_calls(data[3]) as calls, pytest.raises(ValueError):
            restored.step(four_flow_source["batches"][126], [], fault=boundaries.append)
        assert all(count == 0 for count in calls.values()), calls
        assert not boundaries
        assert (output / "regional-work.json").read_bytes() == before
        assert (output / "latest.json").read_bytes() == head
        assert artifact_state(output) == files


@pytest.mark.parametrize("device", DEVICES)
def test_committed_hierarchy_recovers_full_state_without_replaying_mac_pt4_or_feedback(
    native, four_flow_source, tmp_path, device
):
    data = hierarchy_engines(four_flow_source, device, calibrated=True)
    options = dict(collection=True, regional_work_limits=limits())
    with session(native, tmp_path / "reference", data, **options) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        assert attempt(tmp_path / "reference")["units"]["summary_evaluations"] == 40
        first = run.step(four_flow_source["batches"][126], [])
        labels = [native.Feedback(row.id, 0, moment(127), 0.125) for row in first.predictions]
        with observed_calls(data[3]) as issued_calls:
            second = run.step(four_flow_source["batches"][127], labels)
        assert issued_calls["mac"] == 6
        assert issued_calls["predictor_prepare"] == 6
        assert issued_calls["consumer_prepare"] == 2
        assert issued_calls["context_commit"] == 1
        expected, expected_fast = run.snapshot(), run.fast_state()
        expected_state, expected_event = run.mechanism_state(), run.mechanism_event(3)
        later_labels = [
            native.Feedback(row.id, 0, moment(128), 0.125) for row in second.predictions
        ]
        later = run.step(four_flow_source["batches"][128], later_labels)
        following, following_state = run.snapshot(), run.mechanism_state()
    output = tmp_path / "committed"
    with session(native, output, data, **options) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        run.step(four_flow_source["batches"][126], [])

        def stop(boundary):
            if boundary == native.Boundary.committed:
                assert attempt(output)["status"] == "reserved"
                assert json.loads((output / "latest.json").read_text())["generation"] == 3
                raise RuntimeError("Corte posterior a la confirmación nativa")

        with pytest.raises(RuntimeError, match="confirmación nativa"):
            run.step(four_flow_source["batches"][127], labels, fault=stop)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state("cuda:0").clone() if device == "cuda:0" else None
    with observed_calls(data[3]) as calls:
        restored = session(native, output, data, resume=True, **options)
    with restored:
        assert all(count == 0 for count in calls.values()), calls
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        if cuda_rng is not None:
            assert torch.equal(torch.cuda.get_rng_state("cuda:0"), cuda_rng)
        assert restored.snapshot() == expected
        assert_same(restored.fast_state(), expected_fast)
        assert_same(restored.mechanism_state(), expected_state)
        assert restored.mechanism_event(3) == expected_event
        current = attempt(output)
        assert current["status"] == "confirmed" and current["ordinal"] == 1
        assert current["units"]["summary_evaluations"] == 120
        assert len(work_index(output)["cohorts"][-1]["attempts"]) == 1
        actual = restored.step(four_flow_source["batches"][128], later_labels)
        assert predictions(actual) == predictions(later)
        assert restored.snapshot() == following
        assert_same(restored.mechanism_state(), following_state)
        assert restored.snapshot()["applied"] == 8


def test_compacted_hierarchy_receipts_and_actual_availability_survive_native_history(
    native, four_flow_source, tmp_path
):
    data = hierarchy_engines(four_flow_source)
    output = tmp_path / "history"
    options = dict(regional_work_limits=limits())
    with session(native, output, data, **options) as run:
        for index in (126, 127, 128):
            run.step(four_flow_source["batches"][index], [])
        before_fast = run.fast_state()
        run.step([], [], kind="settlement", cutoff=moment(130))
        assert attempt(output)["units"] == dict.fromkeys(RESOURCES, 0)
        assert_same(run.fast_state(), before_fast)
        history = deepcopy(run.mechanism_event(1))
        link = history["payload"]["regional_work"]
        assert link["receipts"][0]["units"]["summary_evaluations"] == 40
        assert link["cohort"]["charged_units"]["summary_evaluations"] == 40
        assert not (output / "checkpoint-1.json").exists()
        assert link["reservation_id"] not in {
            row["reservation_id"]
            for cohort in work_index(output)["cohorts"]
            for row in cohort["attempts"]
        }
        for row in history["payload"]["rows"]:
            trace = row["regional_execution"]
            assert trace["availability"] == [True, False, True, False, False, True, False, True]
            assert trace["work"] == {"local_node_evaluations": 20, "summary_evaluations": 10}
            assert [phase["read_version"] for phase in trace["phases"]] == list(range(10))
            assert [phase["write_version"] for phase in trace["phases"]] == list(range(1, 11))
        snapshot = run.snapshot()
    with session(native, output, data, resume=True, **options) as restored:
        assert restored.snapshot() == snapshot
        assert restored.mechanism_event(1) == history
