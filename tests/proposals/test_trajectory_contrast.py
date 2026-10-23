"""Contraste recuperable de trayectorias propias con parámetros emparejados."""

import copy
import importlib
import importlib.util
import json
import math
import os
import random
import shutil

import numpy as np
import pytest
import torch

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.memory.financial_session import FinancialSession
from mars_titan.memory.native_backend import load_native
from mars_titan.models.titans.financial_inputs import validated_cpu_batch
from mars_titan.proposals.ledger import ProposalLedger
from mars_titan.proposals.mature_review import fingerprint
from tests.proposals.test_error_source import _archive_hashes
from tests.proposals.test_mature_proposals import no_optimizer_steps as no_optimizer_steps
from tests.proposals.trajectory_fixture import (
    FLOWS,
    MANUAL_TARGETS,
    regional_limits,
    trajectory_case,
)


def api(name="trajectory_contrast"):
    path = f"mars_titan.proposals.{name}"
    assert importlib.util.find_spec(path) is not None, f"Falta el módulo {path}"
    return importlib.import_module(path)


def test_registered_trajectory_contrast_api_is_available_in_call():
    assert callable(getattr(api(), "run_registered_contrast", None))


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    assert torch.cuda.is_available(), "La integración requiere CUDA, sin alternativa CPU"
    torch.cuda.set_device("cuda:0")
    library = os.environ.get("MARS_TITAN_EPISODIC_NATIVE")
    assert library, "La integración necesita el enlace nativo existente"
    native = load_native(library)
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with trajectory_case(tmp_path_factory.mktemp("registered-trajectory"), native) as result:
            yield result
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)


def options(case):
    return dict(
        ledger=case["ledger"],
        plan=case["plan"],
        evidence=case["evidence"],
        parent=case["parent"],
        parent_codec=case["codec"],
        source=case["sources"]["s"],
        prefixes=case["prefixes"],
        retention=case["retention"],
        native=case["native"],
        admission="m1",
        block_rows=2,
        selection_min_gain=0.0,
        regional_work_limits=regional_limits(),
    )


def run(case, output, **changes):
    return api().run_registered_contrast(output, **(options(case) | changes))


def stored_trajectory(output, arm):
    path = output / arm["trajectory_relative_path"]
    assert path == output / "trajectories" / f"{arm['configuration_id']}.json"
    record = json.loads(path.read_text())
    assert record["trajectory_id"] == arm["trajectory_id"]
    return record


def session_mae(rows):
    groups = {}
    for row in rows:
        assert row["status"] == "label"
        groups.setdefault((row["flow_id"].split("/")[0], row["prediction_at"]), []).append(
            abs(row["emitted_quantiles"][2] - row["target"])
        )
    return math.fsum(math.fsum(errors) / len(errors) for errors in groups.values()) / len(groups)


def rng_state():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
        (torch.cuda.get_rng_state(0).clone()),
    )


def assert_same_rng(before):
    after = rng_state()
    assert before[0] == after[0] and before[1][0] == after[1][0]
    np.testing.assert_array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2]) and torch.equal(before[3], after[3])


def direct_trajectory(case, configuration_id, output):
    source = case["sources"]["s"]
    specification = source.specification()
    runtime = api("candidate_runtime").materialize_registered_candidate(
        case["registry"],
        configuration_id,
        parent=case["parent"],
        parent_codec=case["codec"],
        input_spec=specification,
        device="cuda:0",
    )
    events = {event.at: event for event in source.events()}
    moment, native = case["moment"], case["native"]
    with FinancialSession(
        output,
        native=native,
        consumer=runtime.consumer,
        codec=runtime.codec,
        prefixes=case["prefixes"],
        retention=case["retention"],
        phase=source.phase,
        world="direct_trajectory_fixture",
        fold="S",
        admission="m1",
        block_rows=2,
        mechanism_collection=None,
        regional_work_limits=regional_limits(),
    ) as session:
        previous = ()
        for index in (130, 131, 132, 133, 134):
            if index in (131, 132, 133):
                assert session.diagnostics()["retained"] == (4 if index == 133 else 0)
            event = events[moment(index)]
            batches = [validated_cpu_batch(raw, specification) for raw in event.inputs]
            labels = (
                [
                    native.Feedback(point.id, 0, moment(index), MANUAL_TARGETS[index - 1])
                    for point in previous
                ]
                if index >= 132
                else []
            )
            result = session.step(
                batches,
                labels,
                kind="warmup" if index == 130 else "decision" if batches else "settlement",
                cutoff=moment(index),
            )
            previous = result.predictions
        session.step([], [], kind="settlement", cutoff=moment(135), close_phase=True)
        assert session.snapshot()["phase_closed"] is True
        assert session.diagnostics()["pending"] == 0
        return (
            api("trajectory_reader")
            .read_mature_trajectory(session, before=case["plan"].selection_end)
            .record()
        )


@pytest.fixture(scope="module")
def completed(case, tmp_path_factory):
    output = tmp_path_factory.mktemp("completed-registered-contrast") / "contrast"
    before = (
        case["generation"].snapshot(),
        case["parent"].identity(),
        case["codec"].identity(),
        case["ledger"].snapshot(),
    )
    rng = rng_state()
    result = run(case, output)
    assert_same_rng(rng)
    assert (
        case["generation"].snapshot(),
        case["parent"].identity(),
        case["codec"].identity(),
        case["ledger"].snapshot(),
    ) == before
    return output, result


def copied_completed(case, completed, output):
    shutil.copytree(completed[0], output)
    assert run(case, output, resume=True) == completed[1]
    return json.loads((output / "state.json").read_text())


@pytest.mark.cuda
@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
class TestRegisteredTrajectoryContrast:
    def test_one_manual_corpus_has_disjoint_g_s_e_and_explicit_view_rebinding(self, case):
        sources, plan = case["sources"], case["plan"]
        assert {source.dataset.identity for source in sources.values()} == {
            case["dataset"].identity
        }
        specifications = {name: source.specification() for name, source in sources.items()}
        assert len({value.view_sha256 for value in specifications.values()}) == 3
        assert specifications["g"].identity() == case["parent"].predictor.config.inputs.identity()
        assert sources["g"].phase.close_at < plan.generation_end <= plan.selection_start
        assert sources["s"].phase.close_at < plan.selection_end <= plan.evaluation_start
        assert plan.evaluation_end - plan.generation_start < 31 * 86_400_000_000
        assert case["dataset"].manifest["configuration"]["backend"] == (
            "manual_trajectory_fixture_no_estimation"
        )
        assert case["evidence"].review.record()["mature_rows"] == 8
        assert case["proposals"]["proposed"] == list(plan.candidate_ids)
        assert all(value is None for value in case["proposals"]["candidate_utilities"].values())

    def test_generation_to_own_native_trajectories_and_confirmed_e_destination(
        self, case, completed
    ):
        output, result = completed
        registry, plan = case["registry"], case["plan"]
        identifiers = [registry.baseline_id, *plan.candidate_ids]
        assert result["schema_version"] == 1
        assert result["kind"] == "registered_trajectory_contrast_v1"
        assert result["status"] == "completed" and result["evaluation_executed"] is False
        assert result["initialization"] == "paired_parent_weights_fresh_state_v1"
        assert result["plan_id"] == plan.plan_id and result["registry_id"] == registry.registry_id
        assert result["baseline_id"] == registry.baseline_id
        source = case["sources"]["s"]
        assert result["source"] == dict(
            index_id=source.identity,
            dataset_id=source.dataset.identity,
            phase=copy.deepcopy(source.metadata["identity"]["phase"]),
        )
        assert [arm["configuration_id"] for arm in result["arms"]] == identifiers
        assert len({arm["consumer_id"] for arm in result["arms"]}) == 3
        assert all(
            arm["effective_configuration_id"] != arm["configuration_id"] for arm in result["arms"]
        )
        frame = None
        for arm in result["arms"]:
            record = stored_trajectory(output, arm)
            rows = record["rows"]
            assert len(rows) == len({row["sample_id"] for row in rows}) == 12
            assert {row["flow_id"] for row in rows} == set(FLOWS)
            assert record["through_generation"] == 6
            assert record["before"] == plan.selection_end
            current = sorted(
                (
                    row["sample_id"],
                    row["flow_id"],
                    row["prediction_at"],
                    row["input_available_at"],
                    row["input_sha256"],
                    row["matures_at"],
                    row["target"],
                )
                for row in rows
            )
            if frame is None:
                frame = current
            assert current == frame
            for index in (131, 132, 133):
                selected = [row for row in rows if row["prediction_at"] == case["moment"](index)]
                assert len(selected) == 4
                assert all(row["target"] == MANUAL_TARGETS[index] for row in selected)
            measured = result["comparison"]["by_configuration_id"][arm["configuration_id"]]
            assert measured["mae"] == pytest.approx(session_mae(rows), rel=1e-12, abs=1e-12)
            assert arm["cost"] and isinstance(arm["cost"], dict)
            assert (output / "arms" / arm["configuration_id"] / "latest.json").is_file()
        metrics = result["comparison"]["by_configuration_id"]
        baseline = metrics[registry.baseline_id]["mae"]
        expected = registry.baseline_id
        for identifier in plan.candidate_ids:
            mae = metrics[identifier]["mae"]
            assert metrics[identifier]["delta_mae"] == pytest.approx(mae - baseline, abs=1e-12)
            assert metrics[identifier]["gain"] == pytest.approx(baseline - mae, abs=1e-12)
            if baseline - mae > 0.0 and mae < metrics[expected]["mae"]:
                expected = identifier
        assert result["comparison"]["selected_configuration_id"] == expected
        confirmation = result["confirmation"]
        assert confirmation["expected_parent_consumer_id"] == case["parent"].model_id
        assert confirmation["selected_configuration_id"] == expected
        assert confirmation["evaluation_start"] == plan.evaluation_start
        assert confirmation["evaluation_end"] == plan.evaluation_end
        assert confirmation["evaluation_executed"] is False
        assert confirmation["canonical_default_changed"] is False
        assert json.loads((output / "confirmation.json").read_text()) == confirmation

    def test_each_arm_matches_a_direct_session_with_the_same_manual_inputs(
        self, case, completed, tmp_path
    ):
        output, result = completed
        for arm in result["arms"]:
            direct = direct_trajectory(
                case, arm["configuration_id"], tmp_path / arm["configuration_id"]
            )
            archived = stored_trajectory(output, arm)
            fields = (
                "sample_id",
                "flow_id",
                "prediction_at",
                "input_available_at",
                "input_sha256",
                "matures_at",
                "emitted_quantiles",
                "target",
                "emitted_point_loss",
                "status",
            )

            def project(record, names):
                return [
                    tuple(row[name] for name in names)
                    for row in sorted(record["rows"], key=lambda row: row["sample_id"])
                ]

            assert project(archived, fields) == project(direct, fields)

    def test_completed_resume_is_idempotent_and_preserves_the_confirmation(self, case, completed):
        output, result = completed
        before = (output / "confirmation.json").read_bytes()
        ledger = case["ledger"].snapshot()
        assert run(case, output, resume=True) == result
        assert (output / "confirmation.json").read_bytes() == before
        assert case["ledger"].snapshot() == ledger

    def test_selection_margin_cannot_change_on_resume(self, case, completed):
        output, _ = completed
        before = (output / "state.json").read_bytes(), (output / "confirmation.json").read_bytes()
        assert json.loads((output / "identity.json").read_text())["selection_min_gain"] == 0.0
        with pytest.raises(ValueError, match="protocolo|identidad"):
            run(case, output, resume=True, selection_min_gain=0.125)
        assert (
            (output / "state.json").read_bytes(),
            (output / "confirmation.json").read_bytes(),
        ) == before

    def test_regional_work_limits_cannot_change_on_resume(self, case, completed):
        output, _ = completed
        before = (output / "state.json").read_bytes(), (output / "confirmation.json").read_bytes()
        changed = regional_limits()
        changed["core_observations"] += 1
        with pytest.raises(ValueError, match="protocolo|identidad"):
            run(case, output, resume=True, regional_work_limits=changed)
        assert (
            (output / "state.json").read_bytes(),
            (output / "confirmation.json").read_bytes(),
        ) == before

    def test_coherently_shifted_cached_statistics_are_rejected_against_native_emissions(
        self, case, completed, tmp_path
    ):
        output = tmp_path / "forged-statistics"
        state = copied_completed(case, completed, output)
        original_confirmation = (output / "confirmation.json").read_bytes()
        native_before = _archive_hashes(output / "arms")
        for entry in state["arms"]:
            statistics = entry["receipt"]["statistics"]
            statistics["parent_mae"] += 1.0
            statistics["excess_mae"] = statistics["parent_mae"] - statistics["zero_mae"]
            statistics["triggered"] = statistics["excess_mae"] > case["plan"].min_excess_mae
            for cohort in statistics["cohorts"]:
                cohort["parent_mae"] += 1.0
                cohort["excess_mae"] = cohort["parent_mae"] - cohort["zero_mae"]
        state["result"]["arms"] = [copy.deepcopy(entry["receipt"]) for entry in state["arms"]]
        for metric in state["result"]["comparison"]["by_configuration_id"].values():
            metric["mae"] += 1.0
        atomic_json(output / "state.json", api()._seal(state))
        before = (output / "state.json").read_bytes()
        with pytest.raises(ValueError, match="recibo.*estadísticas verificadas"):
            run(case, output, resume=True)
        assert (output / "state.json").read_bytes() == before
        assert (output / "confirmation.json").read_bytes() == original_confirmation
        assert _archive_hashes(output / "arms") == native_before
        assert (completed[0] / "confirmation.json").read_bytes() == original_confirmation

    def test_resealed_derived_predictions_cannot_replace_unchanged_native_predictions(
        self, case, completed, tmp_path
    ):
        output = tmp_path / "forged-derived-trajectory"
        state = copied_completed(case, completed, output)
        native_before = _archive_hashes(output / "arms")
        original_confirmation = (completed[0] / "confirmation.json").read_bytes()
        trajectories = []
        for entry in state["arms"]:
            receipt = entry["receipt"]
            path = output / receipt["trajectory_relative_path"]
            body = json.loads(path.read_text())
            old_id = body.pop("trajectory_id")
            native_records = copy.deepcopy(body["native_records"])
            for row in body["rows"]:
                direction = 1.0 if row["emitted_quantiles"][2] >= row["target"] else -1.0
                row["emitted_quantiles"] = [value + direction for value in row["emitted_quantiles"]]
                row["emitted_point_loss"] = abs(row["emitted_quantiles"][2] - row["target"])
            trajectory = api("trajectory_reader").MatureTrajectory(body)
            record = trajectory.record()
            assert record["trajectory_id"] != old_id
            assert record["native_records"] == native_records
            atomic_json(path, record)
            receipt.update(
                trajectory_id=record["trajectory_id"],
                trajectory_file_sha256=sha256(path),
                statistics=api()._statistics(trajectory, case["plan"]),
            )
            trajectories.append(trajectory)
        result = state["result"]
        result["arms"] = [copy.deepcopy(entry["receipt"]) for entry in state["arms"]]
        result["comparison"] = api()._compare(
            trajectories, result["arms"], state["identity"], case["plan"]
        )
        confirmation = result["confirmation"]
        confirmation["trajectory_ids"] = [item.record()["trajectory_id"] for item in trajectories]
        confirmation["selected_configuration_id"] = result["comparison"][
            "selected_configuration_id"
        ]
        confirmation.pop("confirmation_id")
        confirmation["confirmation_id"] = fingerprint(confirmation)
        atomic_json(output / "confirmation.json", confirmation)
        atomic_json(output / "state.json", api()._seal(state))
        before = (output / "state.json").read_bytes(), (output / "confirmation.json").read_bytes()
        with pytest.raises(ValueError, match="trayectoria derivada.*emisiones nativas"):
            run(case, output, resume=True)
        assert (
            (output / "state.json").read_bytes(),
            (output / "confirmation.json").read_bytes(),
        ) == before
        assert _archive_hashes(output / "arms") == native_before
        assert (completed[0] / "confirmation.json").read_bytes() == original_confirmation

    def test_running_attempt_with_unknown_cost_resumes_using_the_second_execution_slot(
        self, case, completed, tmp_path
    ):
        output = tmp_path / "unknown-interrupted-cost"
        state = copied_completed(case, completed, output)
        assert state["identity"]["max_execution_attempts_per_arm"] == 2
        entry = state["arms"][-1]
        entry.update(status="running", receipt=None)
        entry["attempts"] = [dict(status="running", elapsed_seconds=None, error=None)]
        state.update(status="running", result=None)
        (output / "confirmation.json").unlink()
        atomic_json(output / "state.json", api()._seal(state))
        restored = run(case, output, resume=True)
        assert restored["comparison"] == completed[1]["comparison"]
        recovered = json.loads((output / "state.json").read_text())
        attempts = recovered["arms"][-1]["attempts"]
        assert len(attempts) == 2
        assert attempts[0]["status"] == "interrupted"
        assert attempts[0]["elapsed_seconds"] is None and attempts[0]["error"]
        assert attempts[1]["status"] == "completed" and attempts[1]["error"] is None
        assert attempts[1]["elapsed_seconds"] >= 0.0
        assert case["ledger"].snapshot()["charged_attempts"] == 2

    def test_reader_io_failure_keeps_native_progress_and_recovers_without_duplicate_rows(
        self, case, completed, tmp_path, monkeypatch
    ):
        source, output = case["sources"]["s"], tmp_path / "interrupted"
        original = source.events

        def interrupted(*, start_cursor=0):
            iterator = original(start_cursor=start_cursor)
            try:
                for position, event in enumerate(iterator):
                    if position == 2:
                        raise OSError("Interrupción de lectura de la fixture")
                    yield event
            finally:
                iterator.close()

        with monkeypatch.context() as patch:
            patch.setattr(source, "events", interrupted)
            with pytest.raises(OSError, match="Interrupción de lectura"):
                run(case, output)
        baseline_native = output / "arms" / case["registry"].baseline_id / "latest.json"
        assert baseline_native.is_file()
        assert not (output / "confirmation.json").exists()
        restored = run(case, output, resume=True)
        assert restored["status"] == "completed"
        assert restored["comparison"] == completed[1]["comparison"]
        for arm, reference in zip(restored["arms"], completed[1]["arms"], strict=True):
            rows = stored_trajectory(output, arm)["rows"]
            expected = stored_trajectory(completed[0], reference)["rows"]
            assert len(rows) == len({row["sample_id"] for row in rows}) == 12
            assert [
                (row["sample_id"], row["emitted_quantiles"], row["target"]) for row in rows
            ] == [(row["sample_id"], row["emitted_quantiles"], row["target"]) for row in expected]
        assert case["ledger"].snapshot()["charged_attempts"] == 2

    def test_pending_generation_attempt_is_rejected_before_creating_candidate_sessions(
        self, case, tmp_path
    ):
        ledger = ProposalLedger(
            tmp_path / "pending-generation",
            registry=case["registry"],
            review=case["evidence"].review,
            limits=case["limits"],
            generation_plan_id=case["plan"].plan_id,
        )
        ledger.reserve(case["plan"].candidate_ids[0], request_id="pending")
        output = tmp_path / "must-not-run"
        with pytest.raises(ValueError):
            run(case, output, ledger=ledger)
        assert ledger.snapshot()["charged_attempts"] == 1
        assert not (output / "arms").exists()

    def test_e_source_cannot_be_used_as_selection(self, case, tmp_path):
        output = tmp_path / "forbidden-e"
        with pytest.raises(ValueError):
            run(case, output, source=case["sources"]["e"])
        assert not (output / "arms").exists()
        assert case["generation"].snapshot() == case["generation_head"]
