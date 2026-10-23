"""Fuentes nativas completas para propuestas finitas, sin ajustar ni desplegar modelos."""

import copy
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from mars_titan.memory.financial_session import FinancialPhase
from mars_titan.memory.native_backend import load_native
from mars_titan.models.titans.episodic_readout import EpisodicReadoutConfig
from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismBudget
from mars_titan.proposals.contracts import ProposalRegistry, RegisteredConfiguration
from tests.proposals.test_mature_proposals import api
from tests.proposals.test_mature_proposals import no_optimizer_steps as no_optimizer_steps


def test_error_source_api_is_available_in_call():
    assert callable(getattr(api("error_source"), "prepare_error_review", None))
    assert callable(getattr(api("error_source"), "ErrorProposalEvidence", None))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _archive_hashes(output):
    paths = subprocess.run(
        ["rg", "--files", "--hidden", "--no-ignore", "-0", str(output)],
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    result = {}
    for name in paths:
        if name:
            path = Path(os.fsdecode(name))
            result[path.relative_to(output).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return result


@contextmanager
def _memory_helpers():
    with pytest.MonkeyPatch.context() as imports:
        imports.syspath_prepend(str(Path(__file__).resolve().parents[1] / "memory"))
        controls = importlib.import_module("test_financial_session_controls")
        with contextmanager(controls.no_target_estimation.__wrapped__)():
            yield (
                controls,
                importlib.import_module("test_financial_mechanisms").engines,
                importlib.import_module("test_native_mechanism_collection").session,
                importlib.import_module("test_financial_session").moment,
            )


@pytest.fixture(scope="module")
def native_error_sources(tmp_path_factory):
    """Retener sesiones reales y sus datos, con dos emisiones y seis generaciones."""
    assert torch.cuda.is_available(), "La fixture requiere CUDA, sin ruta CPU alternativa"
    torch.cuda.set_device("cuda:0")
    library = os.environ.get("MARS_TITAN_EPISODIC_NATIVE")
    assert library, "La fixture necesita MARS_TITAN_EPISODIC_NATIVE"
    native = load_native(library)
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with ExitStack() as resources:
            controls, engines, session, moment = resources.enter_context(_memory_helpers())
            inputs = controls.four_flow_source.__wrapped__(tmp_path_factory)
            assert inputs["label_origin"] == "manual_fixture_no_estimation"
            data = engines(inputs, "cuda:0")
            consumer = data[3]
            baseline = RegisteredConfiguration(
                predictor=consumer.predictor.config,
                readout=None,
                action=consumer.action,
                catalog=consumer.catalog,
                dtype="torch.float64",
            )
            candidate = replace(
                baseline, predictor=replace(baseline.predictor, seed=baseline.predictor.seed + 1)
            )
            registry = ProposalRegistry((baseline, candidate), baseline.configuration_id)
            flows = tuple(sorted(batch.flow_ids[0] for batch in inputs["batches"][126]))
            cases = {}

            def build(partition="validation", *, closed=True, collected=True, complete=True):
                key = partition, closed, collected, complete
                if key in cases:
                    return cases[key]
                output = tmp_path_factory.mktemp("error-source-native") / "source"
                phase = FinancialPhase(
                    partition, moment(125), moment(126), moment(130), moment(131)
                )
                run = resources.enter_context(
                    session(native, output, data, enabled=collected, phase=phase)
                )
                run.step(inputs["batches"][125], [], kind="warmup")
                first = run.step(inputs["batches"][126], [])
                run.step(
                    [],
                    [
                        native.Feedback(point.id, 0, moment(127), 0.125)
                        for point in first.predictions
                    ],
                    kind="settlement",
                    cutoff=moment(127),
                )
                second = run.step(inputs["batches"][128], [])
                run.step(
                    [],
                    [
                        native.Feedback(point.id, 0, moment(129), -0.25)
                        for point in second.predictions
                    ]
                    if complete
                    else [],
                    kind="settlement",
                    cutoff=moment(129),
                )
                assert len(first.predictions) == len(second.predictions) == 4
                if closed:
                    run.step([], [], kind="settlement", cutoff=moment(131), close_phase=True)
                head = copy.deepcopy(run.snapshot())
                events = [run.mechanism_event(index) for index in range(1, head["cursor"] + 1)]
                assert [event["kind"] for event in events[:5]] == [
                    "warmup",
                    "decision",
                    "settlement",
                    "decision",
                    "settlement",
                ]
                decisions = [event for event in events if event["kind"] == "decision"]
                window = (
                    run.mechanism_collections_at(
                        (2, 4), through_generation=head["cursor"], before=moment(132)
                    )
                    if collected
                    else None
                )
                result = dict(
                    session=run,
                    consumer=consumer,
                    data=data,
                    inputs=inputs,
                    output=output,
                    registry=registry,
                    baseline=baseline,
                    candidate=candidate,
                    flows=flows,
                    moment=moment,
                    head=head,
                    events=events,
                    decisions=decisions,
                    window=window,
                )
                cases[key] = result
                return result

            yield build
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)


def _plan(source, *, registry=None, **changes):
    moment = source["moment"]
    registry = source["registry"] if registry is None else registry
    values = dict(
        source_contract_id=source["session"].contract_id,
        parent_consumer_id=source["consumer"].model_id,
        source_partition="validation",
        declared_before=moment(124),
        parent_selected_before=moment(124),
        generation_start=moment(126),
        generation_end=moment(132),
        selection_start=moment(132),
        selection_end=moment(134),
        evaluation_start=moment(134),
        evaluation_end=moment(136),
        flow_ids=source["flows"],
        market_weights=(("US", 1.0),),
        min_sessions_per_market=2,
        min_excess_mae=0.0,
        candidate_ids=tuple(
            entry.configuration_id
            for entry in registry.entries
            if entry.configuration_id != registry.baseline_id
        ),
        control="mature_parent_vs_zero",
    )
    return api("error_rule").ErrorProposalPlan(**dict(values, **changes))


def _prepare(source, *, registry=None, plan=None):
    return api("error_source").prepare_error_review(
        source["session"],
        registry=source["registry"] if registry is None else registry,
        plan=_plan(source) if plan is None else plan,
    )


@pytest.mark.cuda
@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
class TestClosedErrorSource:
    def test_full_native_frame_keeps_both_decisions_and_exact_provenance(
        self, native_error_sources
    ):
        source = native_error_sources()
        plan = _plan(source)
        evidence = _prepare(source, plan=plan)
        record, review = evidence.record(), evidence.review.record()
        expected_ids = sorted(
            f"{flow}/{source['moment'](index)}" for flow in source["flows"] for index in (126, 128)
        )
        assert source["head"]["cursor"] == 6 and source["head"]["phase_closed"] is True
        assert evidence.plan_id == plan.plan_id
        assert evidence.registry_id == source["registry"].registry_id
        assert record["schema_version"] == 1
        assert record["kind"] == "closed_parent_error_evidence_v1"
        assert record["plan_id"] == plan.plan_id
        assert record["registry_id"] == source["registry"].registry_id
        assert record["review_id"] == review["review_id"]
        assert record["parent_consumer_id"] == source["consumer"].model_id
        assert record["source_contract_id"] == source["session"].contract_id
        assert record["phase"] == asdict(source["session"].phase)
        assert record["source_head"] == {
            "generation": 6,
            "record_sha256": source["head"]["record_sha256"],
        }
        assert record["all_events_sha256"] == _digest(source["events"])
        assert record["decision_generations"] == [2, 4]
        assert record["frame"] == dict(
            sample_ids=expected_ids,
            sha256=_digest(expected_ids),
            observed_flow_ids=list(source["flows"]),
            absent_flow_ids=[],
        )
        assert record["evidence_id"] == _digest(
            {name: value for name, value in record.items() if name != "evidence_id"}
        )
        assert "rows" not in record and "review" not in record
        assert sorted(row["sample_id"] for row in review["rows"]) == expected_ids
        assert len(review["rows"]) == review["mature_rows"] == 8
        assert review["collection_generations"] == [2, 4]
        assert review["through_generation"] == 6
        assert review["review_at"] == source["moment"](132)
        assert review["window_sha256"] == source["window"]["as_of"]["window_sha256"]
        assert review["events_sha256"] == _digest(source["decisions"])
        assert {row["target"] for row in review["rows"]} == {0.125, -0.25}
        assert {row["native_received_at"] for row in review["rows"]} == {
            source["moment"](127),
            source["moment"](129),
        }
        assert source["session"].snapshot() == source["head"]

    @pytest.mark.parametrize("partition", ("train", "validation"))
    def test_train_and_validation_sources_use_their_real_partition(
        self, native_error_sources, partition
    ):
        source = native_error_sources(partition)
        evidence = _prepare(source, plan=_plan(source, source_partition=partition))
        assert evidence.record()["phase"]["partition"] == partition
        assert evidence.review.record()["mature_rows"] == 8

    @pytest.mark.parametrize("partition", ("calibration", "evaluation"))
    def test_reserved_partitions_cannot_be_relabelled_as_generation(
        self, native_error_sources, partition
    ):
        source = native_error_sources(partition)
        with pytest.raises(ValueError):
            _prepare(source, plan=_plan(source, source_partition="validation"))
        assert source["session"].snapshot() == source["head"]

    def test_wrong_parent_contract_and_declared_partition_are_rejected(self, native_error_sources):
        source = native_error_sources()
        for changes in (
            dict(parent_consumer_id="a" * 64),
            dict(source_contract_id="b" * 64),
            dict(source_partition="train"),
        ):
            with pytest.raises(ValueError):
                _prepare(source, plan=_plan(source, **changes))
        assert source["session"].snapshot() == source["head"]

    @pytest.mark.parametrize("offset", (-1, 1))
    def test_generation_start_must_match_the_whole_phase(self, native_error_sources, offset):
        source = native_error_sources()
        with pytest.raises(ValueError):
            _prepare(source, plan=_plan(source, generation_start=source["moment"](126) + offset))

    def test_native_close_is_strictly_before_the_generation_cut(self, native_error_sources):
        source = native_error_sources()
        close_at = source["session"].phase.close_at
        with pytest.raises(ValueError):
            _prepare(source, plan=_plan(source, generation_end=close_at))
        evidence = _prepare(source, plan=_plan(source, generation_end=close_at + 1))
        assert evidence.review.record()["review_at"] == close_at + 1
        assert len(evidence.review.record()["rows"]) == 8

    def test_open_phase_is_rejected_even_when_all_emitted_labels_are_known(
        self, native_error_sources
    ):
        source = native_error_sources(closed=False)
        assert source["head"]["phase_closed"] is False
        with pytest.raises(ValueError):
            _prepare(source)
        assert source["session"].snapshot() == source["head"]

    def test_source_without_native_collection_is_rejected(self, native_error_sources):
        source = native_error_sources(collected=False)
        with pytest.raises(ValueError):
            _prepare(source)
        assert source["session"].snapshot() == source["head"]

    @pytest.mark.parametrize("fault", ("seed", "dtype", "action", "reader", "catalog"))
    def test_registered_baseline_must_match_the_actual_consumer(self, native_error_sources, fault):
        source = native_error_sources()
        baseline = source["baseline"]
        changes = {
            "seed": dict(predictor=replace(baseline.predictor, seed=baseline.predictor.seed + 10)),
            "dtype": dict(dtype="torch.float32"),
            "action": dict(action=MechanismAction(context="parent")),
            "reader": dict(
                readout=EpisodicReadoutConfig(
                    source["data"][2].fingerprint(), hidden_size=32, refinements=1
                )
            ),
            "catalog": dict(
                catalog=replace(baseline.catalog, budget=MechanismBudget(max_reader_steps=2))
            ),
        }[fault]
        wrong = replace(baseline, **changes)
        candidate = replace(
            wrong, predictor=replace(wrong.predictor, seed=wrong.predictor.seed + 1)
        )
        registry = ProposalRegistry((wrong, candidate), wrong.configuration_id)
        with pytest.raises(ValueError):
            _prepare(source, registry=registry, plan=_plan(source, registry=registry))
        assert source["session"].snapshot() == source["head"]

    def test_observed_flows_cannot_be_removed_from_the_declared_universe(
        self, native_error_sources
    ):
        source = native_error_sources()
        with pytest.raises(ValueError):
            _prepare(source, plan=_plan(source, flow_ids=source["flows"][:-1]))

    def test_declared_absent_flow_is_reported_without_inventing_an_emission(
        self, native_error_sources
    ):
        source = native_error_sources()
        evidence = _prepare(source, plan=_plan(source, flow_ids=(*source["flows"], "US/ZZZZ")))
        frame = evidence.record()["frame"]
        assert frame["observed_flow_ids"] == list(source["flows"])
        assert frame["absent_flow_ids"] == ["US/ZZZZ"]
        assert len(frame["sample_ids"]) == 8
        assert all(not sample.startswith("US/ZZZZ/") for sample in frame["sample_ids"])
        assert len(evidence.review.record()["rows"]) == 8

    def test_censored_second_decision_remains_in_the_complete_emission_frame(
        self, native_error_sources
    ):
        source = native_error_sources(complete=False)
        evidence = _prepare(source)
        review = evidence.review.record()
        assert len(review["rows"]) == 8 and review["mature_rows"] == 4
        assert [row["status"] for row in review["rows"]].count("censored") == 4
        assert evidence.record()["decision_generations"] == [2, 4]
        assert len(evidence.record()["frame"]["sample_ids"]) == 8
        for row in review["rows"]:
            if row["prediction_at"] == source["moment"](128):
                assert row["status"] == "censored"
                assert row["target"] is row["emitted_point_loss"] is None

    def test_record_budget_counts_warmup_and_settlements_as_well_as_decisions(
        self, native_error_sources
    ):
        source = native_error_sources()
        with pytest.raises(ValueError):
            _prepare(source, plan=_plan(source, max_source_records=5))
        evidence = _prepare(source, plan=_plan(source, max_source_records=6))
        assert evidence.record()["source_head"]["generation"] == 6
        assert evidence.record()["all_events_sha256"] == _digest(source["events"])

    def test_cumulative_byte_budget_includes_events_and_has_an_exact_boundary(
        self, native_error_sources
    ):
        source = native_error_sources()
        record = _prepare(source).record()
        window_bytes = source["window"]["as_of"]["source_bytes"]
        assert record["logical_source_bytes"] >= window_bytes + sum(
            len(_canonical(event)) for event in source["events"]
        )
        with pytest.raises(ValueError):
            _prepare(source, plan=_plan(source, max_source_bytes=window_bytes))
        boundary = record["logical_source_bytes"]
        accepted = _prepare(source, plan=_plan(source, max_source_bytes=boundary))
        assert accepted.record()["logical_source_bytes"] == boundary
        with pytest.raises(ValueError):
            _prepare(source, plan=_plan(source, max_source_bytes=boundary - 1))
        assert source["session"].snapshot() == source["head"]

    def test_evidence_metadata_has_no_mutable_alias_into_the_saved_frame(
        self, native_error_sources
    ):
        evidence = _prepare(native_error_sources())
        original = evidence.record()
        changed = evidence.record()
        changed["frame"]["sample_ids"].clear()
        changed["phase"]["partition"] = "evaluation"
        changed["source_head"]["generation"] = 0
        assert evidence.record() == original

    def test_preparation_preserves_archive_and_rng_without_constructing_or_running_a_model(
        self, native_error_sources
    ):
        from mars_titan.models.titans.financial import FinancialPredictor

        source = native_error_sources()
        plan = _plan(source)
        before_files = _archive_hashes(source["output"])
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        cpu_rng, cuda_rng = (
            torch.random.get_rng_state().clone(),
            torch.cuda.get_rng_state(0).clone(),
        )

        forbidden_codes = {
            FinancialPredictor.__init__.__code__,
            torch.nn.Module._call_impl.__code__,
        }

        def forbid_execution(frame, event, arg):
            if event == "call" and frame.f_code in forbidden_codes:
                pytest.fail("La preparación volvió a construir o ejecutar un modelo")

        previous = sys.getprofile()
        sys.setprofile(forbid_execution)
        try:
            evidence = _prepare(source, plan=plan)
        finally:
            sys.setprofile(previous)
        assert len(evidence.review.record()["rows"]) == 8
        assert source["session"].snapshot() == source["head"]
        assert torch.equal(torch.random.get_rng_state(), cpu_rng)
        assert torch.equal(torch.cuda.get_rng_state(0), cuda_rng)
        assert random.getstate() == python_rng
        for before, after in zip(numpy_rng, np.random.get_state(), strict=True):
            assert np.array_equal(before, after)
        after_files = _archive_hashes(source["output"])
        assert after_files == before_files

    @pytest.mark.parametrize(
        "fault",
        (
            "frame",
            "phase",
            "parent",
            "missing_frame",
            "frame_hash",
            "observed_flows",
            "absent_flows",
            "head_generation",
            "head_hash",
            "generations",
            "bytes",
            "version",
        ),
    )
    def test_evidence_rejects_incoherent_metadata_with_intact_review(
        self, native_error_sources, fault
    ):
        module = api("error_source")
        evidence = _prepare(native_error_sources())
        metadata = evidence.record()
        metadata.pop("evidence_id")
        if fault == "frame":
            metadata["frame"]["sample_ids"] = []
        elif fault == "phase":
            metadata["phase"]["partition"] = "evaluation"
        elif fault == "parent":
            metadata["parent_consumer_id"] = "f" * 64
        elif fault == "missing_frame":
            metadata.pop("frame")
        elif fault == "frame_hash":
            metadata["frame"]["sha256"] = "f" * 64
        elif fault == "observed_flows":
            metadata["frame"]["observed_flow_ids"] = []
        elif fault == "absent_flows":
            metadata["frame"]["absent_flow_ids"] = metadata["frame"]["observed_flow_ids"][:1]
        elif fault == "head_generation":
            metadata["source_head"]["generation"] = True
        elif fault == "head_hash":
            metadata["source_head"]["record_sha256"] = "f" * 64
        elif fault == "generations":
            metadata["decision_generations"].reverse()
        elif fault == "bytes":
            metadata["logical_source_bytes"] = 0
        elif fault == "version":
            metadata["schema_version"] = True
        with pytest.raises(ValueError):
            module.ErrorProposalEvidence(evidence.review, metadata)
