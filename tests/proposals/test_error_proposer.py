"""Controles finitos con metadatos sintéticos, sin selección ni despliegue."""

from dataclasses import replace

import pytest

from tests.proposals.test_mature_proposals import (
    LIMITS,
    api,
    build_review,
    moment,
)
from tests.proposals.test_mature_proposals import native_archive as native_archive
from tests.proposals.test_mature_proposals import no_optimizer_steps as no_optimizer_steps

pytestmark = [pytest.mark.cuda, pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")]


def setup_case(
    source, directory, *, control="mature_parent_vs_zero", pending=False, max_attempts=2
):
    from mars_titan.proposals.contracts import ProposalRegistry
    from mars_titan.proposals.error_rule import ErrorProposalPlan
    from mars_titan.proposals.error_source import ErrorProposalEvidence
    from mars_titan.proposals.ledger import ProposalLedger
    from mars_titan.proposals.mature_review import MatureReview, fingerprint

    original = source["registry"]
    baseline = original.resolve(original.baseline_id)
    second = original.resolve(source["candidate"])
    third = replace(second, predictor=replace(second.predictor, seed=second.predictor.seed + 1))
    registry = ProposalRegistry((baseline, second, third), baseline.configuration_id)
    body = build_review(source, "empty" if pending else "full").record()
    body.pop("review_id")
    if not pending:
        # Fixture adversarial explícita. No se presenta como emisión del archivo nativo.
        for row in body["rows"]:
            row["emitted_quantiles"] = [row["target"] + 2.0] * 5
            row["emitted_point_loss"] = abs(row["emitted_quantiles"][2] - row["target"])
            row["reference_point_loss"] = row["emitted_point_gain"] = None
            row["candidate_point_losses"] = [None] * 12
            row["candidate_point_gains"] = [None] * 12
    review = MatureReview(body)
    plan = ErrorProposalPlan(
        source_contract_id=body["contract_id"],
        parent_consumer_id=source["parent"],
        source_partition="validation",
        declared_before=moment(124),
        parent_selected_before=moment(124),
        generation_start=moment(126),
        generation_end=body["review_at"],
        selection_start=moment(132),
        selection_end=moment(133),
        evaluation_start=moment(134),
        evaluation_end=moment(135),
        flow_ids=tuple(sorted(row["flow_id"] for row in body["rows"])),
        market_weights=tuple(
            (market, 1.0)
            for market in sorted({row["flow_id"].split("/", 1)[0] for row in body["rows"]})
        ),
        min_sessions_per_market=1,
        min_excess_mae=0.0,
        candidate_ids=(second.configuration_id, third.configuration_id),
        control=control,
    )
    sample_ids = sorted(row["sample_id"] for row in body["rows"])
    close = body["native_records"][-1]["cutoff"]
    evidence = ErrorProposalEvidence(
        review,
        dict(
            schema_version=1,
            kind="closed_parent_error_evidence_v1",
            plan_id=plan.plan_id,
            registry_id=registry.registry_id,
            review_id=review.review_id,
            parent_consumer_id=source["parent"],
            source_contract_id=body["contract_id"],
            phase=dict(
                partition="validation",
                warmup_start=moment(125),
                decision_start=moment(126),
                decision_end=close,
                close_at=close,
            ),
            source_head=dict(
                generation=body["through_generation"],
                record_sha256=body["native_records"][-1]["sha256"],
            ),
            all_events_sha256=body["events_sha256"],
            decision_generations=body["collection_generations"],
            logical_source_bytes=sum(record["bytes"] for record in body["native_records"]),
            frame=dict(
                sample_ids=sample_ids,
                sha256=fingerprint(sample_ids),
                observed_flow_ids=sorted({row["flow_id"] for row in body["rows"]}),
                absent_flow_ids=[],
            ),
        ),
    )
    ledger = ProposalLedger(
        directory,
        registry=registry,
        review=review,
        limits=dict(LIMITS, max_candidates=3, max_attempts=max_attempts),
        generation_plan_id=plan.plan_id,
    )
    return ledger, plan, evidence


def run(ledger, plan, evidence):
    return api("error_proposer").propose_from_mature_errors(ledger, plan=plan, evidence=evidence)


@pytest.mark.parametrize(
    "control,count", [("none", 0), ("fixed", 1), ("enumerate", 2), ("mature_parent_vs_zero", 1)]
)
def test_controls_charge_only_their_declared_candidates(native_archive, tmp_path, control, count):
    ledger, plan, evidence = setup_case(native_archive, tmp_path / "ledger", control=control)
    before = ledger.snapshot()
    result = run(ledger, plan, evidence)
    assert result["complete"] is True
    assert result["attempted"] == list(plan.candidate_ids[:count])
    assert result["proposed"] == list(plan.candidate_ids[:count])
    assert ledger.snapshot()["charged_attempts"] == count
    assert result["candidate_utilities"] == {identifier: None for identifier in plan.candidate_ids}
    assert result["selection_performed"] is result["deployment_performed"] is False
    assert run(ledger, plan, evidence) == result
    if control == "none":
        assert ledger.snapshot() == before


@pytest.mark.parametrize("control", ["fixed", "enumerate", "mature_parent_vs_zero"])
def test_all_active_controls_reject_incomplete_support(native_archive, tmp_path, control):
    ledger, plan, evidence = setup_case(
        native_archive, tmp_path / "ledger", control=control, pending=True
    )
    result = run(ledger, plan, evidence)
    assert result["proposed"] == []
    assert result["statistics"]["reason"] == "incomplete_outcomes"
    assert result["statistics"]["parent_mae"] is None
    assert all(receipt["status"] == "rejected" for receipt in result["receipts"])


@pytest.mark.parametrize("fault", ["plan", "registry", "review"])
def test_mismatched_evidence_is_rejected_before_any_charge(native_archive, tmp_path, fault):
    from mars_titan.proposals.error_source import ErrorProposalEvidence

    ledger, plan, evidence = setup_case(native_archive, tmp_path / "ledger")
    metadata = evidence.record()
    metadata.pop("evidence_id")
    if fault == "review":
        other = build_review(native_archive)
        metadata["review_id"] = other.review_id
    else:
        other = evidence.review
        metadata[f"{fault}_id"] = "d" * 64
    forged = ErrorProposalEvidence(other, metadata)
    before = ledger.snapshot()
    with pytest.raises(ValueError):
        run(ledger, plan, forged)
    assert ledger.snapshot() == before


def test_enumeration_reports_exhaustion_without_renewing_budget(native_archive, tmp_path):
    ledger, plan, evidence = setup_case(
        native_archive, tmp_path / "ledger", control="enumerate", max_attempts=1
    )
    result = run(ledger, plan, evidence)
    assert result["complete"] is False
    assert result["attempted"] == [plan.candidate_ids[0]]
    assert result["not_attempted"] == [plan.candidate_ids[1]]
    assert ledger.snapshot()["charged_attempts"] == 1
    assert run(ledger, plan, evidence) == result


def test_failure_after_reservation_keeps_charge_and_recovers(native_archive, tmp_path, monkeypatch):
    module = api("error_proposer")
    ledger, plan, evidence = setup_case(native_archive, tmp_path / "ledger")
    original = module.parent_zero_statistics

    def fail(*args, **kwargs):
        raise ValueError("Fallo técnico después del cargo")

    monkeypatch.setattr(module, "parent_zero_statistics", fail)
    with pytest.raises(ValueError, match="después del cargo"):
        run(ledger, plan, evidence)
    state = ledger.snapshot()
    assert state["charged_attempts"] == 1
    assert state["attempts"][0]["status"] == "reserved"
    monkeypatch.setattr(module, "parent_zero_statistics", original)
    assert run(ledger, plan, evidence)["proposed"] == [plan.candidate_ids[0]]
    assert ledger.snapshot()["charged_attempts"] == 1


@pytest.mark.parametrize("fault", ("parent_consumer_id", "source_contract_id", "generation_end"))
def test_review_plan_contradiction_is_rejected_before_reserving(native_archive, tmp_path, fault):
    from mars_titan.proposals.error_source import ErrorProposalEvidence
    from mars_titan.proposals.ledger import ProposalLedger

    initial, plan, evidence = setup_case(native_archive, tmp_path / "original")
    value = plan.generation_end + 1 if fault == "generation_end" else "f" * 64
    plan = replace(plan, **{fault: value})
    metadata = evidence.record()
    metadata.pop("evidence_id")
    metadata["plan_id"] = plan.plan_id
    evidence = ErrorProposalEvidence(evidence.review, metadata)
    ledger = ProposalLedger(
        tmp_path / "contradictory",
        registry=initial.registry,
        review=evidence.review,
        limits=dict(initial.limits),
        generation_plan_id=plan.plan_id,
    )
    before = ledger.snapshot()
    with pytest.raises(ValueError):
        run(ledger, plan, evidence)
    assert ledger.snapshot() == before
