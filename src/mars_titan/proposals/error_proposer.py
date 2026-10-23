"""Propuestas registradas con cuota fija. No entrenar, seleccionar ni desplegar aquí."""

from mars_titan.proposals.error_rule import ErrorProposalPlan, parent_zero_statistics
from mars_titan.proposals.error_source import ErrorProposalEvidence
from mars_titan.proposals.ledger import ProposalLedger
from mars_titan.proposals.mature_review import fingerprint, require


def _requests(plan, registry_id, review_id):
    plan_id = plan.plan_id
    candidates = (
        ()
        if plan.control == "none"
        else plan.candidate_ids
        if plan.control == "enumerate"
        else plan.candidate_ids[:1]
    )
    return [
        (
            candidate,
            fingerprint(
                dict(
                    plan_id=plan_id,
                    registry_id=registry_id,
                    review_id=review_id,
                    configuration_id=candidate,
                    position=index,
                )
            ),
        )
        for index, candidate in enumerate(candidates)
    ]


def _bindings(ledger, plan, evidence):
    require(
        type(ledger) is ProposalLedger
        and type(plan) is ErrorProposalPlan
        and type(evidence) is ErrorProposalEvidence,
        "Faltan los contratos de generación",
    )
    plan.check_registry(ledger.registry)
    source = evidence.record()
    review = evidence.review.record()
    state = ledger.snapshot()
    require(
        source["plan_id"] == plan.plan_id == state["identity"].get("generation_plan_id")
        and source["registry_id"] == ledger.registry.registry_id
        and source["review_id"] == ledger.review.review_id == evidence.review.review_id,
        "La evidencia, el plan y la ejecución no corresponden al mismo contraste",
    )
    require(
        review["parent_consumer_id"] == plan.parent_consumer_id
        and review["contract_id"] == plan.source_contract_id
        and review["review_at"] == plan.generation_end
        and source["phase"]["partition"] == plan.source_partition
        and source["phase"]["decision_start"] == plan.generation_start
        and tuple(sorted(source["frame"]["observed_flow_ids"] + source["frame"]["absent_flow_ids"]))
        == plan.flow_ids
        and source["source_head"]["generation"] <= plan.max_source_records
        and source["logical_source_bytes"] <= plan.max_source_bytes,
        "La revisión o su marco no corresponde al plan antes de reservar",
    )
    requests = _requests(plan, source["registry_id"], source["review_id"])
    require(
        len(state["attempts"]) <= len(requests), "La ejecución contiene intentos ajenos al control"
    )
    for receipt, (candidate, request) in zip(state["attempts"], requests, strict=False):
        require(
            receipt["configuration_id"] == candidate and receipt["request_id"] == request,
            "La ejecución cambió el orden o la petición de sus candidatas",
        )
    return source, state, requests


def _disposition(statistics, control):
    if statistics["status"] != "complete":
        return "rejected", statistics["reason"]
    if control == "mature_parent_vs_zero" and not statistics["triggered"]:
        return "rejected", "parent_does_not_exceed_zero_margin"
    return "proposed", "registered_candidate_with_unknown_utility"


def propose_from_mature_errors(ledger, *, plan, evidence):
    """Reservar antes de calcular el disparador y reanudar la misma petición finita.

    Las candidatas conservan utilidad desconocida. Un error posterior a reserve
    deja el cargo y el estado pendiente. La cuota agotada no crea otra ejecución.
    """
    source, state, requests = _bindings(ledger, plan, evidence)
    receipts, statistics = [], None
    for position, (candidate, request) in enumerate(requests):
        if (
            position >= len(state["attempts"])
            and state["charged_attempts"] >= ledger.limits["max_attempts"]
        ):
            break
        receipt = ledger.reserve(candidate, request_id=request).record()
        if receipt["status"] == "abandoned":
            receipts.append(receipt)
            break
        if statistics is None:
            statistics = parent_zero_statistics(evidence.review, plan=plan)
        disposition, reason = _disposition(statistics, plan.control)
        receipt = ledger.finish(
            receipt["proposal_id"], disposition=disposition, reason=reason
        ).record()
        receipts.append(receipt)
        state = ledger.snapshot()
    attempted = [receipt["configuration_id"] for receipt in receipts]
    return dict(
        schema_version=1,
        kind="finite_error_proposals_v1",
        control=plan.control,
        plan_id=plan.plan_id,
        registry_id=source["registry_id"],
        review_id=source["review_id"],
        evidence_id=source["evidence_id"],
        complete=len(receipts) == len(requests)
        and all(receipt["status"] != "abandoned" for receipt in receipts),
        attempted=attempted,
        proposed=[
            receipt["configuration_id"] for receipt in receipts if receipt["status"] == "proposed"
        ],
        not_attempted=[candidate for candidate in plan.candidate_ids if candidate not in attempted],
        candidate_utilities=dict.fromkeys(plan.candidate_ids),
        statistics=statistics,
        frame=source["frame"],
        receipts=receipts,
        selection_performed=False,
        deployment_performed=False,
    )
