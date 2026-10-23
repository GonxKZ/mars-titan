"""Plan cerrado y medias emparejadas de errores maduros por mercado y sesión."""

import hashlib
import importlib
import importlib.util
import json
import math
from dataclasses import replace
from fractions import Fraction

import pytest

DAY = 86_400_000_000
START = 1_420_070_400_000_000
FINAL_TEST = 1_704_067_200_000_000


def api():
    name = "mars_titan.proposals.error_rule"
    assert importlib.util.find_spec(name) is not None, "Falta el contrato de errores maduros"
    return importlib.import_module(name)


def test_error_rule_api_is_available():
    module = api()
    assert callable(getattr(module, "ErrorProposalPlan", None))
    assert callable(getattr(module, "parent_zero_statistics", None))


@pytest.fixture
def registry():
    from mars_titan.proposals.contracts import ProposalRegistry
    from tests.proposals.test_proposal_registry import api as contracts_api
    from tests.proposals.test_proposal_registry import configuration

    module = contracts_api()
    entries = (
        configuration(module),
        configuration(module, program="summary_interleaved_v1"),
        configuration(module, steps=2),
    )
    return ProposalRegistry(entries, entries[0].configuration_id)


def plan_arguments(registry, **changes):
    values = dict(
        source_contract_id="a" * 64,
        parent_consumer_id="b" * 64,
        source_partition="train",
        declared_before=START - DAY,
        parent_selected_before=START - 2 * DAY,
        generation_start=START,
        generation_end=START + 10 * DAY,
        selection_start=START + 10 * DAY,
        selection_end=START + 20 * DAY,
        evaluation_start=START + 20 * DAY,
        evaluation_end=START + 30 * DAY,
        flow_ids=("CN/A", "CN/B", "US/A", "US/B", "US/C"),
        market_weights=(("CN", 0.75), ("US", 0.25)),
        min_sessions_per_market=1,
        min_excess_mae=0.0,
        candidate_ids=tuple(entry.configuration_id for entry in registry.entries[1:]),
        control="mature_parent_vs_zero",
    )
    values.update(changes)
    return values


def plan_for(registry, **changes):
    return api().ErrorProposalPlan(**plan_arguments(registry, **changes))


def us_plan(registry, **changes):
    return plan_for(
        registry,
        **(
            dict(
                flow_ids=("US/A", "US/B", "US/C"),
                market_weights=(("US", 1.0),),
            )
            | changes
        ),
    )


def synthetic_review(samples, plan, *, conditional_losses=None, **changes):
    """Cuerpo sintético completo y válido, sin atribuirle un archivo nativo real."""
    from mars_titan.proposals.mature_review import MatureReview

    times = sorted({START + day * DAY for _, day, _, _, _ in samples})
    records = [
        dict(generation=index, cutoff=at, sha256=f"{index:064x}", bytes=1)
        for index, at in enumerate((*times, times[-1] + 2), 1)
    ]
    receipt = records[-1]
    rows = []
    for flow, day, point, target, status in samples:
        at = START + day * DAY
        loss = abs(point - target) if status == "label" else None
        losses = list(conditional_losses if conditional_losses is not None else [17.0] * 12)
        if status != "label":
            losses = [None] * 12
        reference = losses[0]
        received = status != "pending"
        rows.append(
            dict(
                flow_id=flow,
                sample_id=f"{flow}/{at}",
                prediction_at=at,
                input_available_at=at,
                input_sha256="e" * 64,
                matures_at=at + 1,
                snapshot_sha256=None,
                bank_entries=0,
                requested="off/k1/none",
                emitted="off/k1/none",
                emitted_quantiles=[float(point + offset) for offset in (-2, -1, 0, 1, 2)],
                status=status,
                target=float(target) if status == "label" else None,
                available_at=at + 1 if status in ("label", "excluded") else None,
                native_received_at=receipt["cutoff"] if received else None,
                receipt_generation=receipt["generation"] if received else None,
                native_record_sha256=receipt["sha256"] if received else None,
                emitted_point_loss=float(loss) if loss is not None else None,
                reference_point_loss=reference,
                emitted_point_gain=None if reference is None else reference - loss,
                candidate_point_losses=losses,
                candidate_point_gains=[
                    None if reference is None or value is None else reference - value
                    for value in losses
                ],
            )
        )
    body = dict(
        schema_version=1,
        kind="mature_native_review_v1",
        review_at=plan.generation_end,
        parent_consumer_id=plan.parent_consumer_id,
        window_sha256="c" * 64,
        events_sha256="d" * 64,
        contract_id=plan.source_contract_id,
        through_generation=len(records),
        collection_generations=list(range(1, len(times) + 1)),
        native_records=records,
        rows=rows,
        mature_rows=sum(row["status"] == "label" for row in rows),
        conditional_evidence="issued_trajectory_with_anchored_conditional_alternatives",
        selection_performed=False,
        deployment_performed=False,
    )
    body.update(changes)
    return MatureReview(body)


def unequal_us_samples():
    return [
        ("US/A", 1, 2.0, 0.0, "label"),
        ("US/B", 1, 2.0, 0.0, "label"),
        ("US/C", 1, 2.0, 0.0, "label"),
        ("US/A", 2, 10.0, 0.0, "label"),
    ]


@pytest.mark.parametrize("control", ("none", "fixed", "enumerate", "mature_parent_vs_zero"))
@pytest.mark.parametrize("partition", ("train", "validation"))
def test_plan_identifies_real_regional_and_j_recipes(registry, control, partition):
    plan = plan_for(registry, control=control, source_partition=partition)
    plan.check_registry(registry)
    assert plan.max_source_records == 256 and plan.max_source_bytes == 64 * 1024**2
    assert plan.candidate_ids == tuple(entry.configuration_id for entry in registry.entries[1:])
    assert registry.entries[1].predictor.regional_execution.program == "summary_interleaved_v1"
    assert registry.entries[2].readout.evidence_bridge.recurrence.steps == 2
    assert registry.entries[2].readout.refinements == 1
    identity = plan.identity()
    expected = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert plan.plan_id == expected
    identity["candidate_ids"] = []
    assert tuple(plan.identity()["candidate_ids"]) == plan.candidate_ids


@pytest.mark.parametrize(
    "changes",
    [
        {"source_contract_id": "not-a-hash"},
        {"parent_consumer_id": True},
        {"source_partition": "calibration"},
        {"source_partition": "evaluation"},
        {"source_partition": "test"},
        {"control": "learned"},
        {"declared_before": START},
        {"declared_before": True},
        {"parent_selected_before": START},
        {"parent_selected_before": True},
        {"generation_start": True},
        {"generation_start": float(START)},
        {"generation_end": START},
        {"selection_start": START + 9 * DAY},
        {"selection_end": START + 10 * DAY},
        {"evaluation_start": START + 19 * DAY},
        {"evaluation_end": START + 20 * DAY},
        {"evaluation_end": FINAL_TEST + 1},
        {"flow_ids": ["CN/A", "CN/B", "US/A", "US/B", "US/C"]},
        {"flow_ids": ("CN/A", "CN/A", "US/A")},
        {"flow_ids": ("US/A", "CN/A")},
        {"flow_ids": ("EU/A",)},
        {"flow_ids": ()},
        {"flow_ids": (False,)},
        {"market_weights": (("CN", 0.0), ("US", 1.0))},
        {"market_weights": (("CN", -1.0), ("US", 1.0))},
        {"market_weights": (("CN", math.nan), ("US", 1.0))},
        {"market_weights": (("CN", math.inf), ("US", 1.0))},
        {"market_weights": (("CN", True), ("US", 1.0))},
        {"market_weights": (("US", 1.0),)},
        {"market_weights": (("CN", 0.5), ("CN", 0.5), ("US", 1.0))},
        {"min_sessions_per_market": 0},
        {"min_sessions_per_market": True},
        {"min_sessions_per_market": 8193},
        {"min_excess_mae": -0.01},
        {"min_excess_mae": math.nan},
        {"min_excess_mae": math.inf},
        {"min_excess_mae": True},
        {"candidate_ids": ()},
        {"candidate_ids": ("f" * 64, "f" * 64)},
        {"candidate_ids": tuple(f"{index:064x}" for index in range(8))},
        {"max_source_records": 0},
        {"max_source_records": True},
        {"max_source_records": 257},
        {"max_source_bytes": 0},
        {"max_source_bytes": True},
        {"max_source_bytes": 64 * 1024**2 + 1},
    ],
)
def test_plan_rejects_invalid_closed_fields_before_any_source(registry, changes):
    with pytest.raises(ValueError):
        plan_for(registry, **changes)


def test_empty_candidate_list_is_also_rejected_for_none_control(registry):
    with pytest.raises(ValueError):
        plan_for(registry, control="none", candidate_ids=())


def test_plan_checks_registry_membership_and_excludes_baseline(registry):
    for candidate in (registry.baseline_id, "f" * 64):
        plan = plan_for(registry, candidate_ids=(candidate,))
        with pytest.raises(ValueError):
            plan.check_registry(registry)


def test_plan_accepts_exact_duration_and_final_exclusive_boundary(registry):
    plan_for(registry, evaluation_end=START + 366 * DAY).check_registry(registry)
    with pytest.raises(ValueError):
        plan_for(registry, evaluation_end=START + 366 * DAY + 1)
    base = FINAL_TEST - 30 * DAY
    shifts = {
        field: base + offset * DAY
        for field, offset in (
            ("declared_before", -1),
            ("parent_selected_before", -2),
            ("generation_start", 0),
            ("generation_end", 10),
            ("selection_start", 10),
            ("selection_end", 20),
            ("evaluation_start", 20),
            ("evaluation_end", 30),
        )
    }
    assert plan_for(registry, **shifts).evaluation_end == FINAL_TEST


def test_plan_rejects_generation_before_historical_start(registry):
    base = 946_684_800_000_000 - 1
    shifts = {
        field: base + offset * DAY
        for field, offset in (
            ("declared_before", -1),
            ("parent_selected_before", -2),
            ("generation_start", 0),
            ("generation_end", 10),
            ("selection_start", 10),
            ("selection_end", 20),
            ("evaluation_start", 20),
            ("evaluation_end", 30),
        )
    }
    with pytest.raises(ValueError):
        plan_for(registry, **shifts)


def test_seven_registered_candidates_preserve_order(registry):
    from mars_titan.proposals.contracts import ProposalRegistry
    from tests.proposals.test_proposal_registry import api as contracts_api
    from tests.proposals.test_proposal_registry import configuration

    extra = tuple(configuration(contracts_api(), seed=seed) for seed in range(43, 48))
    expanded = ProposalRegistry(registry.entries + extra, registry.baseline_id)
    identifiers = tuple(entry.configuration_id for entry in expanded.entries[1:])
    plan = plan_for(expanded, candidate_ids=identifiers[::-1])
    plan.check_registry(expanded)
    assert len(plan.candidate_ids) == 7 and plan.candidate_ids == identifiers[::-1]


def test_declared_flow_limit_is_exact(registry):
    flows = tuple(f"US/A{index:04d}" for index in range(8192))
    plan = plan_for(registry, flow_ids=flows, market_weights=(("US", 1.0),))
    assert len(plan.flow_ids) == 8192
    with pytest.raises(ValueError):
        plan_for(registry, flow_ids=flows + ("US/A8192",), market_weights=(("US", 1.0),))


@pytest.mark.parametrize("field", ("control", "min_excess_mae", "candidate_ids"))
def test_plan_id_changes_with_rule_margin_and_order(registry, field):
    plan = plan_for(registry)
    value = {
        "control": "fixed",
        "min_excess_mae": 0.5,
        "candidate_ids": plan.candidate_ids[::-1],
    }[field]
    assert replace(plan, **{field: value}).plan_id != plan.plan_id


@pytest.mark.parametrize("field", ("flow_ids", "market_weights", "candidate_ids"))
def test_json_equivalent_container_tampering_does_not_renew_plan_seal(registry, field):
    plan = plan_for(registry)
    object.__setattr__(plan, field, list(getattr(plan, field)))
    with pytest.raises(ValueError):
        plan.identity()


def test_changed_margin_is_rejected_by_identity_and_statistics(registry):
    plan = us_plan(registry)
    review = synthetic_review(unequal_us_samples(), plan)
    object.__setattr__(plan, "min_excess_mae", 0.5)
    with pytest.raises(ValueError):
        _ = plan.plan_id
    with pytest.raises(ValueError):
        api().parent_zero_statistics(review, plan=plan)


def test_statistics_weights_sessions_instead_of_rows(registry):
    plan = us_plan(registry, min_sessions_per_market=2)
    review = synthetic_review(unequal_us_samples(), plan)
    before = review.record(), plan.identity()
    result = api().parent_zero_statistics(review, plan=plan)
    assert result["status"] == result["reason"] == "complete"
    assert result["samples"] == 4 and result["state_counts"] == dict(
        label=4, pending=0, excluded=0, censored=0
    )
    assert result["sessions_by_market"] == {"US": 2}
    assert result["parent_mae"] == 6.0 and result["parent_mae"] != 4.0
    assert result["zero_mae"] == 0.0 and result["excess_mae"] == 6.0
    assert result["triggered"] is True and result.get("candidate_utility") is None
    assert [(c["market"], c["prediction_at"], c["samples"]) for c in result["cohorts"]] == [
        ("US", START + DAY, 3),
        ("US", START + 2 * DAY, 1),
    ]
    assert [c["parent_mae"] for c in result["cohorts"]] == [2.0, 10.0]
    assert (review.record(), plan.identity()) == before


def test_market_weights_are_independent_of_session_and_row_counts(registry):
    plan = plan_for(registry)
    values = [
        ("CN/A", 1, 5.0, 1.0, "label"),
        ("CN/B", 1, 11.0, 3.0, "label"),
        ("US/A", 1, 2.0, 1.0, "label"),
        ("US/A", 2, 6.0, 2.0, "label"),
        ("US/B", 2, 6.0, 2.0, "label"),
        ("US/C", 2, 6.0, 2.0, "label"),
        ("US/A", 3, 8.0, 2.0, "label"),
    ]
    result = api().parent_zero_statistics(synthetic_review(values, plan), plan=plan)
    parent = Fraction(3, 4) * Fraction(4 + 8, 2) + Fraction(1, 4) * Fraction(1 + 4 + 6, 3)
    zero = Fraction(3, 4) * Fraction(1 + 3, 2) + Fraction(1, 4) * Fraction(1 + 2 + 2, 3)
    assert result["parent_mae"] == pytest.approx(float(parent), rel=1e-14)
    assert result["zero_mae"] == pytest.approx(float(zero), rel=1e-14)
    assert result["excess_mae"] == pytest.approx(float(parent - zero), rel=1e-14)
    assert result["sessions_by_market"] == {"CN": 1, "US": 3}
    assert [(c["market"], c["prediction_at"]) for c in result["cohorts"]] == [
        ("CN", START + DAY),
        ("US", START + DAY),
        ("US", START + 2 * DAY),
        ("US", START + 3 * DAY),
    ]
    reversed_result = api().parent_zero_statistics(synthetic_review(values[::-1], plan), plan=plan)
    assert reversed_result == result


@pytest.mark.parametrize(
    "point,target,margin,excess,triggered",
    [
        (1.0, 1.0, 0.0, -1.0, False),
        (2.0, 1.0, 0.0, 0.0, False),
        (1.5, 0.5, 0.5, 0.5, False),
        (1.5, 0.5, 0.25, 0.5, True),
    ],
)
def test_negative_excess_and_exact_margin_keep_their_sign(
    registry, point, target, margin, excess, triggered
):
    plan = us_plan(registry, min_excess_mae=margin)
    review = synthetic_review([("US/A", 1, point, target, "label")], plan)
    result = api().parent_zero_statistics(review, plan=plan)
    assert result["excess_mae"] == excess and result["triggered"] is triggered


@pytest.mark.parametrize("status", ("pending", "excluded", "censored"))
def test_one_unknown_outcome_prevents_partial_population_scoring(registry, status):
    plan = us_plan(registry)
    values = [("US/A", 1, 9.0, 0.0, "label"), ("US/B", 2, 0.0, 0.0, status)]
    result = api().parent_zero_statistics(synthetic_review(values, plan), plan=plan)
    assert result["status"] == "incomplete" and result["reason"] == "incomplete_outcomes"
    assert result["samples"] == 2 and result["state_counts"]["label"] == 1
    assert result["state_counts"][status] == 1 and result["triggered"] is False
    assert result["parent_mae"] is result["zero_mae"] is result["excess_mae"] is None


def test_missing_market_is_not_removed_or_renormalized(registry):
    plan = plan_for(registry, min_sessions_per_market=2)
    review = synthetic_review([("US/A", 1, 9.0, 0.0, "label")], plan)
    result = api().parent_zero_statistics(review, plan=plan)
    assert result["status"] == "incomplete" and result["reason"] == "missing_market"
    assert result["parent_mae"] is result["zero_mae"] is result["excess_mae"] is None
    assert result["triggered"] is False
    pending = synthetic_review([("US/A", 1, 0.0, 0.0, "pending")], plan)
    assert api().parent_zero_statistics(pending, plan=plan)["reason"] == "incomplete_outcomes"


def test_session_minimum_applies_to_each_declared_market(registry):
    plan = plan_for(registry, min_sessions_per_market=2)
    values = [("CN/A", 1, 8.0, 0.0, "label"), ("US/A", 1, 8.0, 0.0, "label")]
    result = api().parent_zero_statistics(synthetic_review(values, plan), plan=plan)
    assert result["reason"] == "insufficient_sessions" and result["triggered"] is False
    assert result["parent_mae"] is result["zero_mae"] is result["excess_mae"] is None


def test_conditional_candidate_losses_do_not_change_parent_or_unknown_utility(registry):
    plan = us_plan(registry)
    known = synthetic_review(unequal_us_samples(), plan)
    unknown = synthetic_review(unequal_us_samples(), plan, conditional_losses=[None] * 12)
    first = api().parent_zero_statistics(known, plan=plan)
    assert api().parent_zero_statistics(unknown, plan=plan) == first
    assert first.get("candidate_utility") is None
    for control in ("none", "fixed", "enumerate"):
        assert api().parent_zero_statistics(known, plan=replace(plan, control=control)) == first


@pytest.mark.parametrize(
    "changes",
    [
        {"parent_consumer_id": "f" * 64},
        {"contract_id": "f" * 64},
        {"review_at": START + 11 * DAY},
    ],
)
def test_statistics_rejects_review_from_another_parent_contract_or_cut(registry, changes):
    plan = us_plan(registry)
    review = synthetic_review(unequal_us_samples(), plan, **changes)
    with pytest.raises(ValueError):
        api().parent_zero_statistics(review, plan=plan)


def test_statistics_rejects_outside_generation_or_undeclared_flow(registry):
    plan = us_plan(registry)
    for values in (
        [("US/A", -1, 2.0, 0.0, "label")],
        [("US/Z", 1, 2.0, 0.0, "label")],
    ):
        review = synthetic_review(values, plan)
        with pytest.raises(ValueError):
            api().parent_zero_statistics(review, plan=plan)


def test_representable_mean_does_not_overflow_only_from_session_count(registry):
    plan = us_plan(registry, min_sessions_per_market=2)
    review = synthetic_review(
        [("US/A", 1, 1e308, 0.0, "label"), ("US/A", 2, 1e308, 0.0, "label")], plan
    )
    result = api().parent_zero_statistics(review, plan=plan)
    assert math.isfinite(result["parent_mae"]) and result["parent_mae"] == 1e308
    assert result["zero_mae"] == 0.0 and result["excess_mae"] == 1e308


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf))
def test_nonfinite_mature_payload_is_rejected_instead_of_scored(registry, value):
    plan = us_plan(registry)
    with pytest.raises(ValueError):
        review = synthetic_review([("US/A", 1, value, 0.0, "label")], plan)
        api().parent_zero_statistics(review, plan=plan)


def test_plan_and_statistics_do_not_construct_a_predictor(registry, monkeypatch):
    from mars_titan.models.titans.financial import FinancialPredictor

    def forbidden(*args, **kwargs):
        pytest.fail("El plan y la estadística solo pueden revisar metadatos")

    monkeypatch.setattr(FinancialPredictor, "__init__", forbidden)
    plan = us_plan(registry)
    plan.check_registry(registry)
    result = api().parent_zero_statistics(synthetic_review(unequal_us_samples(), plan), plan=plan)
    assert result["status"] == "complete"
