"""Vínculo opcional del plan de generación con la cuota durable de propuestas."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from tests.proposals.test_mature_proposals import (
    LIMITS,
    REQUEST_A,
    REQUEST_B,
    REQUEST_C,
    api,
    build_review,
    directory_bytes,
    ledger_case,
)
from tests.proposals.test_mature_proposals import native_archive as native_archive
from tests.proposals.test_mature_proposals import no_optimizer_steps as no_optimizer_steps

PLAN_ID = "a" * 64
OTHER_PLAN_ID = "b" * 64
pytestmark = [pytest.mark.cuda, pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")]


def canonical_sha(value):
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode()).hexdigest()


def physical_json(value):
    return (json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n").encode()


def planned_case(source, output, *, generation_plan_id=PLAN_ID, resume=False):
    return api("ledger").ProposalLedger(
        output,
        registry=source["registry"],
        review=build_review(source),
        limits=dict(LIMITS),
        resume=resume,
        generation_plan_id=generation_plan_id,
    )


def legacy_state(identity, review, attempts):
    body = dict(
        schema_version=1,
        kind="finite_registered_proposals_v1",
        identity=identity,
        review=review,
        charged_attempts=len(attempts),
        attempts=attempts,
    )
    return dict(body, state_sha256=canonical_sha(body))


def test_explicit_generation_plan_id_binds_new_state(native_archive, tmp_path):
    _, review = ledger_case(native_archive, tmp_path / "legacy")

    ledger = api("ledger").ProposalLedger(
        tmp_path / "planned",
        registry=native_archive["registry"],
        review=review,
        limits=dict(LIMITS),
        generation_plan_id=PLAN_ID,
    )

    state = ledger.snapshot()
    assert state["schema_version"] == 2
    assert state["identity"]["generation_plan_id"] == PLAN_ID
    assert state["charged_attempts"] == 0
    assert state["attempts"] == []


@pytest.mark.parametrize("explicit_none", [False, True], ids=["omitted", "explicit-none"])
def test_legacy_bytes_ids_and_receipts_remain_exact(native_archive, tmp_path, explicit_none):
    output = tmp_path / "legacy-contract"
    if explicit_none:
        ledger = planned_case(native_archive, output, generation_plan_id=None)
        review = build_review(native_archive)
    else:
        ledger, review = ledger_case(native_archive, output)
    record = review.record()
    registry = native_archive["registry"]
    identity = dict(
        registry_id=registry.registry_id,
        baseline_id=registry.baseline_id,
        review_id=record["review_id"],
        parent_consumer_id=record["parent_consumer_id"],
        window_sha256=record["window_sha256"],
        events_sha256=record["events_sha256"],
        limits=dict(LIMITS),
    )
    state = legacy_state(identity, record, [])
    assert ledger.snapshot() == state
    assert (output / "state.json").read_bytes() == physical_json(state)

    configuration_id = native_archive["candidate"]
    proposal_id = canonical_sha(
        dict(
            identity=identity,
            ordinal=1,
            configuration_id=configuration_id,
            request_id=REQUEST_A,
        )
    )
    body = dict(
        schema_version=1,
        kind="registered_configuration_proposal_v1",
        proposal_id=proposal_id,
        ordinal=1,
        registry_id=registry.registry_id,
        review_id=record["review_id"],
        parent_consumer_id=record["parent_consumer_id"],
        window_sha256=record["window_sha256"],
        events_sha256=record["events_sha256"],
        configuration_id=configuration_id,
        request_id=REQUEST_A,
        status="reserved",
        reason=None,
        previous_receipt_sha256=None,
        selection_performed=False,
        deployment_performed=False,
    )
    expected = dict(sorted(body.items()), receipt_sha256=canonical_sha(body))
    assert ledger.reserve(configuration_id, request_id=REQUEST_A).record() == expected
    state = legacy_state(identity, record, [expected])
    assert ledger.snapshot() == state
    assert (output / "state.json").read_bytes() == physical_json(state)

    body.update(status="proposed", reason="registered_configuration")
    expected = dict(sorted(body.items()), receipt_sha256=canonical_sha(body))
    assert (
        ledger.finish(
            proposal_id, disposition="proposed", reason="registered_configuration"
        ).record()
        == expected
    )
    state = legacy_state(identity, record, [expected])
    assert (output / "state.json").read_bytes() == physical_json(state)
    files = directory_bytes(output)
    reopened = planned_case(native_archive, output, generation_plan_id=None, resume=True)
    assert reopened.snapshot() == state
    assert reopened.reserve(configuration_id, request_id=REQUEST_A).record() == expected
    assert directory_bytes(output) == files


@pytest.mark.parametrize(
    "value",
    [True, False, 0, 1, 1.5, b"a" * 64, [], {}, "", "g" * 64, "A" * 64, "a" * 63, "a" * 65],
)
def test_invalid_generation_plan_id_creates_no_output(native_archive, tmp_path, value):
    output = tmp_path / "invalid-plan"
    with pytest.raises(ValueError):
        planned_case(native_archive, output, generation_plan_id=value)
    assert not output.exists()


def test_plan_receipts_keep_idempotence_pending_and_charges_on_reopen(native_archive, tmp_path):
    output = tmp_path / "reopen"
    ledger = planned_case(native_archive, output)
    configuration_id = native_archive["candidate"]
    reserved = ledger.reserve(configuration_id, request_id=REQUEST_A).record()
    assert reserved["schema_version"] == 2
    assert reserved["generation_plan_id"] == PLAN_ID
    before = ledger.snapshot()
    assert before["charged_attempts"] == 1
    assert ledger.reserve(configuration_id, request_id=REQUEST_A).record() == reserved
    with pytest.raises(ValueError):
        ledger.reserve(configuration_id, request_id=REQUEST_B)
    with pytest.raises(ValueError):
        ledger.reserve(native_archive["registry"].baseline_id, request_id=REQUEST_A)
    assert ledger.snapshot() == before

    reopened = planned_case(native_archive, output, resume=True)
    assert reopened.snapshot() == before
    assert reopened.reserve(configuration_id, request_id=REQUEST_A).record() == reserved
    terminal = reopened.finish(
        reserved["proposal_id"], disposition="abandoned", reason="discarded"
    ).record()
    assert terminal["generation_plan_id"] == PLAN_ID and terminal["schema_version"] == 2
    assert (
        reopened.finish(
            reserved["proposal_id"], disposition="abandoned", reason="discarded"
        ).record()
        == terminal
    )
    with pytest.raises(ValueError):
        reopened.finish(reserved["proposal_id"], disposition="proposed", reason="changed")

    resumed = planned_case(native_archive, output, resume=True)
    assert resumed.reserve(configuration_id, request_id=REQUEST_A).record() == terminal
    second = resumed.reserve(configuration_id, request_id=REQUEST_B).record()
    assert second["ordinal"] == 2 and second["generation_plan_id"] == PLAN_ID
    assert second["previous_receipt_sha256"] == terminal["receipt_sha256"]
    resumed.finish(second["proposal_id"], disposition="rejected", reason="discarded")
    before = resumed.snapshot()
    with pytest.raises(ValueError):
        resumed.reserve(configuration_id, request_id=REQUEST_C)
    assert resumed.snapshot() == before and before["charged_attempts"] == 2
    assert all(item["generation_plan_id"] == PLAN_ID for item in before["attempts"])


@pytest.mark.parametrize(
    ("initial", "requested"),
    [(PLAN_ID, OTHER_PLAN_ID), (PLAN_ID, None), (None, PLAN_ID)],
    ids=["other-plan", "plan-to-legacy", "legacy-to-plan"],
)
def test_resume_cannot_rebind_plan_or_clean_unconfirmed_state(
    native_archive, tmp_path, initial, requested
):
    output = tmp_path / "wrong-plan"
    ledger = planned_case(native_archive, output, generation_plan_id=initial)
    reserved = ledger.reserve(native_archive["candidate"], request_id=REQUEST_A).record()
    before = ledger.snapshot()
    temporary = output / ".state.json.pending"
    temporary.write_bytes(b"unconfirmed")
    files = directory_bytes(output)

    with pytest.raises(ValueError):
        planned_case(native_archive, output, generation_plan_id=requested, resume=True)
    assert directory_bytes(output) == files
    assert ledger.snapshot() == before
    reopened = planned_case(native_archive, output, generation_plan_id=initial, resume=True)
    assert reopened.snapshot() == before
    assert reopened.reserve(native_archive["candidate"], request_id=REQUEST_A).record() == reserved
    assert before["charged_attempts"] == 1


@pytest.mark.parametrize(
    ("initial", "changed"),
    [(PLAN_ID, OTHER_PLAN_ID), (PLAN_ID, None), (PLAN_ID, True), (None, PLAN_ID)],
    ids=["other-plan", "remove-plan", "boolean-plan", "add-plan-to-legacy"],
)
def test_in_memory_plan_change_blocks_every_operation_without_effects(
    native_archive, tmp_path, initial, changed
):
    output = tmp_path / "changed-plan"
    ledger = planned_case(native_archive, output, generation_plan_id=initial)
    reserved = ledger.reserve(native_archive["candidate"], request_id=REQUEST_A).record()
    before, files = ledger.snapshot(), directory_bytes(output)
    ledger.generation_plan_id = changed

    with pytest.raises(ValueError):
        ledger.snapshot()
    with pytest.raises(ValueError):
        ledger.reserve(native_archive["candidate"], request_id=REQUEST_B)
    with pytest.raises(ValueError):
        ledger.finish(reserved["proposal_id"], disposition="proposed", reason="registered")
    assert directory_bytes(output) == files

    ledger.generation_plan_id = initial
    assert ledger.snapshot() == before and before["charged_attempts"] == 1


def test_same_request_under_different_plans_has_distinct_ids(native_archive, tmp_path):
    receipts = []
    for index, plan in enumerate((PLAN_ID, OTHER_PLAN_ID)):
        ledger = planned_case(native_archive, tmp_path / f"plan-{index}", generation_plan_id=plan)
        receipts.append(ledger.reserve(native_archive["candidate"], request_id=REQUEST_A).record())
    assert {receipt["generation_plan_id"] for receipt in receipts} == {PLAN_ID, OTHER_PLAN_ID}
    assert len({receipt["proposal_id"] for receipt in receipts}) == 2
    assert len({receipt["receipt_sha256"] for receipt in receipts}) == 2


@pytest.mark.parametrize("fault", ["wrong-receipt-plan", "missing-receipt-plan", "old-schema"])
def test_resigned_incoherent_plan_receipt_is_rejected_without_rewriting(
    native_archive, tmp_path, fault
):
    output = tmp_path / "corrupt-plan"
    ledger = planned_case(native_archive, output)
    ledger.reserve(native_archive["candidate"], request_id=REQUEST_A)
    state = ledger.snapshot()
    receipt = state["attempts"][0]
    if fault == "wrong-receipt-plan":
        receipt["generation_plan_id"] = OTHER_PLAN_ID
    elif fault == "missing-receipt-plan":
        receipt.pop("generation_plan_id")
    else:
        receipt["schema_version"] = 1
    receipt["receipt_sha256"] = canonical_sha(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    state["state_sha256"] = canonical_sha(
        {key: value for key, value in state.items() if key != "state_sha256"}
    )
    (output / "state.json").write_bytes(physical_json(state))
    files = directory_bytes(output)
    with pytest.raises(ValueError):
        ledger.snapshot()
    with pytest.raises(ValueError):
        planned_case(native_archive, output, resume=True)
    assert directory_bytes(output) == files


def test_interrupted_plan_finish_keeps_reserved_charge_until_recovery(
    native_archive, tmp_path, monkeypatch
):
    output = tmp_path / "interrupted"
    ledger = planned_case(native_archive, output)
    reserved = ledger.reserve(native_archive["candidate"], request_id=REQUEST_A).record()
    before, files = ledger.snapshot(), directory_bytes(output)
    replace_file = os.replace

    def interrupted(source, destination):
        if Path(destination) == output / "state.json":
            raise OSError("Interrupción de fixture antes de confirmar el resultado del plan")
        return replace_file(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", interrupted)
        with pytest.raises(OSError):
            ledger.finish(reserved["proposal_id"], disposition="proposed", reason="registered")
    assert directory_bytes(output) == files
    assert ledger.snapshot() == before and before["charged_attempts"] == 1

    reopened = planned_case(native_archive, output, resume=True)
    assert reopened.reserve(native_archive["candidate"], request_id=REQUEST_A).record() == reserved
    terminal = reopened.finish(
        reserved["proposal_id"], disposition="proposed", reason="registered"
    ).record()
    assert terminal["generation_plan_id"] == PLAN_ID
    assert terminal["ordinal"] == 1 and reopened.snapshot()["charged_attempts"] == 1
