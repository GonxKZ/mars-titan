"""Recuperación de contrastes sin renovar cuotas ni confiar en cachés derivadas."""

import copy
import json
import shutil
import sys
from contextlib import contextmanager

import pytest
import torch

from mars_titan.data.storage import atomic_json
from mars_titan.proposals.mature_review import fingerprint
from tests.proposals.test_error_source import _archive_hashes
from tests.proposals.test_mature_proposals import no_optimizer_steps as no_optimizer_steps
from tests.proposals.test_trajectory_contrast import case as case
from tests.proposals.test_trajectory_contrast import completed as completed
from tests.proposals.test_trajectory_contrast import options, run

pytestmark = [pytest.mark.cuda, pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")]


def _copy_state(completed, output):
    shutil.copytree(completed[0], output)
    return json.loads((output / "state.json").read_text())


def _publish_state(output, state):
    body = {key: value for key, value in state.items() if key != "state_sha256"}
    atomic_json(
        output / "state.json",
        dict(body, state_sha256=fingerprint(body, 4 * 1024**2)),
    )


@contextmanager
def _no_forwards():
    forbidden_code = torch.nn.Module._call_impl.__code__

    def guard(frame, event, arg):
        if event == "call" and frame.f_code is forbidden_code:
            pytest.fail("La reconciliación volvió a ejecutar el predictor")

    previous = sys.getprofile()
    sys.setprofile(guard)
    try:
        yield
    finally:
        sys.setprofile(previous)


def test_resealed_effective_configuration_forgery_is_rejected(case, completed, tmp_path):
    output = tmp_path / "forged-effective-configuration"
    state = _copy_state(completed, output)
    receipt = state["arms"][-1]["receipt"]
    assert (
        receipt["effective_configuration_id"]
        == receipt["initialization"]["effective_configuration_id"]
    )
    receipt["effective_configuration_id"] = "f" * 64
    assert (
        receipt["effective_configuration_id"]
        != receipt["initialization"]["effective_configuration_id"]
    )
    state["result"]["arms"] = [copy.deepcopy(entry["receipt"]) for entry in state["arms"]]
    _publish_state(output, state)
    before_state = (output / "state.json").read_bytes()
    before_confirmation = (output / "confirmation.json").read_bytes()
    native_before = _archive_hashes(output / "arms")
    ledger_before = case["ledger"].snapshot()

    with pytest.raises(ValueError):
        run(case, output, resume=True)

    assert (output / "state.json").read_bytes() == before_state
    assert (output / "confirmation.json").read_bytes() == before_confirmation
    assert _archive_hashes(output / "arms") == native_before
    assert case["ledger"].snapshot() == ledger_before


def test_second_closed_native_attempt_recovers_missing_derivation_without_new_work(
    case, completed, tmp_path
):
    output = tmp_path / "second-attempt-closed-native"
    state = _copy_state(completed, output)
    entry = state["arms"][-1]
    configuration_id = entry["configuration_id"]
    original_receipt = copy.deepcopy(entry["receipt"])
    derived = output / original_receipt["trajectory_relative_path"]
    original_trajectory = json.loads(derived.read_text())
    derived.unlink()
    first_attempt = dict(
        status="interrupted",
        elapsed_seconds=None,
        error="Primer proceso interrumpido antes de confirmar su recibo",
    )
    entry.update(
        status="running",
        receipt=None,
        attempts=[
            first_attempt,
            dict(status="running", elapsed_seconds=None, error=None),
        ],
    )
    state.update(status="running", result=None)
    _publish_state(output, state)
    native_before = _archive_hashes(output / "arms")
    ledger_before = case["ledger"].snapshot()
    parent = options(case)["parent"]
    parent_identity = parent.identity()
    confirmation_before = (output / "confirmation.json").read_bytes()

    with _no_forwards():
        result = run(case, output, resume=True)

    recovered = json.loads((output / "state.json").read_text())
    finished = next(
        item for item in recovered["arms"] if item["configuration_id"] == configuration_id
    )
    assert result["status"] == recovered["status"] == "completed"
    assert finished["status"] == "completed"
    assert len(finished["attempts"]) == 2
    assert finished["attempts"][0] == first_attempt
    assert finished["attempts"][1]["status"] == "interrupted"
    assert finished["attempts"][1]["elapsed_seconds"] is None
    assert finished["attempts"][1]["error"]
    assert finished["receipt"]["cost"]["native_closed_reconciliation"] is True
    assert finished["receipt"]["cost"]["execution_seconds"] is None
    assert finished["receipt"]["trajectory_id"] == original_receipt["trajectory_id"]
    assert finished["receipt"]["statistics"] == original_receipt["statistics"]
    assert json.loads(derived.read_text()) == original_trajectory
    assert _archive_hashes(output / "arms") == native_before
    assert case["ledger"].snapshot() == ledger_before
    assert parent.identity() == parent_identity
    assert (output / "confirmation.json").read_bytes() == confirmation_before
    assert result["evaluation_executed"] is False


def test_bootstrap_recovers_only_a_bounded_regular_identity_temporary(case, completed, tmp_path):
    identity_bytes = (completed[0] / "identity.json").read_bytes()
    expected_identity = json.loads(identity_bytes)
    ledger_before = case["ledger"].snapshot()
    output = tmp_path / "interrupted-identity-publication"
    output.mkdir()
    temporary = output / ".identity.json.abcdefgh"
    temporary.write_bytes(identity_bytes)
    assert len(identity_bytes) < 4 * 1024**2

    result = run(case, output, resume=True)

    assert result["status"] == "completed"
    assert json.loads((output / "identity.json").read_text()) == expected_identity
    assert not temporary.exists()
    assert all(
        entry["status"] == "completed"
        for entry in json.loads((output / "state.json").read_text())["arms"]
    )
    assert case["ledger"].snapshot() == ledger_before

    protected = tmp_path / "external-identity.json"
    protected.write_bytes(identity_bytes)
    for fault in ("unrelated", "oversized", "symlink"):
        rejected = tmp_path / fault
        rejected.mkdir()
        candidate = rejected / (
            "unrelated.json" if fault == "unrelated" else ".identity.json.abcdefgh"
        )
        if fault == "symlink":
            candidate.symlink_to(protected)
        else:
            candidate.write_bytes(
                identity_bytes if fault == "unrelated" else b" " * (4 * 1024**2 + 1)
            )
        before = candidate.read_bytes()
        with pytest.raises(ValueError):
            run(case, rejected, resume=True)
        assert candidate.read_bytes() == before
        assert candidate.is_symlink() is (fault == "symlink")
        assert not (rejected / "state.json").exists()
        assert not (rejected / "identity.json").exists()
        assert not (rejected / "arms").exists()
        assert protected.read_bytes() == identity_bytes
    assert case["ledger"].snapshot() == ledger_before
