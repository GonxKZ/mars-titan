"""Recibos operativos con el esquema de campaña y datos sintéticos reducidos."""

import copy
import hashlib
import json
import subprocess
import sys

import pytest

from mars_titan.simulation import campaign_receipts as reader


def digest(value):
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False))
    temporary.replace(path)


def case(stage, variant, *, completed=False):
    return dict(
        id=f"{stage}-{variant}-42",
        stage=stage,
        variant=variant,
        seed=42,
        transitions=0 if stage == "audit" else 128,
        confirmed_transitions=0 if stage == "audit" or not completed else 128,
        output=f"{stage}/{variant}-42",
        config_sha256="b" * 64,
        status="completed" if completed else "running",
        receipt_sha256="c" * 64 if completed else None,
    )


def publications(identity, state):
    report = dict(
        schema_version=1,
        kind="adaptive_campaign",
        status=state["status"],
        phase=state["phase"],
        updated_at=state["updated_at"],
        identity_sha256=digest(identity),
        completed_cases=sum(item["status"] == "completed" for item in state["cases"]),
        final_test_opened=False,
        parent_frozen=True,
        domain="synthetic",
        analysis_domain="technical",
        budget_complete=state["budget_complete"],
        audit_opened=state["audit_opened"],
        selection_frozen=state["freeze_sha256"] is not None,
    )
    registry = dict(
        schema_version=1,
        kind="adaptive_campaign",
        status=state["status"],
        planned_runs=len(state["cases"]),
        runs=[
            dict(
                path=item["output"],
                stage=item["stage"],
                variant=item["variant"],
                seed=item["seed"],
                status=item["status"],
                config_sha256=item["config_sha256"],
                planned_transitions=item["transitions"],
                transitions=item.get("confirmed_transitions", 0),
            )
            for item in state["cases"]
        ],
    )
    return report, registry


def publish(root, identity, state):
    report, registry = publications(identity, state)
    write(root / "identity.json", identity)
    write(root / "campaign.json", dict(payload=state, sha256=digest(state)))
    write(root / "run.json", report)
    write(root / "registry.json", registry)
    return root / "registry.json"


@pytest.fixture
def campaign(tmp_path):
    identity = dict(
        schema_version=1,
        kind="adaptive_campaign",
        final_test_opened=False,
        settings=dict(
            schema_version=1, variants=["ppo", "ppo_hmm"], seeds=[42], final_test_opened=False
        ),
        base_configuration=dict(final_test_opened=False),
    )
    state = dict(
        schema_version=1,
        identity_sha256=digest(identity),
        status="running",
        phase="pilot",
        cases=[case("pilot", "ppo", completed=True), case("pilot", "ppo_hmm")],
        choice=None,
        gate=None,
        freeze_sha256=None,
        audit_opened=False,
        budget_complete=True,
        active_case="pilot-ppo_hmm-42",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    publish(tmp_path, identity, state)
    return tmp_path, identity, state


def finish(identity, state):
    state.update(
        status="completed",
        phase="completed",
        active_case=None,
        choice={"transitions": 128},
        gate={"enabled": False},
        freeze_sha256="d" * 64,
        audit_opened=True,
    )
    state["cases"] = [
        case(stage, variant, completed=True)
        for stage in ("pilot", "main", "audit")
        for variant in identity["settings"]["variants"]
    ]


def test_real_registry_schema_normalizes_only_verified_progress(campaign):
    root, _, _ = campaign
    result = reader.read_adaptive_receipt(root / "registry.json")
    assert result["completed_runs"] == 1 and result["planned_runs"] == 2
    assert result["status"] == "running" and result["final_test_opened"] is False
    assert set(result) == {
        "kind",
        "status",
        "phase",
        "completed_runs",
        "planned_runs",
        "identity_sha256",
        "snapshot_sha256",
        "final_test_opened",
    }


def test_plan_expands_with_main_and_audit_without_inventing_completion(campaign):
    root, identity, state = campaign
    state["cases"][1] = case("pilot", "ppo_hmm", completed=True)
    state.update(phase="main", choice={"transitions": 128}, active_case=None)
    state["cases"].extend(case("main", variant) for variant in identity["settings"]["variants"])
    path = publish(root, identity, state)
    result = reader.read_adaptive_receipt(path)
    assert (result["completed_runs"], result["planned_runs"], result["status"]) == (2, 4, "running")
    finish(identity, state)
    publish(root, identity, state)
    result = reader.read_adaptive_receipt(path, expected=6)
    assert (result["completed_runs"], result["planned_runs"], result["status"]) == (
        6,
        6,
        "completed",
    )
    with pytest.raises(ValueError):
        reader.read_adaptive_receipt(path, expected=4)


@pytest.mark.parametrize("status", ["paused", "blocked", "failed"])
def test_non_success_states_are_preserved(campaign, status):
    root, identity, state = campaign
    state["status"] = status
    result = reader.read_adaptive_receipt(publish(root, identity, state))
    assert result["status"] == status
    assert result["completed_runs"] == 1


@pytest.mark.parametrize("target", ["identity", "settings", "base", "run"])
@pytest.mark.parametrize("value", [None, True, 0])
def test_reservation_requires_explicit_false_in_each_source(campaign, target, value):
    root, identity, state = campaign
    if target == "identity":
        identity["final_test_opened"] = value
    elif target == "settings":
        identity["settings"]["final_test_opened"] = value
    elif target == "base":
        identity["base_configuration"]["final_test_opened"] = value
    state["identity_sha256"] = digest(identity)
    publish(root, identity, state)
    if target == "run":
        report = json.loads((root / "run.json").read_text())
        report["final_test_opened"] = value
        write(root / "run.json", report)
    with pytest.raises(ValueError, match="reserva"):
        reader.read_adaptive_receipt(root / "registry.json")


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "unknown_status",
        "identity",
        "hash",
        "plan",
        "count",
        "future_registry",
        "missing_run",
        "missing_identity",
        "bad_path",
    ],
)
def test_corruption_is_not_a_transient_publication(campaign, change):
    root, identity, state = campaign
    report, registry = publications(identity, state)
    if change == "duplicate":
        state["cases"].append(copy.deepcopy(state["cases"][0]))
        publish(root, identity, state)
    elif change == "unknown_status":
        state["cases"][0]["status"] = "whatever"
        publish(root, identity, state)
    elif change == "identity":
        state["identity_sha256"] = "e" * 64
        publish(root, identity, state)
    elif change == "hash":
        write(root / "campaign.json", dict(payload=state, sha256="e" * 64))
    elif change == "plan":
        registry["planned_runs"] = 3
        write(root / "registry.json", registry)
    elif change == "count":
        report["completed_cases"] = 2
        write(root / "run.json", report)
    elif change == "future_registry":
        registry["runs"][1]["status"] = "completed"
        write(root / "registry.json", registry)
    elif change == "bad_path":
        registry["runs"][0]["path"] = "../outside"
        write(root / "registry.json", registry)
    else:
        (root / ("run.json" if change == "missing_run" else "identity.json")).unlink()
    with pytest.raises(ValueError):
        reader.read_adaptive_receipt(root / "registry.json")


@pytest.mark.parametrize(
    "change", ["pending", "phase", "audit", "freeze", "budget", "missing_main"]
)
def test_terminal_receipt_requires_every_declared_stage(campaign, change):
    root, identity, state = campaign
    finish(identity, state)
    if change == "pending":
        state["cases"][-1]["status"] = "pending"
    elif change == "phase":
        state["phase"] = "pilot"
    elif change == "audit":
        state["audit_opened"] = False
    elif change == "freeze":
        state["freeze_sha256"] = None
    elif change == "budget":
        state["budget_complete"] = False
    else:
        state["cases"] = [row for row in state["cases"] if row["stage"] != "main"]
    with pytest.raises(ValueError):
        reader.read_adaptive_receipt(publish(root, identity, state))


def test_journal_then_report_then_registry_is_a_transient_publication(campaign):
    root, identity, state = campaign
    state["cases"][1] = case("pilot", "ppo_hmm", completed=True)
    state["active_case"] = None
    state["updated_at"] = "2026-01-01T00:00:01+00:00"
    write(root / "campaign.json", dict(payload=state, sha256=digest(state)))
    with pytest.raises(BlockingIOError):
        reader.read_adaptive_receipt(root / "registry.json")
    report, registry = publications(identity, state)
    write(root / "run.json", report)
    with pytest.raises(BlockingIOError):
        reader.read_adaptive_receipt(root / "registry.json")
    write(root / "registry.json", registry)
    assert reader.read_adaptive_receipt(root / "registry.json")["completed_runs"] == 2


def test_registry_rows_are_unique_even_if_completed_count_matches(campaign):
    root, _, _ = campaign
    registry = json.loads((root / "registry.json").read_text())
    registry["runs"][1] = copy.deepcopy(registry["runs"][0])
    write(root / "registry.json", registry)
    with pytest.raises(ValueError):
        reader.read_adaptive_receipt(root / "registry.json")


def test_duplicate_json_fields_and_symlinks_are_rejected(campaign):
    root, _, _ = campaign
    path = root / "registry.json"
    original = path.read_bytes()
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError):
        reader.read_adaptive_receipt(path)
    path.unlink()
    other = root / "other.json"
    other.write_bytes(original)
    path.symlink_to(other)
    with pytest.raises(ValueError):
        reader.read_adaptive_receipt(path)


def test_helper_loads_from_an_external_file_with_only_the_standard_library(campaign):
    root, _, _ = campaign
    script = (
        "import importlib.util,sys\n"
        "spec=importlib.util.spec_from_file_location('operational_receipts',sys.argv[1])\n"
        "module=importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "assert module.read_adaptive_receipt(sys.argv[2])['completed_runs']==1\n"
        "assert 'mars_titan' not in sys.modules\n"
    )
    subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, reader.__file__, str(root / "registry.json")],
        check=True,
    )


def test_child_observes_dynamic_plan_and_waits_for_atomic_publication(campaign):
    from types import SimpleNamespace

    from mars_titan.posttraining.completion import run_child

    root, identity, state = campaign
    finish(identity, state)
    state["updated_at"] = "2026-01-01T00:00:01+00:00"
    report, registry = publications(identity, state)
    documents = {
        "campaign.json": dict(payload=state, sha256=digest(state)),
        "run.json": report,
        "registry.json": registry,
    }
    script = (
        "import json,sys,time\nfrom pathlib import Path\n"
        "root=Path(sys.argv[1]); time.sleep(.08)\n"
        "for name,value in json.loads(sys.argv[2]).items():\n"
        " path=root/name; temporary=path.with_suffix('.pending')\n"
        " temporary.write_text(json.dumps(value)); temporary.replace(path); time.sleep(.05)\n"
    )
    seen = []
    result = run_child(
        [sys.executable, "-c", script, str(root), json.dumps(documents)],
        root / "registry.json",
        None,
        SimpleNamespace(requested=False),
        seen.append,
        poll_seconds=0.01,
        grace_seconds=1,
        receipt_reader=reader.read_adaptive_receipt,
    )
    assert any(row["completed_runs"] == 1 and row["planned_runs"] == 2 for row in seen)
    assert result["completed_runs"] == result["planned_runs"] == 6
    assert seen[-1]["status"] == "completed"


@pytest.mark.parametrize(
    "status,exception",
    [("paused", InterruptedError), ("blocked", RuntimeError), ("failed", RuntimeError)],
)
@pytest.mark.parametrize("code", [0, 2])
def test_runner_preserves_failure_and_pause_states(campaign, status, exception, code):
    from types import SimpleNamespace

    from mars_titan.posttraining.completion import run_child

    root, identity, state = campaign
    state["status"] = status
    publish(root, identity, state)
    with pytest.raises(exception):
        run_child(
            [sys.executable, "-c", f"raise SystemExit({code})"],
            root / "registry.json",
            None,
            SimpleNamespace(requested=False),
            lambda _: None,
            poll_seconds=0.01,
            grace_seconds=1,
            receipt_reader=reader.read_adaptive_receipt,
        )


def test_reader_retries_when_journal_changes_between_file_reads(campaign, monkeypatch):
    root, identity, state = campaign
    original = reader._read
    changed = False

    def read_and_publish(path):
        nonlocal changed
        value = original(path)
        if path.name == "registry.json" and not changed:
            changed = True
            state["cases"][1] = case("pilot", "ppo_hmm", completed=True)
            state["active_case"] = None
            state["updated_at"] = "2026-01-01T00:00:01+00:00"
            publish(root, identity, state)
        return value

    monkeypatch.setattr(reader, "_read", read_and_publish)
    assert reader.read_adaptive_receipt(root / "registry.json")["completed_runs"] == 2


def test_terminal_exit_cannot_confirm_a_half_published_receipt(campaign):
    from types import SimpleNamespace

    from mars_titan.posttraining.completion import run_child

    root, identity, state = campaign
    finish(identity, state)
    state["updated_at"] = "2026-01-01T00:00:01+00:00"
    write(root / "campaign.json", dict(payload=state, sha256=digest(state)))
    with pytest.raises(BlockingIOError):
        run_child(
            [sys.executable, "-c", "pass"],
            root / "registry.json",
            None,
            SimpleNamespace(requested=False),
            lambda _: None,
            poll_seconds=0.1,
            grace_seconds=1,
            receipt_reader=reader.read_adaptive_receipt,
        )


def test_auxiliary_gate_expands_the_declared_audit_plan(campaign):
    root, identity, state = campaign
    finish(identity, state)
    state["gate"] = {"enabled": True}
    auxiliary = ("ppo_recent_aux", "ppo_replay_aux")
    state["cases"] = [
        case(stage, variant, completed=True)
        for stage, variants in (
            ("pilot", identity["settings"]["variants"]),
            ("main", identity["settings"]["variants"]),
            ("auxiliary", auxiliary),
            ("audit", (*identity["settings"]["variants"], *auxiliary)),
        )
        for variant in variants
    ]
    result = reader.read_adaptive_receipt(publish(root, identity, state))
    assert result["completed_runs"] == result["planned_runs"] == 10


def test_publication_does_not_assume_a_monotonic_wall_clock(campaign):
    root, _, state = campaign
    state["cases"][1] = case("pilot", "ppo_hmm", completed=True)
    state["active_case"] = None
    state["updated_at"] = "2025-12-31T23:59:59+00:00"
    write(root / "campaign.json", dict(payload=state, sha256=digest(state)))
    with pytest.raises(BlockingIOError):
        reader.read_adaptive_receipt(root / "registry.json")


def test_nonterminal_plan_can_be_partial_after_configuration_failure(campaign):
    root, identity, state = campaign
    state["cases"][1] = case("pilot", "ppo_hmm", completed=True)
    state["cases"].append(case("main", "ppo"))
    state.update(status="blocked", phase="main", choice={"transitions": 128}, active_case=None)
    result = reader.read_adaptive_receipt(publish(root, identity, state))
    assert (result["completed_runs"], result["planned_runs"], result["status"]) == (2, 3, "blocked")
    state.update(status="completed", phase="completed")
    with pytest.raises(ValueError):
        reader.read_adaptive_receipt(publish(root, identity, state))


@pytest.mark.parametrize("stage", ["pilot", "main"])
@pytest.mark.parametrize("confirmed", [0, 64])
def test_completed_budget_requires_confirmed_transitions(campaign, stage, confirmed):
    root, identity, state = campaign
    finish(identity, state)
    target = next(row for row in state["cases"] if row["stage"] == stage)
    target["confirmed_transitions"] = confirmed
    with pytest.raises(ValueError, match="transiciones|presupuesto"):
        reader.read_adaptive_receipt(publish(root, identity, state))


def early_case(campaign, stage="main"):
    root, identity, state = campaign
    finish(identity, state)
    identity["settings"].update(schema_version=2, evaluation_transitions=16)
    identity["base_configuration"].update(
        environments=4,
        selection=dict(
            early_stopping=True,
            min_transitions=16,
            patience=2,
            min_delta=0.0001,
            metric="ruin_count_then_mean_log_growth",
        ),
    )
    if stage == "auxiliary":
        state["gate"] = {"enabled": True}
        for variant in ("ppo_recent_aux", "ppo_replay_aux"):
            state["cases"].extend(
                (case("auxiliary", variant, completed=True), case("audit", variant, completed=True))
            )
    for row in state["cases"]:
        if row["stage"] != "audit":
            row["stopping_reason"] = "budget_exhausted"
    target = next(row for row in state["cases"] if row["stage"] == stage)
    target.update(
        confirmed_transitions=64,
        stopping_reason="early_stop",
        optimizer_steps=8,
        training_identity_sha256="e" * 64,
    )
    native = dict(
        schema_version=3,
        kind="native_ppo",
        status="completed",
        seed=42,
        agent_variant=target["variant"],
        device="cuda:0",
        diagnostic=False,
        final_test_opened=False,
        parent_frozen=True,
        identity_sha256="e" * 64,
        transitions=64,
        total_steps=128,
        stopping_reason="early_stop",
        optimizer_steps=8,
        selection=dict(identity["base_configuration"]["selection"], policy="greedy_argmax"),
        evaluation_cursors=[0, 16, 32, 48, 64],
        evaluations=5,
        stale_evaluations=3,
        best=dict(transitions=16),
    )
    path = root / target["output"] / "run.json"
    path.parent.mkdir(parents=True)
    write(path, native)
    target["receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    state["identity_sha256"] = digest(identity)
    publish(root, identity, state)
    return target, native, path


@pytest.mark.parametrize("stage", ["main", "auxiliary"])
def test_early_stop_needs_a_hashed_native_receipt_and_permitted_stage(campaign, stage):
    root, _, _ = campaign
    early_case(campaign, stage)
    result = reader.read_adaptive_receipt(root / "registry.json")
    assert result["status"] == "completed"
    assert result["completed_runs"] == result["planned_runs"] == (6 if stage == "main" else 10)


@pytest.mark.parametrize(
    "change",
    [
        "schema",
        "missing_reason",
        "missing_receipt",
        "hash",
        "patience",
        "transitions",
        "reservation",
        "optimizer",
        "pilot",
    ],
)
def test_unverified_early_stop_cannot_complete_the_campaign(campaign, change):
    root, identity, state = campaign
    target, native, path = early_case(campaign, "pilot" if change == "pilot" else "main")
    if change == "schema":
        identity["settings"]["schema_version"] = 1
    elif change == "missing_reason":
        target.pop("stopping_reason")
    elif change == "patience":
        native["stale_evaluations"] = 1
    elif change == "transitions":
        native["transitions"] = 48
    elif change == "reservation":
        native["final_test_opened"] = True
    elif change == "optimizer":
        target["optimizer_steps"] = native["optimizer_steps"] = 0
    elif change == "hash":
        native["unexpected"] = "changed"
    write(path, native)
    if change != "hash":
        target["receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if change == "missing_receipt":
        path.unlink()
    state["identity_sha256"] = digest(identity)
    publish(root, identity, state)
    with pytest.raises(ValueError):
        reader.read_adaptive_receipt(root / "registry.json")


def test_registry_transition_budget_must_remain_an_integer(campaign):
    root, _, _ = campaign
    path = root / "registry.json"
    value = json.loads(path.read_text())
    value["runs"][0]["planned_transitions"] = 128.0
    write(path, value)
    with pytest.raises(ValueError):
        reader.read_adaptive_receipt(path)
