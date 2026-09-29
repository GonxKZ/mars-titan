"""Admitir progreso nativo sin publicar trazas, recuerdos ni métricas reservadas."""

import json

import pytest

from mars_titan.observatory.collector import Collector

VARIANTS = (
    "ppo",
    "double_dqn",
    "ppo_window",
    "ppo_gru",
    "ppo_episodic",
    "ppo_hmm",
    "ppo_episodic_hmm",
    "ppo_recent_aux",
    "ppo_replay_aux",
)


def native_report(model):
    return dict(
        schema_version=2,
        kind="native_ppo",
        activity="rl",
        model=model,
        backend="native_libtorch",
        status="completed",
        domain="synthetic",
        analysis_domain="technical",
        partition="train",
        seed=42,
        global_step=32,
        total_steps=32,
        parent_frozen=True,
        final_test_opened=False,
        macro_coverage=dict(simulated_concepts=3, catalog_concepts=140),
        best=dict(mean_log_growth=987654321, validation_metrics=[dict(net_return=987654321)]),
        trace=dict(path="/private/trace", confirmed_cursor=64),
    )


def collect(tmp_path, report):
    folder = tmp_path / "campaign"
    folder.mkdir(exist_ok=True)
    (folder / "run.json").write_text(json.dumps(report))
    source = dict(id="native-adaptive", path="campaign", kind="archive", domain="synthetic")
    with Collector(tmp_path, tmp_path / "cache.sqlite", max_files=1) as collector:
        return collector.collect([source])


@pytest.mark.parametrize("model", VARIANTS)
def test_native_variants_have_distinct_public_identity(tmp_path, model):
    snapshot = collect(tmp_path, native_report(model))
    run = snapshot["runs"][0]
    assert run["model_id"] == model
    assert run["completed_steps"] == 32
    assert run["financial_validation"] is None
    assert next(row for row in snapshot["models"] if row["id"] == model)["kind"] == "reinforcement"
    assert "987654321" not in json.dumps(snapshot)
    assert "/private/trace" not in json.dumps(snapshot)
    assert any("3" in note and "140" in note for note in snapshot["notes"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "unrecognised"),
        ("backend", "unrecognised"),
        ("model", "unknown"),
        ("analysis_domain", "real"),
        ("final_test_opened", True),
        ("macro_coverage", dict(simulated_concepts=140, catalog_concepts=140)),
    ],
)
def test_unknown_v2_contract_is_rejected(tmp_path, field, value):
    report = native_report("ppo_gru")
    report[field] = value
    with pytest.raises(ValueError):
        collect(tmp_path, report)


@pytest.mark.parametrize("model", VARIANTS)
def test_native_variant_requires_its_activity(tmp_path, model):
    report = native_report(model)
    del report["activity"]
    with pytest.raises(ValueError):
        collect(tmp_path, report)


def test_trace_files_do_not_consume_report_traversal_budget(tmp_path):
    folder = tmp_path / "campaign" / "trace"
    folder.mkdir(parents=True)
    for index in range(32):
        (folder / f"trace-{index}.parquet").write_bytes(b"private")
    assert len(collect(tmp_path, native_report("ppo_episodic"))["runs"]) == 1


def test_native_audit_exposes_only_status(tmp_path):
    report = native_report("ppo_episodic")
    report.update(
        kind="native_ppo_audit", activity="evaluation", phase="evaluation", partition="audit"
    )
    report["financial_validation"] = dict(net_return=987654321)
    run = collect(tmp_path, report)["runs"][0]
    assert run["status"] == "completed"
    assert run["financial_validation"] is None
    assert all(value is None for value in run["metrics"].values())


def test_campaign_summary_does_not_turn_cases_into_optimizer_steps(tmp_path):
    report = native_report("adaptive_comparison")
    report.update(schema_version=1, kind="adaptive_campaign", completed_cases=21)
    del report["global_step"]
    del report["total_steps"]
    snapshot = collect(tmp_path, report)
    run = snapshot["runs"][0]
    assert run["model_id"] == "adaptive_comparison"
    assert run["completed_steps"] is None
    assert run["total_steps"] is None
    assert (
        next(row for row in snapshot["models"] if row["id"] == "adaptive_comparison")["kind"]
        == "summary"
    )


def native_registry(tmp_path, *, report=None, status="pending"):
    folder = tmp_path / "campaign"
    folder.mkdir()
    registry = dict(
        schema_version=1,
        kind="adaptive_campaign",
        status="running",
        planned_runs=1,
        runs=[
            dict(
                path="pilot/ppo_gru-42",
                stage="pilot",
                variant="ppo_gru",
                seed=42,
                status=status,
                planned_transitions=8192,
                transitions=0,
                config_sha256="a" * 64,
            )
        ],
    )
    (folder / "registry.json").write_text(json.dumps(registry))
    aggregate = native_report("adaptive_comparison")
    aggregate.update(schema_version=1, kind="adaptive_campaign")
    (folder / "run.json").write_text(json.dumps(aggregate))
    if report is not None:
        child = folder / "pilot" / "ppo_gru-42"
        child.mkdir(parents=True)
        (child / "run.json").write_text(json.dumps(report))
    source = dict(
        id="adaptive", path="campaign", kind="archive", domain="synthetic", summary="registry.json"
    )
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        return collector.collect([source])


def test_native_registry_imports_planned_cases_without_duplicating_the_coordinator(tmp_path):
    snapshot = native_registry(tmp_path)
    assert len(snapshot["runs"]) == 1
    run = snapshot["runs"][0]
    assert (run["model_id"], run["seed"], run["activity"], run["status"]) == (
        "ppo_gru",
        42,
        "rl",
        "queued",
    )
    assert run["variant_id"] == "piloto"
    assert run["completed_steps"] is None
    assert run["total_steps"] == 8192
    assert run["metadata"]["configuration_sha256"] == "a" * 64
    assert snapshot["campaigns"][0]["planned_runs"] == 1


def test_native_registry_keeps_confirmed_progress_and_a_blocking_verdict(tmp_path):
    report = native_report("ppo_gru")
    report["status"] = "running"
    report["resources"] = dict(
        ram_peak_bytes=16 * 1024**2, ram_peak_method="procfs_VmHWM", vram_peak_bytes=None
    )
    report["invocation_seconds"] = 9.5
    run = native_registry(tmp_path, report=report, status="blocked")["runs"][0]
    assert run["status"] == "blocked"
    assert run["completed_steps"] == 32
    assert run["metrics"]["ram_peak_mib"] == 16
    assert run["metrics"]["vram_peak_mib"] is None
    assert run["metrics"]["elapsed_seconds"] is None


def test_native_registry_rejects_a_different_model_in_the_same_case(tmp_path):
    with pytest.raises(ValueError):
        native_registry(tmp_path, report=native_report("ppo_window"))


@pytest.mark.parametrize("previous_status", [None, "running", "paused"])
def test_waiting_for_gpu_remains_registered_without_claiming_activity(tmp_path, previous_status):
    report = native_report("ppo_gru") if previous_status else None
    if report:
        report["status"] = previous_status
    runs = native_registry(tmp_path, report=report, status="waiting")["runs"]
    assert len(runs) == 1
    assert runs[0]["status"] == "queued"
    assert runs[0]["heartbeat_at"] is None
    assert runs[0]["completed_steps"] == (32 if report else None)


def test_report_written_during_collection_uses_the_actual_observation_time(tmp_path, monkeypatch):
    from datetime import UTC, datetime

    from mars_titan.observatory import collector as module

    class Clock(datetime):
        current = datetime(2026, 9, 29, 12, tzinfo=UTC)

        @classmethod
        def now(cls, tz=None):
            return cls.current

    original_read = Collector.read

    def read(self, path, **kwargs):
        result = original_read(self, path, **kwargs)
        if path.name == "run.json":
            Clock.current = datetime(2026, 9, 29, 12, 0, 2, tzinfo=UTC)
        return result

    monkeypatch.setattr(module, "datetime", Clock)
    monkeypatch.setattr(module, "lock_held", lambda folder: True)
    monkeypatch.setattr(Collector, "read", read)
    report = native_report("ppo_gru")
    report.update(status="running", updated_at="2026-09-29T12:00:01Z")
    snapshot = native_registry(tmp_path, report=report, status="running")
    assert snapshot["runs"][0]["updated_at"] == "2026-09-29T12:00:01Z"
    assert snapshot["runs"][0]["heartbeat_at"] == "2026-09-29T12:00:02Z"
    assert snapshot["generated_at"] == "2026-09-29T12:00:02Z"


def test_native_registry_still_rejects_a_future_receipt(tmp_path):
    report = native_report("ppo_gru")
    report["updated_at"] = "2099-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="posterior"):
        native_registry(tmp_path, report=report)
