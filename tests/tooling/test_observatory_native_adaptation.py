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
