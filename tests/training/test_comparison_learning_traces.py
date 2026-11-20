"""Activación de observaciones sobre los ajustes ya declarados de la campaña."""

import json

import pytest

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training import comparison_campaign as campaign
from tests.training.test_comparison_campaign import declaration, layout


def trace_files(place):
    place.state.mkdir(parents=True, exist_ok=True)
    evidence = dict(
        passed=True,
        models=["titans_mac"],
        every=128,
        max_bytes=2**20,
        max_relative_overhead=0.05,
        measurements={
            "titans_mac": dict(
                report_sha256="e" * 64,
                every=128,
                device="cuda:0",
                exact=True,
                prediction_rows=17,
                gradient_sets=4,
                trace_rows=9,
                timing_pairs=3,
                relative_overhead=0.025,
            )
        },
    )
    measured = place.state / "trace-qualification.json"
    atomic_json(measured, evidence)
    path = place.state / "learning-traces.json"
    atomic_json(
        path,
        dict(
            schema_version=1,
            runtime_commit="b" * 40,
            declaration_sha256="a" * 64,
            qualification_sha256=sha256(measured),
            models=evidence["models"],
            every=evidence["every"],
            max_bytes=evidence["max_bytes"],
        ),
    )
    return path


def test_authorization_binds_the_measured_policy_and_rejects_a_different_cadence(tmp_path):
    place = layout(tmp_path)
    path = trace_files(place)
    permission = campaign.authorize(place, "a" * 64, commit="b" * 40, note="prueba técnica")
    assert permission["learning_trace_policy_sha256"] == sha256(path)
    atomic_json(path, dict(json.loads(path.read_text()), every=1))
    with pytest.raises(ValueError, match="medida|cadencia"):
        campaign.authorize(place, "a" * 64, commit="b" * 40, note="prueba técnica")


def test_qualified_fit_receives_the_policy_and_leaves_a_confirmed_checkpoint_receipt(
    tmp_path, monkeypatch
):
    from mars_titan.training.checkpoints import save_training_state

    place = layout(tmp_path)
    path = trace_files(place)
    campaign.authorize(place, "a" * 64, commit="b" * 40, note="prueba técnica")
    document = campaign.checked_declaration(declaration())
    job = next(j for j in campaign.plan(document, "mars_titan") if j.model == "titans_mac")
    folder = place.output(job.key)
    identity = {"fixture": "sin aprendizaje"}
    cursor = {"sequence": 0, "bytes": 0, "exhausted_at": None}
    seen = []

    def execute(*args, **options):
        seen.append(options.get("traces"))
        state = dict(global_step=129, traces=cursor)
        ck = save_training_state(folder / "fit/checkpoints", identity=identity, state=state)
        atomic_json(
            folder / "fit/run.json",
            dict(
                run_id="f" * 64,
                identity=identity,
                global_step=129,
                recovery_checkpoint=dict(path="checkpoints/" + ck.name, sha256=sha256(ck)),
            ),
        )
        return {"status": "paused"}

    monkeypatch.setattr(campaign, "execute_job", execute)
    assert campaign.run_job(document, place, job, device="cpu") == "paused"
    assert seen == [dict(every=128, max_bytes=2**20)]
    entry = campaign.read_record(place, job.key)["attempts"][0]["learning_traces"]
    receipt = json.loads((folder / "trace-receipt.json").read_text())
    assert entry["start_step"] == entry["resume_step"] == receipt["start_step"] == 0
    assert receipt["policy_sha256"] == sha256(path)
    assert receipt["cursor"] == cursor and receipt["global_step"] == 129
    assert campaign.run_job(document, place, job, device="cpu", preflight=True) == "paused"
    second = campaign.read_record(place, job.key)["attempts"][-1]["learning_traces"]
    assert second["start_step"] == 0 and second["resume_step"] == 129
    assert seen[1] == seen[0]


def test_unowned_trace_directory_is_rejected_before_opening_the_trainer(tmp_path, monkeypatch):
    place = layout(tmp_path)
    trace_files(place)
    campaign.authorize(place, "a" * 64, commit="b" * 40, note="prueba técnica")
    document = campaign.checked_declaration(declaration())
    job = next(j for j in campaign.plan(document, "mars_titan") if j.model == "titans_mac")
    other = place.output(job.key) / "traces"
    other.mkdir(parents=True)
    saved = other / "contenido"
    saved.write_bytes(b"independiente")
    monkeypatch.setattr(campaign, "execute_job", lambda *a, **k: pytest.fail("entrenador abierto"))
    assert campaign.run_job(document, place, job, device="cpu") == campaign.PENDING
    attempt = campaign.read_record(place, job.key)["attempts"][-1]
    assert "no pertenecen" in attempt["error"] and saved.read_bytes() == b"independiente"


@pytest.mark.parametrize(
    ("key", "value"),
    [("exact", False), ("relative_overhead", 0.051), ("trace_rows", 0), ("gradient_sets", 0)],
)
def test_incomplete_cuda_evidence_cannot_qualify_a_model(tmp_path, key, value):
    place = layout(tmp_path)
    policy = trace_files(place)
    measured = place.state / "trace-qualification.json"
    evidence = json.loads(measured.read_text())
    evidence["measurements"]["titans_mac"][key] = value
    atomic_json(measured, evidence)
    atomic_json(policy, dict(json.loads(policy.read_text()), qualification_sha256=sha256(measured)))
    with pytest.raises(ValueError, match="medida CUDA"):
        campaign.authorize(place, "a" * 64, commit="b" * 40, note="prueba técnica")


def test_a_different_cadence_for_one_model_needs_its_own_measurement(tmp_path):
    from mars_titan.training import comparison_traces

    place = layout(tmp_path)
    policy = trace_files(place)
    changed = dict(json.loads(policy.read_text()), every_by_model={"titans_mac": 256})
    atomic_json(policy, changed)
    with pytest.raises(ValueError, match="cadencia"):
        campaign.authorize(place, "a" * 64, commit="b" * 40, note="prueba técnica")
    measured = place.state / "trace-qualification.json"
    evidence = json.loads(measured.read_text())
    evidence["measurements"]["titans_mac"]["every"] = 256
    atomic_json(measured, evidence)
    atomic_json(policy, dict(changed, qualification_sha256=sha256(measured)))
    perm = campaign.authorize(place, "a" * 64, commit="b" * 40, note="prueba técnica")
    document = campaign.checked_declaration(declaration())
    job = next(j for j in campaign.plan(document, "mars_titan") if j.model == "titans_mac")
    entry = comparison_traces.prepare(place, job, document["models"], {"attempts": []}, perm)
    assert entry["every"] == 256 and entry["resume_step"] == 0


def test_retry_rejects_a_trace_manifest_from_another_model_before_it_is_rewritten(tmp_path):
    from mars_titan.training import comparison_traces

    place = layout(tmp_path)
    trace_files(place)
    permission = campaign.authorize(place, "a" * 64, commit="b" * 40, note="prueba técnica")
    document = campaign.checked_declaration(declaration())
    job = next(j for j in campaign.plan(document, "mars_titan") if j.model == "titans_mac")
    record = {"attempts": []}
    entry = comparison_traces.prepare(place, job, document["models"], record, permission)
    record["attempts"].append({"learning_traces": entry})
    root = place.output(job.key)
    foreign = root / "traces/manifest.json"
    atomic_json(foreign, {"identity": {"family": "transformer"}})
    before = foreign.read_bytes()
    with pytest.raises(ValueError, match="manifiesto de trazas"):
        comparison_traces.prepare(place, job, document["models"], record, permission)
    assert foreign.read_bytes() == before
