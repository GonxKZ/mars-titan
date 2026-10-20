"""Contrastes de objetivos con artefactos sintéticos identificados, sin ajustes."""

import json

from tests.tooling.test_quantile_multiseed_analysis import api, multiseed_fixture


def point_fixture(work):
    from mars_titan.data.storage import atomic_json, sha256

    prepared = multiseed_fixture(work)
    prepared["study_version"] = "quantile_point_v1"
    prepared["configuration"].update(
        primary_reference="point_component",
        uncertainty_training={"point_component": "not_trained", "point_l1": "not_trained"},
    )
    names = {"median_isolated": "point_component", "encoder_isolated": "point_l1"}
    for new in names.values():
        path = work / f"recipes/{new}.json"
        atomic_json(path, dict(recipe=dict(loss=new)))
        prepared["artifacts"][f"recipes/{new}.json"] = sha256(path)
    for job in prepared["jobs"]:
        if job["arm"] not in names:
            continue
        old_id = job["id"]
        job["arm"] = names[job["arm"]]
        job["id"] = f"{job['arm']}-{job['seed']}"
        job["recipe"] = f"recipes/{job['arm']}.json"
        folder = work / "runs" / old_id
        folder.rename(work / "runs" / job["id"])
        path = work / "runs" / job["id"] / "run.json"
        report = json.loads(path.read_text())
        report["request"]["recipe_sha256"] = prepared["artifacts"][job["recipe"]]
        atomic_json(path, report)
    atomic_json(work / "prepared.json", prepared)
    digest = sha256(work / "prepared.json")
    for path in (work / "launches").glob("*.json"):
        receipt = json.loads(path.read_text())
        old = receipt["job"]
        arm = names.get(old["arm"], old["arm"])
        receipt["job"] = next(
            job for job in prepared["jobs"] if job["arm"] == arm and job["seed"] == old["seed"]
        )
        receipt["prepared_sha256"] = digest
        atomic_json(path, receipt)
    atomic_json(work / "execution-plan.json", dict(prepared_sha256=digest, prior_receipts=[]))
    return prepared


def test_point_analysis_keeps_the_predeclared_reference_and_marks_untrained_tails(
    tmp_path, monkeypatch, learning_doubles
):
    work = tmp_path / "work"
    work.mkdir()
    prepared = point_fixture(work)
    monkeypatch.setattr(api(), "verify_prepared", lambda path: prepared)
    result = api().analyze(work, tmp_path / "output")
    assert result["kind"] == "quantile_point_objective_multiseed_development_results"
    assert result["primary"]["contrasts"][0]["coefficients"] == {
        "both_isolated": 1.0,
        "point_component": -1.0,
    }
    assert result["primary"]["contrasts"][0]["estimate"] == -5.5
    assert result["primary"]["resampling"]["sessions"] == 2
    assert result["unity_control"]["passed"]
    contrasts = {item["name"] for item in result["secondary"]["contrasts"]}
    assert "point_l1_minus_point_component" in contrasts
    assert "both_isolated_minus_point_component" in contrasts
    assert "both_isolated_minus_point_l1" in contrasts
    assert "interaction" not in contrasts and len(contrasts) == 12
    assert result["objective_controls"] == {
        "point_component": "not_trained",
        "point_l1": "not_trained",
    }
    assert result["final_test_opened"] is False
