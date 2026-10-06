"""Reconocer evaluaciones confirmadas sin publicar sus métricas reservadas."""

import copy
import hashlib
import json

import pytest

from mars_titan.observatory.collector import Collector, digest


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def campaign(tmp_path, family="gru", case=None):
    case = case if case is not None else {"mode": "neural_mae", "seed": 42}
    key = f"posttraining/{family}/seed-42/real/{case.get('mode', 'selected')}"
    job = dict(
        id=key,
        stage="posttraining",
        report="/frozen/parent/run.json",
        sha256="a" * 64,
        comparator=None,
        phase="posttraining",
    )
    path = tmp_path / "campaign" / "runs" / key / "run.json"
    predictions = {}
    for partition in ("calibration", "evaluation"):
        artifact = path.parent / f"{partition}.parquet"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"El recolector no debe abrir este archivo")
        predictions[partition] = dict(
            path=artifact.name,
            sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            metrics={"prediction": {"mae": 987654321, "samples": 999}},
        )
    report = dict(
        schema_version=1,
        status="completed",
        job=job,
        family=family,
        case=case,
        parent_comparison=True,
        checkpoint={"path": "private/model.pt", "sha256": "b" * 64},
        selection={"metric": "session_mae", "value": 987654321},
        predictions=predictions,
        elapsed_seconds=1.25,
        primary="median",
        final_test_opened=False,
    )
    summary = dict(
        schema_version=1,
        kind="frozen_temporal_evaluation",
        identity=dict(
            jobs=[copy.deepcopy(job)],
            sources={},
            partitions=["calibration", "evaluation"],
            ordered_sha256="c" * 64,
            manifest_sha256="d" * 64,
            code={},
        ),
        status="completed",
        runs={key: {"path": str(path.relative_to(tmp_path / "campaign"))}},
        planned_runs=1,
        completed_runs=1,
        final_test_opened=False,
    )
    save(tmp_path, report, summary, path)
    return report, summary, path


def save(tmp_path, report, summary, path):
    dump(path, report)
    next(iter(summary["runs"].values()))["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    dump(tmp_path / "campaign/summary.json", summary)


def collect(collector):
    return collector.collect(
        [{"id": "frozen", "path": "campaign", "kind": "archive", "domain": "real"}]
    )


def test_fixture_matches_the_heldout_producer_confirmation_contract(tmp_path):
    from mars_titan.posttraining.heldout import _confirmed_results

    _, summary, _ = campaign(tmp_path)
    _confirmed_results(tmp_path / "campaign", summary, summary["identity"]["jobs"])


@pytest.mark.parametrize(
    "family,case",
    [
        ("rnn", {"mode": "neural_mae", "seed": 42}),
        ("lstm", {"mode": "neural_mse", "seed": 43}),
        ("gru", {"mode": "klpo_full", "seed": 44}),
        ("dlinear", {"kind": "dlinear", "seed": 42}),
        ("ridge", {"alpha": 1.0}),
        ("xgboost", {"seed": 43}),
        ("xgboost_external_cuda", {"seed": 43}),
    ],
)
def test_frozen_evaluation_keeps_family_seed_and_source_without_opening_artifacts(
    tmp_path, monkeypatch, family, case
):
    original, summary, path = campaign(tmp_path, family, case)
    public_family = "xgboost" if family == "xgboost_external_cuda" else family
    raw = path.read_bytes()
    opened = []
    read = Collector.read

    def checked_read(self, candidate, **kwargs):
        opened.append(candidate)
        assert candidate in {tmp_path / "campaign/summary.json", path}
        return read(self, candidate, **kwargs)

    monkeypatch.setattr(Collector, "read", checked_read)
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        for _ in range(2):
            result = collect(collector)
            assert len(result["runs"]) == 1
            run = result["runs"][0]
            assert run["activity"] == run["phase"] == "evaluation"
            assert run["model_id"] == public_family
            assert run["seed"] == case.get("seed")
            assert run["variant_id"] == original["job"]["id"].replace("/", ".")
            assert run["metadata"]["method"] == "evaluation"
            assert run["metadata"]["source_sha256"] == digest(original)
            assert run["metadata"]["configuration_sha256"] == digest(
                {"kind": public_family, "seed": case.get("seed")}
            )
            assert run["fold"] is None
            assert run["test_released"] is False
            assert run["history"] == []
            assert run["financial_validation"] is None
            assert all(value is None for value in run["metrics"].values())
            assert "987654321" not in json.dumps(result)
    assert len(opened) == 4
    assert path.read_bytes() == raw
    assert json.loads((tmp_path / "campaign/summary.json").read_text()) == summary


def test_frozen_predictive_evaluation_does_not_export_injected_financial_aggregates(tmp_path):
    report, summary, path = campaign(tmp_path)
    report["financial_validation"] = dict(
        partition="validation",
        net_return=0.5,
        max_drawdown=0.1,
        costs=10,
        turnover=2,
        steps=4,
        completed=True,
        invalid_reason=None,
    )
    save(tmp_path, report, summary, path)
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        run = collect(collector)["runs"][0]
    assert run["phase"] == "evaluation"
    assert run["financial_validation"] is None
    assert all(value is None for value in run["metrics"].values())
    assert run["metadata"]["source_sha256"] == digest(report)


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("report", "family", "mars_titan"),
        ("report", "status", "running"),
        ("report", "final_test_opened", True),
        ("report", "schema_version", True),
        ("report", "predictions", {"evaluation": {}}),
        ("report", "predictions", {"calibration": {}, "evaluation": {}, "test": {}}),
        ("report", "job", {"id": "unregistered"}),
        ("report", "identity", {"case": {"kind": "ridge", "mode": "neural_mae"}}),
        ("summary", "final_test_opened", True),
        ("summary", "schema_version", 2),
        ("identity", "partitions", ["validation", "evaluation"]),
        ("identity", "jobs", []),
        ("job", "sha256", "f" * 64),
    ],
)
def test_frozen_evaluation_rejects_a_contradictory_registered_receipt(
    tmp_path, target, field, value
):
    report, summary, path = campaign(tmp_path)
    objects = dict(report=report, summary=summary, identity=summary["identity"], job=report["job"])
    objects[target][field] = value
    save(tmp_path, report, summary, path)
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        with pytest.raises(ValueError):
            collect(collector)


@pytest.mark.parametrize("signature", [None, "0" * 64])
def test_frozen_report_hash_is_required_even_after_caching(tmp_path, signature):
    _, summary, _ = campaign(tmp_path)
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        collect(collector)
        next(iter(summary["runs"].values()))["sha256"] = signature
        dump(tmp_path / "campaign/summary.json", summary)
        with pytest.raises(ValueError, match="huella|registro|evaluaci"):
            collect(collector)


def test_frozen_evaluation_ignores_receipts_not_yet_confirmed_in_summary(tmp_path):
    campaign(tmp_path)
    dump(tmp_path / "campaign/runs/incomplete/run.json", {"status": "running"})
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        assert len(collect(collector)["runs"]) == 1


def test_frozen_evaluation_requires_the_registered_report(tmp_path):
    _, _, path = campaign(tmp_path)
    path.unlink()
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        with pytest.raises(ValueError, match="recibo|evaluaci"):
            collect(collector)
