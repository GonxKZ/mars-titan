"""Encadenado recuperable de procesos con recibos CPU y sin entrenamiento científico."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation.splits import build_folds
from mars_titan.training.checkpoints import StopRequest


def inputs(tmp_path, market="US"):
    root = tmp_path / "inputs"
    root.mkdir()
    encoded = root / "encoded/manifest.json"
    atomic_json(
        encoded,
        dict(
            final_test_opened=False,
            context_sessions=64,
            configuration=dict(encoders={"fixture": "cpu"}),
        ),
    )
    admission = root / "admission.json"
    atomic_json(
        admission, dict(required_indicator_ids=[f"indicator-{index}" for index in range(140)])
    )
    protocol = json.loads(Path("configs/evaluation/real-expanded-walk-forward.json").read_text())
    protocol["market"] = market
    views = root / "views"
    constants = dict(
        parent_sha256="a" * 64, macro_sha256="b" * 64, admission_sha256=sha256(admission)
    )
    report = dict(
        schema_version=1,
        status="temporal_views_prepared",
        final_test_opened=False,
        protocol_sha256="d" * 64,
        folds=[],
        **constants,
    )
    for fold in build_folds(protocol):
        manifest = views / fold["id"] / "manifest.json"
        counts = dict(train=10, validation=4, calibration=3, evaluation=2)
        atomic_json(
            manifest,
            dict(
                final_test_opened=False,
                scope="full_corpus",
                cohort_complete=True,
                markets=[protocol["market"]],
                assets=[dict(market=protocol["market"], symbol="fixture", counts=counts)],
                context_sessions=64,
                configuration=dict(source_manifest_sha256=sha256(encoded)),
                counts=counts,
                temporal_view=dict(
                    protocol=protocol, fold=fold, admission_path=str(admission), **constants
                ),
            ),
        )
        report["folds"].append(
            dict(
                id=fold["id"],
                has_all_partitions=True,
                counts=counts,
                manifest_sha256=sha256(manifest),
            )
        )
    atomic_json(views / "report.json", report)
    files = {}
    for name, filename in (
        ("neural_config", f"convergence-temporal-search-{market.lower()}.json"),
        ("tabular_config", "tabular-convergence-us.json"),
        ("post_config", "real-matched-posttraining.json"),
    ):
        files[name] = root / f"{name}.json"
        files[name].write_bytes((Path("configs/baselines") / filename).read_bytes())
    return SimpleNamespace(
        views=views,
        encoded=encoded,
        **files,
        references=tmp_path / "references",
        completion=tmp_path / "completion",
        state_dir=tmp_path / "state",
    )


def write_child(stage, path, expected, args, identity, *, status="completed", wrong_identity=False):
    if stage == "reliability":
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "cases.csv").write_text("model,accuracy\nfixture,0.5\n")
        report = dict(
            kind="frozen_campaign_reliability",
            status=status,
            final_test_opened=False,
            target_kind="residual_return",
            coverage_guaranteed=False,
            counts=dict(models=expected, prediction_files=2 * expected),
            provenance=dict(
                reference_sha256=sha256(args.references / "summary.json"),
                completion_sha256=sha256(args.completion / "summary.json"),
            ),
            analysis_source_sha256=identity["code"],
            artifacts={"cases.csv": sha256(path.parent / "cases.csv")},
        )
    else:
        inherited = (
            identity["neural"]
            if stage == "neural"
            else dict(
                reference_sha256=sha256(args.references / "summary.json"),
                tabular_config_sha256=sha256(args.tabular_config),
                post_config_sha256=sha256(args.post_config),
                encoded_sha256=sha256(args.encoded),
                code=identity["code"],
                dependency=None,
            )
        )
        if wrong_identity:
            inherited = (
                dict(inherited, config_sha256="f" * 64)
                if stage == "neural"
                else dict(inherited, reference_sha256="f" * 64)
            )
        report = dict(
            kind="temporal_reference_search"
            if stage == "neural"
            else "temporal_posttraining_completion",
            identity=inherited,
            status=status,
            completed_runs=expected if status == "completed" else 1,
            planned_runs=expected,
            final_test_opened=False,
        )
    atomic_json(path, report)
    return report


@pytest.mark.parametrize("market", ["US", "CN"])
def test_plan_derives_all_counts_and_rejects_mismatched_seeds(tmp_path, market):
    from mars_titan.training.real_campaign import prepare_campaign

    args = inputs(tmp_path, market)
    identity, stages = prepare_campaign(args)
    assert len(identity["neural"]["manifests"]) == 10
    assert [(s["name"], s["planned"], s["unit"]) for s in stages] == [
        ("neural", 400, "run"),
        ("completion", 3380, "run"),
        ("reliability", 1890, "model"),
    ]
    for name in ("neural_config", "tabular_config", "post_config"):
        path = getattr(args, name)
        config = json.loads(path.read_text())
        config["seeds" if name == "post_config" else "finalist_seeds"] = [42, 43]
        atomic_json(path, config)
    _, stages = prepare_campaign(args)
    assert [s["planned"] for s in stages] == [280, 2360, 1320]
    config = json.loads(args.post_config.read_text())
    config["seeds"] = [42]
    atomic_json(args.post_config, config)
    with pytest.raises(ValueError, match="semillas"):
        prepare_campaign(args)


@pytest.mark.parametrize("completed_before_cut", [True, False])
def test_chain_resumes_pauses_and_never_rewrites_confirmed_children(
    tmp_path, monkeypatch, completed_before_cut
):
    from mars_titan.training import real_campaign as module

    args = inputs(tmp_path)
    identity, _ = module.prepare_campaign(args)
    monkeypatch.setattr(module, "_code", lambda: identity["code"])
    calls = []

    def child(command, receipt, expected, stop, observe, **options):
        stage = (
            "neural"
            if "mars_titan.training.temporal_search" in command
            else "completion"
            if "mars_titan.posttraining.completion" in command
            else "reliability"
        )
        calls.append((stage, "--resume" in command))
        result = write_child(
            stage,
            receipt,
            expected,
            args,
            identity,
            status="paused" if len(calls) == 1 and not completed_before_cut else "completed",
        )
        observe(options["receipt_reader"](receipt, expected))
        if len(calls) == 1:
            raise InterruptedError("Corte de prueba")
        return result

    monkeypatch.setattr(module, "run_child", child)
    paused = module.run_campaign(args, StopRequest())
    assert paused["status"] == "paused"
    assert [name for name, _ in calls] == ["neural"]
    original = sha256(args.references / "summary.json")
    completed = module.run_campaign(args, StopRequest())
    assert completed["status"] == "completed"
    assert completed["planned_runs"] == completed["completed_runs"] == 3780
    assert completed["completed_reliability_models"] == 1890
    assert [name for name, _ in calls] == (
        ["neural", "completion", "reliability"]
        if completed_before_cut
        else ["neural", "neural", "completion", "reliability"]
    )
    if completed_before_cut:
        assert sha256(args.references / "summary.json") == original
    else:
        assert calls[1] == ("neural", True)
    saved = sha256(args.state_dir / "summary.json")
    assert module.run_campaign(args, StopRequest())["status"] == "completed"
    assert sha256(args.state_dir / "summary.json") == saved
    (args.state_dir / "reliability/cases.csv").write_text("corrupto")
    with pytest.raises(ValueError, match="artefacto"):
        module.run_campaign(args, StopRequest())


@pytest.mark.parametrize("fault", ["failure", "wrong_identity", "source_change"])
def test_failures_stop_the_chain_and_preserve_pending_stages(tmp_path, monkeypatch, fault):
    from mars_titan.training import real_campaign as module

    args = inputs(tmp_path)
    identity, _ = module.prepare_campaign(args)
    monkeypatch.setattr(module, "_code", lambda: identity["code"])
    calls = []

    def child(command, receipt, expected, stop, observe, **options):
        calls.append(command)
        if fault == "failure":
            raise RuntimeError("Fallo del proceso")
        write_child(
            "neural", receipt, expected, args, identity, wrong_identity=fault == "wrong_identity"
        )
        if fault == "source_change":
            args.encoded.write_text('{"final_test_opened":false,"changed":true}')
        return options["receipt_reader"](receipt, expected)

    monkeypatch.setattr(module, "run_child", child)
    with pytest.raises((RuntimeError, ValueError)):
        module.run_campaign(args, StopRequest())
    state = json.loads((args.state_dir / "summary.json").read_text())
    assert state["status"] == "failed"
    assert state["error"]["type"] in {"RuntimeError", "ValueError"}
    assert len(calls) == 1
    assert not args.completion.exists()


def test_initial_stop_and_changed_design_do_not_start_a_process(tmp_path, monkeypatch):
    from mars_titan.training import real_campaign as module

    args = inputs(tmp_path)
    monkeypatch.setattr(
        module, "run_child", lambda *_args, **_kwargs: pytest.fail("No debe ejecutar procesos")
    )
    stop = StopRequest()
    stop.request_stop()
    assert module.run_campaign(args, stop)["status"] == "paused"
    config = json.loads(args.post_config.read_text())
    config["learning_rate"] = 0.0002
    atomic_json(args.post_config, config)
    with pytest.raises(ValueError, match="identidad"):
        module.run_campaign(args, StopRequest())


def test_exclusive_state_lock_and_overlapping_paths_are_rejected(tmp_path, monkeypatch):
    import fcntl
    import os

    from mars_titan.training import real_campaign as module

    args = inputs(tmp_path)
    monkeypatch.setattr(
        module, "run_child", lambda *_args, **_kwargs: pytest.fail("No debe iniciar un proceso")
    )
    args.state_dir.mkdir()
    descriptor = os.open(args.state_dir / ".campaign.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            module.run_campaign(args, StopRequest())
    finally:
        os.close(descriptor)
    args.references = args.views / "output"
    with pytest.raises(ValueError):
        module.prepare_campaign(args)


def test_resource_cache_configuration_is_frozen_on_resume(tmp_path, monkeypatch):
    from mars_titan.training import real_campaign as module

    args = inputs(tmp_path)
    monkeypatch.setattr(
        module, "run_child", lambda *_args, **_kwargs: pytest.fail("No debe iniciar un proceso")
    )
    monkeypatch.delenv("MARS_TITAN_INPUT_CACHE_MIB", raising=False)
    identity, _ = module.prepare_campaign(args)
    monkeypatch.setattr(module, "_code", lambda: identity["code"])
    assert identity["resources"]["MARS_TITAN_INPUT_CACHE_MIB"] is None
    stop = StopRequest()
    stop.request_stop()
    assert module.run_campaign(args, stop)["status"] == "paused"
    monkeypatch.setenv("MARS_TITAN_INPUT_CACHE_MIB", "4096")
    with pytest.raises(ValueError, match="identidad"):
        module.run_campaign(args, StopRequest())


def test_changed_code_is_rejected_on_resume(tmp_path, monkeypatch):
    from mars_titan.training import real_campaign as module

    args = inputs(tmp_path)
    identity, _ = module.prepare_campaign(args)
    monkeypatch.setattr(module, "_code", lambda: identity["code"])
    monkeypatch.setattr(
        module, "run_child", lambda *_args, **_kwargs: pytest.fail("No debe iniciar un proceso")
    )
    stop = StopRequest()
    stop.request_stop()
    assert module.run_campaign(args, stop)["status"] == "paused"
    monkeypatch.setattr(
        module, "_code", lambda: dict(identity["code"], **{"training/real_campaign.py": "f" * 64})
    )
    with pytest.raises(ValueError, match="identidad"):
        module.run_campaign(args, StopRequest())


def test_pause_in_completion_keeps_the_neural_stage_confirmed(tmp_path, monkeypatch):
    from mars_titan.training import real_campaign as module

    args = inputs(tmp_path)
    identity, _ = module.prepare_campaign(args)
    monkeypatch.setattr(module, "_code", lambda: identity["code"])
    calls = []

    def child(command, receipt, expected, stop, observe, **options):
        stage = (
            "neural"
            if "mars_titan.training.temporal_search" in command
            else "completion"
            if "mars_titan.posttraining.completion" in command
            else "reliability"
        )
        calls.append(stage)
        paused = calls == ["neural", "completion"]
        result = write_child(
            stage, receipt, expected, args, identity, status="paused" if paused else "completed"
        )
        observe(options["receipt_reader"](receipt, expected))
        if paused:
            raise InterruptedError("Pausa de la continuación")
        return result

    monkeypatch.setattr(module, "run_child", child)
    paused = module.run_campaign(args, StopRequest())
    assert paused["status"] == "paused"
    assert paused["stages"][0]["status"] == "completed"
    assert paused["completed_runs"] == 401
    digest = sha256(args.references / "summary.json")
    assert module.run_campaign(args, StopRequest())["status"] == "completed"
    assert calls == ["neural", "completion", "completion", "reliability"]
    assert sha256(args.references / "summary.json") == digest


@pytest.mark.parametrize("fault", ["encoding", "context", "indicator_count", "admission_hash"])
def test_representation_and_indicator_admission_fail_before_any_child(tmp_path, monkeypatch, fault):
    from mars_titan.training import real_campaign as module

    args = inputs(tmp_path)
    report_path = args.views / "report.json"
    report = json.loads(report_path.read_text())
    admission_path = args.views.parent / "admission.json"
    if fault == "encoding":
        encoded = json.loads(args.encoded.read_text())
        encoded["configuration"]["encoders"] = {"fixture": "different"}
        atomic_json(args.encoded, encoded)
    elif fault == "admission_hash":
        admission_path.write_text(admission_path.read_text() + " ")
    else:
        if fault == "indicator_count":
            atomic_json(
                admission_path,
                dict(required_indicator_ids=[f"indicator-{index}" for index in range(139)]),
            )
            report["admission_sha256"] = sha256(admission_path)
        for row in report["folds"]:
            path = args.views / row["id"] / "manifest.json"
            metadata = json.loads(path.read_text())
            if fault == "context":
                metadata["context_sessions"] = 32
            else:
                metadata["temporal_view"]["admission_sha256"] = sha256(admission_path)
            atomic_json(path, metadata)
            row["manifest_sha256"] = sha256(path)
        atomic_json(report_path, report)
    monkeypatch.setattr(
        module,
        "run_child",
        lambda *_args, **_kwargs: pytest.fail("La admisión debe impedir cualquier proceso"),
    )
    with pytest.raises(ValueError):
        module.run_campaign(args, StopRequest())
    assert not args.references.exists()


@pytest.mark.parametrize("status,exit_code", [("completed", 0), ("paused", 2)])
def test_cli_passes_paths_and_keeps_the_paused_exit_code(
    tmp_path, monkeypatch, capsys, status, exit_code
):
    from mars_titan.training import real_campaign as module

    args = inputs(tmp_path)
    received = []

    def run(options, stop):
        received.append((options.references, options.completion, options.state_dir, stop.requested))
        return dict(
            status=status, completed_runs=0, planned_runs=3780, completed_reliability_models=0
        )

    monkeypatch.setattr(module, "run_campaign", run)
    argv = []
    for option, name in (
        ("views", "views"),
        ("encoded", "encoded"),
        ("neural-config", "neural_config"),
        ("tabular-config", "tabular_config"),
        ("post-config", "post_config"),
        ("references-output", "references"),
        ("completion-output", "completion"),
        ("state-dir", "state_dir"),
    ):
        argv.extend(("--" + option, str(getattr(args, name))))
    assert module.main(argv) == exit_code
    assert received == [(args.references, args.completion, args.state_dir, False)]
    assert status in capsys.readouterr().out
