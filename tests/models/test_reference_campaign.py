"""Contratos del ejecutor de rejillas, sin entrenamientos ni resultados experimentales."""

import importlib
import json

import pytest

from mars_titan.data.storage import atomic_json, sha256


def module():
    try:
        return importlib.import_module("mars_titan.models.baselines.campaign")
    except ModuleNotFoundError:
        pytest.fail("Falta el ejecutor reutilizable de las rejillas de referencias")


@pytest.fixture
def inputs(tmp_path):
    prepared, samples = tmp_path / "prepared", tmp_path / "samples"
    for root in (prepared, samples):
        root.mkdir()
        (root / "source.txt").write_text("Conservar el origen", encoding="utf-8")
    config = {
        "schema_version": 1,
        "purpose": "orchestration_test_only",
        "models": ["rnn", "lstm"],
        "seeds": [7, 9],
        "losses": ["mse", "mae"],
        "learning_rates": [0.001, 0.0001],
        "epochs": 2,
        "huber_delta": 0.01,
        "batch_size": 16,
        "workers": 0,
        "device": "cuda:0",
        "training_through": "2022-12-31",
        "validation_year": 2023,
        "final_test_opened": False,
        "posttraining": {
            "base_loss": "mse",
            "base_learning_rate": 0.001,
            "losses": ["mae", "mse"],
            "learning_rate": 0.0001,
            "epochs": 5,
            "selection": "fixed_control_not_validation_winner",
            "optimizer_policy": "new_adamw",
        },
    }
    path = tmp_path / "config.json"
    atomic_json(path, config)
    return path, prepared, samples, tmp_path / "campaign", config


@pytest.fixture
def probe_double(monkeypatch):
    calls = []

    def run(prepared, samples, output, report_path, **options):
        calls.append((prepared, samples, output, report_path, options))
        output.mkdir(parents=True)
        checkpoint = output / f"epoch-{options['epochs']}.pt"
        checkpoint.write_bytes(b"test-only-checkpoint:" + output.name.encode())
        predictions = output / "predictions.parquet"
        predictions.write_bytes(b"test-only-predictions:" + output.name.encode())
        training_predictions = output / "training-predictions.parquet"
        training_predictions.write_bytes(b"test-only-training-predictions:" + output.name.encode())
        parent = options.get("initialize_from")
        report = {
            "status": "completed",
            "model": options["kind"],
            "config": {
                "initialization": {
                    "sha256": sha256(parent),
                    "source_completed_epochs": 2,
                    "policy": "weights_only_new_optimizer_and_rng",
                }
                if parent is not None
                else None,
            },
            "final_test_opened": False,
            "epochs": [
                {
                    "epoch": options["epochs"],
                    "checkpoint_sha256": sha256(checkpoint),
                }
            ],
            "predictions_sha256": sha256(predictions),
            "training_predictions_sha256": sha256(training_predictions),
            "samples": {"train": 2, "validation": 1},
            "parameters": 3,
            "diagnostic_row_metrics": {options["kind"]: {"mae": 0.5, "mse": 0.25}},
            "resume_check": {"exact_weights": True},
            "restored_predictions_equal": True,
        }
        atomic_json(report_path, report)
        return report

    monkeypatch.setattr(module(), "run_temporal_probe", run)
    return calls, run


def test_campaign_runs_cartesian_grid_and_both_controls_from_same_final_parent(
    inputs, probe_double
):
    config_path, prepared, samples, output, config = inputs
    calls, _ = probe_double
    result = module().run_campaign(config_path, prepared, samples, output)

    assert result["status"] == "completed"
    assert result["planned_runs"] == result["completed_runs"] == 24
    assert result["failed_runs"] == 0
    assert result["final_test_opened"] is False
    assert result["resolved_config"] == config
    assert result["config_sha256"] == sha256(config_path)
    assert json.loads((output / "summary.json").read_text()) == result
    assert len(calls) == 24
    base = [row for row in result["runs"] if row["stage"] == "base"]
    assert {(row["kind"], row["loss"], row["learning_rate"], row["seed"]) for row in base} == {
        (kind, loss, rate, seed)
        for kind in ("rnn", "lstm")
        for loss in ("mse", "mae")
        for rate in (0.001, 0.0001)
        for seed in (7, 9)
    }
    for row, (actual_prepared, actual_samples, run, report, options) in zip(
        result["runs"], calls, strict=True
    ):
        assert actual_prepared == prepared and actual_samples == samples
        assert run == output / row["output_path"]
        assert report == output / row["report_path"]
        assert options["kind"] == row["kind"]
        assert options["loss"] == row["loss"]
        assert options["seed"] == row["seed"]
        assert options["huber_delta"] == 0.01
        assert row["status"] == "completed" and row["elapsed_seconds"] >= 0
        assert row["report_sha256"] == sha256(report)
        assert row["predictions_sha256"] == sha256(output / row["predictions_path"])
        assert row["training_predictions_sha256"] == sha256(
            output / row["training_predictions_path"]
        )
        assert row["checkpoint_sha256"] == sha256(output / row["checkpoint_path"])
        assert row["exact_recovery"] is True
        assert row["restored_predictions_equal"] is True
        if row["stage"] == "base":
            assert options["epochs"] == 2
            assert options["learning_rate"] == row["learning_rate"]
            assert "initialize_from" not in options
        else:
            parent = f"{row['kind']}-mse-0.001-s{row['seed']}"
            assert row["parent"] == parent
            assert options["epochs"] == 5 and options["learning_rate"] == 0.0001
            assert options["loss"] in {"mae", "mse"}
            assert options["initialize_from"] == output / "runs" / parent / "epoch-2.pt"
            assert row["parent_checkpoint_sha256"] == sha256(options["initialize_from"])
            assert row["options"]["initialize_from"] == f"runs/{parent}/epoch-2.pt"
    assert [row["loss"] for row in result["runs"][16:]] == ["mae", "mse"] * 4
    assert result["inputs"] == {
        "config": "../config.json",
        "prepared": "../prepared",
        "samples": "../samples",
    }
    assert str(output.parent) not in json.dumps(result)
    for source in (config_path, prepared / "source.txt", samples / "source.txt"):
        assert source.exists()
    assert (prepared / "source.txt").read_text() == "Conservar el origen"
    assert (samples / "source.txt").read_text() == "Conservar el origen"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("purpose", " "),
        ("purpose", []),
        ("models", []),
        ("models", ["gru", "gru"]),
        ("models", ["unknown"]),
        ("models", "rnn"),
        ("models", [["rnn"]]),
        ("seeds", []),
        ("seeds", [7, 7]),
        ("seeds", [-1]),
        ("seeds", [2**32]),
        ("seeds", [False]),
        ("losses", ["mse", "mse"]),
        ("losses", ["mae"]),
        ("losses", ["mse", "other"]),
        ("learning_rates", [0.0001]),
        ("learning_rates", [0.001, 0]),
        ("learning_rates", [0.001, True]),
        ("learning_rates", [0.001, 0.001]),
        ("learning_rates", [0.001, 9007199254740992, 9007199254740993]),
        ("epochs", 1),
        ("epochs", 31),
        ("epochs", 2.0),
        ("huber_delta", 0),
        ("huber_delta", False),
        ("batch_size", 32),
        ("workers", 1),
        ("workers", False),
        ("device", "cpu"),
        ("training_through", "2023-12-31"),
        ("validation_year", 2024),
        ("validation_year", 2023.0),
        ("final_test_opened", True),
        ("final_test_opened", 0),
        ("unexpected", "value"),
    ],
)
def test_invalid_configuration_is_rejected_without_outputs_or_runner(
    inputs, monkeypatch, key, value
):
    config_path, prepared, samples, output, config = inputs
    config[key] = value
    atomic_json(config_path, config)
    unit = module()
    monkeypatch.setattr(unit, "run_temporal_probe", lambda *args, **kw: pytest.fail("Se entrenó"))
    with pytest.raises(ValueError):
        unit.run_campaign(config_path, prepared, samples, output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("base_loss", "mae"),
        ("base_learning_rate", 0.0001),
        ("losses", ["mae"]),
        ("losses", ["mae", "mae"]),
        ("losses", ["mae", "mse", "huber"]),
        ("learning_rate", 0.001),
        ("epochs", 4),
        ("epochs", 5.0),
        ("selection", "validation_winner"),
        ("optimizer_policy", "resume_adamw"),
        ("loss", "mae"),
    ],
)
def test_changed_posttraining_control_is_rejected_before_outputs(inputs, key, value):
    config_path, prepared, samples, output, config = inputs
    config["posttraining"][key] = value
    atomic_json(config_path, config)
    with pytest.raises(ValueError):
        module().run_campaign(config_path, prepared, samples, output)
    assert not output.exists()


def test_campaign_budget_includes_both_posttraining_controls(inputs):
    config_path, prepared, samples, output, config = inputs
    config.update(models=["rnn", "lstm", "gru", "dlinear"], seeds=list(range(6)))
    atomic_json(config_path, config)
    with pytest.raises(ValueError, match="128"):
        module().run_campaign(config_path, prepared, samples, output)
    assert not output.exists()


def test_exact_case_budget_and_maximum_epochs_are_accepted(inputs, probe_double):
    config_path, prepared, samples, output, config = inputs
    config.update(
        models=["rnn", "lstm", "gru", "dlinear"],
        seeds=list(range(8)),
        losses=["mse"],
        epochs=30,
    )
    atomic_json(config_path, config)
    result = module().run_campaign(config_path, prepared, samples, output)
    assert result["completed_runs"] == 128
    for row in result["runs"]:
        if row["stage"] == "posttraining":
            assert row["parent_checkpoint_path"].endswith("/epoch-30.pt")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("key", ["learning_rates", "huber_delta"])
def test_nonfinite_numbers_are_rejected_before_outputs(inputs, value, key):
    config_path, prepared, samples, output, config = inputs
    config[key] = [0.001, value] if key == "learning_rates" else value
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        module().run_campaign(config_path, prepared, samples, output)
    assert not output.exists()


def test_config_read_is_bounded_before_json_parsing(inputs):
    config_path, prepared, samples, output, _ = inputs
    config_path.write_text(" " * (64 * 1024) + "{}")
    with pytest.raises(ValueError, match="64 KiB"):
        module().run_campaign(config_path, prepared, samples, output)
    assert not output.exists()


@pytest.mark.parametrize("source", ["prepared", "samples"])
def test_missing_input_directory_is_rejected_before_outputs(inputs, source):
    config_path, prepared, samples, output, _ = inputs
    missing = config_path.parent / "missing"
    with pytest.raises(ValueError):
        module().run_campaign(
            config_path,
            missing if source == "prepared" else prepared,
            missing if source == "samples" else samples,
            output,
        )
    assert not output.exists()


@pytest.mark.parametrize("text", ['{"schema_version":1,"schema_version":1}', "[]", "{"])
def test_malformed_or_duplicate_json_is_rejected_before_outputs(inputs, text):
    config_path, prepared, samples, output, _ = inputs
    config_path.write_text(text)
    with pytest.raises(ValueError):
        module().run_campaign(config_path, prepared, samples, output)
    assert not output.exists()


@pytest.mark.parametrize("protected", ["prepared", "samples", "dataset"])
@pytest.mark.parametrize("symlink", [False, True])
def test_output_cannot_modify_sources_even_through_symlinks(
    inputs, monkeypatch, protected, symlink
):
    config_path, prepared, samples, output, _ = inputs
    monkeypatch.chdir(config_path.parent)
    dataset = config_path.parent / "dataset"
    dataset.mkdir()
    root = {"prepared": prepared, "samples": samples, "dataset": dataset}[protected]
    if symlink:
        alias = config_path.parent / "alias"
        alias.symlink_to(root, target_is_directory=True)
        output = alias / "campaign"
    else:
        output = root / "campaign"
    with pytest.raises(ValueError, match="origen"):
        module().run_campaign(config_path, prepared, samples, output)
    assert not output.exists()
    assert (prepared / "source.txt").read_text() == "Conservar el origen"


@pytest.mark.parametrize("target", ["directory", "file", "dangling_symlink"])
def test_existing_destination_is_never_reused_or_replaced(inputs, target):
    config_path, prepared, samples, output, _ = inputs
    if target == "directory":
        output.mkdir()
        (output / "sentinel").write_text("Conservar")
    elif target == "file":
        output.write_text("Conservar")
    else:
        output.symlink_to(output.parent / "missing")
    with pytest.raises(ValueError, match="nuevo"):
        module().run_campaign(config_path, prepared, samples, output)
    if target == "directory":
        assert (output / "sentinel").read_text() == "Conservar"
    elif target == "file":
        assert output.read_text() == "Conservar"
    else:
        assert output.is_symlink()


def test_failure_is_recorded_and_propagated_without_starting_later_cases(
    inputs, probe_double, monkeypatch
):
    config_path, prepared, samples, output, _ = inputs
    calls, run = probe_double
    failure = RuntimeError("Fallo controlado de prueba")

    def fail_second(prepared, samples, output, report_path, **options):
        if len(calls) == 1:
            output.mkdir(parents=True)
            (output / "partial.pt").write_bytes(b"test-only-partial")
            raise failure
        return run(prepared, samples, output, report_path, **options)

    monkeypatch.setattr(module(), "run_temporal_probe", fail_second)
    with pytest.raises(RuntimeError) as caught:
        module().run_campaign(config_path, prepared, samples, output)
    assert caught.value is failure
    result = json.loads((output / "summary.json").read_text())
    assert result["status"] == "failed"
    assert result["completed_runs"] == result["failed_runs"] == 1
    assert len(calls) == 1
    assert [row["status"] for row in result["runs"][:3]] == ["completed", "failed", "pending"]
    failed = result["runs"][1]
    assert failed["error_type"] == "RuntimeError" and failed["error"] == str(failure)
    assert (output / failed["output_path"] / "partial.pt").read_bytes() == b"test-only-partial"
    assert failed["elapsed_seconds"] >= 0
    assert (output / result["runs"][0]["checkpoint_path"]).exists()


@pytest.mark.parametrize(
    "fault", ["test", "parent_hash", "prediction_hash", "training_hash", "checkpoint_hash"]
)
def test_unverified_artifact_or_parent_is_not_marked_completed(
    inputs, probe_double, monkeypatch, fault
):
    config_path, prepared, samples, output, _ = inputs
    _, run = probe_double

    def invalid_report(prepared, samples, output, report_path, **options):
        report = run(prepared, samples, output, report_path, **options)
        if fault == "test":
            report["final_test_opened"] = True
        elif fault == "parent_hash" and options.get("initialize_from"):
            report["config"]["initialization"]["sha256"] = "0" * 64
        elif fault == "prediction_hash":
            report["predictions_sha256"] = "0" * 64
        elif fault == "training_hash":
            report["training_predictions_sha256"] = "0" * 64
        elif fault == "checkpoint_hash":
            report["epochs"][-1]["checkpoint_sha256"] = "0" * 64
        atomic_json(report_path, report)
        return report

    monkeypatch.setattr(module(), "run_temporal_probe", invalid_report)
    with pytest.raises(ValueError):
        module().run_campaign(config_path, prepared, samples, output)
    result = json.loads((output / "summary.json").read_text())
    assert result["status"] == "failed" and result["failed_runs"] == 1
    if fault == "test":
        assert result["final_test_opened"] is True
    assert next(row for row in result["runs"] if row["status"] == "failed")["error_type"] == (
        "ValueError"
    )


def test_main_exposes_the_same_campaign_paths(inputs, probe_double, monkeypatch, capsys):
    config_path, prepared, samples, output, _ = inputs
    monkeypatch.setattr(
        "sys.argv",
        [
            "campaign",
            "--config",
            str(config_path),
            "--prepared",
            str(prepared),
            "--samples",
            str(samples),
            "--output",
            str(output),
        ],
    )
    module().main()
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert (output / "summary.json").is_file()
