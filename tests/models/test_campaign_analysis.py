"""Análisis de artefactos pequeños con poblaciones y errores conocidos."""

import importlib
import json
from datetime import UTC, datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import atomic_json, sha256


def module():
    try:
        return importlib.import_module("mars_titan.models.baselines.analysis")
    except ModuleNotFoundError:
        pytest.fail("Falta el análisis de campañas de referencias")


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / "campaign"
    plan = {
        "models": ["rnn"],
        "seeds": [7, 9],
        "losses": ["mse"],
        "learning_rates": [0.001],
        "epochs": 2,
        "training_through": "2022-12-31",
        "validation_year": 2023,
        "final_test_opened": False,
        "posttraining": {
            "base_loss": "mse",
            "base_learning_rate": 0.001,
            "losses": ["mae", "mse"],
            "learning_rate": 0.0001,
            "epochs": 5,
        },
    }
    runs = []
    for seed in (7, 9):
        parent = f"rnn-mse-0.001-s{seed}"
        for stage, loss, scale, mae, mse in (
            (
                "base",
                "mse",
                0.5 if seed == 7 else 0,
                1 / 3 if seed == 7 else 2 / 3,
                1 / 6 if seed == 7 else 2 / 3,
            ),
            ("posttraining", "mae", 0.75, 1 / 6, 1 / 24),
            ("posttraining", "mse", 0.25, 1 / 2, 3 / 8),
        ):
            identifier = parent if stage == "base" else f"rnn-post-{loss}-s{seed}"
            run = {
                "id": identifier,
                "kind": "rnn",
                "loss": loss,
                "seed": seed,
                "learning_rate": 0.001 if stage == "base" else 0.0001,
                "epochs": 2 if stage == "base" else 5,
                "stage": stage,
                "status": "completed",
                "final_test_opened": False,
                "elapsed_seconds": 1.0,
                "output_path": f"runs/{identifier}",
                "report_path": f"reports/{identifier}.json",
                "samples": {"train": 6, "validation": 6},
            }
            output = root / run["output_path"]
            output.mkdir(parents=True)
            checkpoint = output / f"epoch-{run['epochs']}.pt"
            checkpoint.write_bytes(identifier.encode())
            run.update(
                checkpoint_path=checkpoint.relative_to(root).as_posix(),
                checkpoint_sha256=sha256(checkpoint),
            )
            for partition, name in (
                ("train", "training_predictions"),
                ("validation", "predictions"),
            ):
                days = (
                    [(2022, 11, 1), (2022, 12, 1)]
                    if partition == "train"
                    else [(2023, 1, 3), (2023, 2, 1)]
                )
                rows = [
                    {
                        "asset_id": asset,
                        "prediction_at": datetime(*day, tzinfo=UTC),
                        "target": target,
                        "rnn": target * scale,
                        "zero": 0.0,
                    }
                    for day in days
                    for asset, target in (("US/A", -1.0), ("US/B", 0.0), ("US/C", 1.0))
                ]
                path = output / (
                    "training-predictions.parquet"
                    if partition == "train"
                    else "predictions.parquet"
                )
                pq.write_table(pa.Table.from_pylist(rows), path)
                run.update(
                    {
                        f"{name}_path": path.relative_to(root).as_posix(),
                        f"{name}_sha256": sha256(path),
                    }
                )
            initialization = None
            if stage == "posttraining":
                base = next(item for item in runs if item["id"] == parent)
                run.update(
                    parent=parent,
                    parent_checkpoint_path=base["checkpoint_path"],
                    parent_checkpoint_sha256=base["checkpoint_sha256"],
                )
                initialization = {
                    "sha256": base["checkpoint_sha256"],
                    "policy": "weights_only_new_optimizer_and_rng",
                    "source_completed_epochs": 2,
                }
                run["initialization"] = initialization
            config = {
                "kind": "rnn",
                "loss": loss,
                "seed": seed,
                "lr": run["learning_rate"],
                "epochs": run["epochs"],
                "initialization": initialization,
            }
            report = {
                "status": "completed",
                "model": "rnn",
                "config": config,
                "samples": run["samples"],
                "training_cutoff": "2022-12-31",
                "validation_year": 2023,
                "final_test_opened": False,
                "restored_predictions_equal": True,
                "resume_check": {"exact_weights": True},
                "numerics": {"deterministic_algorithms": True},
                "input_hashes": {
                    "data/inputs.parquet": "a" * 64,
                    f"{output}/targets/A-targets.parquet": "c" * 64,
                    "src/model.py": "b" * 64,
                },
                "epochs": [
                    {
                        "epoch": run["epochs"],
                        "checkpoint_sha256": run["checkpoint_sha256"],
                        "train": {"diagnostic_mae": 99.0},
                        "validation": {"diagnostic_mae": 99.0},
                    }
                ],
                "predictions_sha256": run["predictions_sha256"],
                "training_predictions_sha256": run["training_predictions_sha256"],
                "final_training_row_metrics": {
                    "rnn": {"mae": mae, "mse": mse},
                    "zero": {"mae": 2 / 3, "mse": 2 / 3},
                },
                "diagnostic_row_metrics": {
                    "rnn": {"mae": mae, "mse": mse},
                    "zero": {"mae": 2 / 3, "mse": 2 / 3},
                },
            }
            atomic_json(root / run["report_path"], report)
            run["report_sha256"] = sha256(root / run["report_path"])
            runs.append(run)
    summary = {
        "schema_version": 1,
        "status": "completed",
        "planned_runs": len(runs),
        "completed_runs": len(runs),
        "failed_runs": 0,
        "final_test_opened": False,
        "config_sha256": "d" * 64,
        "resolved_config": plan,
        "runs": runs,
    }
    path = root / "summary.json"
    atomic_json(path, summary)
    return path


def analyze(campaign):
    return module().analyze_campaign(campaign, block_length=1, repetitions=20, seed=7)


def edit_summary(campaign, edit):
    summary = json.loads(campaign.read_text())
    edit(summary)
    atomic_json(campaign, summary)


def edit_report(campaign, index, edit):
    summary = json.loads(campaign.read_text())
    run = summary["runs"][index]
    path = campaign.parent / run["report_path"]
    report = json.loads(path.read_text())
    edit(report)
    atomic_json(path, report)
    run["report_sha256"] = sha256(path)
    atomic_json(campaign, summary)


def edit_predictions(campaign, index, partition, edit):
    summary = json.loads(campaign.read_text())
    run = summary["runs"][index]
    name = "training_predictions" if partition == "train" else "predictions"
    path = campaign.parent / run[f"{name}_path"]
    frame = pq.read_table(path).to_pandas()
    pq.write_table(pa.Table.from_pandas(edit(frame), preserve_index=False), path)
    digest = sha256(path)
    run[f"{name}_sha256"] = digest
    report_path = campaign.parent / run["report_path"]
    report = json.loads(report_path.read_text())
    report[f"{name}_sha256"] = digest
    atomic_json(report_path, report)
    run["report_sha256"] = sha256(report_path)
    atomic_json(campaign, summary)


def test_analysis_recomputes_final_weights_errors_and_keeps_one_common_population(campaign):
    result = analyze(campaign)

    assert result["campaign_sha256"] == sha256(campaign)
    assert result["population"]["train"]["n"] == result["population"]["validation"]["n"] == 6
    assert result["population"]["validation"]["assets"] == ["US/A", "US/B", "US/C"]
    assert result["population"]["validation"]["n_dates"] == 2
    run = next(item for item in result["runs"] if item["id"] == "rnn-mse-0.001-s7")
    for partition in ("train", "validation"):
        assert run[partition]["metrics"]["mae"] == pytest.approx(1 / 3)
        assert run[partition]["metrics"]["mse"] == pytest.approx(1 / 6)
        assert run[partition]["by_asset"]["US/A"]["mae"] == 0.5
        assert run[partition]["by_asset"]["US/B"]["mae"] == 0
    assert run["validation"]["by_month"]["2023-01"]["n"] == 3
    assert run["validation"]["by_month"]["2023-02"]["mae"] == pytest.approx(1 / 3)
    assert run["validation"]["bootstrap_vs_zero"]["estimate"] == pytest.approx(-1 / 3)
    assert result["final_test_opened"] is False
    assert result["cohort_policy"] == "one_campaign_common_population"
    encoded = json.dumps(result, allow_nan=False, ensure_ascii=False).encode()
    assert len(encoded) < 5 * 1024**2
    assert str(campaign.parent.parent).encode() not in encoded
    assert all(item["value"] is None and item["reason"] for item in result["not_computed"].values())


def test_cross_sectional_correlations_use_dates_with_three_distinct_nonconstant_assets(campaign):
    result = analyze(campaign)
    good = next(item for item in result["runs"] if item["id"] == "rnn-mse-0.001-s7")
    constant = next(item for item in result["runs"] if item["id"] == "rnn-mse-0.001-s9")

    assert good["validation"]["cross_sectional"]["eligible_dates_n"] == 2
    assert good["validation"]["cross_sectional"]["pearson"]["mean"] == pytest.approx(1)
    assert good["validation"]["cross_sectional"]["spearman"]["mean"] == pytest.approx(1)
    assert constant["validation"]["cross_sectional"]["eligible_dates_n"] == 0
    assert constant["validation"]["cross_sectional"]["excluded_constant_n"] == 2
    assert constant["validation"]["cross_sectional"]["pearson"] is None


def test_seed_ranges_are_descriptive_and_do_not_merge_configuration_variants(campaign):
    groups = analyze(campaign)["seed_groups"]
    base = next(group for group in groups if group["stage"] == "base")

    assert len(groups) == 3
    assert base["seeds"] == [7, 9] and base["n_seeds"] == 2
    assert base["validation"]["mae"] == {
        "mean": pytest.approx(0.5),
        "std": pytest.approx(np.sqrt(2) / 6),
        "min": pytest.approx(1 / 3),
        "max": pytest.approx(2 / 3),
    }
    assert base["range_kind"] == "observed_seed_range_not_confidence_interval"


def test_posttraining_controls_are_paired_with_their_own_parent_and_seed(campaign):
    pairs = analyze(campaign)["posttraining_pairs"]

    assert len(pairs) == 2
    first = next(pair for pair in pairs if pair["seed"] == 7)
    assert first["parent"] == "rnn-mse-0.001-s7"
    assert first["post_mae"] == "rnn-post-mae-s7"
    assert first["post_mse"] == "rnn-post-mse-s7"
    assert first["validation"]["mae_vs_mse"]["paired_mae_difference"] == pytest.approx(-1 / 3)
    assert first["validation"]["mae_vs_parent"]["paired_mae_difference"] == pytest.approx(-1 / 6)
    assert first["validation"]["mse_vs_parent"]["paired_mae_difference"] == pytest.approx(1 / 6)
    assert first["validation"]["mae_vs_mse"]["bootstrap"]["interval"] == {
        "lower": pytest.approx(-1 / 3),
        "upper": pytest.approx(-1 / 3),
    }


@pytest.mark.parametrize("field", ["report", "checkpoint", "predictions", "training_predictions"])
def test_modified_artifacts_are_rejected_before_analysis(campaign, field):
    summary = json.loads(campaign.read_text())
    path = campaign.parent / summary["runs"][0][f"{field}_path"]
    path.write_bytes(path.read_bytes() + b"modified")

    with pytest.raises(ValueError, match="huella"):
        analyze(campaign)


@pytest.mark.parametrize("fault", ["absolute", "escape", "symlink"])
def test_artifact_paths_cannot_escape_the_campaign(campaign, fault):
    outside = campaign.parent.parent / "outside.pt"
    outside.write_bytes(b"outside")
    if fault == "symlink":
        (campaign.parent / "alias.pt").symlink_to(outside)
    path = {"absolute": str(outside), "escape": "../outside.pt", "symlink": "alias.pt"}[fault]
    edit_summary(campaign, lambda summary: summary["runs"][1].update(checkpoint_path=path))

    with pytest.raises(ValueError, match="ruta|campaña"):
        analyze(campaign)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "running"),
        ("final_test_opened", True),
        ("completed_runs", 5),
        ("planned_runs", 7),
        ("failed_runs", 1),
        ("schema_version", True),
    ],
)
def test_incomplete_or_changed_summary_is_rejected(campaign, field, value):
    edit_summary(campaign, lambda summary: summary.update({field: value}))
    with pytest.raises(ValueError):
        analyze(campaign)


@pytest.mark.parametrize("fault", ["duplicate_id", "missing_case", "changed_parent", "parent_hash"])
def test_identity_or_grid_mismatches_are_rejected(campaign, fault):
    def change(summary):
        if fault == "duplicate_id":
            summary["runs"][1]["id"] = summary["runs"][0]["id"]
        elif fault == "missing_case":
            summary["runs"].pop()
            summary.update(planned_runs=5, completed_runs=5)
        elif fault == "changed_parent":
            summary["runs"][1]["parent"] = "rnn-mse-0.001-s9"
        else:
            summary["runs"][1]["parent_checkpoint_sha256"] = "0" * 64

    edit_summary(campaign, change)
    with pytest.raises(ValueError):
        analyze(campaign)


@pytest.mark.parametrize(
    "fault", ["model", "seed", "samples", "metrics", "input_hash", "test", "epoch"]
)
def test_report_must_match_its_receipt_and_recomputed_values(campaign, fault):
    def change(report):
        if fault == "model":
            report["model"] = "lstm"
        elif fault == "seed":
            report["config"]["seed"] = 1
        elif fault == "samples":
            report["samples"]["train"] = 5
        elif fault == "metrics":
            report["final_training_row_metrics"]["rnn"]["mae"] = 99
        elif fault == "input_hash":
            report["input_hashes"]["data/inputs.parquet"] = "0" * 64
        elif fault == "epoch":
            report["epochs"][-1]["epoch"] = 1
        else:
            report["final_test_opened"] = True

    edit_report(campaign, 1, change)
    with pytest.raises(ValueError):
        analyze(campaign)


@pytest.mark.parametrize("partition", ["train", "validation"])
@pytest.mark.parametrize(
    "fault", ["nonfinite", "zero", "duplicate", "target", "population", "date", "missing_column"]
)
def test_prediction_contract_rejects_invalid_or_unpaired_rows(campaign, partition, fault):
    def change(frame):
        if fault == "nonfinite":
            frame.loc[0, "rnn"] = np.nan
        elif fault == "zero":
            frame.loc[0, "zero"] = 1
        elif fault == "duplicate":
            frame.loc[1] = frame.loc[0]
        elif fault == "target":
            frame.loc[0, "target"] = 42
        elif fault == "population":
            frame.loc[0, "asset_id"] = "US/D"
        elif fault == "date":
            frame.loc[0, "prediction_at"] = datetime(2024, 1, 1, tzinfo=UTC)
        else:
            return frame.drop(columns="asset_id")
        return frame

    edit_predictions(campaign, 1, partition, change)
    with pytest.raises(ValueError):
        analyze(campaign)


def test_row_order_does_not_change_errors_or_paired_comparisons(campaign):
    expected = analyze(campaign)
    edit_predictions(campaign, 1, "validation", lambda frame: frame.iloc[::-1])
    actual = analyze(campaign)

    assert actual["posttraining_pairs"] == expected["posttraining_pairs"]
    assert actual["seed_groups"] == expected["seed_groups"]


def test_target_identity_is_checked_even_when_reported_errors_stay_unchanged(campaign):
    def change(frame):
        frame.loc[0, ["target", "rnn"]] *= -1
        return frame

    edit_predictions(campaign, 1, "validation", change)
    with pytest.raises(ValueError, match="objetivos|población"):
        analyze(campaign)


def test_cli_writes_a_new_small_analysis_without_modifying_the_campaign(
    campaign, monkeypatch, capsys
):
    destination = campaign.parent.parent / "analysis.json"
    before = sha256(campaign)
    monkeypatch.setattr(
        "sys.argv",
        [
            "analysis",
            "--campaign",
            str(campaign),
            "--output",
            str(destination),
            "--block-length",
            "1",
            "--repetitions",
            "20",
            "--seed",
            "7",
        ],
    )
    module().main()

    assert sha256(campaign) == before
    assert json.loads(destination.read_text())["campaign_sha256"] == before
    assert destination.stat().st_size < 5 * 1024**2
    assert "6" in capsys.readouterr().out


@pytest.mark.parametrize("destination", ["existing", "inside", "symlink"])
def test_cli_refuses_existing_outputs_and_campaign_artifact_destinations(
    campaign, monkeypatch, destination
):
    path = campaign.parent.parent / "analysis.json"
    if destination == "existing":
        path.write_text("Conservar")
    elif destination == "inside":
        path = campaign.parent / "analysis.json"
    else:
        path.symlink_to(campaign)
    monkeypatch.setattr(
        "sys.argv", ["analysis", "--campaign", str(campaign), "--output", str(path)]
    )

    with pytest.raises(ValueError):
        module().main()
    if destination == "existing":
        assert path.read_text() == "Conservar"


def test_cross_sectional_correlations_are_not_pooled_correlations(campaign):
    for index in range(6):

        def rows(frame):
            frame["target"] = [-1, 0, 1, 9, 10, 11]
            frame["rnn"] = [1, 0, -1, 11, 10, 9]
            return frame

        edit_predictions(campaign, index, "validation", rows)
        edit_report(
            campaign,
            index,
            lambda report: report.update(
                diagnostic_row_metrics={
                    "rnn": {"mae": 4 / 3, "mse": 8 / 3},
                    "zero": {"mae": 16 / 3, "mse": 152 / 3},
                }
            ),
        )

    validation = analyze(campaign)["runs"][0]["validation"]

    assert validation["metrics"]["pearson"] == pytest.approx(73 / 77)
    assert validation["cross_sectional"]["pearson"]["mean"] == pytest.approx(-1)
    assert validation["cross_sectional"]["spearman"]["mean"] == pytest.approx(-1)


def test_cross_sectional_dates_with_fewer_than_three_assets_are_counted_but_excluded(campaign):
    for index in range(6):

        def rows(frame):
            frame.loc[5, "prediction_at"] = datetime(2023, 2, 2, tzinfo=UTC)
            return frame

        edit_predictions(campaign, index, "validation", rows)
    result = analyze(campaign)["runs"][0]["validation"]["cross_sectional"]

    assert result["eligible_dates_n"] == 1
    assert result["excluded_insufficient_assets_n"] == 2
    assert result["pearson"]["std"] is None


def test_one_seed_has_no_sample_standard_deviation(campaign):
    def change(summary):
        summary["resolved_config"]["seeds"] = [7]
        summary["runs"] = summary["runs"][:3]
        summary.update(planned_runs=3, completed_runs=3)

    edit_summary(campaign, change)
    base = next(group for group in analyze(campaign)["seed_groups"] if group["stage"] == "base")

    assert base["n_seeds"] == 1
    assert base["validation"]["mae"]["std"] is None
    assert base["validation"]["mae"]["min"] == base["validation"]["mae"]["max"]


def test_duplicate_declared_seeds_are_not_silently_collapsed(campaign):
    edit_summary(campaign, lambda summary: summary["resolved_config"].update(seeds=[7, 9, 9]))
    with pytest.raises(ValueError):
        analyze(campaign)


def test_parent_epoch_evidence_must_match_the_final_parent(campaign):
    edit_report(
        campaign,
        1,
        lambda report: report["config"]["initialization"].update(source_completed_epochs=1),
    )
    with pytest.raises(ValueError):
        analyze(campaign)


@pytest.mark.parametrize("content", ["[]", '{"schema_version":1,"schema_version":1}', "{}"])
def test_malformed_summary_contract_is_rejected(campaign, content):
    campaign.write_text(content)
    with pytest.raises(ValueError):
        analyze(campaign)


def test_analysis_refuses_json_outputs_of_five_mebibytes_or_more():
    with pytest.raises(ValueError, match="5 MiB"):
        module()._encoded({"large": "x" * (5 * 1024**2)})


@pytest.mark.parametrize("name", ["prepared", "samples"])
def test_cli_cannot_write_inside_declared_data_inputs(campaign, monkeypatch, name):
    origin = campaign.parent.parent / name
    origin.mkdir()
    edit_summary(campaign, lambda summary: summary.update(inputs={name: f"../{name}"}))
    monkeypatch.setattr(
        "sys.argv",
        ["analysis", "--campaign", str(campaign), "--output", str(origin / "analysis.json")],
    )
    with pytest.raises(ValueError, match="origen"):
        module().main()
    assert not (origin / "analysis.json").exists()


def test_colliding_normalized_target_paths_are_rejected_even_with_equal_hashes(campaign):
    edit_report(
        campaign,
        0,
        lambda report: report["input_hashes"].update(
            {
                "another-run/targets/A-targets.parquet": "c" * 64,
            }
        ),
    )
    with pytest.raises(ValueError, match="fuente|colisión"):
        analyze(campaign)


@pytest.mark.parametrize("changed", [False, True])
def test_optional_campaign_data_provenance_must_match_verified_reports(campaign, changed):
    provenance = {
        "input_hashes": {"data/inputs.parquet": "a" * 64, "targets/A-targets.parquet": "c" * 64},
        "samples": {"train": 6, "validation": 5 if changed else 6},
    }
    edit_summary(campaign, lambda summary: summary.update(data_provenance=provenance))
    if changed:
        with pytest.raises(ValueError, match="procedencia|fuentes"):
            analyze(campaign)
    else:
        assert analyze(campaign)["population"]["validation"]["n"] == 6


def test_resource_report_separates_historical_process_peak_from_sampled_run_peak(campaign):
    def change(report):
        report.update(peak_rss_mib=9000.0, total_seconds=12.0)
        final = report["epochs"][-1]
        report["epochs"] = [dict(final, epoch=1), final]
        for index, epoch in enumerate(report["epochs"]):
            for partition, elapsed, rss in (
                ("train", 2.0 + index, 100.0 + index * 50),
                ("validation", 1.0, 200.0 - index * 20),
            ):
                epoch[partition] = {
                    "sampled_tree_peak": {"rss_mib": rss, "pss_mib": rss / 2},
                    "ram_poll_seconds": 0.25,
                    "elapsed_seconds": elapsed,
                    "samples": 6,
                    "samples_per_second": 6 / elapsed,
                    "step_p50_ms": 2.0,
                    "step_p95_ms": 3.0,
                    "step_p99_ms": 4.0,
                    "latency_scope": "loader_plus_transfer_plus_synchronized_gpu_step",
                }

    edit_report(campaign, 0, change)
    resource = analyze(campaign)["runs"][0]["resources"]

    assert resource["process_lifetime_peak_rss_mib"] == 9000
    assert resource["sampled_run_peak_rss_mib"] == 200
    assert resource["ram_poll_seconds"] == [0.25]
    assert "muestreo" in resource["sampled_memory_limit"]
    assert resource["runner_wall_seconds"] == 12
    assert resource["campaign_case_wall_seconds"] == 1
    assert resource["epochs"]["train"]["elapsed_seconds_total"] == 5
    assert resource["epochs"]["validation"]["elapsed_seconds_total"] == 2
    assert resource["epochs"]["train"]["processed_samples"] == 12
    assert resource["epochs"]["train"]["samples_per_second"] == pytest.approx(12 / 5)
    assert resource["epochs"]["train"]["step_latency_by_epoch"][0]["p99_ms"] == 4
    assert resource["kernel_seconds"] is None and resource["kernel_seconds_reason"]


def test_grid_budget_includes_controls_before_calling_the_cartesian_producer(campaign, monkeypatch):
    edit_summary(campaign, lambda summary: summary["resolved_config"].update(seeds=list(range(64))))
    monkeypatch.setattr(
        module(), "product", lambda *args: pytest.fail("Se invocó el producto cartesiano")
    )

    with pytest.raises(ValueError, match="128"):
        analyze(campaign)


def test_large_json_grid_is_rejected_without_generating_its_millions_of_cases(
    campaign, monkeypatch
):
    producer = module().product
    edit_summary(
        campaign,
        lambda summary: summary["resolved_config"].update(
            models=["rnn", "lstm", "gru", "dlinear"],
            losses=["mse", "mae", "huber"],
            seeds=list(range(2000)),
            learning_rates=[0.001 + index * 1e-7 for index in range(2000)],
        ),
    )
    monkeypatch.setattr(
        module(), "product", lambda *args: pytest.fail("Se invocó el producto cartesiano")
    )

    with pytest.raises(ValueError, match="128"):
        analyze(campaign)
    monkeypatch.setattr(module(), "product", producer)
    with pytest.raises(ValueError, match="128"):
        analyze(campaign)


def test_grid_accepts_exactly_128_cases_including_posttraining_controls(campaign):
    plan = json.loads(campaign.read_text())["resolved_config"]
    plan.update(
        models=["rnn", "lstm", "gru", "dlinear"],
        seeds=[1, 2, 3, 4],
        losses=["mse", "mae", "huber"],
        learning_rates=[0.001, 0.0001],
    )
    cases = []
    for kind in plan["models"]:
        for seed in plan["seeds"]:
            for stage, loss, rate in (
                ("base", "mse", 0.001),
                ("base", "mse", 0.0001),
                ("base", "mae", 0.001),
                ("base", "mae", 0.0001),
                ("base", "huber", 0.001),
                ("base", "huber", 0.0001),
                ("posttraining", "mae", 0.0001),
                ("posttraining", "mse", 0.0001),
            ):
                cases.append(
                    {
                        "id": f"{kind}-{seed}-{stage}-{loss}-{rate}",
                        "stage": stage,
                        "kind": kind,
                        "seed": seed,
                        "loss": loss,
                        "learning_rate": rate,
                    }
                )

    assert len(cases) == 128
    assert module()._check_grid(cases, plan) is None


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("models", ["unknown"]),
        ("models", [["rnn"]]),
        ("seeds", [-1]),
        ("seeds", [2**32]),
        ("seeds", [True]),
        ("seeds", ["7"]),
        ("losses", ["unknown"]),
        ("learning_rates", [0]),
        ("learning_rates", [-0.1]),
        ("learning_rates", [True]),
        ("learning_rates", ["0.001"]),
        ("epochs", True),
        ("epochs", 1),
        ("epochs", 31),
    ],
)
def test_grid_values_are_validated_before_calling_the_cartesian_producer(
    campaign, monkeypatch, key, value
):
    edit_summary(campaign, lambda summary: summary["resolved_config"].update({key: value}))
    monkeypatch.setattr(
        module(), "product", lambda *args: pytest.fail("Se invocó el producto cartesiano")
    )

    with pytest.raises(ValueError):
        analyze(campaign)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("losses", ["mae", "mae"]),
        ("losses", "mae"),
        ("losses", [["mae"], "mse"]),
        ("learning_rate", -0.1),
        ("epochs", True),
        ("base_learning_rate", 0.01),
    ],
)
def test_posttraining_plan_is_validated_before_calling_the_cartesian_producer(
    campaign, monkeypatch, key, value
):
    edit_summary(
        campaign, lambda summary: summary["resolved_config"]["posttraining"].update({key: value})
    )
    monkeypatch.setattr(
        module(), "product", lambda *args: pytest.fail("Se invocó el producto cartesiano")
    )

    with pytest.raises(ValueError):
        analyze(campaign)
