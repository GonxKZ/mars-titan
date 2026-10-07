"""Exportación de recibos técnicos pequeños, sin entrenamientos ni datos reservados."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from mars_titan.data.storage import sha256

SPEC = importlib.util.spec_from_file_location(
    "export_comparison", Path(__file__).parents[2] / "scripts/export_campaign_comparison.py"
)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def table(path, records):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def receipt(directory, report, name="comparison.json"):
    if "method" in report and name == "comparison.json":
        for field, filename in (
            ("overall", "methods.csv"),
            ("by_fold", "folds.csv"),
            ("intervals", "intervals.csv"),
        ):
            with (directory / filename).open() as stream:
                report[field] = list(csv.DictReader(stream))
    report["artifacts"] = {path.name: sha256(path) for path in directory.glob("*.csv")}
    (directory / name).write_text(json.dumps(report), encoding="utf-8")


def campaign(tmp_path, count=4, *, legacy=False):
    directory = tmp_path / "predictive"
    directory.mkdir()
    folds, cases, intervals, aggregates = [], [], [], []
    for index in range(count):
        month = 13 - count + index
        end = f"2023-{month + 1:02d}-01" if month < 12 else "2024-01-01"
        fold = dict(
            id=f"fold-{index:03d}",
            counts=dict(calibration=20, evaluation=20),
            windows={
                "calibration": [f"2023-{month - 1:02d}-01", f"2023-{month:02d}-01"],
                "evaluation": [f"2023-{month:02d}-01", end],
            },
            manifest_sha256=digest(str(index)),
        )
        folds.append(fold)
        for family in ("rnn", "lstm", "gru", "dlinear"):
            for method in ("reference", "neural_mae", "neural_mse"):
                identifier = f"{family}-{method}"
                for partition in ("calibration", "evaluation"):
                    row = dict(
                        fold=fold["id"],
                        id=identifier,
                        partition=partition,
                        stage="reference",
                        phase="finalist",
                        family=family,
                        method=method,
                        seed=42,
                        included=True,
                        primary="continuous",
                        parent_id=None,
                        samples=20,
                        sessions=10,
                        predictions_sha256=digest(f"{index}/{identifier}/{partition}"),
                    )
                    if not legacy:
                        row.update(
                            checkpoint_sha256=digest(f"{index}/{identifier}"),
                            source_report_sha256=digest(f"report/{index}/{identifier}"),
                        )
                    cases.append(row)
                    aggregates.append(
                        dict(
                            fold=fold["id"],
                            partition=partition,
                            family=family,
                            method=method,
                            models=1,
                            sessions=10,
                        )
                    )
                intervals.append(
                    dict(
                        fold=fold["id"],
                        partition="evaluation",
                        family=family,
                        method=method,
                        reference="zero",
                        models=1,
                        block_length=5,
                        sessions=10,
                        repetitions=37,
                        seed=42,
                        estimate=-0.0002 - index * 0.00001,
                        lower=-0.0006,
                        upper=0.0001,
                        reason=None,
                    )
                )
    report = dict(
        schema_version=1,
        status="completed",
        final_test_opened=False,
        provenance=dict(
            reference_sha256=digest("reference"),
            completion_sha256=digest("done"),
            final_test_opened=False,
            domain="real",
            folds=folds,
        ),
        counts=dict(models=len(cases) // 2, prediction_files=len(cases)),
        method=dict(confidence=0.95, repetitions=37, block_lengths=[5], seed=42),
        versions={},
        analysis_source_sha256={},
        metadata=[
            dict(fold=row["fold"], id=row["id"], metadata=dict(case=dict(seed=row["seed"])))
            for row in cases
            if row["partition"] == "evaluation"
        ],
    )
    table(directory / "cases.csv", cases)
    table(directory / "intervals.csv", intervals)
    table(directory / "folds.csv", aggregates)
    table(
        directory / "methods.csv",
        [
            dict(partition=partition, family=family, method=method, folds=count, models=count)
            for partition in ("calibration", "evaluation")
            for family in ("rnn", "lstm", "gru", "dlinear")
            for method in ("reference", "neural_mae", "neural_mse")
        ],
    )
    receipt(directory, report)
    return directory, report


def joint_campaign(tmp_path, count=4):
    directory, report = campaign(tmp_path, count)
    report["method"].update(market_stratification="separate_markets", markets=["CN", "US"])
    for fold in report["provenance"]["folds"]:
        fold["market_counts"] = {market: dict(fold["counts"]) for market in ("US", "CN")}
        fold["markets"] = ["CN", "US"]
        fold["counts"] = {part: 40 for part in fold["counts"]}
    for filename in ("cases.csv", "methods.csv", "folds.csv", "intervals.csv"):
        records = list(csv.DictReader((directory / filename).open()))
        updated = []
        for row in records:
            if filename == "cases.csv":
                updated.append(
                    dict(
                        row,
                        samples=40,
                        sessions=18,
                        samples_US=20,
                        samples_CN=20,
                        sessions_US=10,
                        sessions_CN=8,
                    )
                )
            else:
                for market in ("CN", "US"):
                    copy = dict(row, market=market)
                    if "sessions" in copy:
                        copy["sessions"] = 8 if market == "CN" else 10
                    updated.append(copy)
        table(directory / filename, updated)
    receipt(directory, report)
    return directory, report


def test_joint_export_preserves_states_and_renders_each_market(tmp_path):
    directory, _ = joint_campaign(tmp_path)
    output = tmp_path / "export"
    exporter.main(["--predictive", str(directory), "--output", str(output)])
    source = list(csv.DictReader((directory / "cases.csv").open()))
    actual = list(csv.DictReader((output / "predictive-cases.csv").open()))
    assert actual == source
    svg = (output / "predictive-periods.svg").read_text()
    assert "CN · RNN" in svg and "US · RNN" in svg
    assert "8 sesiones" in svg and "10 sesiones" in svg


def test_joint_export_rejects_lost_or_changed_market_counts_before_publication(tmp_path):
    directory, report = joint_campaign(tmp_path)
    records = list(csv.DictReader((directory / "cases.csv").open()))
    records[0]["samples_CN"] = "19"
    table(directory / "cases.csv", records)
    receipt(directory, report)
    output = tmp_path / "export"
    with pytest.raises(ValueError):
        exporter.main(["--predictive", str(directory), "--output", str(output)])
    assert not output.exists()


@pytest.mark.parametrize(
    "fault", ["missing_marker", "missing_market_list", "incomplete_market_list"]
)
def test_joint_export_requires_market_stratification_matching_provenance(tmp_path, fault):
    if fault == "missing_marker":
        directory, report = campaign(tmp_path)
        for fold in report["provenance"]["folds"]:
            fold["markets"] = ["CN", "US"]
            fold["market_counts"] = {
                market: {part: count // 2 for part, count in fold["counts"].items()}
                for market in ("US", "CN")
            }
    else:
        directory, report = joint_campaign(tmp_path)
        fold = report["provenance"]["folds"][0]
        if fault == "missing_market_list":
            fold.pop("markets")
        else:
            fold["markets"] = ["US"]
    receipt(directory, report)
    output = tmp_path / "export"
    with pytest.raises(ValueError):
        exporter.main(["--predictive", str(directory), "--output", str(output)])
    assert not output.exists()


def test_joint_reliability_links_each_market_to_the_same_frozen_states(tmp_path):
    from mars_titan.evaluation.campaign_comparison import compare_campaigns
    from mars_titan.evaluation.campaign_reliability import evaluate_campaign_reliability
    from tests.evaluation.test_comparison_sources import Campaign

    source = Campaign(tmp_path / "sources", materialize=True, matching=True, markets=("US", "CN"))
    comparison = tmp_path / "comparison"
    reliability = tmp_path / "reliability"
    compare_campaigns(source.reference, source.completion, comparison, repetitions=100)
    evaluate_campaign_reliability(source.reference, source.completion, reliability)
    predictive, _, tables = exporter.load_source("predictive", comparison, tmp_path / "export")
    diagnostic, _, diagnoses = exporter.load_source("reliability", reliability, tmp_path / "export")
    cases = exporter.validate_predictive(
        predictive, tables, exporter.windows(predictive["provenance"])
    )
    exporter.validate_reliability(diagnostic, diagnoses["cases.csv"], predictive, cases)
    assert len(cases) == 16 and len(diagnoses["cases.csv"]) == 48
    assert {row["market"] for row in diagnoses["cases.csv"]} == {"US", "CN"}


def reliability(tmp_path, predictive, report):
    directory = tmp_path / "reliability"
    directory.mkdir()
    cases = list(csv.DictReader((predictive / "cases.csv").open()))
    result = []
    for calibration, evaluation in zip(cases[::2], cases[1::2], strict=True):
        row = {
            name: evaluation[name]
            for name in (
                "fold",
                "id",
                "stage",
                "phase",
                "family",
                "method",
                "seed",
                "included",
                "primary",
                "parent_id",
                "checkpoint_sha256",
            )
        }
        fold = next(fold for fold in report["provenance"]["folds"] if fold["id"] == row["fold"])
        for partition, case in (("calibration", calibration), ("evaluation", evaluation)):
            row.update(
                {
                    f"{partition}_{name}": case[name]
                    for name in (
                        "predictions_sha256",
                        "source_report_sha256",
                        "samples",
                        "sessions",
                    )
                }
            )
            row[f"{partition}_start"], row[f"{partition}_end"] = fold["windows"][partition]
        for confidence in (None, 0.9, 0.95):
            result.append(
                row
                | dict(
                    prediction_kind="point" if confidence is None else "interval",
                    confidence=confidence,
                    coverage_guaranteed=False,
                    target_kind="residual_return",
                    conditional_accuracy=None,
                    calls=0,
                    samples=20,
                    radius=None if confidence is None else 0.1,
                )
            )
    reliability_report = dict(
        schema_version=1,
        kind="frozen_campaign_reliability",
        status="completed",
        final_test_opened=False,
        target_kind="residual_return",
        coverage_guaranteed=False,
        provenance=report["provenance"],
        cases=result,
        counts=dict(
            models=len(cases) // 2,
            prediction_files=len(cases),
            folds=len(report["provenance"]["folds"]),
        ),
        method=dict(
            confidence_levels=[0.9, 0.95],
            calibration_partition="calibration",
            evaluation_partition="evaluation",
            selection="frozen_before_calibration",
            seed_or_fold_pooling=False,
        ),
    )
    table(directory / "cases.csv", result)
    receipt(directory, reliability_report, "reliability.json")
    return directory, reliability_report


def financial_campaign(tmp_path):
    directory = tmp_path / "financial"
    directory.mkdir()
    variants = (
        "cash",
        "hold_initial",
        "rebalance_50",
        "rebalance_100",
        "ppo",
        "double_dqn",
        "ppo_window",
        "ppo_gru",
        "ppo_hmm",
        "ppo_episodic",
        "ppo_episodic_hmm",
        "ppo_recent_aux",
        "ppo_replay_aux",
    )
    records = []
    for family, worlds in (("all", 5), ("signal", 2), ("noise", 3)):
        for index, variant in enumerate(variants):
            control = index < 4
            for seed in (None,) if control else (42, 43):
                records.append(
                    dict(
                        kind="control" if control else "learned",
                        variant=variant,
                        seed=seed,
                        family=family,
                        cost_bps=10,
                        worlds=worlds,
                        mean_net_return=0.01,
                    )
                )
    report = dict(
        schema_version=1,
        status="completed",
        final_test_opened=False,
        domain="synthetic",
        split="audit",
        provenance={},
        counts={},
        population=dict(
            worlds=5,
            worlds_by_family={"signal": 2, "noise": 3},
            policy_seeds=[42, 43],
            cost_bps=[10],
        ),
    )
    table(directory / "aggregates.csv", records)
    table(
        directory / "paired_summary.csv",
        [
            dict(
                variant="ppo",
                control="cash",
                family="all",
                cost_bps=10,
                worlds=5,
                policy_seeds=2,
                delta_net_return=0.01,
            )
        ],
    )
    receipt(directory, report)
    return directory, report


@pytest.mark.parametrize("count", [4, 10])
def test_predictive_only_uses_all_provenance_windows_and_real_artifacts(
    tmp_path, count, monkeypatch
):
    source, report = campaign(tmp_path, count)
    output = tmp_path / "export"
    rendered = []
    original = exporter.save

    def inspect(figure, directory, name):
        rendered.append(figure)
        for ax in figure.axes:
            assert len(ax.get_yticks()) == count
            assert ax.get_ylim() == (count - 0.5, -0.5)
            assert all(
                len(collection.get_offsets()) == count
                for collection in ax.collections
                if type(collection).__name__ == "PathCollection"
            )
        assert "37" in " ".join(text.get_text() for text in figure.texts)
        original(figure, directory, name)

    monkeypatch.setattr(exporter, "save", inspect)
    exporter.main(["--predictive", str(source), "--output", str(output)])
    assert len(rendered) == 1
    evidence = json.loads((output / "evidence.json").read_text())
    assert evidence["predictive"]["provenance"] == report["provenance"]
    assert "financial" not in evidence
    with Image.open(output / "predictive-periods.png") as image:
        assert image.width >= 1500 and image.height >= 1000
        assert image.convert("L").getextrema()[0] < 80
    for name, expected in evidence["artifacts"].items():
        assert sha256(output / name) == expected


def test_legacy_comparison_still_exports_without_reliability(tmp_path):
    source, _ = campaign(tmp_path, legacy=True)
    exporter.main(["--predictive", str(source), "--output", str(tmp_path / "out")])


@pytest.mark.parametrize(
    "corruption", ["wrong_fold", "duplicate", "missing", "temporal", "sealed", "hash"]
)
def test_rejects_inconsistent_windows_and_bytes_without_publishing(tmp_path, corruption):
    source, report = campaign(tmp_path)
    records = list(csv.DictReader((source / "intervals.csv").open()))
    if corruption == "wrong_fold":
        records[0]["fold"] = "fold-010"
    elif corruption == "duplicate":
        records.append(records[0])
    elif corruption == "missing":
        records.pop()
    elif corruption == "temporal":
        report["provenance"]["folds"].reverse()
    elif corruption == "sealed":
        report["provenance"]["folds"][-1]["windows"]["evaluation"][1] = "2024-02-01"
    table(source / "intervals.csv", records)
    receipt(source, report)
    if corruption == "hash":
        with (source / "intervals.csv").open("a") as stream:
            stream.write("\n")
    output = tmp_path / "out"
    with pytest.raises(ValueError):
        exporter.main(["--predictive", str(source), "--output", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("failure", ["csv", "figure", "receipt"])
def test_publication_failure_leaves_no_output_and_allows_retry(tmp_path, monkeypatch, failure):
    source, _ = campaign(tmp_path)
    output = tmp_path / "out"
    with monkeypatch.context() as context:

        def fail(*args, **kwargs):
            raise OSError("corte simulado")

        context.setattr(
            exporter,
            {"csv": "write_csv", "figure": "save", "receipt": "atomic_json"}[failure],
            fail,
        )
        with pytest.raises(OSError, match="corte simulado"):
            exporter.main(["--predictive", str(source), "--output", str(output)])
    assert not output.exists()
    exporter.main(["--predictive", str(source), "--output", str(output)])
    assert (output / "evidence.json").is_file()


def test_reliability_links_exact_models_and_preserves_undefined_metrics(tmp_path):
    source, report = campaign(tmp_path)
    reliable, _ = reliability(tmp_path, source, report)
    output = tmp_path / "out"
    exporter.main(
        ["--predictive", str(source), "--reliability", str(reliable), "--output", str(output)]
    )
    evidence = json.loads((output / "evidence.json").read_text())
    assert evidence["reliability"]["linkage"] == "checkpoint_predictions_and_source_reports"
    records = list(csv.DictReader((output / "reliability-cases.csv").open()))
    assert len(records) == 144
    assert all(row["conditional_accuracy"] == "" for row in records)


@pytest.mark.parametrize(
    "corruption",
    [
        "reference",
        "completion",
        "checkpoint",
        "prediction",
        "source_report",
        "window",
        "model",
        "duplicate",
        "missing",
        "summary",
        "target",
        "guarantee",
        "legacy",
    ],
)
def test_reliability_rejects_cross_campaign_or_damaged_cases(tmp_path, corruption):
    source, report = campaign(tmp_path)
    reliable, reliability_report = reliability(tmp_path, source, report)
    record = reliability_report["cases"][0]
    if corruption in {"reference", "completion"}:
        reliability_report["provenance"][f"{corruption}_sha256"] = digest("otra campaña")
    elif corruption == "checkpoint":
        record["checkpoint_sha256"] = digest("otro checkpoint")
    elif corruption in {"prediction", "source_report"}:
        field = "predictions" if corruption == "prediction" else "source_report"
        record[f"evaluation_{field}_sha256"] = digest("otra fuente")
    elif corruption == "window":
        record["evaluation_end"] = "2024-02-01"
    elif corruption == "model":
        record["id"] = "otro modelo"
    elif corruption == "duplicate":
        reliability_report["cases"].append(record)
    elif corruption == "missing":
        reliability_report["cases"].pop()
    elif corruption == "target":
        reliability_report["target_kind"] = "stock_return"
    elif corruption == "guarantee":
        record["coverage_guaranteed"] = True
    elif corruption == "legacy":
        records = list(csv.DictReader((source / "cases.csv").open()))
        for row in records:
            del row["checkpoint_sha256"]
        table(source / "cases.csv", records)
        receipt(source, report)
    table(reliable / "cases.csv", reliability_report["cases"])
    if corruption == "summary":
        record["calls"] = 1
    receipt(reliable, reliability_report, "reliability.json")
    output = tmp_path / "out"
    with pytest.raises(ValueError):
        exporter.main(
            ["--predictive", str(source), "--reliability", str(reliable), "--output", str(output)]
        )
    assert not output.exists()


def test_financial_uses_declared_worlds_and_seeds_without_exporting_world_pairs(tmp_path):
    source, _ = campaign(tmp_path)
    financial, _ = financial_campaign(tmp_path)
    output = tmp_path / "out"
    exporter.main(
        ["--predictive", str(source), "--financial", str(financial), "--output", str(output)]
    )
    assert (output / "financial-scenarios.svg").is_file()
    assert not (output / "financial-paired_worlds.csv").exists()


@pytest.mark.parametrize("corruption", ["duplicate", "model", "fold_summary", "method_summary"])
def test_rejects_predictive_tables_with_inconsistent_membership(tmp_path, corruption):
    source, report = campaign(tmp_path)
    name = {"fold_summary": "folds.csv", "method_summary": "methods.csv"}.get(
        corruption, "cases.csv"
    )
    records = list(csv.DictReader((source / name).open()))
    if corruption == "duplicate":
        records.append(records[0])
    elif corruption == "model":
        records[0]["id"] = "otro modelo"
    else:
        records[0]["models"] = "2"
    table(source / name, records)
    receipt(source, report)
    with pytest.raises(ValueError):
        exporter.main(["--predictive", str(source), "--output", str(tmp_path / "out")])


@pytest.mark.parametrize(
    "corruption",
    ["worlds", "population", "seed", "duplicate", "family", "cost", "nan", "domain", "pair"],
)
def test_rejects_financial_population_or_pair_mismatch(tmp_path, corruption):
    source, _ = campaign(tmp_path)
    financial, report = financial_campaign(tmp_path)
    records = list(csv.DictReader((financial / "aggregates.csv").open()))
    if corruption == "worlds":
        records[0]["worlds"] = "512"
    elif corruption == "population":
        report["population"]["worlds"] = 512
    elif corruption == "seed":
        records[4]["seed"] = "99"
    elif corruption == "duplicate":
        records.append(records[0])
    elif corruption == "family":
        records[-1]["family"] = "otra"
    elif corruption == "cost":
        records[-1]["cost_bps"] = "25"
    elif corruption == "nan":
        records[0]["mean_net_return"] = "nan"
    elif corruption == "domain":
        report["domain"] = "real"
    elif corruption == "pair":
        pairs = list(csv.DictReader((financial / "paired_summary.csv").open()))
        pairs[0]["variant"] = "desconocida"
        table(financial / "paired_summary.csv", pairs)
    table(financial / "aggregates.csv", records)
    receipt(financial, report)
    with pytest.raises(ValueError):
        exporter.main(
            [
                "--predictive",
                str(source),
                "--financial",
                str(financial),
                "--output",
                str(tmp_path / "out"),
            ]
        )


@pytest.mark.parametrize("limit", ["bytes", "rows"])
def test_csv_reads_are_bounded(tmp_path, monkeypatch, limit):
    source, _ = campaign(tmp_path)
    monkeypatch.setattr(exporter, "MAX_CSV_BYTES" if limit == "bytes" else "MAX_ROWS", 1)
    with pytest.raises(ValueError):
        exporter.main(["--predictive", str(source), "--output", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_publish_race_preserves_the_other_output(tmp_path, monkeypatch):
    source, _ = campaign(tmp_path)
    output = tmp_path / "out"
    original = exporter._publish_directory

    def raced(stage, destination):
        destination.mkdir()
        (destination / "existing").write_text("otra ejecución")
        original(stage, destination)

    monkeypatch.setattr(exporter, "_publish_directory", raced)
    with pytest.raises(OSError):
        exporter.main(["--predictive", str(source), "--output", str(output)])
    assert [path.name for path in output.iterdir()] == ["existing"]
    assert (output / "existing").read_text() == "otra ejecución"
    assert not list(tmp_path.glob(".campaign-export-*"))


@pytest.mark.parametrize("corruption", ["duplicate_header", "extra_cell", "empty", "symlink"])
def test_rejects_invalid_csv_structure_and_links(tmp_path, corruption):
    source, report = campaign(tmp_path)
    path = source / "methods.csv"
    if corruption == "duplicate_header":
        path.write_text("method,method\na,b\n")
    elif corruption == "extra_cell":
        path.write_text("method\na,b\n")
    elif corruption == "empty":
        path.write_text("method\n")
    else:
        saved = tmp_path / "saved.csv"
        path.rename(saved)
        path.symlink_to(saved)
    receipt(source, report)
    with pytest.raises(ValueError):
        exporter.main(["--predictive", str(source), "--output", str(tmp_path / "out")])


def test_existing_destination_is_preserved(tmp_path):
    source, _ = campaign(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    (output / "evidence.json").write_text("existente")
    with pytest.raises(SystemExit):
        exporter.main(["--predictive", str(source), "--output", str(output)])
    assert (output / "evidence.json").read_text() == "existente"


@pytest.mark.parametrize("damage", ["bounds", "inverted", "nan", "repetitions"])
def test_interval_errors_are_rejected_before_publication(tmp_path, damage):
    source, report = campaign(tmp_path)
    records = list(csv.DictReader((source / "intervals.csv").open()))
    if damage == "bounds":
        records[0]["lower"] = ""
    elif damage == "inverted":
        records[0]["lower"] = "1"
    elif damage == "nan":
        records[0]["estimate"] = "nan"
    else:
        records[0]["repetitions"] = "2000"
    table(source / "intervals.csv", records)
    receipt(source, report)
    with pytest.raises(ValueError):
        exporter.main(["--predictive", str(source), "--output", str(tmp_path / "out")])


def test_undefined_and_asymmetric_intervals_are_not_replaced(tmp_path, monkeypatch):
    source, report = campaign(tmp_path)
    records = list(csv.DictReader((source / "intervals.csv").open()))
    records[0].update(lower="", upper="", reason="Intervalo indefinido")
    records[12].update(estimate="0.001", lower="-0.0006", upper="0.0001")
    table(source / "intervals.csv", records)
    receipt(source, report)
    original = exporter.save

    def inspect(figure, directory, name):
        segments = figure.axes[0].collections[0].get_segments()
        assert len(segments) == 3
        assert segments[0][:, 0].tolist() == pytest.approx([-6.0, 1.0])
        assert figure.axes[0].collections[1].get_offsets()[1, 0] == 10.0
        original(figure, directory, name)

    monkeypatch.setattr(exporter, "save", inspect)
    exporter.main(["--predictive", str(source), "--output", str(tmp_path / "out")])


@pytest.mark.parametrize("field", ["artifact", "reference", "completion"])
def test_missing_hashes_cannot_disable_verification(tmp_path, field):
    source, report = campaign(tmp_path)
    if field == "artifact":
        report["artifacts"]["cases.csv"] = None
    else:
        report["provenance"][f"{field}_sha256"] = None
    (source / "comparison.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        exporter.main(["--predictive", str(source), "--output", str(tmp_path / "out")])


@pytest.mark.parametrize(
    "field", ["seed", "confidence", "block_lengths", "intervals", "by_fold", "overall"]
)
def test_predictive_summary_and_resampling_metadata_are_checked(tmp_path, field):
    source, report = campaign(tmp_path)
    if field == "seed":
        report["method"][field] = 43
    elif field == "confidence":
        report["method"][field] = 2
    elif field == "block_lengths":
        report["method"][field] = [1]
    else:
        report[field][0]["models"] = "999"
    (source / "comparison.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        exporter.main(["--predictive", str(source), "--output", str(tmp_path / "out")])


@pytest.mark.parametrize("field", ["samples", "seed"])
def test_case_samples_and_seed_match_the_frozen_receipt(tmp_path, field):
    source, report = campaign(tmp_path)
    records = list(csv.DictReader((source / "cases.csv").open()))
    if field == "samples":
        records[0]["samples"] = "19"
    else:
        records[0]["seed"] = records[1]["seed"] = "912345"
    table(source / "cases.csv", records)
    receipt(source, report)
    with pytest.raises(ValueError):
        exporter.main(["--predictive", str(source), "--output", str(tmp_path / "out")])


def test_incomplete_financial_population_is_rejected_before_cartesian_expansion():
    class CountedVariant(str):
        calls = 0

        def __hash__(self):
            self.calls += 1
            assert self.calls <= 20, "Se expandió una población incompatible con una sola fila"
            return super().__hash__()

    report = dict(
        domain="synthetic",
        split="audit",
        population=dict(
            worlds=20,
            worlds_by_family={f"f{index}": 1 for index in range(20)},
            policy_seeds=[42],
            cost_bps=list(range(20)),
        ),
    )
    tables = {
        "aggregates.csv": [
            dict(
                kind="control",
                variant=CountedVariant("cash"),
                family="all",
                seed="",
                cost_bps="10",
                worlds="20",
                mean_net_return="0",
            )
        ],
        "paired_summary.csv": [],
    }
    with pytest.raises(ValueError, match="Faltan agregados"):
        exporter.validate_financial(report, tables)
