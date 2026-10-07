"""Calibración por checkpoint y evaluación posterior con recibos congelados."""

import copy
import csv
import json
from datetime import UTC, datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256
from mars_titan.evaluation.comparison_sources import predictive_sources
from mars_titan.evaluation.session_metrics import SessionErrors
from tests.evaluation.test_comparison_sources import Campaign, save


def prepared_campaign(root, change=None, *, calibration_samples=20, markets=("US",)):
    campaign = Campaign(root, matching=True, markets=markets)
    campaign.counts.update(calibration=calibration_samples, evaluation=4)
    if len(markets) > 1:
        assert calibration_samples % 4 == 0
        for asset in campaign.manifest["assets"]:
            asset["counts"].update(calibration=calibration_samples // 4, evaluation=1)
    campaign.manifest_hash = save(campaign.manifest_path, campaign.manifest)
    for original in campaign.originals.values():
        if "identity" in original:
            original["identity"]["manifest_sha256"] = campaign.manifest_hash
            if "grid" in original["identity"]:
                original["identity"]["grid"]["source_sha256"] = campaign.manifest_hash
        else:
            original["manifest_sha256"] = campaign.manifest_hash
        if "selection" in original:
            original["selection"]["best_epoch"] = 1
    campaign.publish()
    evaluation = campaign.completion / campaign.fold / "evaluation/summary.json"
    summary = json.loads(evaluation.read_text())
    for identifier, row in summary["runs"].items():
        report_path = evaluation.parent / row["path"]
        report = json.loads(report_path.read_text())
        for partition in ("calibration", "evaluation"):
            count = campaign.counts[partition]
            month = 11 if partition == "calibration" else 12
            values = dict(
                sample_id=[f"{partition}-{index}" for index in range(count)],
                asset_id=[f"US/A{index}" for index in range(count)],
                market=["US"] * count,
                prediction_at=[
                    datetime(2023, month, 2 if index < count - 1 else 3, tzinfo=UTC)
                    for index in range(count)
                ],
                target=[0.1] * count if partition == "calibration" else [-0.1, 0.2, 0.0, 0.5],
                prediction=[0.0] * count if partition == "calibration" else [0.0, 0.3, 0.2, 0.2],
                parent=[0.0] * count,
                zero=[0.0] * count,
            )
            if len(markets) > 1:
                values["market"] = [
                    markets[min(index * len(markets) // count, len(markets) - 1)]
                    for index in range(count)
                ]
                local_rows = [index % (count // len(markets)) for index in range(count)]
                values["asset_id"] = [
                    f"{market}/{'AB'[index % 2]}"
                    for index, market in zip(local_rows, values["market"], strict=True)
                ]
                values["prediction_at"] = [
                    datetime(2023, month, 2 + index // 2, 7 if market == "CN" else 21, tzinfo=UTC)
                    for index, market in zip(local_rows, values["market"], strict=True)
                ]
                if partition == "calibration":
                    values["target"] = [
                        0.1 if market == "US" else 1.0 for market in values["market"]
                    ]
            if change is not None:
                change(identifier, partition, values)
            values["prediction_at"] = pa.array(
                values["prediction_at"], pa.timestamp("us", tz="UTC")
            )
            table = pa.table(values)
            destination = report_path.parent / f"{partition}.parquet"
            pq.write_table(table, destination)
            metrics = {}
            for role in ("prediction", "parent", "zero"):
                errors = SessionErrors()
                errors.update(
                    np.asarray(values["market"]),
                    table["prediction_at"].to_numpy(),
                    np.asarray(values[role]) - values["target"],
                )
                metrics[role] = errors.summary()
            report["predictions"][partition] = dict(
                path=destination.name, sha256=sha256(destination), metrics=metrics
            )
        row["sha256"] = save(report_path, report)
    signature = save(evaluation, summary)
    completion_path = campaign.completion / "summary.json"
    completion = json.loads(completion_path.read_text())
    for stage in completion["stages"]:
        if stage["stage"] == "evaluation":
            stage["sha256"] = signature
    save(completion_path, completion)
    return campaign


def test_pipeline_calibrates_the_same_frozen_checkpoint_and_reports_real_denominators(tmp_path):
    from mars_titan.evaluation.campaign_reliability import evaluate_campaign_reliability

    campaign = prepared_campaign(tmp_path / "sources")
    output = tmp_path / "reliability"
    before = sha256(campaign.completion / "summary.json")
    report = evaluate_campaign_reliability(campaign.reference, campaign.completion, output)
    assert report["status"] == "completed"
    assert report["final_test_opened"] is False
    assert report["target_kind"] == "residual_return"
    assert report["counts"]["models"] == 8
    assert report["counts"]["prediction_files"] == 16
    rows = [row for row in report["cases"] if row["id"] == "reference/search/winner"]
    point = next(row for row in rows if row["prediction_kind"] == "point")
    assert point["samples"] == 4
    assert point["eligible_targets"] == 3
    assert point["zero_targets"] == 1
    assert point["calls"] == 2
    assert point["direction_accuracy"] == pytest.approx(2 / 3)
    assert point["mean_session_direction_accuracy"] == 0.75
    assert point["checkpoint_sha256"] == campaign.originals[point["id"]]["checkpoint"]["sha256"]
    for confidence in (0.9, 0.95):
        interval = next(row for row in rows if row["confidence"] == confidence)
        assert interval["radius"] == 0.1
        assert interval["calibration_samples"] == 20
        assert interval["covered"] == 2
        assert interval["coverage"] == 0.5
        assert interval["mean_session_coverage"] == pytest.approx(1 / 3)
        assert interval["mean_width"] == 0.2
        assert interval["conditional_accuracy"] == 1
    assert sha256(campaign.completion / "summary.json") == before
    assert str(campaign.reference.parent) not in json.dumps(report)
    csv_rows = list(csv.DictReader((output / "cases.csv").open()))
    assert len(csv_rows) == 24
    assert all("market" not in row for row in report["cases"])
    assert "market_pooling" not in report["method"]
    assert sha256(output / "cases.csv") == report["artifacts"]["cases.csv"]
    with pytest.raises(ValueError, match="salida"):
        evaluate_campaign_reliability(campaign.reference, campaign.completion, output)


def test_pairing_rejects_another_checkpoint_or_temporal_block_before_reading_predictions(tmp_path):
    from mars_titan.evaluation.campaign_reliability import _pairs

    campaign = prepared_campaign(tmp_path / "sources")
    rows, _ = predictive_sources(campaign.reference, campaign.completion)
    original = next(row for row in rows if row["partition"] == "evaluation")
    for change in (
        {"checkpoint_sha256": "f" * 64},
        {"bounds": ["2023-11-02", "2023-12-01"]},
        {"bounds": ["2024-01-01", "2024-02-01"]},
        {"partition": "calibration"},
    ):
        changed = [copy.deepcopy(row) for row in rows]
        index = rows.index(original)
        changed[index].update(change)
        with pytest.raises(ValueError):
            _pairs(changed)
    original["source_report_sha256"] = "e" * 64
    assert len(_pairs(rows)) == 8


def test_evaluation_errors_cannot_change_the_calibration_radius(tmp_path):
    from mars_titan.evaluation.campaign_reliability import evaluate_campaign_reliability

    def change(_identifier, partition, values):
        if partition == "evaluation":
            values["target"] = [100.0] * 4

    campaign = prepared_campaign(tmp_path / "sources", change)
    report = evaluate_campaign_reliability(
        campaign.reference, campaign.completion, tmp_path / "output"
    )
    for row in report["cases"]:
        if row["prediction_kind"] == "interval":
            assert row["radius"] == 0.1
            assert row["coverage"] == 0


def test_small_calibration_remains_undefined_in_json_and_csv(tmp_path):
    from mars_titan.evaluation.campaign_reliability import evaluate_campaign_reliability

    campaign = prepared_campaign(tmp_path / "sources", calibration_samples=3)
    output = tmp_path / "output"
    report = evaluate_campaign_reliability(campaign.reference, campaign.completion, output)
    for row in report["cases"]:
        if row["prediction_kind"] == "interval":
            assert row["reason"]
            assert row["coverage"] is row["radius"] is row["calls"] is None
            assert row["mean_session_direction_accuracy"] is None
            assert row["defined_sessions_direction_accuracy"] == 0
    for row in csv.DictReader((output / "cases.csv").open()):
        if row["prediction_kind"] == "interval":
            assert row["coverage"] == row["calls"] == ""


def test_model_specific_target_changes_are_rejected_before_writing_an_output(tmp_path):
    from mars_titan.evaluation.campaign_reliability import evaluate_campaign_reliability

    def change(identifier, partition, values):
        if identifier == "reference/search/winner" and partition == "evaluation":
            values["target"][0] += 1

    campaign = prepared_campaign(tmp_path / "sources", change)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="población|etiquetas"):
        evaluate_campaign_reliability(campaign.reference, campaign.completion, output)
    assert not output.exists()


def test_reusing_calibration_sample_ids_in_evaluation_is_rejected(tmp_path):
    from mars_titan.evaluation.campaign_reliability import evaluate_campaign_reliability

    def change(_identifier, partition, values):
        if partition == "evaluation":
            values["sample_id"][0] = "calibration-0"

    campaign = prepared_campaign(tmp_path / "sources", change)
    with pytest.raises(ValueError, match="muestras"):
        evaluate_campaign_reliability(campaign.reference, campaign.completion, tmp_path / "output")


def test_reserved_targets_are_not_loaded_by_the_reliability_pipeline(tmp_path, monkeypatch):
    from mars_titan.evaluation.campaign_reliability import evaluate_campaign_reliability

    def change(_identifier, partition, values):
        if partition == "evaluation":
            values["prediction_at"] = [datetime(2024, 1, 2, tzinfo=UTC)] * 4

    campaign = prepared_campaign(tmp_path / "sources", change)
    original = pq.ParquetFile

    class ObservedFile:
        def __init__(self, path, **kwargs):
            self.file = original(path, **kwargs)
            self.reserved = path.name == "evaluation.parquet"

        def __getattr__(self, name):
            return getattr(self.file, name)

        def read(self, columns, **kwargs):
            assert not self.reserved or "target" not in columns
            return self.file.read(columns=columns, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", ObservedFile)
    with pytest.raises(ValueError, match="fechas"):
        evaluate_campaign_reliability(campaign.reference, campaign.completion, tmp_path / "output")


def test_cli_reports_completion_and_refuses_writing_inside_the_sources(tmp_path, capsys):
    from mars_titan.evaluation.campaign_reliability import main

    campaign = prepared_campaign(tmp_path / "sources")
    args = ["--reference", str(campaign.reference), "--completion", str(campaign.completion)]
    with pytest.raises(ValueError):
        main([*args, "--output", str(campaign.completion / "output")])
    assert main([*args, "--output", str(tmp_path / "output")]) == 0
    assert "8" in capsys.readouterr().out


@pytest.mark.parametrize("failure", ["csv", "receipt"])
def test_interrupted_publication_leaves_no_destination_and_can_be_retried(
    tmp_path, monkeypatch, failure
):
    from mars_titan.evaluation import campaign_reliability as module

    campaign = prepared_campaign(tmp_path / "sources")
    output = tmp_path / "output"
    original = csv.DictWriter.writerows

    def interrupted_rows(writer, rows):
        original(writer, rows[:1])
        raise OSError("Escritura interrumpida")

    def interrupted_receipt(*_args, **_kwargs):
        raise OSError("Escritura interrumpida")

    with monkeypatch.context() as patch:
        if failure == "csv":
            patch.setattr(csv.DictWriter, "writerows", interrupted_rows)
        else:
            patch.setattr(module, "atomic_json", interrupted_receipt)
        with pytest.raises(OSError, match="interrumpida"):
            module.evaluate_campaign_reliability(campaign.reference, campaign.completion, output)
    assert not output.exists()
    completed = module.evaluate_campaign_reliability(
        campaign.reference, campaign.completion, output
    )
    assert completed["status"] == "completed"
    assert sha256(output / "cases.csv") == completed["artifacts"]["cases.csv"]


def joint_reviews(tmp_path, *, cn_calibration_samples=20, cn_error=1.0, cn_evaluation=True):
    identity = dict(
        fold="fold-000",
        id="reference/search/winner",
        stage="reference",
        phase="search",
        family="rnn",
        method="supervised",
        seed=42,
        included=True,
        primary=True,
        parent_id=None,
        checkpoint_sha256="a" * 64,
        metadata={},
    )
    sources, reviews = [], []
    for partition in ("calibration", "evaluation"):
        calibrating = partition == "calibration"
        us_target = [0.1] * 20 if calibrating else [-1.0, 1.0, 0.0, 1.0]
        cn_target = [cn_error] * cn_calibration_samples if calibrating else [1.0, -1.0, 0.0]
        if not calibrating and not cn_evaluation:
            cn_target = []
        us_prediction = [0.0] * 20 if calibrating else [0.0, 2.0, 3.0, 1.0]
        cn_prediction = (
            [0.0] * len(cn_target) if calibrating else [-2.0, -0.5, 2.0][: len(cn_target)]
        )
        markets = ["US"] * len(us_target) + ["CN"] * len(cn_target)
        times = [
            datetime(2023, 11 if calibrating else 12, 2 if i < count - 1 else 3, hour, tzinfo=UTC)
            for count, hour in ((len(us_target), 21), (len(cn_target), 7))
            for i in range(count)
        ]
        target = np.asarray(us_target + cn_target)
        prediction = np.asarray(us_prediction + cn_prediction)
        cohort = pa.table(
            dict(
                sample_id=[f"{partition}-{i}" for i in range(len(markets))],
                asset_id=[f"{market}/A{i}" for i, market in enumerate(markets)],
                market=markets,
                prediction_at=pa.array(times, pa.timestamp("us", tz="UTC")),
                target=target,
            )
        )
        errors = SessionErrors()
        errors.update(markets, cohort["prediction_at"].to_numpy(), prediction - target)
        reviews.append(
            dict(cohort=cohort, prediction=prediction, metrics=dict(prediction=errors.summary()))
        )
        path = tmp_path / f"{partition}.parquet"
        pq.write_table(cohort.append_column("prediction", pa.array(prediction)), path)
        bounds = ["2023-11-01", "2023-12-01"] if calibrating else ["2023-12-01", "2024-01-01"]
        sources.append(
            identity
            | dict(
                partition=partition,
                path=path,
                sha256=sha256(path),
                source_report_sha256=("b" if calibrating else "c") * 64,
                bounds=bounds,
                bounds_by_market={market: list(bounds) for market in ("US", "CN")},
            )
        )
    return (*sources, *reviews)


def test_joint_calibration_uses_each_market_and_preserves_its_denominators(tmp_path):
    from mars_titan.evaluation.campaign_reliability import _case_rows

    rows = list(_case_rows(*joint_reviews(tmp_path)))
    assert len(rows) == 6
    assert {(row["market"], row["prediction_kind"], row["confidence"]) for row in rows} == {
        (market, kind, confidence)
        for market in ("US", "CN")
        for kind, confidence in (("point", None), ("interval", 0.9), ("interval", 0.95))
    }
    for row in rows:
        assert row["fold"] == "fold-000" and row["seed"] == 42
        assert row["checkpoint_sha256"] == "a" * 64
        assert row["calibration_samples"] == 20
        assert row["calibration_sessions"] == row["evaluation_sessions"] == 2
        assert row["samples"] == row["evaluation_samples"] == (4 if row["market"] == "US" else 3)
        assert row["coverage_guaranteed"] is False
    us, cn = [
        next(row for row in rows if row["market"] == market and row["prediction_kind"] == "point")
        for market in ("US", "CN")
    ]
    assert (us["eligible_targets"], us["calls"], us["abstentions"]) == (3, 2, 1)
    assert us["direction_accuracy"] == pytest.approx(2 / 3)
    assert us["positive_precision"] == 1 and us["negative_precision"] is None
    assert us["mean_session_direction_accuracy"] == 0.75
    assert us["defined_sessions_direction_accuracy"] == 2
    assert (cn["eligible_targets"], cn["calls"], cn["abstentions"]) == (2, 2, 0)
    assert cn["direction_accuracy"] == cn["negative_precision"] == 0.5
    assert cn["positive_precision"] is None
    assert cn["defined_sessions_direction_accuracy"] == 1
    for row in rows:
        if row["prediction_kind"] != "interval":
            continue
        if row["market"] == "US":
            assert row["radius"] == 0.1 and row["coverage"] == 0.25
        else:
            assert row["radius"] == 1.0 and row["coverage"] == pytest.approx(1 / 3)
            assert (row["calls"], row["abstentions"], row["correct_calls"]) == (1, 1, 0)
            assert row["mean_session_coverage"] == 0.25


@pytest.mark.parametrize("cn_samples", [0, 3])
def test_joint_calibration_does_not_borrow_rows_from_the_other_market(tmp_path, cn_samples):
    from mars_titan.evaluation.campaign_reliability import _case_rows

    rows = list(_case_rows(*joint_reviews(tmp_path, cn_calibration_samples=cn_samples)))
    assert len(rows) == 6
    for row in rows:
        if row["prediction_kind"] != "interval":
            continue
        if row["market"] == "US":
            assert row["radius"] == 0.1 and row["calibration_samples"] == 20
        else:
            assert row["calibration_samples"] == cn_samples
            assert row["radius"] is row["coverage"] is row["calls"] is None
            assert row["reason"] and row["defined_sessions_direction_accuracy"] == 0


def test_joint_market_changes_do_not_move_another_market_radius(tmp_path):
    from mars_titan.evaluation.campaign_reliability import _case_rows

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    ordinary = list(_case_rows(*joint_reviews(first)))
    changed = list(_case_rows(*joint_reviews(second, cn_error=1000.0)))
    assert len(ordinary) == len(changed) == 6
    for rows in (ordinary, changed):
        assert {
            row["radius"]
            for row in rows
            if row["market"] == "US" and row["prediction_kind"] == "interval"
        } == {0.1}


def test_joint_market_without_evaluation_rows_keeps_zero_denominators(tmp_path):
    from mars_titan.evaluation.campaign_reliability import _case_rows

    rows = list(_case_rows(*joint_reviews(tmp_path, cn_evaluation=False)))
    assert len(rows) == 6
    for row in rows:
        if row["market"] == "CN":
            assert row["samples"] == row["evaluation_samples"] == row["evaluation_sessions"] == 0
            assert row["direction_accuracy"] is row["coverage"] is None
            assert row["defined_sessions_direction_accuracy"] == 0


def test_joint_pairs_validate_chronology_for_each_market(tmp_path):
    from mars_titan.evaluation.campaign_reliability import _pairs

    calibration, evaluation, *_ = joint_reviews(tmp_path)
    evaluation["bounds_by_market"]["CN"] = ["2023-11-15", "2023-12-15"]
    with pytest.raises(ValueError, match="mercado|anterior"):
        _pairs([calibration, evaluation])


@pytest.mark.parametrize("bounds", [None, {}, {"US": ["2023-11-01", "2023-12-01"]}])
def test_joint_pairs_reject_missing_or_incomplete_market_bounds(tmp_path, bounds):
    from mars_titan.evaluation.campaign_reliability import _pairs

    calibration, evaluation, *_ = joint_reviews(tmp_path)
    calibration["bounds_by_market"] = bounds
    with pytest.raises(ValueError, match="mercados"):
        _pairs([calibration, evaluation])


def test_joint_cohort_cannot_silently_fall_back_to_a_pooled_calibrator(tmp_path):
    from mars_titan.evaluation.campaign_reliability import _case_rows

    calibration, evaluation, *reviews = joint_reviews(tmp_path)
    calibration.pop("bounds_by_market")
    evaluation.pop("bounds_by_market")
    with pytest.raises(ValueError, match="límites por mercado"):
        list(_case_rows(calibration, evaluation, *reviews))


def test_joint_pipeline_publishes_market_rows_without_counting_models_twice(tmp_path):
    from mars_titan.evaluation.campaign_reliability import evaluate_campaign_reliability

    campaign = prepared_campaign(tmp_path / "sources", calibration_samples=40, markets=("US", "CN"))
    output = tmp_path / "reliability"
    report = evaluate_campaign_reliability(campaign.reference, campaign.completion, output)
    assert report["counts"]["models"] == 8
    assert report["counts"]["prediction_files"] == 16
    assert len(report["cases"]) == 48
    assert report["method"]["calibration_grouping"] == "model_fold_seed_market"
    assert report["method"]["market_pooling"] is False
    assert report["coverage_guaranteed"] is False
    for row in report["cases"]:
        assert row["evaluation_samples"] == row["samples"] == 2
        assert row["calibration_samples"] == 20
        if row["prediction_kind"] == "interval":
            assert row["radius"] == (0.1 if row["market"] == "US" else 1.0)
    csv_rows = list(csv.DictReader((output / "cases.csv").open()))
    assert len(csv_rows) == 48 and {row["market"] for row in csv_rows} == {"US", "CN"}
    assert sha256(output / "cases.csv") == report["artifacts"]["cases.csv"]
