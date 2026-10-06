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


def prepared_campaign(root, change=None, *, calibration_samples=20):
    campaign = Campaign(root, matching=True)
    campaign.counts.update(calibration=calibration_samples, evaluation=4)
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
