"""Escritura opcional de predicciones de PC1 a través de la cola de comparación."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.batches import atomic_parquet_batches
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation.comparison_bundle import ResultsBundle, export_results
from mars_titan.evaluation.comparison_metrics import panel_from_table
from mars_titan.training import comparison_campaign as campaign
from tests.evaluation.comparison_results_fixtures import campaign_state
from tests.training.test_comparison_campaign import declaration, layout

ENV = "MARS_TITAN_COMPARISON_PREDICTION_STORAGE"
PROFILE = "compact_v1"


def encodings(path, field):
    meta = pq.read_metadata(path)
    return meta.row_group(0).column(meta.schema.names.index(field)).encodings


def floats_with_metadata():
    bits = np.array(
        [0, 2**63, 0x7FF8000000000007, 0x7FF8000000000008, 0xFFF0000000000000, 1], dtype="u8"
    )
    scores = pa.array(bits.view("f8"))
    masked = pa.array([1.0, None, -0.0, 0.125, 0.25, 0.5], type=pa.float32())
    return pa.table(
        {"target": scores, "prediction": scores, "extra": masked, "asset_id": list("aabbcc")}
    ).replace_schema_metadata({b"version": b"1"})


@pytest.mark.parametrize(
    ("model", "machine", "setting", "split_target", "split_prediction"),
    [
        ("transformer", "references", PROFILE, True, True),
        ("rule", "references", PROFILE, True, False),
        ("area_network", "references", PROFILE, True, True),
        ("titans_mac", "mars_titan", PROFILE, False, False),
        ("transformer", "references", None, False, False),
    ],
)
def test_run_job_limits_encoding_to_its_three_outputs_and_preserves_bits(
    tmp_path, monkeypatch, model, machine, setting, split_target, split_prediction
):
    place, doc = layout(tmp_path), declaration()
    if model == "area_network":
        doc["execution"]["neural_blocks"] = {
            f"{origin['year']}/seed-{seed}": "references"
            for origin in doc["origins"]
            for seed in doc["seeds"]
        }
    job = next(j for j in campaign.plan(doc, machine) if j.model == model)
    root = place.output(job.key)
    root.mkdir(parents=True)
    outside = tmp_path / "other"
    outside.mkdir()
    link = root / "external"
    link.symlink_to(outside, target_is_directory=True)
    source = floats_with_metadata()
    batches = [source.slice(0, 3), source.slice(3)]
    paths = [root / f"{part}-predictions.parquet" for part in ("validation", "calibration")]
    paths.append(root / "seed-42/evaluation-predictions.parquet")
    unrelated = [
        root / "other-predictions.parquet",
        outside / "validation-predictions.parquet",
        link / "evaluation-predictions.parquet",
    ]
    if setting:
        monkeypatch.setenv(ENV, setting)
    else:
        monkeypatch.delenv(ENV, raising=False)

    def execute(*args, **options):
        for path in paths + unrelated:
            assert atomic_parquet_batches(path, batches) == 6
        return {"status": "paused"}

    monkeypatch.setattr(campaign, "execute_job", execute)
    assert campaign.run_job(doc, place, job, device="cpu") == "paused"
    for path in paths:
        assert ("BYTE_STREAM_SPLIT" in encodings(path, "target")) is split_target
        assert ("BYTE_STREAM_SPLIT" in encodings(path, "prediction")) is split_prediction
        with pq.ParquetFile(path) as parquet:
            assert [parquet.metadata.row_group(i).num_rows for i in range(2)] == [3, 3]
            restored = parquet.read(use_threads=False)
        assert restored.schema.equals(source.schema, check_metadata=True)
        assert restored["asset_id"].equals(source["asset_id"])
        assert restored["extra"].is_null().equals(source["extra"].is_null())
        assert np.array_equal(
            restored["extra"].fill_null(0).to_numpy().view("u4"),
            source["extra"].fill_null(0).to_numpy().view("u4"),
        )
        for name in ("target", "prediction"):
            assert np.array_equal(
                restored[name].to_numpy().view("u8"), source[name].to_numpy().view("u8")
            )
    outside_context = root / "evaluation-predictions.parquet"
    atomic_parquet_batches(outside_context, batches)
    for path in [*unrelated, outside_context]:
        assert "BYTE_STREAM_SPLIT" not in encodings(path, "target")
    recorded = campaign.read_record(place, job.key)["attempts"][-1].get("prediction_storage")
    assert recorded == (PROFILE if split_target else None)


def test_invalid_storage_fails_before_opening_the_runner(tmp_path, monkeypatch):
    place, doc = layout(tmp_path), declaration()
    job = campaign.plan(doc, "references")[0]
    monkeypatch.setenv(ENV, "desconocido")
    monkeypatch.setattr(campaign, "execute_job", lambda *a, **k: pytest.fail("ejecutor abierto"))
    assert campaign.run_job(doc, place, job, device="cpu") == "pending"
    attempt = campaign.read_record(place, job.key)["attempts"][-1]
    assert attempt["error_type"] == "ValueError" and ENV in attempt["error"]


def test_writer_failure_keeps_previously_committed_bytes_and_releases_context(
    tmp_path, monkeypatch
):
    place, doc = layout(tmp_path), declaration()
    job = campaign.plan(doc, "references")[0]
    old = place.output(job.key) / "validation-predictions.parquet"
    batch = floats_with_metadata()
    atomic_parquet_batches(old, [batch])
    original = old.read_bytes()
    monkeypatch.setenv(ENV, PROFILE)

    def broken():
        yield batch
        raise OSError("corte de escritura")

    def execute(*args, **options):
        atomic_parquet_batches(old, broken())

    monkeypatch.setattr(campaign, "execute_job", execute)
    assert campaign.run_job(doc, place, job, device="cpu") == "pending"
    assert old.read_bytes() == original and list(old.parent.iterdir()) == [old]
    assert campaign.read_record(place, job.key)["attempts"][-1]["prediction_storage"] == PROFILE
    atomic_parquet_batches(old, [batch])
    assert old.read_bytes() == original


def test_comparison_zip_and_panel_read_the_compact_predictions(tmp_path, monkeypatch):
    doc, digest, place = campaign_state(tmp_path / "source")
    job = campaign.plan(doc, "references")[0]
    folder = place.output(job.key)
    report = json.loads((folder / "run.json").read_text())
    expected = {}
    for partition, entry in report["predictions"].items():
        path = folder / entry["path"]
        expected[partition] = pq.read_table(path)
        path.unlink()
    monkeypatch.setenv(ENV, PROFILE)

    def execute(*args, **options):
        for partition, table in expected.items():
            entry = report["predictions"][partition]
            path = folder / entry["path"]
            written = atomic_parquet_batches(path, [table])
            entry.update(rows=written, bytes=path.stat().st_size, sha256=sha256(path))
        atomic_json(folder / "run.json", report)
        return report

    monkeypatch.setattr(campaign, "execute_job", execute)
    assert campaign.run_job(doc, place, job, device="cpu") == "completed"
    archive = tmp_path / "results.zip"
    export_results(doc, digest, place, "references", archive)
    with ResultsBundle(archive) as uploaded:
        files = uploaded.manifest["jobs"][job.key]["tables"][str(job.seed)]
        for partition, name in files.items():
            assert "BYTE_STREAM_SPLIT" in encodings(
                folder / report["predictions"][partition]["path"], "target"
            )
            table = uploaded.read_table(name)
            panel = panel_from_table(table)
            reference = panel_from_table(expected[partition])
            assert np.array_equal(panel.prediction, reference.prediction)
            assert np.array_equal(panel.target, reference.target)
