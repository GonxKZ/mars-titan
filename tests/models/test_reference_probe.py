"""Precondiciones de las sondas antes de tocar GPU o escribir resultados."""

import importlib

import pytest


def module():
    try:
        return importlib.import_module("mars_titan.reference_probe")
    except ModuleNotFoundError:
        pytest.fail("Falta el ejecutor de referencias sobre datos estrictos")


def test_reference_probe_refuses_outputs_inside_prepared_inputs(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    sentinel = prepared / "manifest.json"
    sentinel.write_text("conservar")
    with pytest.raises(ValueError, match="origen"):
        module().run_reference_probe(
            prepared, tmp_path / "samples", prepared / "run", tmp_path / "report.json"
        )
    assert sentinel.read_text() == "conservar"


def test_reference_probe_does_not_replace_an_existing_run(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    with pytest.raises(ValueError, match="nuevo"):
        module().run_reference_probe(
            tmp_path / "prepared", tmp_path / "samples", output, tmp_path / "report.json"
        )


@pytest.mark.parametrize("unverified", ["prepared", "samples"])
def test_strict_preflight_requires_both_verified_manifests(tmp_path, monkeypatch, unverified):
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    unit = module()
    roots = {name: tmp_path / name for name in ("prepared", "samples")}
    for name, root in roots.items():
        path = root / "US/A"
        path.mkdir(parents=True)
        policy = "technical" if name == unverified else "verified_full_articles"
        (path / "manifest.json").write_text(json.dumps({"news_content_policy": policy}))
    pq.write_table(pa.table({"row": [1]}), roots["samples"] / "US/A/samples.parquet")
    monkeypatch.setattr(unit, "prepare_targets", lambda *args: pytest.fail("Se llegó a etiquetas"))
    with pytest.raises(ValueError, match="noticias completas verificadas"):
        unit.prepare_probe(
            roots["prepared"], roots["samples"], tmp_path / "run", tmp_path / "report"
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "artifact", ["ridge.npz", "predictions.parquet", "targets/A-targets.parquet"]
)
def test_reference_report_cannot_overwrite_run_artifacts(tmp_path, artifact):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="origen"):
        module().run_reference_probe(
            tmp_path / "prepared", tmp_path / "samples", output, output / artifact
        )
    assert not output.exists()


def test_boosting_probe_is_explicitly_cpu_and_never_calls_cuda(tmp_path, monkeypatch):
    import json
    from datetime import UTC, datetime

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    unit = module()
    prepared, samples = tmp_path / "prepared", tmp_path / "samples"
    source = prepared / "US/A"
    sample = samples / "US/A"
    source.mkdir(parents=True)
    sample.mkdir(parents=True)
    for path in (source / "manifest.json", sample / "manifest.json"):
        path.write_text(json.dumps({"news_content_policy": "verified_full_articles"}))
    pq.write_table(pa.table({"example": [1]}), sample / "samples.parquet")
    train, validation = datetime(2022, 1, 3, tzinfo=UTC), datetime(2023, 1, 3, tzinfo=UTC)
    targets = {
        "US/A": {train.isoformat(): (0.1, "train"), validation.isoformat(): (0.2, "validation")}
    }

    def labels(paths, prepared, output):
        output.mkdir(parents=True)
        return targets, {"assets": {"A": {"train": 20, "validation": 1}}}, {}

    def records(*args, **kwargs):
        for i, day in enumerate([train] * 20 + [validation]):
            yield {
                "cursor": ("US/A", i),
                "prediction_at": day,
                "inputs": {
                    name: np.array([i])
                    for name in ("prices", "news", "charts", "fundamentals", "macro")
                },
            }

    monkeypatch.setattr(unit, "prepare_targets", labels)
    monkeypatch.setattr(unit, "iter_windows", records)
    monkeypatch.setattr(
        unit, "require_cuda", lambda: pytest.fail("La referencia CPU ha pedido CUDA")
    )
    result = unit.run_reference_probe(
        prepared, samples, tmp_path / "run", tmp_path / "report.json", kind="boosting"
    )
    assert result["device"] == "cpu"
    assert result["samples"] == {"train": 20, "validation": 1}
    assert result["restored_predictions_equal"] is True
    assert result["peak_vram_allocated_bytes"] is None
