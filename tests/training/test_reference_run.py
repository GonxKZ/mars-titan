import importlib
import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.storage import sha256
from mars_titan.training.checkpoints import StopRequest, load_training_state
from tests.training.test_corpus_inputs import corpus


def training_corpus(tmp_path, *, markets=("US",)):
    path = corpus(tmp_path, assets=2, rows=9, markets=markets, group_size=3)
    metadata = json.loads(path.read_text())
    metadata["context_sessions"] = 64
    for asset in metadata["assets"]:
        price_path = tmp_path / "prepared" / asset["market"] / asset["symbol"] / "prices.parquet"
        prices = pa.table(
            {
                **{
                    name: [value + i for i in range(64)]
                    for name, value in dict(
                        open=10.0, high=12.0, low=9.0, close=11.0, volume=100.0
                    ).items()
                },
                "available_at": [
                    datetime(2017, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(64)
                ],
            }
        )
        pq.write_table(prices, price_path)
        asset["prices_sha256"] = sha256(price_path)
        for kind, filename in (("samples", "samples.parquet"), ("labels", "labels.parquet")):
            source = tmp_path / kind / asset["market"] / asset["symbol"] / filename
            rows = pq.read_table(source).to_pylist()
            for i, row in enumerate(rows):
                if kind == "samples":
                    row["price_end_index"] = 63
                if i >= 6:
                    row["prediction_at"] = row["prediction_at"].replace(year=2023)
                    if kind == "labels":
                        row["target_available_at"] = row["target_available_at"].replace(year=2023)
                        row["partition"] = "validation"
            pq.write_table(pa.Table.from_pylist(rows), source, row_group_size=3)
            asset[kind + "_sha256"] = sha256(source)
        asset["counts"] = {"train": 6, "validation": 3}
    metadata["counts"] = {
        "train": len(metadata["assets"]) * 6,
        "validation": len(metadata["assets"]) * 3,
    }
    path.write_text(json.dumps(metadata))
    return path


def module():
    return importlib.import_module("mars_titan.training.reference_run")


def case(kind="gru"):
    return dict(kind=kind, loss="mse", learning_rate=0.001, seed=42, epochs=2, huber_delta=0.01)


@pytest.mark.parametrize("kind", ["rnn", "lstm", "gru", "dlinear"])
def test_every_epoch_and_prediction_include_all_admissible_rows(tmp_path, kind):
    manifest = training_corpus(tmp_path / "data")
    output = tmp_path / "run"
    report = module().run_reference_case(manifest, output, case(kind), batch_size=5)
    assert report["status"] == "completed" and report["final_test_opened"] is False
    assert report["samples"] == {"train": 12, "validation": 6}
    assert len(report["epochs"]) == 2
    assert [e["train"]["samples"] for e in report["epochs"]] == [12, 12]
    for partition, expected in (("train", 12), ("validation", 6)):
        rows = pq.read_table(output / f"{partition}-predictions.parquet").to_pylist()
        assert len({r["sample_id"] for r in rows}) == len(rows) == expected
        assert all(np.isfinite(r["prediction"]) for r in rows)
    assert report["device"] == "cuda:0"


def test_interruption_resumes_exact_next_batch_and_final_weights(tmp_path, monkeypatch):
    manifest = training_corpus(tmp_path / "data")
    engine = module()
    reference = engine.run_reference_case(manifest, tmp_path / "continuous", case(), batch_size=5)
    stop = StopRequest()
    real_save = engine.save_training_state

    def pause(directory, state, **kwargs):
        saved = real_save(directory, state, **kwargs)
        if state["global_step"] == 2:
            stop.request_stop()
        return saved

    monkeypatch.setattr(engine, "save_training_state", pause)
    report = engine.run_reference_case(
        manifest, tmp_path / "interrupted", case(), batch_size=5, checkpoint_steps=1, stop=stop
    )
    assert report["status"] == "paused"
    monkeypatch.setattr(engine, "save_training_state", real_save)
    completed = engine.run_reference_case(
        manifest, tmp_path / "interrupted", case(), batch_size=5, checkpoint_steps=1, resume=True
    )
    assert completed["status"] == "completed"
    assert completed["global_step"] == reference["global_step"] == 6
    first = load_training_state(
        tmp_path / "continuous/checkpoints", expected_identity=reference["identity"]
    )
    second = load_training_state(
        tmp_path / "interrupted/checkpoints", expected_identity=completed["identity"]
    )
    assert all(torch.equal(v, second["model"][k]) for k, v in first["model"].items())
    assert pq.read_table(tmp_path / "continuous/validation-predictions.parquet").equals(
        pq.read_table(tmp_path / "interrupted/validation-predictions.parquet")
    )


def test_resume_rejects_changed_corpus_and_loss(tmp_path):
    manifest = training_corpus(tmp_path / "data")
    output = tmp_path / "run"
    module().run_reference_case(manifest, output, case(), batch_size=5)
    with pytest.raises(ValueError, match="identidad|configuración"):
        module().run_reference_case(
            manifest, output, {**case(), "loss": "mae"}, batch_size=5, resume=True
        )
    data = json.loads(manifest.read_text())
    data["scope"] = "full_corpus"
    data["cohort_complete"] = True
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identidad|configuración"):
        module().run_reference_case(manifest, output, case(), batch_size=5, resume=True)


def test_posttraining_requires_same_parent_population_and_architecture(tmp_path):
    manifest = training_corpus(tmp_path / "data")
    parent = tmp_path / "parent"
    module().run_reference_case(manifest, parent, case(), batch_size=5)
    child = module().run_reference_case(
        manifest,
        tmp_path / "child",
        {**case(), "loss": "mae"},
        batch_size=5,
        initialize_from=parent,
    )
    assert child["initialization"]["parent_checkpoint_sha256"]
    assert child["initialization"]["optimizer_policy"] == "new_adamw"
    with pytest.raises(ValueError, match="origen|arquitectura"):
        module().run_reference_case(
            manifest, tmp_path / "invalid", case("lstm"), batch_size=5, initialize_from=parent
        )


def test_missing_cuda_fails_before_creating_run(tmp_path, monkeypatch):
    manifest = training_corpus(tmp_path / "data")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA|cuda"):
        module().run_reference_case(manifest, tmp_path / "run", case())
    assert not (tmp_path / "run").exists()


def test_stop_already_requested_does_not_apply_an_extra_update(tmp_path):
    manifest = training_corpus(tmp_path / "data")
    stop = StopRequest()
    stop.request_stop()
    report = module().run_reference_case(manifest, tmp_path / "run", case(), stop=stop)
    assert report["status"] == "paused" and report["global_step"] == 0


def test_nonfinite_prediction_does_not_publish_completed_run(tmp_path, monkeypatch):
    manifest = training_corpus(tmp_path / "data")
    engine = module()
    forward = engine.CostProbe.forward

    def invalid(model, inputs):
        return forward(model, inputs) * float("nan")

    monkeypatch.setattr(engine.CostProbe, "forward", invalid)
    with pytest.raises(ValueError, match="finita"):
        engine.run_reference_case(manifest, tmp_path / "run", case())
    report = json.loads((tmp_path / "run/run.json").read_text())
    assert report["status"] == "failed" and report["final_test_opened"] is False


def test_code_change_during_training_cannot_confirm_a_new_state(tmp_path, monkeypatch):
    manifest = training_corpus(tmp_path / "data")
    engine = module()
    original = engine.CostProbe.forward
    digest = engine.sha256

    def change_source(model, inputs):
        monkeypatch.setattr(
            engine, "sha256", lambda p: "0" * 64 if p.name == "reference_run.py" else digest(p)
        )
        return original(model, inputs)

    monkeypatch.setattr(engine.CostProbe, "forward", change_source)
    with pytest.raises(ValueError, match="código"):
        engine.run_reference_case(manifest, tmp_path / "run", case())
    report = json.loads((tmp_path / "run/run.json").read_text())
    state = load_training_state(tmp_path / "run/checkpoints", expected_identity=report["identity"])
    assert state["global_step"] == 0


def test_completed_case_rejects_corrupt_final_checkpoint(tmp_path):
    manifest = training_corpus(tmp_path / "data")
    output = tmp_path / "run"
    report = module().run_reference_case(manifest, output, case(), batch_size=5)
    (output / report["checkpoint"]["path"]).write_bytes(b"incompleto")
    with pytest.raises(ValueError, match="punto|control|checkpoint"):
        module().run_reference_case(manifest, output, case(), batch_size=5, resume=True)


def test_resume_rejects_changed_numerical_backend_policy(tmp_path):
    manifest = training_corpus(tmp_path / "data")
    output = tmp_path / "run"
    original = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        module().run_reference_case(manifest, output, case(), batch_size=5)
        torch.set_float32_matmul_precision("high")
        with pytest.raises(ValueError, match="identidad|configuración"):
            module().run_reference_case(manifest, output, case(), batch_size=5, resume=True)
    finally:
        torch.set_float32_matmul_precision(original)


def test_parent_final_checkpoint_cannot_fall_back_to_another_epoch(tmp_path):
    manifest = training_corpus(tmp_path / "data")
    parent = tmp_path / "parent"
    report = module().run_reference_case(manifest, parent, case(), batch_size=5)
    (parent / report["checkpoint"]["path"]).write_bytes(b"incompleto")
    with pytest.raises(ValueError, match="origen|punto|control"):
        module().run_reference_case(
            manifest,
            tmp_path / "child",
            {**case(), "loss": "mae"},
            batch_size=5,
            initialize_from=parent,
        )
    assert not (tmp_path / "child").exists()


def test_changed_price_transformation_cannot_resume_checkpoint(tmp_path, monkeypatch):
    manifest = training_corpus(tmp_path / "data")
    output = tmp_path / "run"
    engine = module()
    engine.run_reference_case(manifest, output, case(), batch_size=5)
    digest = engine.sha256
    monkeypatch.setattr(
        engine, "sha256", lambda p: "0" * 64 if p.name == "streaming.py" else digest(p)
    )
    with pytest.raises(ValueError, match="identidad|configuración"):
        engine.run_reference_case(manifest, output, case(), batch_size=5, resume=True)
