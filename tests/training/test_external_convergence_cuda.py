"""Selección y recuperación con XGBoost CUDA real sobre un corpus sintético pequeño."""

import json
import math
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.embeddings import require_cuda
from mars_titan.data.storage import sha256
from mars_titan.models.baselines import external_boosting
from mars_titan.models.baselines.external_boosting import ExternalBoostingModel, _device
from mars_titan.training import external_corpus
from mars_titan.training.checkpoints import StopRequest
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.tabular_corpus import _matrix
from tests.suite_support import requires_cuda
from tests.training.test_reference_run import training_corpus


def _validation_disagrees_with_training(directory):
    """Crear sesiones desiguales y etiquetas que favorezcan la primera ronda."""
    manifest = training_corpus(directory, markets=("US", "CN"))
    metadata = json.loads(manifest.read_text())
    for asset in metadata["assets"]:
        path = directory / "labels" / asset["market"] / asset["symbol"] / "labels.parquet"
        table = pq.read_table(path)
        rows = []
        for row in table.to_pylist():
            if row["partition"] == "validation":
                last = 8 if asset["market"] == "US" else 7
                if asset["symbol"] == "A0001" and row["sample_row"] >= last:
                    continue
                row["target"] = -0.04 if asset["market"] == "CN" and row["sample_row"] == 8 else 0.0
            rows.append(row)
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path, row_group_size=3)
        asset["labels_sha256"] = sha256(path)
        samples = directory / "samples" / asset["market"] / asset["symbol"] / "samples.parquet"
        pq.write_table(pq.read_table(samples).slice(0, len(rows)), samples, row_group_size=3)
        asset["samples_sha256"] = sha256(samples)
        asset["counts"]["validation"] = sum(row["partition"] == "validation" for row in rows)
    metadata["counts"]["validation"] = sum(
        asset["counts"]["validation"] for asset in metadata["assets"]
    )
    manifest.write_text(json.dumps(metadata))
    return manifest


@requires_cuda
def test_cuda_selection_recovers_patience_best_model_and_session_metric(
    tmp_path, monkeypatch, recwarn
):
    require_cuda()
    manifest = _validation_disagrees_with_training(tmp_path / "data")
    options = dict(
        rounds=6,
        selection=dict(schema_version=1, minimum_rounds=2, patience_rounds=2, min_delta=1e-5),
        max_depth=1,
        max_bin=16,
        learning_rate=0.1,
        seed=42,
        batch_size=5,
        max_batch_bytes=1024**2,
        max_host_cache_bytes=32 * 1024**2,
        max_validation_cache_bytes=32 * 1024**2,
        checkpoint_interval=1,
        on_host=False,
    )
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    continuous = external_corpus.run_external_reference(manifest, full, **options)
    assert continuous["device"] == continuous["audit"]["device"] == "cuda:0"
    assert continuous["status"] == "completed"
    assert continuous["stop_reason"] == "validation_plateau"
    assert continuous["completed_rounds"] == 4 and continuous["selected_round"] == 1
    assert continuous["selection"]["rounds_without_improvement"] == 2
    assert continuous["samples"] == {"train": 24, "validation": 9}
    assert continuous["consumed_training_rows"] == 96

    stop = StopRequest()
    confirm = external_corpus._confirm_selection

    def pause_after_confirmation(output, report, model):
        confirm(output, report, model)
        if report["completed_rounds"] == 3:
            stop.request_stop()

    # La instrumentación solo solicita la pausa después de guardar el modelo real.
    with monkeypatch.context() as instrumentation:
        instrumentation.setattr(external_corpus, "_confirm_selection", pause_after_confirmation)
        paused = external_corpus.run_external_reference(manifest, interrupted, stop=stop, **options)
    assert paused["status"] == "paused" and paused["completed_rounds"] == 3
    assert paused["selected_round"] == 1
    assert paused["selection"]["rounds_without_improvement"] == 1
    checkpoint_before = dict(paused["recovery_checkpoint"])
    selection_before = dict(paused["selection"])
    replay_stop = StopRequest()
    create_callback = external_boosting.selection_callback

    def pause_replay(*args, **kwargs):
        callback = create_callback(*args, **kwargs)
        after_iteration = callback.after_iteration

        def after(model, epoch, log):
            requested = after_iteration(model, epoch, log)
            if callback.replayed_rounds == 2:
                replay_stop.request_stop()
                return True
            return requested

        callback.after_iteration = after
        return callback

    with monkeypatch.context() as instrumentation:
        instrumentation.setattr(external_boosting, "selection_callback", pause_replay)
        rebuilding = external_corpus.run_external_reference(
            manifest, interrupted, resume=True, stop=replay_stop, **options
        )
    assert rebuilding["status"] == "paused"
    assert rebuilding["completed_rounds"] == 3
    assert rebuilding["selection"] == selection_before
    assert rebuilding["recovery_checkpoint"] == checkpoint_before
    assert sha256(interrupted / checkpoint_before["path"]) == checkpoint_before["sha256"]
    assert rebuilding["attempts"][-1]["replayed_rounds"] == 2
    assert rebuilding["attempts"][-1]["replayed_training_rows"] == 48
    resumed = external_corpus.run_external_reference(manifest, interrupted, resume=True, **options)
    assert resumed["status"] == "completed" and resumed["completed_rounds"] == 4
    assert resumed["selection"] == continuous["selection"]
    assert resumed["predictions"] == continuous["predictions"]
    assert resumed["attempts"][-1]["replayed_rounds"] == 3
    assert resumed["attempts"][-1]["replayed_training_rows"] == 72

    for folder, report in ((full, continuous), (interrupted, resumed)):
        keep = {
            record["path"] for record in report["recovery_checkpoints"] + [report["checkpoint"]]
        }
        assert len(keep) == 3
        assert {
            str(path.relative_to(folder)) for path in (folder / "checkpoints").iterdir()
        } == keep
        best = ExternalBoostingModel.load(
            folder / report["checkpoint"]["path"], report["checkpoint"]["sha256"], training_rows=24
        )
        last = ExternalBoostingModel.load(
            folder / report["recovery_checkpoint"]["path"],
            report["recovery_checkpoint"]["sha256"],
            training_rows=24,
        )
        assert _device(best.booster) == _device(last.booster) == "cuda:0"
        assert best.booster.num_boosted_rounds() == 1
        assert last.booster.num_boosted_rounds() == 4
        batch = next(
            CorpusDataset(manifest).batches(partition="validation", batch_size=5, epoch=0, seed=0)
        )
        values = _matrix(batch, np.float32)
        assert not np.array_equal(best.predict(values), last.predict(values))

        table = pq.read_table(folder / report["predictions"]["validation"]["path"])
        errors = defaultdict(list)
        for row in table.to_pylist():
            errors[row["market"], row["prediction_at"]].append(
                abs(row["prediction"] - row["target"])
            )
        assert len(errors) == 6
        session_mae = math.fsum(math.fsum(group) / len(group) for group in errors.values()) / 6
        row_mae = math.fsum(value for group in errors.values() for value in group) / 9
        assert math.isclose(session_mae, report["selection"]["best_session_mae"], abs_tol=1e-12)
        assert not math.isclose(session_mae, row_mae, abs_tol=1e-8)
        assert report["predictions"]["validation"]["metrics"]["session_count"] == 6
        assert report["final_test_opened"] is False

    assert (
        external_corpus.run_external_reference(manifest, interrupted, resume=True, **options)
        == resumed
    )
    assert not [
        warning for warning in recwarn if "External memory cache file" in str(warning.message)
    ]
