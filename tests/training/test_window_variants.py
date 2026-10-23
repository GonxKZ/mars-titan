"""Contrato y recuperación del ajuste experimental de ventanas independientes."""

import importlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training.checkpoints import StopRequest, load_training_state
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.edition_subsets import materialize
from mars_titan.training.kernel_policy import FP32_STRICT, declared_policy
from mars_titan.training.learning_hold import HOLD_ENV, LearningHoldError
from mars_titan.training.temporal_corpus import prepare_temporal_corpus
from tests.suite_support import requires_cuda
from tests.training.edition_catalog_fixture import built_catalog


def engine():
    name = "mars_titan.training.window_variant_run"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        pytest.fail("Falta el runner experimental de ventanas independientes")


def recipe(**changes):
    options = dict(
        train_start="2015-01-01",
        train_end="2015-04-01",
        validation_start="2015-04-01",
        validation_end="2015-05-01",
        temporal_encoder=None,
        estimator="backprop",
        optimizer="adamw",
        seed=42,
        epochs=2,
        batch_size=2,
        learning_rate=0.001,
        weight_decay=0.0,
        loss="mse",
        huber_delta=0.01,
        max_grad_norm=1.0,
        hidden_size=32,
        layers=1,
        max_seconds=120.0,
        max_output_bytes=64 * 1024**2,
        max_cuda_bytes=512 * 1024**2,
        cache_bytes=1024**2,
        checkpoint_steps=1,
        selection=dict(metric="session_mae", stopping="fixed_budget", patience=3, min_delta=0.0),
        dust=None,
        final_test_access=False,
    )
    options.update(changes)
    return engine().WindowVariantRecipe(**options)


def dust_options():
    return dict(
        population=2,
        chunk_size=1,
        sigma=0.05,
        seed=42,
        max_forward_calls=512,
        max_workspace_bytes=16 * 1024**2,
    )


def same_state(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            same_state(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(same_state(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def short_protocol():
    configuration = json.loads(
        Path("configs/evaluation/historical-masked-us-walk-forward-v2.json").read_text()
    )
    configuration.update(
        first_validation_start="2015-04-01",
        minimum_train_months=3,
        validation_months=1,
        calibration_months=1,
        evaluation_months=1,
        step_months=120,
        seeds=[42],
        selection=dict(
            metric="session_mae", stopping="fixed_budget", patience=3, min_delta=0.0, max_epochs=2
        ),
    )
    return configuration


@pytest.fixture
def window_source(tmp_path):
    days = [
        "2014-12-15",
        "2015-01-06",
        "2015-02-10",
        "2015-03-17",
        "2015-04-07",
        "2015-04-21",
        "2015-05-12",
        "2015-06-16",
        "2023-01-05",
        "2023-01-06",
    ]
    data = built_catalog(tmp_path / "source", markets=("US",), assets=2, days={"US": days})
    assert data.metadata["counts"] == dict(train=16, validation=4)
    return data


@pytest.fixture
def short_view(window_source, tmp_path):
    definition = tmp_path / "definition.json"
    atomic_json(
        definition,
        dict(
            schema_version=1,
            kind="edition_subset_definition",
            name="window-2015",
            markets=["US"],
            years=dict(first=2015, last=2015),
            labels="usable",
        ),
    )
    materialize(window_source.catalog, definition, tmp_path / "subset")
    protocol = tmp_path / "protocol.json"
    atomic_json(protocol, short_protocol())
    prepare_temporal_corpus(
        window_source.parent,
        protocol,
        None,
        None,
        tmp_path / "views",
        recover_annual_boundaries=True,
        input_policy=HISTORICAL_MASKED,
        subset=tmp_path / "subset/manifest.json",
    )
    manifest = tmp_path / "views/fold-000/manifest.json"
    assert json.loads(manifest.read_text())["counts"] == dict(
        train=6, validation=4, calibration=2, evaluation=2
    )
    return manifest


@pytest.fixture
def allowed_child(tmp_path, monkeypatch):
    hold = tmp_path / "child-hold.json"
    hold.write_text(json.dumps(dict(training_allowed=True)))
    monkeypatch.setenv(HOLD_ENV, str(hold))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return hold


@pytest.mark.parametrize(
    "encoder,estimator,optimizer",
    [
        (None, "backprop", "adamw"),
        ("kronlstm_v1", "backprop", "adamw"),
        ("vector_delta_v1", "backprop", "adamw"),
        (None, "dust", "adamw"),
        (None, "backprop", "sgd"),
        (None, "dust", "sgd"),
    ],
)
def test_six_arms_are_explicit_noncanonical_recipes(encoder, estimator, optimizer):
    dust = dust_options() if estimator == "dust" else None
    value = recipe(temporal_encoder=encoder, estimator=estimator, optimizer=optimizer, dust=dust)
    identity = value.identity()
    assert identity["temporal_encoder"] == encoder
    assert identity["estimator"] == estimator and identity["optimizer"] == optimizer
    assert identity["final_test_access"] is False
    assert identity["selection"]["stopping"] == "fixed_budget"
    assert asdict(value)["validation_end"] == "2015-05-01"


@pytest.mark.parametrize(
    "changes",
    [
        dict(temporal_encoder="gru"),
        dict(estimator="noise"),
        dict(optimizer="adam"),
        dict(estimator="dust", temporal_encoder="kronlstm_v1", dust=dust_options()),
        dict(estimator="dust", temporal_encoder="vector_delta_v1", dust=dust_options()),
        dict(estimator="dust"),
        dict(dust=dict(population=2)),
        dict(final_test_access=True),
        dict(final_test_access=0),
        dict(train_start="2013-01-01"),
        dict(train_end="2015-04-02"),
        dict(validation_end="2024-01-01"),
        dict(validation_start="2015-03-01"),
        dict(validation_end="2015-08-01"),
        dict(epochs=0),
        dict(seed=True),
        dict(batch_size=0),
        dict(learning_rate=float("nan")),
        dict(max_seconds=float("inf")),
        dict(max_output_bytes=0),
        dict(checkpoint_steps=0),
        dict(selection=dict(metric="session_mae", stopping="plateau", patience=1, min_delta=0.0)),
    ],
)
def test_recipe_rejects_invalid_scope_before_accessing_data(changes):
    with pytest.raises((ValueError, TypeError)):
        recipe(**changes)


def test_total_train_and_validation_must_fit_366_days():
    value = recipe(
        train_start="2019-01-01",
        train_end="2019-12-01",
        validation_start="2019-12-01",
        validation_end="2020-01-01",
    )
    assert value.identity()["validation_end"] == "2020-01-01"
    with pytest.raises(ValueError):
        recipe(
            train_start="2019-01-01",
            train_end="2020-01-01",
            validation_start="2020-01-01",
            validation_end="2020-02-01",
        )


def test_recipe_rejects_unknown_fields():
    with pytest.raises(TypeError):
        recipe(use_final_test=True)


def test_learning_hold_precedes_manifest_cuda_and_output(tmp_path, monkeypatch):
    api = engine()
    hold = tmp_path / "hold.json"
    hold.write_text(json.dumps(dict(training_allowed=False)))
    before = hold.read_bytes()
    monkeypatch.setenv(HOLD_ENV, str(hold))

    def forbidden(*args, **kwargs):
        pytest.fail("El bloqueo debe preceder a la apertura del corpus y CUDA")

    monkeypatch.setattr(api, "CorpusDataset", forbidden)
    monkeypatch.setattr(api, "require_cuda", forbidden)
    destination = tmp_path / "absent-output"
    with pytest.raises(LearningHoldError):
        api.run_window_variant(tmp_path / "missing-manifest.json", destination, recipe())
    assert not destination.exists() and hold.read_bytes() == before


def test_builder_rejects_non_january_start_before_sources_or_output(tmp_path):
    output = tmp_path / "unprepared"
    with pytest.raises(ValueError, match="enero|inicio|subconjunto"):
        engine().prepare_window_variant_view(
            tmp_path / "missing-parent.json",
            tmp_path / "missing-catalog.json",
            output,
            recipe(train_start="2015-02-01"),
        )
    assert not output.exists()


def test_builder_keeps_2000_contract_but_subsets_effective_2015(window_source, tmp_path):
    api = engine()
    manifest = api.prepare_window_variant_view(
        window_source.parent, window_source.catalog, tmp_path / "built", recipe()
    )
    document = json.loads(Path(manifest).read_text())
    temporal = document["temporal_view"]
    assert temporal["protocol"]["train_start"] == "2000-01-01"
    assert temporal["protocol"]["final_test_start"] == "2024-01-01"
    assert temporal["protocol"]["step_months"] == 120
    assert document["counts"] == dict(train=6, validation=4, calibration=2, evaluation=2)
    assert document["final_test_opened"] is False and document["data_subset"]
    for asset in document["assets"]:
        labels = pq.read_table(
            Path(document["roots"]["labels"]) / asset["market"] / asset["symbol"] / "labels.parquet"
        ).to_pylist()
        old = [row for row in labels if row["prediction_at"].year == 2014]
        assert len(old) == 1 and old[0]["reason"] == "outside_subset"
        assert old[0]["partition"] is None and old[0]["target"] is None
        future = [row for row in labels if row["prediction_at"].year == 2023]
        assert len(future) == 2
        assert all(row["reason"] == "outside_subset" for row in future)
        assert all(row["partition"] is None and row["target"] is None for row in future)


def test_real_rows_outside_recipe_are_rejected_before_forward(
    short_view, tmp_path, monkeypatch, allowed_child
):
    api = engine()

    def forbidden(*args, **kwargs):
        pytest.fail("Una fila fuera del intervalo debe rechazarse antes del forward")

    monkeypatch.setattr(api, "_forward", forbidden)
    destination = tmp_path / "rejected"
    with pytest.raises(ValueError, match="intervalo|ventana|fecha"):
        api.run_window_variant(short_view, destination, recipe(train_start="2015-02-01"))
    assert not destination.exists()


def test_admission_timeout_records_zero_updates_before_cuda(
    short_view, tmp_path, monkeypatch, allowed_child
):
    api = engine()
    clock = iter((0.0, 121.0))
    monkeypatch.setattr(api.time, "perf_counter", lambda: next(clock, 121.0))

    def forbidden(*args, **kwargs):
        pytest.fail("Una admisión interrumpida no debe abrir CUDA")

    monkeypatch.setattr(api, "require_cuda", forbidden)
    output = tmp_path / "admission-timeout"
    report = api.run_window_variant(short_view, output, recipe())
    assert report["status"] == "partial" and report["stage"] == "admission"
    assert report["global_step"] == 0 and report["checkpoint"] is None
    assert not (output / "checkpoints").exists() and not (output / "config.json").exists()
    assert json.loads((output / "report.json").read_text()) == report


@requires_cuda
@pytest.mark.parametrize(
    ("previous_status", "interruption"),
    [
        pytest.param("paused", "timeout", id="paused-timeout"),
        pytest.param("paused", "stop", id="paused-stop"),
        pytest.param("completed", "timeout", id="completed-timeout"),
        pytest.param("completed", "stop", id="completed-stop"),
        pytest.param("checkpoint_without_report", "timeout", id="hardkill-timeout"),
        pytest.param("checkpoint_without_report", "stop", id="hardkill-stop"),
    ],
)
def test_resume_admission_preserves_confirmed_runtime(
    short_view, tmp_path, monkeypatch, allowed_child, previous_status, interruption
):
    api = engine()
    output = tmp_path / "readmission"
    configuration = recipe(epochs=1)
    initial_stop = StopRequest()
    real_save, real_update = api.save_training_state, api._update_window
    updates = []

    class CheckpointInterruption(BaseException):
        pass

    def observe_update(*args, **kwargs):
        result = real_update(*args, **kwargs)
        updates.append(1)
        return result

    def interrupt_after_checkpoint(directory, state, **kwargs):
        path = real_save(directory, state, **kwargs)
        if state["global_step"] == 1:
            if previous_status == "checkpoint_without_report":
                raise CheckpointInterruption()
            initial_stop.request_stop()
        return path

    monkeypatch.setattr(api, "_update_window", observe_update)
    with monkeypatch.context() as initial:
        if previous_status != "completed":
            initial.setattr(api, "save_training_state", interrupt_after_checkpoint)
        if previous_status == "checkpoint_without_report":
            with pytest.raises(CheckpointInterruption):
                api.run_window_variant(short_view, output, configuration, stop=initial_stop)
            assert not (output / "report.json").exists()
        else:
            previous = api.run_window_variant(short_view, output, configuration, stop=initial_stop)
            assert previous["status"] == previous_status

    expected_step = 3 if previous_status == "completed" else 1
    identity = json.loads((output / "config.json").read_text())["identity"]
    confirmed = load_training_state(output / "checkpoints", expected_identity=identity)
    assert confirmed["global_step"] == expected_step == len(updates)
    if expected_step == 1:
        assert confirmed["cursor"] is not None and confirmed["statistics"]["samples"] == 2

    index = json.loads((output / "checkpoints/latest.json").read_text())
    records = [*index["latest"], *index["pinned"], index["best"]]
    paths = {
        output / "config.json",
        output / "report.json",
        output / "checkpoints/latest.json",
        output / "checkpoints/identity.json",
        *(output / "checkpoints" / row["name"] for row in records if row is not None),
    }
    if previous_status == "completed":
        assert previous["global_step"] == 3
        for category in ("predictions", "session_metrics"):
            paths.update(output / row["path"] for row in previous[category].values())
    before = {path: path.read_bytes() if path.exists() else None for path in paths}

    def forbidden(*args, **kwargs):
        pytest.fail("La readmisión interrumpida no debe abrir CUDA")

    stop = StopRequest()
    with monkeypatch.context() as admission:
        admission.setattr(api, "require_cuda", forbidden)
        if interruption == "stop":
            stop.request_stop()
        else:
            admission.setattr(api._Budget, "expired", property(lambda budget: True))
        interrupted = api.run_window_variant(
            short_view, output, configuration, resume=True, stop=stop
        )
    assert interrupted["global_step"] == expected_step
    assert len(updates) == expected_step
    assert {path: path.read_bytes() if path.exists() else None for path in paths} == before
    intact = load_training_state(output / "checkpoints", expected_identity=identity)
    assert same_state(intact, confirmed)

    restored = []

    def observe_restore(directory, state, **kwargs):
        path = real_save(directory, state, **kwargs)
        if not restored:
            saved = load_training_state(directory, expected_identity=identity)
            assert same_state(saved, confirmed)
            restored.append(saved["global_step"])
        return path

    monkeypatch.setattr(api, "save_training_state", observe_restore)
    resumed = api.run_window_variant(short_view, output, configuration, resume=True)
    assert resumed["status"] == "completed" and resumed["global_step"] == 3
    assert resumed["work"]["optimizer_updates"] == 3 and len(updates) == 3
    assert (output / "config.json").read_bytes() == before[output / "config.json"]
    final = load_training_state(output / "checkpoints", expected_identity=identity)
    assert final["global_step"] == 3 and final["epoch"] == 1 and final["cursor"] is None
    assert final["history"][0]["train"]["samples"] == 6
    if previous_status == "completed":
        assert not restored and resumed == previous
        assert {path: path.read_bytes() if path.exists() else None for path in paths} == before
    else:
        assert restored == [1]


@requires_cuda
def test_full_short_epochs_seal_configuration_before_first_update(
    short_view, tmp_path, monkeypatch, allowed_child
):
    api = engine()
    output = tmp_path / "run"
    real_step = torch.optim.AdamW.step
    real_batch_update, real_statistics = api._update_window, api._update
    sealed, updates = [], []
    original_targets = []

    def keep_targets(model, estimator, optimizer, batch, configuration, device):
        original_targets.append(batch["target"].copy())
        return real_batch_update(model, estimator, optimizer, batch, configuration, device)

    def exact_statistics(statistics, prediction, target):
        assert target.dtype == torch.float64
        np.testing.assert_array_equal(target.detach().cpu().numpy(), original_targets.pop(0))
        return real_statistics(statistics, prediction, target)

    def observed(optimizer, *args, **kwargs):
        config = output / "config.json"
        assert config.is_file()
        sealed.append(config.read_bytes())
        updates.append(
            sum(
                parameter.numel()
                for group in optimizer.param_groups
                for parameter in group["params"]
            )
        )
        return real_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", observed)
    monkeypatch.setattr(api, "_update_window", keep_targets)
    monkeypatch.setattr(api, "_update", exact_statistics)
    report = api.run_window_variant(short_view, output, recipe())
    assert report["status"] == "completed" and report["final_test_opened"] is False
    assert report["device"] == "cuda:0"
    assert report["samples"] == dict(train=6, validation=4)
    assert [row["train"]["samples"] for row in report["epochs"]] == [6, 6]
    assert updates and min(updates) > 1000
    assert not original_targets
    assert all(value == sealed[0] for value in sealed)
    config = json.loads(sealed[0])
    assert config["identity"] == report["identity"]
    assert report["identity"]["manifest_sha256"] == sha256(short_view)
    assert report["identity"]["sources"]
    assert allowed_child.read_text() == json.dumps(dict(training_allowed=True))
    for partition, count in (("train", 6), ("validation", 4)):
        rows = pq.read_table(output / f"{partition}-predictions.parquet").to_pylist()
        assert len(rows) == len({row["sample_id"] for row in rows}) == count
        assert all(row["prediction_at"].year == 2015 for row in rows)


@requires_cuda
@pytest.mark.parametrize(
    "arm",
    [
        "transformer-bp-adamw",
        "kronlstm-bp-adamw",
        "vector-delta-bp-adamw",
        "transformer-dust-adamw",
        "transformer-bp-sgd",
        "transformer-dust-sgd",
    ],
)
def test_six_published_routes_complete_fixture_updates_and_recover_checkpoint(
    short_view, tmp_path, monkeypatch, allowed_child, arm
):
    from mars_titan.models.baselines.transformer import CompactPriceTransformer
    from mars_titan.models.kronlstm import CompactPriceKronLSTM
    from mars_titan.training.dust import DustEstimator, FinancialDustAdapter

    api = engine()
    path = (
        Path(__file__).resolve().parents[2]
        / "configs/experiments/window-variants-pilot-month-v3"
        / f"{arm}.json"
    )
    published = json.loads(path.read_text())
    # Solo se adapta el intervalo de la fixture, una época, su lote y los límites técnicos.
    adapted = dict(
        train_start="2015-01-01",
        train_end="2015-04-01",
        validation_start="2015-04-01",
        validation_end="2015-05-01",
        epochs=1,
        batch_size=2,
        max_seconds=120.0,
        max_output_bytes=64 * 1024**2,
        max_cuda_bytes=512 * 1024**2,
        cache_bytes=1024**2,
        checkpoint_steps=1,
    )
    configuration = api.WindowVariantRecipe(**(published | adapted))
    assert configuration.loss == published["loss"] == "mae"
    optimizer_type = torch.optim.AdamW if published["optimizer"] == "adamw" else torch.optim.SGD
    real_update, real_step = api._update_window, optimizer_type.step
    seen, steps = [], []

    def observed_update(model, estimator, optimizer, batch, options, device):
        predictor = model.predictor
        encoder = predictor.price_encoder
        expected_encoder = (
            CompactPriceTransformer
            if published["temporal_encoder"] is None
            else CompactPriceKronLSTM
        )
        assert type(encoder) is expected_encoder
        assert predictor.config.temporal_encoder == published["temporal_encoder"]
        if published["temporal_encoder"] is not None:
            expected_rule = "scalar" if arm.startswith("kronlstm") else "vector"
            assert encoder.configuration["rule"] == expected_rule
            assert all(block.rule == expected_rule for block in encoder.blocks)
        assert type(optimizer) is optimizer_type
        parameters = sum(p.numel() for p in predictor.parameters())
        assert parameters > 1000 and str(device) == "cuda:0"
        if published["estimator"] == "dust":
            assert type(estimator) is DustEstimator
            assert type(estimator.adapter) is FinancialDustAdapter
            assert estimator.loss == "mae"
            assert estimator.config.population == 16 and estimator.config.chunk_size == 8
            assert estimator.config.max_workspace_bytes == 512 * 1024**2
        else:
            assert estimator is None
        result = real_update(model, estimator, optimizer, batch, options, device)
        seen.append((tuple(batch["sample_ids"]), parameters))
        return result

    def observed_step(optimizer, *args, **kwargs):
        steps.append(type(optimizer))
        return real_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(api, "_update_window", observed_update)
    monkeypatch.setattr(optimizer_type, "step", observed_step)
    output = tmp_path / arm
    report = api.run_window_variant(short_view, output, configuration)
    assert report["status"] == "completed" and report["device"] == "cuda:0"
    assert report["global_step"] == report["work"]["optimizer_updates"] == 3
    assert steps == [optimizer_type] * 3 and len(seen) == 3
    assert report["parameters"] == seen[0][1] > 1000
    assert report["samples"] == dict(train=6, validation=4)
    assert len(report["epochs"]) == 1 and report["epochs"][0]["train"]["samples"] == 6
    assert report["identity"]["recipe"] == configuration.identity()
    assert report["final_test_opened"] is False
    assert report["identity"]["final_test_access"] is False
    assert report["evaluation_scope"]["final_test"] == "closed"
    trained_ids = [sample for samples, _ in seen for sample in samples]
    assert len(trained_ids) == len(set(trained_ids)) == 6
    for partition, count in (("train", 6), ("validation", 4)):
        reference = report["predictions"][partition]
        prediction = output / reference["path"]
        rows = pq.read_table(prediction).to_pylist()
        assert len(rows) == len({row["sample_id"] for row in rows}) == count
        if partition == "train":
            assert {row["sample_id"] for row in rows} == set(trained_ids)
    saved = load_training_state(output / "checkpoints", expected_identity=report["identity"])
    assert saved["global_step"] == 3 and saved["epoch"] == 1 and saved["cursor"] is None
    assert {"model", "optimizer", "rng", "selection", "estimator"} <= saved.keys()
    if published["estimator"] == "dust":
        assert saved["estimator"]["steps"] == 3
        assert saved["estimator"]["totals"]["logical_rows"] == 6
    else:
        assert saved["estimator"] is report["estimator"] is None


@requires_cuda
@pytest.mark.parametrize(
    "estimator,temporal_encoder",
    [
        pytest.param("backprop", None, id="backprop"),
        pytest.param("dust", None, id="dust"),
        pytest.param("backprop", "kronlstm_precomputed_v1", id="precomputed-scalar"),
    ],
)
def test_resume_restores_exact_optimizer_rng_cursor_and_weights(
    short_view, tmp_path, monkeypatch, allowed_child, estimator, temporal_encoder
):
    api = engine()
    configuration = recipe(
        epochs=1,
        estimator=estimator,
        temporal_encoder=temporal_encoder,
        dust=dust_options() if estimator == "dust" else None,
    )
    reference = api.run_window_variant(short_view, tmp_path / "continuous", configuration)
    if estimator == "dust":
        source = "models/titans/frozen_financial.py"
        assert reference["identity"]["sources"][source] == sha256(
            Path(api.__file__).resolve().parents[1] / source
        )
    stop = StopRequest()
    real_save = api.save_training_state

    def pause(directory, state, **kwargs):
        saved = real_save(directory, state, **kwargs)
        if state["global_step"] == 1:
            stop.request_stop()
        return saved

    monkeypatch.setattr(api, "save_training_state", pause)
    partial = api.run_window_variant(short_view, tmp_path / "interrupted", configuration, stop=stop)
    assert partial["status"] == "paused" and partial["global_step"] == 1
    saved = load_training_state(
        tmp_path / "interrupted/checkpoints", expected_identity=partial["identity"]
    )
    assert {"model", "optimizer", "rng", "cursor", "selection", "estimator"} <= saved.keys()
    assert (saved["estimator"] is None) == (estimator == "backprop")
    assert saved["cursor"] is not None
    monkeypatch.setattr(api, "save_training_state", real_save)
    resumed = api.run_window_variant(
        short_view, tmp_path / "interrupted", configuration, resume=True
    )
    assert resumed["status"] == "completed"
    assert resumed["global_step"] == reference["global_step"]
    left = load_training_state(
        tmp_path / "continuous/checkpoints", expected_identity=reference["identity"]
    )
    right = load_training_state(
        tmp_path / "interrupted/checkpoints", expected_identity=resumed["identity"]
    )
    assert same_state(left["optimizer"], right["optimizer"])
    assert same_state(left["rng"], right["rng"])
    assert same_state(left["estimator"], right["estimator"])
    assert same_state(left["model"], right["model"])
    assert pq.read_table(tmp_path / "continuous/validation-predictions.parquet").equals(
        pq.read_table(tmp_path / "interrupted/validation-predictions.parquet")
    )


@requires_cuda
def test_selection_can_keep_initial_checkpoint_without_stopping_fixed_updates(
    short_view, tmp_path, monkeypatch, allowed_child
):
    api = engine()
    real_evaluate = api._evaluate
    validations = []

    def observed(*args, **kwargs):
        result = real_evaluate(*args, **kwargs)
        if kwargs.get("partition", "validation") == "validation":
            result["session_mae"] = 0.0 if not validations else 1.0
            validations.append(result["session_mae"])
        return result

    monkeypatch.setattr(api, "_evaluate", observed)
    report = api.run_window_variant(short_view, tmp_path / "initial-best", recipe())
    expected = sum(
        len(
            list(
                CorpusDataset(short_view, input_policy=HISTORICAL_MASKED).batches(
                    partition="train", batch_size=2, epoch=epoch, seed=42
                )
            )
        )
        for epoch in range(2)
    )
    assert report["selection"]["best_epoch"] == 0
    assert report["global_step"] == expected and len(report["epochs"]) == 2


@requires_cuda
def test_timeout_is_partial_and_has_recoverable_confirmed_checkpoint(
    short_view, tmp_path, monkeypatch, allowed_child
):
    api = engine()
    clock = [0.0]
    real_save = api.save_training_state
    monkeypatch.setattr(api.time, "perf_counter", lambda: clock[0])

    def expire(directory, state, **kwargs):
        saved = real_save(directory, state, **kwargs)
        if state["global_step"] == 1:
            clock[0] = 121.0
        return saved

    monkeypatch.setattr(api, "save_training_state", expire)
    output = tmp_path / "timeout"
    report = api.run_window_variant(short_view, output, recipe())
    assert report["status"] == "partial" and report["stop_reason"] == "timeout"
    assert report["global_step"] == 1
    saved = load_training_state(output / "checkpoints", expected_identity=report["identity"])
    assert saved["global_step"] == 1 and saved["cursor"] is not None
    assert not (output / "validation-predictions.parquet").exists()


@requires_cuda
@pytest.mark.parametrize("drift", [False, True], ids=["apply-before-forward", "reject-drift"])
def test_fp32_strict_is_applied_and_enforced_before_checkpoint(
    short_view, tmp_path, monkeypatch, allowed_child, drift
):
    api = engine()
    previous = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    output = tmp_path / "precision"
    real_evaluate, real_step = api._evaluate, torch.optim.AdamW.step
    checkpoints, evaluations = [], []

    def observed_evaluate(*args, **kwargs):
        if not drift:
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.backends.cudnn.allow_tf32 is False
            assert torch.get_float32_matmul_precision() == "highest"
        evaluations.append(kwargs.get("partition", "validation"))
        return real_evaluate(*args, **kwargs)

    def changed_policy(optimizer, *args, **kwargs):
        checkpoints.append((output / "checkpoints/latest.json").read_bytes())
        result = real_step(optimizer, *args, **kwargs)
        torch.backends.cudnn.allow_tf32 = True
        return result

    monkeypatch.setattr(api, "_evaluate", observed_evaluate)
    try:
        if drift:
            declared_policy(FP32_STRICT)
            monkeypatch.setattr(torch.optim.AdamW, "step", changed_policy)
            with pytest.raises(ValueError, match="precisión|numérica|política|núcleos"):
                api.run_window_variant(short_view, output, recipe(epochs=1))
            assert len(checkpoints) == 1
            assert (output / "checkpoints/latest.json").read_bytes() == checkpoints[0]
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            report = api.run_window_variant(short_view, output, recipe(epochs=1))
            assert evaluations and report["status"] == "completed"
            assert report["identity"].get("precision") == FP32_STRICT
            assert report["identity"].get("kernel_policy") == declared_policy(FP32_STRICT)
    finally:
        torch.set_float32_matmul_precision(previous[2])
        torch.backends.cuda.matmul.allow_tf32 = previous[0]
        torch.backends.cudnn.allow_tf32 = previous[1]


def test_fp32_strict_rejects_cpu_autocast_before_reading_sources(
    tmp_path, monkeypatch, allowed_child
):
    api = engine()

    def forbidden(*args, **kwargs):
        pytest.fail("La precisión debe rechazarse antes de leer el corpus")

    monkeypatch.setattr(api, "CorpusDataset", forbidden)
    output = tmp_path / "autocast-rejected"
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="autocast|precisión"):
            api.run_window_variant(tmp_path / "missing-manifest.json", output, recipe())
    assert not output.exists()


@requires_cuda
def test_selected_session_artifacts_reuse_predictions_and_completed_resume_verifies_them(
    short_view, tmp_path, monkeypatch, allowed_child
):
    api = engine()
    output = tmp_path / "selected-sessions"
    real_forward, real_batches, real_evaluate = (
        api._WindowModel.forward,
        CorpusDataset.batches,
        api._evaluate,
    )
    reader = CorpusDataset(short_view, input_policy=HISTORICAL_MASKED)
    counts = {
        partition: len(list(reader.batches(partition=partition, batch_size=2, epoch=0, seed=42)))
        for partition in ("train", "validation")
    }
    forward_calls, exported = [], []

    def observed_forward(model, inputs, presence=None):
        assert len(exported) < 2, "Las métricas por sesión no requieren otro forward"
        forward_calls.append(len(inputs["prices"]))
        return real_forward(model, inputs, presence)

    def observed_batches(dataset, **kwargs):
        assert len(exported) < 2, "Las métricas por sesión deben reutilizar las predicciones"
        yield from real_batches(dataset, **kwargs)

    def observed_evaluate(*args, **kwargs):
        result = real_evaluate(*args, **kwargs)
        if kwargs.get("destination") is not None:
            exported.append(kwargs["partition"])
        return result

    monkeypatch.setattr(api._WindowModel, "forward", observed_forward)
    monkeypatch.setattr(CorpusDataset, "batches", observed_batches)
    monkeypatch.setattr(api, "_evaluate", observed_evaluate)
    report = api.run_window_variant(short_view, output, recipe(epochs=1))
    assert report["status"] == "completed" and exported == ["train", "validation"]
    expected_calls = 2 * counts["train"] + 3 * counts["validation"]
    assert len(forward_calls) == report["work"]["clean_forward_calls"] == expected_calls
    references = report.get("session_metrics")
    assert isinstance(references, dict) and set(references) == {"train", "validation"}
    prediction_bytes = {}
    for partition, reference in references.items():
        path = output / reference["path"]
        assert sha256(path) == reference["sha256"]
        prediction = output / report["predictions"][partition]["path"]
        prediction_bytes[prediction] = prediction.read_bytes()
        groups = {}
        for row in pq.read_table(prediction).to_pylist():
            key = row["market"], row["prediction_at"]
            values = groups.setdefault(key, [0, 0.0, 0.0, 0.0, 0.0])
            error = float(row["prediction"]) - float(row["target"])
            values[0] += 1
            for index, value in enumerate(
                (abs(error), error**2, abs(float(row["target"])), float(row["target"]) ** 2), 1
            ):
                values[index] += value
        rows = pq.read_table(path).to_pylist()
        assert len(rows) == len(groups)
        assert sum(row["samples"] for row in rows) == report["samples"][partition]
        seen = []
        for row in rows:
            key = row["market"], row["prediction_at"]
            seen.append(key)
            expected = groups[key]
            assert row["samples"] == expected[0]
            for name, value in zip(
                ("absolute_error", "squared_error", "zero_absolute_error", "zero_squared_error"),
                expected[1:],
                strict=True,
            ):
                assert row[name] == pytest.approx(value, rel=1e-12, abs=1e-12)
        assert len(set(seen)) == len(groups)
    index_before = (output / "checkpoints/latest.json").read_bytes()
    chosen = output / references["validation"]["path"]
    chosen.write_bytes(b"corrupt session artifact")
    exported.clear()
    with pytest.raises(ValueError, match="sesión|sesiones|métrica|confirmad"):
        api.run_window_variant(short_view, output, recipe(epochs=1), resume=True)
    assert (output / "checkpoints/latest.json").read_bytes() == index_before
    assert all(path.read_bytes() == values for path, values in prediction_bytes.items())


@requires_cuda
def test_published_precomputed_recipe_completes_fixture_updates_and_checkpoint(
    short_view, tmp_path, monkeypatch, allowed_child
):
    api = engine()
    path = Path(__file__).resolve().parents[2] / (
        "configs/experiments/kronlstm-precomputed/window-month.json"
    )
    published = json.loads(path.read_text())
    adapted = dict(
        train_start="2015-01-01",
        train_end="2015-04-01",
        validation_start="2015-04-01",
        validation_end="2015-05-01",
        epochs=1,
        batch_size=2,
        max_seconds=120.0,
        max_output_bytes=64 * 1024**2,
        max_cuda_bytes=512 * 1024**2,
        cache_bytes=1024**2,
        checkpoint_steps=1,
    )
    configuration = api.WindowVariantRecipe(**(published | adapted))
    assert configuration.loss == "mae" and configuration.estimator == "backprop"
    real_update, real_step = api._update_window, torch.optim.AdamW.step
    updated_ids, steps = [], []

    def observed_update(model, estimator, optimizer, batch, options, device):
        predictor, encoder = model.predictor, model.predictor.price_encoder
        assert predictor.config.temporal_encoder == "kronlstm_precomputed_v1"
        assert encoder.execution == "precomputed" and encoder.configuration["rule"] == "scalar"
        assert all(b.execution == "precomputed" and b.rule == "scalar" for b in encoder.blocks)
        assert type(optimizer) is torch.optim.AdamW and estimator is None
        assert str(device) == "cuda:0" and sum(p.numel() for p in predictor.parameters()) > 1000
        updated_ids.extend(batch["sample_ids"])
        return real_update(model, estimator, optimizer, batch, options, device)

    def observed_step(optimizer, *args, **kwargs):
        steps.append(type(optimizer))
        return real_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(api, "_update_window", observed_update)
    monkeypatch.setattr(torch.optim.AdamW, "step", observed_step)
    output = tmp_path / "precomputed-fixture"
    report = api.run_window_variant(short_view, output, configuration)
    assert report["status"] == "completed" and report["device"] == "cuda:0"
    assert steps == [torch.optim.AdamW] * 3
    assert report["global_step"] == report["work"]["optimizer_updates"] == 3
    assert report["samples"] == dict(train=6, validation=4) and len(report["epochs"]) == 1
    assert len(updated_ids) == len(set(updated_ids)) == 6
    assert report["identity"]["recipe"] == configuration.identity()
    assert report["final_test_opened"] is report["identity"]["final_test_access"] is False
    assert report["evaluation_scope"]["final_test"] == "closed"
    identifiers = {}
    for partition, count in (("train", 6), ("validation", 4)):
        rows = pq.read_table(output / report["predictions"][partition]["path"]).to_pylist()
        identifiers[partition] = {row["sample_id"] for row in rows}
        assert len(rows) == len(identifiers[partition]) == count
    assert identifiers["train"] == set(updated_ids)
    assert identifiers["train"].isdisjoint(identifiers["validation"])
    saved = load_training_state(output / "checkpoints", expected_identity=report["identity"])
    assert saved["global_step"] == 3 and saved["epoch"] == 1 and saved["cursor"] is None
    assert saved["selection"] == report["selection"] and saved["rng"]
    assert saved["model"] and saved["optimizer"]["state"] and saved["estimator"] is None
    contract = saved["model"]["price_encoder._extra_state"]
    assert contract["execution"] == "precomputed"
    assert contract["backend"] == "pytorch_precomputed_cpu_qkv64_v1"
