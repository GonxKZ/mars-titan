"""Entradas del adaptador y normalización por ventana con la edición con máscaras."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED, policy_identity
from mars_titan.environments.actions import ActionGrid
from mars_titan.posttraining import run
from mars_titan.posttraining.augmented_inputs import AugmentedInputs
from mars_titan.posttraining.inputs import PairedInputs, fit_normalization
from mars_titan.posttraining.parent_cache import ParentCache
from tests.posttraining.masked_fixture import change_after_training, masked_ordered, sources

MODALITIES = ("prices", "news", "charts", "fundamentals", "macro")


class Predictor:
    """Padre técnico que depende de los bits y registra cada llamada."""

    def __init__(self):
        self.calls = []

    def __call__(self, inputs, presence=None):
        self.calls.append(presence)
        return inputs["charts"].mean(axis=1) + 0.01 * presence.sum(axis=1)


def paired(ordered, folder, predictor=None):
    train, validation = sources(ordered)
    cache = ParentCache(folder / "parent.sqlite", "a" * 64, "masked", predictor or Predictor())
    return PairedInputs(train, validation, cache), (train, validation, cache)


def close(handles):
    for handle in handles:
        handle.close()


def test_features_add_the_five_bits_before_the_parent_prediction(tmp_path):
    _, ordered, report = masked_ordered(tmp_path)
    predictor = Predictor()
    data, handles = paired(ordered, tmp_path, predictor)
    widths = {name: int(np.prod(shape)) for name, shape in data.shapes.items()}
    assert data.features == sum(widths.values()) + 5 + 1
    assert data.identity["input_policy"] == HISTORICAL_MASKED
    rows = 0
    for batch in data.batches(partition="train", condition="real", batch_size=2):
        count = len(batch["target"])
        rows += count
        features = batch["features"]
        assert features.shape == (count, data.features) and features.dtype == np.float32
        column = 0
        for name in MODALITIES:
            width = widths[name]
            np.testing.assert_array_equal(
                features[:, column : column + width], batch["inputs"][name].reshape(count, -1)
            )
            column += width
        np.testing.assert_array_equal(features[:, column : column + 5], batch["presence"])
        np.testing.assert_array_equal(features[:, -1], batch["parent"].astype(np.float32))
        np.testing.assert_allclose(
            batch["parent"],
            batch["inputs"]["charts"].mean(axis=1) + 0.01 * batch["presence"].sum(axis=1),
        )
    assert rows == report["counts"]["train"]
    assert all(call is not None and call.dtype == np.bool_ for call in predictor.calls)
    close(handles)


@pytest.mark.parametrize("condition", ["real_resampled", "real_synthetic"])
def test_masked_edition_only_admits_the_real_condition(tmp_path, condition):
    _, ordered, _ = masked_ordered(tmp_path)
    data, handles = paired(ordered, tmp_path)
    with pytest.raises(ValueError, match="real"):
        next(data.batches(partition="train", condition=condition, batch_size=2))
    with pytest.raises(ValueError, match="real"):
        data.budget(condition, 2)
    train, validation, cache = handles
    with pytest.raises(ValueError, match="real"):
        AugmentedInputs(train, validation, cache, synthetic=lambda index: None)
    with pytest.raises(TypeError):
        PairedInputs(train, validation, cache, synthetic=lambda index: None)
    close(handles)


def test_normalizer_is_bound_to_the_training_window_only(tmp_path):
    _, ordered, report = masked_ordered(tmp_path)
    data, handles = paired(ordered, tmp_path)
    statistics = fit_normalization(data, batch_size=2)
    assert {key: statistics[key] for key in ("input_policy", "mask_contract")} == (
        policy_identity(HISTORICAL_MASKED)
    )
    assert "train_sha256" not in statistics
    assert statistics["feature_order"] == [*MODALITIES, "presence", "parent_prediction"]
    assert statistics["window"] == dict(
        train_partition_sha256=report["partitions"]["train"]["sha256"],
        train_rows=report["counts"]["train"],
        train_cohorts=len(report["partitions"]["train"]["cohorts"]),
        train_bounds={"US": list(data.train.market_bounds["US"])},
    )
    assert len(statistics["mean"]) == len(statistics["scale"]) == data.features
    # Los bits de precios y gráficos son constantes y conservan escala uno.
    presence = slice(data.features - 6, data.features - 1)
    assert statistics["scale"][presence][0] == statistics["scale"][presence][2] == 1.0
    close(handles)


class Sealed:
    """Fuente de validación que falla si alguien lee una de sus cohortes."""

    def __init__(self, source):
        self.source = source

    def __getattr__(self, name):
        return getattr(self.source, name)

    def __len__(self):
        return len(self.source)

    def __call__(self, position):
        raise AssertionError("La normalización ha leído validación")


def test_fit_never_opens_validation(tmp_path):
    _, ordered, _ = masked_ordered(tmp_path)
    data, handles = paired(ordered, tmp_path)
    data.validation = Sealed(data.validation)
    fit_normalization(data, batch_size=2)
    with pytest.raises(AssertionError, match="validación"):
        next(data.batches(partition="validation", condition="real", batch_size=2))
    close(handles)


def test_changing_later_partitions_leaves_the_window_normalizer_unchanged(tmp_path):
    _, ordered, before = masked_ordered(tmp_path / "before")
    train_moments = {row[0] for row in before["partitions"]["train"]["cohorts"]}
    _, changed, after = masked_ordered(tmp_path / "after", change_after_training(train_moments))
    first, handles_first = paired(ordered, tmp_path / "before")
    second, handles_second = paired(changed, tmp_path / "after")
    # El cambio existe: validación cambia sus objetivos y su archivo, el ajuste no.
    assert (
        before["partitions"]["validation"]["sha256"] != after["partitions"]["validation"]["sha256"]
    )
    assert before["partitions"]["train"] == after["partitions"]["train"]
    assert json.loads(ordered.read_text())["source_sha256"] != after["source_sha256"]
    targets = [
        np.concatenate(
            [
                b["target"]
                for b in data.batches(partition="validation", condition="real", batch_size=2)
            ]
        )
        for data in (first, second)
    ]
    assert not np.array_equal(*targets)
    assert fit_normalization(first, batch_size=2) == fit_normalization(second, batch_size=2)
    grids = [ActionGrid.from_dict(report["grid"]).to_dict() for report in (before, after)]
    assert {k: v for k, v in grids[0].items() if k != "source_sha256"} == {
        k: v for k, v in grids[1].items() if k != "source_sha256"
    }
    close(handles_first + handles_second)


def validation_arguments(data, ordered):
    report = json.loads(ordered.read_text())
    parent = SimpleNamespace(
        identity=dict(
            checkpoint_sha256="a" * 64,
            source_sha256=data.train.source_sha256,
            counts=dict(report["counts"]),
            **policy_identity(HISTORICAL_MASKED),
        )
    )
    case = dict(
        mode="mae",
        condition="real",
        seed=1,
        epochs=1,
        learning_rate=0.001,
        weight_decay=0.0,
        clip_norm=1.0,
        beta=0.1,
        behavior_epsilon=0.001,
        auxiliary_samples=4,
    )
    options = dict(
        parent=parent,
        batch_size=2,
        device="cuda:0",
        diagnostic=False,
        lease=None,
        checkpoint_seconds=60,
        max_updates=None,
    )
    return case, ActionGrid.from_dict(report["grid"]), options


def test_run_rejects_a_normalizer_from_another_window_or_edition(tmp_path, monkeypatch):
    _, ordered, _ = masked_ordered(tmp_path)
    data, handles = paired(ordered, tmp_path)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(run, "require_device", lambda *args: None)
    case, grid, options = validation_arguments(data, ordered)
    statistics = fit_normalization(data, batch_size=2)
    assert run._validate_run(data, case, grid, statistics, **options)[1]["rows"] == 3
    other = json.loads(json.dumps(statistics))
    other["window"]["train_partition_sha256"] = "f" * 64
    legacy = dict(statistics, train_sha256=data.train.manifest_sha256)
    strict = {k: v for k, v in statistics.items() if k not in {"input_policy", "mask_contract"}}
    for changed in (other, legacy, strict):
        with pytest.raises(ValueError, match="normalización"):
            run._validate_run(data, case, grid, changed, **options)
    options["parent"].identity.pop("input_policy")
    with pytest.raises(ValueError, match="política"):
        run._validate_run(data, case, grid, statistics, **options)
    close(handles)
