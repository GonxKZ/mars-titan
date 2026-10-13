"""Padres neuronales y tabulares de la edición con máscaras, con su identidad."""

import json

import numpy as np
import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED, policy_identity
from mars_titan.data.storage import atomic_json
from mars_titan.models.baselines.multimodal import STRICT_FUSION
from mars_titan.posttraining.parent_cache import ParentCache
from mars_titan.posttraining.parents import FrozenParent, load_parent
from mars_titan.training.predictive_parents import prepare_parent_cache
from tests.posttraining.masked_fixture import masked_ordered, masked_parent, sources

MODALITIES = ("prices", "news", "charts", "fundamentals", "macro")


def cohort(ordered):
    train, validation = sources(ordered)
    with train, validation:
        return [train(i) for i in range(len(train))]


@pytest.mark.parametrize("kind", ["gru", "dlinear", "transformer"])
def test_masked_neural_parent_loads_presence_fusion_and_uses_the_bits(tmp_path, kind):
    _, ordered, _ = masked_ordered(tmp_path)
    report, model = masked_parent(ordered, tmp_path / "parent", kind)
    parent = load_parent(ordered, report, device="cpu", diagnostic=True)
    assert parent.masked and parent.input_policy == HISTORICAL_MASKED
    assert parent.identity["mask_fusion"] == model.configuration["mask_fusion"]
    assert {key: parent.identity[key] for key in ("input_policy", "mask_contract")} == (
        policy_identity(HISTORICAL_MASKED)
    )
    assert parent.identity["prediction_retention"] == "heldout_full_train_sessions_v1"
    model.eval()
    for raw in cohort(ordered):
        inputs, presence = raw["inputs"], raw["presence"]
        with torch.inference_mode():
            expected = model(
                {name: torch.tensor(value) for name, value in inputs.items()},
                torch.tensor(presence),
            )
        np.testing.assert_array_equal(parent.predict(inputs, presence), expected.double().numpy())
        flipped = presence.copy()
        flipped[:, 1] = ~flipped[:, 1]
        # Una ausencia sin relleno nulo se rechaza. Con relleno nulo cambia la predicción.
        assert not np.array_equal(parent.predict(inputs, flipped), parent.predict(inputs, presence))
        with pytest.raises(ValueError, match="bits"):
            parent.predict(inputs)
        broken = dict(inputs, macro=inputs["macro"] + 1)
        if not presence[:, 4].all():
            with pytest.raises(ValueError, match="cero"):
                parent.predict(broken, presence)


def test_parent_and_corpus_must_share_the_policy_and_fusion(tmp_path):
    _, ordered, _ = masked_ordered(tmp_path)
    report, _ = masked_parent(ordered, tmp_path / "strict-parent", policy=False)
    with pytest.raises(ValueError, match="política"):
        load_parent(ordered, report, device="cpu", diagnostic=True)
    report, _ = masked_parent(ordered, tmp_path / "strict-fusion", fusion=STRICT_FUSION)
    with pytest.raises(ValueError, match="fusión"):
        load_parent(ordered, report, device="cpu", diagnostic=True)


@pytest.mark.parametrize("change", ["summary", "predictions", "retention", "code"])
def test_masked_parent_receipt_must_match_its_retention_and_code(tmp_path, change):
    _, ordered, _ = masked_ordered(tmp_path)
    path, _ = masked_parent(ordered, tmp_path / "parent")
    report = json.loads(path.read_text())
    if change == "summary":
        report.pop("train_summary")
    elif change == "predictions":
        report["predictions"]["train"] = {}
    elif change == "retention":
        report["identity"].pop("prediction_retention")
    else:
        report["identity"]["code"]["data/input_policy.py"] = "0" * 64
    atomic_json(path, report)
    with pytest.raises(ValueError):
        load_parent(ordered, path, device="cpu", diagnostic=True)


def test_aligned_parent_cache_rejects_masked_parents(tmp_path):
    _, ordered, _ = masked_ordered(tmp_path)
    path, _ = masked_parent(ordered, tmp_path / "parent")
    report = json.loads(path.read_text())
    report["predictions"] = {"train": {}, "validation": {}}
    report["identity"].pop("prediction_retention")
    report.pop("train_summary")
    atomic_json(path, report)
    with pytest.raises(ValueError, match="estrictos"):
        prepare_parent_cache(ordered, path, tmp_path / "cache")


class Matrix:
    """Modelo tabular técnico que registra la matriz recibida."""

    def __init__(self):
        self.matrices = []

    def predict(self, matrix):
        self.matrices.append(matrix)
        return matrix.astype(np.float64).sum(axis=1)


@pytest.mark.parametrize("masked", [False, True])
def test_tabular_parent_appends_the_bits_after_the_modalities(masked, tmp_path):
    shapes = dict(prices=(4, 5), news=(3,), charts=(2,), fundamentals=(3,), macro=(3,))
    identity = dict(model="ridge", checkpoint_sha256="a" * 64)
    if masked:
        identity |= policy_identity(HISTORICAL_MASKED)
    model = Matrix()
    parent = FrozenParent(model, identity, shapes, "cpu")
    generator = np.random.default_rng(1)
    inputs = {
        name: generator.normal(size=(2, *shape)).astype(np.float32)
        for name, shape in shapes.items()
    }
    presence = np.array([[True, False, True, True, False], [True, True, True, False, True]])
    if masked:
        for index, name in enumerate(MODALITIES):
            inputs[name][~presence[:, index]] = 0
    parent.predict(inputs, presence if masked else None)
    matrix = model.matrices[-1]
    blocks = [inputs[name].reshape(2, -1) for name in MODALITIES]
    assert matrix.shape == (2, sum(block.shape[1] for block in blocks) + 5 * masked)
    np.testing.assert_array_equal(matrix[:, : matrix.shape[1] - 5 * masked], np.hstack(blocks))
    if masked:
        np.testing.assert_array_equal(matrix[:, -5:], presence.astype(np.float32))
        with pytest.raises(ValueError):
            parent.predict(inputs)
    else:
        with pytest.raises(ValueError, match="estricto"):
            parent.predict(inputs, presence)


def test_parent_cache_keys_include_the_presence_bits(tmp_path):
    calls = []

    def predict(inputs, presence=None):
        calls.append(presence)
        base = inputs["charts"].sum(axis=1)
        return base if presence is None else base + presence.sum(axis=1)

    raw = dict(
        prediction_at=10,
        asset_ids=["US/B", "US/A"],
        available_at=np.array([1, 2]),
        inputs={name: np.zeros((2, 3), dtype=np.float32) for name in MODALITIES},
    )
    raw["inputs"]["charts"] = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32)
    with ParentCache(tmp_path / "cache.sqlite", "a" * 64, "masked", predict) as cache:
        strict = cache.predict(raw)
        assert calls == [None]
        first = cache.predict(
            dict(raw, presence=np.array([[1, 0, 1, 0, 0], [1, 1, 1, 0, 0]], bool))
        )
        # La predicción vuelve al orden de entrada aunque la caché ordene los activos.
        np.testing.assert_array_equal(first, strict + [2, 3])
        np.testing.assert_array_equal(calls[-1], [[1, 1, 1, 0, 0], [1, 0, 1, 0, 0]])
        second = cache.predict(dict(raw, presence=np.ones((2, 5), bool)))
        assert len(calls) == 3 and not np.array_equal(first, second)
        cache.predict(dict(raw, presence=np.ones((2, 5), bool)))
        assert len(calls) == 3
        with pytest.raises(ValueError, match="cinco"):
            cache.predict(dict(raw, presence=np.ones((2, 4), bool)))


def test_quantile_parent_contributes_its_median_and_rejects_scalar_matrices(tmp_path):
    from mars_titan.models.quantile_head import QUANTILE_HEAD, median
    from mars_titan.posttraining import adapter_matrix

    _, ordered, _ = masked_ordered(tmp_path)
    path, model = masked_parent(ordered, tmp_path / "parent", "transformer", head=QUANTILE_HEAD)
    parent = load_parent(ordered, path, device="cpu", diagnostic=True)
    assert parent.quantiles and parent.identity["output_head"]["name"] == QUANTILE_HEAD
    model.eval()
    for raw in cohort(ordered):
        with torch.inference_mode():
            expected = median(
                model(
                    {name: torch.tensor(value) for name, value in raw["inputs"].items()},
                    torch.tensor(raw["presence"]),
                )
            )
        np.testing.assert_array_equal(
            parent.predict(raw["inputs"], raw["presence"]), expected.double().numpy()
        )
    # Continuación y adaptadores conservan la cabeza ordenada. El objetivo lo fija el caso.
    assert parent.continuation().emits_quantiles
    assert all(value.requires_grad for value in parent.continuation().parameters())
    assert not any(value.requires_grad for value in parent.model.parameters())
    # La versión 1 solo declara objetivos escalares y no se reinterpreta para cuantiles.
    declared, digest = adapter_matrix.read_matrix("configs/posttraining/adapter-matrix-v1.json")
    with pytest.raises(ValueError, match="escalar"):
        adapter_matrix.plan(
            declared, digest, "transformer", parent.model, updates_per_epoch=1, linear_features=8
        )


@pytest.mark.parametrize("change", ["contract", "head", "code"])
def test_quantile_parent_needs_a_consistent_output_contract(tmp_path, change):
    from mars_titan.models.quantile_head import QUANTILE_HEAD

    _, ordered, _ = masked_ordered(tmp_path)
    path, _ = masked_parent(ordered, tmp_path / "parent", head=QUANTILE_HEAD)
    report = json.loads(path.read_text())
    if change == "contract":
        report["identity"].pop("output_head")
    elif change == "head":
        report["identity"]["case"].pop("head")
    else:
        report["identity"]["code"]["models/quantile_head.py"] = "0" * 64
    atomic_json(path, report)
    with pytest.raises(ValueError, match="inferencia" if change == "code" else "cabeza"):
        load_parent(ordered, path, device="cpu", diagnostic=True)


@pytest.mark.parametrize("change", ["policy", "precision", "flags"])
def test_parent_with_declared_precision_needs_the_policy_its_fit_recorded(tmp_path, change):
    from mars_titan.training.kernel_policy import FP32_STRICT

    _, ordered, _ = masked_ordered(tmp_path)
    path, _ = masked_parent(ordered, tmp_path / "parent", precision=FP32_STRICT)
    report = json.loads(path.read_text())
    if change == "policy":
        report["identity"].pop("kernel_policy")
    elif change == "precision":
        report["identity"]["case"].pop("precision")
    else:
        report["identity"]["kernel_policy"]["cudnn_allow_tf32"] = True
    atomic_json(path, report)
    with pytest.raises(ValueError, match="política de precisión"):
        load_parent(ordered, path, device="cpu", diagnostic=True)
