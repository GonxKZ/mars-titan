"""Adaptadores y continuación sobre padres `quantile_head_v1` con su pinball, sin pasos.

El optimizador del fixture `recorder` no hereda de `torch.optim.Optimizer`, registra los
gradientes y exige pesos sin cambios en cada paso. Las pruebas llegan hasta ese paso.
"""

import json
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json
from mars_titan.environments.actions import ActionGrid
from mars_titan.models.baselines.multimodal import PRESENCE_FUSION, MultimodalReference
from mars_titan.models.predictive_adaptation import adapted_copy, parent_copy
from mars_titan.models.quantile_head import (
    LEVELS,
    QUANTILE_COLUMNS,
    QUANTILE_HEAD,
    pinball_loss,
)
from mars_titan.posttraining import adapter_matrix
from mars_titan.posttraining.heldout import _adjustment, evaluate_partition
from mars_titan.posttraining.inputs import PairedInputs, fit_normalization
from mars_titan.posttraining.parent_cache import ParentCache
from mars_titan.posttraining.parents import load_parent
from mars_titan.posttraining.run import PINBALL_MODE, _loss, run_case
from mars_titan.training.checkpoints import StopRequest
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.posttraining.masked_fixture import masked_ordered, masked_parent, sources

MATRIX = "configs/posttraining/adapter-matrix-v2.json"
DIMENSIONS = dict(prices=5, news=7, charts=6, fundamentals=3, macro=6)


def quantile_parent(kind, layers=2):
    torch.manual_seed(5)
    return (
        MultimodalReference(
            kind,
            DIMENSIONS,
            context=8,
            hidden_size=32,
            layers=layers,
            dropout=0.0,
            transformer=dict(heads=2, feedforward_multiplier=2) if kind == "transformer" else None,
            mask_fusion=PRESENCE_FUSION,
            head=QUANTILE_HEAD,
        )
        .eval()
        .requires_grad_(False)
    )


def batch(size=6, seed=2):
    generator = torch.Generator().manual_seed(seed)
    inputs = {
        name: torch.randn(
            (size, 8, width) if name == "prices" else (size, width), generator=generator
        )
        for name, width in DIMENSIONS.items()
    }
    presence = torch.ones((size, 5), dtype=torch.bool)
    presence[::2, 3] = False
    inputs["fundamentals"] = inputs["fundamentals"] * presence[:, 3:4]
    target = torch.randn(size, generator=generator, dtype=torch.float64) * 0.02
    return inputs, presence, target


def matrix_cases(family):
    declared, digest = adapter_matrix.read_matrix(MATRIX)
    items = adapter_matrix.cases(declared, digest, family, head=QUANTILE_HEAD)
    return {item["id"]: item for item in items}, declared


def model_for(parent, case):
    if "adapter" not in case:
        # Igual que `FrozenParent.continuation`.
        return parent_copy(parent).requires_grad_(True)
    return adapted_copy(
        parent,
        adapter_matrix.targets(case["adapter"], parent),
        seed=adapter_matrix.adapter_seed(case),
    )


def pinball(model, inputs, presence, target):
    case = dict(mode=PINBALL_MODE)
    batch = dict(target=target.numpy())
    output = model(inputs, presence).double()
    grid = SimpleNamespace(values=np.linspace(-0.1, 0.1, 21), scale=0.01)
    return _loss(output, batch, grid, case, None, "cpu"), output


@pytest.mark.parametrize(
    ("kind", "arm"),
    [
        ("gru", "head"),
        ("gru", "head+fusion"),
        ("lstm", "fusion_full_rank"),
        ("dlinear", "fusion"),
        ("transformer", "readout"),
        ("transformer", "head+readout+fusion"),
    ],
)
def test_pinball_reaches_only_the_adapter_and_keeps_the_parent(kind, arm):
    parent = quantile_parent(kind)
    before = {k: v.clone() for k, v in parent.state_dict().items() if torch.is_tensor(v)}
    cases, _ = matrix_cases(kind)
    case = cases[f"seed-42/{arm}"]["case"]
    assert case["mode"] == PINBALL_MODE
    model = model_for(parent, case)
    inputs, presence, target = batch()
    with torch.inference_mode():
        assert torch.equal(model(inputs, presence), parent(inputs, presence))
    model.train()
    loss, output = pinball(model, inputs, presence, target)
    expected = pinball_loss(output, target, reduction="none").mean()
    assert torch.equal(loss, expected)
    loss.backward()
    adapters = {
        name: value
        for name, value in model.named_parameters()
        if "parametrizations" in name and not name.endswith("original")
    }
    assert adapters and all(value.requires_grad for value in adapters.values())
    assert all(value.grad is not None for value in adapters.values())
    # Con U nula el gradiente de V es cero en el primer paso. U o la corrección completa no.
    assert any(value.grad.abs().sum() > 0 for value in adapters.values())
    frozen = [value for name, value in model.named_parameters() if name not in adapters]
    assert frozen and all(not value.requires_grad and value.grad is None for value in frozen)
    assert all(value.grad is None for value in parent.parameters())
    assert all(torch.equal(value, parent.state_dict()[key]) for key, value in before.items())


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_continuation_optimizes_every_weight_with_the_same_pinball(kind):
    parent = quantile_parent(kind)
    cases, _ = matrix_cases(kind)
    case = cases["seed-43/full_continuation"]["case"]
    assert case["mode"] == PINBALL_MODE and "adapter" not in case
    model = model_for(parent, case)
    inputs, presence, target = batch()
    loss, _ = pinball(model.train(), inputs, presence, target)
    loss.backward()
    assert all(value.grad is not None for value in model.parameters())
    assert all(value.grad is None for value in parent.parameters())


@pytest.mark.parametrize("arm", ["head", "head+readout+fusion", "full_continuation"])
def test_quantile_order_survives_any_adapter_or_continuation_weights(arm):
    parent = quantile_parent("transformer")
    cases, _ = matrix_cases("transformer")
    model = model_for(parent, cases[f"seed-44/{arm}"]["case"])
    generator = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for value in model.parameters():
            if value.requires_grad:
                value.add_(torch.randn(value.shape, generator=generator) * 3.0)
    inputs, presence, _ = batch(size=32, seed=4)
    with torch.inference_mode():
        output = model(inputs, presence)
    assert output.shape == (32, len(LEVELS)) and torch.isfinite(output).all()
    assert (torch.diff(output, dim=1) >= 0).all()
    assert not torch.equal(output, parent(inputs, presence))


@pytest.fixture
def edition(tmp_path):
    view, ordered, report = masked_ordered(tmp_path / "data")
    return SimpleNamespace(view=view, ordered=ordered, report=report)


def setup(edition, folder, kind, head=QUANTILE_HEAD):
    path, _ = masked_parent(edition.ordered, folder / "parent", kind, head=head)
    parent = load_parent(edition.ordered, path, device="cpu", diagnostic=True)
    train, validation = sources(edition.ordered)
    cache = ParentCache(
        folder / "parent.sqlite", parent.identity["checkpoint_sha256"], "masked", parent.predict
    )
    data = PairedInputs(train, validation, cache)
    grid = ActionGrid.from_dict(edition.report["grid"])
    return parent, data, grid, fit_normalization(data, batch_size=2), (train, validation, cache)


@pytest.mark.parametrize(
    ("kind", "arm"),
    [("transformer", "head+readout+fusion"), ("gru", "full_continuation"), ("gru", "head")],
)
def test_quantile_case_runs_until_each_step_and_selects_with_the_median(
    tmp_path, recorder, edition, kind, arm
):
    parent, data, grid, scale, handles = setup(edition, tmp_path, kind)
    cases, declared = matrix_cases(kind)
    case = cases[f"seed-42/{arm}"]["case"]
    before = {k: v.clone() for k, v in parent.model.state_dict().items() if torch.is_tensor(v)}
    output = tmp_path / "arm"
    report = run_case(
        data, output, case, grid, scale, parent=parent, batch_size=2, device="cpu", diagnostic=True
    )
    updates = declared["budget"]["epochs"] * data.budget("real", 2)["updates"]
    assert report["status"] == "completed" and report["mode"] == PINBALL_MODE
    assert report["global_step"] == report["total_steps"] == updates
    (optimizer,) = recorder.optimizers
    assert len(optimizer.calls) == updates
    assert all(all(grad is not None for grad in call) for call in optimizer.calls)
    identity = report["identity"]
    assert identity["objective"]["loss"] == "pinball"
    assert identity["objective"]["levels"] == list(LEVELS)
    assert "models/quantile_head.py" in identity["code"]
    assert identity["parent"]["output_head"]["name"] == QUANTILE_HEAD
    # Selección con el MAE por sesión de la mediana. Sin actualizaciones gana el padre.
    validation = report["predictions"]["validation"]["metrics"]
    assert validation["primary"] == "quantile_median" and validation["pinball"] >= 0
    assert validation["session_mae"] == report["baseline"]["session_mae"]
    assert report["selection"]["best_epoch"] == 0
    # Las estadísticas del ajuste usan la mediana. Sin pasos coincide con la caché del padre.
    expected = sum(
        float(np.abs(item["parent"] - item["target"]).sum())
        for item in data.batches(partition="train", condition="real", batch_size=2, epoch=0, seed=0)
    )
    assert len(report["epochs"]) == declared["budget"]["epochs"]
    for epoch in report["epochs"]:
        assert epoch["train"]["absolute_error"] == pytest.approx(expected, rel=1e-6)
    table = pq.read_table(output / "validation-predictions.parquet").to_pydict()
    levels = np.column_stack([table[name] for name in QUANTILE_COLUMNS])
    assert (np.diff(levels, axis=1) >= 0).all()
    np.testing.assert_array_equal(table["prediction"], levels[:, 2])
    np.testing.assert_array_equal(table["prediction"], table["parent"])
    after = parent.model.state_dict()
    assert all(torch.equal(value, after[key]) for key, value in before.items())
    # La evaluación reservada reconstruye el brazo y escribe el esquema común.
    model, action_grid, neural = _adjustment(output / "run.json", report, parent, "cpu")
    dataset = CorpusDataset(edition.view, input_policy=HISTORICAL_MASKED)
    path = tmp_path / "evaluation.parquet"
    evaluate_partition(
        dataset,
        parent,
        "evaluation",
        path,
        stop=StopRequest(),
        device="cpu",
        model=model,
        grid=action_grid,
        neural=neural,
    )
    held = pq.read_table(path).to_pydict()
    assert {"asset_id", "market", "prediction_at", "target", "prediction"} <= set(held)
    levels = np.column_stack([held[name] for name in QUANTILE_COLUMNS])
    np.testing.assert_array_equal(held["prediction"], levels[:, 2])
    np.testing.assert_array_equal(held["prediction"], held["parent"])
    for handle in handles:
        handle.close()


def test_objective_must_match_the_parent_head(tmp_path, recorder, edition):
    options = dict(batch_size=2, device="cpu", diagnostic=True)
    parent, data, grid, scale, handles = setup(edition, tmp_path / "q", "gru")
    declared, digest = adapter_matrix.read_matrix(MATRIX)
    scalar = {item["id"]: item["case"] for item in adapter_matrix.cases(declared, digest, "gru")}
    for name in ("seed-42/head", "seed-42/full_continuation"):
        with pytest.raises(ValueError, match="cuantiles"):
            run_case(data, tmp_path / name, scalar[name], grid, scale, parent=parent, **options)
    for handle in handles:
        handle.close()
    parent, data, grid, scale, handles = setup(edition, tmp_path / "s", "gru", head=None)
    quantile, _ = matrix_cases("gru")
    with pytest.raises(ValueError, match="cuantiles"):
        run_case(
            data,
            tmp_path / "pinball",
            quantile["seed-42/head"]["case"],
            grid,
            scale,
            parent=parent,
            **options,
        )
    assert recorder.optimizers == []
    for handle in handles:
        handle.close()


def test_scalar_parents_keep_their_objective_and_identity(tmp_path, recorder, edition):
    declared, digest = adapter_matrix.read_matrix(MATRIX)
    first, _ = adapter_matrix.read_matrix("configs/posttraining/adapter-matrix-v1.json")
    assert adapter_matrix.objectives(declared, "scalar") == first["objectives"]
    parent, data, grid, scale, handles = setup(edition, tmp_path, "gru", head=None)
    case = {item["id"]: item["case"] for item in adapter_matrix.cases(declared, digest, "gru")}[
        "seed-42/head"
    ]
    assert case["mode"] == "neural_mae"
    report = run_case(
        data,
        tmp_path / "arm",
        case,
        grid,
        scale,
        parent=parent,
        batch_size=2,
        device="cpu",
        diagnostic=True,
    )
    assert "objective" not in report["identity"]
    assert "models/quantile_head.py" not in report["identity"]["code"]
    assert report["predictions"]["validation"]["metrics"]["primary"] == "median"
    for handle in handles:
        handle.close()


def test_matrix_v2_records_the_excluded_linear_correction(tmp_path):
    declared, digest = adapter_matrix.read_matrix(MATRIX)
    excluded = adapter_matrix.excluded_controls(declared, QUANTILE_HEAD)
    assert set(excluded) == {"linear_residual"} and "pinball" in excluded["linear_residual"]
    assert adapter_matrix.excluded_controls(declared, "scalar") == {}
    for family in adapter_matrix.FAMILIES:
        items = adapter_matrix.cases(declared, digest, family, head=QUANTILE_HEAD)
        assert all(item["control"] != "linear_residual" for item in items)
        assert {item["case"]["mode"] for item in items} == {PINBALL_MODE}
    value = json.loads(open(MATRIX).read())
    value["objectives"][QUANTILE_HEAD]["linear_residual"] = "mae"
    atomic_json(tmp_path / "matrix.json", value)
    with pytest.raises(ValueError, match="objetivo"):
        adapter_matrix.read_matrix(tmp_path / "matrix.json")
