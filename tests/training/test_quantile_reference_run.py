"""Runner de referencias con `quantile_head_v1` en CPU, sin modificar pesos.

Se reutiliza el sustituto de AdamW que solo registra gradientes. Las pruebas
recorren lectura, forward, pinball, backward, checkpoint, tablas por fila y panel
de evaluación. CUDA queda sustituida por CPU únicamente dentro de cada prueba.
"""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.forecast_scores import score_sessions, selective_risk
from mars_titan.models.baselines.multimodal import MultimodalReference
from mars_titan.models.quantile_head import (
    CONTRACT,
    LEVELS,
    MEDIAN_INDEX,
    PINBALL,
    QUANTILE_COLUMNS,
    QUANTILE_HEAD,
    pinball_loss,
)
from mars_titan.training.checkpoints import load_training_state
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.training.test_masked_reference_run import (
    HELDOUT,
    cpu_runner,  # noqa: F401
    engine,
    masked_case,
    masked_view,
    run_masked,
)


def quantile_case(kind="transformer", **options):
    return {**masked_case(kind, **options), "head": QUANTILE_HEAD, "loss": PINBALL}


def model_of(report):
    identity = report["identity"]
    return MultimodalReference(
        identity["case"]["kind"],
        identity["dimensions"],
        context=identity["context"],
        mask_fusion=identity["mask_fusion"],
        head=identity["case"].get("head", "scalar"),
        **identity["case"]["architecture"],
    )


@pytest.mark.parametrize("kind", ["transformer", "gru"])
def test_quantile_run_records_its_identity_and_trains_with_pinball(tmp_path, cpu_runner, kind):  # noqa: F811
    manifest = masked_view(tmp_path)
    report = run_masked(manifest, tmp_path / "run", quantile_case(kind, epochs=1))
    assert report["status"] == "completed" and report["final_test_opened"] is False
    identity = report["identity"]
    assert identity["case"]["head"] == QUANTILE_HEAD and identity["case"]["loss"] == PINBALL
    assert identity["output_head"] == CONTRACT
    assert "models/quantile_head.py" in identity["code"]
    # El primer paso registrado debe ser el gradiente de la pinball sobre el primer lote.
    dataset = CorpusDataset(manifest, input_policy=HISTORICAL_MASKED)
    first = next(dataset.batches(partition="train", batch_size=2, epoch=0, seed=42))
    # El sustituto nunca cambia pesos, así que el estado guardado es el inicial.
    state = load_training_state(
        tmp_path / "run/checkpoints", expected_identity=identity, selection="best"
    )
    model = model_of(report)
    model.load_state_dict(state["model"])
    model.train()
    output = engine._forward(model, first, "cpu")
    target = torch.from_numpy(first["target"]).to(torch.float32)
    loss = pinball_loss(output, target, reduction="none").mean()
    expected = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
    for value, recorded in zip(expected, cpu_runner.calls[0], strict=True):
        if value is None:
            assert recorded is None or torch.count_nonzero(recorded) == 0
        else:
            torch.testing.assert_close(recorded, value, rtol=1e-6, atol=1e-9)
    names = [name for name, _ in model.named_parameters()]
    head = dict(zip(names, cpu_runner.calls[0], strict=True))["head.weight"]
    # Los incrementos de las colas reciben gradiente. Una L1 sobre la mediana no lo daría.
    assert head[[0, 1, 3, 4]].abs().sum() > 0


def test_quantile_tables_fit_the_forecast_panel_contract(tmp_path, cpu_runner):  # noqa: F811
    manifest = masked_view(tmp_path)
    report = run_masked(manifest, tmp_path / "run", quantile_case(epochs=1))
    for partition in HELDOUT:
        table = pq.read_table(tmp_path / "run" / report["predictions"][partition]["path"])
        assert set(QUANTILE_COLUMNS) <= set(table.column_names)
        levels = np.column_stack([table[name].to_numpy() for name in QUANTILE_COLUMNS])
        assert (np.diff(levels, axis=1) >= 0).all()
        point = table["prediction"].to_numpy()
        assert point.dtype == levels.dtype == np.float32
        assert np.array_equal(point, levels[:, MEDIAN_INDEX])
        panel = ForecastPanel.from_arrow(table, quantile_columns=QUANTILE_COLUMNS, levels=LEVELS)
        scores = score_sessions(panel)
        assert scores.point_equals_median is True
        assert [interval[0] for interval in scores.intervals] == [0.8, 0.95]
        assert selective_risk(panel, nominal=0.8).full_mae.shape == (panel.sessions,)
        metrics = report["predictions"][partition]["metrics"]
        absolute = np.abs(point.astype(np.float64) - table["target"].to_numpy())
        assert metrics["mae"] == pytest.approx(absolute.mean(), rel=1e-12)


def test_scalar_runs_keep_their_identity_and_tables(tmp_path, cpu_runner):  # noqa: F811
    manifest = masked_view(tmp_path)
    report = run_masked(manifest, tmp_path / "run", masked_case("transformer", epochs=1))
    identity = report["identity"]
    assert "head" not in identity["case"] and "output_head" not in identity
    assert "models/quantile_head.py" not in identity["code"]
    for partition in HELDOUT:
        table = pq.read_table(tmp_path / "run" / report["predictions"][partition]["path"])
        assert table.column_names == [
            "sample_id",
            "asset_id",
            "market",
            "prediction_at",
            "target",
            "prediction",
            "zero",
        ]


def test_scientific_identity_adds_the_head_fingerprint_only_with_the_head(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "cpu-technical-fixture")
    scalar = engine.scientific_identity(kind="transformer", input_policy=HISTORICAL_MASKED)
    quantile = engine.scientific_identity(
        kind="transformer", input_policy=HISTORICAL_MASKED, head=QUANTILE_HEAD
    )
    assert set(quantile["code"]) - set(scalar["code"]) == {"models/quantile_head.py"}
    assert {k: v for k, v in quantile.items() if k != "code"} == {
        k: v for k, v in scalar.items() if k != "code"
    }


@pytest.mark.parametrize("parent_head", ["scalar", QUANTILE_HEAD])
def test_continuations_never_cross_output_heads(tmp_path, cpu_runner, parent_head):  # noqa: F811
    manifest = masked_view(tmp_path)
    scalar, quantile = masked_case("dlinear", epochs=1), quantile_case("dlinear", epochs=1)
    first, second = (scalar, quantile) if parent_head == "scalar" else (quantile, scalar)
    parent = tmp_path / "parent"
    run_masked(manifest, parent, first)
    # El rechazo debe llegar por la comprobación de la cabeza, antes de comparar código.
    with pytest.raises(ValueError, match="población, arquitectura y semilla"):
        run_masked(manifest, tmp_path / "child", second, initialize_from=parent)
    assert not (tmp_path / "child").exists()
    report = json.loads((parent / "run.json").read_text())
    assert report["identity"]["case"].get("head") == first.get("head")


@pytest.mark.parametrize(
    "change,message",
    [
        (dict(loss="mae"), "pinball"),
        (dict(loss="huber"), "pinball"),
        (dict(head="quantile_head_v2"), "cuantiles"),
        (dict(head="scalar"), "cuantiles"),
        (dict(head=None), "cuantiles"),
        (dict(huber_delta=0.0), "Huber"),
    ],
)
def test_invalid_quantile_cases_fail_before_reading_or_creating_a_run(tmp_path, change, message):
    case = {**quantile_case(), **change}
    with pytest.raises(ValueError, match=message):
        run_masked(tmp_path / "missing.json", tmp_path / "run", case)
    assert not (tmp_path / "run").exists()


def test_quantile_head_requires_a_scientific_architecture(tmp_path):
    case = quantile_case("gru")
    case.pop("architecture")
    with pytest.raises(ValueError, match="arquitectura"):
        engine.run_reference_case(tmp_path / "missing.json", tmp_path / "run", case)
    assert not (tmp_path / "run").exists()


def test_scalar_case_cannot_declare_pinball(tmp_path):
    case = {**masked_case(), "loss": PINBALL}
    with pytest.raises(ValueError, match="pérdida"):
        run_masked(tmp_path / "missing.json", tmp_path / "run", case)


def test_point_split_rejects_mismatched_shapes():
    target = np.zeros(3)
    with pytest.raises(ValueError, match="cinco niveles"):
        engine._point(torch.zeros(3), target, True)
    with pytest.raises(ValueError, match="misma forma"):
        engine._point(torch.zeros(3, 5), target, False)
    assert engine._point(torch.arange(15.0).reshape(3, 5), target, True).tolist() == [2, 7, 12]
