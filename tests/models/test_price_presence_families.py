"""El relleno de una sesión ausente en todo el mercado no llega a ninguna familia.

Cada prueba cambia solo el valor de los huecos, con el bit de presencia a cero, y exige las
mismas predicciones y los mismos gradientes, o el rechazo de la frontera que valida en CPU.
No se ejecuta ningún paso de optimizador.
"""

import numpy as np
import pytest
import torch

from mars_titan.data.price_windows import price_window_contract
from mars_titan.memory.write_scores import decision_features
from mars_titan.models.baselines.multimodal import PRESENCE_FUSION, MultimodalReference
from mars_titan.training.tabular_corpus import _matrix
from tests.models.titans.test_financial_adapter import DIMENSIONS, api, raw_batch, specification

WITH_BIT = {**DIMENSIONS, "prices": 6}
KINDS = ("rnn", "lstm", "gru", "dlinear", "transformer")
FILLS = (0.0, 3.75, -1e3)


def contract():
    calendar = dict(start="2019-01-02", end="2019-12-31", decisions_sha256="a" * 64)
    return price_window_contract({"CN": calendar}, {"CN": ["2019-04-29", "2019-04-30"]})


def windows(batch=3, context=64, *, fill=0.0, seed=5):
    generator = np.random.default_rng(seed)
    values = generator.normal(size=(batch, context, 6))
    values[..., 5] = 1
    values[0, [10, 11], 5] = 0
    values[-1, 0, 5] = 0
    values[..., :5] = np.where(values[..., 5:] > 0, values[..., :5], fill)
    return values


def inputs(fill, context=64):
    generator = torch.Generator().manual_seed(3)
    result = {
        name: torch.randn((3, width), generator=generator, dtype=torch.float64)
        for name, width in WITH_BIT.items()
        if name != "prices"
    }
    result["prices"] = torch.tensor(windows(context=context, fill=fill), dtype=torch.float64)
    return result


@pytest.mark.parametrize("kind", KINDS)
def test_reference_predictions_and_gradients_ignore_the_fill(kind):
    outputs, gradients = [], []
    for fill in FILLS:
        torch.manual_seed(7)
        options = dict(transformer=dict(heads=2, feedforward_multiplier=2))
        model = MultimodalReference(
            kind,
            WITH_BIT,
            context=64,
            hidden_size=32,
            layers=1,
            dropout=0.0,
            mask_fusion=PRESENCE_FUSION,
            **(options if kind == "transformer" else {}),
        ).double()
        values = inputs(fill)
        values["prices"].requires_grad_(True)
        presence = torch.ones(3, 5, dtype=torch.bool)
        output = model(values, presence)
        output.sum().backward()
        outputs.append(output.detach())
        gradients.append([p.grad.clone() for p in model.parameters() if p.grad is not None])
        absent = values["prices"][..., 5] == 0
        assert (values["prices"].grad[absent][:, :5] == 0).all()
    for output, grads in zip(outputs[1:], gradients[1:], strict=True):
        assert torch.equal(output, outputs[0])
        assert all(torch.equal(a, b) for a, b in zip(grads, gradients[0], strict=True))


def test_tabular_matrix_ignores_the_fill():
    def batch(fill):
        values = {
            name: np.ones((3, width), dtype=np.float32)
            for name, width in DIMENSIONS.items()
            if name != "prices"
        }
        values["prices"] = windows(fill=fill).astype(np.float32)
        return dict(inputs=values, target=np.zeros(3), presence=np.ones((3, 5), dtype=bool))

    reference = _matrix(batch(0.0), presence=True)
    assert reference.shape[1] == 64 * 6 + sum(DIMENSIONS.values()) - 5 + 5
    for fill in FILLS[1:]:
        assert np.array_equal(_matrix(batch(fill), presence=True), reference)


def test_m3_anomaly_chains_observed_closes_only():
    prices = windows(fill=0.0)
    presence = np.ones((3, 5), dtype=bool)
    presence[:, 3] = False
    fundamentals = np.zeros((3, 3))
    base = decision_features(prices, presence, fundamentals)
    noisy = windows(fill=7.0)
    assert [f.anomaly for f in decision_features(noisy, presence, fundamentals)] == [
        f.anomaly for f in base
    ]
    # Una fila completa coincide con la ventana de cinco canales.
    assert base[1].anomaly == decision_features(prices[:, :, :5], presence, fundamentals)[1].anomaly
    closes = prices[0, prices[0, :, 5] == 1, 3]
    steps = np.diff(closes)
    middle = np.median(steps[:-1])
    expected = abs(steps[-1]) / max(np.median(np.abs(steps[:-1] - middle)), 1e-12)
    assert base[0].anomaly == pytest.approx(expected, rel=0, abs=0)


def titans_specification(*, declared=True, prices=6):
    base = specification()
    representation = dict(base.representation)
    if declared:
        representation["price_window"] = contract()
    return api().FinancialInputSpec(
        source_sha256="a" * 64,
        view_sha256="b" * 64,
        representation=representation,
        dimensions={**DIMENSIONS, "prices": prices},
        input_policy=base.input_policy,
    )


def titans_batch(fill):
    raw = raw_batch()
    raw["inputs"]["prices"] = windows(batch=2, fill=fill).astype(np.float32)
    return raw


def test_titans_accepts_the_bit_only_when_the_representation_declares_it():
    assert titans_specification().dimensions["prices"] == 6
    with pytest.raises(ValueError, match="canales"):
        titans_specification(declared=False)
    with pytest.raises(ValueError, match="canales"):
        titans_specification(prices=5)


def test_titans_rejects_a_nonzero_fill_before_building_tensors_and_predicts_with_gaps():
    spec = titans_specification()
    with pytest.raises(ValueError, match="presencia"):
        api().DecisionBatch.from_corpus(titans_batch(2.5), spec, dtype=torch.float64)
    config = api().FinancialConfig(spec, variant="mac_online", hidden_size=32, layers=1, seed=42)
    model = api().FinancialPredictor(config, dtype=torch.float64)
    batch = api().DecisionBatch.from_corpus(titans_batch(0.0), spec, dtype=torch.float64)
    result = model.prepare(batch, model.initial_state(batch.flow_ids))
    assert torch.isfinite(result.point_predictions).all()


def test_titans_rejects_a_fill_written_after_the_cpu_validation():
    spec = titans_specification()
    config = api().FinancialConfig(spec, variant="mac_online", hidden_size=32, layers=1, seed=42)
    model = api().FinancialPredictor(config, dtype=torch.float64)
    batch = api().DecisionBatch.from_corpus(titans_batch(0.0), spec, dtype=torch.float64)
    first = model.prepare(batch, model.initial_state(batch.flow_ids)).point_predictions
    absent = batch.inputs["prices"][..., 5] == 0
    assert absent.any()
    # La huella de la validación detecta el relleno escrito después sobre el tensor.
    batch.inputs["prices"][..., :5][absent] = 9.5
    with pytest.raises(ValueError, match="después de su validación"):
        model.prepare(batch, model.initial_state(batch.flow_ids))
    again = api().DecisionBatch.from_corpus(titans_batch(0.0), spec, dtype=torch.float64)
    second = model.prepare(again, model.initial_state(again.flow_ids)).point_predictions
    assert torch.equal(first, second)
