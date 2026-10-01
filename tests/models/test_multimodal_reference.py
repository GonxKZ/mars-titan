"""Referencias configurables con cinco entradas y estado independiente por ventana."""

import importlib

import pytest
import torch

from mars_titan.data.embeddings import require_cuda


def model_type():
    try:
        return importlib.import_module("mars_titan.models.baselines.multimodal").MultimodalReference
    except ModuleNotFoundError:
        pytest.fail("Falta la referencia científica configurable")


DIMENSIONS = dict(prices=5, news=8, charts=7, fundamentals=6, macro=9)


def inputs(device, *, batch=3, context=32):
    return {
        key: torch.randn(
            (batch, context, width) if key == "prices" else (batch, width),
            device=device,
            requires_grad=True,
        )
        for key, width in DIMENSIONS.items()
    }


@pytest.mark.parametrize("kind", ["rnn", "lstm", "gru", "dlinear"])
@pytest.mark.parametrize("layers", [1, 2])
def test_every_modality_receives_gradients_with_explicit_architecture(kind, layers):
    cls = model_type()
    device = require_cuda()
    torch.manual_seed(14)
    model = cls(kind, DIMENSIONS, context=32, hidden_size=64, layers=layers, dropout=0.1).to(device)
    values = inputs(device)
    output = model(values)
    assert output.shape == (3,)
    assert model.encode(values).shape == (3, 64)
    output.square().sum().backward()
    assert all(
        v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0
        for v in values.values()
    )
    assert model.architecture == dict(hidden_size=64, layers=layers, dropout=0.1)


@pytest.mark.parametrize("kind", ["rnn", "lstm", "gru", "dlinear"])
def test_windows_do_not_retain_hidden_state_and_evaluation_disables_dropout(kind):
    cls = model_type()
    device = require_cuda()
    torch.manual_seed(19)
    model = (
        cls(kind, DIMENSIONS, context=32, hidden_size=32, layers=2, dropout=0.2).to(device).eval()
    )
    values = inputs(device)
    with torch.no_grad():
        expected = model(values)
        model({key: value * 20 for key, value in values.items()})
        repeated = model(values)
        reversed_batch = model({key: value.flip(0) for key, value in values.items()})
    torch.testing.assert_close(repeated, expected, rtol=0, atol=0)
    torch.testing.assert_close(reversed_batch.flip(0), expected, rtol=1e-5, atol=1e-6)


def test_capacity_changes_with_width_and_depth_and_preserves_input_contract():
    cls = model_type()
    device = require_cuda()
    sizes = []
    for hidden, layers in ((32, 1), (64, 1), (128, 2)):
        model = cls("gru", DIMENSIONS, context=32, hidden_size=hidden, layers=layers).to(device)
        sizes.append(sum(p.numel() for p in model.parameters()))
    assert sizes[0] < sizes[1] < sizes[2]
    values = inputs(device)
    with pytest.raises(ValueError, match="modalidades|entradas"):
        model({k: v for k, v in values.items() if k != "news"})
    with pytest.raises(ValueError, match="forma|dimensi|lote"):
        model({**values, "macro": values["macro"][:1]})
    with pytest.raises(ValueError, match="forma|contexto|ventana"):
        model({**values, "prices": values["prices"][:, :16]})


@pytest.mark.parametrize(
    "options",
    [
        dict(hidden_size=True),
        dict(hidden_size=0),
        dict(layers=0),
        dict(layers=True),
        dict(dropout=float("nan")),
        dict(dropout=1.0),
    ],
)
def test_invalid_architecture_fails_before_allocating_model(options):
    with pytest.raises(ValueError, match="arquitectura|dimensi|regularizaci"):
        model_type()("gru", DIMENSIONS, context=32, **options)
