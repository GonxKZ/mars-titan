"""Salida recurrente contrastada con ecuaciones escalares e independencia temporal."""

import math

import pytest


@pytest.fixture
def torch_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("La comprobación requiere CUDA y no se sustituye por CPU")
    return torch


@pytest.mark.parametrize("kind", ["rnn", "lstm"])
def test_recurrent_readout_uses_last_hidden_state_not_cell_or_sequence_mean(torch_cuda, kind):
    from mars_titan.profiling import MODALITIES, CostProbe

    torch = torch_cuda
    model = CostProbe(kind, dict.fromkeys(MODALITIES, 1), context=3).to("cuda:0")
    model.head = torch.nn.Linear(160, 1, bias=False).to("cuda:0")
    with torch.no_grad():
        for parameter in model.price_encoder.parameters():
            parameter.zero_()
        model.head.weight.zero_()
        model.head.weight[0, 0] = 1
        if kind == "rnn":
            model.price_encoder.bias_ih_l0.fill_(1)
            model.price_encoder.weight_hh_l0.copy_(torch.eye(32, device="cuda:0") * 0.5)
            expected = 0.0
            for _ in range(3):
                expected = math.tanh(1 + 0.5 * expected)
        else:
            model.price_encoder.bias_ih_l0[64:96] = 1
            cell = 0.5 * math.tanh(1) * (1 + 0.5 + 0.25)
            expected = 0.5 * math.tanh(cell)
    inputs = {
        name: torch.zeros((2, 3, 1) if name == "prices" else (2, 1), device="cuda:0")
        for name in MODALITIES
    }
    torch.testing.assert_close(model(inputs), torch.full((2,), expected, device="cuda:0"))


@pytest.mark.parametrize("kind", ["rnn", "lstm"])
def test_recurrent_windows_do_not_inherit_state_from_previous_predictions(torch_cuda, kind):
    from mars_titan.profiling import MODALITIES, CostProbe

    torch = torch_cuda
    torch.manual_seed(17)
    model = CostProbe(kind, dict.fromkeys(MODALITIES, 2), context=4).to("cuda:0").eval()
    first = {
        name: torch.randn((1, 4, 2) if name == "prices" else (1, 2), device="cuda:0")
        for name in MODALITIES
    }
    second = {name: value + 7 for name, value in first.items()}
    with torch.inference_mode():
        before = model(first)
        model(second)
        after = model(first)
        joint = model({name: torch.cat([first[name], second[name]]) for name in MODALITIES})
    assert torch.equal(before, after)
    torch.testing.assert_close(before, joint[:1], atol=1e-6, rtol=1e-6)
