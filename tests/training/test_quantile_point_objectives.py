"""Controles de objetivo puntual con la arquitectura de cinco salidas intacta."""

import importlib

import pytest
import torch

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]
LOSSES = [("quantile_point_mae_v1", 2.0, 0.5), ("quantile_point_component_v1", 0.2, 0.05)]


def objective(prediction, target, kind):
    return importlib.import_module("mars_titan.training.quantile_objectives").point_objective(
        prediction, target, kind
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind,value,derivative", LOSSES)
def test_only_the_median_receives_the_declared_absolute_error_gradient(
    device, dtype, kind, value, derivative
):
    predictions = torch.tensor(
        [[-2, -1, 0, 1, 2], [-2, -1, 1, 2, 3]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    target = torch.tensor([2, -1], dtype=dtype, device=device)
    loss = objective(predictions, target, kind)
    assert loss.item() == pytest.approx(value)
    gradient = torch.autograd.grad(loss, predictions)[0]
    expected = torch.tensor(
        [[0, 0, -derivative, 0, 0], [0, 0, derivative, 0, 0]], dtype=dtype, device=device
    )
    torch.testing.assert_close(gradient, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind,_,__", LOSSES)
def test_a_zero_residual_keeps_the_midpoint_subgradient(kind, _, __):
    predictions = torch.zeros(2, 5, dtype=torch.float64, requires_grad=True)
    loss = objective(predictions, torch.zeros(2, dtype=torch.float64), kind)
    assert loss.item() == 0
    assert torch.count_nonzero(torch.autograd.grad(loss, predictions)[0]) == 0


@pytest.mark.parametrize(
    "prediction,target",
    [
        (None, torch.zeros(2)),
        (torch.zeros(2, 5), None),
        (torch.zeros(2, 4), torch.zeros(2)),
        (torch.zeros(2, 5, 1), torch.zeros(2)),
        (torch.zeros(2, 5), torch.zeros(2, 1)),
        (torch.zeros(2, 5), torch.zeros(3)),
        (torch.zeros(0, 5), torch.zeros(0)),
        (torch.zeros(2, 5), torch.zeros(2, dtype=torch.float64)),
        (torch.zeros(2, 5, dtype=torch.int64), torch.zeros(2, dtype=torch.int64)),
    ],
)
def test_incompatible_shapes_precision_and_empty_batches_are_rejected(prediction, target):
    with pytest.raises(ValueError):
        objective(prediction, target, "quantile_point_mae_v1")


@pytest.mark.parametrize("kind", [None, "mae", "pinball", "", [], {}])
def test_a_point_control_cannot_be_enabled_by_another_loss_name(kind):
    with pytest.raises(ValueError):
        objective(torch.zeros(2, 5), torch.zeros(2), kind)


@pytest.mark.cuda
def test_mixing_cuda_predictions_and_cpu_labels_is_rejected():
    with pytest.raises(ValueError):
        objective(torch.zeros(2, 5, device="cuda:0"), torch.zeros(2), "quantile_point_mae_v1")


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_labels_are_not_silently_corrected(value):
    loss = objective(torch.zeros(1, 5), torch.tensor([value]), "quantile_point_mae_v1")
    assert not torch.isfinite(loss)


@pytest.mark.parametrize("device", DEVICES)
def test_equal_encoder_gradients_can_change_after_clipping_the_full_model(device):
    from dataclasses import replace

    from mars_titan.models.quantile_head import pinball_loss
    from mars_titan.models.quantile_routing import QuantileRouting
    from mars_titan.models.titans.financial import FinancialPredictor
    from mars_titan.models.titans.financial_inputs import DecisionBatch
    from mars_titan.training.titans_walk_forward import unfused_attention
    from tests.models.titans.test_financial_adapter import raw_batch
    from tests.models.titans.test_financial_quantile_routing import configuration

    base = configuration()
    with unfused_attention():
        models = [
            FinancialPredictor(c, device=device, dtype=torch.float64)
            for c in (base, replace(base, quantile_routing=QuantileRouting(False, False)))
        ]
        batch = DecisionBatch.from_corpus(
            raw_batch(), base.inputs, device=device, dtype=torch.float64
        )
        target = torch.tensor([0.1, -0.2], device=device, dtype=torch.float64)
        before, after, norms = [], [], []
        for i, model in enumerate(models):
            output = model.prepare(
                batch, model.initial_state(batch.flow_ids, differentiable=True), differentiable=True
            )
            loss = (
                objective(output.quantiles, target, "quantile_point_component_v1")
                if i == 0
                else pinball_loss(output.quantiles, target)
            )
            loss.backward()
            parameters = [p for name, p in model.named_parameters() if not name.startswith("head.")]
            before.append(torch.cat([p.grad.flatten() for p in parameters if p.grad is not None]))
            norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3))
            after.append(torch.cat([p.grad.flatten() for p in parameters if p.grad is not None]))
        torch.testing.assert_close(before[0], before[1], rtol=1e-12, atol=1e-14)
        assert norms[1] > norms[0] > 1e-3
        assert not torch.allclose(after[0], after[1], rtol=1e-4, atol=1e-12)
