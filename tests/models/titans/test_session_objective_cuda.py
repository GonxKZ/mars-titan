"""Paridad CUDA pendiente de una ventana libre, con MATH y sin optimizador."""

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmarks.objective_variants_cost import BASE, MODELS, batches, specification
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.training.kernel_policy import FP32_STRICT, apply_kernel_policy


@pytest.mark.cuda
@pytest.mark.parametrize("name", list(MODELS))
def test_complete_session_predictions_and_gradients_match_cpu(name):
    # Tolerancias declaradas antes de ejecutar CUDA. No se usan para ocultar no finitos.
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    apply_kernel_policy(FP32_STRICT)
    try:
        spec = specification()
        results = {}
        with sdpa_kernel(SDPBackend.MATH):
            for device in ("cpu", "cuda:0"):
                config = FinancialConfig(
                    spec,
                    variant="mac_online",
                    seed=42,
                    **(BASE | dict(hidden_size=32, max_batch=16) | MODELS[name]),
                )
                model = FinancialPredictor(config, dtype=torch.float32, device=device)
                inputs = batches(spec, 8, 2, device=device)
                state = model.initial_state(inputs[0].flow_ids, differentiable=True)
                points, losses = [], []
                for batch in inputs:
                    prepared = model.prepare(
                        batch, state, differentiable=True, session_complete=True
                    )
                    state = prepared.next_state
                    target = torch.linspace(-0.02, 0.03, 8, device=device)
                    losses.append(
                        model.output_objective.loss(
                            prepared.outputs, target, [("US", batch.prediction_at[0])] * 8
                        )
                        if model.output_objective is not None
                        else (prepared.point_predictions - target).abs().mean()
                    )
                    points.append(prepared.point_predictions.detach().cpu())
                torch.stack(losses).mean().backward()
                gradients = {
                    key: None if p.grad is None else p.grad.detach().cpu()
                    for key, p in model.named_parameters()
                }
                assert all(
                    value is None or torch.isfinite(value).all() for value in gradients.values()
                )
                results[device] = torch.stack(points), gradients
        cpu, cuda = results["cpu"], results["cuda:0"]
        torch.testing.assert_close(cuda[0], cpu[0], rtol=1e-4, atol=1e-5)
        assert set(cuda[1]) == set(cpu[1])
        for key, expected in cpu[1].items():
            if expected is None:
                assert cuda[1][key] is None
            else:
                torch.testing.assert_close(cuda[1][key], expected, rtol=1e-3, atol=1e-5)
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
