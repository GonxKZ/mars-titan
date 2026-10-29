"""R1, U1 y U2 de #786 en CUDA frente a CPU, con λ no nulo y sin pasos de optimizador."""

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmarks.core_variants_cost import arms, walk
from benchmarks.objective_variants_cost import BASE, batches, specification
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.training.kernel_policy import FP32_STRICT, apply_kernel_policy

LAMBDAS = [0.5, -0.25, 0.1, 0.3]


@pytest.mark.cuda
@pytest.mark.parametrize("name", [name for name, changes in arms().items() if changes])
def test_core_variant_predictions_and_gradients_match_cpu(name):
    # Tolerancias declaradas antes de ejecutar CUDA, las mismas de los objetivos de #749.
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    apply_kernel_policy(FP32_STRICT)
    try:
        spec = specification()
        targets = [torch.linspace(-0.02, 0.03, 8) for _ in range(3)]
        results = {}
        with sdpa_kernel(SDPBackend.MATH):
            for device in ("cpu", "cuda:0"):
                config = FinancialConfig(
                    spec,
                    variant="mac_online",
                    seed=42,
                    **(BASE | dict(hidden_size=32, max_batch=16) | arms()[name]),
                )
                model = FinancialPredictor(config, dtype=torch.float32, device=device)
                if model.mac.differential is not None:
                    with torch.no_grad():
                        model.mac.differential.weight.copy_(torch.tensor(LAMBDAS))
                    model._seal_parameters()
                inputs = batches(spec, 8, 3, device=device)
                device_targets = [target.to(device) for target in targets]
                prepared = walk(model, inputs, differentiable=True)
                points = torch.cat([value.point_predictions for value in prepared])
                (points - torch.cat(device_targets)).abs().mean().backward()
                gradients = {
                    key: None if p.grad is None else p.grad.detach().cpu()
                    for key, p in model.named_parameters()
                }
                assert all(
                    value is None or torch.isfinite(value).all() for value in gradients.values()
                )
                results[device] = points.detach().cpu(), gradients
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
