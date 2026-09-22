import importlib

import pytest


def module():
    try:
        return importlib.import_module("mars_titan.profiling")
    except ModuleNotFoundError:
        pytest.fail("Las referencias de coste multimodal todavía no existen")


@pytest.mark.parametrize("kind", ["mlp", "gru", "dlinear", "rnn", "lstm"])
def test_cost_probes_use_every_modality_and_macro(kind):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("La prueba de aceleración requiere CUDA, no se sustituye por CPU")
    torch.manual_seed(42)
    dimensions = {"prices": 5, "news": 4, "charts": 3, "fundamentals": 2, "macro": 2}
    context = 64 if kind == "dlinear" else 4
    model = module().CostProbe(kind, dimensions, context=context).to("cuda:0")
    batch = {
        k: torch.ones(
            (2, context, d) if k == "prices" else (2, d), device="cuda:0", requires_grad=True
        )
        for k, d in dimensions.items()
    }
    output = model(batch)
    assert tuple(output.shape) == (2,)
    output.sum().backward()
    assert all(v.grad is not None and torch.count_nonzero(v.grad) > 0 for v in batch.values())
    with pytest.raises(ValueError, match="modalidades"):
        model({k: v for k, v in batch.items() if k != "charts"})


def test_cost_projection_does_not_claim_unknown_architecture_speed():
    pytest.importorskip("torch")
    result = module().project_cost(
        samples=10000, seconds_per_sample=0.01, epochs=20, folds=5, seeds=3, models=2
    )
    assert result["training_seconds"] == 60000
    assert result["architecture_scope"] == "measured_cost_probes_only"
