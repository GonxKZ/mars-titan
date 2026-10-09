"""Piezas propias del arnés de paridad con referencias públicas, sin código de terceros.

El arnés compara el núcleo con `lucidrains/titans-pytorch` y `fla` en un entorno aislado.
Aquí se comprueban sus transcripciones y su criterio de comparación con el núcleo del
proyecto, para que un error del arnés no pase por una diferencia de la referencia.
Solo hay pasos hacia delante, escrituras internas de la memoria y `autograd.grad`.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def harness():
    path = ROOT / "benchmarks/titans_reference_parity.py"
    specification = importlib.util.spec_from_file_location("titans_reference_parity", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def small_case(module, **changes):
    values = dict(
        name="fixture", dim=6, depth=2, flows=2, tokens=7, gate_bias=True, residual=True, seed=5
    )
    values.update(changes)
    return module.Case(**values)


def fixture(module, case):
    ours = module.our_memory(case, torch.float64, "cpu")
    generator = torch.Generator().manual_seed(case.seed)
    x = module.INPUT_SCALE * torch.randn(
        case.flows, case.tokens, case.dim, generator=generator, dtype=torch.float64
    )
    query = torch.randn(case.dim, case.dim, generator=generator, dtype=torch.float64) / 3
    return ours, x, query


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("depth", [1, 2])
def test_minibatch_transcription_with_block_one_is_the_sequential_core(depth, residual):
    module = harness()
    case = small_case(module, depth=depth, residual=residual)
    ours, x, query = fixture(module, case)
    with torch.no_grad():
        reads, state, _ = module.run_ours(ours, x, query, differentiable=False)
    paper_reads, paper_weights = module.paper_minibatch(ours, x, query, block=1)
    torch.testing.assert_close(paper_reads, reads, rtol=1e-12, atol=1e-14)
    for actual, expected in zip(paper_weights, state.weights, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)


def test_minibatch_transcription_evaluates_gradients_at_the_block_start():
    module = harness()
    case = small_case(module)
    ours, x, query = fixture(module, case)
    with torch.no_grad():
        reads, _, _ = module.run_ours(ours, x, query, differentiable=False)
    paper_reads, _ = module.paper_minibatch(ours, x, query, block=4)
    # El primer token de cada bloque usa los mismos pesos en ambos recorridos.
    torch.testing.assert_close(paper_reads[:, 0], reads[:, 0], rtol=1e-12, atol=1e-14)
    assert (paper_reads[:, 1:4] - reads[:, 1:4]).abs().max() > 1e-6


def test_alpha_rows_are_tied_so_forgetting_is_a_scalar_per_token():
    module = harness()
    case = small_case(module)
    ours, x, _ = fixture(module, case)
    weight = ours.alpha_projection.weight
    assert torch.equal(weight, weight[:1].expand_as(weight))
    alpha = ours.alpha_projection(x).sigmoid()
    torch.testing.assert_close(alpha, alpha[..., :1].expand_as(alpha), rtol=0, atol=0)


def test_parenthesised_layer_norm_gradient_matches_autograd():
    module = harness()
    generator = torch.Generator().manual_seed(11)
    dim = 8
    memory = torch.randn(dim, dim, generator=generator, dtype=torch.float64)
    key = torch.randn(1, dim, generator=generator, dtype=torch.float64)
    value = torch.randn(1, dim, generator=generator, dtype=torch.float64)
    scale = 1 + 0.1 * torch.randn(dim, generator=generator, dtype=torch.float64)
    shift = 0.1 * torch.randn(dim, generator=generator, dtype=torch.float64)
    local = memory.clone().requires_grad_(True)
    (expected,) = torch.autograd.grad(
        module.ttt_layer_norm_loss(local, key, value, scale, shift), local
    )
    v_new = module.corrected_layer_norm_gradient(key @ memory, value - key, scale, shift)
    torch.testing.assert_close(2 * key.T @ v_new, expected, rtol=1e-12, atol=1e-12)


def test_comparison_uses_absolute_plus_relative_tolerance_and_rejects_other_shapes():
    module = harness()
    rtol, atol = module.TOLERANCES["float32"]["forward"]
    # Valores en FP64 para que el redondeo de la prueba no altere el cociente medido.
    expected = torch.tensor([0.0, 10.0], dtype=torch.float64)
    inside = expected + torch.tensor([0.9 * atol, 0.9 * (atol + rtol * 10)], dtype=torch.float64)
    outside = expected + torch.tensor([0.0, 1.1 * (atol + rtol * 10)], dtype=torch.float64)
    assert module.compare(inside, expected, torch.float32, "forward")["passed"]
    result = module.compare(outside, expected, torch.float32, "forward")
    assert not result["passed"]
    assert result["worst_tolerance_ratio"] == pytest.approx(1.1, rel=1e-5)
    nan_values = torch.tensor([float("nan"), 10.0], dtype=torch.float64)
    nan = module.compare(nan_values, expected, torch.float32, "forward")
    assert not nan["passed"]
    with pytest.raises(ValueError):
        module.compare(expected[:1], expected, torch.float32, "forward")


def test_reference_checkout_must_be_the_pinned_commit(tmp_path):
    module = harness()
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("0" * 40 + "\n")
    with pytest.raises(RuntimeError, match="exige"):
        module.require_commit(tmp_path, module.LUCIDRAINS_COMMIT)
    (tmp_path / ".git" / "HEAD").write_text(module.LUCIDRAINS_COMMIT + "\n")
    assert module.require_commit(tmp_path, module.LUCIDRAINS_COMMIT) == module.LUCIDRAINS_COMMIT
