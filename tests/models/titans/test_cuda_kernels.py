"""Paridad de la escritura recurrente y de la selección estable FP64 de Titans."""

import pytest
import torch

from mars_titan.models.titans import cuda_kernels as kernels


def one_layer(weight, previous, gradient, gates, *, differentiable):
    result = kernels.try_memory_write(
        (weight,), (previous,), (gradient,), gates, differentiable=differentiable
    )
    return None if result is None else (result[0][0], result[1][0])


def reference_write(weight, previous, gradient, alpha, eta, theta):
    velocity = eta * previous - theta * gradient
    return (1 - alpha) * weight + velocity, velocity


def assert_bits(left, right):
    assert left.shape == right.shape
    assert left.dtype == right.dtype
    assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or not kernels._supported_device(torch.device("cuda:0")),
    reason="Se necesita una GPU NVIDIA con Triton",
)


def test_cpu_keeps_reference_behavior():
    matrix = torch.ones(1, 64, 64)
    gates = torch.zeros(1, 64, 1), torch.ones(1, 1, 1), torch.ones(1, 1, 1)
    assert one_layer(matrix, matrix, matrix, gates, differentiable=False) is None
    x = torch.tensor([[0.0, -0.0, -1, 9, 9]], dtype=torch.float64)
    assert torch.equal(kernels.stable_neighbors(x, 3), torch.tensor([[3, 4, 0]]))


@cuda
@pytest.mark.parametrize("columns", [1, 31, 32, 33, 127, 255, 511, 512, 1003, 1024, 1025])
def test_stable_neighbors_keeps_exact_ids(columns):
    generator = torch.Generator().manual_seed(42)
    x = torch.randint(-4, 5, (11, columns), generator=generator).double().cuda()
    x[0].fill_(float("-inf"))
    x[1].fill_(float("nan"))
    x[2, ::2] = -0.0
    x[2, 1::2] = 0.0
    x[3, ::4] = float("nan")
    x[3, 1::4] = float("inf")
    x[4] = torch.linspace(1, 1 + 2.0e-13, columns, dtype=torch.float64, device=x.device)
    x[5, ::2] = -float("inf")
    x[5, 1::2] = -torch.finfo(torch.float64).max
    for scores in (x, x[:, ::2]):
        for count in {1, min(8, scores.shape[1]), min(3, scores.shape[1])}:
            expected = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :count]
            assert torch.equal(kernels.stable_neighbors(scores, count), expected)


@cuda
@pytest.mark.parametrize("batch", [1, 7, 128])
@pytest.mark.parametrize("outputs", [(0,), (1,), (0, 1)])
def test_write_preserves_values_and_two_derivatives(batch, outputs):
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(120)
    dims = [(batch, 64, 64)] * 3 + [(batch, 64, 1), (batch, 1, 1), (batch, 1, 1)]
    raw = [torch.randn(s, device=device, generator=generator) for s in dims]
    vectors = [torch.randn(s, device=device, generator=generator) for s in dims]
    out_seeds = [torch.randn(dims[0], device=device, generator=generator) for _ in outputs]

    def evaluate(fused):
        args = [x.detach().clone().requires_grad_() for x in raw]
        out = (
            one_layer(*args[:3], args[3:], differentiable=True) if fused else reference_write(*args)
        )
        assert out is not None
        grad = torch.autograd.grad(
            tuple(out[i] for i in outputs), args, out_seeds, create_graph=True, allow_unused=True
        )
        terms = [(g * v).sum() for g, v in zip(grad, vectors, strict=True) if g is not None]
        higher = torch.autograd.grad(sum(terms), args, allow_unused=True)
        return (*out, *grad, *higher)

    for actual, expected in zip(evaluate(True), evaluate(False), strict=True):
        if expected is None:
            assert actual is None
        else:
            assert_bits(actual, expected)


@cuda
def test_write_respects_signed_zeros_and_range():
    z = torch.zeros(3, 64, 64, device="cuda")
    z[0, ::2] = -0.0
    z[1] = torch.finfo(torch.float32).tiny * 0.5
    z[2] = torch.finfo(torch.float32).max * 0.25
    alpha = torch.zeros(3, 64, 1, device=z.device)
    eta = torch.full((3, 1, 1), 0.5, device=z.device)
    theta = torch.full_like(eta, 0.25)
    expected = reference_write(z, z, z, alpha, eta, theta)
    got = one_layer(z, z, z, (alpha, eta, theta), differentiable=False)
    for result, reference in zip(got, expected, strict=True):
        assert_bits(result, reference)


@cuda
def test_memory_layers_validate_before_launch_and_support_transposed_gradient(monkeypatch):
    matrix = torch.randn(3, 64, 64, device="cuda")
    grad = matrix.transpose(1, 2)
    gates = (
        matrix[:, :, :1].contiguous(),
        matrix[:, :1, :1].contiguous(),
        matrix[:, :1, :1].contiguous(),
    )
    got = one_layer(matrix, matrix, grad, gates, differentiable=False)
    for item, reference in zip(got, reference_write(matrix, matrix, grad, *gates), strict=True):
        assert_bits(item, reference)

    def unexpected(*args):
        pytest.fail("El despacho debe comprobar la última capa antes de escribir la primera")

    monkeypatch.setattr(kernels, "_launch_write", unexpected)
    invalid = kernels.try_memory_write(
        (matrix, grad), (matrix, matrix), (grad, grad), gates, differentiable=False
    )
    assert invalid is None


@cuda
@pytest.mark.parametrize("out_of_memory", [False, True])
def test_parameter_seal_matches_individual_transfers(monkeypatch, out_of_memory):
    import hashlib

    from mars_titan.models.titans import financial

    m = torch.nn.Module()
    m.register_parameter("weight", torch.nn.Parameter(torch.randn(4, 6, device="cuda")))
    m.register_buffer("strided", torch.randn(12, 9, device="cuda")[::2, 1::2])
    m.register_buffer("empty", torch.zeros(0, device="cuda"))
    m.register_buffer("flag", torch.tensor(True, device="cuda"))
    m.register_buffer("positive", torch.tensor([0.0, -0.0, float("nan")], device="cuda"))
    m.register_buffer("identifiers", torch.arange(5, device="cuda", dtype=torch.int64))
    if out_of_memory:

        def fail_transfer(*args, **kwargs):
            raise torch.OutOfMemoryError("ensayo de reserva")

        monkeypatch.setattr(torch, "cat", fail_transfer)
    for _ in range(2):
        per_tensor = {
            k: financial._tensor_digest(v) for k, v in (*m.named_parameters(), *m.named_buffers())
        }
        expected = hashlib.sha256(financial.canonical(per_tensor).encode()).hexdigest()
        assert financial.FinancialPredictor._parameter_digest(m) == expected
        with torch.no_grad():
            m.weight[0, 0] += 1


@cuda
def test_missing_scaling_statistics_are_checked_at_the_original_call():
    from mars_titan.models.learning_scale import LearningScale, LearningScaler
    from mars_titan.models.titans.state import deferred_checks

    dims = dict(prices=6, news=2, charts=2, fundamentals=3, macro=3)
    m = LearningScaler(LearningScale(clip=5.0, price_returns=False, zero_median=False), dims).cuda()
    inputs = {
        name: torch.ones((1, 64, size) if name == "prices" else (1, size), device="cuda")
        for name, size in dims.items()
    }
    presence = torch.ones((1, 5), device="cuda", dtype=torch.bool)
    with pytest.raises(ValueError, match="Faltan las estadísticas"):
        with deferred_checks():
            m(inputs, presence)
            m.fitted.fill_(True)


def test_importing_titans_leaves_the_jit_backend_unloaded():
    import subprocess
    import sys

    # Registrar el estado después de PyTorch. Un runtime que ya haya cargado Triton
    # también debe poder importar los modelos sin tocar el controlador GPU.
    program = """
import sys
import torch
before = ('triton' in sys.modules, torch.cuda.is_initialized())
from mars_titan.models.titans import cuda_kernels
assert ('triton' in sys.modules, torch.cuda.is_initialized()) == before
"""
    subprocess.run([sys.executable, "-c", program], check=True, timeout=30)
