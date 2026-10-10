"""Comprobaciones en CPU del estudio de divergencia de la memoria de Titans-MAC."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F


def benchmark():
    path = Path(__file__).resolve().parents[3] / "benchmarks/titans_divergence.py"
    if not path.is_file():
        pytest.fail("Falta el estudio reproducible de divergencia de Titans-MAC")
    specification = importlib.util.spec_from_file_location("titans_divergence_benchmark", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_bf16_emulation_rounds_matmuls_and_restores_the_runtime():
    study = benchmark()
    values = torch.randn(3, 5, dtype=torch.float32)
    weight = torch.randn(4, 5, dtype=torch.float32)
    original = F.linear
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    with study.arithmetic("bf16"):
        emulated = F.linear(values, weight)
        product = torch.bmm(values.unsqueeze(0), weight.t().unsqueeze(0))
    expected = original(values.bfloat16().float(), weight.bfloat16().float()).bfloat16().float()
    assert torch.equal(emulated, expected)
    assert torch.equal(product[0], expected)
    assert not torch.equal(emulated, original(values, weight))
    assert F.linear is original
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == flags


def test_tf32_mode_is_scoped():
    study = benchmark()
    before = torch.backends.cuda.matmul.allow_tf32
    with study.arithmetic("tf32"):
        assert torch.backends.cuda.matmul.allow_tf32
    with study.arithmetic("exact"):
        assert not torch.backends.cuda.matmul.allow_tf32
    assert torch.backends.cuda.matmul.allow_tf32 == before


def test_precision_study_shares_parameters_and_reproduces_itself():
    study = benchmark()
    exact = study.model(
        study.CAMPAIGN, flows=2, dim=8, parameters=torch.float32, dtype=torch.float64, device="cpu"
    )
    single = study.model(
        study.CAMPAIGN, flows=2, dim=8, parameters=torch.float32, dtype=torch.float32, device="cpu"
    )
    for left, right in zip(exact.parameters(), single.parameters(), strict=True):
        assert left.dtype == torch.float64
        assert torch.equal(left, right.double())
    tokens = study.stream_tokens("unit_variance", study="precision", dim=8, flows=2, length=24)
    assert torch.equal(tokens, tokens.float().double())
    first = study.run_trajectory(
        study.CAMPAIGN, tokens, precision="cpu_float64", study="precision", dim=8
    )
    second = study.run_trajectory(
        study.CAMPAIGN, tokens, precision="cpu_float64", study="precision", dim=8
    )
    assert torch.equal(first["outputs"], second["outputs"])
    # La reproducción conserva los parámetros FP64 del recibo anterior. El estudio de
    # precisión parte de los mismos valores redondeados a FP32 en todas las precisiones.
    native = study.run_trajectory(
        study.CAMPAIGN, tokens, precision="cpu_float64", study="reproduction", dim=8
    )
    assert not torch.equal(native["outputs"], first["outputs"])
    assert set(first["states"]) == {24}
    assert first["normalization"]["layer_norm_calls"] == 3 * 24
    lower = study.run_trajectory(
        study.CAMPAIGN, tokens, precision="cpu_float32", study="precision", dim=8
    )
    gap = study.relative_curve(lower["outputs"], first["outputs"])
    assert 0 < gap[0] < 1e-5


def test_state_point_round_trip_preserves_layers_and_storage():
    study = benchmark()
    mac = study.model(
        study.CAMPAIGN, flows=3, dim=4, parameters=torch.float32, dtype=torch.float64, device="cpu"
    )
    state = mac.initial_state(3)
    point = study.state_point(state)
    shifted = point + torch.arange(point.numel(), dtype=torch.float64).reshape(point.shape)
    rebuilt = study._with_point(state, shifted, 4)
    mac._validate_state(rebuilt)
    assert torch.equal(study.state_point(rebuilt), shifted)
    assert torch.equal(rebuilt.memory.momentum[0][1].flatten(), shifted[1, 32:48])


def test_identical_rows_stay_at_roundoff_and_perturbations_are_measured():
    study = benchmark()
    tokens = study.stream_tokens("unit_variance", study="precision", dim=8, flows=2, length=16)
    # Las copias exactas solo se separan por el redondeo, que depende de su posición en el
    # lote de la GEMM y de SDPA en CPU. Por eso el piso no es exactamente cero.
    still, outputs = study.free_growth(study.CAMPAIGN, tokens, dim=8, epsilon=0.0)
    assert still.max() < 1e-13
    assert outputs.shape == (16, 3, 2, 8)
    moved, outputs = study.free_growth(study.CAMPAIGN, tokens, dim=8, epsilon=1e-7)
    assert torch.all(moved[0] > 0) and torch.all(moved < 1e-3)
    gap = (outputs[:, 1:] - outputs[:, :1]).flatten(2).norm(dim=2)
    assert torch.allclose(gap / outputs[:, 0].flatten(1).norm(dim=1)[:, None], moved)
    exponent = study.lyapunov(study.CAMPAIGN, tokens, dim=8, renormalize=4, directions=2)
    assert math.isfinite(exponent["exponent_per_step"])
    assert exponent["exponent_min"] <= exponent["exponent_per_step"] <= exponent["exponent_max"]


def test_growth_rate_and_horizons_follow_the_curve():
    study = benchmark()
    steps = torch.arange(1, 2001, dtype=torch.float64)
    curve = 1e-16 * torch.exp(0.02 * steps)
    assert study.growth_rate(curve) == pytest.approx(0.02, rel=1e-9)
    horizons = study.horizons(curve)
    expected = math.ceil(math.log(1e-6 / 1e-16) / 0.02)
    assert horizons["1e-06"] in (expected, expected + 1)
    assert horizons["0.01"] > horizons["0.001"] > horizons["1e-06"]
    assert study.growth_rate(torch.full((100,), 1e-20, dtype=torch.float64)) is None


def test_lyapunov_separates_contracting_and_chaotic_memories():
    study = benchmark()
    tokens = study.stream_tokens("unit_variance", study="precision", dim=64, flows=1, length=64)
    contracting = study.lyapunov("gate_bias", tokens, dim=64, directions=2)
    chaotic = study.lyapunov("v1_residual_layer_norm", tokens, dim=64, directions=2)
    assert contracting["exponent_max"] < 0 < chaotic["exponent_min"]
    assert chaotic["exponent_per_step"] > 0.1
    # En régimen lineal, el exponente no depende del tamaño de la perturbación.
    smaller = study.lyapunov("v1_residual_layer_norm", tokens, dim=64, directions=2, epsilon=1e-10)
    assert abs(smaller["exponent_per_step"] - chaotic["exponent_per_step"]) < 0.02
