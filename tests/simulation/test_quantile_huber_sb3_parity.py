"""Paridad de la pérdida cuantílica de Huber de QR-DQN con sb3-contrib 2.9.0.

Se compara `sb3_contrib.common.utils.quantile_huber_loss` con el oráculo FP64 de
`native/tests/rl_variety_reference.py`, su archivo versionado y, cuando está compilado, el
núcleo C++ de `native/src/quantile_dqn.cpp`. Solo se usa esa función. Los algoritmos, los
entornos y `set_random_seed` de Stable-Baselines3 quedan fuera, porque tocan los
generadores globales y su PPO recorta en lugar de penalizar con KL. No se da ningún paso de
entorno ni de optimizador.

sb3-contrib fija κ = 1 y devuelve la media del lote de sum_i mean_j |τ_i − 1{u_ij < 0}| L₁(u_ij).
La pérdida por transición se obtiene con lotes de una fila. Sin `cum_prob` calcula los
niveles τ en FP32, así que aquí se le pasan los puntos medios FP64 de forma explícita.

Tolerancias declaradas antes de ejecutar:

- FP64: |a − b| ≤ 1e-12 elemento a elemento en pérdidas y gradientes.
- FP32: la de la ruta FP32 de la etapa nativa en `rl_variety`, |a − b| ≤ 1e-9 + 1e-5 · S,
  con b el oráculo FP64 evaluado sobre las mismas entradas redondeadas a FP32. La etapa
  nativa la declara para la pérdida, una suma de términos no negativos, donde S = |b|. En
  el gradiente los términos cambian de signo y se cancelan, así que S es la suma de sus
  valores absolutos, (1/N) Σ_j |τ_i − 1{u_ij < 0}| |L₁'(u_ij)|, la escala habitual de la
  cota de error de una suma. Con S = |b| también en el gradiente la comparación falló en un
  elemento casi nulo por cancelación (2,4·10⁻⁸ frente a una cota de 10⁻⁹ más 10⁻⁵ |b|).
- Núcleo nativo: el ejecutable `rl_variety_tests` aplica su propia tolerancia,
  |a − b| ≤ 1e-15 + 1e-12 · |b|, a los valores de sb3-contrib escritos en una copia del archivo.

Intercambiar la suma sobre i y la media sobre j no cambia el valor, porque hay tantos
cuantiles predichos como átomos del objetivo. Sí se detectan una media en lugar de la suma
sobre i, los niveles τ en el eje del objetivo, el signo de la indicadora y otro κ.
"""

import copy
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.suite_support import reference_module

pytestmark = pytest.mark.external_reference

ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / "native" / "tests" / "rl_variety_reference.py"
FIXTURE = ROOT / "native" / "tests" / "fixtures" / "rl_variety_reference.json"
PPO = ROOT / "build" / "native" / "native-ppo-release" / "mars-titan-ppo"
ATOL64 = 1e-12
RTOL32, ATOL32 = 1e-5, 1e-9
# Geometría de las identidades qr_dqn y qr_dqn_cvar (quantile_dqn.hpp).
QUANTILES, BATCH = 32, 64


@pytest.fixture(scope="module")
def loss():
    module = reference_module("sb3_contrib")
    assert module.__version__ == "2.9.0"
    return reference_module("sb3_contrib.common.utils").quantile_huber_loss


@pytest.fixture(scope="module")
def oracle():
    """El script de referencia como módulo, sin ejecutar su `main`."""
    spec = importlib.util.spec_from_file_location("rl_variety_reference", ORACLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def per_transition(loss, oracle, predicted, targets, dtype=torch.float64):
    """Pérdida de sb3-contrib por transición y su gradiente respecto a los cuantiles."""
    current = torch.tensor(predicted, dtype=dtype, requires_grad=True)
    target = torch.tensor(targets, dtype=dtype)
    tau = torch.tensor(oracle.midpoints(current.shape[1]), dtype=dtype).view(1, -1, 1)
    values = torch.stack(
        [loss(current[b : b + 1], target[b : b + 1], cum_prob=tau) for b in range(len(current))]
    )
    (gradient,) = torch.autograd.grad(values.sum(), current)
    return values.detach().double().numpy(), gradient.double().numpy()


def campaign_case(seed=31):
    """Lote con la geometría de la campaña y errores a los dos lados de κ = 1."""
    rng = np.random.default_rng(seed)
    predicted = np.sort(rng.normal(0.0, 1.0, (BATCH, QUANTILES)), axis=1)
    targets = rng.normal(0.0, 1.0, (BATCH, QUANTILES))
    error = targets[:, None, :] - predicted[:, :, None]
    assert (np.abs(error) < 1).any() and (np.abs(error) > 1).any()
    assert (error < 0).any() and (error > 0).any()
    return predicted, targets


def close(actual, expected, label, rtol=0.0, atol=ATOL64, scale=None):
    difference = np.abs(np.asarray(actual) - np.asarray(expected))
    bound = atol + rtol * np.abs(expected if scale is None else scale)
    assert difference.shape == np.shape(expected), label
    assert bool(np.all(difference <= bound)), f"{label}: máximo {difference.max():.3e}"


def kappa_one_cases():
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["quantile"]
    selected = [case for case in cases if case["kappa"] == 1.0]
    # qr_linear usa κ = 0,01 y solo lo cubre el oráculo propio.
    assert [case["name"] for case in selected] == ["qr_mean", "qr_cvar"]
    return selected


def test_sb3_matches_the_versioned_oracle_with_kappa_one(loss, oracle):
    for case in kappa_one_cases():
        predicted, targets = np.array(case["predicted"]), np.array(case["targets"])
        values, gradient = per_transition(loss, oracle, predicted, targets)
        close(values, case["loss"], f"{case['name']} pérdida")
        close(gradient, case["gradient"], f"{case['name']} gradiente")
        tau = torch.tensor(case["midpoints"]).view(1, -1, 1)
        batch = loss(torch.tensor(predicted), torch.tensor(targets), cum_prob=tau)
        close(float(batch), np.mean(case["loss"]), f"{case['name']} media del lote")


def test_sb3_matches_the_oracle_on_the_campaign_geometry(loss, oracle):
    predicted, targets = campaign_case()
    expected = oracle.quantile_huber(predicted, targets, 1.0)
    values, gradient = per_transition(loss, oracle, predicted, targets)
    close(values, expected[0], "pérdida")
    close(gradient, expected[1], "gradiente")


def gradient_scale(oracle, predicted, targets):
    """Suma de los valores absolutos de los términos de cada componente del gradiente."""
    error = targets[:, None, :] - predicted[:, :, None]
    tau = oracle.midpoints(predicted.shape[1])[None, :, None]
    weight = np.abs(tau - (error < 0))
    return (weight * np.minimum(np.abs(error), 1.0)).mean(axis=2)


def test_fp32_stays_within_the_native_fp32_tolerance(loss, oracle):
    predicted, targets = campaign_case(37)
    predicted = predicted.astype(np.float32).astype(np.float64)
    targets = targets.astype(np.float32).astype(np.float64)
    expected = oracle.quantile_huber(predicted, targets, 1.0)
    values, gradient = per_transition(loss, oracle, predicted, targets, torch.float32)
    close(values, expected[0], "pérdida FP32", RTOL32, ATOL32)
    scale = gradient_scale(oracle, predicted, targets)
    close(gradient, expected[1], "gradiente FP32", RTOL32, ATOL32, scale=scale)


def test_default_levels_are_float32_and_exact_only_for_powers_of_two(loss, oracle):
    """Sin `cum_prob`, sb3-contrib calcula τ en FP32. Con 32 niveles son exactos, con 200 no."""
    for count, exact in ((QUANTILES, True), (200, False)):
        rng = np.random.default_rng(count)
        current = torch.tensor(rng.normal(size=(8, count)))
        target = torch.tensor(rng.normal(size=(8, count)))
        tau = torch.tensor(oracle.midpoints(count)).view(1, -1, 1)
        explicit = float(loss(current, target, cum_prob=tau))
        default = float(loss(current, target))
        assert (default == explicit) is exact, count
        assert abs(default - explicit) <= 1e-6 * abs(explicit)


def variant(predicted, targets, change):
    """La pérdida por transición con una convención cambiada, para medir la sensibilidad."""
    count = predicted.shape[1]
    error = targets[:, None, :] - predicted[:, :, None]
    kappa = {"kappa_half": 0.5, "kappa_two": 2.0}.get(change, 1.0)
    magnitude = np.abs(error)
    huber = np.where(magnitude <= kappa, 0.5 * error**2, kappa * (magnitude - 0.5 * kappa))
    tau = (2 * np.arange(count) + 1) / (2 * count)
    tau = tau[None, None, :] if change == "tau_on_target_axis" else tau[None, :, None]
    below = (error > 0) if change == "indicator_sign" else (error < 0)
    weighted = np.abs(tau - below) * huber
    if change == "mean_over_quantiles":
        return weighted.mean(axis=(1, 2))
    return weighted.mean(axis=2).sum(axis=1)


@pytest.mark.parametrize(
    "change",
    ["mean_over_quantiles", "tau_on_target_axis", "indicator_sign", "kappa_half", "kappa_two"],
)
def test_the_parity_detects_a_changed_convention(loss, oracle, change):
    predicted, targets = campaign_case()
    values, _ = per_transition(loss, oracle, predicted, targets)
    close(variant(predicted, targets, "none"), values, "fórmula sin cambios")
    difference = np.abs(variant(predicted, targets, change) - values)
    assert difference.max() > 1e3 * ATOL64


def native_check():
    """Ejecutable `rl_variety_tests` junto al mars-titan-ppo declarado o compilado."""
    declared = os.environ.get("MARS_TITAN_PPO_EXECUTABLE")
    executable = (Path(declared) if declared else PPO).parent / "rl_variety_tests"
    if not executable.is_file():
        pytest.skip(
            "Falta rl_variety_tests compilado junto a mars-titan-ppo "
            "(preset native-ppo-release o MARS_TITAN_PPO_EXECUTABLE)"
        )
    return executable


def run_native(executable, document, path):
    path.write_text(json.dumps(document, indent=1, sort_keys=True, allow_nan=False) + "\n")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1")
    return subprocess.run(
        [str(executable), str(path)], env=environment, capture_output=True, text=True, timeout=120
    )


@pytest.mark.native_binding
def test_native_loss_matches_sb3_through_the_rl_variety_check(loss, oracle, tmp_path):
    """El núcleo C++ reproduce los valores de sb3-contrib con la tolerancia de `rl_variety`.

    Se sustituyen en una copia del archivo la pérdida y el gradiente de los casos con κ = 1
    por los de sb3-contrib. Un control con una pérdida alterada en 1e-9 relativo debe fallar,
    lo que muestra que el ejecutable compara de verdad esos valores.
    """
    executable = native_check()
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for case in document["quantile"]:
        if case["kappa"] == 1.0:
            predicted, targets = np.array(case["predicted"]), np.array(case["targets"])
            values, gradient = per_transition(loss, oracle, predicted, targets)
            case["loss"], case["gradient"] = values.tolist(), gradient.tolist()
    result = run_native(executable, document, tmp_path / "sb3.json")
    assert result.returncode == 0, result.stdout + result.stderr
    broken = copy.deepcopy(document)
    case = next(item for item in broken["quantile"] if item["name"] == "qr_mean")
    case["loss"][0] *= 1 + 1e-9
    result = run_native(executable, broken, tmp_path / "broken.json")
    assert result.returncode != 0
    assert "qr_mean" in result.stdout + result.stderr
