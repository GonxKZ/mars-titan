"""Comprueba ejemplos algebraicos, sin datos financieros ni entrenamiento."""

import argparse
import hashlib
import itertools
import json
import math
import platform
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def entropy(rows):
    counts = Counter(rows)
    return -sum((n / len(rows)) * math.log2(n / len(rows)) for n in counts.values())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_scalar_delta():
    differences = []
    for seed in (42, 43, 44):
        rng = np.random.default_rng(seed)
        for dimension in (2, 8, 32):
            for _ in range(40):
                key = rng.normal(size=dimension)
                key /= np.linalg.norm(key)
                decay = float(rng.uniform(0, 1))
                step = float(rng.uniform(0, 2 - decay))
                operator = (1 - decay) * np.eye(dimension) - step * np.outer(key, key)
                norm = float(np.linalg.norm(operator, 2))
                bound = max(abs(1 - decay), abs(1 - decay - step * float(key @ key)))
                np.testing.assert_allclose(norm, bound, atol=2e-14, rtol=2e-14)
                require(norm <= 1 + 2e-14, "La regla escalar amplifica la norma.")
                differences.append(abs(norm - bound))
    counterexample = 0.9 * np.eye(2) - 1.95 * np.diag([1.0, 0.0])
    amplification = float(np.linalg.norm(counterexample, 2))
    np.testing.assert_allclose(amplification, 1.05, atol=2e-14, rtol=2e-14)
    return {
        "random_cases": len(differences),
        "seeds": [42, 43, 44],
        "dimensions": [2, 8, 32],
        "maximum_norm_formula_difference": max(differences),
        "atol": 2e-14,
        "rtol": 2e-14,
        "decay_counterexample": {
            "forgetting_fraction": 0.1,
            "step": 1.95,
            "key_norm": 1,
            "spectral_norm": amplification,
        },
    }


def check_channel_gate():
    key = np.array([1.0, 1.0]) / math.sqrt(2)
    erase = np.array([0.9, 0.1])
    operator = 0.95 * (np.eye(2) - np.outer(key, erase * key))
    probe = np.array([1.0, -2.0])
    ratio = float(np.linalg.norm(operator @ probe) ** 2 / np.linalg.norm(probe) ** 2)
    radius = float(max(abs(np.linalg.eigvals(operator))))
    np.testing.assert_allclose(ratio, 1.0730725, atol=1e-14, rtol=1e-14)
    require(ratio > 1 and radius < 1, "El contraejemplo no cumple sus condiciones.")
    boundary = np.eye(2) - np.outer(key, np.array([1.0, 0.0]) * key)
    np.testing.assert_allclose(
        np.linalg.norm(boundary, 2), math.sqrt((3 + math.sqrt(5)) / 4), rtol=1e-14
    )
    return {
        "key": key.tolist(),
        "erase": erase.tolist(),
        "retention": 0.95,
        "operator": operator.tolist(),
        "probe": probe.tolist(),
        "squared_norm_gain": ratio,
        "spectral_norm": float(np.linalg.norm(operator, 2)),
        "spectral_radius": radius,
        "boundary_operator_norm": float(np.linalg.norm(boundary, 2)),
    }


def check_proximal_cohort():
    differences = []
    residuals = []
    conditions = []
    for seed, dimension, rows, step, retention in itertools.product(
        (42, 43, 44), (2, 8, 32), (1, 17), (0.0, 0.2, 100.0), (0.0, 0.95, 1.0)
    ):
        rng = np.random.default_rng(seed)
        keys = rng.normal(size=(rows, dimension))
        keys /= np.linalg.norm(keys, axis=1, keepdims=True)
        weights = rng.random(rows)
        weights /= weights.sum()
        keys *= np.sqrt(weights)[:, None]
        values = rng.normal(size=(rows, 3)) * np.sqrt(weights)[:, None]
        previous = rng.normal(size=(dimension, 3))
        other = rng.normal(size=(dimension, 3))
        gram, cross = keys.T @ keys, keys.T @ values
        system = np.eye(dimension) + step * gram
        rhs = retention * previous + step * cross
        solution = np.linalg.solve(system, rhs)
        alternative = np.linalg.solve(system, retention * other + step * cross)
        require(np.isfinite(solution).all(), "La solución contiene valores no finitos.")
        require(
            np.linalg.norm(solution - alternative)
            <= retention * np.linalg.norm(previous - other) + 2e-11,
            "La solución no cumple la cota de contracción.",
        )
        residual = float(np.linalg.norm(system @ solution - rhs) / (1 + np.linalg.norm(rhs)))
        require(residual <= 2e-11, "Residuo excesivo en el sistema lineal.")
        residuals.append(residual)
        condition = float(np.linalg.cond(system))
        require(condition <= 1 + step + 2e-11, "Se excede la cota de condición.")
        conditions.append(condition)
        permutation = rng.permutation(rows)
        permuted_keys, permuted_values = keys[permutation], values[permutation]
        permuted = np.linalg.solve(
            np.eye(dimension) + step * permuted_keys.T @ permuted_keys,
            retention * previous + step * permuted_keys.T @ permuted_values,
        )
        blocks = np.array_split(np.arange(rows), min(3, rows))
        block_gram = sum((keys[b].T @ keys[b] for b in blocks), np.zeros_like(gram))
        block_cross = sum((keys[b].T @ values[b] for b in blocks), np.zeros_like(cross))
        blocked = np.linalg.solve(
            np.eye(dimension) + step * block_gram, retention * previous + step * block_cross
        )
        for candidate in (permuted, blocked):
            np.testing.assert_allclose(candidate, solution, atol=2e-11, rtol=2e-11)
            differences.append(float(np.max(np.abs(candidate - solution))))
    return {
        "cases": len(residuals),
        "seeds": [42, 43, 44],
        "key_dimensions": [2, 8, 32],
        "value_dimension": 3,
        "cohort_rows": [1, 17],
        "steps": [0, 0.2, 100],
        "retentions": [0, 0.95, 1],
        "maximum_scaled_equation_residual": max(residuals),
        "maximum_permutation_or_block_difference": max(differences),
        "maximum_condition_number": max(conditions),
        "atol": 2e-11,
        "rtol": 2e-11,
    }


def check_information_and_losses():
    samples = [(x, c, x * c) for x, c in itertools.product((-1, 1), repeat=2)]
    xx, cc, yy = map(list, zip(*samples, strict=True))
    information = entropy(cc) + entropy(yy) - entropy(list(zip(cc, yy, strict=True)))
    conditional = (
        entropy(list(zip(cc, xx, strict=True)))
        + entropy(list(zip(yy, xx, strict=True)))
        - entropy(xx)
        - entropy(samples)
    )
    covariance = sum(x * y for x, _, y in samples) / len(samples)
    require(information == 0 and conditional == 1 and covariance == 0, "Falla el ejemplo MI.")
    gains = [(1 + q) / (1 + q + r) for q, r in ((3, 1), (1, 3))]
    np.testing.assert_allclose(gains, [0.8, 0.4])
    predictions = np.array([0.0, 4.0])
    clipped_mixture = float(np.minimum(1, abs(predictions.mean())))
    mixture_clipped = float(np.minimum(1, abs(predictions)).mean())
    require(clipped_mixture > mixture_clipped, "Falla la no convexidad de la pérdida recortada.")
    return {
        "conditional_information": {
            "distribution": "X y C Rademacher independientes, Y=X*C",
            "atoms": samples,
            "covariance_x_y": covariance,
            "mutual_information_c_y_bits": information,
            "conditional_information_c_y_given_x_bits": conditional,
        },
        "kalman_example": {"prior_variance": 1, "innovation_variance": 5, "gains": gains},
        "clipped_loss": {
            "target": 0,
            "predictions": predictions.tolist(),
            "clip": 1,
            "loss_of_mixture": clipped_mixture,
            "mixture_of_losses": mixture_clipped,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Archivo JSON para conservar la comprobación.")
    args = parser.parse_args()
    report = {
        "schema_version": 2,
        "checked_at": datetime.now(UTC).isoformat(),
        "kind": "algebraic_checks",
        "candidate_implemented": False,
        "model_trained": False,
        "device": "cpu",
        "dtype": "float64",
        "numpy": np.__version__,
        "python": platform.python_version(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scalar_delta": check_scalar_delta(),
        "channel_gate": check_channel_gate(),
        "proximal_cohort": check_proximal_cohort(),
        **check_information_and_losses(),
        "limits": [
            "Comprobaciones finitas. No demuestran estabilidad global ni resultados financieros.",
            "Las cotas comparan estados con las mismas entradas, pesos y puertas.",
            "Amplificar una norma en un paso no demuestra divergencia al repetir el operador.",
            "La verificación proximal usa cohortes no vacías y tolerancias, no paridad bit a bit.",
            "Sin datos experimentales, acceso al test final ni mediciones de rendimiento.",
        ],
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized)
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
