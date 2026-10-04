"""Comprueba límites de replay y señales con ejemplos finitos, sin entrenar modelos."""

import argparse
import hashlib
import itertools
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_importance():
    expectation_errors, bias_errors = [], []
    for seed, size, epsilon in itertools.product((42, 43, 44), (2, 17, 64), (0.05, 0.2, 1.0)):
        rng = np.random.default_rng(seed)
        target = rng.random(size)
        target /= target.sum()
        priority = rng.random(size)
        priority /= priority.sum()
        probability = (1 - epsilon) * priority + epsilon * target
        gradient = rng.normal(size=(size, 4))
        weight = target / probability
        expected = (probability * weight) @ gradient
        reference = target @ gradient
        require(np.isfinite(expected).all(), "La esperanza no es finita.")
        np.testing.assert_allclose(expected, reference, atol=2e-14, rtol=2e-14)
        require(float(weight.max()) <= 1 / epsilon + 2e-14, "Peso fuera de su cota.")
        square_norm = np.sum(gradient**2, axis=1)
        moment = float((probability * weight**2) @ square_norm)
        bound = float(target @ square_norm / epsilon)
        require(moment <= bound + 2e-12, "Segundo momento fuera de su cota.")
        expectation_errors.append(float(np.max(np.abs(expected - reference))))
        for limit in (0.5, 1.0, 2.0):
            actual_bias = (probability * np.minimum(weight, limit)) @ gradient - reference
            formula = -np.maximum(target - limit * probability, 0) @ gradient
            np.testing.assert_allclose(actual_bias, formula, atol=2e-14, rtol=2e-14)
            bias_errors.append(float(np.max(np.abs(actual_bias - formula))))
    return {
        "cases": len(expectation_errors),
        "clipping_cases": len(bias_errors),
        "seeds": [42, 43, 44],
        "buffer_sizes": [2, 17, 64],
        "epsilon": [0.05, 0.2, 1],
        "gradient_dimension": 4,
        "maximum_expectation_error": max(expectation_errors),
        "maximum_clipping_bias_formula_error": max(bias_errors),
        "second_moment_atol": 2e-12,
        "atol": 2e-14,
        "rtol": 2e-14,
    }


def check_zero_support():
    gradient = np.array([[-1.0, 2.0], [3.0, -2.0], [999.0, 999.0]])
    cases = (
        ([0.5, 0.5, 0.0], [1.0, 0.0, 0.0], 0.2),
        ([1.0, 0.0, 0.0], [0.0, 0.5, 0.5], 0.2),
        ([0.5, 0.5, 0.0], [1.0, 0.0, 0.0], 1.0),
    )
    differences = []
    for target, priority, epsilon in cases:
        target, priority = np.array(target), np.array(priority)
        probability = epsilon * target + (1 - epsilon) * priority
        support = probability > 0
        require(np.all((target == 0) | support), "El objetivo tiene masa fuera del soporte.")
        with np.errstate(divide="raise", invalid="raise"):
            weight = np.divide(target, probability, out=np.zeros_like(target), where=support)
        expected = (probability * weight) @ gradient
        reference = target @ gradient
        np.testing.assert_allclose(expected, reference, atol=2e-14, rtol=2e-14)
        differences.append(float(np.max(np.abs(expected - reference))))
    return {"cases": len(cases), "maximum_expectation_error": max(differences)}


def check_repetition():
    cases = []
    for step, noise in itertools.product((0.1, 0.5, 0.9), (-2.0, 2.0)):
        trajectory = [0.0]
        for _ in range(64):
            trajectory.append((1 - step) * trajectory[-1] + step * noise)
        trajectory = np.array(trajectory)
        closed_form = noise * (1 - (1 - step) ** np.arange(65))
        np.testing.assert_allclose(trajectory, closed_form, atol=2e-14, rtol=2e-14)
        observed_error = np.abs(noise - trajectory)
        future_error = np.abs(trajectory)
        require(np.all(np.diff(observed_error) <= 2e-14), "No disminuye el error observado.")
        require(np.all(np.diff(future_error) >= -2e-14), "No aumenta el error futuro.")
        require(future_error[-1] > future_error[0], "No se obtiene el contraejemplo.")
        cases.append({"step": step, "noise": noise, "future_error_after_64": future_error[-1]})
    groups, copies, variance = 7, 4, 2.5
    covariance = variance * np.kron(np.eye(groups), np.ones((copies, copies)))
    coefficients = np.full(groups * copies, 1 / (groups * copies))
    mean_variance = float(coefficients @ covariance @ coefficients)
    np.testing.assert_allclose(mean_variance, variance / groups, atol=2e-14, rtol=2e-14)
    require(mean_variance > variance / (groups * copies), "Copias tratadas como independientes.")
    return {
        "noisy_label_cases": cases,
        "duplicate_groups": groups,
        "copies_per_group": copies,
        "individual_variance": variance,
        "actual_mean_variance": mean_variance,
        "incorrect_independence_variance": variance / (groups * copies),
        "effective_count_in_this_example": variance / mean_variance,
    }


def check_signal_and_distillation():
    values = np.array([-1.0, 1.0])
    predictions = np.linspace(-2, 2, 401)
    risks = np.mean(np.abs(values[:, None] - predictions[None, :]), axis=0)
    np.testing.assert_allclose(risks.min(), 1.0)
    oracle_risk = float(np.mean(np.abs(values - values)))
    asymmetric_values = np.array([0.0, 10.0])
    probabilities = np.array([0.9, 0.1])
    mean = float(probabilities @ asymmetric_values)
    median = 0.0
    mean_risk = float(probabilities @ np.abs(asymmetric_values - mean))
    median_risk = float(probabilities @ np.abs(asymmetric_values - median))
    np.testing.assert_allclose([mean, mean_risk, median_risk], [1, 1.8, 1])
    return {
        "unobserved_independent_signal": {
            "values": values.tolist(),
            "observed_signal_mae": oracle_risk,
            "reduced_signal_minimum_mae_on_grid": float(risks.min()),
            "grid": [-2, 2, 401],
        },
        "distillation": {
            "values": asymmetric_values.tolist(),
            "probabilities": probabilities.tolist(),
            "mean_target": mean,
            "median_target": median,
            "mean_mae": mean_risk,
            "median_mae": median_risk,
        },
    }


def posterior(prior, likelihood_ratio):
    return likelihood_ratio * prior / (1 + (likelihood_ratio - 1) * prior)


def check_bayes_and_calibration():
    prior, likelihood_ratio = 0.5, 3.0
    first = posterior(prior, likelihood_ratio)
    repeated = prior
    for _ in range(5):
        repeated = posterior(repeated, likelihood_ratio)
    np.testing.assert_allclose([first, repeated], [0.75, 243 / 244])
    priors = np.array([0.01, 0.011])
    amplification = float(np.diff(posterior(priors, 100))[0] / np.diff(priors)[0])
    require(amplification > 1, "El ejemplo de Bayes no amplifica la diferencia.")
    errors = np.array([0.0, 0.0, 1.0, 1.0])
    constant = np.full(4, 0.5)
    np.testing.assert_allclose(constant.mean(), errors.mean())
    brier = float(np.mean((constant - errors) ** 2))
    np.testing.assert_allclose(brier, 0.25)
    return {
        "duplicate_evidence": {
            "prior": prior,
            "likelihood_ratio": likelihood_ratio,
            "one_observation": first,
            "five_copies_counted_as_independent": repeated,
        },
        "bayes_prior_difference_amplification": amplification,
        "constant_calibrated_risk": {
            "binary_errors": errors.tolist(),
            "forecast": constant.tolist(),
            "brier": brier,
            "ideal_brier": float(np.mean((errors - errors) ** 2)),
        },
    }


def check_curvature():
    theta, gradient, step, smoothness = 1.0, 1.0, 3.0, 1.0
    before = theta**2 / 2
    after = (theta - step * gradient) ** 2 / 2
    bound = -step * theta * gradient + smoothness * step**2 * gradient**2 / 2
    np.testing.assert_allclose(after - before, bound)
    require(theta * gradient > 0 and after > before, "Falla el contraejemplo de curvatura.")
    return {"step": step, "alignment": theta * gradient, "loss_before": before, "loss_after": after}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Archivo JSON para conservar la comprobación.")
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "checked_at": datetime.now(UTC).isoformat(),
        "kind": "finite_algebraic_examples",
        "candidate_implemented": False,
        "model_trained": False,
        "device": "cpu",
        "dtype": "float64",
        "numpy": np.__version__,
        "python": platform.python_version(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "importance": check_importance(),
        "zero_support": check_zero_support(),
        "repetition": check_repetition(),
        "information": check_signal_and_distillation(),
        "bayes_and_calibration": check_bayes_and_calibration(),
        "curvature": check_curvature(),
        "limits": [
            "Ejemplos finitos que acompañan derivaciones, sin resultados financieros.",
            "No se entrenan modelos ni se leen datasets o el test final.",
            "La independencia y la estructura de covarianza solo rigen los ejemplos declarados.",
            "La cuadrícula de predicciones comprueba el ejemplo, no demuestra la cota universal.",
            "No son benchmarks ni comprobaciones de una implementación CUDA.",
        ],
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized)
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
