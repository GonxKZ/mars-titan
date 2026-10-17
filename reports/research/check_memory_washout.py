"""Alcance del calentamiento de la memoria lineal, con aritmética racional."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from fractions import Fraction as F
from pathlib import Path

from reports.research import check_titans_memory_certificate as certificate


def contraction_steps(rho, tolerance, *, prefactor=F(1), max_steps=100_000):
    """Primer entero n con prefactor·rhoⁿ ≤ tolerancia, sin redondear el criterio."""
    if any(not isinstance(value, F) for value in (rho, tolerance, prefactor)):
        raise ValueError("La comprobación exacta requiere fracciones")
    if not 0 <= rho < 1 or tolerance <= 0 or prefactor <= 0:
        raise ValueError("Se requiere contracción estricta y cotas positivas")
    if type(max_steps) is not int or not 1 <= max_steps <= 100_000:
        raise ValueError("El límite debe ser un entero entre 1 y 100000")
    if prefactor <= tolerance:
        return 0
    low, high = 0, 1
    while prefactor * rho**high > tolerance:
        if high == max_steps:
            raise ValueError("La cota no alcanza la tolerancia dentro del límite")
        low, high = high, min(2 * high, max_steps)
    while high - low > 1:
        middle = (low + high) // 2
        if prefactor * rho**middle <= tolerance:
            high = middle
        else:
            low = middle
    return high


def check_washout():
    """Verificar el testigo PT1 y el horizonte que permite afirmar cada norma."""
    rho, p, q = F(9997, 10000), F(181, 1000), F(49, 5)
    valid, vertices = certificate.exact_certificate(
        certificate.CERTIFIED_BOXES["pt1_semivida_346_momentum_0_3"], p, q, rho
    )
    lower, upper, prefactor = F(99, 100), F(981, 100), F(63, 20)
    conversion = (
        certificate.psd_2x2(((1 - lower, p), (p, q - lower)))
        and certificate.psd_2x2(((upper - 1, -p), (-p, upper - q)))
        and prefactor**2 >= upper / lower
    )
    if not valid or not conversion:
        raise AssertionError("El testigo o su conversión de norma no se han verificado")
    target = F(1, 100)
    result = dict(
        certificate_valid=valid,
        witness=dict(p=str(p), q=str(q), norm_lower=str(lower), norm_upper=str(upper)),
        vertices=vertices,
        rho=str(rho),
        relative_tolerance=str(target),
        euclidean_prefactor=str(prefactor),
        euclidean_prefactor_valid=conversion,
        full_mac_covered=False,
        two_layer_memory_covered=False,
        protocol_changed=False,
        assumptions="Memorias lineales con las mismas claves, valores y puertas efectivas",
    )
    for name, factor in (("p_norm", F(1)), ("euclidean", prefactor)):
        steps = contraction_steps(rho, target, prefactor=factor)
        minimal = factor * rho**steps <= target < factor * rho ** (steps - 1)
        if not minimal:
            raise AssertionError("El horizonte no es el primer entero que cumple la cota")
        result[name] = dict(
            steps=steps,
            minimality_verified_exactly=minimal,
            approximate_upper_factor_at_252=float(factor * rho**252),
        )
    return result


def check_feedback_counterexample():
    """Dos puntos fijos acotados con valores realimentados, pese al control local."""
    alpha, eta, theta, limit = F(1, 2), F(1, 10), F(1, 20), F(1)
    box = certificate.CERTIFIED_BOXES["pt1_semivida_346_momentum_0_3"]
    inside = box[0] <= alpha <= box[1] and 0 <= eta <= box[2] and 0 <= theta <= box[3]
    states = ((F(1, 9), F(1, 18)), (F(-1, 9), F(-1, 18)))
    for weight, momentum in states:
        value = 21 * weight
        gradient = 2 * (weight - value)
        clipped = max(-limit, min(limit, gradient))
        next_momentum = eta * momentum - theta * clipped
        next_weight = (1 - alpha) * weight + next_momentum
        if (next_weight, next_momentum) != (weight, momentum) or abs(gradient) <= limit:
            raise AssertionError("El estado no es un punto fijo con recorte estricto")
    if not inside or states[0] == states[1]:
        raise AssertionError("El contraejemplo necesita dos estados distintos dentro de la caja")
    return dict(
        kind="algebraic_feedback_counterexample",
        gates_inside_pt1_box=inside,
        fixed_points=[[str(item) for item in state] for state in states],
        fixed_points_verified_exactly=True,
        local_jacobian_frobenius_squared=str((1 - alpha) ** 2 + 2 * eta**2),
        global_incremental_contraction=False,
        full_model_experiment=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Recibo nuevo, sin sobrescritura"
    )
    args = parser.parse_args()
    result = dict(
        schema_version=1,
        kind="linear_memory_washout_bound",
        recorded_at_utc=datetime.now(UTC).isoformat(),
        program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        certificate_program_sha256=hashlib.sha256(
            Path(certificate.__file__).read_bytes()
        ).hexdigest(),
        **check_washout(),
        feedback_counterexample=check_feedback_counterexample(),
    )
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
