"""Comprueba el certificado de contracción de la memoria lineal de Titans, sin datos ni ajuste.

Con una capa lineal D × D, clave unitaria k, olvido α por fila, momentum η y paso θ, cada fila
w de la memoria y su momentum s siguen

    s' = η s − 2 θ (w·k − v) k,     w' = (1 − α) w + s'.

En la base {span(k), k⊥} la transición homogénea se reduce a dos bloques 2 × 2. El programa
contrasta esa reducción con `NeuralMemory`, verifica con fracciones exactas los certificados
cuadráticos en los vértices de cada caja de puertas, construye un contraejemplo racional con
factores estables cuyo producto periódico diverge y comprueba la cota de estado con escrituras
acotadas. No carga datos financieros, no crea optimizadores y no usa CUDA. La derivación está en
`docs/research/titans-memory-certificate.md`.
"""

import argparse
import hashlib
import itertools
import json
import math
import platform
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import numpy as np

# Cajas declaradas antes de ejecutar: (α mínimo, α máximo, η máximo, θ máximo). α mínimo
# 27/10000 queda por debajo de 1 − 2^(−1/256) ≈ 0,0027040, el olvido de semivida 256 de la
# receta de campaña, y θ máximo 1/10 es `theta_max` de `MemoryConfig`.
CERTIFIED_BOXES = {
    "semivida_256_momentum_0_3": (
        Fraction(27, 10000),
        Fraction(1),
        Fraction(3, 10),
        Fraction(1, 10),
    ),
    "semivida_69_momentum_0_5": (Fraction(1, 100), Fraction(1), Fraction(1, 2), Fraction(1, 10)),
    # Caja de la propuesta PT1: α ≥ 1/500 (semivida máxima de unas 346 observaciones).
    "pt1_semivida_346_momentum_0_3": (
        Fraction(1, 500),
        Fraction(1),
        Fraction(3, 10),
        Fraction(1, 10),
    ),
}
# Caja sin certificado: α como la receta, η libre hasta 0,9 y θ hasta theta_max.
COUNTEREXAMPLE_BOX = (Fraction(27, 10000), Fraction(1), Fraction(9, 10), Fraction(1, 10))
# Una sola dirección de clave basta: la inestabilidad aparece al alternar pasos con y sin
# escritura, porque los bloques B(α, η, θ) y B(α, η, 0) no conmutan.
KEY = (Fraction(1), Fraction(0))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def block(alpha, eta, theta):
    """Bloque en la dirección de k. Con θ = 0 coincide con el bloque ortogonal."""
    return ((1 - alpha - 2 * theta, eta), (-2 * theta, eta))


def transition(alpha, eta, theta, key):
    """Transición homogénea de (w; s) para una fila, en coma flotante."""
    key = np.asarray(key, dtype=float)
    identity = np.eye(key.size)
    projector = np.outer(key, key)
    return np.block(
        [
            [(1 - alpha) * identity - 2 * theta * projector, eta * identity],
            [-2 * theta * projector, eta * identity],
        ]
    )


def rational_transition(alpha, eta, theta, key):
    """La misma transición con D = 2 y fracciones exactas."""
    projector = [[key[i] * key[j] for j in range(2)] for i in range(2)]
    top = [
        [(1 - alpha) * (i == j) - 2 * theta * projector[i][j] for j in range(2)]
        + [eta * (i == j) for j in range(2)]
        for i in range(2)
    ]
    bottom = [
        [-2 * theta * projector[i][j] for j in range(2)] + [eta * (i == j) for j in range(2)]
        for i in range(2)
    ]
    return top + bottom


def matmul(left, right):
    inner, width = len(right), len(right[0])
    return [[sum(row[k] * right[k][j] for k in range(inner)) for j in range(width)] for row in left]


def psd_2x2(matrix):
    """Una matriz simétrica 2 × 2 es semidefinida positiva si y solo si sus menores lo son."""
    (a, b), (c, d) = matrix
    require(b == c, "La matriz no es simétrica")
    return a >= 0 and d >= 0 and a * d - b * c >= 0


def schur_stable_2x2(matrix):
    """Criterio de Jury: λ² + a₁λ + a₀ tiene sus raíces dentro del disco unidad abierto."""
    (a, b), (c, d) = matrix
    a1, a0 = -(a + d), a * d - b * c
    return abs(a0) < 1 and 1 + a1 + a0 > 0 and 1 - a1 + a0 > 0


def check_decomposition():
    """T = B⊥ ⊗ Π⊥ + B∥ ⊗ Π y Tᵀ(P₀ ⊗ I)T se separa por bloques, en coma flotante."""
    rng = np.random.default_rng(42)
    worst = 0.0
    cases = 0
    for dimension in (1, 2, 3, 8, 32):
        for _ in range(25):
            key = rng.normal(size=dimension)
            key /= np.linalg.norm(key)
            alpha, eta, theta = rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 0.5)
            projector = np.outer(key, key)
            orthogonal = np.eye(dimension) - projector
            parallel = np.array(block(alpha, eta, theta), dtype=float)
            ortho = np.array(block(alpha, eta, 0.0), dtype=float)
            expected = np.kron(ortho, orthogonal) + np.kron(parallel, projector)
            actual = transition(alpha, eta, theta, key)
            p, q = rng.uniform(-0.9, 0.9), rng.uniform(1.0, 10.0)
            base = np.array([[1.0, p], [p, q]])
            quadratic = actual.T @ np.kron(base, np.eye(dimension)) @ actual
            reduced = np.kron(ortho.T @ base @ ortho, orthogonal) + np.kron(
                parallel.T @ base @ parallel, projector
            )
            worst = max(
                worst,
                float(np.abs(actual - expected).max()),
                float(np.abs(quadratic - reduced).max()),
            )
            cases += 1
    require(worst < 1e-12, "La reducción por bloques no reproduce la transición")
    return {"cases": cases, "dimensions": [1, 2, 3, 8, 32], "max_abs_difference": worst}


def check_code_transition():
    """La actualización de `NeuralMemory` con una capa coincide con T z + b fila a fila."""
    import torch

    from mars_titan.models.titans.config import GateBias, MemoryConfig
    from mars_titan.models.titans.neural_memory import NeuralMemory

    torch.set_num_threads(2)
    worst = 0.0
    cases = 0
    for seed, dimension in itertools.product((42, 43), (2, 8, 16)):
        config = MemoryConfig(
            dim=dimension,
            depth=1,
            normalize_qk=True,
            max_batch=4,
            max_tokens=4,
            parameter_seed=seed,
            gate_bias=GateBias(),
        )
        memory = NeuralMemory(config, dtype=torch.float64)
        generator = torch.Generator().manual_seed(seed)
        state = memory.initial_state(3)
        # Un primer paso deja momentum no nulo para contrastar también ese término.
        first = torch.randn(3, 1, dimension, generator=generator, dtype=torch.float64)
        state = memory.update(first, state)
        token = torch.randn(3, 1, dimension, generator=generator, dtype=torch.float64)
        updated = memory.update(token, state)
        with torch.no_grad():
            y = token[:, 0]
            key = torch.nn.functional.normalize(memory.key_projection(y), dim=-1, eps=1e-12)
            value = memory.value_projection(y)
            alpha = memory.alpha_projection(y).sigmoid()
            eta = memory.eta_projection(y).sigmoid()[:, 0]
            theta = config.theta_max * memory.theta_projection(y).sigmoid()[:, 0]
        for row in range(3):
            weights = state.weights[0][row].numpy()
            momentum = state.momentum[0][row].numpy()
            k = key[row].numpy()
            for output in range(dimension):
                z = np.concatenate([weights[output], momentum[output]])
                gates = (float(alpha[row, output]), float(eta[row]), float(theta[row]))
                forcing = 2 * gates[2] * float(value[row, output]) * np.concatenate([k, k])
                predicted = transition(*gates, k) @ z + forcing
                actual = np.concatenate(
                    [
                        updated.weights[0][row, output].numpy(),
                        updated.momentum[0][row, output].numpy(),
                    ]
                )
                worst = max(worst, float(np.abs(predicted - actual).max()))
                cases += 1
    require(worst < 1e-12, "La transición derivada no coincide con NeuralMemory")
    return {
        "module": "mars_titan.models.titans.neural_memory.NeuralMemory",
        "depth": 1,
        "dtype": "float64",
        "rows_checked": cases,
        "max_abs_difference": worst,
        "optimizer_steps": 0,
    }


def vertex_gates(box):
    alpha_low, alpha_high, eta_high, theta_high = box
    zero = Fraction(0) if isinstance(alpha_low, Fraction) else 0.0
    return list(itertools.product((alpha_low, alpha_high), (zero, eta_high), (zero, theta_high)))


def contraction_grid(box, p, q):
    """ρ mínimo con ρ²P₀ ⪰ BᵀP₀B en todos los vértices, para cada (p, q) de la rejilla.

    Para matrices 2 × 2 el mayor autovalor generalizado de (BᵀP₀B, P₀) tiene forma cerrada.
    """
    det_p = q - p * p
    worst = np.zeros_like(p)
    for gates in vertex_gates(box):
        (b11, b12), (b21, b22) = (tuple(float(x) for x in row) for row in block(*gates))
        # X = BᵀP₀B con P₀ = [[1, p], [p, q]].
        x11 = b11 * b11 + 2 * p * b11 * b21 + q * b21 * b21
        x12 = b11 * b12 + p * (b11 * b22 + b21 * b12) + q * b21 * b22
        x22 = b12 * b12 + 2 * p * b12 * b22 + q * b22 * b22
        trace_term = x11 * q + x22 - 2 * x12 * p
        det_x = (b11 * b22 - b12 * b21) ** 2 * det_p
        discriminant = np.maximum(trace_term * trace_term - 4 * det_p * det_x, 0.0)
        largest = (trace_term + np.sqrt(discriminant)) / (2 * det_p)
        worst = np.maximum(worst, np.sqrt(largest))
    return worst


def float_certificate(box, grid=1201):
    """Busca P₀ = [[1, p], [p, q]] que minimice la contracción común en los vértices."""
    p_values = np.linspace(-0.99, 0.99, grid)
    q_values = np.exp(np.linspace(-6.0, 10.0, grid))
    p, q = np.meshgrid(p_values, q_values, indexing="ij")
    valid = q > p * p
    radius = np.where(valid, contraction_grid(box, np.where(valid, p, 0.0), q), np.inf)
    index = np.unravel_index(np.argmin(radius), radius.shape)
    return float(radius[index]), float(p[index]), float(q[index])


def exact_certificate(box, p, q, rho):
    """P₀ ≻ 0 y ρ²P₀ − BᵀP₀B ⪰ 0 en los ocho vértices, con fracciones exactas.

    La condición equivale por complemento de Schur a una LMI afín en B, y B es afín en
    (α, η, θ). Por eso cumplirla en los vértices la extiende a toda la caja.
    """
    base = ((Fraction(1), p), (p, q))
    require(q - p * p > 0, "P₀ no es definida positiva")
    vertices = []
    for gates in vertex_gates(box):
        matrix = block(*gates)
        transposed = [list(row) for row in zip(*matrix, strict=True)]
        quadratic = matmul(matmul(transposed, base), matrix)
        slack = [[rho * rho * base[i][j] - quadratic[i][j] for j in range(2)] for i in range(2)]
        vertices.append(
            {
                "alpha": str(gates[0]),
                "eta": str(gates[1]),
                "theta": str(gates[2]),
                "psd": psd_2x2(slack),
            }
        )
    return all(vertex["psd"] for vertex in vertices), vertices


def rationalize_certificate(box, estimate):
    """Redondea el óptimo flotante a fracciones y acepta el menor ρ con prueba exacta."""
    rho_float, p_float, q_float = estimate
    for numerator in range(math.floor(rho_float * 100000) + 1, 100000):
        rho = Fraction(numerator, 100000)
        for dp, dq in itertools.product(range(-3, 4), range(-3, 4)):
            p = Fraction(round(p_float * 1000) + dp, 1000)
            q = Fraction(round(q_float * 100) + dq, 100)
            accepted, vertices = exact_certificate(box, p, q, rho)
            if accepted:
                return rho, p, q, vertices
    raise AssertionError("No se encontró un certificado racional")


def check_certified_boxes():
    results = {}
    for name, box in CERTIFIED_BOXES.items():
        estimate = float_certificate(box)
        require(estimate[0] < 1, f"La caja {name} no admite certificado cuadrático")
        rho, p, q, vertices = rationalize_certificate(box, estimate)
        matrix = np.array([[1.0, float(p)], [float(p), float(q)]])
        results[name] = {
            "box": {
                "alpha": [str(box[0]), str(box[1])],
                "eta": ["0", str(box[2])],
                "theta": ["0", str(box[3])],
            },
            "float_estimate": {"rho": estimate[0], "p": estimate[1], "q": estimate[2]},
            "exact": {"rho": str(rho), "p": str(p), "q": str(q), "vertices": vertices},
            "rho_float": float(rho),
            "contraction_half_life_steps": math.log(2) / -math.log(float(rho)),
            "condition_number_P0": float(np.linalg.cond(matrix)),
        }
    return results


def check_random_products():
    """En la caja certificada, la norma inducida por P₀ ⊗ I nunca supera ρ por paso."""
    box = CERTIFIED_BOXES["semivida_256_momentum_0_3"]
    rho, p, q, _ = rationalize_certificate(box, float_certificate(box))
    base = np.array([[1.0, float(p)], [float(p), float(q)]])
    rng = np.random.default_rng(7)
    worst_step = 0.0
    worst_product = 0.0
    sequences = 0
    corners = [[float(x) for x in gates] for gates in vertex_gates(box)]
    for dimension in (2, 4, 8):
        root = np.linalg.cholesky(np.kron(base, np.eye(dimension)))
        inverse = np.linalg.inv(root)
        for _ in range(200):
            product = np.eye(2 * dimension)
            for _ in range(32):
                key = rng.normal(size=dimension)
                key /= np.linalg.norm(key)
                if rng.uniform() < 0.5:
                    gates = corners[rng.integers(len(corners))]
                else:
                    gates = [
                        rng.uniform(float(box[0]), 1.0),
                        rng.uniform(0, float(box[2])),
                        rng.uniform(0, float(box[3])),
                    ]
                matrix = transition(*gates, key)
                step = np.linalg.norm(root.T @ matrix @ inverse.T, 2)
                worst_step = max(worst_step, float(step) / float(rho))
                product = matrix @ product
            norm = np.linalg.norm(root.T @ product @ inverse.T, 2)
            worst_product = max(worst_product, float(norm) / float(rho) ** 32)
            sequences += 1
    require(worst_step <= 1 + 1e-12, "Un paso supera la contracción certificada")
    require(worst_product <= 1 + 1e-10, "Un producto supera la cota certificada")
    return {
        "box": "semivida_256_momentum_0_3",
        "sequences": sequences,
        "steps_per_sequence": 32,
        "dimensions": [2, 4, 8],
        "max_step_norm_over_rho": worst_step,
        "max_product_norm_over_rho_power": worst_product,
    }


def beam_counterexample(box, length=8, width=80):
    """Búsqueda en haz, sobre los bloques de los vértices, del producto con mayor crecimiento."""
    labels = vertex_gates(box)
    blocks = [np.array(block(*gates), dtype=float) for gates in labels]
    beam = [((), np.eye(2))]
    best = (0.0, ())
    for step in range(1, length + 1):
        candidates = {}
        for sequence, matrix in beam:
            for index, factor in enumerate(blocks):
                product = factor @ matrix
                radius = float(max(abs(np.linalg.eigvals(product)))) ** (1 / step)
                candidates[sequence + (index,)] = (radius, product)
        ranked = sorted(candidates.items(), key=lambda item: -item[1][0])
        if ranked[0][1][0] > best[0]:
            best = (ranked[0][1][0], ranked[0][0])
        beam = [(sequence, product) for sequence, (_, product) in ranked[:width]]
    return best, labels


def integer_power_trace(matrix, exponent):
    """tr(Nⁿ) con enteros exactos, por cuadrados sucesivos (n potencia de dos)."""
    require(exponent & (exponent - 1) == 0, "El exponente debe ser potencia de dos")
    power = matrix
    for _ in range(exponent.bit_length() - 1):
        power = matmul(power, power)
    return sum(power[i][i] for i in range(len(power)))


def real_nonnegative_spectrum(matrix):
    """Autovalores reales y no negativos de una matriz 2 × 2, comprobados sin raíces."""
    (a, b), (c, d) = matrix
    trace, determinant = a + d, a * d - b * c
    return trace * trace - 4 * determinant >= 0 and trace >= 0 and determinant >= 0


def check_counterexample():
    """Factores estables (Jury exacto) cuyo producto periódico tiene radio espectral > 1."""
    (radius, sequence), labels = beam_counterexample(COUNTEREXAMPLE_BOX)
    require(radius > 1, "La búsqueda no encontró un producto divergente")
    product = [[Fraction(int(i == j)) for j in range(4)] for i in range(4)]
    real_spectra = True
    for index in sequence:
        gates = labels[index]
        require(schur_stable_2x2(block(*gates)), "Un bloque paralelo no es estable por Jury")
        require(0 <= 1 - gates[0] < 1 and 0 <= gates[1] < 1, "Un bloque ortogonal no es estable")
        real_spectra &= real_nonnegative_spectrum(block(*gates))
        real_spectra &= real_nonnegative_spectrum(block(gates[0], gates[1], Fraction(0)))
        product = matmul(rational_transition(*gates, KEY), product)
    denominator = math.lcm(*(entry.denominator for row in product for entry in row))
    integer = [[int(entry * denominator) for entry in row] for row in product]
    exponent = 64
    trace = integer_power_trace(integer, exponent)
    # |tr(Mⁿ)| ≤ 4 ρ(M)ⁿ con M = N/d, así que |tr(Nⁿ)| > 4 dⁿ prueba ρ(M) > 1 sin redondeos.
    require(abs(trace) > 4 * denominator**exponent, "La traza exacta no certifica divergencia")
    log_lower = (math.log(abs(trace)) - exponent * math.log(denominator) - math.log(4)) / exponent
    numeric = float(max(abs(np.linalg.eigvals(np.array(product, dtype=float)))))
    return {
        "box": {
            "alpha": [str(COUNTEREXAMPLE_BOX[0]), str(COUNTEREXAMPLE_BOX[1])],
            "eta": ["0", str(COUNTEREXAMPLE_BOX[2])],
            "theta": ["0", str(COUNTEREXAMPLE_BOX[3])],
        },
        "sequence": [
            {"alpha": str(labels[i][0]), "eta": str(labels[i][1]), "theta": str(labels[i][2])}
            for i in sequence
        ],
        "key": [str(x) for x in KEY],
        "every_factor_schur_stable_exact": True,
        # Un espectro real y no negativo por paso tampoco evita la divergencia del producto.
        "every_factor_real_nonnegative_spectrum_exact": real_spectra,
        "exact_trace_power": exponent,
        "certified_lower_bound_product_spectral_radius": math.exp(log_lower),
        "numeric_product_spectral_radius": numeric,
        "numeric_growth_per_step": numeric ** (1 / len(sequence)),
    }


def check_bounded_writes():
    """Con |ψ| ≤ δ y puertas acotadas, ‖w‖ ≤ ‖w₀‖ + (‖s₀‖ + 2θδ/(1 − η))/α por fila."""
    rng = np.random.default_rng(11)
    alpha_low, eta_high, theta_high, delta = 0.0027, 0.9, 0.1, 1.0
    worst_ratio = 0.0
    for dimension in (2, 8):
        for _ in range(50):
            w = rng.normal(size=dimension)
            s = rng.normal(size=dimension) * 0.1
            drift = (np.linalg.norm(s) + 2 * theta_high * delta / (1 - eta_high)) / alpha_low
            bound = np.linalg.norm(w) + drift
            for _ in range(3000):
                key = rng.normal(size=dimension)
                key /= np.linalg.norm(key)
                # Valores de colas pesadas con saltos extremos para forzar escrituras grandes.
                value = rng.standard_t(1.5) * (1000 if rng.uniform() < 0.01 else 1)
                alpha = rng.uniform(alpha_low, 1)
                eta = rng.uniform(0, eta_high)
                theta = rng.uniform(0, theta_high)
                residual = float(np.clip(w @ key - value, -delta, delta))
                s = eta * s - 2 * theta * residual * key
                w = (1 - alpha) * w + s
                worst_ratio = max(worst_ratio, float(np.linalg.norm(w) / bound))
    require(worst_ratio <= 1, "La cota con escrituras acotadas no se cumple")
    return {
        "alpha_low": alpha_low,
        "eta_high": eta_high,
        "theta_high": theta_high,
        "huber_delta": delta,
        "max_norm_over_bound": worst_ratio,
        "note": "Valores t de Student con 1,5 grados y saltos de factor 1000 en el 1 % de pasos",
    }


def check_incremental_contraction():
    """Dos trayectorias con las mismas entradas y escritura recortada se acercan a ritmo ρ.

    El recorte es monótono y 1-Lipschitz, así que ψ(r₁) − ψ(r₂) = c (r₁ − r₂) con c ∈ [0, 1].
    La diferencia sigue la transición con paso cθ, que sigue dentro de la caja certificada.
    """
    box = CERTIFIED_BOXES["pt1_semivida_346_momentum_0_3"]
    rho, p, q, _ = rationalize_certificate(box, float_certificate(box))
    base = np.array([[1.0, float(p)], [float(p), float(q)]])
    rng = np.random.default_rng(13)
    worst = 0.0
    trajectories = 0
    for dimension in (2, 8):
        root = np.linalg.cholesky(np.kron(base, np.eye(dimension)))
        for _ in range(100):
            first = rng.normal(size=2 * dimension) * 3
            second = rng.normal(size=2 * dimension) * 3
            initial = np.linalg.norm(root.T @ (first - second))
            for step in range(1, 401):
                key = rng.normal(size=dimension)
                key /= np.linalg.norm(key)
                value = rng.standard_t(1.5)
                alpha = rng.uniform(float(box[0]), 1.0)
                eta = rng.uniform(0, float(box[2]))
                theta = rng.uniform(0, float(box[3]))
                updated = []
                for state in (first, second):
                    w, s = state[:dimension], state[dimension:]
                    residual = float(np.clip(w @ key - value, -0.5, 0.5))
                    s = eta * s - 2 * theta * residual * key
                    updated.append(np.concatenate([(1 - alpha) * w + s, s]))
                first, second = updated
                distance = np.linalg.norm(root.T @ (first - second))
                worst = max(worst, float(distance / (float(rho) ** step * initial)))
            trajectories += 1
    require(worst <= 1 + 1e-9, "La escritura recortada no contrae la diferencia")
    return {
        "box": "pt1_semivida_346_momentum_0_3",
        "rho": str(rho),
        "trajectories": trajectories,
        "steps": 400,
        "clip_delta": 0.5,
        "max_distance_over_rho_power": worst,
    }


def check_frontier():
    """α mínimo con certificado cuadrático para cada η máximo, θ ≤ 0,1. Solo estimación."""
    rows = []
    for eta_high in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):

        def certified(alpha_low, eta_high=eta_high):
            return float_certificate((alpha_low, 1.0, eta_high, 0.1), grid=401)[0] < 1

        if not certified(0.5):
            rows.append({"eta_high": eta_high, "min_alpha_estimate": None})
            continue
        low, high = 1e-5, 0.5
        for _ in range(24):
            middle = math.sqrt(low * high)
            low, high = (low, middle) if certified(middle) else (middle, high)
        rows.append(
            {
                "eta_high": eta_high,
                "min_alpha_estimate": high,
                "half_life_estimate": math.log(2) / -math.log1p(-high),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Archivo JSON para conservar la comprobación.")
    args = parser.parse_args()
    import torch

    report = {
        "schema_version": 1,
        "kind": "titans_linear_memory_certificate_check",
        "started_utc": datetime.now(UTC).isoformat(),
        "program_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "decomposition": check_decomposition(),
        "code_transition": check_code_transition(),
        "certified_boxes": check_certified_boxes(),
        "random_products": check_random_products(),
        "counterexample": check_counterexample(),
        "bounded_writes": check_bounded_writes(),
        "incremental_contraction": check_incremental_contraction(),
        "frontier_float_estimate": check_frontier(),
        "limits": [
            "Memoria lineal de una capa con claves unitarias. No cubre la memoria de dos capas"
            " con LayerNorm residual de la receta de campaña.",
            "Los certificados racionales son exactos en los vértices. La frontera es una"
            " estimación flotante y no demuestra que fuera de ella no exista certificado.",
            "La cota con escrituras acotadas vale para cualquier secuencia de claves, valores y"
            " puertas. La contracción incremental exige además que las puertas estén dentro"
            " de una caja certificada.",
        ],
    }
    report["finished_utc"] = datetime.now(UTC).isoformat()
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized)
    print(serialized)


if __name__ == "__main__":
    main()
