"""Término de IC por sesión que una receta suma a su pérdida (`session_ic_term_v1`).

Las pruebas comparan el término con una referencia independiente escrita con la biblioteca
estándar, sin optimizador ni pasos.
"""

from statistics import NormalDist

import pytest
import torch

from mars_titan.models.titans.session_ic_objective import (
    SESSION_IC_TERM,
    SessionICTerm,
    parse_session_ic_term,
)


def normal_scores(values):
    """Puntuaciones normales de los rangos, con los empates promediados."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks, start = [0.0] * len(values), 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        for position in range(start, stop + 1):
            ranks[order[position]] = (start + stop) / 2 + 1
        start = stop + 1
    return [NormalDist().inv_cdf((rank - 0.5) / len(values)) for rank in ranks]


def pearson(left, right):
    mean_left, mean_right = sum(left) / len(left), sum(right) / len(right)
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True))
    spread_left = sum((a - mean_left) ** 2 for a in left) ** 0.5
    spread_right = sum((b - mean_right) ** 2 for b in right) ** 0.5
    return covariance / (spread_left * spread_right)


def reference(point, target, sessions, weight, minimum_rows):
    groups = {}
    for row, key in enumerate(sessions):
        groups.setdefault(key, []).append(row)
    values = [
        pearson([point[row] for row in rows], normal_scores([target[row] for row in rows]))
        for rows in groups.values()
        if len(rows) >= minimum_rows
    ]
    return -weight * sum(values) / len(values) if values else 0.0


def sample(rows=40, seed=5):
    generator = torch.Generator().manual_seed(seed)
    point = torch.randn(rows, dtype=torch.float64, generator=generator)
    target = 0.3 * point + torch.randn(rows, dtype=torch.float64, generator=generator)
    # Tres sesiones de dos mercados, con filas intercaladas y una sesión demasiado corta.
    sessions = [("US", 1), ("US", 2), ("CN", 1)] * (rows // 3) + [("CN", 9)] * (rows % 3)
    return point, target, sessions


def test_the_term_is_the_weighted_mean_of_the_session_correlations():
    point, target, sessions = sample()
    term = SessionICTerm(weight=2.5, minimum_rows=5)
    expected = reference(point.tolist(), target.tolist(), sessions, 2.5, 5)
    assert term.term(point, target, sessions).item() == pytest.approx(expected, rel=1e-8)
    # La sesión corta no entra: retirarla no cambia el término.
    kept = [row for row, key in enumerate(sessions) if key != ("CN", 9)]
    without = term.term(point[kept], target[kept], [sessions[row] for row in kept])
    assert len(kept) < len(sessions)
    torch.testing.assert_close(without, term.term(point, target, sessions), rtol=0, atol=0)


def test_ties_use_average_ranks():
    point = torch.tensor([0.1, 0.4, -0.2, 0.9, 0.3, -0.5], dtype=torch.float64)
    target = torch.tensor([1.0, 1.0, 0.0, 2.0, 2.0, 2.0], dtype=torch.float64)
    sessions = [("US", 1)] * 6
    expected = reference(point.tolist(), target.tolist(), sessions, 1.0, 3)
    value = SessionICTerm(minimum_rows=3).term(point, target, sessions)
    assert value.item() == pytest.approx(expected, rel=1e-8)


def test_the_term_only_depends_on_the_order_of_the_target_and_not_on_level_or_scale():
    point, target, sessions = sample()
    term = SessionICTerm(minimum_rows=5)
    base = term.term(point, target, sessions)
    # Nivel y escala del punto dentro de cada sesión, y cualquier transformación creciente del
    # objetivo, dejan igual la correlación con sus puntuaciones normales.
    shifted = point.clone()
    for index, key in enumerate(sessions):
        shifted[index] = 3.0 * point[index] + (0.7 if key[1] == 1 else -0.2)
    torch.testing.assert_close(term.term(shifted, target, sessions), base, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(
        term.term(point, torch.exp(target), sessions), base, rtol=1e-12, atol=0
    )
    # Y un valor extremo del objetivo pesa lo que su rango, no su tamaño.
    extreme = target.clone()
    top = int(target.argmax())
    extreme[top] = target[top] + 1e6
    torch.testing.assert_close(term.term(point, extreme, sessions), base, rtol=1e-12, atol=0)


def test_the_gradient_moves_the_point_towards_the_order_of_the_target():
    point, target, sessions = sample()
    point = point.clone().requires_grad_(True)
    term = SessionICTerm(minimum_rows=5)
    value = term.term(point, target, sessions)
    (gradient,) = torch.autograd.grad(value, point)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    moved = point.detach() - 0.05 * gradient
    assert term.term(moved, target, sessions) < value
    # El gradiente de cada sesión suma cero salvo el suelo relativo de la normalización
    # (1e-10): el término no mueve el nivel de la sesión.
    for key in set(sessions):
        rows = [row for row, other in enumerate(sessions) if other == key]
        assert abs(gradient[rows].sum().item()) <= 1e-9 * gradient[rows].abs().sum().item()
    # La sesión corta no recibe gradiente.
    short = [row for row, key in enumerate(sessions) if key == ("CN", 9)]
    assert short and torch.count_nonzero(gradient[short]) == 0


def test_without_sessions_of_the_minimum_size_the_term_is_zero_with_a_graph():
    point, target, sessions = sample(rows=6)
    point = point.clone().requires_grad_(True)
    value = SessionICTerm(minimum_rows=8).term(point, target, sessions)
    assert value.item() == 0
    (gradient,) = torch.autograd.grad(value, point)
    assert torch.count_nonzero(gradient) == 0


def test_a_perfect_order_reaches_minus_the_weight():
    target = torch.linspace(-1, 1, 12, dtype=torch.float64)
    normal = torch.tensor(normal_scores(target.tolist()), dtype=torch.float64)
    value = SessionICTerm(weight=3.0, minimum_rows=3).term(normal, target, [("US", 1)] * 12)
    assert value.item() == pytest.approx(-3.0, abs=1e-8)
    reverse = SessionICTerm(weight=3.0, minimum_rows=3).term(-normal, target, [("US", 1)] * 12)
    assert reverse.item() == pytest.approx(3.0, abs=1e-8)


def test_the_identity_names_the_term_and_its_values():
    term = SessionICTerm(weight=3, minimum_rows=8)
    identity = term.identity()
    assert identity["name"] == SESSION_IC_TERM == "session_ic_term_v1"
    assert identity["weight"] == 3.0 and type(identity["weight"]) is float
    assert identity["minimum_rows"] == 8
    assert identity != SessionICTerm(weight=1.0, minimum_rows=8).identity()
    assert parse_session_ic_term(None) is None
    assert parse_session_ic_term(term) is term
    assert parse_session_ic_term(dict(weight=3.0, minimum_rows=8)) == term


@pytest.mark.parametrize(
    "value",
    [
        dict(weight=1.0),
        dict(minimum_rows=8),
        dict(weight=0.0, minimum_rows=8),
        dict(weight=-1.0, minimum_rows=8),
        dict(weight=float("nan"), minimum_rows=8),
        dict(weight=2e3, minimum_rows=8),
        dict(weight=True, minimum_rows=8),
        dict(weight=1.0, minimum_rows=2),
        dict(weight=1.0, minimum_rows=8.0),
        dict(weight=1.0, minimum_rows=8, otra=1),
        "session_ic_term_v1",
        3.0,
    ],
)
def test_invalid_declarations_are_rejected(value):
    with pytest.raises(ValueError):
        parse_session_ic_term(value)
