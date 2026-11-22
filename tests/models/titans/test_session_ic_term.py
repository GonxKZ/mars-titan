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
    known_offset,
    liquidity_weights,
    parse_session_ic_term,
    session_correlations,
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


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks, start = [0.0] * len(values), 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        for position in range(start, stop + 1):
            ranks[order[position]] = (start + stop) / 2 + 1
        start = stop + 1
    return ranks


def reference_weights(liquidity, power):
    ranks = average_ranks(liquidity)
    raw = [(0.05 + (rank - 0.5) / len(liquidity)) ** power for rank in ranks]
    return [value / sum(raw) for value in raw]


def weighted_pearson(left, right, weights):
    mean_left = sum(w * a for w, a in zip(weights, left, strict=True))
    mean_right = sum(w * b for w, b in zip(weights, right, strict=True))
    covariance = sum(
        w * (a - mean_left) * (b - mean_right) for w, a, b in zip(weights, left, right, strict=True)
    )
    spread_left = sum(w * (a - mean_left) ** 2 for w, a in zip(weights, left, strict=True))
    spread_right = sum(w * (b - mean_right) ** 2 for w, b in zip(weights, right, strict=True))
    return covariance / (spread_left * spread_right) ** 0.5


def weighted_reference(point, target, sessions, liquidity, weight, minimum_rows, power):
    groups = {}
    for row, key in enumerate(sessions):
        groups.setdefault(key, []).append(row)
    values = []
    for rows in groups.values():
        if len(rows) < minimum_rows:
            continue
        weights = reference_weights([liquidity[row] for row in rows], power)
        scores = normal_scores([target[row] for row in rows])
        values.append(weighted_pearson([point[row] for row in rows], scores, weights))
    return -weight * sum(values) / len(values) if values else 0.0


def liquid_sample(rows=40, seed=5):
    point, target, sessions = sample(rows, seed)
    generator = torch.Generator().manual_seed(seed + 100)
    liquidity = 12 + 3 * torch.randn(rows, dtype=torch.float64, generator=generator)
    # Empates de liquidez dentro de una sesión.
    liquidity[3], liquidity[6] = liquidity[0], liquidity[0]
    return point, target, sessions, liquidity


@pytest.mark.parametrize("power", [0.5, 1.0, 2.0])
def test_the_weighted_term_is_the_weighted_correlation_of_each_session(power):
    point, target, sessions, liquidity = liquid_sample()
    term = SessionICTerm(weight=2.5, minimum_rows=5, liquidity_column=-3, liquidity_power=power)
    expected = weighted_reference(
        point.tolist(), target.tolist(), sessions, liquidity.tolist(), 2.5, 5, power
    )
    value = term.term(point, target, sessions, liquidity)
    assert value.item() == pytest.approx(expected, rel=1e-8)
    assert value.item() != pytest.approx(
        reference(point.tolist(), target.tolist(), sessions, 2.5, 5)
    )
    # Los pesos de una sesión suman uno, crecen con la liquidez y los empates lo comparten.
    rows = [row for row, key in enumerate(sessions) if key == ("US", 1)]
    weights = liquidity_weights(liquidity[rows], power)
    torch.testing.assert_close(
        weights,
        torch.tensor(reference_weights(liquidity[rows].tolist(), power), dtype=torch.float64),
        rtol=1e-12,
        atol=0,
    )
    assert weights.sum().item() == pytest.approx(1.0)
    order = liquidity[rows].argsort()
    assert bool((weights[order][1:] >= weights[order][:-1]).all())
    # Cualquier transformación creciente de la liquidez deja los mismos pesos.
    torch.testing.assert_close(
        liquidity_weights(torch.exp(liquidity[rows] / 4), power), weights, rtol=1e-12, atol=0
    )


def test_equal_liquidity_gives_the_unweighted_term():
    point, target, sessions, _ = liquid_sample()
    plain = SessionICTerm(weight=2.0, minimum_rows=5).term(point, target, sessions)
    term = SessionICTerm(weight=2.0, minimum_rows=5, liquidity_column=-1, liquidity_power=1.0)
    flat = torch.full_like(point, 7.0)
    torch.testing.assert_close(term.term(point, target, sessions, flat), plain, rtol=1e-9, atol=0)
    # Y la correlación ponderada es la de siempre cuando no se declara la liquidez.
    torch.testing.assert_close(
        session_correlations(point, target, sessions, 5),
        session_correlations(point, target, sessions, 5, liquidity=None, power=None),
        rtol=0,
        atol=0,
    )


def test_the_liquid_rows_weigh_more_and_the_liquidity_has_no_gradient():
    point, target, sessions, liquidity = liquid_sample()
    point = point.clone().requires_grad_(True)
    liquidity = liquidity.clone().requires_grad_(True)
    term = SessionICTerm(minimum_rows=5, liquidity_column=-1, liquidity_power=2.0)
    value = term.term(point, target, sessions, liquidity)
    (gradient,) = torch.autograd.grad(value, point, retain_graph=True)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert torch.autograd.grad(value, liquidity, allow_unused=True)[0] is None
    # En cada sesión, la mitad más líquida recibe más gradiente que la menos líquida.
    for key in {key for key in sessions if key != ("CN", 9)}:
        rows = [row for row, other in enumerate(sessions) if other == key]
        order = sorted(rows, key=lambda row: liquidity[row].item())
        half = len(order) // 2
        assert gradient[order[half:]].abs().sum() > gradient[order[:half]].abs().sum()
    # Mover el punto contra el gradiente mejora el término ponderado.
    moved = point.detach() - 0.05 * gradient
    assert term.term(moved, target, sessions, liquidity.detach()) < value


def test_the_liquidity_is_declared_whole_and_enters_the_identity_only_then():
    plain = SessionICTerm(weight=3.0, minimum_rows=8)
    declared = SessionICTerm(weight=3.0, minimum_rows=8, liquidity_column=-29, liquidity_power=0.5)
    assert not plain.weighted and declared.weighted
    assert "liquidity_column" not in plain.identity()
    identity = declared.identity()
    assert identity["liquidity_column"] == -29 and identity["liquidity_power"] == 0.5
    assert {k: v for k, v in identity.items() if k in plain.identity()} == plain.identity()
    values = dict(weight=3.0, minimum_rows=8, liquidity_column=-29, liquidity_power=0.5)
    assert parse_session_ic_term(values) == declared
    point, target, sessions, liquidity = liquid_sample()
    with pytest.raises(ValueError, match="solo si la declara"):
        plain.term(point, target, sessions, liquidity)
    with pytest.raises(ValueError, match="solo si la declara"):
        declared.term(point, target, sessions)
    for changes in (
        dict(liquidity_column=-29),
        dict(liquidity_power=0.5),
        dict(liquidity_column=0, liquidity_power=0.5),
        dict(liquidity_column=3, liquidity_power=0.5),
        dict(liquidity_column=-1.0, liquidity_power=0.5),
        dict(liquidity_column=True, liquidity_power=0.5),
        dict(liquidity_column=-1, liquidity_power=0.0),
        dict(liquidity_column=-1, liquidity_power=-1.0),
        dict(liquidity_column=-1, liquidity_power=9.0),
        dict(liquidity_column=-1, liquidity_power=float("nan")),
    ):
        with pytest.raises(ValueError):
            parse_session_ic_term(dict(weight=3.0, minimum_rows=8, **changes))


SHIFT = dict(offset_column=-3, offset_scale=0.01)


def offsets(rows=40, seed=11):
    generator = torch.Generator().manual_seed(seed)
    return 0.8 * torch.randn(rows, dtype=torch.float64, generator=generator)


def test_the_shifted_term_correlates_point_and_target_after_adding_the_known_offset():
    point, target, sessions = sample()
    offset = offsets()
    term = SessionICTerm(weight=2.5, minimum_rows=5, **SHIFT)
    expected = reference((point + offset).tolist(), (target + offset).tolist(), sessions, 2.5, 5)
    value = term.term(point, target, sessions, offset=offset)
    assert value.item() == pytest.approx(expected, rel=1e-8)
    # Con desplazamiento cero es el término de siempre, bit a bit.
    plain = SessionICTerm(weight=2.5, minimum_rows=5)
    zero = torch.zeros_like(offset)
    torch.testing.assert_close(
        term.term(point, target, sessions, offset=zero),
        plain.term(point, target, sessions),
        rtol=0,
        atol=0,
    )


def test_the_shifted_term_depends_on_the_scale_of_the_point_and_the_offset_has_no_gradient():
    point, target, sessions = sample()
    offset = offsets().requires_grad_()
    shifted, plain = SessionICTerm(minimum_rows=5, **SHIFT), SessionICTerm(minimum_rows=5)
    scale = torch.ones((), dtype=torch.float64, requires_grad=True)
    # La escala del punto no mueve el término de siempre y sí el desplazado: sumado al alfa
    # conocido, un punto con otra escala ordena de otra forma.
    (moved,) = torch.autograd.grad(
        shifted.term(scale * point, target, sessions, offset=offset), scale
    )
    (still,) = torch.autograd.grad(plain.term(scale * point, target, sessions), scale)
    assert abs(moved.item()) > 1e-4 and abs(still.item()) < 1e-9
    leaf = point.clone().requires_grad_()
    value = shifted.term(leaf, target, sessions, offset=offset)
    gradient, through = torch.autograd.grad(value, (leaf, offset), allow_unused=True)
    assert gradient.abs().sum() > 0 and through is None


def test_the_offset_is_declared_whole_and_enters_the_identity_only_then():
    plain = SessionICTerm(weight=3.0, minimum_rows=8)
    shifted = SessionICTerm(weight=3.0, minimum_rows=8, **SHIFT)
    assert not plain.shifted and shifted.shifted
    identity, before = shifted.identity(), plain.identity()
    assert {key: identity[key] for key in before} == before
    assert set(identity) - set(before) == {
        "offset_column",
        "offset_scale",
        "offset",
        "offset_source",
    }
    assert identity["offset_column"] == -3 and identity["offset_scale"] == 0.01
    assert parse_session_ic_term(dict(weight=3.0, minimum_rows=8, **SHIFT)) == shifted
    both = dict(weight=3.0, minimum_rows=8, liquidity_column=-1, liquidity_power=1.0, **SHIFT)
    parsed = parse_session_ic_term(both)
    assert parsed.weighted and parsed.shifted
    point, target, sessions = sample()
    with pytest.raises(ValueError, match="desplazamiento"):
        shifted.term(point, target, sessions)
    with pytest.raises(ValueError, match="desplazamiento"):
        plain.term(point, target, sessions, offset=offsets())
    with pytest.raises(ValueError, match="desplazamiento"):
        shifted.term(point, target, sessions, offset=offsets()[:-1])


@pytest.mark.parametrize(
    "wrong",
    [
        dict(offset_column=-3),
        dict(offset_scale=0.01),
        dict(offset_column=0, offset_scale=0.01),
        dict(offset_column=3, offset_scale=0.01),
        dict(offset_column=True, offset_scale=0.01),
        dict(offset_column=-3, offset_scale=0.0),
        dict(offset_column=-3, offset_scale=-1.0),
        dict(offset_column=-3, offset_scale=float("nan")),
        dict(offset_column=-3, offset_scale="0.01"),
    ],
)
def test_invalid_offsets_are_rejected(wrong):
    with pytest.raises(ValueError):
        SessionICTerm(**wrong)
    with pytest.raises(ValueError):
        parse_session_ic_term(dict(weight=1.0, minimum_rows=8, **wrong))


def test_the_known_offset_is_the_column_times_its_scale_and_zero_where_it_is_unknown():
    column = torch.tensor([1.5, float("nan"), -2.0, float("inf"), 0.0], dtype=torch.float64)
    result = known_offset(column.requires_grad_(), 0.01)
    assert result.tolist() == [0.015, 0.0, -0.02, 0.0, 0.0]
    assert not result.requires_grad
    assert known_offset(column.float(), 0.01).dtype == torch.float32
