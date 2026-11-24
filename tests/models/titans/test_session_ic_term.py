"""Término de IC por sesión que una receta suma a su pérdida (`session_ic_term_v1`).

Las pruebas comparan el término con una referencia independiente escrita con la biblioteca
estándar, sin optimizador ni pasos.
"""

from statistics import NormalDist

import numpy as np
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


def test_the_plain_weight_adds_the_term_on_the_label_to_the_shifted_one():
    point, target, sessions = sample()
    offset = offsets()
    both = SessionICTerm(weight=2.5, minimum_rows=5, plain_weight=1.5, **SHIFT)
    shifted = reference((point + offset).tolist(), (target + offset).tolist(), sessions, 2.5, 5)
    plain = reference(point.tolist(), target.tolist(), sessions, 1.5, 5)
    value = both.term(point, target, sessions, offset=offset)
    assert value.item() == pytest.approx(shifted + plain, rel=1e-8)
    # Es la suma de los dos términos de siempre, cada uno con su peso, bit a bit.
    alone = SessionICTerm(weight=2.5, minimum_rows=5, **SHIFT)
    label = SessionICTerm(weight=1.5, minimum_rows=5)
    torch.testing.assert_close(
        value,
        alone.term(point, target, sessions, offset=offset) + label.term(point, target, sessions),
        rtol=0,
        atol=0,
    )
    # Y no es el término desplazado con el peso sumado: el de la etiqueta no ve el alfa.
    heavier = SessionICTerm(weight=4.0, minimum_rows=5, **SHIFT)
    assert abs((value - heavier.term(point, target, sessions, offset=offset)).item()) > 1e-3
    # Sin sesiones con suficientes filas vale cero y conserva el grafo.
    leaf = point.clone().requires_grad_()
    short = SessionICTerm(minimum_rows=64, plain_weight=1.0, **SHIFT)
    zero = short.term(leaf, target, sessions, offset=offset)
    assert zero.item() == 0.0 and zero.requires_grad


def test_the_gradient_with_the_plain_weight_is_the_sum_of_the_two_gradients():
    point, target, sessions = sample()
    offset = offsets().requires_grad_()
    leaf = point.clone().requires_grad_()
    both = SessionICTerm(weight=2.0, minimum_rows=5, plain_weight=3.0, **SHIFT)
    shifted = SessionICTerm(weight=2.0, minimum_rows=5, **SHIFT)
    label = SessionICTerm(weight=3.0, minimum_rows=5)
    together, through = torch.autograd.grad(
        both.term(leaf, target, sessions, offset=offset), (leaf, offset), allow_unused=True
    )
    (first,) = torch.autograd.grad(shifted.term(leaf, target, sessions, offset=offset), leaf)
    (second,) = torch.autograd.grad(label.term(leaf, target, sessions), leaf)
    torch.testing.assert_close(together, first + second, rtol=1e-12, atol=1e-15)
    assert through is None
    # El término de la etiqueta no depende de la escala del punto; el desplazado sí.
    scale = torch.ones((), dtype=torch.float64, requires_grad=True)
    (moved,) = torch.autograd.grad(both.term(scale * point, target, sessions, offset=offset), scale)
    (only,) = torch.autograd.grad(
        shifted.term(scale * point, target, sessions, offset=offset), scale
    )
    torch.testing.assert_close(moved, only, rtol=1e-9, atol=1e-12)


def test_the_plain_weight_needs_the_offset_and_enters_the_identity_only_then():
    shifted = SessionICTerm(weight=3.0, minimum_rows=8, **SHIFT)
    both = SessionICTerm(weight=3.0, minimum_rows=8, plain_weight=2.0, **SHIFT)
    identity, before = both.identity(), shifted.identity()
    assert {key: identity[key] for key in before} == before
    assert set(identity) - set(before) == {"plain_weight", "plain"}
    assert identity["plain_weight"] == 2.0
    declared = dict(weight=3.0, minimum_rows=8, plain_weight=2.0, **SHIFT)
    assert parse_session_ic_term(declared) == both
    weighted = parse_session_ic_term(dict(declared, liquidity_column=-1, liquidity_power=1.0))
    assert weighted.weighted and weighted.shifted and weighted.plain_weight == 2.0
    # Un entero se guarda como número real, igual que el peso.
    assert type(SessionICTerm(plain_weight=2, **SHIFT).plain_weight) is float


@pytest.mark.parametrize(
    "wrong",
    [
        dict(plain_weight=1.0),
        dict(plain_weight=1.0, liquidity_column=-1, liquidity_power=1.0),
        dict(plain_weight=0.0, **SHIFT),
        dict(plain_weight=-1.0, **SHIFT),
        dict(plain_weight=1e4, **SHIFT),
        dict(plain_weight=float("nan"), **SHIFT),
        dict(plain_weight="1.0", **SHIFT),
        dict(plain_weight=True, **SHIFT),
    ],
)
def test_invalid_plain_weights_are_rejected(wrong):
    with pytest.raises(ValueError):
        SessionICTerm(**wrong)
    with pytest.raises(ValueError):
        parse_session_ic_term(dict(weight=1.0, minimum_rows=8, **wrong))


def test_the_known_offset_is_zero_where_the_scaled_value_overflows():
    # Finito en FP32 y fuera de rango al multiplicar por la escala: antes salía infinito.
    column = torch.tensor([1e38, -1e38, 1.5, float("nan"), float("inf")], dtype=torch.float32)
    result = known_offset(column, 100.0)
    assert torch.isfinite(result).all()
    assert result.tolist() == pytest.approx([0.0, 0.0, 150.0, 0.0, 0.0])


def test_an_offset_that_is_not_finite_is_refused_by_the_term():
    point, target, sessions = sample()
    term = SessionICTerm(minimum_rows=5, **SHIFT)
    for wrong in (float("inf"), float("nan")):
        offset = offsets()
        offset[3] = wrong
        with pytest.raises(ValueError, match="desplazamiento"):
            term.term(point, target, sessions, offset=offset)


def numpy_top_third_weights(liquidity_values, weight):
    """Referencia independiente de pesos brutos del tercio superior de liquidez de una sesión."""
    arr = np.asarray(liquidity_values, dtype=np.float64)
    result = np.ones(len(arr), dtype=np.float64)
    if weight == 1.0:
        return result
    known = np.isfinite(arr)
    if not np.any(known):
        return result
    ordered = np.sort(arr[known])
    threshold = ordered[(2 * len(ordered)) // 3]
    result[known & (arr >= threshold)] = float(weight)
    return result


def numpy_weighted_pearson(left, right, weights):
    """Correlación ponderada independiente calculada con numpy.cov(..., aweights=...)."""
    cov = np.cov(
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
        aweights=np.asarray(weights, dtype=np.float64),
    )
    return float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))


def numpy_session_term_reference(
    point,
    target,
    sessions,
    *,
    weight,
    minimum_rows,
    liquidity=None,
    liquidity_weight=1.0,
    stability=0.0,
):
    """Referencia independiente con numpy.cov(aweights) y numpy.std(ddof=1)."""
    groups = {}
    for row, key in enumerate(sessions):
        groups.setdefault(key, []).append(row)
    correlations = []
    for rows in groups.values():
        if len(rows) < minimum_rows:
            continue
        p_rows = [point[row] for row in rows]
        g_rows = normal_scores([target[row] for row in rows])
        if liquidity is not None and liquidity_weight > 1.0:
            row_w = numpy_top_third_weights([liquidity[row] for row in rows], liquidity_weight)
            correlations.append(numpy_weighted_pearson(p_rows, g_rows, row_w))
        else:
            correlations.append(pearson(p_rows, g_rows))
    if not correlations:
        return 0.0
    arr = np.asarray(correlations, dtype=np.float64)
    value = -float(weight) * float(arr.mean())
    if stability > 0.0 and len(arr) >= 2:
        value += float(stability) * float(weight) * float(np.std(arr, ddof=1))
    return value


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_absent_and_default_wp1_options_match_baseline_bitwise(dtype):
    point, target, sessions, liquidity = liquid_sample()
    point = point.to(dtype)
    target = target.to(dtype)
    liquidity = liquidity.to(dtype)
    baseline = SessionICTerm(weight=2.5, minimum_rows=5)
    base_leaf = point.clone().requires_grad_(True)
    base_val = baseline.term(base_leaf, target, sessions)
    (base_grad,) = torch.autograd.grad(base_val, base_leaf)
    neutral_variants = [
        (SessionICTerm(weight=2.5, minimum_rows=5, liquidity_weight=1.0), None),
        (SessionICTerm(weight=2.5, minimum_rows=5, stability=0.0), None),
        (SessionICTerm(weight=2.5, minimum_rows=5, liquidity_weight=1.0, stability=0.0), None),
        (
            SessionICTerm(weight=2.5, minimum_rows=5, liquidity_column=-1, liquidity_weight=1.0),
            None,
        ),
        (
            SessionICTerm(weight=2.5, minimum_rows=5, liquidity_column=-1, liquidity_weight=1.0),
            liquidity,
        ),
        (
            parse_session_ic_term(
                dict(weight=2.5, minimum_rows=5, liquidity_weight=1, stability=0)
            ),
            None,
        ),
    ]
    for variant, liq in neutral_variants:
        assert variant.identity() == baseline.identity()
        leaf = point.clone().requires_grad_(True)
        val = variant.term(leaf, target, sessions, liquidity=liq)
        (grad,) = torch.autograd.grad(val, leaf)
        torch.testing.assert_close(val, base_val, rtol=0, atol=0)
        torch.testing.assert_close(grad, base_grad, rtol=0, atol=0)


def test_top_third_weights_handle_sizes_ties_missing_and_permutation_invariance():
    from mars_titan.models.titans.session_ic_objective import (
        session_row_weights,
        top_third_weights,
    )

    for size in (3, 4, 5, 6, 8, 9, 12):
        # Incluye valores negativos, igual que log_dollar_volume_21.
        values = torch.linspace(-4.0, 2.0, size, dtype=torch.float64)
        actual = top_third_weights(values, 3.5)
        expected = torch.tensor(numpy_top_third_weights(values.tolist(), 3.5), dtype=torch.float64)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert int((actual == 3.5).sum()) == size - (2 * size) // 3
        assert int((actual == 1.0).sum()) == (2 * size) // 3
    # Empates en el umbral: todas las filas empatadas comparten peso sin depender del orden.
    tied = torch.tensor([1.0, 2.0, 3.0, 5.0, 5.0, 5.0], dtype=torch.float64)
    perm = torch.tensor([4, 0, 5, 2, 3, 1])
    tied_weights = top_third_weights(tied, 4.0)
    assert tied_weights.tolist() == [1.0, 1.0, 1.0, 4.0, 4.0, 4.0]
    torch.testing.assert_close(
        top_third_weights(tied[perm], 4.0), tied_weights[perm], rtol=0, atol=0
    )
    # Liquidez no finita (NaN, +inf, -inf) siempre pesa 1 y no entra en el umbral.
    mixed = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), -2.0, 0.0, 1.5], dtype=torch.float64
    )
    assert top_third_weights(mixed, 5.0).tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 5.0]
    # Toda la liquidez ausente: todas pesan 1.
    all_missing = torch.tensor([float("nan"), float("inf"), -float("inf")], dtype=torch.float64)
    assert top_third_weights(all_missing, 5.0).tolist() == [1.0, 1.0, 1.0]
    # Por sesión y mercado: otra sesión o mercado con liquidez extrema no cambia nada.
    sessions = [("US", 1)] * 6 + [("US", 2)] * 6 + [("CN", 1)] * 6
    base_liq = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 3, dtype=torch.float64)
    perturbed = base_liq.clone()
    perturbed[6:12] = perturbed[6:12] + 1000.0
    perturbed[12:18] = float("nan")
    w_base = session_row_weights(base_liq, sessions, 3.0)
    w_pert = session_row_weights(perturbed, sessions, 3.0)
    torch.testing.assert_close(w_base[:6], w_pert[:6], rtol=0, atol=0)
    assert w_pert[12:18].tolist() == [1.0] * 6


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("liquidity_weight", [1.0, 2.0, 3.0, 5.0])
def test_liquidity_weight_matches_numpy_cov_aweights_and_reduces_at_one(dtype, liquidity_weight):
    point, target, sessions, liquidity = liquid_sample(rows=42, seed=17)
    point = point.to(dtype)
    target = target.to(dtype)
    liquidity = liquidity.to(dtype)
    term = SessionICTerm(
        weight=2.5, minimum_rows=5, liquidity_column=-29, liquidity_weight=liquidity_weight
    )
    leaf = point.clone().requires_grad_(True)
    liq_leaf = liquidity.clone().requires_grad_(True)
    actual = term.term(leaf, target, sessions, liquidity=liq_leaf)
    expected = numpy_session_term_reference(
        point.tolist(),
        target.tolist(),
        sessions,
        weight=2.5,
        minimum_rows=5,
        liquidity=liquidity.tolist(),
        liquidity_weight=liquidity_weight,
    )
    rel = 1e-5 if dtype == torch.float32 else 1e-8
    assert actual.item() == pytest.approx(expected, rel=rel, abs=1e-7)
    grad, liq_grad = torch.autograd.grad(actual, (leaf, liq_leaf), allow_unused=True)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert liq_grad is None
    plain = SessionICTerm(weight=2.5, minimum_rows=5).term(point, target, sessions)
    if liquidity_weight == 1.0:
        torch.testing.assert_close(actual, plain, rtol=0, atol=0)
    else:
        assert abs(actual.item() - plain.item()) > 1e-4
        # Mover el punto contra el gradiente reduce el término ponderado.
        moved = point.detach() - 0.05 * grad
        assert term.term(moved, target, sessions, liquidity=liquidity) < actual


def test_top_third_selection_never_leaks_across_sessions_or_markets():
    point, target, sessions, liquidity = liquid_sample(rows=36, seed=23)
    # 3 sesiones: ("US", 1), ("US", 2), ("CN", 1), 12 filas cada una, intercaladas.
    term = SessionICTerm(weight=2.0, minimum_rows=5, liquidity_column=-29, liquidity_weight=4.0)
    target_rows = [i for i, key in enumerate(sessions) if key == ("US", 1)]
    other_rows = [i for i, key in enumerate(sessions) if key != ("US", 1)]
    perturbed_liq = liquidity.clone()
    # Cambiar radicalmente la liquidez de las otras sesiones (otro instante y otro mercado).
    for row in other_rows:
        perturbed_liq[row] = 1e6 if sessions[row][0] == "US" else float("nan")
    p1 = point.clone().requires_grad_(True)
    p2 = point.clone().requires_grad_(True)
    c1 = session_correlations(p1, target, sessions, 5, liquidity=liquidity, liquidity_weight=4.0)
    c2 = session_correlations(
        p2, target, sessions, 5, liquidity=perturbed_liq, liquidity_weight=4.0
    )
    # La primera sesión aparece primero -> índice 0: correlación y gradiente idénticos bit a bit.
    torch.testing.assert_close(c1[0], c2[0], rtol=0, atol=0)
    (g1,) = torch.autograd.grad(term.term(p1, target, sessions, liquidity=liquidity), p1)
    (g2,) = torch.autograd.grad(term.term(p2, target, sessions, liquidity=perturbed_liq), p2)
    torch.testing.assert_close(g1[target_rows], g2[target_rows], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stability_penalizes_cross_session_dispersion_in_the_right_direction(dtype):
    # Dos lotes con exactamente la misma correlación media por sesión y distinta dispersión.
    n = 12
    target_one = torch.linspace(-1.0, 1.0, n, dtype=dtype)
    g = torch.tensor(normal_scores(target_one.tolist()), dtype=dtype)
    # Vector ortogonal a g con la misma norma para fijar la correlación exactamente a ρ.
    orth = torch.cos(torch.linspace(0.0, 3.0 * np.pi, n, dtype=dtype))
    orth = orth - orth.mean()
    orth = orth - ((orth * g).sum() / (g * g).sum()) * g
    orth = orth * (g.norm() / orth.norm())

    def with_corr(rho):
        return rho * g + float(np.sqrt(1.0 - rho * rho)) * orth

    sessions = [("US", 1)] * n + [("US", 2)] * n + [("US", 3)] * n
    target = torch.cat([target_one, target_one, target_one])
    # Lote regular: correlaciones (0.3, 0.3, 0.3), media 0.3, std 0.0.
    regular = torch.cat([with_corr(0.3), with_corr(0.3), with_corr(0.3)])
    # Lote irregular: correlaciones (-0.1, 0.3, 0.7), misma media 0.3, std = 0.4.
    irregular = torch.cat([with_corr(-0.1), with_corr(0.3), with_corr(0.7)])

    term_plain = SessionICTerm(weight=2.0, minimum_rows=5, stability=0.0)
    term_stable = SessionICTerm(weight=2.0, minimum_rows=5, stability=0.5)
    # Sin stability ambos lotes tienen exactamente la misma pérdida.
    torch.testing.assert_close(
        term_plain.term(regular, target, sessions),
        term_plain.term(irregular, target, sessions),
        rtol=1e-5,
        atol=1e-6,
    )
    # Con stability el lote irregular tiene pérdida estrictamente mayor en stability * weight * std.
    loss_reg = term_stable.term(regular, target, sessions)
    loss_irr = term_stable.term(irregular, target, sessions)
    expected_std = float(np.std([-0.1, 0.3, 0.7], ddof=1))
    assert loss_irr.item() > loss_reg.item()
    assert (loss_irr - loss_reg).item() == pytest.approx(
        0.5 * 2.0 * expected_std, rel=1e-5, abs=1e-6
    )
    expected_irr = numpy_session_term_reference(
        irregular.tolist(), target.tolist(), sessions, weight=2.0, minimum_rows=5, stability=0.5
    )
    assert loss_irr.item() == pytest.approx(expected_irr, rel=1e-5, abs=1e-6)
    # Con exactamente dos sesiones (frontera de truncation=2) stability ya penaliza.
    two_sessions = [("US", 1)] * n + [("US", 2)] * n
    two_target = torch.cat([target_one, target_one])
    two_reg = torch.cat([with_corr(0.3), with_corr(0.3)])
    two_irr = torch.cat([with_corr(-0.1), with_corr(0.7)])
    two_std = float(np.std([-0.1, 0.7], ddof=1))
    assert (
        term_stable.term(two_irr, two_target, two_sessions)
        - term_stable.term(two_reg, two_target, two_sessions)
    ).item() == pytest.approx(0.5 * 2.0 * two_std, rel=1e-5, abs=1e-6)
    # Con una sola sesión válida, stability no añade nada (coincide bit a bit con stability=0).
    single_sessions = [("US", 1)] * n
    p_single = with_corr(0.3)
    torch.testing.assert_close(
        term_stable.term(p_single, target_one, single_sessions),
        term_plain.term(p_single, target_one, single_sessions),
        rtol=0,
        atol=0,
    )
    # El gradiente con stability empuja la sesión rezagada más que la aventajada.
    leaf = irregular.clone().requires_grad_(True)
    (g_plain,) = torch.autograd.grad(term_plain.term(leaf, target, sessions), leaf)
    (g_stab,) = torch.autograd.grad(term_stable.term(leaf, target, sessions), leaf)
    # En la sesión 0 (ρ = -0.1 < media), stability refuerza la dirección que sube su correlación.
    assert g_stab[:n].norm() > g_plain[:n].norm()
    # En la sesión 2 (ρ = 0.7 > media), stability reduce el empuje frente a la media sola.
    assert g_stab[2 * n :].norm() < g_plain[2 * n :].norm()


def test_liquidity_weight_and_stability_work_with_label_offset_and_both_modes():
    point, target, sessions, liquidity = liquid_sample(rows=42, seed=31)
    offset = offsets(rows=42, seed=41)
    # Modo 1: sobre la etiqueta.
    label_term = SessionICTerm(
        weight=2.0, minimum_rows=5, liquidity_column=-29, liquidity_weight=3.0, stability=0.4
    )
    exp_label = numpy_session_term_reference(
        point.tolist(),
        target.tolist(),
        sessions,
        weight=2.0,
        minimum_rows=5,
        liquidity=liquidity.tolist(),
        liquidity_weight=3.0,
        stability=0.4,
    )
    val_label = label_term.term(point, target, sessions, liquidity=liquidity)
    assert val_label.item() == pytest.approx(exp_label, rel=1e-8)

    # Modo 2: sobre lo operable con offset.
    op_term = SessionICTerm(
        weight=2.5,
        minimum_rows=5,
        liquidity_column=-29,
        liquidity_weight=3.0,
        stability=0.4,
        **SHIFT,
    )
    exp_op = numpy_session_term_reference(
        (point + offset).tolist(),
        (target + offset).tolist(),
        sessions,
        weight=2.5,
        minimum_rows=5,
        liquidity=liquidity.tolist(),
        liquidity_weight=3.0,
        stability=0.4,
    )
    val_op = op_term.term(point, target, sessions, liquidity=liquidity, offset=offset)
    assert val_op.item() == pytest.approx(exp_op, rel=1e-8)

    # Modo 3: ambos a la vez (offset + plain_weight): suma exacta bit a bit de los dos.
    both_term = SessionICTerm(
        weight=2.5,
        minimum_rows=5,
        liquidity_column=-29,
        liquidity_weight=3.0,
        stability=0.4,
        plain_weight=2.0,
        **SHIFT,
    )
    val_both = both_term.term(point, target, sessions, liquidity=liquidity, offset=offset)
    torch.testing.assert_close(val_both, val_op + val_label, rtol=0, atol=0)
    assert val_both.item() == pytest.approx(exp_op + exp_label, rel=1e-8)

    # Gradientes: offset y liquidity sin gradiente; point igual a la suma de ambos.
    p_leaf = point.clone().requires_grad_(True)
    l_leaf = liquidity.clone().requires_grad_(True)
    o_leaf = offset.clone().requires_grad_(True)
    g_both, g_l, g_o = torch.autograd.grad(
        both_term.term(p_leaf, target, sessions, liquidity=l_leaf, offset=o_leaf),
        (p_leaf, l_leaf, o_leaf),
        allow_unused=True,
    )
    (g_op,) = torch.autograd.grad(
        op_term.term(p_leaf, target, sessions, liquidity=liquidity, offset=offset), p_leaf
    )
    (g_lbl,) = torch.autograd.grad(
        label_term.term(p_leaf, target, sessions, liquidity=liquidity), p_leaf
    )
    assert g_l is None and g_o is None
    torch.testing.assert_close(g_both, g_op + g_lbl, rtol=1e-12, atol=1e-15)


def test_wp1_options_enter_identity_only_when_non_default_and_validate_inputs():
    plain = SessionICTerm(weight=3.0, minimum_rows=8)
    base_id = plain.identity()
    assert "liquidity_weight" not in base_id and "stability" not in base_id
    # Con valores por defecto explícitos no entran en identity().
    assert SessionICTerm(weight=3.0, minimum_rows=8, liquidity_weight=1.0).identity() == base_id
    assert (
        SessionICTerm(
            weight=3.0, minimum_rows=8, liquidity_column=-29, liquidity_weight=1.0
        ).identity()
        == base_id
    )
    assert SessionICTerm(weight=3.0, minimum_rows=8, stability=0.0).identity() == base_id

    # Con liquidity_weight > 1 entra en identity().
    lw = SessionICTerm(weight=3.0, minimum_rows=8, liquidity_column=-29, liquidity_weight=3)
    assert lw.weighted and lw.top_third_weighted and type(lw.liquidity_weight) is float
    lw_id = lw.identity()
    assert {k: lw_id[k] for k in base_id} == base_id
    assert lw_id["liquidity_column"] == -29 and lw_id["liquidity_weight"] == 3.0
    assert "stability" not in lw_id
    assert (
        parse_session_ic_term(
            dict(weight=3.0, minimum_rows=8, liquidity_column=-29, liquidity_weight=3)
        )
        == lw
    )

    # Con stability > 0 entra en identity().
    stab = SessionICTerm(weight=3.0, minimum_rows=8, stability=0.5)
    assert not stab.weighted and stab.stability == 0.5
    stab_id = stab.identity()
    assert {k: stab_id[k] for k in base_id} == base_id
    assert stab_id["stability"] == 0.5 and "liquidity_weight" not in stab_id
    assert parse_session_ic_term(dict(weight=3.0, minimum_rows=8, stability=0.5)) == stab

    # Combinadas juntas y con offset + plain_weight.
    full_dict = dict(
        weight=3.0,
        minimum_rows=8,
        liquidity_column=-29,
        liquidity_weight=3.0,
        stability=0.5,
        plain_weight=2.0,
        **SHIFT,
    )
    full = parse_session_ic_term(full_dict)
    full_id = full.identity()
    assert full_id["liquidity_weight"] == 3.0 and full_id["stability"] == 0.5
    assert full_id["plain_weight"] == 2.0 and full_id["offset_column"] == -3

    # También stability se combina con liquidity_power existente.
    power_stab = parse_session_ic_term(
        dict(weight=3.0, minimum_rows=8, liquidity_column=-29, liquidity_power=0.5, stability=0.25)
    )
    assert power_stab.weighted and not power_stab.top_third_weighted
    assert power_stab.identity()["liquidity_power"] == 0.5
    assert power_stab.identity()["stability"] == 0.25

    # `liquidity` obligatorio si liquidity_weight > 1.
    point, target, sessions, _ = liquid_sample()
    with pytest.raises(ValueError, match="liquidez"):
        lw.term(point, target, sessions)

    # Validación estricta de declaraciones inválidas.
    for wrong in (
        dict(liquidity_weight=2.0),
        dict(liquidity_column=-29, liquidity_weight=0.5),
        dict(liquidity_column=-29, liquidity_weight=0.0),
        dict(liquidity_column=-29, liquidity_weight=-1.0),
        dict(liquidity_column=-29, liquidity_weight=1e4),
        dict(liquidity_column=-29, liquidity_weight=float("nan")),
        dict(liquidity_column=-29, liquidity_weight=float("inf")),
        dict(liquidity_column=-29, liquidity_weight=True),
        dict(liquidity_column=-29, liquidity_weight="3.0"),
        dict(liquidity_column=-29, liquidity_weight=None),
        dict(liquidity_column=0, liquidity_weight=2.0),
        dict(liquidity_column=3, liquidity_weight=2.0),
        dict(liquidity_column=-29, liquidity_power=0.5, liquidity_weight=2.0),
        dict(stability=-0.1),
        dict(stability=1e4),
        dict(stability=float("nan")),
        dict(stability=float("inf")),
        dict(stability=True),
        dict(stability=False),
        dict(stability="0.5"),
        dict(stability=None),
    ):
        with pytest.raises(ValueError):
            parse_session_ic_term(dict(weight=3.0, minimum_rows=8, **wrong))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_finite_deterministic_gradients_for_constant_single_short_and_missing_liquidity(dtype):
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        term = SessionICTerm(
            weight=3.0, minimum_rows=5, liquidity_column=-29, liquidity_weight=3.0, stability=1.0
        )
        point, target, sessions, liquidity = liquid_sample(rows=30, seed=7)
        point, target, liquidity = point.to(dtype), target.to(dtype), liquidity.to(dtype)

        # 1. Predicción constante cero producida por una cabeza lineal que arranca en cero.
        weight = torch.zeros(16, dtype=dtype, requires_grad=True)
        features = torch.randn(30, 16, dtype=dtype, generator=torch.Generator().manual_seed(9))
        zero_pred = features @ weight
        loss_zero = term.term(zero_pred, target, sessions, liquidity=liquidity)
        assert torch.isfinite(loss_zero)
        (g_weight,) = torch.autograd.grad(loss_zero, weight)
        assert torch.isfinite(g_weight).all() and torch.isfinite(torch.linalg.vector_norm(g_weight))

        # 2. Predicción constante no nula.
        const_pred = torch.full_like(point, 5.0).requires_grad_(True)
        loss_const = term.term(const_pred, target, sessions, liquidity=liquidity)
        (g_const,) = torch.autograd.grad(loss_const, const_pred)
        assert torch.isfinite(loss_const) and torch.isfinite(g_const).all()

        # 3. Dos sesiones idénticas (correlaciones idénticas != 0 -> varianza exacta 0).
        s_one = point[:10]
        t_one = target[:10]
        l_one = liquidity[:10]
        dup_p = torch.cat([s_one, s_one]).requires_grad_(True)
        dup_t = torch.cat([t_one, t_one])
        dup_l = torch.cat([l_one, l_one])
        dup_s = [("US", 1)] * 10 + [("US", 2)] * 10
        loss_dup = term.term(dup_p, dup_t, dup_s, liquidity=dup_l)
        (g_dup,) = torch.autograd.grad(loss_dup, dup_p)
        assert torch.isfinite(loss_dup) and torch.isfinite(g_dup).all() and g_dup.abs().sum() > 0

        # 4. Una sola sesión válida.
        one_p = s_one.clone().requires_grad_(True)
        loss_one = term.term(one_p, t_one, [("US", 1)] * 10, liquidity=l_one)
        (g_one,) = torch.autograd.grad(loss_one, one_p)
        assert torch.isfinite(loss_one) and torch.isfinite(g_one).all()

        # 5. Todas las sesiones por debajo de minimum_rows.
        short_p = point[:8].clone().requires_grad_(True)
        short_s = [("US", 1)] * 4 + [("US", 2)] * 4
        loss_short = term.term(short_p, target[:8], short_s, liquidity=liquidity[:8])
        (g_short,) = torch.autograd.grad(loss_short, short_p)
        assert loss_short.item() == 0.0 and torch.count_nonzero(g_short) == 0

        # 6. Liquidez ausente parcial y totalmente ausente (NaN / ±inf).
        for bad_liq in (
            torch.where(torch.arange(30) % 2 == 0, float("nan"), liquidity),
            torch.full_like(liquidity, float("nan")),
            torch.tensor([float("nan"), float("inf"), -float("inf")] * 10, dtype=dtype),
        ):
            p_miss = point.clone().requires_grad_(True)
            loss_miss = term.term(p_miss, target, sessions, liquidity=bad_liq)
            (g_miss,) = torch.autograd.grad(loss_miss, p_miss)
            assert torch.isfinite(loss_miss) and torch.isfinite(g_miss).all()

        # 7. Objetivo constante y puntuaciones grandes sin desbordamiento.
        p_ct = point.clone().requires_grad_(True)
        loss_ct = term.term(p_ct, torch.ones_like(target), sessions, liquidity=liquidity)
        (g_ct,) = torch.autograd.grad(loss_ct, p_ct)
        assert torch.isfinite(loss_ct) and torch.isfinite(g_ct).all()

        scale = 1e30 if dtype == torch.float32 else 1e200
        big_p = (point * scale).detach().requires_grad_(True)
        loss_big = term.term(big_p, target, sessions, liquidity=liquidity)
        (g_big,) = torch.autograd.grad(loss_big, big_p)
        assert torch.isfinite(loss_big) and torch.isfinite(g_big).all()
    finally:
        torch.use_deterministic_algorithms(previous)
