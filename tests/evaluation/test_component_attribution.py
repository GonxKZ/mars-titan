"""Álgebra de la atribución por componentes con linajes de juguete.

Los valores de cada brazo son funciones conocidas de sus componentes, así que cada efecto
esperado se calcula a mano. No hay predicciones, modelos ni datos.
"""

import itertools
import math

import numpy as np
import pytest

from mars_titan.evaluation import component_attribution as attribution


def lineage(**changes):
    declared = dict(
        question="¿Cuánto aporta cada componente?",
        components={
            "a": dict(label="Componente a", requires=[]),
            "b": dict(label="Componente b", requires=["a"]),
            "c": dict(label="Componente c", requires=[]),
        },
        arms={
            "base": [],
            "with_a": ["a"],
            "with_ab": ["a", "b"],
            "with_c": ["c"],
            "with_ac": ["a", "c"],
            "full": ["a", "b", "c"],
        },
        ladder=["a", "b", "c"],
        full=["a", "b", "c"],
        interactions=[dict(first="a", second="c", context="base")],
        games=[dict(players=["a", "c"], context="base")],
    )
    declared.update(changes)
    return attribution.load_lineage("toy", declared)


def value(members, weights, pair=0.0):
    """Valor conocido de un conjunto, que suma los pesos y añade una interacción entre a y c."""
    total = sum(weights[c] for c in members)
    return total + (pair if {"a", "c"} <= set(members) else 0.0)


def apply(toy, terms, weights, pair=0.0):
    coefficients, unnamed = attribution.resolve(toy, terms)
    assert unnamed == []
    return math.fsum(c * value(toy.arms[arm], weights, pair) for arm, c in coefficients.items())


WEIGHTS = dict(a=1.5, b=-0.25, c=0.75)


def test_ladder_steps_follow_the_declared_order_and_add_up_to_the_total():
    toy = lineage()
    steps = attribution.ladder(toy)
    assert list(steps) == ["+a", "+b", "+c", "total"]
    resolved = {name: attribution.resolve(toy, terms)[0] for name, terms in steps.items()}
    assert resolved["+b"] == {"with_ab": 1.0, "with_a": -1.0}
    assert resolved["+c"] == {"full": 1.0, "with_ab": -1.0}
    assert resolved["total"] == {"full": 1.0, "base": -1.0}
    summed = {}
    for name in ("+a", "+b", "+c"):
        for arm, c in resolved[name].items():
            summed[arm] = summed.get(arm, 0.0) + c
    assert {arm: c for arm, c in summed.items() if c} == resolved["total"]
    for name, component in (("+a", "a"), ("+b", "b"), ("+c", "c")):
        assert apply(toy, steps[name], WEIGHTS, pair=0.4) == pytest.approx(
            WEIGHTS[component] + (0.4 if component == "c" else 0.0)
        )


def test_leaving_a_component_out_also_removes_what_depends_on_it():
    toy = lineage()
    loo = attribution.leave_one_out(toy)
    assert attribution.resolve(toy, loo["-a"])[0] == {"full": 1.0, "with_c": -1.0}
    assert attribution.removed_with(toy, "a") == ["a", "b"]
    assert attribution.removed_with(toy, "b") == ["b"]
    assert attribution.resolve(toy, loo["-b"])[0] == {"full": 1.0, "with_ac": -1.0}
    assert attribution.resolve(toy, loo["-c"])[0] == {"full": 1.0, "with_ab": -1.0}


def test_conditional_effects_cover_every_named_pair_that_differs_in_one_component():
    toy = lineage()
    effects = attribution.conditional_effects(toy)
    expected = {
        "a@base": ("base", "with_a"),
        "c@base": ("base", "with_c"),
        "b@with_a": ("with_a", "with_ab"),
        "c@with_a": ("with_a", "with_ac"),
        "a@with_c": ("with_c", "with_ac"),
        "c@with_ab": ("with_ab", "full"),
        "b@with_ac": ("with_ac", "full"),
    }
    assert set(effects) == set(expected)
    for name, (smaller, larger) in expected.items():
        assert attribution.resolve(toy, effects[name])[0] == {larger: 1.0, smaller: -1.0}
    # El efecto de c depende de que esté a, y la diferencia entre contextos es la interacción.
    assert apply(toy, effects["c@with_a"], WEIGHTS, pair=0.4) - apply(
        toy, effects["c@base"], WEIGHTS, pair=0.4
    ) == pytest.approx(0.4)


def test_the_interaction_is_the_double_difference_and_vanishes_when_effects_add():
    toy = lineage()
    terms = attribution.interactions(toy)["axc@base"]
    assert attribution.resolve(toy, terms)[0] == {
        "with_ac": 1.0,
        "with_a": -1.0,
        "with_c": -1.0,
        "base": 1.0,
    }
    assert apply(toy, terms, WEIGHTS, pair=0.4) == pytest.approx(0.4)
    assert apply(toy, terms, WEIGHTS) == pytest.approx(0.0)


def test_two_player_shapley_splits_the_interaction_in_halves_and_is_efficient():
    toy = lineage()
    values, limitations = attribution.shapley(toy)
    assert limitations == []
    assert set(values) == {"phi_a@base", "phi_c@base", "total@base"}
    assert apply(toy, values["phi_a@base"], WEIGHTS, pair=0.4) == pytest.approx(1.5 + 0.2)
    assert apply(toy, values["phi_c@base"], WEIGHTS, pair=0.4) == pytest.approx(0.75 + 0.2)
    assert attribution.resolve(toy, values["phi_a@base"])[0] == {
        "with_a": 0.5,
        "base": -0.5,
        "with_ac": 0.5,
        "with_c": -0.5,
    }


def random_game(rng, players):
    names = [f"p{i}" for i in range(players)]
    components = {name: dict(label=name, requires=[]) for name in names}
    arms = {
        "_".join(subset) or "none": list(subset)
        for size in range(players + 1)
        for subset in itertools.combinations(names, size)
    }
    declared = dict(
        question="juego",
        components=components,
        arms=arms,
        ladder=names,
        full=names,
        interactions=[],
        games=[dict(players=names, context="none")],
    )
    values = {arm: float(rng.normal()) for arm in arms}
    return attribution.load_lineage("game", declared), values, names


@pytest.mark.parametrize("players", [2, 3, 4, 5])
def test_shapley_values_are_efficient_symmetric_and_zero_for_dummies(players):
    rng = np.random.default_rng(players)
    toy, values, names = random_game(rng, players)
    phi, _ = attribution.shapley(toy)

    def evaluate(terms, table):
        coefficients, unnamed = attribution.resolve(toy, terms)
        assert unnamed == []
        return math.fsum(c * table[arm] for arm, c in coefficients.items())

    total = evaluate(phi["total@none"], values)
    assert total == pytest.approx(values["_".join(names)] - values["none"])
    shares = [evaluate(phi[f"phi_{name}@none"], values) for name in names]
    assert math.fsum(shares) == pytest.approx(total, abs=1e-12)
    # Un jugador nulo recibe cero y dos jugadores simétricos reciben lo mismo.
    weights = rng.normal(size=players)
    additive = {
        arm: math.fsum(weights[names.index(c)] for c in toy.arms[arm] if c != names[0])
        for arm in values
    }
    assert evaluate(phi[f"phi_{names[0]}@none"], additive) == pytest.approx(0.0, abs=1e-12)
    for i, name in enumerate(names[1:], start=1):
        assert evaluate(phi[f"phi_{name}@none"], additive) == pytest.approx(weights[i])


def test_a_game_with_coalitions_that_break_dependencies_is_a_declared_limitation():
    toy = lineage(games=[dict(players=["a", "b", "c"], context="base")])
    values, limitations = attribution.shapley(toy)
    assert values == {}
    assert limitations == [
        dict(
            players=["a", "b", "c"],
            context="base",
            coalitions=8,
            structurally_invalid=2,
            reason=limitations[0]["reason"],
        )
    ]


def test_sets_without_a_named_arm_stay_unidentified():
    toy = lineage(arms={"base": [], "with_a": ["a"], "full": ["a", "b", "c"]}, games=[])
    loo = attribution.leave_one_out(toy)
    coefficients, unnamed = attribution.resolve(toy, loo["-b"])
    assert coefficients == {"full": 1.0}
    assert unnamed == [["a", "c"]]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            dict(
                components={
                    "a": dict(label="a", requires=["b"]),
                    "b": dict(label="b", requires=["a"]),
                    "c": dict(label="c", requires=[]),
                }
            ),
            "ciclo",
        ),
        (dict(arms={"base": [], "only_b": ["b"]}), "dependencias"),
        (dict(arms={"base": [], "again": []}), "mismos componentes"),
        (dict(ladder=["b", "a"]), "antes de sus dependencias"),
        (dict(full=["b"]), "no está cerrado"),
        (dict(interactions=[dict(first="a", second="c", context="with_a")]), "ya contiene"),
        (dict(interactions=[dict(first="b", second="c", context="base")]), "dependencias"),
        (dict(games=[dict(players=["a", "c"], context="with_c")]), "ya contiene algún jugador"),
        (dict(games=[dict(players=["a"], context="base")]), "entre 2"),
        (dict(arms={"base": ["z"]}), "no declarados"),
    ],
)
def test_inconsistent_lineages_are_rejected(changes, message):
    with pytest.raises(ValueError, match=message):
        lineage(**changes)
