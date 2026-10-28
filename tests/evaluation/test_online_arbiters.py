"""Núcleo AdaNormalHedge de los árbitros en línea (#789), en CPU y sin ajustar nada."""

import math

import numpy as np
import pyarrow as pa
import pytest

from mars_titan.evaluation import online_arbiters as arbiters
from mars_titan.evaluation import scale_diagnostics as sd
from mars_titan.evaluation.forecast_panel import ForecastPanel

DAY_US = 86_400_000_000


def reference_rounds(prior, losses, awake):
    """Algoritmo 1 de Luo y Schapire (2015) con la ecuación (3), escrito opción a opción."""
    total = math.fsum(prior)
    q = [value / total for value in prior]
    regret, magnitude = [0.0] * len(q), [0.0] * len(q)

    def phi(r, c):
        return 1.0 if r <= 0 else math.exp(r * r / (3 * c))

    mixtures = []
    for loss, mask in zip(losses, awake, strict=True):
        mass = [
            q[i]
            * 0.5
            * (phi(regret[i] + 1, magnitude[i] + 1) - phi(regret[i] - 1, magnitude[i] + 1))
            if mask[i]
            else 0.0
            for i in range(len(q))
        ]
        if not any(mask):
            mixtures.append(None)
            continue
        if math.fsum(mass) == 0:
            mass = [q[i] if mask[i] else 0.0 for i in range(len(q))]
        mixture = [value / math.fsum(mass) for value in mass]
        mixed = math.fsum(mixture[i] * loss[i] for i in range(len(q)) if mask[i])
        for i in range(len(q)):
            instant = mixed - loss[i] if mask[i] else 0.0
            regret[i] += instant
            magnitude[i] += abs(instant)
        mixtures.append(mixture)
    return mixtures, regret, magnitude


def sequence(seed, rounds=300, options=5, sleeping=0.2):
    rng = np.random.default_rng(seed)
    # Pérdidas con una opción mejor que cambia a mitad de la secuencia.
    base = rng.uniform(0.3, 0.7, size=(rounds, options))
    base[: rounds // 2, 0] -= 0.25
    base[rounds // 2 :, 3] -= 0.25
    losses = np.clip(base + rng.normal(0, 0.1, size=base.shape), 0, 1)
    awake = rng.random((rounds, options)) > sleeping
    awake[rng.random(rounds) < 0.05] = False
    return losses, awake


def run(prior, losses, awake):
    state = arbiters.AdaNormalHedge.start(prior)
    mixtures, bounds = [], []
    for loss, mask in zip(losses, awake, strict=True):
        mixtures.append(state.distribution(mask))
        state.update(loss, mask)
        bounds.append(state.potential_bound())
    return state, mixtures, bounds


def test_the_weight_is_the_difference_of_potentials_of_the_article():
    for regret, magnitude in [(0.0, 0.0), (0.4, 1.0), (-0.5, 2.0), (3.0, 5.0), (12.0, 40.0)]:
        expected = 0.5 * (
            math.exp(max(regret + 1, 0) ** 2 / (3 * (magnitude + 1)))
            - math.exp(max(regret - 1, 0) ** 2 / (3 * (magnitude + 1)))
        )
        assert arbiters.weights(regret, magnitude) == pytest.approx(expected, rel=1e-13)
    assert arbiters.potential(0.0, 0.0) == 1.0 and arbiters.potential(-2.0, 3.0) == 1.0
    assert arbiters.potential(3.0, 3.0) == pytest.approx(math.e)
    # Sin arrepentimiento por encima de −1 la opción no recibe peso.
    assert arbiters.weights(-1.0, 4.0) == 0.0 and arbiters.weights(-7.0, 9.0) == 0.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_rounds_match_a_reference_written_option_by_option(seed):
    losses, awake = sequence(seed)
    prior = [1.0, 2.0, 1.0, 0.5, 1.5]
    state, mixtures, _ = run(prior, losses, awake)
    expected, regret, magnitude = reference_rounds(prior, losses.tolist(), awake.tolist())
    # La referencia resta exponenciales casi iguales cerca de R = −1, por eso la tolerancia
    # relativa no baja de 1e-9. Un error en la fórmula daría diferencias de orden uno.
    for got, want in zip(mixtures, expected, strict=True):
        assert (got is None) == (want is None)
        if want is not None:
            np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(state.regret, regret, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(state.magnitude, magnitude, rtol=1e-9, atol=1e-12)
    assert state.rounds == sum(mask.any() for mask in awake)


def adversarial(rounds=400, options=4):
    """La mejor opción rota cada 25 rondas y las demás pierden todo."""
    losses = np.ones((rounds, options))
    for t in range(rounds):
        losses[t, (t // 25) % options] = 0.0
    return losses, np.ones_like(losses, dtype=bool)


@pytest.mark.parametrize("case", ["random", "sleeping", "adversarial", "constant"])
def test_the_potential_respects_the_bound_of_lemma_2_after_every_round(case):
    if case == "adversarial":
        losses, awake = adversarial()
    elif case == "constant":
        losses = np.tile([0.0, 1.0, 1.0], (300, 1))
        awake = np.ones_like(losses, dtype=bool)
    else:
        losses, awake = sequence(3, sleeping=0.0 if case == "random" else 0.4)
    _, _, bounds = run(np.ones(losses.shape[1]), losses, awake)
    for total, bound in bounds:
        assert total <= bound * (1 + 1e-12)


@pytest.mark.parametrize("case", ["random", "sleeping", "adversarial"])
def test_the_regret_to_any_fixed_mixture_respects_theorem_1(case):
    losses, awake = (
        adversarial()
        if case == "adversarial"
        else sequence(5, sleeping=0.0 if case == "random" else 0.3)
    )
    options = losses.shape[1]
    prior = np.arange(1, options + 1, dtype=float)
    state, _, _ = run(prior, losses, awake)
    q = state.prior
    _, bound = state.potential_bound()
    rng = np.random.default_rng(9)
    competitors = [np.eye(options)[i] for i in range(options)]
    competitors += list(rng.dirichlet(np.ones(options), size=20))
    for u in competitors:
        regret = float(u @ state.regret)
        weight = float(u @ state.magnitude)
        support = u > 0
        entropy = float(np.sum(u[support] * np.log(u[support] / q[support])))
        # Ecuación (2) para una sola opción, que es uniforme sobre un subconjunto, y (1) para
        # las demás mezclas.
        single = support.sum() == 1
        extra = 1.0 if single else math.log(1 + math.log(options))
        assert regret <= math.sqrt(3 * weight * (entropy + math.log(bound) + extra)) + 1e-9


def reference_tracking(prior, losses, awake):
    """AdaNormalHedge.TV con la fórmula de la sección 5.1: sumas de arrepentimiento por
    intervalo de las opciones originales, sin copias explícitas."""
    q = np.asarray(prior, dtype=float) / np.sum(prior)
    history, mixtures = [], []
    for loss, mask in zip(losses, awake, strict=True):
        if not mask.any():
            mixtures.append(None)
            continue
        mass = np.zeros(len(q))
        for start in range(len(history) + 1):
            interval = np.array(history[start:]).reshape(-1, len(q))
            regret, magnitude = interval.sum(axis=0), np.abs(interval).sum(axis=0)
            for i in range(len(q)):
                upper = max(regret[i] + 1, 0) ** 2 / (3 * (magnitude[i] + 1))
                lower = max(regret[i] - 1, 0) ** 2 / (3 * (magnitude[i] + 1))
                mass[i] += 0.5 * (math.exp(upper) - math.exp(lower)) / (start + 1) ** 2
        mass = np.where(mask, q * mass, 0.0)
        mixture = mass / mass.sum()
        mixed = float(mixture[mask] @ loss[mask])
        history.append(np.where(mask, mixed - loss, 0.0))
        mixtures.append(mixture)
    return mixtures


def switching(rounds=200, options=3, switch=100):
    """La mejor opción pasa de la 0 a la 1 a mitad de la secuencia."""
    losses = np.full((rounds, options), 0.6)
    losses[:switch, 0] = 0.2
    losses[switch:, 1] = 0.2
    return losses, np.ones_like(losses, dtype=bool)


def tracking_run(prior, losses, awake):
    state = arbiters.AdaNormalHedgeTV.start(prior)
    mixtures, bounds = [], []
    for loss, mask in zip(losses, awake, strict=True):
        mixtures.append(state.distribution(mask))
        state.update(loss, mask)
        bounds.append(state.potential_bound())
    return state, mixtures, bounds


@pytest.mark.parametrize("seed", [0, 1])
def test_the_tracking_form_matches_the_interval_sums_of_the_article(seed):
    losses, awake = sequence(seed, rounds=60, options=4)
    prior = [1.0, 2.0, 1.0, 0.5]
    state, mixtures, _ = tracking_run(prior, losses, awake)
    for got, want in zip(mixtures, reference_tracking(prior, losses, awake), strict=True):
        assert (got is None) == (want is None)
        if want is not None:
            np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)
    assert state.rounds == state.regret.shape[0] == sum(mask.any() for mask in awake)


@pytest.mark.parametrize("case", ["random", "sleeping", "adversarial", "switching"])
def test_the_tracking_form_respects_lemma_2_and_the_bound_on_every_interval(case):
    if case == "adversarial":
        losses, awake = adversarial(rounds=160)
    elif case == "switching":
        losses, awake = switching()
    else:
        losses, awake = sequence(8, rounds=160, sleeping=0.0 if case == "random" else 0.3)
    state, _, bounds = tracking_run(np.ones(losses.shape[1]), losses, awake)
    for total, bound in bounds:
        assert total <= bound * (1 + 1e-12)
    # Teorema 3 con la ecuación (2) para cada copia (τ, i): el intervalo que empieza en τ.
    _, bound = state.potential_bound()
    start = np.arange(1, state.rounds + 2, dtype=float)
    prior = state.prior[None, :] / start[:, None] ** 2
    prior = prior / prior.sum()
    for row in range(state.rounds):
        for option in range(state.options):
            limit = math.sqrt(
                3
                * state.magnitude[row, option]
                * (math.log(1 / prior[row, option]) + math.log(bound) + 1)
            )
            assert state.regret[row, option] <= limit + 1e-9


def test_the_tracking_form_follows_a_change_that_the_fixed_one_misses():
    losses, awake = switching()
    _, fixed, _ = run(np.ones(3), losses, awake)
    _, tracking, _ = tracking_run(np.ones(3), losses, awake)
    # Sesenta rondas después del cambio la forma TV ya ha girado y la fija sigue con la
    # opción anterior. Con estas pérdidas la TV supera 0,9 en la ronda 143 y la fija, con
    # una secuencia más larga, en la 217.
    assert fixed[160][0] > 0.9 and tracking[160][1] > 0.9
    assert tracking[199][1] > fixed[199][1]


def test_a_sleeping_option_gets_no_weight_and_keeps_its_state():
    state = arbiters.AdaNormalHedge.start([1, 1, 1])
    state.update([0.2, 0.8, 0.5], np.array([True, True, True]))
    before = (state.regret.copy(), state.magnitude.copy())
    mask = np.array([True, False, True])
    mixture = state.distribution(mask)
    assert mixture[1] == 0.0 and mixture.sum() == pytest.approx(1.0)
    state.update([0.1, np.nan, 0.9], mask)
    assert state.regret[1] == before[0][1] and state.magnitude[1] == before[1][1]
    assert state.regret[0] != before[0][0]


def test_a_round_without_options_mixes_nothing_and_changes_nothing():
    state = arbiters.AdaNormalHedge.start([1, 1])
    nobody = np.array([False, False])
    assert state.distribution(nobody) is None
    assert state.update([np.nan, np.nan], nobody) is None
    assert state.rounds == 0 and not state.regret.any() and not state.magnitude.any()


def test_the_prior_decides_when_every_awake_weight_is_zero():
    state = arbiters.AdaNormalHedge.start([1, 3, 4])
    state.regret = np.array([-1.5, -1.0, 2.0])
    state.magnitude = np.array([3.0, 2.0, 2.0])
    mixture = state.distribution(np.array([True, True, False]))
    np.testing.assert_allclose(mixture, [0.25, 0.75, 0.0])
    # Con la opción de peso positivo despierta, el prior ya no decide.
    assert state.distribution(np.array([True, True, True]))[2] == 1.0


@pytest.mark.parametrize(
    "prior, losses, mask, message",
    [
        ([1, 0], [0.5, 0.5], [True, True], "prior"),
        ([1, math.inf], [0.5, 0.5], [True, True], "prior"),
        ([], [], [], "prior"),
        ([1, 1], [0.5, 1.5], [True, True], r"\[0, 1\]"),
        ([1, 1], [0.5, np.nan], [True, True], r"\[0, 1\]"),
        ([1, 1], [0.5, -0.1], [True, True], r"\[0, 1\]"),
        ([1, 1], [0.5], [True, True], "una pérdida"),
        ([1, 1], [0.5, 0.5], [1, 1], "booleana"),
        ([1, 1], [0.5, 0.5], [True], "booleana"),
    ],
)
def test_invalid_rounds_are_rejected(prior, losses, mask, message):
    with pytest.raises(ValueError, match=message):
        state = arbiters.AdaNormalHedge.start(prior)
        state.update(losses, np.asarray(mask))


def delayed(seed=4, sessions=60, options=3, lag=3):
    rng = np.random.default_rng(seed)
    decided = 1_546_380_000_000_000 + np.arange(sessions, dtype=np.int64) * DAY_US
    matures = decided + lag * DAY_US + rng.integers(0, DAY_US, size=sessions)
    losses = rng.uniform(0, 1, size=(sessions, options))
    awake = rng.random((sessions, options)) > 0.15
    return decided, matures, losses, awake


def test_only_mature_losses_reach_a_decision():
    decided, matures, losses, awake = delayed()
    mixtures, _ = arbiters.mature_mixture(decided, matures, losses, awake, [1, 1, 1])
    rng = np.random.default_rng(1)
    for session in range(len(decided)):
        immature = matures > decided[session]
        altered = losses.copy()
        altered[immature] = rng.uniform(0, 1, size=(int(immature.sum()), losses.shape[1]))
        again, _ = arbiters.mature_mixture(decided, matures, altered, awake, [1, 1, 1])
        np.testing.assert_array_equal(again[session], mixtures[session])
    # La prueba no es vacía: invertir las pérdidas ya maduras sí cambia la decisión.
    altered = losses.copy()
    known = matures <= decided[10]
    altered[known] = 1 - altered[known]
    again, _ = arbiters.mature_mixture(decided, matures, altered, awake, [1, 1, 1])
    assert known.any() and not np.allclose(again[10], mixtures[10], equal_nan=True)


def test_the_mixtures_do_not_depend_on_the_order_of_the_sessions():
    decided, matures, losses, awake = delayed(seed=6)
    mixtures, state = arbiters.mature_mixture(decided, matures, losses, awake, [2, 1, 1])
    order = np.random.default_rng(2).permutation(len(decided))
    shuffled, other = arbiters.mature_mixture(
        decided[order], matures[order], losses[order], awake[order], [2, 1, 1]
    )
    np.testing.assert_array_equal(shuffled, mixtures[order])
    np.testing.assert_array_equal(other.regret, state.regret)
    # Sin opciones despiertas la sesión no tiene mezcla.
    awake[5] = False
    mixtures, _ = arbiters.mature_mixture(decided, matures, losses, awake, [2, 1, 1])
    assert np.isnan(mixtures[5]).all() and not np.isnan(mixtures[6]).any()


def test_a_session_without_lag_is_applied_before_the_next_decision():
    decided = np.array([0, 10, 20], dtype=np.int64)
    matures = np.array([10, 20, 30], dtype=np.int64)
    losses = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    awake = np.ones((3, 2), dtype=bool)
    mixtures, state = arbiters.mature_mixture(decided, matures, losses, awake, [1, 1])
    # La sesión que madura justo al decidir la siguiente ya cuenta, como en el resto de etapas.
    np.testing.assert_array_equal(mixtures[0], [0.5, 0.5])
    assert mixtures[1][0] > 0.5 and mixtures[2][0] > mixtures[1][0]
    assert state.rounds == 2


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d, m, x, a: (d, d, x, a), "maduran después"),
        (lambda d, m, x, a: (d * 0, m, x, a), "distintos"),
        (lambda d, m, x, a: (d.astype(float), m, x, a), "instantes enteros"),
        (lambda d, m, x, a: (d, m, x[:, :2], a), "una pérdida"),
        (lambda d, m, x, a: (d, m, x, a.astype(int)), "disponibilidad"),
        (lambda d, m, x, a: (d[:0], m[:0], x[:0], a[:0]), "instantes enteros"),
    ],
)
def test_invalid_sessions_are_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        arbiters.mature_mixture(*change(*delayed()), [1, 1, 1])


def version_table(seed=3, sessions=60, assets=12, markets=("US",), thin=None):
    """Filas de calibración y S con varios mercados. `thin` deja una sesión con dos activos."""
    rng = np.random.default_rng(seed)
    columns = dict(sample_id=[], market=[], prediction_at=[], target=[], prediction=[])
    for offset, market in enumerate(markets):
        for day in range(sessions):
            at = 1_546_380_000_000_000 + day * DAY_US + offset * 3_600_000_000
            count = 2 if day == thin else assets
            for asset in range(count):
                columns["sample_id"].append(f"{market}/A{asset}/{day}")
                columns["market"].append(market)
                columns["prediction_at"].append(at)
                columns["target"].append(float(rng.standard_normal()))
                columns["prediction"].append(0.0)
    columns["prediction_at"] = pa.array(columns["prediction_at"], type=pa.timestamp("us", tz="UTC"))
    return pa.table(columns)


def version_inputs(table, seed=5, signs=(1.0, 0.0, 0.0)):
    """Panel, predicción de cada versión y madurez a dos días de cada etiqueta."""
    panel = ForecastPanel.from_arrow(table)
    rng = np.random.default_rng(seed)
    columns = [sign * panel.target + rng.standard_normal(panel.rows) for sign in signs]
    matures = np.asarray(panel.prediction_at, dtype=np.int64) + 2 * DAY_US
    return panel, np.column_stack(columns), matures


def with_targets(table, change):
    """Tabla con los objetivos que devuelve `change` a partir de los de la tabla.

    Se trabaja en el orden de la tabla, que no es el canónico del panel: `ForecastPanel`
    ordena las identidades como texto.
    """
    at = table["prediction_at"].cast(pa.int64()).to_numpy()
    market = np.asarray(table["market"].to_pylist())
    target = change(table["target"].to_numpy().copy(), at, market)
    return ForecastPanel.from_arrow(
        table.set_column(table.schema.get_field_index("target"), "target", pa.array(target))
    )


def arbiter(panel, predictions, matures, **options):
    settings = dict(margin=0.0, window=10, min_assets=3)
    settings.update(options)
    return arbiters.online_mixture(panel, predictions, matures, **settings)


def test_each_session_is_standardized_and_a_constant_one_is_zero():
    panel, predictions, _ = version_inputs(version_table())
    tipified = arbiters.standardized(panel, predictions[:, 0])
    for start, count in zip(panel.session_starts, panel.session_samples, strict=True):
        values = tipified[start : start + count]
        assert abs(values.mean()) < 1e-12 and values.std() == pytest.approx(1.0)
    assert not arbiters.standardized(panel, np.ones(panel.rows)).any()


def test_the_loss_is_half_of_one_minus_the_session_ic():
    panel, predictions, _ = version_inputs(version_table(thin=4))
    losses, scored = arbiters.session_losses(panel, predictions, min_assets=3)
    for option in range(predictions.shape[1]):
        values, _ = sd.session_correlations(panel, predictions[:, option], min_assets=3)
        np.testing.assert_allclose(losses[:, option], (1 - values) / 2)
    # La sesión con dos activos no puntúa y las demás sí.
    assert not scored[4] and scored[np.arange(panel.sessions) != 4].all()


def test_the_arbiter_moves_to_the_version_that_ranks_the_assets():
    panel, predictions, matures = version_inputs(version_table(sessions=120))
    result = arbiter(panel, predictions, matures)
    # Hasta que madura la primera sesión no hay fiabilidad y el árbitro se abstiene.
    assert result["abstained"][:2].all() and np.isnan(result["reliability"][:2]).all()
    late = result["mixtures"][60:]
    assert (late[:, 0] > 0.9).all()
    assert 0 < result["coverage"] < 1
    tipified = arbiters.standardized(panel, predictions[:, 0])
    rows = slice(panel.session_starts[100], panel.session_starts[100] + panel.session_samples[100])
    np.testing.assert_allclose(
        result["prediction"][rows],
        np.column_stack([arbiters.standardized(panel, c) for c in predictions.T])[rows]
        @ result["mixtures"][100],
    )
    assert np.corrcoef(result["prediction"][rows], tipified[rows])[0, 1] > 0.99


def test_the_arbiter_abstains_when_no_version_is_reliable_enough():
    panel, predictions, matures = version_inputs(version_table(), signs=(-1.0, -1.0))
    result = arbiter(panel, predictions, matures)
    assert result["abstained"].all() and result["coverage"] == 0.0
    assert not result["prediction"].any()
    # El margen se fija antes: con −1 solo se abstiene mientras no hay resultados maduros.
    lenient = arbiter(panel, predictions, matures, margin=-1.0)
    assert lenient["abstained"].sum() == 2


def test_without_a_margin_the_mixture_never_abstains():
    panel, predictions, matures = version_inputs(version_table(), signs=(-1.0, -1.0))
    result = arbiter(panel, predictions, matures, margin=None)
    assert not result["abstained"].any() and result["coverage"] == 1.0
    # Antes de la primera sesión madura la mezcla es la del prior, uniforme.
    np.testing.assert_allclose(result["mixtures"][0], [0.5, 0.5])
    assert result["prediction"].any()


@pytest.mark.parametrize("seed", [None, 7])
def test_no_future_target_reaches_a_decision(seed):
    table = version_table(sessions=40)
    panel, predictions, matures = version_inputs(table)
    result = arbiter(panel, predictions, matures, seed=seed)
    rng = np.random.default_rng(3)
    for session in (5, 17, 33):
        decided = panel.session_time[session]

        def redraw(target, at, market, decided=decided):
            future = at + 2 * DAY_US > decided
            target[future] = rng.standard_normal(int(future.sum()))
            return target

        altered = with_targets(table, redraw)
        future = matures > decided
        assert future.any() and (~future).any()
        np.testing.assert_array_equal(altered.target[~future], panel.target[~future])
        assert not np.array_equal(altered.target[future], panel.target[future])
        again = arbiter(altered, predictions, matures, seed=seed)
        for key in ("mixtures", "reliability"):
            np.testing.assert_array_equal(again[key][session], result[key][session])
        assert again["abstained"][session] == result["abstained"][session]
        start, count = panel.session_starts[session], panel.session_samples[session]
        rows = slice(start, start + count)
        np.testing.assert_array_equal(again["prediction"][rows], result["prediction"][rows])


def test_the_shuffled_control_loses_the_order_but_not_the_known_outcomes():
    panel, predictions, matures = version_inputs(version_table(sessions=50))
    ordered = arbiter(panel, predictions, matures)
    shuffled = arbiter(panel, predictions, matures, seed=7)
    again = arbiter(panel, predictions, matures, seed=7)
    np.testing.assert_array_equal(shuffled["mixtures"], again["mixtures"])
    assert not np.allclose(shuffled["mixtures"][10:], ordered["mixtures"][10:])
    # Las dos versiones conocen los mismos resultados: la abstención inicial coincide.
    np.testing.assert_array_equal(shuffled["abstained"][:2], ordered["abstained"][:2])


def test_the_arbiter_follows_a_change_of_the_better_version_and_its_control_does_not():
    table = version_table(sessions=90)
    panel = ForecastPanel.from_arrow(table)
    early = panel.prediction_at < panel.session_time[45]
    noise = np.random.default_rng(2).standard_normal((panel.rows, 2))
    predictions = np.column_stack(
        [
            np.where(early, panel.target, 0.0) + 0.7 * noise[:, 0],
            np.where(early, 0.0, panel.target) + 0.7 * noise[:, 1],
        ]
    )
    matures = np.asarray(panel.prediction_at, dtype=np.int64) + 2 * DAY_US
    ordered = arbiter(panel, predictions, matures)
    shuffled = arbiter(panel, predictions, matures, seed=7)
    assert ordered["mixtures"][40, 0] > 0.9 and ordered["mixtures"][85, 1] > 0.9
    # Sin el orden de los resultados el control no sabe que la versión buena ha cambiado.
    assert shuffled["mixtures"][85, 1] < ordered["mixtures"][85, 1] - 0.2


def test_each_market_has_its_own_state():
    table = version_table(markets=("US", "CN"), sessions=30)
    panel, predictions, matures = version_inputs(table)
    result = arbiter(panel, predictions, matures)
    rng = np.random.default_rng(1)

    def redraw(target, at, market):
        target[market == "CN"] = rng.standard_normal(int((market == "CN").sum()))
        return target

    altered = arbiter(with_targets(table, redraw), predictions, matures)
    us = np.asarray(panel.markets)[panel.session_market] == "US"
    np.testing.assert_array_equal(altered["mixtures"][us], result["mixtures"][us])
    assert not np.array_equal(altered["mixtures"][~us], result["mixtures"][~us])


def test_the_uniform_control_is_the_mean_of_the_standardized_options():
    panel, predictions, _ = version_inputs(version_table())
    expected = np.mean([arbiters.standardized(panel, c) for c in predictions.T], axis=0)
    np.testing.assert_allclose(arbiters.uniform_mixture(panel, predictions), expected)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda p, m: (p[:, :1], m, {}), "dos opciones"),
        (lambda p, m: (np.where(p > 2, np.nan, p), m, {}), "finita"),
        (lambda p, m: (p, m - 2 * DAY_US, {}), "madura después"),
        (lambda p, m: (p, m.astype(float), {}), "madura después"),
        (lambda p, m: (p, m, dict(window=0)), "ventana"),
        (lambda p, m: (p, m, dict(margin=math.nan)), "margen"),
    ],
)
def test_invalid_arbiter_inputs_are_rejected(change, message):
    panel, predictions, matures = version_inputs(version_table())
    predictions, matures, options = change(predictions, matures)
    with pytest.raises(ValueError, match=message):
        arbiter(panel, predictions, matures, **options)
