"""Árbitros en línea con resultados maduros (#789).

El núcleo `adanormalhedge_v1` es AdaNormalHedge (Luo y Schapire, 2015,
https://arxiv.org/abs/1502.05934): el algoritmo 1 del artículo con la extensión de su
sección 4 para opciones que duermen. No tiene parámetros que ajustar. Cada opción i guarda
su arrepentimiento acumulado R_i y la magnitud acumulada C_i de sus arrepentimientos, y la
mezcla de una ronda es

    p_i ∝ q_i I_i w(R_i, C_i),   w(R, C) = (Φ(R + 1, C + 1) − Φ(R − 1, C + 1)) / 2,

con Φ(R, C) = exp([R]₊² / 3C), Φ(0, 0) = 1, el prior q y la disponibilidad I_i ∈ {0, 1}.
Conocidas las pérdidas ℓ_i ∈ [0, 1] de las opciones despiertas, la ronda suma
r_i = I_i (p·ℓ − ℓ_i) a R_i y |r_i| a C_i.

Si todos los pesos de las opciones despiertas son cero, el artículo deja la mezcla libre
(nota 1 del algoritmo 1) y aquí se usa el prior restringido a esas opciones. La igualdad
Σ_i q_i w_i r_i = 0 de la prueba de su lema 2 se mantiene, porque los w_i de las opciones
despiertas son cero y las dormidas no suman arrepentimiento. Una ronda sin opciones
despiertas no mezcla nada ni cambia el estado.

El lema 2 da Σ_i q_i Φ(R_i, C_i) ≤ B = 1 + (3/2) Σ_i q_i (1 + ln(1 + C_i)) tras cada ronda,
y el teorema 1 acota el arrepentimiento frente a cualquier mezcla fija u por
√(3 (u·C) (RE(u‖q) + ln B + ln(1 + ln N))). `AdaNormalHedgeTV` es la reducción de la
sección 5.1 que extiende la cota a cualquier intervalo, de modo que compite también con una
mejor opción que cambia en el tiempo. Las pruebas comprueban las cotas sobre secuencias
construidas, sin ajustar nada.

Las pérdidas financieras maduran después de decidir. `mature_mixture` aplica cada sesión
cuando madura, en orden de madurez, con la mezcla vigente en ese momento, que es la que
conserva la igualdad anterior. Las cotas valen así para la secuencia de sesiones maduras y no
exactamente para las mezclas emitidas antes de conocer sus pérdidas.

`online_mixture` reparte la confianza entre las opciones de cada sesión con la pérdida
(1 − IC)/2 de las sesiones maduras y mezcla sus predicciones tipificadas. Dos identidades la
usan. `version_arbiter_v1` mezcla versiones de un mismo ajuste y se abstiene si ninguna
tiene fiabilidad madura por encima de un margen fijado antes. Sus controles son la media
uniforme (`uniform_mixture`), el mismo árbitro con los resultados desordenados en el tiempo
y la época fija elegida en validación, que es la emisión ya existente del ajuste.
`model_combination_v1` mezcla todos los modelos comparados, sin abstención y con la media
simple como control obligatorio por el enigma de la combinación de previsiones. La garantía
es sobre la pérdida lineal de la mezcla, no sobre el IC medio de la predicción mezclada.
"""

import math
from dataclasses import dataclass

import numpy as np

from . import scale_diagnostics as sd

ADANORMALHEDGE = "adanormalhedge_v1"
VERSION_ARBITER = "version_arbiter_v1"
MODEL_COMBINATION = "model_combination_v1"
DECLARATION_KIND = "online_arbiter_variants"
# La combinación de modelos mezcla unos ochenta modelos de la criba.
MAX_OPTIONS = 128
MAX_WINDOW = 252


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def potential(regret, magnitude):
    """Φ(R, C) de cada opción, con Φ(0, 0) = 1."""
    regret = np.asarray(regret, dtype=np.float64)
    magnitude = np.asarray(magnitude, dtype=np.float64)
    positive = np.maximum(regret, 0.0)
    exponent = np.divide(
        positive**2, 3.0 * magnitude, out=np.zeros_like(positive), where=positive > 0
    )
    return np.exp(exponent)


def weights(regret, magnitude):
    """w(R, C) de cada opción.

    Se escribe como exp(b) · expm1(a − b) / 2, con a y b los exponentes de Φ(R + 1, C + 1) y
    Φ(R − 1, C + 1), para no restar dos exponenciales casi iguales cuando C es grande. Vale
    cero si R ≤ −1.
    """
    regret = np.asarray(regret, dtype=np.float64)
    scale = 3.0 * (np.asarray(magnitude, dtype=np.float64) + 1.0)
    upper = np.maximum(regret + 1.0, 0.0) ** 2 / scale
    lower = np.maximum(regret - 1.0, 0.0) ** 2 / scale
    return 0.5 * np.exp(lower) * np.expm1(upper - lower)


def _instant(mixture, losses, awake):
    """Arrepentimiento instantáneo de cada opción y pérdida de la mezcla, p·ℓ.

    Solo se leen las pérdidas de las opciones despiertas, que deben estar en [0, 1].
    """
    awake = np.asarray(awake)
    losses = np.asarray(losses, dtype=np.float64)
    _require(losses.shape == mixture.shape, "Hay una pérdida por opción")
    seen = losses[awake]
    _require(
        np.isfinite(seen).all() and (seen >= 0).all() and (seen <= 1).all(),
        "Las pérdidas de las opciones despiertas están en [0, 1]",
    )
    seen = np.where(awake, losses, 0.0)
    mixed = float(mixture @ seen)
    return np.where(awake, mixed - seen, 0.0), mixed


def _awake(awake, options):
    mask = np.asarray(awake)
    _require(
        mask.shape == (options,) and mask.dtype == np.bool_,
        "La disponibilidad es una máscara booleana con una entrada por opción",
    )
    return mask


@dataclass
class AdaNormalHedge:
    """Estado de AdaNormalHedge sobre un número fijo de opciones."""

    prior: np.ndarray
    regret: np.ndarray
    magnitude: np.ndarray
    rounds: int = 0

    @classmethod
    def start(cls, prior):
        """Estado inicial con R = C = 0. El prior se normaliza y debe ser positivo."""
        prior = np.asarray(prior, dtype=np.float64)
        _require(
            prior.ndim == 1
            and 1 <= prior.size <= MAX_OPTIONS
            and np.isfinite(prior).all()
            and (prior > 0).all(),
            f"El prior necesita de 1 a {MAX_OPTIONS} pesos positivos y finitos",
        )
        zeros = np.zeros(prior.size)
        return cls(prior / prior.sum(), zeros, zeros.copy())

    @property
    def options(self):
        return self.prior.size

    def distribution(self, awake):
        """Mezcla de una ronda entre las opciones despiertas, o None si no hay ninguna."""
        awake = _awake(awake, self.options)
        if not awake.any():
            return None
        mass = np.where(awake, self.prior * weights(self.regret, self.magnitude), 0.0)
        total = mass.sum()
        if total == 0.0:
            # Todas las despiertas tienen R ≤ −1: el artículo deja la mezcla libre.
            mass = np.where(awake, self.prior, 0.0)
            total = mass.sum()
        return mass / total

    def update(self, losses, awake):
        """Aplicar una ronda y devolver la pérdida de la mezcla, p·ℓ, o None sin opciones.

        Solo se leen las pérdidas de las opciones despiertas, que deben estar en [0, 1].
        """
        mixture = self.distribution(awake)
        if mixture is None:
            return None
        instant, mixed = _instant(mixture, losses, awake)
        self.regret = self.regret + instant
        self.magnitude = self.magnitude + np.abs(instant)
        self.rounds += 1
        return mixed

    def potential_bound(self):
        """Σ_i q_i Φ(R_i, C_i) y la cota B de su lema 2 con el estado actual."""
        total = math.fsum(self.prior * potential(self.regret, self.magnitude))
        bound = 1.0 + 1.5 * math.fsum(self.prior * (1.0 + np.log1p(self.magnitude)))
        return total, bound


@dataclass
class AdaNormalHedgeTV:
    """AdaNormalHedge.TV, la reducción de la sección 5.1 del artículo para competir con una
    mejor opción que cambia en el tiempo.

    Cada ronda τ despierta una copia (τ, i) de cada opción, con prior q_i / τ², que desde
    entonces sufre las pérdidas de la opción i. La mezcla suma los pesos de las copias,

        p_i ∝ q_i I_i Σ_τ τ⁻² w(R_{[τ,t−1],i}, C_{[τ,t−1],i}),

    y el arrepentimiento de la copia (τ, i) es el de la opción i en el intervalo que empieza
    en τ. Así la cota del teorema 3 vale para cualquier intervalo, y el horizonte completo
    es el caso de la mejor opción fija. Las copias que despiertan en la ronda siguiente
    tienen R = C = 0 y peso positivo, de modo que la mezcla nunca queda libre. Cuesta
    O(N t) por ronda.
    """

    prior: np.ndarray
    # Fila τ − 1: las copias que despertaron en la ronda τ.
    regret: np.ndarray
    magnitude: np.ndarray
    rounds: int = 0

    @classmethod
    def start(cls, prior):
        base = AdaNormalHedge.start(prior)
        empty = np.zeros((0, base.options))
        return cls(base.prior, empty, empty.copy())

    @property
    def options(self):
        return self.prior.size

    def _copies(self):
        """Estado con las copias que despiertan en la ronda siguiente, todavía a cero."""
        zero = np.zeros((1, self.options))
        return np.vstack([self.regret, zero]), np.vstack([self.magnitude, zero])

    def _start_weights(self):
        start = np.arange(1, self.rounds + 2, dtype=np.float64)
        return 1.0 / start[:, None] ** 2

    def distribution(self, awake):
        """Mezcla de una ronda entre las opciones despiertas, o None si no hay ninguna."""
        awake = _awake(awake, self.options)
        if not awake.any():
            return None
        regret, magnitude = self._copies()
        mass = (self._start_weights() * weights(regret, magnitude)).sum(axis=0)
        mass = np.where(awake, self.prior * mass, 0.0)
        return mass / mass.sum()

    def update(self, losses, awake):
        """Aplicar una ronda y devolver la pérdida de la mezcla, p·ℓ, o None sin opciones."""
        mixture = self.distribution(awake)
        if mixture is None:
            return None
        instant, mixed = _instant(mixture, losses, awake)
        regret, magnitude = self._copies()
        self.regret = regret + instant
        self.magnitude = magnitude + np.abs(instant)
        self.rounds += 1
        return mixed

    def potential_bound(self):
        """Lema 2 con el prior de las copias normalizado sobre las rondas jugadas y la
        siguiente, que es un prior fijo válido para ese horizonte."""
        regret, magnitude = self._copies()
        prior = self._start_weights() * self.prior
        prior = prior / prior.sum()
        total = math.fsum((prior * potential(regret, magnitude)).ravel())
        bound = 1.0 + 1.5 * math.fsum((prior * (1.0 + np.log1p(magnitude))).ravel())
        return total, bound


def mature_mixture(decided_at, matures_at, losses, awake, prior, scored=None, hedge=None):
    """Mezcla de cada sesión con AdaNormalHedge y solo las pérdidas maduras al decidirla.

    `decided_at[s]` es el instante de decisión de la sesión s y `matures_at[s]` el de su
    última etiqueta, ambos en microsegundos UTC y con `matures_at > decided_at`.
    `losses[s, i]` es la pérdida de la opción i en esa sesión, `awake[s, i]` si estaba
    disponible al decidir y `scored[s, i]` si su pérdida cuenta al madurar, que por defecto
    es `awake`. Antes de decidir una sesión se aplican, en orden de madurez y después de
    decisión, las sesiones con `matures_at <= decided_at`, la convención del resto de etapas.
    Los instantes de decisión no se repiten, así que el resultado no depende del orden de
    las filas de entrada. `hedge` es `AdaNormalHedge`, por defecto, o `AdaNormalHedgeTV`.

    Devuelve las mezclas, con NaN en las sesiones sin opciones, y el estado final, que solo
    incluye las sesiones maduras antes de la última decisión.
    """
    decided_at = np.asarray(decided_at)
    matures_at = np.asarray(matures_at)
    losses = np.asarray(losses, dtype=np.float64)
    awake = np.asarray(awake)
    scored = awake if scored is None else np.asarray(scored)
    sessions = decided_at.shape[0] if decided_at.ndim == 1 else -1
    state = (hedge or AdaNormalHedge).start(prior)
    _require(
        sessions >= 1
        and decided_at.dtype.kind == "i"
        and matures_at.shape == decided_at.shape
        and matures_at.dtype.kind == "i"
        and losses.shape == (sessions, state.options)
        and awake.shape == losses.shape == scored.shape
        and awake.dtype == np.bool_ == scored.dtype,
        "Cada sesión necesita instantes enteros, una pérdida y una disponibilidad por opción",
    )
    _require(
        np.unique(decided_at).size == sessions and (matures_at > decided_at).all(),
        "Las sesiones tienen instantes de decisión distintos y maduran después de decidir",
    )
    _require(not (scored & ~awake).any(), "Solo puntúa una opción disponible al decidir")
    order = np.argsort(decided_at)
    mature = np.lexsort((decided_at, matures_at))
    mixtures = np.full(losses.shape, np.nan)
    applied = 0
    for session in order:
        while applied < sessions and matures_at[mature[applied]] <= decided_at[session]:
            known = mature[applied]
            state.update(losses[known], scored[known])
            applied += 1
        mixture = state.distribution(awake[session])
        if mixture is not None:
            mixtures[session] = mixture
    return mixtures, state


def standardized(panel, prediction):
    """Predicción tipificada dentro de cada sesión. Una sesión constante queda a cero."""
    prediction = np.asarray(prediction, dtype=np.float64)
    _require(prediction.shape == (panel.rows,), "Hay una predicción por fila del panel")
    result = np.zeros(panel.rows)
    for start, count in zip(panel.session_starts, panel.session_samples, strict=True):
        values = prediction[start : start + count]
        if np.ptp(values) > 0:
            result[start : start + count] = (values - values.mean()) / values.std()
    return result


def session_losses(panel, predictions, *, min_assets):
    """Pérdida (1 − IC)/2 de cada opción en cada sesión y las sesiones que puntúan.

    Puntúa una sesión con al menos `min_assets` filas y objetivo no constante. Una opción
    constante en una sesión que puntúa no ordena los activos: su IC vale 0 y pierde 1/2.
    """
    starts, counts = panel.session_starts, panel.session_samples
    spread = np.array(
        [np.ptp(panel.target[s : s + n]) for s, n in zip(starts, counts, strict=True)]
    )
    scored = (counts >= min_assets) & (spread > 0)
    values = np.column_stack(
        [
            sd.session_correlations(panel, predictions[:, option], min_assets=min_assets)[0]
            for option in range(predictions.shape[1])
        ]
    )
    return (1.0 - values) / 2.0, scored


def _reliability(ic, scored, decided_at, matures_at, window):
    """IC medio de las últimas `window` sesiones maduras y puntuadas antes de cada decisión."""
    mature = np.lexsort((decided_at, matures_at))
    known = np.searchsorted(matures_at[mature], decided_at, side="right")
    reliability = np.full(ic.shape, np.nan)
    for session, count in enumerate(known):
        seen = mature[:count][scored[mature[:count]]][-window:]
        if seen.size:
            reliability[session] = ic[seen].mean(axis=0)
    return reliability


def _ordered(losses, scored, ic, decided_at, matures_at, window):
    """Mezclas y fiabilidades de un mercado en el orden en que maduran sus sesiones."""
    available = np.ones(losses.shape, dtype=bool)
    mixtures, _ = mature_mixture(
        decided_at,
        matures_at,
        losses,
        available,
        np.ones(losses.shape[1]),
        scored=available & scored[:, None],
        hedge=AdaNormalHedgeTV,
    )
    return mixtures, _reliability(ic, scored, decided_at, matures_at, window)


def _shuffled(losses, scored, ic, decided_at, matures_at, window, keys):
    """Control con los resultados desordenados en el tiempo, sin ninguno sin madurar.

    Cada sesión tiene una clave aleatoria fija. En cada decisión el estado se reconstruye
    desde cero con las sesiones ya maduras ordenadas por su clave, y la fiabilidad usa las
    últimas `window` de ese orden. Se pierde el orden temporal de los resultados y se
    conserva cuáles se conocen en cada momento.
    """
    sessions, options = losses.shape
    available = np.ones(options, dtype=bool)
    mixtures = np.full((sessions, options), np.nan)
    reliability = np.full((sessions, options), np.nan)
    for session in range(sessions):
        known = np.flatnonzero(matures_at <= decided_at[session])
        known = known[np.argsort(keys[known])]
        state = AdaNormalHedgeTV.start(np.ones(options))
        for item in known:
            state.update(losses[item], available & scored[item])
        mixtures[session] = state.distribution(available)
        seen = known[scored[known]][-window:]
        if seen.size:
            reliability[session] = ic[seen].mean(axis=0)
    return mixtures, reliability


def online_mixture(panel, predictions, matures_at, *, window, min_assets, margin=None, seed=None):
    """Mezcla en línea de varias opciones con sus resultados maduros.

    `predictions[:, i]` es la predicción de la opción i en las filas del panel, que reúne la
    calibración y S en orden canónico, y `matures_at` la madurez de la etiqueta de cada fila.
    Cada mercado tiene su propio estado de AdaNormalHedge.TV con prior uniforme. La pérdida de
    una opción en una sesión es (1 − IC)/2 y su fiabilidad, el IC medio de sus últimas
    `window` sesiones maduras. Con `margin`, la sesión se abstiene, con predicción cero, si
    ninguna opción supera ese margen, y también mientras no haya ninguna sesión madura. Sin
    `margin` no hay abstención. Las sesiones que no se abstienen emiten la mezcla de las
    predicciones tipificadas por sesión. Con `seed` se calcula el control con los resultados
    desordenados en el tiempo (`_shuffled`), que solo usa resultados maduros.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    matures_at = np.asarray(matures_at)
    _require(
        predictions.ndim == 2
        and predictions.shape[0] == panel.rows
        and 2 <= predictions.shape[1] <= MAX_OPTIONS
        and np.isfinite(predictions).all(),
        "Cada fila necesita una predicción finita de cada opción, con al menos dos opciones",
    )
    _require(
        matures_at.shape == (panel.rows,)
        and matures_at.dtype.kind == "i"
        and (matures_at > panel.prediction_at).all(),
        "Cada etiqueta madura después de su decisión",
    )
    _require(
        type(window) is int
        and window >= 1
        and (margin is None or math.isfinite(margin))
        and type(min_assets) is int,
        "La fiabilidad necesita una ventana entera positiva y un margen finito",
    )
    options = predictions.shape[1]
    tipified = np.column_stack([standardized(panel, predictions[:, i]) for i in range(options)])
    losses, scored = session_losses(panel, predictions, min_assets=min_assets)
    ic = 1.0 - 2.0 * losses
    session_matures = np.maximum.reduceat(matures_at, panel.session_starts)
    mixtures = np.full((panel.sessions, options), np.nan)
    reliability = np.full((panel.sessions, options), np.nan)
    keys = None if seed is None else np.random.default_rng(seed).random(panel.sessions)
    for market in np.unique(panel.session_market):
        sessions = np.flatnonzero(panel.session_market == market)
        market_args = (
            losses[sessions],
            scored[sessions],
            ic[sessions],
            panel.session_time[sessions],
            session_matures[sessions],
            window,
        )
        if keys is None:
            mixtures[sessions], reliability[sessions] = _ordered(*market_args)
        else:
            mixtures[sessions], reliability[sessions] = _shuffled(*market_args, keys[sessions])
    if margin is None:
        abstained = np.zeros(panel.sessions, dtype=bool)
    else:
        abstained = ~np.any(reliability > margin, axis=1)
    prediction = np.zeros(panel.rows)
    for session in np.flatnonzero(~abstained):
        start, count = panel.session_starts[session], panel.session_samples[session]
        prediction[start : start + count] = tipified[start : start + count] @ mixtures[session]
    return dict(
        prediction=prediction,
        mixtures=mixtures,
        reliability=reliability,
        abstained=abstained,
        coverage=float(1.0 - abstained.mean()),
    )


def uniform_mixture(panel, predictions):
    """Control de media uniforme de las opciones tipificadas, sin abstención."""
    predictions = np.asarray(predictions, dtype=np.float64)
    columns = [standardized(panel, predictions[:, i]) for i in range(predictions.shape[1])]
    return np.mean(columns, axis=0)


def _declared_stage(document, identity):
    """Etapa `identity` de una declaración con AdaNormalHedge.TV, desactivada y sin test final."""
    core = document.get("core", {}) if isinstance(document, dict) else {}
    _require(
        isinstance(document, dict)
        and document.get("kind") == DECLARATION_KIND
        and document.get("default_enabled") is False
        and document.get("final_test_access") is False
        and core.get("identity") == ADANORMALHEDGE
        and core.get("form") == "adanormalhedge_tv",
        "La declaración de los árbitros debe usar AdaNormalHedge.TV, desactivada y sin test final",
    )
    stages = document.get("stages")
    stage = stages.get(identity) if isinstance(stages, dict) else None
    _require(
        identity in (VERSION_ARBITER, MODEL_COMBINATION) and isinstance(stage, dict),
        f"{identity} no está declarada",
    )
    return stage


def _version_controls(margin, controls, analysis):
    """Semilla del control desordenado y comparación con las medias de pesos en el tiempo."""
    shuffled = controls.get("shuffled_outcomes", {})
    seed = shuffled.get("seed") if isinstance(shuffled, dict) else None
    _require(
        type(margin) in (int, float)
        and math.isfinite(margin)
        and set(controls) == {"common_epoch", "uniform_mixture", "shuffled_outcomes"}
        and type(seed) is int
        and seed >= 0,
        "El árbitro de versiones declara un margen finito y sus tres controles",
    )
    _require(
        analysis.get("required") == ["uniform_mixture", "shuffled_outcomes"],
        "El árbitro de versiones exige superar la media uniforme y el control desordenado",
    )
    averaging = analysis.get("time_averaging")
    averaging = averaging if isinstance(averaging, dict) else {}
    compared = averaging.get("compared_with")
    _require(
        isinstance(averaging.get("arbiter_of"), str)
        and isinstance(compared, list)
        and compared
        and all(isinstance(name, str) and "/" in name for name in compared)
        and len(set(compared)) == len(compared)
        and averaging.get("family") == "reported_together_outside_every_decision",
        "El árbitro de versiones declara su comparación con las medias de pesos, fuera de "
        "toda decisión",
    )
    return seed, dict(arbiter_of=averaging["arbiter_of"], compared_with=compared)


def stage_options(document, identity):
    """Parámetros declarados de una etapa, comprobados antes de abrir ningún dato.

    Devuelve la ventana de fiabilidad, el margen de abstención, los activos mínimos por
    sesión, la semilla del control desordenado, la métrica de su análisis y la comparación
    con las medias de pesos en el tiempo. `version_arbiter_v1` necesita un margen finito, ese
    control y esa comparación. `model_combination_v1` no se abstiene y solo tiene la media
    simple.
    """
    stage = _declared_stage(document, identity)
    window, margin = stage.get("reliability_window_sessions"), stage.get("abstention_margin")
    min_assets, controls = stage.get("min_assets"), stage.get("controls")
    _require(isinstance(controls, dict), f"{identity} declara sus controles")
    analysis = stage.get("analysis")
    # La pérdida de los árbitros es (1 − IC)/2, así que su análisis se mide con el IC.
    _require(
        isinstance(analysis, dict) and analysis.get("metric") == "session_ic",
        f"{identity} se analiza con el IC por sesión",
    )
    _require(
        type(window) is int
        and 1 <= window <= MAX_WINDOW
        and type(min_assets) is int
        and min_assets >= 3,
        f"{identity} necesita una ventana de 1 a {MAX_WINDOW} sesiones y al menos tres activos",
    )
    if identity == VERSION_ARBITER:
        seed, averaging = _version_controls(margin, controls, analysis)
    else:
        _require(
            margin is None and set(controls) == {"simple_mean"},
            "La combinación de modelos no se abstiene y su control es la media simple",
        )
        seed = averaging = None
    return dict(
        window=window,
        margin=margin,
        min_assets=min_assets,
        seed=seed,
        metric=analysis["metric"],
        time_averaging=averaging,
    )
