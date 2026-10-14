"""Varianza y correlación de cohorte de la regla kalman de B6, estimadas por momentos (PT3).

La regla kalman de `associative_memory` modela los resultados de una cohorte madura con
Σ = σ²[(1 − ϱ)I + ϱ11ᵀ]: σ² es la varianza de un residuo etiqueta menos núcleo y ϱ la
correlación común de los residuos de una misma sesión. Aquí se estiman los dos con los
residuos de un tramo anterior a todo lo que la corrección predice, con el estimador de
componentes de la varianza del ANOVA de un factor (efecto aleatorio de sesión, Baltagi,
2021). El mercado entra como estrato fijo, para que un sesgo medio distinto en EE. UU. y en
China no se confunda con correlación dentro de la sesión.

Con g sesiones en M mercados, N residuos, n_i residuos en la sesión i y N_m en el mercado m:

    SSB = Σ_i n_i (ȳ_i − ȳ_m(i))²          MSB = SSB / (g − M)
    SSW = Σ_i Σ_j (y_ij − ȳ_i)²            MSW = SSW / (N − g)
    n₀ = (N − Σ_m Σ_{i∈m} n_i² / N_m) / (g − M)

E[MSW] = σ_w² y E[MSB] = σ_w² + n₀ σ_b², así que σ_b² = max(0, (MSB − MSW) / n₀),
σ² = σ_b² + MSW y ϱ = min(tope, σ_b² / σ²). Con sesiones del mismo tamaño n₀ es ese tamaño.
El tope mantiene ϱ < 1, que la regla exige para invertir Σ.

La corrección escribe juntas las etiquetas de un mismo evento, es decir, las que maduran en el
mismo instante, y la regla trata cada una de esas cohortes con la ϱ de una sesión.
`cohort_composition` cuenta en qué medida una cohorte de escritura no coincide con una sesión:
cohortes que mezclan sesiones o mercados y sesiones repartidas entre varias cohortes.
"""

import numpy as np

METHOD = "one_way_anova_moments_session_effect_market_strata_v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def estimate_cohort_noise(residuals, markets, sessions, *, correlation_cap):
    """Estimar σ² y ϱ con los residuos de cada sesión de cada mercado.

    `residuals` son etiqueta menos predicción del núcleo, `markets` el mercado de cada fila y
    `sessions` el instante de decisión en microsegundos. Una sesión es un par de mercado e
    instante. Un mercado con una sola sesión no aporta grados de libertad entre sesiones, pero
    sí dentro de ella. Devuelve las dos varianzas, la correlación sin truncar y los recuentos.
    """
    values = np.asarray(residuals, dtype=np.float64)
    markets = np.asarray(markets)
    sessions = np.asarray(sessions)
    rows = len(values)
    _require(
        values.shape == (rows,)
        and markets.shape == sessions.shape == (rows,)
        and markets.dtype.kind == "U"
        and sessions.dtype.kind in "iu"
        and bool(np.isfinite(values).all()),
        "Los residuos necesitan valores finitos con su mercado y su sesión alineados",
    )
    _require(
        type(correlation_cap) is float and 0.0 < correlation_cap < 1.0,
        "El tope de la correlación de cohorte debe estar en (0, 1)",
    )
    # Índices de mercado y de sesión. Una sesión pertenece a un único mercado.
    market_names, market = np.unique(markets, return_inverse=True)
    pairs = np.stack([market.astype(np.int64), sessions.astype(np.int64)], axis=1)
    keys, session = np.unique(pairs, axis=0, return_inverse=True)
    session = session.reshape(-1)
    groups, strata = len(keys), len(market_names)
    _require(
        rows > groups > strata,
        "La estimación necesita más sesiones que mercados y alguna sesión con dos residuos",
    )
    sizes = np.bincount(session, minlength=groups).astype(np.float64)
    means = np.bincount(session, weights=values, minlength=groups) / sizes
    totals = np.bincount(market, minlength=strata).astype(np.float64)
    centers = np.bincount(market, weights=values, minlength=strata) / totals
    owner = keys[:, 0]
    between = float(np.sum(sizes * (means - centers[owner]) ** 2))
    within = float(np.sum((values - means[session]) ** 2))
    msb = between / (groups - strata)
    msw = within / (rows - groups)
    _require(msw > 0.0, "Los residuos de cada sesión no varían y σ² no puede estimarse")
    concentration = float(np.sum(np.bincount(owner, weights=sizes**2, minlength=strata) / totals))
    n0 = (rows - concentration) / (groups - strata)
    common = max(0.0, (msb - msw) / n0)
    variance = common + msw
    raw = (msb - msw) / (msb + (n0 - 1.0) * msw)
    return dict(
        method=METHOD,
        observation_noise=variance,
        cohort_correlation=min(correlation_cap, common / variance),
        unclipped_correlation=raw,
        between_mean_square=msb,
        within_mean_square=msw,
        mean_session_size=n0,
        rows=rows,
        sessions=groups,
        markets=market_names.tolist(),
    )


def cohort_composition(maturity, markets, sessions):
    """Comparar las cohortes de escritura con las sesiones de las etiquetas.

    `maturity` es el instante en que madura cada etiqueta, que fija su cohorte, y `markets` y
    `sessions` su mercado y su instante de decisión, que fijan su sesión como en
    `estimate_cohort_noise`. Devuelve los recuentos de cohortes, sesiones y filas.
    """
    maturity = np.asarray(maturity)
    markets = np.asarray(markets)
    sessions = np.asarray(sessions)
    rows = len(maturity)
    _require(
        rows >= 1
        and maturity.shape == markets.shape == sessions.shape == (rows,)
        and maturity.dtype.kind in "iu"
        and sessions.dtype.kind in "iu"
        and markets.dtype.kind == "U",
        "Las etiquetas necesitan su madurez, su mercado y su sesión alineados",
    )
    _, market = np.unique(markets, return_inverse=True)
    pairs = np.stack([market.astype(np.int64), sessions.astype(np.int64)], axis=1)
    session_keys, session = np.unique(pairs, axis=0, return_inverse=True)
    cohort_keys, cohort = np.unique(maturity, return_inverse=True)
    session, cohort = session.reshape(-1), cohort.reshape(-1)

    def distinct(left, right, size):
        """Número de valores distintos de `right` para cada valor de `left`."""
        found = np.unique(np.stack([left, right], axis=1), axis=0)
        return np.bincount(found[:, 0], minlength=size)

    sessions_per_cohort = distinct(cohort, session, len(cohort_keys))
    markets_per_cohort = distinct(cohort, market, len(cohort_keys))
    cohorts_per_session = distinct(session, cohort, len(session_keys))
    rows_per_cohort = np.bincount(cohort, minlength=len(cohort_keys))
    mixed_sessions, mixed_markets = sessions_per_cohort > 1, markets_per_cohort > 1
    return dict(
        rows=rows,
        cohorts=len(cohort_keys),
        sessions=len(session_keys),
        cohorts_mixing_sessions=int(mixed_sessions.sum()),
        rows_in_cohorts_mixing_sessions=int(rows_per_cohort[mixed_sessions].sum()),
        cohorts_mixing_markets=int(mixed_markets.sum()),
        rows_in_cohorts_mixing_markets=int(rows_per_cohort[mixed_markets].sum()),
        max_sessions_per_cohort=int(sessions_per_cohort.max()),
        sessions_split_across_cohorts=int((cohorts_per_session > 1).sum()),
        max_cohorts_per_session=int(cohorts_per_session.max()),
    )
