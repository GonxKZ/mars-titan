"""Eventos declarados de D2 dentro de la comparación walk-forward (#453 y #454).

Los eventos se fijaron con una cronología externa antes de mirar resultados
(`docs/research/memory-diagnostics.md`, D2): la quiebra de Lehman (15/09/2008), la caída de
marzo de 2020 (24/02/2020 a 23/03/2020) y el inicio del ciclo de subidas de tipos de 2022
(16/03/2022) en EE. UU., cada uno con ±60 sesiones, y las sesiones de junio a agosto de 2015 en
China. Un evento ocupa las sesiones evaluadas de su mercado cuyo día UTC cae entre su primer y
su último día, más las `sessions_before` anteriores y las `sessions_after` posteriores en el
orden de esas mismas sesiones. En los dos mercados la decisión cae en el mismo día UTC que su
sesión local. Si el tramo del evento no tiene ninguna sesión evaluada, el evento queda fuera y
el informe lo indica. Si la serie empieza o termina antes de completar el margen, se usan las
sesiones que haya y el informe da cuántas.

Para cada modelo da, por evento y con los eventos de cada vista unidos, el número de sesiones,
la media y el percentil declarado del MAE por sesión, después de promediar las semillas sesión
a sesión. El percentil usa la interpolación lineal de NumPy. Para cada contraste de dos modelos
de las familias declaradas da el cambio relativo de ese percentil, variante frente a base, que
es lo que pide la regla de PT1 (no empeorar más de un 1 %). Su intervalo sale de un bootstrap
circular por bloques dentro de cada evento, con la longitud, las réplicas, la confianza y la
semilla de la comparación, y remuestrea las mismas sesiones para todos los modelos de la vista.
Con unas 120 sesiones por evento, el percentil 95 depende de muy pocas sesiones, y el
intervalo muestra cuánto.

La calibración en línea de PT2 usa estas mismas sesiones para dar su cobertura en cada evento.

No elige modelos ni cambia los contrastes principales. Trabaja sobre las mismas series por
sesión que ellos, sin leer filas.
"""

import re
from datetime import date

import numpy as np

from mars_titan.evaluation.paired_comparisons import circular_block_indices

FIELD = "declared_events"
STATUS = "declared_not_executed"
MARKETS = ("US", "CN")
LAST_DAY = date(2023, 12, 31)
DAY_MICROSECONDS = 86_400_000_000
_FIELDS = {"status", "declared_at", "source", "use", "quantile", "events"}
_EVENT_FIELDS = {"market", "first", "last", "sessions_before", "sessions_after"}
_NAME = re.compile(r"^[a-z0-9_]+$")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _date(value, label, last=LAST_DAY):
    try:
        day = date.fromisoformat(value)
    except (TypeError, ValueError):
        day = None
    _require(day is not None and day <= last, f"{label} debe ser un día anterior a {last.year + 1}")
    return day


def declaration(section):
    """Validar los eventos declarados antes de evaluar. No lee predicciones."""
    _require(
        isinstance(section, dict) and set(section) == _FIELDS and section["status"] == STATUS,
        "Los eventos declarados no cumplen su contrato",
    )
    _date(section["declared_at"], "La fecha de la declaración", date.max)
    _require(
        all(isinstance(section[key], str) and section[key].strip() for key in ("source", "use")),
        "Los eventos declarados necesitan su fuente y su uso",
    )
    quantile = section["quantile"]
    _require(type(quantile) is float and 0 < quantile < 1, "El percentil debe estar en (0, 1)")
    events = section["events"]
    _require(isinstance(events, dict) and events, "Faltan los eventos")
    for name, event in events.items():
        _require(
            isinstance(name, str)
            and _NAME.match(name)
            and isinstance(event, dict)
            and set(event) == _EVENT_FIELDS
            and event["market"] in MARKETS
            and all(
                type(event[key]) is int and 0 <= event[key] <= 252
                for key in ("sessions_before", "sessions_after")
            ),
            f"El evento {name} no está bien declarado",
        )
        first = _date(event["first"], f"El primer día de {name}")
        _require(first <= _date(event["last"], f"El último día de {name}"), f"{name} al revés")
    return section


def _day(text):
    return (date.fromisoformat(text) - date(1970, 1, 1)).days


def event_sessions(event, session_time, session_market, markets):
    """Posiciones de las sesiones del evento en la serie, en orden temporal.

    Devuelve las posiciones y cuántas sesiones quedan antes, dentro y después del tramo
    declarado, o None si el tramo no tiene ninguna sesión evaluada de su mercado.
    """
    if event["market"] not in markets:
        return None
    own = np.flatnonzero(np.asarray(session_market) == list(markets).index(event["market"]))
    own = own[np.argsort(np.asarray(session_time)[own], kind="stable")]
    days = np.asarray(session_time)[own] // DAY_MICROSECONDS
    inside = np.flatnonzero((days >= _day(event["first"])) & (days <= _day(event["last"])))
    if not len(inside):
        return None
    start = max(int(inside[0]) - event["sessions_before"], 0)
    end = min(int(inside[-1]) + event["sessions_after"] + 1, len(own))
    counts = dict(
        before=int(inside[0]) - start, inside=len(inside), after=end - 1 - int(inside[-1])
    )
    return own[start:end], counts


def _statistics(values, quantile):
    return dict(
        sessions=len(values),
        mean=float(values.mean()),
        quantile=float(np.quantile(values, quantile)),
    )


def _pairs(families, arms):
    """Contrastes de dos modelos (variante menos base) de las familias declaradas."""
    pairs = {}
    for family, contrasts in families.items():
        for name, coefficients in contrasts.items():
            if len(coefficients) != 2 or sorted(coefficients.values()) != [-1.0, 1.0]:
                continue
            base = next(arm for arm, value in coefficients.items() if value == -1.0)
            variant = next(arm for arm, value in coefficients.items() if value == 1.0)
            if base in arms and variant in arms:
                pairs.setdefault(family, {})[name] = (base, variant)
    return pairs


def _relative(base, variant, confidence):
    """Cambio relativo del percentil de la variante frente a la base, con intervalo."""
    estimate, draws = base
    if estimate == 0 or np.any(draws == 0):
        return dict(estimate=None, interval=None, reason="El percentil de la base es cero")
    ratios = variant[1] / draws - 1
    tail = (1 - confidence) / 2
    return dict(
        estimate=float(variant[0] / estimate - 1),
        interval=[float(value) for value in np.quantile(ratios, [tail, 1 - tail])],
    )


def _view(section, comparison, pairs, arms, series_of, view):
    first = series_of(arms[0], view)
    times, market_codes, markets = first[2], first[3], first[4]
    present = {markets[code] for code in np.unique(market_codes)}
    defined = first[1]
    _require(
        all(np.array_equal(series_of(arm, view)[1], defined) for arm in arms),
        "Los modelos no tienen las mismas sesiones definidas",
    )
    events, positions = {}, {}
    for name, event in section["events"].items():
        if event["market"] not in present:
            continue
        found = event_sessions(event, times, market_codes, markets)
        if found is None:
            events[name] = dict(market=event["market"], reason="Sin sesiones evaluadas en el tramo")
            continue
        index = found[0][defined[found[0]]]
        if not len(index):
            events[name] = dict(market=event["market"], reason="Sin sesiones definidas")
            continue
        events[name] = dict(market=event["market"], **found[1], defined=len(index))
        positions[name] = index
    if not positions:
        return dict(events=events, reason="Ningún evento tiene sesiones evaluadas en la vista")
    quantile = section["quantile"]
    rng = np.random.default_rng(comparison["seed"])
    replicates, length = comparison["replicates"], comparison["block_length"]
    # Las mismas réplicas para todos los modelos: cada evento se remuestrea dentro de sí.
    draws = {
        name: index[circular_block_indices(rng, replicates, len(index), length)]
        for name, index in positions.items()
    }
    pooled_index = np.concatenate(list(positions.values()))
    pooled_draws = np.concatenate(list(draws.values()), axis=1)
    summaries, tails = {}, {}
    for arm in arms:
        values = series_of(arm, view)[0]
        summaries[arm] = dict(
            events={
                name: _statistics(values[index], quantile) for name, index in positions.items()
            },
            pooled=_statistics(values[pooled_index], quantile),
        )
        tails[arm] = dict(
            pooled=(
                summaries[arm]["pooled"]["quantile"],
                np.quantile(values[pooled_draws], quantile, axis=1),
            ),
            **{
                name: (
                    summaries[arm]["events"][name]["quantile"],
                    np.quantile(values[draws[name]], quantile, axis=1),
                )
                for name in positions
            },
        )
    contrasts = {}
    for family, named in pairs.items():
        contrasts[family] = {
            name: dict(
                base=base,
                variant=variant,
                pooled=_relative(
                    tails[base]["pooled"], tails[variant]["pooled"], comparison["confidence"]
                ),
                events={
                    event: _relative(
                        tails[base][event], tails[variant][event], comparison["confidence"]
                    )
                    for event in positions
                },
            )
            for name, (base, variant) in named.items()
        }
    return dict(events=events, arms=summaries, quantile_relative_change=contrasts)


def report(section, comparison, families, arms, views, series_of):
    """Sección del informe con las métricas de cada vista en sus eventos declarados.

    `series_of(arm, view)` devuelve el MAE por sesión promediado entre semillas, su máscara,
    el instante, el mercado de cada sesión y los mercados de la serie.
    """
    arms = list(arms)
    pairs = _pairs(families, set(arms))
    return dict(
        declaration=section,
        statistic="seed_averaged_session_mae",
        quantile_method="numpy_linear",
        bootstrap=dict(
            kind="circular_blocks_within_each_event",
            block_length=comparison["block_length"],
            replicates=comparison["replicates"],
            confidence=comparison["confidence"],
            seed=comparison["seed"],
        ),
        views={view: _view(section, comparison, pairs, arms, series_of, view) for view in views},
    )
