"""Retención e interferencia de la memoria cuando reaparece un régimen observable.

Es un análisis secundario y descriptivo de la comparación walk-forward, declarado el 10 de
octubre de 2026 antes de cualquier resultado. No elige modelos ni cambia la conclusión
principal. Trabaja sobre las mismas series por sesión que los contrastes, sin volver a
leer filas ni a predecir.

Unidad y beneficio. Para cada par declarado (memoria, control) y cada mercado, el beneficio
de una sesión es b = MAE del control menos MAE de la memoria, después de promediar las
semillas sesión a sesión. Positivo favorece a la memoria.

Regímenes y recorrido. Cada sesión lleva la ruta del calendario de `regime_calendar`, que
solo usa precios hasta la decisión. Las memorias empiezan cada recorrido en su estado
inicial y no conservan nada entre ventanas, así que la historia de una sesión es la de su
ventana, desde que la memoria del par empieza a escribir hasta la sesión anterior. Cada par
declara ese comienzo. Los pesos rápidos de Titans escriben con las entradas desde el
principio del calentamiento (12 meses antes de la evaluación). El banco episódico y el
Transformer en línea solo reciben etiquetas maduras dentro del tramo medido, así que su
historia empieza con la evaluación: un régimen que solo apareció en el calentamiento no
está guardado en el banco y cuenta como nuevo. Un tramo es una racha de sesiones
clasificadas con la misma ruta. Las sesiones sin clasificar no cuentan, no cortan rachas y
quedan fuera.

Clases de cada sesión evaluada, con h la posición dentro de su tramo:

- `novel_entry`: h ≤ H y el régimen del tramo no había aparecido antes en el recorrido.
- `revisit_entry`: h ≤ H y el régimen ya apareció antes en el recorrido, con otro régimen
  entre medias. La ausencia es el número de sesiones clasificadas desde su última aparición.
- `continuing`: h > H.

Estadísticos, cada uno como diferencia de medias de b por clase:

- `predictive_effect`: media de todas las sesiones clasificadas.
- `retention`: revisitas menos regímenes nuevos. Si la memoria conserva algo útil de un
  régimen, debería recuperarse antes cuando vuelve que cuando aparece por primera vez.
- `interference`: revisitas tras una ausencia larga menos revisitas tras una corta. Un valor
  negativo indica que lo escrito entre medias desplaza lo útil.
- `plasticity`: entrada en un régimen nuevo menos tramos ya asentados.

Control de descarte. La entrada de un régimen nuevo suele caer al principio del recorrido y
las revisitas después, cuando la memoria ya tiene contenido. Para que eso no se confunda con
retención, el mismo cálculo se repite con un placebo: la ruta de la sesión que está L
sesiones antes en el calendario. Conserva rachas, ausencias y su posición en el año, pero
deja de coincidir con el estado del mercado.

Decisiones declaradas para cada par y mercado, con intervalos simultáneos:

- Retención: se mantiene si `retention` y `retention` menos su placebo quedan por encima de
  cero. En otro caso se descarta.
- Interferencia: se detecta si `interference` e `interference` menos su placebo quedan por
  debajo de cero. En otro caso no se detecta.

Si faltan sesiones de alguna clase necesaria, la decisión queda sin tomar. Con 12 meses de
calentamiento casi todos los regímenes aparecen antes de la evaluación, así que los pares de
pesos rápidos apenas tienen regímenes nuevos y su retención quedará normalmente sin decidir.
Para ellos la pregunta contrastable es la interferencia.

Los pares separan qué se mide: Titans-MAC en línea frente al congelado de igual capacidad
(escribir en inferencia), el congelado frente al desactivado (la lectura aprendida sin
escrituras), M1 a M3 frente a M0 (el contenido del banco con los mismos parámetros del
lector), M2 y M3 frente a M1 (qué se guarda con la misma capacidad) y el Transformer en línea
frente al congelado (aprender en línea sin memoria). No se inspecciona el contenido del banco
ni de los pesos rápidos, así que almacenamiento y recuperación se separan por diseño de los
pares y de las clases, no por diagnósticos internos.

Los intervalos usan el bootstrap circular por bloques de días UTC de la comparación, con su
longitud, réplicas, confianza y semilla. Todas las clases, pares y el placebo de un mercado
se remuestrean con los mismos bloques.
"""

from datetime import date

import numpy as np

from mars_titan.evaluation.paired_comparisons import circular_block_counts, family_intervals
from mars_titan.memory.regimes import REGIME_RULE, UNCLASSIFIED

STATUS = "secondary_descriptive"
USE = "description_only_not_for_model_selection_or_primary_conclusion"
REGIMES = f"{REGIME_RULE}_market_calendar"
HISTORY = "calendar_sessions_of_the_window_since_the_memory_of_the_pair_starts_writing"
# Comienzo de la historia de cada par: el calentamiento o el tramo medido.
STARTS = ("since_warmup_start", "since_measured_start")
METRIC = "session_mae_control_minus_memory_after_averaging_seeds"
PLACEBO = "route_of_the_calendar_session_lagged_by_the_declared_sessions"
MULTIPLICITY = "max_t_over_pairs_and_decision_statistics_within_each_market"
DECISION = (
    "retention_supported_if_it_and_its_placebo_difference_are_above_zero_"
    "interference_detected_if_it_and_its_placebo_difference_are_below_zero"
)
# Decisiones: estadístico, grupos que necesita y sentido que confirma la hipótesis.
DECISIONS = {
    "retention": (("novel_entry", "revisit_entry"), 1.0, ("supported", "discarded")),
    "interference": (("revisit_long", "revisit_short"), -1.0, ("detected", "not_detected")),
}
CLASSES = ("novel_entry", "revisit_entry", "continuing")
STATISTICS = {
    "predictive_effect": "mean_of_all_classified_sessions",
    "retention": "revisit_entry_minus_novel_entry",
    "interference": "long_absence_revisit_entry_minus_short_absence_revisit_entry",
    "plasticity": "novel_entry_minus_continuing",
}
DOES_NOT_MEASURE = [
    "contents_of_the_episodic_bank_or_of_the_fast_weights",
    "economic_causes_of_the_regimes",
    "retention_across_windows_of_states_refit_from_scratch",
]
FIELDS = {
    "status",
    "declared_at",
    "use",
    "hypothesis",
    "regimes",
    "history",
    "warmup_months",
    "metric",
    "entry_sessions",
    "long_absence_sessions",
    "absence_bins",
    "placebo_lag_sessions",
    "min_sessions",
    "pairs",
    "statistics",
    "placebo",
    "multiplicity",
    "decision",
    "does_not_measure",
}
_PAIR_FIELDS = {"memory", "control", "history", "separates"}
MAX_PAIRS = 16
MAX_SESSIONS = 1_000
# Grupos de sesiones de cada recorrido, en este orden en las matrices del bootstrap.
_GROUPS = ("classified", *CLASSES, "revisit_long", "revisit_short")
_NOT_IN_SCOPE = "Algún modelo del par no se compara en este ámbito"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _count(value, minimum, maximum):
    return type(value) is int and minimum <= value <= maximum


def declaration(section, arms):
    """Validar la declaración previa sin abrir ningún dato."""
    _require(
        isinstance(section, dict) and set(section) == FIELDS,
        "La declaración de retención e interferencia no tiene exactamente sus campos",
    )
    declared = section["declared_at"]
    try:
        valid_date = isinstance(declared, str) and date.fromisoformat(declared).isoformat()
    except ValueError:
        valid_date = False
    _require(
        section["status"] == STATUS
        and section["use"] == USE
        and valid_date == declared
        and isinstance(section["hypothesis"], str)
        and section["hypothesis"].strip()
        and section["does_not_measure"] == DOES_NOT_MEASURE,
        "La retención debe declararse como análisis secundario con su hipótesis y sus límites",
    )
    _require(
        section["regimes"] == REGIMES
        and section["history"] == HISTORY
        and section["metric"] == METRIC
        and section["statistics"] == STATISTICS
        and section["placebo"] == PLACEBO
        and section["multiplicity"] == MULTIPLICITY
        and section["decision"] == DECISION,
        "La retención usa el calendario de regímenes, el beneficio por sesión y el placebo",
    )
    entry, long_absence = section["entry_sessions"], section["long_absence_sessions"]
    bins = section["absence_bins"]
    _require(
        _count(section["warmup_months"], 0, 60)
        and _count(entry, 1, MAX_SESSIONS)
        and _count(long_absence, 2, MAX_SESSIONS)
        and _count(section["placebo_lag_sessions"], 1, MAX_SESSIONS)
        and _count(section["min_sessions"], 1, 1_000_000)
        and isinstance(bins, list)
        and bins
        and all(_count(edge, 2, MAX_SESSIONS) for edge in bins)
        and bins == sorted(set(bins))
        and long_absence in bins,
        "Los umbrales de la retención deben ser enteros acotados y fijados de antemano",
    )
    pairs = section["pairs"]
    _require(
        isinstance(pairs, dict) and 1 <= len(pairs) <= MAX_PAIRS,
        "La retención necesita entre 1 y 16 pares declarados",
    )
    seen = set()
    for name, pair in pairs.items():
        _require(
            isinstance(name, str)
            and name.isidentifier()
            and isinstance(pair, dict)
            and set(pair) == _PAIR_FIELDS
            and pair["history"] in STARTS
            and isinstance(pair["separates"], str)
            and pair["separates"].strip(),
            f"El par {name} necesita memoria, control, comienzo de su historia y lo que separa",
        )
        memory, control = pair["memory"], pair["control"]
        _require(
            memory in arms and control in arms and memory != control,
            f"El par {name} debe comparar dos modelos distintos de la comparación",
        )
        _require(
            all(arms[arm]["seeds"] for arm in (memory, control)),
            f"El par {name} no puede usar el control cero, que no tiene memoria ni semillas",
        )
        _require((memory, control) not in seen, f"El par {name} está repetido")
        seen.add((memory, control))
    return section


def _first_of_month_before(day, months):
    """Primer día del mes `months` meses antes, la misma regla que el calentamiento."""
    position = day.year * 12 + day.month - 1 - months
    return date(position // 12, position % 12 + 1, 1)


def _micros(day):
    return int(np.datetime64(day, "us").astype(np.int64))


def classify(routes, entry):
    """Clase, ausencia y posición en el tramo de cada sesión de un recorrido.

    `routes` son las rutas del calendario desde el comienzo del recorrido, en orden.
    Devuelve tres vectores: la clase (índice en CLASSES, o -1 si la sesión no está
    clasificada), la ausencia en sesiones clasificadas de un tramo de revisita (0 en los
    demás) y la posición h dentro del tramo (0 en las no clasificadas).
    """
    routes = np.asarray(routes, dtype=np.int64)
    kind = np.full(len(routes), -1, dtype=np.int64)
    absence = np.zeros(len(routes), dtype=np.int64)
    position = np.zeros(len(routes), dtype=np.int64)
    last_seen, current, start, revisit, gap, step = {}, None, 0, False, 0, -1
    for index, route in enumerate(routes.tolist()):
        if route == UNCLASSIFIED:
            continue
        step += 1
        if route != current:
            revisit = route in last_seen
            gap = step - last_seen[route] - 1 if revisit else 0
            current, start = route, step
        last_seen[route] = step
        h = step - start + 1
        position[index] = h
        if h > entry:
            kind[index] = CLASSES.index("continuing")
        else:
            kind[index] = CLASSES.index("revisit_entry" if revisit else "novel_entry")
            absence[index] = gap if revisit else 0
    return kind, absence, position


def placebo_routes(routes, lag):
    """Ruta de la sesión `lag` posiciones antes. Las primeras quedan sin clasificar."""
    routes = np.asarray(routes, dtype=np.int64)
    lagged = np.full(len(routes), UNCLASSIFIED, dtype=np.int64)
    lagged[lag:] = routes[: len(routes) - lag] if lag < len(routes) else []
    return lagged


def session_classes(times, calendar, windows, section, start, *, placebo=False):
    """Clases de las sesiones evaluadas de un mercado según el recorrido de su ventana.

    `times` son los instantes UTC de las sesiones de la serie, `calendar` los instantes y
    rutas del mercado, `windows` los intervalos de evaluación [inicio, fin) de cada ventana y
    `start` el comienzo de la historia (STARTS). Una sesión fuera de todas las ventanas o
    ausente del calendario detiene el análisis.
    """
    _require(start in STARTS, "El comienzo de la historia no está declarado")
    at, routes = calendar
    if placebo:
        routes = placebo_routes(routes, section["placebo_lag_sessions"])
    times = np.asarray(times, dtype=np.int64)
    found = np.searchsorted(at, times)
    _require(
        np.all(found < len(at)) and np.array_equal(at[np.minimum(found, len(at) - 1)], times),
        "Alguna sesión evaluada no está en el calendario de regímenes",
    )
    kind = np.full(len(times), -1, dtype=np.int64)
    absence = np.zeros(len(times), dtype=np.int64)
    position = np.zeros(len(times), dtype=np.int64)
    covered = np.zeros(len(times), dtype=bool)
    for begin_day, end in windows:
        first, last = _micros(begin_day), _micros(end)
        history = first
        if start == "since_warmup_start":
            history = _micros(
                _first_of_month_before(date.fromisoformat(begin_day), section["warmup_months"])
            )
        inside = (times >= first) & (times < last)
        _require(not (covered & inside).any(), "Las ventanas de evaluación se solapan")
        if not inside.any():
            continue
        covered |= inside
        begin, stop = np.searchsorted(at, [history, last])
        classes = classify(routes[begin:stop], section["entry_sessions"])
        local = found[inside] - begin
        for target, values in zip((kind, absence, position), classes, strict=True):
            target[inside] = values[local]
    _require(covered.all(), "Alguna sesión evaluada no pertenece a ninguna ventana")
    return kind, absence, position


def _groups(kind, absence, long_absence):
    """Máscara de cada grupo de _GROUPS para las sesiones de una serie."""
    revisit = kind == CLASSES.index("revisit_entry")
    masks = [kind >= 0, *(kind == index for index in range(len(CLASSES)))]
    masks += [revisit & (absence >= long_absence), revisit & (absence < long_absence)]
    return np.stack(masks, axis=1)


def _statistics(means):
    """Los cuatro estadísticos a partir de las medias de los grupos (última dimensión)."""
    group = {name: means[..., index] for index, name in enumerate(_GROUPS)}
    return np.stack(
        [
            group["classified"],
            group["revisit_entry"] - group["novel_entry"],
            group["revisit_long"] - group["revisit_short"],
            group["novel_entry"] - group["continuing"],
        ],
        axis=-1,
    )


def _curves(benefit, defined, kind, absence, position, section):
    """Medias descriptivas por posición en el tramo y por ausencia, sin intervalos."""

    def summary(mask):
        mask = mask & defined
        count = int(mask.sum())
        return dict(sessions=count, mean=float(benefit[mask].mean()) if count else None)

    recovery = {
        name: [
            summary((kind == CLASSES.index(name)) & (position == h))
            for h in range(1, section["entry_sessions"] + 1)
        ]
        for name in ("novel_entry", "revisit_entry")
    }
    revisit = kind == CLASSES.index("revisit_entry")
    lows, highs = [1, *section["absence_bins"]], [*section["absence_bins"], None]
    absence_bins = [
        dict(
            from_sessions=low,
            to_sessions=high,
            **summary(revisit & (absence >= low) & (high is None or absence < high)),
        )
        for low, high in zip(lows, highs, strict=True)
    ]
    return dict(recovery=recovery, absence=absence_bins)


def _pair_cells(benefit, defined, period, classes, placebo, section, periods):
    """Sumas y recuentos diarios de los grupos reales seguidos de los del placebo."""
    long_absence = section["long_absence_sessions"]
    masks = np.concatenate(
        [_groups(*classes[:2], long_absence), _groups(*placebo[:2], long_absence)], axis=1
    )
    masks &= defined[:, None]
    weights = masks.astype(np.float64)
    values = np.where(defined, benefit, 0.0)
    sums = np.stack(
        [np.bincount(period, weights=values * column, minlength=periods) for column in weights.T],
        axis=1,
    )
    sizes = np.stack(
        [np.bincount(period, weights=column, minlength=periods) for column in weights.T], axis=1
    )
    return sums, sizes


def _means(sums, sizes):
    """Media de cada grupo, NaN si el grupo no tiene sesiones."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sizes > 0, sums / np.where(sizes > 0, sizes, 1.0), np.nan)


def _both(means):
    """Estadísticos reales y del placebo a partir de las medias de los dos bloques de grupos."""
    groups = len(_GROUPS)
    return np.stack([_statistics(means[..., :groups]), _statistics(means[..., groups:])], axis=-2)


def _float(value):
    return float(value) if np.isfinite(value) else None


def _bootstrap(cells, options, periods):
    """Estadísticos de cada réplica para todos los pares, con los mismos bloques de días."""
    replicates, block = options["replicates"], options["block_length"]
    draws = {name: np.empty((replicates, 2, len(STATISTICS))) for name in cells}
    rng = np.random.default_rng(options["seed"])
    for offset in range(0, replicates, 256):
        size = min(256, replicates - offset)
        counts = circular_block_counts(rng, size, periods, block).astype(np.float64)
        for name, (sums, sizes) in cells.items():
            draws[name][offset : offset + size] = _both(_means(counts @ sums, counts @ sizes))
    return draws


def market_report(section, options, pairs, periods):
    """Estadísticos, intervalos y decisiones de cada par de un mercado.

    `pairs` asigna a cada par sus vectores (beneficio, definido, día) y las clases de sus
    sesiones con las rutas reales y con las retrasadas, o None si no está en el ámbito.
    """
    result = {name: dict(status="not_in_scope", reason=_NOT_IN_SCOPE) for name in pairs}
    names = [name for name, value in pairs.items() if value is not None]
    if not names:
        return result
    cells = {name: _pair_cells(*pairs[name], section, periods) for name in names}
    draws = None if options["block_length"] >= periods else _bootstrap(cells, options, periods)
    family = []
    for name in names:
        sums, sizes = cells[name]
        totals = sizes.sum(axis=0)
        estimate = _both(_means(sums.sum(axis=0), totals))
        counts = [
            {group: int(totals[k * len(_GROUPS) + i]) for i, group in enumerate(_GROUPS)}
            for k in range(2)
        ]
        record = dict(
            status="computed",
            sessions=counts[0],
            placebo_sessions=counts[1],
            estimates={key: _float(estimate[0, i]) for i, key in enumerate(STATISTICS)},
            placebo_estimates={key: _float(estimate[1, i]) for i, key in enumerate(STATISTICS)},
            intervals={key: None for key in STATISTICS},
            decisions={},
        )
        result[name] = record
        for i, key in enumerate(STATISTICS):
            column = None if draws is None else draws[name][:, 0, i]
            if column is not None and np.isfinite(column).all():
                bounds = family_intervals(
                    estimate[0, i : i + 1], column[:, None], options["confidence"]
                )
                record["intervals"][key] = [float(bounds["lower"][0]), float(bounds["upper"][0])]
        for key, (groups, _, _) in DECISIONS.items():
            i = list(STATISTICS).index(key)
            decision = dict(
                status="undetermined",
                difference_from_placebo=_float(estimate[0, i] - estimate[1, i]),
                intervals=None,
                reason=None,
            )
            record["decisions"][key] = decision
            if draws is None:
                decision["reason"] = "Se necesitan más días que la longitud del bloque"
            elif any(
                counts[k][group] < section["min_sessions"] for k in range(2) for group in groups
            ):
                decision["reason"] = "Alguna clase necesaria tiene menos sesiones de las declaradas"
            elif not np.isfinite(draws[name][:, :, i]).all():
                decision["reason"] = "Alguna réplica no contiene sesiones de una clase necesaria"
            else:
                column = draws[name][:, :, i]
                family.append((name, key, estimate[0, i], estimate[0, i] - estimate[1, i], column))
    if family:
        _decide(result, family, options["confidence"])
    return result


def _decide(result, family, confidence):
    """Intervalos simultáneos de todas las decisiones de un mercado y su resultado."""
    estimate = np.array([value for _, _, real, diff, _ in family for value in (real, diff)])
    draws = np.stack(
        [
            column
            for *_, replicates in family
            for column in (replicates[:, 0], replicates[:, 0] - replicates[:, 1])
        ],
        axis=1,
    )
    joint = family_intervals(estimate, draws, confidence)
    for position, (name, key, *_) in enumerate(family):
        _, sign, (confirmed, rejected) = DECISIONS[key]
        bounds = {
            label: [
                float(joint["joint_lower"][2 * position + k]),
                float(joint["joint_upper"][2 * position + k]),
            ]
            for k, label in enumerate((key, "difference_from_placebo"))
        }
        # Confirmar exige que los dos intervalos queden enteros en el sentido declarado.
        holds = [(low > 0) if sign > 0 else (high < 0) for low, high in bounds.values()]
        if all(holds):
            reason = None
        elif not holds[0]:
            reason = f"El intervalo de {key} no excluye el cero en el sentido declarado"
        else:
            reason = f"{key} no se separa de la del placebo"
        result[name]["decisions"][key].update(
            status=confirmed if all(holds) else rejected,
            intervals=bounds,
            critical=joint["critical"],
            reason=reason,
        )


def report(section, options, windows, series_of, calendar, markets):
    """Sección del informe para cada mercado del ámbito.

    `windows` son los intervalos [inicio, fin) de evaluación de las ventanas, `series_of`
    devuelve para un modelo y un mercado (MAE por sesión, definido, instantes, días) o None
    si el modelo no se compara en el ámbito, y `calendar` asigna a cada mercado sus
    instantes y rutas.
    """
    result = {}
    for market in markets:
        _require(market in calendar, f"El calendario no contiene el mercado {market}")
        reference, inputs = None, {}
        for name, pair in section["pairs"].items():
            memory, control = (series_of(pair[key], market) for key in ("memory", "control"))
            if memory is None or control is None:
                inputs[name] = None
                continue
            reference = memory if reference is None else reference
            for series in (memory, control):
                _require(
                    np.array_equal(series[2], reference[2])
                    and np.array_equal(series[3], reference[3]),
                    "Los modelos del análisis no evalúan las mismas sesiones",
                )
            inputs[name] = (control[0] - memory[0], memory[1] & control[1])
        if reference is None:
            result[market] = dict(pairs=market_report(section, options, inputs, 0))
            continue
        times, period = reference[2], reference[3]
        # Clases con rutas reales y retrasadas para cada comienzo de historia que se use.
        classes = {}
        for name, value in inputs.items():
            start = section["pairs"][name]["history"]
            if value is not None and start not in classes:
                classes[start] = tuple(
                    session_classes(times, calendar[market], windows, section, start, placebo=lag)
                    for lag in (False, True)
                )
        pairs = market_report(
            section,
            options,
            {
                name: None
                if value is None
                else (*value, period, *classes[section["pairs"][name]["history"]])
                for name, value in inputs.items()
            },
            int(period.max()) + 1,
        )
        for name, value in inputs.items():
            if value is not None:
                real = classes[section["pairs"][name]["history"]][0]
                pairs[name]["curves"] = _curves(*value, *real, section)
        result[market] = dict(
            sessions=len(times),
            classes={
                start: {name: int(np.sum(real[0] == i)) for i, name in enumerate(CLASSES)}
                | dict(unclassified=int(np.sum(real[0] < 0)))
                for start, (real, _) in classes.items()
            },
            pairs=pairs,
        )
    return dict(declaration=section, status="computed", markets=result)


def pending(section):
    """Sección declarada sin calendario de regímenes todavía."""
    return dict(
        declaration=section,
        status="not_computed",
        reason="Falta el calendario de regímenes de la edición evaluada",
    )
