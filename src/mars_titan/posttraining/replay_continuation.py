"""Repaso de filas anteriores en la continuación del walk-forward por etapas (#55, MT-053).

En la ventana k la continuación completa ajusta el padre de k-1 solo con las filas nuevas,
unos meses de datos tras veinte años de ajuste. El repaso es la forma más directa de
consolidar lo aprendido antes sin otra regularización: cada época añade a las filas nuevas
visitas de sesiones que el padre ya usó (Rolnick et al., 2019, Chaudhry et al., 2019). Se
repasan las entradas reales y no representaciones guardadas, así que el modelo las vuelve a
codificar con sus pesos de cada momento y no hay claves ni valores obsoletos que reconstruir
(el problema que plantea el repaso latente de Pellegrini et al., 2020).

La sección opcional `replay_continuation` de la matriz de versión 3 declara la proporción de
actualizaciones de repaso por actualización de filas nuevas y los ámbitos donde se propone.
De ella salen tres controles con el mismo objetivo, presupuesto, λ y selección que la
continuación completa, y con las mismas actualizaciones entre ellos:

- `replay_uniform` repasa sesiones anteriores elegidas al azar, en proporción al tiempo.
- `replay_regime_balanced` reparte las mismas actualizaciones por igual entre los regímenes
  observables de su mercado, con la regla `observable_volatility_trend_v1` de #523 aplicada a
  la serie del factor de mercado de la vista. Es la variante con diversidad: da más peso a
  los regímenes poco frecuentes, como los tramos turbulentos y bajistas.
- `repeat_new_rows` repite con el mismo número de actualizaciones sesiones de las filas
  nuevas. Es el control que descarta el repaso: si iguala a los otros dos, la ganancia viene
  de hacer más actualizaciones y no de volver a ver el pasado.

Las sesiones de repaso y su orden se fijan una vez por caso, antes del primer ajuste, con su
propio generador. Cada época recorre todas las filas nuevas en el mismo orden que la
continuación completa y las mismas visitas adicionales en posiciones nuevas. Una sesión
anterior solo entra si el padre ajustó todas sus filas y todas sus etiquetas maduraron antes
de la primera fila nueva. La validación y la calibración del padre, que el ajuste por etapas
deja fuera, tampoco entran en el repaso. Las familias con entrenador cronológico quedan
fuera: repasar una sesión antigua exige reconstruir el estado de sus flujos desde un
calentamiento propio.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from mars_titan.data.price_windows import CONTEXT_SESSIONS, PRESENCE_CHANNEL, PRICE_WINDOW_CHANNELS
from mars_titan.memory.regimes import REGIME_RULE, SLOTS, UNCLASSIFIED, RegimeRule

from . import adapter_variety

SECTION_NAME = "replay_continuation"
UNIFORM, REGIME, REPEAT = "replay_uniform", "replay_regime_balanced", "repeat_new_rows"
CONTROLS = (UNIFORM, REGIME, REPEAT)
# Clave del caso con la política y la proporción de su repaso.
KEY = "replay"
SECTION = {"ratio", "campaign"}
EARLIER, NEW = "earlier_rows", "new_rows"
SOURCES = {UNIFORM: EARLIER, REGIME: EARLIER, REPEAT: NEW}
# La regla de #523 resume un mercado con la mediana transversal de sus activos. Aquí el
# mercado es la serie de su factor, así que basta un activo. El resto de la regla no cambia.
RULE = RegimeRule(REGIME_RULE, min_assets=1)
# Flujos de los generadores del repaso, separados de la permutación de cada época.
SELECTION_STREAM, ORDER_STREAM = 5501, 5502
_CLOSE = PRICE_WINDOW_CHANNELS.index("log_close_to_anchor_close")
_PRESENT = PRICE_WINDOW_CHANNELS.index(PRESENCE_CHANNEL)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ratio(value):
    _require(
        type(value) is float and math.isfinite(value) and 0 < value <= 1,
        "La proporción del repaso es una fracción entre 0 y 1 de las actualizaciones nuevas",
    )
    return value


def validate(section, matrix):
    """Comprobar la sección frente a la matriz de versión 3."""
    _require(
        isinstance(section, dict)
        and set(section) == SECTION
        and isinstance(section["campaign"], list)
        and len(set(section["campaign"])) == len(section["campaign"])
        and set(section["campaign"]) <= {adapter_variety.REFERENCES},
        "El repaso declara su proporción y solo se propone para las referencias",
    )
    _ratio(section["ratio"])
    _require(
        matrix["selection"]["patience"] is None,
        "El repaso conserva todas las actualizaciones de sus épocas",
    )
    return section


def declared(matrix):
    """Sección del repaso en la matriz, o None si no la declara."""
    return matrix.get(SECTION_NAME) if matrix["schema_version"] >= 3 else None


def proposed(matrix, scope, *, reserve=False):
    """Si la matriz declara el repaso y lo propone en el ámbito, o en todos con `reserve`."""
    section = declared(matrix)
    return section is not None and (reserve or scope in section["campaign"])


def spec(control, ratio):
    """Política y proporción que el caso de un control declara en su clave `replay`."""
    _require(control in CONTROLS, "El control de repaso no está declarado")
    return dict(policy=control, ratio=_ratio(ratio))


def check_spec(value):
    _require(
        isinstance(value, dict)
        and set(value) == {"policy", "ratio"}
        and value["policy"] in CONTROLS,
        "El repaso del caso declara una política conocida y su proporción",
    )
    _ratio(value["ratio"])
    return value


def epoch_updates(counts, batch_size):
    """Actualizaciones de una época que recorre una vez cada sesión en lotes de `batch_size`.

    Cada sesión se parte en sus propios lotes, sin juntar filas de sesiones distintas.
    """
    return sum(math.ceil(count / batch_size) for count in counts)


def extra_updates(updates, ratio):
    """Actualizaciones de repaso de cada época: la proporción de las nuevas, por exceso."""
    _require(type(updates) is int and updates >= 1, "Faltan las actualizaciones nuevas")
    return math.ceil(_ratio(ratio) * updates)


def estimated_epoch_rows(rows, ratio, batch_size):
    """Filas de una época con repaso cuando solo se conocen las filas nuevas.

    Supone lotes completos: las actualizaciones nuevas se acotan por debajo con
    ceil(filas / lote) y cada actualización adicional repasa un lote completo. Sirve para
    presupuestar, no para fijar el plan, que cuenta las cohortes reales.
    """
    _require(
        type(rows) is int and rows >= 1 and type(batch_size) is int and batch_size >= 1,
        "Las filas nuevas y el lote deben ser positivos",
    )
    return rows + extra_updates(math.ceil(rows / batch_size), ratio) * batch_size


def market_routes(available_at, closes, moments, market):
    """Régimen observable de cada decisión con la serie de cierres de un factor de mercado.

    `available_at` son los instantes en microsegundos en que se conoce cada cierre, en orden
    creciente. Para cada decisión se usa la ventana de las últimas 64 sesiones conocidas en
    ese instante, con la misma regla que #523 aplica a la mediana de los activos. Con menos
    rendimientos de los declarados la decisión queda sin clasificar.
    """
    available_at = np.asarray(available_at, dtype=np.int64)
    closes = np.asarray(closes, dtype=np.float64)
    _require(
        available_at.ndim == 1
        and available_at.shape == closes.shape
        and np.all(np.diff(available_at) > 0)
        and np.all(np.isfinite(closes))
        and np.all(closes > 0),
        "El factor de mercado necesita cierres positivos ordenados por disponibilidad",
    )
    routes = []
    for at in moments:
        known = int(np.searchsorted(available_at, at, side="right"))
        start = max(0, known - CONTEXT_SESSIONS)
        if known - start < 2:
            routes.append(UNCLASSIFIED)
            continue
        window = np.zeros((1, CONTEXT_SESSIONS, len(PRICE_WINDOW_CHANNELS)))
        values = np.log(closes[start:known])
        steps = slice(CONTEXT_SESSIONS - len(values), CONTEXT_SESSIONS)
        window[0, steps, _CLOSE] = values - values[0]
        window[0, steps, _PRESENT] = 1.0
        routes.append(RULE.state(market, window, int(at)).route)
    return routes


def order(candidates, policy, *, seed, routes=None):
    """Orden de las sesiones candidatas a repasar, fijado una vez por caso.

    Uniforme y repetición permutan las candidatas. El reparto por régimen permuta cada
    régimen y toma después una sesión de cada uno por turno, en el orden de las rutas, hasta
    agotarlos. Así las primeras actualizaciones ya cubren todos los regímenes presentes.
    """
    rng = np.random.default_rng([seed, SELECTION_STREAM])
    candidates = list(candidates)
    _require(candidates, "No hay sesiones candidatas para el repaso")
    if policy != REGIME:
        return [candidates[i] for i in rng.permutation(len(candidates))]
    _require(
        routes is not None and len(routes) == len(candidates),
        "El reparto por régimen necesita la ruta de cada sesión",
    )
    groups = {}
    for candidate, route in zip(candidates, routes, strict=True):
        _require(type(route) is int and 0 <= route < SLOTS, "La ruta no pertenece a la regla")
        groups.setdefault(route, []).append(candidate)
    queues = {
        route: [items[i] for i in rng.permutation(len(items))]
        for route, items in sorted(groups.items())
    }
    result = []
    while queues:
        for route in list(queues):
            result.append(queues[route].pop(0))
            if not queues[route]:
                del queues[route]
    return result


def plan(ordered, rows, *, batch_size, updates):
    """Sesiones del repaso y lotes de cada una hasta sumar exactamente `updates`.

    `rows` da las filas de cada sesión. La última sesión puede quedar recortada a sus primeros
    lotes. Devuelve pares (sesión, lotes).
    """
    _require(type(batch_size) is int and batch_size >= 1, "El lote del repaso no es válido")
    remaining, result = updates, []
    for session in ordered:
        if not remaining:
            break
        batches = min(math.ceil(rows[session] / batch_size), remaining)
        result.append((session, batches))
        remaining -= batches
    _require(not remaining, "Las sesiones candidatas no bastan para el repaso declarado")
    return result


def positions(count, extra, *, seed, epoch):
    """Posiciones de las visitas adicionales entre las `count` visitas nuevas de una época.

    Las visitas nuevas conservan su orden. Las adicionales ocupan posiciones al azar y en un
    orden propio de la época.
    """
    rng = np.random.default_rng([seed, epoch, ORDER_STREAM])
    slots = np.sort(rng.choice(count + extra, size=extra, replace=False))
    return slots.tolist(), rng.permutation(extra).tolist()


def earlier_sessions(dataset, since, fitted, fitted_samples):
    """Sesiones de ajuste de la vista anteriores a `since` que el padre ajustó completas.

    `fitted` es el tramo `train` de la vista del padre según `staged_rows.partition_rows`,
    `(mercado, activo) -> (filas, decisiones, madurez)`, y `fitted_samples` la huella del
    archivo de muestras de cada activo. Una sesión entra si todas sus filas de ajuste en esta
    vista son filas que el padre ajustó (mismo activo y misma fila de su archivo de
    muestras) y todas sus etiquetas maduran antes de `since`. Así el repaso nunca añade
    filas de la validación o la calibración del padre, que el ajuste por etapas deja fuera.

    Devuelve, en orden de decisión, los instantes, las filas y los mercados de cada sesión
    admitida, y cuántas sesiones anteriores quedan fuera por tener filas que el padre no
    ajustó o etiquetas que maduran en `since` o después.
    """
    from .staged_rows import partition_rows

    _require(type(since) is int and since > 0, "Falta la primera decisión nueva")
    found, samples = partition_rows(dataset, ("train",))
    markets = sorted({market for market, _ in found["train"]})
    decisions, maturity, codes, known = [], [], [], []
    for key, (indices, decided, mature) in found["train"].items():
        before = decided < since
        parent = fitted.get(key)
        _require(
            parent is None or samples[key] == fitted_samples.get(key),
            "Un activo cambia de archivo de muestras entre la vista del padre y la ventana",
        )
        decisions.append(decided[before])
        maturity.append(mature[before])
        codes.append(np.full(int(before.sum()), markets.index(key[0]), dtype=np.int64))
        known.append(
            np.zeros(int(before.sum()), dtype=bool)
            if parent is None
            else np.isin(indices[before], parent[0])
        )
    decisions, maturity, codes, known = (
        np.concatenate(values) for values in (decisions, maturity, codes, known)
    )
    _require(len(decisions), "No hay sesiones de ajuste anteriores a las filas nuevas")
    moments, inverse, rows = np.unique(decisions, return_inverse=True, return_counts=True)
    latest = np.full(len(moments), np.iinfo(np.int64).min, dtype=np.int64)
    np.maximum.at(latest, inverse, maturity)
    complete = np.ones(len(moments), dtype=bool)
    np.logical_and.at(complete, inverse, known)
    present = np.zeros((len(moments), len(markets)), dtype=bool)
    present[inverse, codes] = True
    late = latest >= since
    admitted = complete & ~late
    _require(admitted.any(), "Ninguna sesión anterior la ajustó el padre completa")
    return dict(
        moments=moments[admitted].tolist(),
        rows=rows[admitted].tolist(),
        markets=[[markets[j] for j in np.flatnonzero(row)] for row in present[admitted]],
        not_fitted=int((~complete).sum()),
        late=int((complete & late).sum()),
        since=since,
    )


def factor_routes(manifest, sessions):
    """Régimen de cada sesión admitida con el factor de mercado que declara la vista.

    El archivo del factor se comprueba con su huella. Una sesión con filas de dos mercados o
    de un mercado sin factor queda sin clasificar.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from mars_titan.data.storage import sha256

    factors = manifest.get("market_factors") or {}
    series = {}
    for market, factor in factors.items():
        path = Path(factor["prices_path"])
        _require(sha256(path) == factor["prices_sha256"], f"El factor de {market} ha cambiado")
        table = pq.read_table(path, columns=["close", "available_at"])
        moments = table["available_at"].cast(pa.timestamp("us", tz="UTC")).cast(pa.int64())
        series[market] = (moments.to_numpy(), table["close"].to_numpy())
    routes = [UNCLASSIFIED] * len(sessions["moments"])
    for market, (available_at, closes) in series.items():
        chosen = [i for i, names in enumerate(sessions["markets"]) if names == [market]]
        moments = [sessions["moments"][i] for i in chosen]
        found = market_routes(available_at, closes, moments, market)
        for i, route in zip(chosen, found, strict=True):
            routes[i] = route
    return routes


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
