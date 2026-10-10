"""Calendario de regímenes observables por sesión, para describir la evaluación.

El análisis de retención e interferencia (`evaluation.retention_interference`) necesita
saber en qué régimen estaba cada mercado en cada sesión, desde el comienzo del
calentamiento de la memoria hasta el final de la evaluación. Este módulo calcula esa
etiqueta una vez por mercado y sesión con la regla declarada en `memory.regimes`, la misma
que enruta B6, y la guarda con su identidad.

La etiqueta solo usa precios publicados hasta la decisión. Para la sesión t se forman las
ventanas de 64 sesiones del calendario que terminan en t, con la presencia del contrato
v3.1: una sesión ausente en todo el mercado queda con presencia cero y un activo al que le
falta otra sesión de la ventana no entra. Con todas las ventanas válidas del mercado se
llama a `RegimeRule.state`. Por tanto, cambiar o quitar precios posteriores a t no cambia
la etiqueta de t, y una prueba lo comprueba recortando la historia.

La cohorte es el mercado completo de la edición preparada, no las filas de un brazo. Así
la etiqueta es la misma para todos los brazos y no depende de qué activos tengan objetivo
en una vista. B6 resume en cambio las filas de cada evento, así que su ruta puede diferir
en una sesión concreta. Las dos usan la misma regla y los mismos mínimos.

Las sesiones sin ninguna ventana válida, o con menos activos o rendimientos de los
declarados, quedan en la ruta 0 sin clasificar. Las etiquetas son un descriptor del
evaluador. Ningún modelo las recibe ni decide con ellas.

El calendario no lee ninguna sesión desde 2024: una sesión del test final en los precios
preparados detiene la construcción antes de escribir nada.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.price_windows import CONTEXT_SESSIONS, PRESENCE_CHANNEL, PRICE_WINDOW_CHANNELS
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.memory.regimes import MARKETS, REGIME_RULE, SLOTS, UNCLASSIFIED, RegimeRule

KIND = "observable_regime_calendar"
SCHEMA_VERSION = 1
FINAL_TEST_START = "2024-01-01"
ABSENT_FILE = "market-absent-sessions.json"
_CLOSE = PRICE_WINDOW_CHANNELS.index("log_close_to_anchor_close")
_PRESENT = PRICE_WINDOW_CHANNELS.index(PRESENCE_CHANNEL)
_FIELDS = ("sessions", "at", "route", "assets", "returns")
_MAX_BYTES = 64 * 1024**2
_MAX_ASSETS = 20_000
_SOURCES = ("evaluation/regime_calendar.py", "memory/regimes.py", "data/price_windows.py")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def code_sha256():
    root = Path(__file__).parents[1]
    return {name: sha256(root / name) for name in _SOURCES}


def market_routes(market, sessions, at, closes, absent, rule):
    """Ruta, activos y rendimientos de cada sesión de un mercado.

    `sessions` son las fechas ISO del calendario en orden, `at` el instante UTC en
    microsegundos de cada sesión con filas (-1 en las ausentes), `closes` una matriz
    [activos, sesiones] con NaN donde el activo no tiene precio y `absent` la máscara de las
    sesiones ausentes en todo el mercado. Devuelve una fila por sesión con filas.
    """
    closes = np.asarray(closes, dtype=np.float64)
    absent = np.asarray(absent, dtype=bool)
    at = np.asarray(at, dtype=np.int64)
    count = len(sessions)
    _require(
        closes.ndim == 2
        and closes.shape[1] == count
        and absent.shape == (count,)
        and at.shape == (count,)
        and list(sessions) == sorted(sessions)
        and len(set(sessions)) == count,
        "El calendario de un mercado no es coherente",
    )
    present = np.isfinite(closes)
    _require(not present[:, absent].any(), "Una sesión ausente en todo el mercado tiene precios")
    _require(np.all((at >= 0) != absent), "Cada sesión con filas necesita su instante")
    _require(np.all(np.diff(at[~absent]) > 0), "Los instantes de las sesiones no son crecientes")
    _require(np.all(closes[present] > 0), "Los cierres deben ser positivos para tomar su logaritmo")
    logs = np.where(present, np.log(np.where(present, closes, 1.0)), np.nan)
    # Activos presentes acumulados: una ventana es válida si el activo está en todas sus
    # sesiones con filas. Las ausentes de mercado no cuentan para ningún activo.
    cumulative = np.concatenate(
        [np.zeros((len(closes), 1), dtype=np.int64), np.cumsum(present, axis=1)], axis=1
    )
    rows = []
    for end in np.flatnonzero(~absent):
        first = end - CONTEXT_SESSIONS + 1
        if first < 0:
            # La ventana empezaría antes de la primera sesión observada del mercado.
            rows.append((end, UNCLASSIFIED, 0, int((~absent[: end + 1]).sum()) - 1))
            continue
        slots = ~absent[first : end + 1]
        needed = int(slots.sum())
        valid = np.flatnonzero(cumulative[:, end + 1] - cumulative[:, first] == needed)
        if not len(valid):
            rows.append((end, UNCLASSIFIED, 0, needed - 1))
            continue
        window = logs[valid, first : end + 1]
        anchor = window[:, int(np.argmax(slots))]
        prices = np.zeros((len(valid), CONTEXT_SESSIONS, len(PRICE_WINDOW_CHANNELS)))
        prices[:, slots, _CLOSE] = window[:, slots] - anchor[:, None]
        prices[:, :, _PRESENT] = slots
        state = rule.state(market, prices, int(at[end]))
        rows.append((end, state.route, state.assets, state.returns))
    index = np.array([row[0] for row in rows], dtype=np.int64)
    return dict(
        sessions=[sessions[i] for i in index],
        at=at[index].tolist(),
        route=[row[1] for row in rows],
        assets=[row[2] for row in rows],
        returns=[row[3] for row in rows],
    )


def _read_market(prepared, market, absent_record):
    """Cierres e instantes de todos los activos preparados de un mercado."""
    folder = prepared / market
    _require(folder.is_dir() and not folder.is_symlink(), f"Falta el mercado {market}")
    last = absent_record["last_considered_session"]
    _require(last < FINAL_TEST_START, "La edición considera sesiones del test final")
    series, moments = [], {}
    paths = sorted(folder.glob("*/prices.parquet"))
    _require(1 <= len(paths) <= _MAX_ASSETS, f"El mercado {market} no tiene activos acotados")
    for path in paths:
        table = pq.read_table(path, columns=["close", "session", "available_at"])
        days = [str(day) for day in table["session"].to_pylist()]
        _require(
            all(day < FINAL_TEST_START for day in days),
            f"{path.parent.name} contiene sesiones del test final",
        )
        stamps = table["available_at"].cast("int64").to_numpy()
        for day, stamp in zip(days, stamps.tolist(), strict=True):
            _require(
                moments.setdefault(day, stamp) == stamp,
                f"La sesión {day} de {market} tiene instantes distintos según el activo",
            )
        series.append((days, table["close"].to_numpy()))
    absent = set(absent_record["sessions"])
    _require(not absent & set(moments), "Una sesión declarada ausente tiene precios")
    first = absent_record["first_observed_session"]
    sessions = sorted(day for day in set(moments) | absent if first <= day <= last)
    position = {day: i for i, day in enumerate(sessions)}
    closes = np.full((len(series), len(sessions)), np.nan)
    for row, (days, values) in enumerate(series):
        kept = [i for i, day in enumerate(days) if day in position]
        closes[row, [position[days[i]] for i in kept]] = values[kept]
    at = np.array([moments.get(day, -1) for day in sessions], dtype=np.int64)
    mask = np.array([day in absent for day in sessions])
    return sessions, at, closes, mask


def build(prepared_root, *, rule=None, markets=MARKETS):
    """Calcular el calendario de los mercados pedidos sobre una edición preparada."""
    rule = RegimeRule(REGIME_RULE) if rule is None else rule
    prepared_root = Path(prepared_root)
    manifest, manifest_sha256 = read_manifest(prepared_root / "manifest.json")
    absent_records, absent_sha256 = read_manifest(prepared_root / ABSENT_FILE)
    _require(
        manifest.get("kind") == "prepared_cohort" and manifest.get("status") == "completed",
        "La edición preparada no está completa",
    )
    _require(markets and set(markets) <= set(MARKETS), "Los mercados deben ser US o CN")
    calendar = {}
    for market in markets:
        sessions, at, closes, absent = _read_market(
            prepared_root / "prepared", market, absent_records[market]
        )
        calendar[market] = market_routes(market, sessions, at, closes, absent, rule)
    return dict(
        schema_version=SCHEMA_VERSION,
        kind=KIND,
        rule=rule.identity(),
        cohort="all_prepared_assets_of_the_market_with_a_valid_window_at_the_session",
        causality="windows_of_64_calendar_sessions_ending_at_the_decision",
        prepared=dict(
            manifest_sha256=manifest_sha256,
            market_absent_sessions_sha256=absent_sha256,
            cohort_id=manifest.get("cohort_id"),
            input_policy=manifest.get("input_policy"),
        ),
        code_sha256=code_sha256(),
        final_test_opened=False,
        markets={market: calendar[market] for market in markets},
    )


def check(calendar):
    """Validar un calendario leído y devolver por mercado instantes y rutas en NumPy."""
    _require(
        isinstance(calendar, dict)
        and calendar.get("schema_version") == SCHEMA_VERSION
        and calendar.get("kind") == KIND
        and calendar.get("final_test_opened") is False
        and isinstance(calendar.get("markets"), dict)
        and calendar["markets"]
        and set(calendar["markets"]) <= set(MARKETS),
        "El calendario de regímenes no cumple su contrato",
    )
    rule = calendar.get("rule")
    _require(
        isinstance(rule, dict) and rule.get("rule") == REGIME_RULE and rule.get("slots") == SLOTS,
        "El calendario no usa la regla de régimen declarada",
    )
    resolved = {}
    for market, record in calendar["markets"].items():
        _require(
            isinstance(record, dict) and set(record) == set(_FIELDS),
            f"El calendario de {market} no tiene sus campos",
        )
        size = len(record["sessions"])
        _require(
            size >= 1 and all(len(record[name]) == size for name in _FIELDS),
            f"Las columnas del calendario de {market} no tienen la misma longitud",
        )
        _require(
            all(isinstance(day, str) and day < FINAL_TEST_START for day in record["sessions"]),
            f"El calendario de {market} contiene sesiones del test final",
        )
        at = np.array(record["at"], dtype=np.int64)
        route = np.array(record["route"], dtype=np.int64)
        _require(np.all(np.diff(at) > 0), f"Los instantes de {market} no son crecientes")
        _require(
            np.all((route >= UNCLASSIFIED) & (route < SLOTS)),
            f"Las rutas de {market} están fuera de rango",
        )
        resolved[market] = (at, route)
    return resolved


def read(path):
    """Leer un calendario con su huella. La huella identifica el calendario en el informe."""
    path = Path(path)
    calendar, digest = read_manifest(path, _MAX_BYTES)
    return check(calendar), digest, calendar["rule"]


def write(calendar, output, *, sources=()):
    """Publicar el calendario en un archivo nuevo, fuera de las fuentes."""
    output = Path(output)
    safe_destination(output)
    for source in sources:
        outside_source(Path(source), output)
    _require(not output.exists(), "El calendario necesita un archivo nuevo")
    check(calendar)
    atomic_json(output, calendar)
    return sha256(output)


def summary(calendar):
    """Recuento de sesiones por mercado, año y ruta, sin ningún dato de resultado."""
    result = {}
    for market, record in calendar["markets"].items():
        years = {}
        for day, route in zip(record["sessions"], record["route"], strict=True):
            counts = years.setdefault(day[:4], [0] * SLOTS)
            counts[route] += 1
        result[market] = dict(
            sessions=len(record["sessions"]),
            first=record["sessions"][0],
            last=record["sessions"][-1],
            routes_by_year=years,
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--market", action="append", choices=MARKETS)
    args = parser.parse_args(argv)
    markets = tuple(args.market or MARKETS)
    calendar = build(args.prepared, markets=markets)
    digest = write(calendar, args.output, sources=(args.prepared,))
    print(json.dumps(dict(output=str(args.output), sha256=digest, **summary(calendar)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
