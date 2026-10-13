"""Estado de cotización acreditado de cada activo: admisión, advertencia de riesgo y salida.

La tabla procede de fuentes oficiales capturadas (`data/china_listing_status.py`) y solo
contiene hechos fechados hasta el corte de la edición, el 31 de diciembre de 2023. Un estado
que seguía vigente en el corte se guarda sin final, de modo que la tabla no revela nada de
2024. Para cada acción A guarda la fecha oficial de admisión, la primera sesión con banda
diaria tras una salida a bolsa sin límites, los tramos con advertencia de riesgo (ST o *ST),
los tramos sin reforma accionarial (prefijo S, con la misma banda reducida) y los días sin
límite que siguen a una reforma o a una reanudación de cotización.
Para cualquier mercado puede guardar el precio de salida de una baja, con su fecha de pago y
su fuente. Sin esa entrada, la baja se trata sin precio de salida.

Las reglas que se derivan de estos datos están en `market_rules.py` y en el lector C++ de la
cinta reconstruida, con la misma tabla y el mismo algoritmo.
"""

import hashlib
import json
import math
import re
from datetime import date
from pathlib import Path

TABLE_KIND = "listing_status_table"
CUTOFF = "2023-12-31"
# Primera fecha desde la que los tramos ST están comprobados. Las cintas chinas empiezan en
# 2011, y un tramo que ya estaba vigente en esta fecha empieza en ella.
COVERAGE_FROM = "2010-01-01"
MAX_TABLE_BYTES = 16 * 1024**2
CHINA_FIELDS = (
    "listed_on",
    "limit_free_until",
    "special_treatment",
    "share_reform_pending",
    "limit_free_days",
)
EXIT_FIELDS = ("last_session", "price", "currency", "paid_on", "source")
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
_HEX = re.compile(r"[a-f0-9]{64}")
_ASSET = re.compile(r"(US|CN)/[A-Za-z0-9.\-]{1,32}")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def day(value, *, optional=False, bounded=True):
    """Fecha ISO de calendario, no posterior al corte si `bounded`. None solo si se admite."""
    if value is None and optional:
        return None
    _require(
        isinstance(value, str) and _DAY.fullmatch(value) and (not bounded or value <= CUTOFF),
        "Una fecha del estado de cotización no es ISO o es posterior al corte",
    )
    date.fromisoformat(value)
    return value


def _spans(spans, label):
    """Tramos [inicio, fin) por días de Pekín, ordenados, sin solapes y dentro de la cobertura.

    Solo el último puede no tener fin, si seguía vigente en el corte.
    """
    _require(
        isinstance(spans, list) and len(spans) <= 30, f"Los tramos {label} deben formar una lista"
    )
    previous = None
    for index, span in enumerate(spans):
        _require(
            isinstance(span, list) and len(span) == 2, f"Un tramo {label} necesita inicio y fin"
        )
        start = day(span[0])
        end = day(span[1], optional=index == len(spans) - 1)
        _require(
            start >= COVERAGE_FROM, f"Un tramo {label} empieza antes de la cobertura comprobada"
        )
        _require(
            (previous is None or previous <= start) and (end is None or start < end),
            f"Los tramos {label} deben estar ordenados y sin solapes",
        )
        previous = end


def china_entry(asset, entry):
    """Comprobar la entrada de una acción A.

    Hay dos tramos con banda reducida en el tablero principal: la advertencia de riesgo (ST o
    *ST) y la reforma accionarial pendiente (prefijo S). Los días sin límite tras una reforma
    o una reanudación de cotización son sesiones de XSHG ordenadas y sin repetir. La primera
    sesión con banda tras la salida a bolsa se recalcula con el calendario y puede caer en
    enero de 2024 si la admisión fue en los últimos días de 2023. Es lo único que no procede
    de la fuente, sino de la regla de las cinco sesiones.
    """
    from .market_rules import limit_free_until, xshg_session_set

    _require(
        isinstance(entry, dict) and set(entry) == set(CHINA_FIELDS),
        "La entrada de una acción A no conserva sus campos",
    )
    listed = day(entry["listed_on"])
    free = day(entry["limit_free_until"], optional=True, bounded=False)
    _require(
        free == limit_free_until(asset, listed),
        "La exención de límites no corresponde a la admisión y a la regla de su tablero",
    )
    _spans(entry["special_treatment"], "ST")
    _spans(entry["share_reform_pending"], "S")
    days = entry["limit_free_days"]
    _require(
        isinstance(days, list) and len(days) <= 30,
        "Los días sin límite deben formar una lista",
    )
    sessions = xshg_session_set()
    for index, value in enumerate(days):
        _require(
            day(value) >= COVERAGE_FROM
            and date.fromisoformat(value) in sessions
            and (index == 0 or days[index - 1] < value),
            "Un día sin límite no es una sesión ordenada de la cobertura",
        )
    return entry


def exit_entry(entry, sources, currency):
    """Precio de salida acreditado de una baja: importe por acción, pago y fuente."""
    _require(
        isinstance(entry, dict)
        and set(entry) == set(EXIT_FIELDS)
        and type(entry["price"]) is float
        and math.isfinite(entry["price"])
        and entry["price"] >= 0
        and entry["currency"] == currency
        and entry["source"] in sources,
        "La salida no conserva su precio, su moneda o su fuente",
    )
    _require(
        day(entry["last_session"]) < day(entry["paid_on"]),
        "El cobro de una baja debe ser posterior a la última sesión de la serie",
    )
    return entry


def read_listing_status(path):
    """Leer la tabla y comprobar su contrato. Devuelve la tabla y la huella de sus bytes."""
    path = Path(path)
    _require(
        path.is_file() and 0 < path.stat().st_size <= MAX_TABLE_BYTES,
        "La tabla del estado de cotización no es legible o supera su presupuesto",
    )
    data = path.read_bytes()
    table = json.loads(data)
    _require(
        isinstance(table, dict)
        and set(table) == {"kind", "schema_version", "cutoff", "sources", "china", "exits"}
        and table["kind"] == TABLE_KIND
        and table["schema_version"] == 1
        and table["cutoff"] == CUTOFF
        and isinstance(table["sources"], dict)
        and all(
            isinstance(value, dict) and _HEX.fullmatch(str(value.get("sha256", "")))
            for value in table["sources"].values()
        )
        and isinstance(table["china"], dict)
        and isinstance(table["exits"], dict),
        "La tabla no declara su contrato, su corte y sus fuentes",
    )
    for asset, entry in table["china"].items():
        _require(_ASSET.fullmatch(asset) and asset.startswith("CN/"), "Activo chino no válido")
        china_entry(asset, entry)
    for asset, entry in table["exits"].items():
        _require(_ASSET.fullmatch(asset), "La salida pertenece a un activo no válido")
        exit_entry(entry, table["sources"], "CNY" if asset.startswith("CN/") else "USD")
    return table, hashlib.sha256(data).hexdigest()


def tape_status(table, market, assets):
    """Entradas chinas de los activos de una cinta. Una cinta de EE. UU. no tiene ninguna."""
    if market != "CN":
        return {}
    missing = sorted(set(assets) - set(table["china"]))
    _require(not missing, f"Falta el estado de cotización de {len(missing)} activos chinos")
    return {asset: table["china"][asset] for asset in sorted(assets)}


def require_tape_status(market, assets, entries, first_close):
    """La auditoría de una cinta china cubre cada activo y la de EE. UU. está vacía.

    Una cinta china no puede empezar antes de la cobertura comprobada de los tramos, porque
    antes de esa fecha la tabla no afirma nada. `first_close` es el primer cierre de la
    cinta en microsegundos UTC.
    """
    from .market_rules import beijing_day

    _require(
        isinstance(entries, dict) and set(entries) == (set(assets) if market == "CN" else set()),
        "El estado de cotización de la cinta no cubre exactamente sus activos",
    )
    _require(
        market != "CN"
        or first_close >= beijing_day(*date.fromisoformat(COVERAGE_FROM).timetuple()[:3]),
        "Una cinta china empieza antes de la cobertura del estado de cotización",
    )
    for asset, entry in entries.items():
        china_entry(asset, entry)
