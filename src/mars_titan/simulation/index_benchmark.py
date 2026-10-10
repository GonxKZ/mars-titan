"""Índice de mercado chino comprado y mantenido, calculado con niveles diarios del índice.

La edición de precios negociados no contiene ningún fondo cotizado que replique el CSI 300.
Por eso la referencia `market_index` china no pasa por el motor y se calcula aquí, con los
niveles diarios del índice y estas reglas, declaradas antes de ver resultados:

- La decisión se toma al primer cierre del episodio y la compra se ejecuta en la primera
  apertura con nivel de una sesión posterior, como una orden del motor. Paga `cost_bps`
  sobre el importe comprado y deja la cartera invertida por completo.
- El CSI 300 es un índice de precios. No incluye dividendos, de modo que infravalora la
  rentabilidad de una cartera que los cobrase. No hay participación máxima, lotes ni bandas.
- Cada cierre del episodio se valora con el nivel de cierre de su sesión local. Una sesión
  sin nivel conserva el último cierre conocido, como un activo sin fila en la cinta.
- La venta hipotética al último cierre paga `cost_bps` y no paga timbre, porque el índice
  no es una acción.

Los niveles solo se leen después de evaluar y ninguna decisión los usa.
"""

import math
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.storage import sha256

from .evaluation import EQUITY_BASIS

BENCHMARKS = {"csi300_price_index_v1": dict(market="CN", name="CSI 300")}
ZONES = dict(US="America/New_York", CN="Asia/Shanghai")
BASIS = "price_index_levels_without_dividends"
UNAVAILABLE = "index_levels_unavailable"
MAX_FILE_BYTES = 16 * 1024**2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def read_levels(path, digest):
    """Sesiones, aperturas y cierres del índice, con la huella declarada del archivo."""
    _require(
        path.stat().st_size <= MAX_FILE_BYTES and sha256(path) == digest,
        f"Los niveles de {path.name} no coinciden con su huella o son demasiado grandes",
    )
    table = pq.read_table(path, columns=["session", "open", "close"], use_threads=False)
    _require(
        table["session"].type == pa.string()
        and table["session"].null_count == 0
        and all(table[name].type == pa.float64() for name in ("open", "close")),
        "Los niveles necesitan sesiones de texto y aperturas y cierres en coma flotante",
    )
    sessions = np.array(table["session"].to_pylist(), dtype="datetime64[D]")
    opens = table["open"].to_numpy(zero_copy_only=False).astype(np.float64)
    closes = table["close"].to_numpy(zero_copy_only=False).astype(np.float64)
    for values in (opens, closes):
        # Un nivel ausente es NaN. Un infinito o un nivel no positivo es un error de la fuente.
        _require(
            not np.isinf(values).any() and (values[np.isfinite(values)] > 0).all(),
            "Los niveles del índice deben ser positivos o estar ausentes",
        )
    _require(
        len(sessions) >= 2 and (np.diff(sessions.astype(np.int64)) > 0).all(),
        "Las sesiones del índice deben ser únicas y crecientes",
    )
    return dict(session=sessions, open=opens, close=closes, sha256=digest)


def local_sessions(close_times, market):
    """Sesión local de cada cierre de la cinta, en la zona horaria de su mercado."""
    zone = ZoneInfo(ZONES[market])
    return np.array(
        [
            datetime.fromtimestamp(int(value) / 1e6, UTC).astimezone(zone).date().isoformat()
            for value in close_times
        ],
        dtype="datetime64[D]",
    )


def index_episode(levels, close_times, *, market, capital, cost_bps):
    """Patrimonio por cierre y retornos del índice comprado y mantenido en un episodio.

    `close_times` son los cierres del episodio evaluado, en microsegundos UTC. Devuelve el
    registro con la forma de un episodio de la etapa, o un motivo si no hay niveles.
    """
    _require(
        len(close_times) >= 2
        and type(capital) in (int, float)
        and math.isfinite(capital)
        and capital > 0
        and type(cost_bps) in (int, float)
        and 0 <= cost_bps <= 1000,
        "El episodio necesita dos cierres, capital positivo y un coste declarado",
    )
    days = local_sessions(close_times, market)
    # Último nivel conocido en cada sesión del episodio, sin mirar sesiones posteriores.
    known = np.isfinite(levels["close"])
    position = np.searchsorted(levels["session"][known], days, side="right") - 1
    if position[0] < 0:
        return None, UNAVAILABLE
    closes = levels["close"][known][position]
    exact = dict(zip(levels["session"].tolist(), levels["open"].tolist(), strict=True))
    fill = next(
        (k for k in range(1, len(days)) if math.isfinite(exact.get(days[k].item(), math.nan))),
        None,
    )
    if fill is None:
        return None, UNAVAILABLE
    rate = cost_bps / 10_000
    units = capital / (exact[days[fill].item()] * (1 + rate))
    nav = [float(capital)] * fill + [float(units * close) for close in closes[fill:]]
    peaks = np.maximum.accumulate(nav)
    final = nav[-1] * (1 - rate)
    return (
        dict(
            net_return=nav[-1] / capital - 1,
            liquidated_net_return=final / capital - 1,
            max_drawdown=float(np.max(1 - np.asarray(nav) / peaks)),
            costs=capital - units * exact[days[fill].item()],
            turnover=units * exact[days[fill].item()] / capital,
            steps=len(nav) - 1,
            equity=dict(basis=EQUITY_BASIS, close_times=[int(t) for t in close_times], nav=nav),
        ),
        None,
    )
