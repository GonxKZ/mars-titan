"""Ejecución de apertura a cierre en la sesión siguiente con precios negociados reconstruidos.

La cartera larga y corta del protocolo decide al cierre de t, abre en la apertura de la
sesión siguiente del calendario y liquida al cierre de esa misma sesión, que es el
horizonte del objetivo residual. Este módulo responde, para cada fila (activo, decisión),
si esa apertura es ejecutable y con qué precios, con las reglas de la cinta reconstruida
de la RL (``reconstructed_tape``):

- Solo filas verificadas de la edición. Sin fila, sin verificar o con volumen cero no hay
  ejecución. Nada se rellena.
- Los precios en la rejilla se llevan a su múltiplo exacto. Una apertura fuera de rejilla
  no es ejecutable, igual que la apertura de un día con split y dividendo a la vez.
- En China, las bandas diarias y el timbre de ``market_rules`` (``cn_a_share_v1``). La
  referencia es el último cierre negociado verificado anterior, menos el dividendo por
  acción y dividida por el split de esa apertura. Una compra no se ejecuta con la apertura
  en el límite superior o por encima, y una venta en corto con la apertura en el límite
  inferior o por debajo. Sin referencia o sin tablero acreditado no se opera.

Dividendos y splits de la fecha ex no cambian una posición de apertura a cierre: los dos
precios comparten base y quien compra en la apertura ex no cobra el dividendo. La edición
termina el 31 de diciembre de 2023, así que nada de este módulo lee la reserva de 2024.
Cada instancia conserva solo las sesiones de su intervalo, para acotar la memoria.
"""

import hashlib
import io
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from mars_titan.data.temporal import MarketClock
from mars_titan.data.unadjusted_edition import CUTOFF, EVENT_SCHEMA, PRICE_SCHEMA

from .market_rules import RULES, china_a_share_instrument
from .reconstructed_tape import MAX_FILE_BYTES, _snap, read_edition

REASONS = (
    "executable",
    "no_row",
    "unverified",
    "no_volume",
    "off_grid_open",
    "ambiguous_event",
    "board_without_rules",
    "no_reference_close",
)
EXECUTABLE, NO_ROW, UNVERIFIED, NO_VOLUME, OFF_GRID, AMBIGUOUS, NO_BOARD, NO_REFERENCE = range(8)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _microseconds(moment):
    return int(moment.timestamp()) * 1_000_000 + moment.microsecond


def _table(root, manifest, key, name, schema):
    """Leer un artefacto de la edición tras comprobar su huella, sin convertirlo a Python."""
    expected = manifest["receipts"][key][name]
    path = Path(root) / "assets" / key / name
    _require(
        path.is_file() and path.stat().st_size <= MAX_FILE_BYTES,
        f"Falta un artefacto de la edición: {key}/{name}",
    )
    data = path.read_bytes()
    _require(
        hashlib.sha256(data).hexdigest() == expected,
        f"Un artefacto de la edición ha cambiado: {key}/{name}",
    )
    table = pq.read_table(io.BytesIO(data))
    _require(table.schema.equals(schema), f"El artefacto no conserva su esquema: {key}/{name}")
    return table


class _Asset:
    """Sesiones de un activo desde ``first``, con precios ejecutables y su motivo."""

    def __init__(self, market, prices, events, instrument, first):
        sessions = np.array(prices["session"].to_pylist(), dtype="datetime64[D]")
        _require(
            not len(sessions) or sessions.max() <= np.datetime64(CUTOFF),
            "La edición contiene filas posteriores a su corte",
        )
        _require(
            len(sessions) < 2 or bool((np.diff(sessions) > np.timedelta64(0, "D")).all()),
            "Las sesiones de un activo no están ordenadas",
        )
        verified = prices["verified"].to_numpy(zero_copy_only=False).astype(bool)
        volume = prices["volume"].to_numpy()
        opened, grid = _snap(market, sessions, prices["open"].to_numpy())
        closed, _ = _snap(market, sessions, prices["close"].to_numpy())
        traded = verified & (volume > 0)
        _require(
            bool(np.isfinite(volume[verified]).all())
            and bool((opened[traded] > 0).all())
            and bool((closed[traded] > 0).all()),
            "La edición contiene precios o volúmenes no válidos en filas verificadas",
        )
        reason = np.select(
            [~verified, ~(volume > 0), ~grid], [UNVERIFIED, NO_VOLUME, OFF_GRID], EXECUTABLE
        ).astype(np.int8)
        cash, split = self._events(events, sessions)
        reason[(cash > 0) & (split > 0) & (reason == EXECUTABLE)] = AMBIGUOUS
        # Último cierre negociado verificado anterior a cada sesión, como en la cinta.
        last = np.maximum.accumulate(np.where(traded, np.arange(len(sessions)), -1))
        previous = np.r_[-1, last[:-1]] if len(sessions) else last
        reference = np.where(previous >= 0, closed[np.maximum(previous, 0)], np.nan)
        reference = (reference - cash) / np.where(split > 0, split, 1.0)
        keep = sessions >= first
        executable = reason == EXECUTABLE
        self.sessions, self.reason, self.instrument = sessions[keep], reason[keep], instrument
        self.open = np.where(executable, opened, np.nan)[keep]
        self.close = np.where(executable, closed, np.nan)[keep]
        self.reference = reference[keep] if instrument else None

    @staticmethod
    def _events(events, sessions):
        """Dividendo por acción negociada y razón de split de cada sesión, cero si no hay."""
        days = np.array(events["session"].to_pylist(), dtype="datetime64[D]")
        published = events["provider_dividend"].to_numpy()
        raw = events["raw_dividend"].to_numpy()
        ratio = events["provider_split"].to_numpy()
        cash = np.zeros(len(sessions))
        split = np.zeros(len(sessions))
        position = np.searchsorted(sessions, days)
        known = position < len(sessions)
        known[known] = sessions[position[known]] == days[known]
        value = np.where(published[known] > 0, raw[known], 0.0)
        _require(
            bool(np.isfinite(value).all()) and bool((value >= 0).all()),
            "La edición contiene un dividendo no válido",
        )
        factor = ratio[known]
        cash[position[known]] = value
        split[position[known]] = np.where((factor > 0) & (factor != 1), factor, 0.0)
        return cash, split


class SessionPrices:
    """Precios de ejecución de un mercado para las decisiones de ``[start, end)``."""

    def __init__(self, edition, market, start, end):
        _require(market in ("US", "CN"), "El mercado debe ser US o CN")
        self.root, self.market = Path(edition), market
        self.manifest = read_edition(self.root)
        self.edition_id = self.manifest["edition_id"]
        last = min(date.fromisoformat(end) + timedelta(days=15), date.fromisoformat(CUTOFF))
        clock = MarketClock(market, start, last.isoformat())
        self.days = np.array([day.isoformat() for day in clock.days], dtype="datetime64[D]")
        self.decisions = np.array([_microseconds(m) for m in clock.decisions], dtype=np.int64)
        self.opens = np.array([_microseconds(m) for m in clock.opens], dtype=np.int64)
        self.first = np.datetime64(start, "D")
        prefix = f"{market}/"
        self.available = {key for key in self.manifest["receipts"] if key.startswith(prefix)}
        self.assets = {}

    def _asset(self, key):
        if key not in self.assets:
            asset = None
            if key in self.available:
                instrument = None
                if self.market == "CN":
                    try:
                        instrument = china_a_share_instrument(key)
                    except ValueError:
                        instrument = False
                prices = _table(self.root, self.manifest, key, "prices.parquet", PRICE_SCHEMA)
                events = _table(self.root, self.manifest, key, "events.parquet", EVENT_SCHEMA)
                asset = _Asset(self.market, prices, events, instrument, self.first)
            self.assets[key] = asset
        return self.assets[key]

    def next_session(self, decisions):
        """Índice de calendario de la sesión siguiente a cada decisión, que debe ser exacta."""
        decisions = np.asarray(decisions, dtype=np.int64)
        index = np.minimum(np.searchsorted(self.decisions, decisions), len(self.decisions) - 1)
        _require(
            bool((self.decisions[index] == decisions).all()),
            "Una decisión no corresponde a un cierre del calendario del mercado",
        )
        _require(
            bool((index + 1 < len(self.days)).all()),
            "La sesión siguiente queda fuera de la edición, que termina en 2023",
        )
        return index + 1

    def executions(self, assets, decisions):
        """Precios y permisos de entrada de cada fila en la sesión siguiente a su decisión.

        Devuelve vectores alineados con las filas: ``open`` y ``close`` (NaN si no hay
        ejecución), ``reason`` (código de ``REASONS``), ``long_entry`` y ``short_entry``,
        ``blocked_long`` y ``blocked_short`` (apertura en la banda que impide entrar),
        ``exit_limit_long`` y ``exit_limit_short`` (el cierre toca la banda que bloquearía
        esa salida) y ``buy_tax`` y ``sell_tax``. El timbre se fija con la apertura: la
        apertura y el cierre de una sesión china caen en el mismo día de Pekín.
        """
        assets = np.asarray(assets, dtype=object)
        following = self.next_session(decisions)
        rows = len(assets)
        result = dict(
            open=np.full(rows, np.nan),
            close=np.full(rows, np.nan),
            reason=np.full(rows, NO_ROW, dtype=np.int8),
            long_entry=np.zeros(rows, dtype=bool),
            short_entry=np.zeros(rows, dtype=bool),
            blocked_long=np.zeros(rows, dtype=bool),
            blocked_short=np.zeros(rows, dtype=bool),
            exit_limit_long=np.zeros(rows, dtype=bool),
            exit_limit_short=np.zeros(rows, dtype=bool),
            buy_tax=np.zeros(rows),
            sell_tax=np.zeros(rows),
        )
        order = np.argsort(assets, kind="stable")
        keys, starts = np.unique(assets[order], return_index=True)
        bounds = np.r_[starts, rows]
        for key, begin, finish in zip(keys, bounds[:-1], bounds[1:], strict=True):
            selected = order[begin:finish]
            asset = self._asset(str(key))
            if asset is None or not len(asset.sessions):
                continue
            days = self.days[following[selected]]
            position = np.minimum(np.searchsorted(asset.sessions, days), len(asset.sessions) - 1)
            present = asset.sessions[position] == days
            local = np.full(len(selected), NO_ROW, dtype=np.int8)
            local[present] = asset.reason[position[present]]
            if asset.instrument is False:
                local[local == EXECUTABLE] = NO_BOARD
            if asset.instrument:
                self._china(asset, selected, following, position, local, result)
            ready = local == EXECUTABLE
            index = selected[ready]
            result["open"][index] = asset.open[position[ready]]
            result["close"][index] = asset.close[position[ready]]
            result["reason"][selected] = local
        executable = result["reason"] == EXECUTABLE
        result["long_entry"] = executable & ~result["blocked_long"]
        result["short_entry"] = executable & ~result["blocked_short"]
        return result

    def _china(self, asset, selected, following, position, local, result):
        """Bandas y timbre de las filas ejecutables de un activo A, con la regla de la cinta."""
        rows = np.flatnonzero(local == EXECUTABLE)
        for j in rows:
            row, where = selected[j], position[j]
            reference, at = asset.reference[where], int(self.opens[following[row]])
            if not np.isfinite(reference) or reference <= 0:
                local[j] = NO_REFERENCE
                continue
            result["buy_tax"][row] = asset.instrument.tax(at, "buy")
            result["sell_tax"][row] = asset.instrument.tax(at, "sell")
            limits = asset.instrument.limits(float(reference), at)
            if limits is None:
                continue
            upper, lower = limits
            opened, closed = asset.open[where], asset.close[where]
            result["blocked_long"][row] = opened >= upper
            result["blocked_short"][row] = opened <= lower
            result["exit_limit_long"][row] = closed <= lower
            result["exit_limit_short"][row] = closed >= upper

    def identity(self):
        return dict(
            market=self.market,
            edition_id=self.edition_id,
            price_basis="unadjusted_reconstructed",
            market_rules=RULES if self.market == "CN" else None,
            builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
