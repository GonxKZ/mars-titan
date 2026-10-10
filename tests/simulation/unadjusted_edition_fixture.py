"""Edición sintética con el formato real de la edición sin ajustar, identificada como fixture.

Escribe ``prices.parquet``, ``events.parquet`` y ``receipt.json`` por activo y un manifiesto
cuya identidad se calcula como en ``build_edition``. Los precios están en céntimos exactos y
se almacenan con un residuo relativo de 4e-7, del orden del que deja la reconstrucción.
"""

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.temporal import MarketClock
from mars_titan.data.unadjusted_edition import EVENT_SCHEMA, PRICE_SCHEMA
from mars_titan.data.unadjusted_prices import price_grid
from mars_titan.simulation.listing_status import CUTOFF, TABLE_KIND, read_listing_status
from tests.environments.walk_forward_fixture import window

RESIDUAL = 4e-7
HISTORY_START = "2022-09-01"


@dataclass
class Asset:
    symbol: str
    base: float = 20.0
    start: int | None = None
    end: int | None = None
    # None verifica todas las filas. Un entero verifica desde esa posición de 2023.
    verified_from: int | None = None
    unverified: tuple = ()
    never_verified: bool = False
    zero_volume: tuple = ()
    missing: tuple = ()
    off_grid_open: tuple = ()
    # (posición en 2023, dividendo por acción negociada, razón de split o 0)
    events: tuple = ()
    overrides: dict = field(default_factory=dict)
    leading_zero_volume: bool = False


def tape_days(market):
    clock = MarketClock(market, "2023-01-01", "2023-12-31")
    return [day.isoformat() for day in clock.days]


def history_days(market):
    clock = MarketClock(market, HISTORY_START, "2023-12-31")
    return [day.isoformat() for day in clock.days]


def _rows(market, spec):
    days = history_days(market)
    year = tape_days(market)
    position = {day: index for index, day in enumerate(year)}
    first = year[spec.start] if spec.start is not None else days[0]
    last = year[spec.end] if spec.end is not None else days[-1]
    missing = {year[i] for i in spec.missing}
    selected = [d for d in days if first <= d <= last and d not in missing]
    splits = [(year[i], ratio) for i, _, ratio in spec.events if ratio]
    rows, previous = [], None
    for k, day in enumerate(selected):
        close = round(spec.base + 0.01 * ((k * 37) % 50), 2)
        opened = round((previous if previous is not None else close) + 0.01 * ((k * 13) % 7 - 3), 2)
        # La serie se genera en acciones posteriores a cada split y se escala hacia atrás.
        factor = float(np.prod([ratio for when, ratio in splits if day < when] or [1.0]))
        values = dict(
            open=round(opened * factor, 2), close=round(close * factor, 2), volume=1e6 + 1000 * k
        )
        index = position.get(day)
        if index in spec.zero_volume or (spec.leading_zero_volume and index == 0):
            # El proveedor rellena la suspensión con volumen cero y un cierre que puede moverse.
            values = dict(open=rows[-1]["close"] + 0.37, close=rows[-1]["close"] + 0.37, volume=0.0)
        values.update(spec.overrides.get(index, {}))
        values["high"] = max(values["open"], values["close"]) + 0.05
        values["low"] = min(values["open"], values["close"]) - 0.05
        rows.append(dict(session=day, **values))
        previous = values["close"] / factor
    for row in rows:
        for name in ("open", "high", "low", "close"):
            row[name] = row[name] * (1 + RESIDUAL)
        if position.get(row["session"]) in spec.off_grid_open:
            row["open"] = row["open"] + 0.003
    return rows, year


def _verified(spec, sessions, year):
    if spec.never_verified:
        return np.zeros(len(sessions), dtype=bool)
    verified = np.ones(len(sessions), dtype=bool)
    if spec.verified_from is not None:
        verified &= np.array(sessions) >= year[spec.verified_from]
    for index in spec.unverified:
        verified &= np.array(sessions) != year[index]
    return verified


def _write_asset(folder, market, spec):
    rows, year = _rows(market, spec)
    sessions = [row["session"] for row in rows]
    close = np.array([row["close"] for row in rows])
    on_grid, _ = price_grid(market, np.array(sessions, dtype="datetime64[D]"), close)
    prices = pa.Table.from_arrays(
        [
            pa.array(sessions, pa.string()),
            *(pa.array([row[name] for row in rows]) for name in ("open", "high", "low", "close")),
            pa.array([row["volume"] for row in rows]),
            pa.array(np.ones(len(rows))),
            pa.array(np.zeros(len(rows))),
            pa.array(on_grid),
            pa.array(_verified(spec, sessions, year)),
        ],
        schema=PRICE_SCHEMA,
    )
    events = sorted(spec.events)
    later = [np.prod([r for j, _, r in events if j > i and r] or [1.0]) for i, _, _ in events]
    table = pa.Table.from_arrays(
        [
            pa.array([year[i] for i, _, _ in events], pa.string()),
            pa.array([cash / k for (_, cash, _), k in zip(events, later, strict=True)]),
            pa.array([float(ratio) for _, _, ratio in events]),
            pa.array([float(cash) for _, cash, _ in events]),
        ],
        schema=EVENT_SCHEMA,
    )
    folder.mkdir(parents=True)
    pq.write_table(prices, folder / "prices.parquet")
    pq.write_table(table, folder / "events.parquet")
    artifacts = {
        name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
        for name in ("prices.parquet", "events.parquet")
    }
    receipt = dict(market=market, symbol=spec.symbol, fixture=True, artifacts=artifacts)
    (folder / "receipt.json").write_text(json.dumps(receipt))
    return artifacts


def write_edition(root, assets, *, basis="unadjusted_reconstructed"):
    """Escribir la edición sintética de ``assets``, un dict mercado -> lista de Asset."""
    identity = dict(schema_version=1, policy=dict(cutoff="2023-12-31"), code={"fixture": "1"})
    receipts = {}
    for market in sorted(assets):
        for spec in sorted(assets[market], key=lambda item: item.symbol):
            folder = root / "assets" / market / spec.symbol
            receipts[f"{market}/{spec.symbol}"] = _write_asset(folder, market, spec)
    receipts = dict(sorted(receipts.items()))
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    for artifacts in receipts.values():
        digest.update(json.dumps(artifacts, sort_keys=True).encode())
    manifest = dict(
        kind="unadjusted_price_edition",
        edition_id=digest.hexdigest(),
        identity=identity,
        price_basis=basis,
        corporate_actions_complete=False,
        receipts=receipts,
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def predictions(market, symbols, *, score=None, start="2023-01-01", end="2023-12-31"):
    """Puntuaciones sintéticas en cada decisión, no producidas por ningún modelo."""
    clock = MarketClock(market, start, end)
    times, assets, values = [], [], []
    for k, moment in enumerate(clock.decisions):
        at = int(moment.timestamp()) * 1_000_000
        for i, symbol in enumerate(sorted(symbols)):
            times.append(at)
            assets.append(f"{market}/{symbol}")
            values.append(0.01 * ((i + k) % 5 - 1) if score is None else score(k, i))
    return dict(
        prediction_at=np.array(times, dtype=np.int64), asset_id=assets, score=np.array(values)
    )


def evaluation_window(market, values, **options):
    return window(market, values=values, **options)


def listing_status(root, *, china=None, exits=None, name="listing-status.json"):
    """Tabla del estado de cotización de la edición sintética, identificada como fixture.

    Cada acción A de la edición recibe una admisión anterior a sus filas, sin exención inicial,
    sin tramos ST o S y sin días sin límite, salvo lo que fijen las entradas de `china`. `exits` añade salidas con precio, cuya fuente
    es la propia fixture. Devuelve la tabla y la huella de sus bytes, como `read_listing_status`.
    """
    manifest = json.loads((root / "manifest.json").read_text())
    empty = dict(
        listed_on="2000-01-04",
        limit_free_until=None,
        special_treatment=[],
        share_reform_pending=[],
        limit_free_days=[],
    )
    entries = {key: dict(empty) for key in manifest["receipts"] if key.startswith("CN/")}
    entries.update({key: {**empty, **value} for key, value in (china or {}).items()})
    digest = hashlib.sha256(b"listing-status-fixture").hexdigest()
    table = dict(
        kind=TABLE_KIND,
        schema_version=1,
        cutoff=CUTOFF,
        sources={"fixture": dict(sha256=digest, fixture=True)},
        china=dict(sorted(entries.items())),
        exits={key: dict(value, source="fixture") for key, value in (exits or {}).items()},
    )
    path = root.parent / name
    path.write_text(json.dumps(table, indent=1, sort_keys=True))
    return read_listing_status(path)
