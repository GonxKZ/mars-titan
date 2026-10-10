"""Cinta histórica real desde la edición de precios negociados reconstruidos.

La edición de #379 no satisface por sí sola ``MarketTape``. Este módulo la convierte en
una cinta de un mercado con estas reglas, todas declaradas en su identidad:

- Solo se leen filas verificadas. Un activo con una fila sin verificar dentro de la cinta,
  sin cierre negociado verificado al empezar o cuya serie termina dentro de la cinta se
  excluye con su motivo. Nada se rellena con precios inventados.
- Una fila con volumen cero o una sesión del calendario sin fila es una sesión sin
  negociación. No tiene apertura ejecutable ni precio nuevo. La valoración conserva el
  último cierre negociado verificado. La última sesión sigue la misma regla cuando la serie
  continúa después de la cinta: el activo seguía cotizando y solo falta su fila.
- Los precios en la rejilla de cotización se llevan a su múltiplo exacto con la tolerancia
  de la edición. Una apertura fuera de rejilla no es ejecutable.
- Splits y dividendos proceden de los eventos del proveedor dentro de las filas
  verificadas. Si coinciden en la misma sesión, el importe por acción es ambiguo. Se abona
  el menor de los dos importes posibles y esa apertura no es ejecutable.
- Las predicciones proceden de recibos walk-forward y deben coincidir con sus huellas. El
  último dato de ajuste de cada sesión sale del recibo de su ventana.

La población está condicionada a seguir cotizando en marzo de 2025 y no hay retornos de
salida. Por eso cada activo de la cinta tiene cierre valorado en todas sus sesiones.
"""

import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from mars_titan.data.temporal import MarketClock
from mars_titan.data.unadjusted_edition import CUTOFF, EVENT_SCHEMA, PRICE_SCHEMA
from mars_titan.data.unadjusted_prices import DEFAULT_RELATIVE_TOLERANCE, price_increments
from mars_titan.environments.walk_forward_receipt import (
    WalkForwardWindow,
    prediction_fingerprint,
)

from .market import CURRENCIES, RECONSTRUCTED, RECONSTRUCTED_CONTRACT, MarketTape
from .portfolio import CorporateAction

EDITION_KIND = "unadjusted_price_edition"
# Origen declarado de toda cinta construida aquí, con precios reales de la edición.
SOURCE_KIND = "unadjusted_edition_tape"
MAX_FILE_BYTES = 64 * 1024**2
# Una última sesión sin fila se valora con el último cierre negociado si la serie tiene
# alguna fila posterior a la cinta. Si la serie termina dentro de la cinta haría falta un
# retorno de salida, que la edición no tiene, y el activo se excluye.
FINAL_SESSION_RULE = "missing_row_valued_at_last_traded_close_when_series_continues_v1"
REASONS = (
    "no_verified_rows",
    "unverified_rows_in_tape",
    "row_outside_calendar",
    "no_verified_traded_close_at_start",
    "series_ends_in_tape",
    "event_outside_calendar",
    "invalid_event",
)


class NoAdmittedAssets(ValueError):
    """Ningún activo pedido cumple las condiciones de la cinta. Conserva sus motivos."""

    def __init__(self, excluded):
        super().__init__("Ningún activo cumple las condiciones de la cinta reconstruida")
        self.excluded = dict(sorted(excluded.items()))


def _microseconds(moment):
    return int(moment.timestamp()) * 1_000_000 + moment.microsecond


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_edition(root):
    """Leer el manifiesto y comprobar su identidad sin abrir todavía los activos."""
    root = Path(root)
    path = root / "manifest.json"
    if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("La edición no tiene un manifiesto legible")
    manifest = json.loads(path.read_text())
    identity, receipts = manifest.get("identity"), manifest.get("receipts")
    if (
        manifest.get("kind") != EDITION_KIND
        or manifest.get("price_basis") != RECONSTRUCTED
        or manifest.get("corporate_actions_complete") is not False
        or not isinstance(identity, dict)
        or identity.get("policy", {}).get("cutoff") != CUTOFF
        or not isinstance(receipts, dict)
        or list(receipts) != sorted(receipts)
    ):
        raise ValueError("La edición no declara precios reconstruidos con su corte")
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    for artifacts in receipts.values():
        digest.update(json.dumps(artifacts, sort_keys=True).encode())
    if digest.hexdigest() != manifest.get("edition_id"):
        raise ValueError("El manifiesto de la edición no conserva su identidad")
    return manifest


def _artifact(root, manifest, key, name, schema):
    """Leer una tabla una sola vez y comprobar su huella antes de interpretarla."""
    expected = manifest["receipts"][key][name]
    path = root / "assets" / key / name
    if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"Falta un artefacto de la edición: {key}/{name}")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Un artefacto de la edición ha cambiado: {key}/{name}")
    table = pq.read_table(io.BytesIO(data))
    if not table.schema.equals(schema):
        raise ValueError(f"El artefacto no conserva el esquema de la edición: {key}/{name}")
    return table.to_pydict(), expected


def _snap(market, days, values):
    """Llevar a su múltiplo cada precio que la edición sitúa en la rejilla de cotización."""
    values = np.asarray(values, dtype=np.float64)
    ticks = price_increments(market, days, values)
    result, on_grid = values.copy(), np.zeros(len(values), dtype=bool)
    with np.errstate(invalid="ignore"):
        for column in range(ticks.shape[1]):
            # 1/256, 0,01 y 0,0001 tienen inversas enteras exactas en coma flotante.
            inverse = np.round(1 / ticks[:, column])
            exact = np.round(values * inverse) / inverse
            hit = ~on_grid & (np.abs(values - exact) <= DEFAULT_RELATIVE_TOLERANCE * values)
            result[hit], on_grid[hit] = exact[hit], True
    return result, on_grid


def _calendar(market, segments):
    """Sesiones cuya decisión cae en algún tramo, con apertura y decisión en UTC."""
    first = datetime.fromtimestamp(segments[0][0] / 1e6, UTC).date()
    last = datetime.fromtimestamp((segments[-1][1] - 1) / 1e6, UTC).date()
    clock = MarketClock(market, first.isoformat(), last.isoformat())
    decisions = np.array([_microseconds(value) for value in clock.decisions], dtype=np.int64)
    opens = np.array([_microseconds(value) for value in clock.opens], dtype=np.int64)
    owner = np.full(len(decisions), -1)
    for index, (start, end) in enumerate(segments):
        owner[(decisions >= start) & (decisions < end)] = index
    keep = owner >= 0
    days = np.array([day.isoformat() for day in clock.days], dtype="datetime64[D]")
    return days[keep], opens[keep], decisions[keep], owner[keep]


def _asset(market, prices, events, days):
    """Columnas de la cinta para un activo, o el motivo de su exclusión."""
    sessions = np.array(prices["session"], dtype="datetime64[D]")
    if len(sessions) and sessions.max() > np.datetime64(CUTOFF):
        raise ValueError("La edición contiene filas posteriores a su corte")
    if len(sessions) > 1 and (np.diff(sessions) <= np.timedelta64(0, "D")).any():
        raise ValueError("Las sesiones de un activo no están ordenadas")
    verified = np.array(prices["verified"], dtype=bool)
    volume = np.array(prices["volume"], dtype=np.float64)
    if not verified.any():
        return "no_verified_rows", None
    inside = (sessions >= days[0]) & (sessions <= days[-1])
    if not verified[inside].all():
        return "unverified_rows_in_tape", None
    if not np.isin(sessions[inside], days).all():
        return "row_outside_calendar", None
    # Último cierre negociado no posterior a la primera sesión, sin filas dudosas después.
    before = np.flatnonzero((sessions <= days[0]) & (volume > 0))
    first = int(np.searchsorted(sessions, days[0], side="right"))
    if not len(before) or not verified[before[-1] : first].all():
        return "no_verified_traded_close_at_start", None
    if not (sessions >= days[-1]).any():
        return "series_ends_in_tape", None
    reason, found = _events(events, days)
    if reason is not None:
        return reason, None
    position = np.searchsorted(sessions, days)
    present = position < len(sessions)
    present[present] = sessions[position[present]] == days[present]
    rows = position[present]
    traded = np.zeros(len(days), dtype=bool)
    traded[present] = volume[rows] > 0
    used = np.concatenate(([before[-1]], rows[volume[rows] > 0]))
    values = {
        name: np.array(prices[name], dtype=np.float64) for name in ("open", "high", "low", "close")
    }
    if (
        not np.isfinite(volume[rows]).all()
        or (volume[rows] < 0).any()
        or any(not (values[name][used] > 0).all() for name in values)
    ):
        raise ValueError("La edición contiene precios o volúmenes no válidos en filas verificadas")
    snapped, grid = {}, {}
    for name, column in values.items():
        snapped[name], grid[name] = _snap(market, sessions, column)
    index = rows[volume[rows] > 0]
    frame = np.full((len(days), 5), np.nan)
    frame[present, 4] = volume[rows]
    frame[traded, 0] = np.where(grid["open"][index], snapped["open"][index], np.nan)
    frame[traded, 1], frame[traded, 2] = snapped["high"][index], snapped["low"][index]
    # Sin negociación no hay precio nuevo: se arrastra el último cierre negociado.
    last = np.maximum.accumulate(np.where(traded, np.arange(len(days)), -1))
    traded_closes = np.zeros(len(days))
    traded_closes[traded] = snapped["close"][index]
    frame[:, 3] = np.where(
        last >= 0, traded_closes[np.maximum(last, 0)], snapped["close"][before[-1]]
    )
    for event_position, _, _, ambiguous in found:
        if ambiguous:
            frame[event_position, 0] = np.nan
    quiet = rows[volume[rows] == 0]
    counts = dict(
        traded_sessions=int(traded.sum()),
        zero_volume_sessions=len(quiet),
        zero_volume_moved_closes=int(
            (snapped["close"][quiet] != frame[present & ~traded, 3]).sum()
        ),
        missing_rows=int((~present).sum()),
        final_sessions_without_row=int(not present[-1]),
        off_grid_opens=int((~grid["open"][index]).sum()),
        off_grid_closes=int((~grid["close"][index]).sum()),
        ambiguous_same_day_events=sum(1 for event in found if event[3]),
    )
    return None, (frame, found, counts)


def _events(events, days):
    """Eventos posteriores a la primera sesión, como (posición, dividendo, split, ambiguo)."""
    sessions = np.array(events["session"], dtype="datetime64[D]")
    dividends = np.array(events["provider_dividend"], dtype=np.float64)
    raw = np.array(events["raw_dividend"], dtype=np.float64)
    splits = np.array(events["provider_split"], dtype=np.float64)
    selected = (sessions > days[0]) & (sessions <= days[-1])
    found = []
    for day, dividend, cash, split in zip(
        sessions[selected], dividends[selected], raw[selected], splits[selected], strict=True
    ):
        position = int(np.searchsorted(days, day))
        if position >= len(days) or days[position] != day:
            return "event_outside_calendar", None
        if (
            not np.isfinite([dividend, split]).all()
            or dividend < 0
            or split < 0
            or (dividend > 0 and not (np.isfinite(cash) and cash > 0))
        ):
            return "invalid_event", None
        cash = float(cash) if dividend > 0 else 0.0
        split = float(split) if split > 0 and split != 1 else 0.0
        if cash or split:
            found.append((position, cash, split, bool(cash and split)))
    return None, found


def _actions(asset, events, opens, closes, lag):
    """Dividendo antes que split en la misma apertura, como en el precio de referencia."""
    actions = []
    for position, cash, split, ambiguous in events:
        at = int(opens[position])
        day = f"{asset}/{at}"
        if cash:
            # Con split el mismo día, el importe por acción anterior es cash o cash * split.
            value = cash * min(1.0, split) if ambiguous else cash
            paid = position + lag
            pay_at = int(opens[paid]) if paid < len(opens) else int(closes[-1]) + 1
            actions.append(
                CorporateAction(f"{day}/dividend", asset, "dividend", at, value, pay_at, True)
            )
        if split:
            actions.append(CorporateAction(f"{day}/split", asset, "split", at, split, None, True))
    return actions


def _scores(windows, predictions, segment, market, decisions, owner, assets):
    """Situar cada predicción en su sesión tras comprobar la huella del recibo."""
    scores = np.full((len(decisions), len(assets)), np.nan)
    columns = {asset: index for index, asset in enumerate(assets)}
    dropped = 0
    for index, (window, values) in enumerate(zip(windows, predictions, strict=True)):
        if not isinstance(values, dict) or set(values) != {"prediction_at", "asset_id", "score"}:
            raise ValueError("Las predicciones de una ventana no conservan su formato")
        record = prediction_fingerprint(**values)
        if record != window.prediction_record(segment):
            raise ValueError("Las predicciones no coinciden con la huella de su recibo")
        local = np.flatnonzero(owner == index)
        times = np.asarray(values["prediction_at"], dtype=np.int64)
        where = np.minimum(np.searchsorted(decisions[local], times), len(local) - 1)
        if (decisions[local][where] != times).any():
            raise ValueError("Una predicción no corresponde a una decisión de su tramo")
        for row, asset, score in zip(
            local[where], values["asset_id"], values["score"], strict=True
        ):
            if not str(asset).startswith(f"{market}/"):
                raise ValueError("Una predicción pertenece a otro mercado")
            if asset in columns:
                scores[row, columns[asset]] = float(score)
            else:
                dropped += 1
    return scores, dropped


def build_reconstructed_tape(
    edition,
    windows,
    predictions,
    *,
    market,
    partition,
    dividend_payment_lag_sessions,
    segment="evaluation",
    symbols=None,
):
    """Construir la cinta de un mercado sobre los tramos ``segment`` de ventanas consecutivas.

    ``predictions`` contiene, por ventana, ``prediction_at``, ``asset_id`` y ``score``. El
    plazo de pago de los dividendos no tiene fuente en la edición y se declara como supuesto.
    Devuelve la cinta y un informe con los activos excluidos y sus motivos.
    """
    root = Path(edition)
    manifest = read_edition(root)
    windows = tuple(windows)
    if (
        market not in CURRENCIES
        or not windows
        or any(not isinstance(w, WalkForwardWindow) or w.market != market for w in windows)
        or len(predictions) != len(windows)
        or type(dividend_payment_lag_sessions) is not int
        or not 0 <= dividend_payment_lag_sessions <= 252
    ):
        raise ValueError("Las ventanas, el mercado o el plazo de pago no son válidos")
    segments = [window.segment(segment) for window in windows]
    if any(a[1] > b[0] for a, b in zip(segments, segments[1:], strict=False)):
        raise ValueError("Los tramos de las ventanas deben ser consecutivos y sin solapes")
    days, opens, decisions, owner = _calendar(market, segments)
    if len(days) < 2 or len(set(owner.tolist())) != len(windows):
        raise ValueError("Cada tramo necesita sesiones y la cinta al menos dos")
    prefix = f"{market}/"
    available = sorted(key[len(prefix) :] for key in manifest["receipts"] if key.startswith(prefix))
    chosen = available if symbols is None else sorted(set(symbols))
    if not chosen or not set(chosen) <= set(available):
        raise ValueError("Los activos pedidos no pertenecen a la edición")
    excluded, frames, actions, counts, artifacts = {}, [], [], {}, {}
    for symbol in chosen:
        key = f"{market}/{symbol}"
        prices, prices_sha = _artifact(root, manifest, key, "prices.parquet", PRICE_SCHEMA)
        events, events_sha = _artifact(root, manifest, key, "events.parquet", EVENT_SCHEMA)
        artifacts[key] = [prices_sha, events_sha]
        reason, result = _asset(market, prices, events, days)
        if reason is not None:
            excluded[key] = reason
            continue
        frame, events_, asset_counts = result
        frames.append((key, frame))
        actions.extend(_actions(key, events_, opens, decisions, dividend_payment_lag_sessions))
        for name, value in asset_counts.items():
            counts[name] = counts.get(name, 0) + value
    if not frames:
        raise NoAdmittedAssets(excluded)
    assets = [key for key, _ in frames]
    scores, dropped = _scores(windows, predictions, segment, market, decisions, owner, assets)
    exclusions = {reason: list(excluded.values()).count(reason) for reason in REASONS}
    walk_forward = [
        dict(
            receipt_sha256=window.sha256,
            fold=window.fold,
            partition=segment,
            start=start,
            end=end,
            labels_used_until=window.labels_used_until,
        )
        for window, (start, end) in zip(windows, segments, strict=True)
    ]
    audit = dict(
        price_basis=RECONSTRUCTED,
        market=market,
        edition_id=manifest["edition_id"],
        evidence_sha256=_canonical(dict(artifacts=artifacts, excluded=excluded)),
        walk_forward=walk_forward,
        prediction_fit_ends=[windows[i].labels_used_until for i in owner],
        assumptions=dict(dividend_payment_lag_sessions=dividend_payment_lag_sessions),
        **RECONSTRUCTED_CONTRACT,
    )
    code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source = dict(
        kind=SOURCE_KIND,
        builder_sha256=code,
        final_session=FINAL_SESSION_RULE,
        segment=segment,
        parents=[[window.fold, *window.parent] for window in windows],
        requested_assets=len(chosen),
        exclusions=exclusions,
        counts=dict(sorted(counts.items())),
        dropped_predictions=dropped,
    )
    tape = MarketTape(
        np.stack([frame for _, frame in frames], axis=1),
        decisions,
        assets,
        scores,
        domain="real",
        currency=CURRENCIES[market],
        partition=partition,
        prediction_times=decisions,
        open_times=opens,
        audit=audit,
        actions=sorted(actions, key=lambda a: (a.effective_at, a.asset, a.kind != "dividend")),
        parent_id="walk_forward:" + _canonical(source["parents"]),
        source_identity=source,
    )
    report = dict(
        tape_sha256=tape.sha256,
        assets=len(assets),
        excluded=dict(sorted(excluded.items())),
        exclusions=exclusions,
        counts=source["counts"],
        dropped_predictions=dropped,
        sessions=len(days),
        first_session=str(days[0]),
        last_session=str(days[-1]),
    )
    return tape, report
