"""Estratos de liquidez causales y peso de las filas extremas en las métricas cuadráticas.

Es un análisis secundario y descriptivo de la comparación walk-forward, declarado el 10 de
octubre de 2026 antes de cualquier resultado (#532). No sirve para elegir modelos ni cambia
la conclusión principal, que sigue siendo el MAE residual por sesión y el pinball de la
cabeza de cuantiles. El diagnóstico del residual (#16) mostró que las colas de US las dominan
acciones de menos de 1 USD o con muy pocos títulos negociados. Una métrica cuadrática queda
decidida por esas filas, así que el MSE solo se informa junto a su versión por estratos y al
peso de las filas más extremas. Ninguna fila se descarta.

Cada fila evaluada (activo, instante de decisión) se asigna con la edición sin ajustar, que
reconstruye los precios y volúmenes negociados. Los precios preparados están ajustados hacia
atrás por splits y dividendos posteriores, así que un umbral de 1 USD sobre ellos usaría el
futuro: un valor que más tarde se desdobla aparecería como acción de céntimos.

- Sesión publicada: la última del mercado cuya decisión (cierre más cinco minutos) es anterior
  o igual al instante de la predicción. Es la misma barrera que admite los precios del corpus.
- Precio: último cierre verificado con volumen positivo hasta esa sesión, llevado a su múltiplo
  de cotización.
- Volumen: mediana de las acciones negociadas en las últimas ``volume_sessions`` filas de la
  edición hasta esa sesión, todas verificadas. Una sesión rellenada con volumen cero cuenta
  como cero.
- Sin precio verificado o sin ventana completa, la fila queda en ``unclassified`` con su motivo.

Ninguna fila posterior a la sesión publicada interviene. La edición sí usó acciones
corporativas y una constante terminal posteriores para deshacer el ajuste del proveedor, y su
verificación por tramos decide qué filas se pueden clasificar. El valor recuperado es el precio
negociado que ya era público al emitir, pero la pertenencia a ``unclassified`` depende de esa
reconstrucción. Todos los brazos se evalúan sobre las mismas filas, así que esa selección no
favorece a ninguno.

Las filas extremas se miden sin quitarlas. Para cada brazo, semilla y vista se informa qué
parte de la suma de cuadrados (MSE por fila) y del MSE por sesión con la ponderación declarada
aportan las ``k`` filas de mayor contribución. En el MSE por sesión una fila pesa su error al
cuadrado dividido por las filas de su sesión y por las sesiones de la vista. Cada ventana
guarda las ``max(extremes)`` filas mayores de cada mercado y criterio. Como el peso de una
fila solo depende de su mercado dentro de una vista, las mayores de la unión son exactamente
las mayores de todas las ventanas.
"""

from datetime import UTC, date, datetime, timedelta
from math import fsum
from pathlib import Path

import numpy as np
import pyarrow as pa
from numpy.lib.stride_tricks import sliding_window_view

from mars_titan.data.temporal import MarketClock
from mars_titan.data.unadjusted_edition import CUTOFF, PRICE_SCHEMA
from mars_titan.simulation.reconstructed_tape import _snap, read_edition
from mars_titan.simulation.session_prices import _table

from .forecast_panel import weighted_session_mean

STATUS = "secondary_descriptive"
USE = "description_only_not_for_model_selection_or_primary_conclusion"
SOURCE = "unadjusted_reconstructed_verified_rows_up_to_the_last_published_session_v1"
PRICE_BASIS = "unadjusted_reconstructed"
STRATA = {
    "liquid": dict(low_price=False, thin_volume=False),
    "low_price": dict(low_price=True, thin_volume=False),
    "thin_volume": dict(low_price=False, thin_volume=True),
    "low_price_and_thin_volume": dict(low_price=True, thin_volume=True),
    "unclassified": None,
}
NAMES = tuple(STRATA)
UNCLASSIFIED = NAMES.index("unclassified")
# Motivos por los que una fila queda sin clasificar, en el orden de comprobación.
REASONS = (
    "classified",
    "asset_not_in_edition",
    "no_published_row",
    "no_verified_traded_close",
    "incomplete_volume_window",
)
QUADRATIC = ("mse",)
MULTIPLICITY = "bonferroni_over_strata_and_views_with_max_t_within_family"
FIELDS = {
    "status",
    "declared_at",
    "use",
    "source",
    "price_below",
    "volume_below",
    "volume_sessions",
    "strata",
    "metrics",
    "extremes",
    "listed_rows",
    "recalibrate",
    "min_rows",
    "min_sessions",
    "multiplicity",
}
MAX_THRESHOLD = 10_000_000
MAX_VOLUME_SESSIONS = 250
MAX_EXTREMES = 10_000
MAX_LISTED = 100
_CODES = np.empty((2, 2), dtype=np.int8)
for _code, _pattern in enumerate(STRATA.values()):
    if _pattern is not None:
        _CODES[int(_pattern["low_price"]), int(_pattern["thin_volume"])] = _code


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _iso_date(value):
    try:
        return isinstance(value, str) and date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def declaration(section, series_metrics, contrasted):
    """Validar la declaración previa sin abrir ningún dato.

    ``contrasted`` son las métricas de los contrastes principales. Cada métrica cuadrática que
    aparezca allí debe tener también su versión por estratos.
    """
    _require(
        isinstance(section, dict) and set(section) == FIELDS,
        "La declaración de estratos de liquidez no tiene exactamente sus campos",
    )
    _require(
        section["status"] == STATUS
        and section["use"] == USE
        and _iso_date(section["declared_at"])
        and section["source"] == SOURCE
        and section["multiplicity"] == MULTIPLICITY,
        "Los estratos de liquidez deben declararse como análisis secundario con su fuente",
    )
    _require(section["strata"] == STRATA, "Los estratos son los cuatro de precio y volumen")
    price = section["price_below"]
    _require(
        type(price) in (int, float) and 0 < price <= 1_000,
        "El umbral de precio debe ser positivo y fijado de antemano",
    )
    _require(
        _integer(section["volume_below"], 1, MAX_THRESHOLD)
        and _integer(section["volume_sessions"], 1, MAX_VOLUME_SESSIONS),
        "El umbral de volumen y su ventana deben ser enteros positivos acotados",
    )
    metrics = section["metrics"]
    quadratic = [metric for metric in contrasted if metric in QUADRATIC]
    _require(
        isinstance(metrics, list)
        and metrics
        and metrics[0] == "mae"
        and len(set(metrics)) == len(metrics)
        and set(metrics) <= set(series_metrics)
        and set(quadratic) <= set(metrics),
        "Los estratos de liquidez empiezan por el MAE e incluyen cada métrica cuadrática",
    )
    extremes = section["extremes"]
    _require(
        isinstance(extremes, list)
        and extremes
        and all(_integer(k, 1, MAX_EXTREMES) for k in extremes)
        and all(a < b for a, b in zip(extremes, extremes[1:], strict=False)),
        "Las filas extremas son recuentos enteros crecientes y acotados",
    )
    _require(
        _integer(section["listed_rows"], 1, min(MAX_LISTED, extremes[-1])),
        "Las filas listadas no pueden superar las extremas declaradas",
    )
    _require(section["recalibrate"] is False, "Ningún estrato puede volver a calibrar")
    for name in ("min_rows", "min_sessions"):
        _require(
            _integer(section[name], 1, MAX_THRESHOLD),
            f"El umbral {name} debe ser un entero positivo fijado de antemano",
        )
    return section


def _microseconds(moment):
    return int(moment.timestamp()) * 1_000_000 + moment.microsecond


def asset_codes(market, sessions, close, volume, verified, section):
    """Código y motivo de cada fila de un activo como sesión publicada de una decisión.

    La fila ``i`` solo depende de las filas ``0..i``: el último cierre verificado con volumen
    positivo y la mediana del volumen de las ``volume_sessions`` filas que terminan en ella.
    """
    _require(
        not len(sessions) or sessions.max() <= np.datetime64(CUTOFF),
        "La edición contiene filas posteriores a su corte",
    )
    _require(
        len(sessions) < 2 or bool((np.diff(sessions) > np.timedelta64(0, "D")).all()),
        "Las sesiones de un activo no están ordenadas",
    )
    closed, _ = _snap(market, sessions, close)
    traded = verified & (volume > 0)
    _require(
        bool(np.isfinite(volume[verified]).all())
        and bool((volume[verified] >= 0).all())
        and bool((closed[traded] > 0).all()),
        "La edición contiene precios o volúmenes no válidos en filas verificadas",
    )
    rows = len(sessions)
    last = np.maximum.accumulate(np.where(traded, np.arange(rows), -1))
    price = np.where(last >= 0, closed[np.maximum(last, 0)], np.nan)
    width = section["volume_sessions"]
    median = np.full(rows, np.nan)
    if rows >= width:
        counts = np.cumsum(np.r_[0, verified.astype(np.int64)])
        # Ventanas que terminan en las filas width - 1, ..., rows - 1.
        complete = counts[width:] - counts[:-width] == width
        windows = sliding_window_view(np.where(verified, volume, np.nan), width)
        median[width - 1 :][complete] = np.median(windows[complete], axis=1)
    low = price < section["price_below"]
    thin = median < section["volume_below"]
    reason = np.select(
        [~np.isfinite(price), ~np.isfinite(median)],
        [REASONS.index("no_verified_traded_close"), REASONS.index("incomplete_volume_window")],
        REASONS.index("classified"),
    ).astype(np.int8)
    code = np.where(
        reason == 0, _CODES[low.astype(np.intp), thin.astype(np.intp)], UNCLASSIFIED
    ).astype(np.int8)
    return code, reason


class EditionLiquidity:
    """Estrato de liquidez de cada fila con la edición sin ajustar y la declaración.

    Cada activo se lee una sola vez por instancia, con su huella, y se guardan sus días y
    los códigos de todas sus filas (seis bytes por fila, unos 115 MB para la edición entera).
    Una comparación con todas sus ventanas lee así la edición una vez y no una por ventana.
    """

    def __init__(self, edition, section):
        self.root = Path(edition)
        self.manifest = read_edition(self.root)
        self.section = section
        self._assets = {}

    def identity(self):
        return dict(edition_id=self.manifest["edition_id"], price_basis=PRICE_BASIS, source=SOURCE)

    def _published_days(self, market, moments):
        """Última sesión del mercado cuya decisión no es posterior a cada instante."""
        first = datetime.fromtimestamp(int(moments.min()) / 1e6, UTC).date()
        last = datetime.fromtimestamp(int(moments.max()) / 1e6, UTC).date()
        # Quince días antes bastan para encontrar una sesión incluso tras festivos largos.
        clock = MarketClock(market, (first - timedelta(days=15)).isoformat(), last.isoformat())
        decisions = np.array([_microseconds(value) for value in clock.decisions], dtype=np.int64)
        days = np.array([day.isoformat() for day in clock.days], dtype="datetime64[D]")
        index = np.searchsorted(decisions, moments, side="right") - 1
        _require(bool((index >= 0).all()), "Un instante precede a todas las sesiones del reloj")
        return days[index]

    def _asset(self, market, key):
        """Días del activo (desde 1970) con el código y el motivo de cada fila de la edición."""
        if key not in self._assets:
            prices = _table(self.root, self.manifest, key, "prices.parquet", PRICE_SCHEMA)
            sessions = np.array(prices["session"].to_pylist(), dtype="datetime64[D]")
            code, reason = asset_codes(
                market,
                sessions,
                prices["close"].to_numpy(),
                prices["volume"].to_numpy(),
                prices["verified"].to_numpy(zero_copy_only=False).astype(bool),
                self.section,
            )
            self._assets[key] = (sessions.astype(np.int32), code, reason)
        return self._assets[key]

    def assign(self, markets, assets, moments):
        """Código de estrato y motivo de cada fila, en el orden de entrada.

        ``markets`` y ``assets`` son textos por fila (``asset_id`` con el formato
        ``MERCADO/símbolo``) y ``moments`` los instantes de decisión en microsegundos UTC.
        """
        markets = np.asarray(markets, dtype=object)
        assets = np.asarray(assets, dtype=object)
        moments = np.asarray(moments, dtype=np.int64)
        _require(
            markets.shape == assets.shape == moments.shape and moments.ndim == 1,
            "Los estratos de liquidez necesitan mercado, activo e instante por fila",
        )
        codes = np.full(len(moments), UNCLASSIFIED, dtype=np.int8)
        reasons = np.full(len(moments), REASONS.index("asset_not_in_edition"), dtype=np.int8)
        for market in sorted(set(markets.tolist())):
            _require(market in ("US", "CN"), "El mercado debe ser US o CN")
            rows = np.flatnonzero(markets == market)
            days = self._published_days(market, moments[rows])
            order = rows[np.argsort(assets[rows], kind="stable")]
            keys, starts = np.unique(assets[order], return_index=True)
            for key, group in zip(keys, np.split(order, starts[1:]), strict=True):
                _require(str(key).startswith(f"{market}/"), "El activo no es de su mercado")
                if key not in self.manifest["receipts"]:
                    continue
                sessions, code, reason = self._asset(market, key)
                published = days[np.searchsorted(rows, group)].astype(np.int32)
                position = np.searchsorted(sessions, published, "right")
                position -= 1
                present = position >= 0
                reasons[group] = REASONS.index("no_published_row")
                codes[group[present]] = code[position[present]]
                reasons[group[present]] = reason[position[present]]
        return codes, reasons


def record(codes, reasons):
    """Filas por estrato y filas sin clasificar por motivo, de una ventana."""
    return dict(
        rows=len(codes),
        strata={name: int(np.count_nonzero(codes == k)) for k, name in enumerate(NAMES)},
        unclassified={
            name: int(np.count_nonzero(reasons == k))
            for k, name in enumerate(REASONS)
            if name != "classified"
        },
    )


def _top(values, rows, count):
    """Las ``count`` filas de ``rows`` con mayor valor. Los empates siguen el orden canónico."""
    return rows[np.argsort(-values[rows], kind="stable")[:count]]


def window_extremes(panel, row_codes, count):
    """Filas de mayor error cuadrático y de mayor peso en el MSE por sesión de cada mercado.

    ``row_codes`` sigue el orden canónico del panel. Los valores se calculan en float64 con
    la misma diferencia que el MSE de ``score_sessions``.
    """
    _require(
        isinstance(row_codes, np.ndarray) and row_codes.shape == (panel.rows,),
        "Los códigos de liquidez no siguen las filas del panel",
    )
    error = np.square(panel.prediction - panel.target)
    rows_in_session = panel.session_samples[panel.session]
    per_session = error / rows_in_session
    result = {}
    for code, market in enumerate(panel.markets):
        rows = np.flatnonzero(panel.market == code)
        if not len(rows):
            continue
        entry = {}
        for name, values in (("row", error), ("session", per_session)):
            chosen = _top(values, rows, count)
            entry[name] = dict(
                value=values[chosen],
                squared_error=error[chosen],
                session_rows=rows_in_session[chosen].astype(np.int64),
                prediction_at=panel.prediction_at[chosen].astype(np.int64),
                row_id=panel.row_id.take(pa.array(chosen)).to_pylist(),
                stratum=row_codes[chosen].astype(np.int8),
            )
        result[market] = entry
    return result


def _join(parts, market, name):
    """Candidatos de un mercado y criterio de todas las ventanas, en orden de ventana."""
    pieces = [part[market][name] for part in parts if market in part]
    if not pieces:
        return None
    return {
        key: (
            [value for piece in pieces for value in piece[key]]
            if key == "row_id"
            else np.concatenate([piece[key] for piece in pieces])
        )
        for key in pieces[0]
    }


def _asset_of(row_id, market):
    """``asset_id`` de una identidad de fila ``mercado/asset_id/instante``."""
    return row_id[len(market) + 1 : row_id.rindex("/")]


def _criterion(parts, scores, view_markets, criterion, weighting, declared):
    """Parte del total que aportan las k filas mayores con un criterio, y su listado."""
    count = declared["extremes"][-1]
    candidates, weights, labels = [], [], []
    sessions = {
        m: int(np.sum(scores.session_market == scores.markets.index(m))) for m in view_markets
    }
    present = [market for market in view_markets if sessions[market]]
    for market in present:
        joined = _join(parts, market, criterion)
        if joined is None:
            continue
        if criterion == "row":
            weight = 1.0
        elif weighting == "session":
            weight = 1.0 / sum(sessions.values())
        else:
            weight = 1.0 / (len(present) * sessions[market])
        candidates.append(joined)
        weights.append(np.full(len(joined["value"]), weight))
        labels += [market] * len(joined["value"])
    if criterion == "row":
        total = fsum((scores.samples * scores.mse).tolist())
    else:
        everywhere = np.ones(len(scores.mse), dtype=bool)
        total = weighted_session_mean(
            scores.mse, everywhere, scores.session_market, len(scores.markets), weighting
        )
    if not candidates or not total:
        reason = "No hay filas" if not candidates else "El total es cero"
        return dict(total=total, shares=None, reason=reason, strata=None, rows=[])
    values = {
        key: (
            [v for c in candidates for v in c[key]]
            if key == "row_id"
            else np.concatenate([c[key] for c in candidates])
        )
        for key in candidates[0]
    }
    contribution = values["value"] * np.concatenate(weights)
    order = np.argsort(-contribution, kind="stable")[:count]
    shares, strata = {}, {}
    for k in declared["extremes"]:
        top = order[:k]
        shares[str(k)] = fsum(contribution[top].tolist()) / total
        strata[str(k)] = {
            name: int(np.count_nonzero(values["stratum"][top] == code))
            for code, name in enumerate(NAMES)
        }
    listed = []
    for index in order[: declared["listed_rows"]]:
        market = labels[index]
        moment = datetime.fromtimestamp(int(values["prediction_at"][index]) / 1e6, UTC)
        listed.append(
            dict(
                market=market,
                asset_id=_asset_of(values["row_id"][index], market),
                prediction_at=moment.isoformat(),
                stratum=NAMES[int(values["stratum"][index])],
                squared_error=float(values["squared_error"][index]),
                session_rows=int(values["session_rows"][index]),
                share=float(contribution[index] / total),
            )
        )
    return dict(total=total, shares=shares, reason=None, strata=strata, rows=listed)


def extremes_report(parts, views, markets, weighting, declared):
    """Peso de las filas extremas de un brazo y semilla en cada vista.

    ``parts`` son los ``window_extremes`` de cada ventana y ``views`` las puntuaciones por
    sesión unidas de todas las ventanas, por vista (ámbito y mercados).
    """
    result = {}
    for view, scores in views.items():
        view_markets = list(markets) if view not in markets else [view]
        result[view] = {
            name: _criterion(parts, scores, view_markets, name, weighting, declared)
            for name in ("row", "session")
        }
    return result


def pending(declared, reason):
    """Sección declarada que no se ha podido calcular, con su motivo."""
    return dict(declaration=declared, status="pending", reason=reason)
