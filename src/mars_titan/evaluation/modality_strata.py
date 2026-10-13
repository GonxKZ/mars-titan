"""Estratos por presencia de noticias y fundamentales para la comparación walk-forward.

Es un análisis secundario y descriptivo, declarado el 9 de octubre de 2026 antes de
cualquier resultado. No sirve para elegir modelos ni cambia la conclusión principal.
En la edición desde 2000, precios, gráficos y macro están presentes en todas las
filas, así que los cuatro estratos solo distinguen noticias y fundamentales. Si una
fila evaluada no tuviera alguna de esas tres modalidades, la declaración dejaría de
describir la población y la sección queda no estimable con sus recuentos.

Los bits salen de la columna ``presence`` de ``samples.parquet`` para las filas que
cada vista asigna al tramo de evaluación, leídas con ``CorpusDataset``, que comprueba
las huellas de muestras y etiquetas. Cada estrato se puntúa con ``score_sessions`` sobre
el subconjunto de filas del panel ya validado. Los intervalos calibrados son los del
calibrador común de la ventana, ajustado una vez por mercado con todas las filas de
calibración. Ningún estrato lo vuelve a ajustar.
"""

import dataclasses
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mars_titan.data.input_policy import MODALITIES
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.forecast_scores import score_sessions
from mars_titan.training.corpus_inputs import CorpusDataset, _times

STATUS = "secondary_descriptive"
USE = "description_only_not_for_model_selection_or_primary_conclusion"
PRESENCE_SOURCE = "view_evaluation_labels_and_samples_presence_v1"
ALWAYS_PRESENT = ("prices", "charts", "macro")
STRATA = {
    "news_and_fundamentals": dict(news=True, fundamentals=True),
    "news_only": dict(news=True, fundamentals=False),
    "fundamentals_only": dict(news=False, fundamentals=True),
    "neither": dict(news=False, fundamentals=False),
}
MULTIPLICITY = "bonferroni_over_strata_and_views_with_max_t_within_family"
FIELDS = {
    "status",
    "declared_at",
    "use",
    "presence_source",
    "always_present",
    "strata",
    "focus",
    "metrics",
    "recalibrate",
    "min_rows",
    "min_sessions",
    "multiplicity",
}
MAX_SAMPLE_ROWS = 1_000_000
MAX_THRESHOLD = 10_000_000
_NEWS, _FUNDAMENTALS = MODALITIES.index("news"), MODALITIES.index("fundamentals")
# Código de cada fila: posición de su patrón (noticias, fundamentales) en STRATA.
_CODES = np.empty((2, 2), dtype=np.int8)
for _code, _pattern in enumerate(STRATA.values()):
    _CODES[int(_pattern["news"]), int(_pattern["fundamentals"])] = _code


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _iso_date(value):
    try:
        return isinstance(value, str) and date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def declaration(section, series_metrics):
    """Validar la declaración previa de los estratos sin abrir ningún dato."""
    _require(
        isinstance(section, dict) and set(section) == FIELDS,
        "La declaración de estratos no tiene exactamente sus campos",
    )
    metrics = section["metrics"]
    _require(
        section["status"] == STATUS
        and section["use"] == USE
        and _iso_date(section["declared_at"])
        and section["presence_source"] == PRESENCE_SOURCE
        and section["always_present"] == list(ALWAYS_PRESENT)
        and section["multiplicity"] == MULTIPLICITY,
        "Los estratos deben declararse como análisis secundario y descriptivo con su fuente",
    )
    _require(
        section["strata"] == STRATA and section["focus"] in STRATA,
        "Los estratos son los cuatro patrones de noticias y fundamentales",
    )
    _require(
        isinstance(metrics, list)
        and metrics
        and metrics[0] == "mae"
        and len(set(metrics)) == len(metrics)
        and set(metrics) <= set(series_metrics),
        "Los contrastes por estrato deben empezar por el MAE y usar métricas por sesión",
    )
    _require(section["recalibrate"] is False, "Ningún estrato puede volver a calibrar")
    for name in ("min_rows", "min_sessions"):
        value = section[name]
        _require(
            type(value) is int and 1 <= value <= MAX_THRESHOLD,
            f"El umbral {name} debe ser un entero positivo fijado de antemano",
        )
    return section


def _bits(table):
    """Bits de presencia de cada muestra con la misma comprobación de forma del corpus."""
    column = table["presence"].combine_chunks()
    _require(
        (pa.types.is_list(column.type) or pa.types.is_fixed_size_list(column.type))
        and pa.types.is_boolean(column.type.value_type)
        and column.null_count == 0
        and column.flatten().null_count == 0
        and bool((pc.list_value_length(column).to_numpy() == len(MODALITIES)).all()),
        "La presencia necesita cinco booleanos por muestra",
    )
    bits = column.flatten().to_numpy(zero_copy_only=False).reshape(len(table), len(MODALITIES))
    counts = table["news_count"]
    _require(
        pa.types.is_integer(counts.type) and counts.null_count == 0,
        "El recuento de noticias debe ser un entero conocido",
    )
    events = counts.to_numpy()
    _require(
        bool((events >= 0).all()) and np.array_equal(events > 0, bits[:, _NEWS]),
        "La presencia de noticias no coincide con sus eventos admitidos",
    )
    return bits


def view_presence(path, input_policy, expected_sha256):
    """Filas de evaluación de una vista, con su objetivo y la presencia de cada muestra.

    Devuelve una tabla con ``asset_id``, ``market``, ``prediction_at`` y ``target``, en el
    formato de las predicciones, y una matriz booleana de cinco columnas por fila.
    """
    dataset = CorpusDataset(Path(path), cache_bytes=0, input_policy=input_policy)
    _require(
        dataset.masked and dataset.temporals,
        "La presencia necesita una vista walk-forward con máscaras",
    )
    _require(dataset.identity == expected_sha256, "La vista cambió antes de leer la presencia")
    columns = dict(asset_id=[], market=[], prediction_at=[], target=[])
    presence = []
    for asset in sorted(dataset.assets, key=lambda row: (row["market"], row["symbol"])):
        if not asset["counts"]["evaluation"]:
            continue
        with pq.ParquetFile(dataset._file(asset, "samples")) as file:
            rows = file.metadata.num_rows
            _require(1 <= rows <= MAX_SAMPLE_ROWS, "Las muestras superan el presupuesto de filas")
            table = file.read(
                columns=["prediction_at", "presence", "news_count"], use_threads=False
            )
        positions, moments, targets, _ = dataset._labels(asset, "evaluation", rows)
        _require(
            np.array_equal(_times(table["prediction_at"])[positions], moments),
            "Las etiquetas de evaluación no corresponden a las decisiones de sus muestras",
        )
        presence.append(_bits(table)[positions])
        columns["asset_id"] += [f"{asset['market']}/{asset['symbol']}"] * len(positions)
        columns["market"] += [asset["market"]] * len(positions)
        columns["prediction_at"].append(moments)
        columns["target"].append(targets)
    _require(presence, "La vista no tiene filas de evaluación")
    bits = np.concatenate(presence)
    moments = pa.array(np.concatenate(columns["prediction_at"]), type=pa.int64())
    table = pa.table(
        dict(
            asset_id=pa.array(columns["asset_id"], type=pa.string()),
            market=pa.array(columns["market"], type=pa.string()),
            prediction_at=moments.cast(pa.timestamp("us", tz="UTC")),
            target=np.concatenate(columns["target"]).astype(np.float64),
        )
    )
    bits.setflags(write=False)
    return table, bits


def incomplete_rows(bits):
    """Filas a las que les falta alguna modalidad que la declaración da por presente."""
    required = [MODALITIES.index(name) for name in ALWAYS_PRESENT]
    return int(np.count_nonzero(~bits[:, required].all(axis=1)))


def codes(bits):
    """Índice del estrato de cada fila según la presencia de noticias y fundamentales."""
    return _CODES[bits[:, _NEWS].astype(np.intp), bits[:, _FUNDAMENTALS].astype(np.intp)]


def _subset(panel, mask):
    """Panel con las filas de la máscara, que ya están en orden canónico y lo conservan."""
    return ForecastPanel.from_columns(
        panel.row_id.filter(pa.array(mask)),
        np.asarray(panel.markets)[panel.market[mask]],
        panel.prediction_at[mask],
        panel.target[mask],
        panel.prediction[mask],
        quantiles=None if panel.quantiles is None else panel.quantiles[mask],
        levels=panel.levels,
        markets=panel.markets,
    )


def score_strata(panel, row_codes, *, rank_ic_min_assets, calibrated=None, names=tuple(STRATA)):
    """Puntuaciones por sesión de cada estrato, en bruto y con los cuantiles ya calibrados.

    ``row_codes`` sigue el orden canónico del panel y cada código es la posición de su
    estrato en ``names``. ``calibrated`` son los cuantiles del panel completo corregidos por
    el calibrador común de la ventana. Un estrato sin filas en la ventana queda sin
    puntuación. Los estratos de liquidez usan la misma función con sus nombres.
    """
    _require(
        isinstance(row_codes, np.ndarray) and row_codes.shape == (panel.rows,),
        "Los códigos de presencia no siguen las filas del panel",
    )
    result = {}
    for code, name in enumerate(names):
        mask = row_codes == code
        entry = dict(raw=None, calibrated=None)
        if mask.any():
            part = _subset(panel, mask)
            entry["raw"] = score_sessions(part, rank_ic_min_assets=rank_ic_min_assets)
            if calibrated is not None:
                quantiles = np.ascontiguousarray(calibrated[mask])
                quantiles.setflags(write=False)
                entry["calibrated"] = score_sessions(
                    dataclasses.replace(part, quantiles=quantiles),
                    rank_ic_min_assets=rank_ic_min_assets,
                )
        result[name] = entry
    return result


def estimability(rows, sessions, *, min_rows, min_sessions):
    """Decidir con los umbrales declarados si una celda se informa o por qué no."""
    if rows == 0:
        return False, "El estrato no tiene filas en esta celda"
    if rows < min_rows or sessions < min_sessions:
        return False, (
            f"{rows} filas y {sessions} sesiones, por debajo del mínimo declarado de "
            f"{min_rows} filas y {min_sessions} sesiones"
        )
    return True, None


def subset_views(scores, markets, names):
    """Ámbito y mercados de un subconjunto de filas. Un mercado sin sesiones queda vacío."""
    if scores is None:
        return dict.fromkeys(names)
    views = {names[0]: scores}
    if len(markets) > 1:
        for code, market in enumerate(scores.markets):
            mask = scores.session_market == code
            views[market] = scores.select_sessions(mask, label=market) if mask.any() else None
    return views


def cell(scores, thresholds, market=None):
    """Filas, sesiones y estimabilidad de un subconjunto en el ámbito o en un mercado."""
    rows, sessions = 0, 0
    if scores is not None and market is None:
        rows, sessions = int(scores.samples.sum()), len(scores.samples)
    elif scores is not None:
        mask = scores.session_market == scores.markets.index(market)
        rows, sessions = int(scores.samples[mask].sum()), int(mask.sum())
    estimable, reason = estimability(rows, sessions, **thresholds)
    return dict(rows=rows, sessions=sessions, estimable=estimable, reason=reason)


def adjusted_confidence(confidence, cells):
    """Bonferroni entre las celdas de estrato y ámbito sobre el nivel de cada familia."""
    _require(type(cells) is int and cells >= 1, "El número de celdas debe ser positivo")
    return 1 - (1 - confidence) / cells
