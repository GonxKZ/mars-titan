"""Puntuación de escritura M3: error maduro, anomalía de precios y relevancia publicada.

Los tres componentes se calculan por separado. La anomalía y la relevancia salen de las
entradas de la decisión, disponibles en su corte. El error necesita la etiqueta madura de la
predicción emitida y llega al banco después de la maduración. Las escalas se estiman solo con
el tramo de entrenamiento de la ventana y se congelan con su huella. Cada componente se lleva
a [0, 1] con `x / (x + m)`, donde `m` es su mediana de entrenamiento, así que la mediana
corresponde a 0,5. La frescura contable usa `m / (m + edad)` y vale 1 con edad cero. Los
pesos están fijados de antemano y no se ajustan en validación.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np

# Pesos iguales sobre componentes normalizados en la misma escala, fijados antes de ejecutar.
WEIGHTS = (1 / 3, 1 / 3, 1 / 3)
# Valor de un componente de relevancia desconocido: la mediana de entrenamiento, no cero.
NEUTRAL = 0.5
# Escala mínima de la dispersión de rendimientos, en unidades de log-rendimiento.
MAD_FLOOR = 1e-12
SAMPLE_CAPACITY = 65_536
SCALER_SEED = 19
_NEWS, _FUNDAMENTALS = 1, 3
_HEX = frozenset("0123456789abcdef")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _integer(value, minimum, maximum):
    return type(value) is int and minimum <= value <= maximum


def _positive(value):
    return type(value) is float and math.isfinite(value) and value > 0


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def declaration():
    """Definición que repite `configs/titans/mars-titan-extensions.json` sin desviarse."""
    return dict(
        recipe="episodic_m3_three_index_v1",
        score="w_e*e_norm + w_a*a_norm + w_r*r_norm",
        weights=list(WEIGHTS),
        normalization="x_over_x_plus_training_median",
        error="abs(label - issued_prediction) after maturity",
        anomaly="abs(last log return between observed closes) / median absolute deviation of "
        "the previous returns between observed closes of the 64-session window, 62 when every "
        "session is observed",
        relevance="max over known components: filing freshness m_f/(m_f+age) and news "
        "presence 1-p_news/2",
        missing_relevance="unknown and never zero: neutral 0.5 when no component is known",
        macro_relevance="not_included",
        diversity="not_included",
        thresholds="none: the 50/25/25 quotas bound the writes",
        scalers=dict(
            partition="train",
            sample="uniform_reservoir",
            sample_capacity=SAMPLE_CAPACITY,
            seed=SCALER_SEED,
        ),
        search="fixed_weights_two_learning_rate_cases",
    )


@dataclass(frozen=True)
class DecisionFeatures:
    """Rasgos de la decisión conocidos en su corte. None y False no significan cero."""

    anomaly: float
    filing_age_days: float | None
    news: bool

    def __post_init__(self):
        age = self.filing_age_days
        if (
            type(self.anomaly) is not float
            or not math.isfinite(self.anomaly)
            or self.anomaly < 0
            or (age is not None and (type(age) is not float or not math.isfinite(age) or age < 0))
            or type(self.news) is not bool
        ):
            raise ValueError("Los rasgos M3 necesitan anomalía, antigüedad y noticias válidas")


def decision_features(prices, presence, fundamentals):
    """Anomalía y relevancia de cada decisión con sus entradas de la edición con máscaras.

    `prices` es la ventana [n, 64, 5] de log-precios relativos a su primer cierre, o [n, 64, 6]
    con el bit de presencia desde la v3.1. La anomalía compara el último log-rendimiento entre
    cierres observados con la mediana de las desviaciones absolutas de los anteriores de la
    misma ventana, sin incluir el último. Una sesión ausente en todo el mercado no aporta cierre,
    así que el rendimiento que la atraviesa une los dos cierres observados que la rodean y la
    ventana tiene menos rendimientos. Cada fila se calcula con aritmética elemental sobre sus
    propios valores, así que no depende del lote.
    """
    prices, presence = np.asarray(prices), np.asarray(presence)
    fundamentals = np.asarray(fundamentals)
    size = len(prices)
    if (
        prices.ndim != 3
        or prices.shape[1] < 4
        or prices.shape[2] not in (5, 6)
        or presence.shape != (size, 5)
        or presence.dtype != np.bool_
        or fundamentals.ndim != 2
        or len(fundamentals) != size
        or fundamentals.shape[1] % 3
        or not presence[:, [0, 2]].all()
    ):
        raise ValueError("M3 necesita la ventana de precios, la presencia y los fundamentales")
    closes = prices[:, :, 3].astype(np.float64)
    observed_steps = (
        prices[:, :, 5] == 1 if prices.shape[2] == 6 else np.ones(closes.shape, dtype=bool)
    )
    full = observed_steps.all(axis=1)
    anomaly = np.empty(size, dtype=np.float64)
    returns = np.diff(closes[full], axis=1)
    past = returns[:, :-1]
    center = np.median(past, axis=1, keepdims=True)
    spread = np.median(np.abs(past - center), axis=1)
    anomaly[full] = np.abs(returns[:, -1]) / np.maximum(spread, MAD_FLOOR)
    for row in np.flatnonzero(~full):
        # Solo se encadenan cierres observados. El relleno de un hueco nunca entra.
        steps = np.diff(closes[row, observed_steps[row]])
        if len(steps) < 3 or not observed_steps[row, -1]:
            raise ValueError("M3 necesita al menos tres rendimientos y la sesión de la decisión")
        middle = np.median(steps[:-1])
        anomaly[row] = abs(steps[-1]) / max(np.median(np.abs(steps[:-1] - middle)), MAD_FLOOR)
    width = fundamentals.shape[1] // 3
    observed = fundamentals[:, width : 2 * width] == 1
    log_age = fundamentals[:, 2 * width :]
    result = []
    for row in range(size):
        age = None
        if presence[row, _FUNDAMENTALS]:
            if not observed[row].any():
                raise ValueError("Un bloque de fundamentales presente no tiene conceptos")
            # expm1 es creciente: la menor edad es la de la publicación más reciente.
            age = math.expm1(float(log_age[row][observed[row]].min()))
        result.append(DecisionFeatures(float(anomaly[row]), age, bool(presence[row, _NEWS])))
    return result


def batch_features(batch):
    """Rasgos M3 de una vista CPU validada, en el orden de sus flujos."""
    return decision_features(batch.inputs["prices"], batch.presence, batch.inputs["fundamentals"])


@dataclass(frozen=True)
class WriteScalers:
    """Medianas del tramo de entrenamiento de una ventana, congeladas después."""

    source_sha256: str
    dataset_sha256: str
    decision_start: int
    decision_end: int
    decisions: int
    labels: int
    filing_decisions: int
    news_decisions: int
    error_median: float
    anomaly_median: float
    filing_age_median: float | None
    news_share: float
    sample_capacity: int = SAMPLE_CAPACITY
    seed: int = SCALER_SEED

    def __post_init__(self):
        if not (
            _digest(self.source_sha256)
            and _digest(self.dataset_sha256)
            and _integer(self.decision_start, 1, 2**63 - 1)
            and _integer(self.decision_end, self.decision_start + 1, 2**63 - 1)
            and _integer(self.decisions, 1, 2**63 - 1)
            and _integer(self.labels, 1, 2**63 - 1)
            and _integer(self.filing_decisions, 0, self.decisions)
            and _integer(self.news_decisions, 0, self.decisions)
            and _positive(self.error_median)
            and _positive(self.anomaly_median)
            and (self.filing_age_median is None) == (self.filing_decisions == 0)
            and (self.filing_age_median is None or _positive(self.filing_age_median))
            and type(self.news_share) is float
            and self.news_share == self.news_decisions / self.decisions
            and _integer(self.sample_capacity, 1, 2**20)
            and _integer(self.seed, 0, 2**64 - 1)
        ):
            raise ValueError("Las escalas M3 no son medianas positivas de entrenamiento")

    def identity(self):
        return dict(schema_version=1, recipe="m3_training_scalers_v1", **asdict(self))

    def fingerprint(self):
        return hashlib.sha256(_canonical(self.identity()).encode()).hexdigest()

    @classmethod
    def from_fields(cls, values):
        """Reconstruir las escalas guardadas en una configuración y exigir el mismo JSON."""
        if not isinstance(values, dict):
            raise ValueError("Faltan las escalas M3")
        try:
            result = cls(**values)
        except TypeError as error:
            raise ValueError("Las escalas M3 no conservan sus campos") from error
        if _canonical(asdict(result)) != _canonical(values):
            raise ValueError("Las escalas M3 o sus tipos han cambiado")
        return result


def normalized(scalers, raw):
    """Componentes en [0, 1] y máscara de relevancia conocida (1 contable, 2 noticias).

    `raw` es `[abs_error, anomalía, antigüedad o None, noticias]`.
    """
    error, anomaly, age, news = raw
    e = error / (error + scalers.error_median)
    a = anomaly / (anomaly + scalers.anomaly_median)
    known, mask = [], 0
    if age is not None and scalers.filing_age_median is not None:
        known.append(scalers.filing_age_median / (scalers.filing_age_median + age))
        mask |= 1
    if news:
        known.append(1.0 - scalers.news_share / 2)
        mask |= 2
    return e, a, max(known) if known else NEUTRAL, mask


def score(weights, e, a, r):
    """Combinación lineal en orden fijo y FP64."""
    return weights[0] * e + weights[1] * a + weights[2] * r


def raw_components(error, features):
    """Componentes sin normalizar que conserva el banco por episodio retenido."""
    if (
        type(error) is not float
        or not math.isfinite(error)
        or type(features) is not DecisionFeatures
    ):
        raise ValueError("M3 necesita el error finito y los rasgos de cada candidato")
    return [abs(error), features.anomaly, features.filing_age_days, features.news]


class _Sample:
    """Muestra uniforme de tamaño fijo con el algoritmo R y un generador propio."""

    def __init__(self, capacity, seed, stream):
        self.values = np.empty(capacity, dtype=np.float64)
        self.size = self.seen = 0
        self._random = np.random.Generator(np.random.PCG64([seed, stream]))

    def extend(self, values):
        values = np.asarray(values, dtype=np.float64)
        free = min(len(self.values) - self.size, len(values))
        self.values[self.size : self.size + free] = values[:free]
        self.size += free
        self.seen += free
        rest = values[free:]
        if len(rest):
            # El elemento número n, contando desde cero, ocupa una plaza uniforme de [0, n].
            slots = self._random.integers(0, self.seen + 1 + np.arange(len(rest)))
            accepted = slots < len(self.values)
            for value, slot in zip(rest[accepted], slots[accepted], strict=True):
                self.values[slot] = value
            self.seen += len(rest)

    def median(self):
        return float(np.median(self.values[: self.size])) if self.size else None


def fit_write_scalers(source, *, block_rows=128, capacity=SAMPLE_CAPACITY, seed=SCALER_SEED):
    """Medianas de |etiqueta|, anomalía y antigüedad contable en el tramo de entrenamiento.

    Recorre una vez el índice de observaciones del tramo `train`. Solo usa decisiones dentro
    de su intervalo de decisión y etiquetas maduras de ese tramo. La escala del error es el
    error mediano del pronóstico nulo, que no depende de ningún modelo.
    """
    from mars_titan.models.titans.financial_inputs import validated_cpu_batch

    from .financial_observations import FinancialObservationSource

    if type(source) is not FinancialObservationSource or source.phase.partition != "train":
        raise ValueError("Las escalas M3 se estiman solo con el tramo de entrenamiento")
    phase, specification = source.phase, source.specification()
    if specification.input_policy != "historical_masked_2000_v1":
        raise ValueError("M3 necesita la edición histórica con máscaras")
    errors, anomalies, ages = (_Sample(capacity, seed, stream) for stream in range(3))
    decisions = labels = filings = news = 0
    for event in source.batched_events(block_rows=block_rows):
        values = [abs(value) for _, _, value in event.labels]
        errors.extend(values)
        labels += len(values)
        if not event.inputs or event.at < phase.decision_start:
            continue
        for raw in event.inputs:
            features = batch_features(validated_cpu_batch(raw, specification))
            anomalies.extend([row.anomaly for row in features])
            known = [row.filing_age_days for row in features if row.filing_age_days is not None]
            ages.extend(known)
            decisions += len(features)
            filings += len(known)
            news += sum(row.news for row in features)
    if not decisions or not labels:
        raise ValueError("El tramo de entrenamiento no tiene decisiones o etiquetas para M3")
    return WriteScalers(
        source_sha256=source.identity,
        dataset_sha256=source.dataset.identity,
        decision_start=phase.decision_start,
        decision_end=phase.decision_end,
        decisions=decisions,
        labels=labels,
        filing_decisions=filings,
        news_decisions=news,
        error_median=errors.median(),
        anomaly_median=anomalies.median(),
        filing_age_median=ages.median(),
        news_share=news / decisions,
        sample_capacity=capacity,
        seed=seed,
    )
