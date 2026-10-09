"""Ventanas expansivas por calendario, con purga de etiquetas y test reservado."""

from datetime import date

import numpy as np

from mars_titan.training.selection import validate_selection

PARTITIONS = ("train", "validation", "calibration", "evaluation")
LABEL_INTERVAL_PURGE = "label_interval"
_DATES = ("train_start", "first_validation_start", "final_test_start", "final_test_end")
_MONTHS = ("validation_months", "calibration_months", "evaluation_months", "step_months")
_COMMON = set(_DATES + _MONTHS) | {
    "schema_version",
    "market",
    "minimum_train_months",
    "primary_metric",
    "seeds",
}
# La versión 1 conserva su margen fijo. La 2 purga por el intervalo de cada etiqueta y
# declara la regla de parada común antes de preparar las vistas.
_FIELDS = {1: _COMMON | {"gap_sessions"}, 2: _COMMON | {"purge", "selection"}}
_SELECTION = {"metric", "patience", "min_delta", "minimum_epochs", "max_epochs"}


def _add_months(day, months):
    position = day.year * 12 + day.month - 1 + months
    return date(position // 12, position % 12 + 1, 1)


def _timestamp(day):
    return int(np.datetime64(day, "us").astype(np.int64))


def _validate(config):
    version = config.get("schema_version") if isinstance(config, dict) else None
    if type(version) is not int or version not in _FIELDS or set(config) != _FIELDS[version]:
        raise ValueError("La configuración temporal tiene campos ausentes o desconocidos")
    if config["market"] not in {"US", "CN"}:
        raise ValueError("Versión o mercado no admitidos")
    if config["primary_metric"] != "session_mae":
        raise ValueError("El criterio primario de selección debe ser session_mae")
    dates = {key: date.fromisoformat(config[key]) for key in _DATES}
    if any(day.day != 1 or not 1900 <= day.year <= 2100 for day in dates.values()):
        raise ValueError("Los límites deben ser comienzos de mes entre 1900 y 2100")
    for key in _MONTHS + ("minimum_train_months",):
        if type(config[key]) is not int or not 1 <= config[key] <= 240:
            raise ValueError("Las duraciones deben ser meses enteros positivos y acotados")
    if version == 1 and config["step_months"] != config["evaluation_months"]:
        raise ValueError("Las ventanas de evaluación deben ser consecutivas y no solaparse")
    if version == 1 and (
        type(config["gap_sessions"]) is not int or not 0 <= config["gap_sessions"] <= 20
    ):
        raise ValueError("El margen debe contener entre cero y veinte sesiones")
    if version == 2:
        # Un paso múltiplo de la evaluación permite declarar un subconjunto de años sin solapes.
        if config["step_months"] % config["evaluation_months"]:
            raise ValueError("Las ventanas de evaluación no pueden solaparse")
        if config["purge"] != LABEL_INTERVAL_PURGE:
            raise ValueError("La purga debe derivarse del intervalo de cada etiqueta")
        stopping_rule(config)
    seeds = config["seeds"]
    if (
        not isinstance(seeds, list)
        or not 1 <= len(seeds) <= 10
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Las semillas deben ser enteros distintos y acotados")
    minimum = _add_months(dates["train_start"], config["minimum_train_months"])
    if (
        not minimum
        <= dates["first_validation_start"]
        < dates["final_test_start"]
        < dates["final_test_end"]
    ):
        raise ValueError("El orden temporal o la historia mínima de entrenamiento no son válidos")
    return dates


def stopping_rule(config):
    """Devolver la regla común de selección y parada declarada por un protocolo v2."""
    rule = config.get("selection") if isinstance(config, dict) else None
    if (
        not isinstance(rule, dict)
        or set(rule) != _SELECTION
        or type(rule["max_epochs"]) is not int
        or not 1 <= rule["max_epochs"] <= 1000
    ):
        raise ValueError("La regla de parada necesita métrica, paciencia, mínimo y máximo")
    validate_selection(
        {key: value for key, value in rule.items() if key != "max_epochs"},
        epochs=rule["max_epochs"],
    )
    return dict(rule)


def build_folds(config):
    """Fijar fechas sin observar etiquetas, resultados ni valores del test."""
    dates = _validate(config)
    validation = dates["first_validation_start"]
    folds = []
    while validation < dates["final_test_start"]:
        calibration = _add_months(validation, config["validation_months"])
        evaluation = _add_months(calibration, config["calibration_months"])
        end = _add_months(evaluation, config["evaluation_months"])
        if end > dates["final_test_start"]:
            break
        if len(folds) >= 128:
            raise ValueError("El protocolo supera el límite de 128 ventanas")
        folds.append(
            dict(
                id=f"fold-{len(folds):03d}",
                train=[dates["train_start"].isoformat(), validation.isoformat()],
                validation=[validation.isoformat(), calibration.isoformat()],
                calibration=[calibration.isoformat(), evaluation.isoformat()],
                evaluation=[evaluation.isoformat(), end.isoformat()],
            )
        )
        validation = _add_months(validation, config["step_months"])
    if not folds:
        raise ValueError("No cabe una ventana de evaluación antes del test reservado")
    return folds


class FoldPartitioner:
    """Preparar una vez los cortes y reutilizarlos en lotes de metadatos del panel."""

    def __init__(self, fold, clock, config):
        if fold not in build_folds(config) or clock.market != config["market"]:
            raise ValueError("La ventana o el calendario no corresponden al protocolo")
        self.sessions = np.array(
            [int(t.timestamp() * 1_000_000) for t in clock.decisions], dtype=np.int64
        )
        self.bounds = []
        for name in PARTITIONS:
            start, end = map(_timestamp, fold[name])
            position = int(np.searchsorted(self.sessions, end))
            # En la versión 2 la única purga es que la etiqueta madure antes del final del tramo.
            fixed = config["schema_version"] == 1 and name != "evaluation"
            gap = config["gap_sessions"] if fixed else 0
            cutoff = self.sessions[position - gap] if gap and position >= gap else end
            self.bounds.append((name, start, end, cutoff))
        self.test_start = _timestamp(config["final_test_start"])
        self.test_end = _timestamp(config["final_test_end"])

    def nominal_partitions(self, prediction_at):
        """Nombrar el tramo nominal de cada decisión para registrar la frontera de una purga."""
        prediction = np.asarray(prediction_at)
        names = np.full(len(prediction), "outside_window", dtype="U14")
        for name, start, end, _ in self.bounds:
            names[(prediction >= start) & (prediction < end)] = name
        return names

    def assign(self, prediction_at, available_at, label_available_at, *, eligible=None):
        """Asignar sin usar activos ni valores del objetivo para elegir los cortes."""
        prediction, available, maturity = (
            np.asarray(v) for v in (prediction_at, available_at, label_available_at)
        )
        if (
            prediction.ndim != 1
            or len(prediction) > 1_000_000
            or any(
                v.shape != prediction.shape or v.dtype != np.dtype("int64")
                for v in (prediction, available, maturity)
            )
        ):
            raise ValueError("Los metadatos deben ser vectores int64 de microsegundos UTC")
        if np.any(maturity <= prediction) or np.any(available < 0):
            raise ValueError("La etiqueta debe madurar después de la predicción")
        positions = np.searchsorted(self.sessions, prediction)
        if np.any(positions == len(self.sessions)) or not np.array_equal(
            self.sessions[positions], prediction
        ):
            raise ValueError("Las decisiones no pertenecen al calendario declarado")
        eligible = (
            np.ones(len(prediction), dtype=bool) if eligible is None else np.asarray(eligible)
        )
        if eligible.shape != prediction.shape or eligible.dtype != np.dtype(bool):
            raise ValueError("La admisión debe ser una máscara booleana alineada")
        partition = np.full(len(prediction), "excluded", dtype="U13")
        reason = np.full(len(prediction), "outside_window", dtype="U32")
        for name, start, end, cutoff in self.bounds:
            inside = (prediction >= start) & (prediction < end)
            gap = inside & (prediction >= cutoff)
            late = inside & (maturity >= end)
            accepted = inside & ~gap & ~late
            partition[accepted] = name
            reason[accepted] = "accepted"
            reason[late] = "label_crosses_boundary"
            reason[gap] = "session_gap"
        incomplete = ~eligible
        future = available > prediction
        partition[incomplete | future] = "excluded"
        reason[incomplete] = "incomplete_inputs"
        reason[future] = "future_inputs"
        reserved = (prediction >= self.test_start) & (prediction < self.test_end)
        partition[reserved] = "test_reserved"
        reason[reserved] = "final_test_reserved"
        return dict(partition=partition, reason=reason)


def assign_partitions(
    prediction_at, available_at, label_available_at, fold, clock, config, *, eligible=None
):
    """Consultar una ventana aislada. Para recorrer archivos, reutilizar FoldPartitioner."""
    return FoldPartitioner(fold, clock, config).assign(
        prediction_at, available_at, label_available_at, eligible=eligible
    )
