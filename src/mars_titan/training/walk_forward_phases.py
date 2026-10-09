"""Fases de una ventana walk-forward con el calentamiento común de los brazos con memoria.

Titans-MAC, MARS-TITAN, CM-v1 y la GRU episódica construyen sus fases con esta función,
así que todos observan la misma ventana de calentamiento. El ajuste empieza en el origen
de la ventana sin calentamiento. Validación, calibración y evaluación observan antes las
entradas de los `warmup_months` previos a su tramo, sin etiquetas ni predicciones
emitidas, y el calentamiento nunca empieza antes del origen del ajuste. Ninguna fase
termina después del final de su tramo, que es también el corte de maduración de sus
etiquetas en las vistas v2.
"""

from datetime import date

import numpy as np

from mars_titan.memory.financial_session import FinancialPhase

PREDICTED = ("validation", "calibration", "evaluation")
# Límite de la receta: hasta cinco años de entradas antes del tramo medido.
MAX_WARMUP_MONTHS = 60


def micros(day):
    """Microsegundos UTC del inicio de un día ISO."""
    return int(np.datetime64(day, "us").astype(np.int64))


def months_before(day, months):
    """Primer día del mes que queda `months` meses antes del de `day`."""
    start = date.fromisoformat(day)
    position = start.year * 12 + start.month - 1 - months
    return date(position // 12, position % 12 + 1, 1).isoformat()


def checked_warmup(months):
    """Meses de calentamiento declarados por una receta, entre 0 y 60."""
    if type(months) is not int or not 0 <= months <= MAX_WARMUP_MONTHS:
        raise ValueError("La receta necesita walk_forward.warmup_months entero entre 0 y 60")
    return months


def window_phases(fold, warmup_months):
    """Fases del ajuste y de los tres tramos medidos, con el calentamiento acotado."""
    checked_warmup(warmup_months)
    origin, train_end = (micros(day) for day in fold["train"])
    phases = {"train": FinancialPhase("train", origin, origin, train_end, train_end)}
    for name in PREDICTED:
        start, end = (micros(day) for day in fold[name])
        warmup = max(origin, micros(months_before(fold[name][0], warmup_months)))
        phases[name] = FinancialPhase(name, warmup, start, end, end)
    return phases
