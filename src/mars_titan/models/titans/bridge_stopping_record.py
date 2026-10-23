"""Validar decisiones de parada archivadas mediante configuraciones, sin ejecutar lectores."""

from .bridge_recurrence import BridgeRecurrenceConfig
from .config import bounded_integer

STOPPING_FIELDS = ("selected_steps", "evaluated_steps", "criterion_met")


def verify_stopping_record(value, recurrence, *, episodes, mode, active_views):
    """Comprobar tipos y trabajo declarado, sin autenticar el residual numérico."""
    if type(recurrence) is not BridgeRecurrenceConfig or recurrence.stopping is None:
        raise ValueError("Los metadatos de parada necesitan una recurrencia adaptativa")
    recurrence.__post_init__()
    bounded_integer(episodes, "episodios de la parada", 0, 1024)
    if (
        type(mode) is not str
        or mode not in ("bank", "no_bank")
        or type(active_views) not in (tuple, list)
        or not 1 <= len(active_views) <= 8
        or any(type(active) is not bool for active in active_views)
    ):
        raise ValueError("La parada necesita modo y disponibilidad regional explícitos")
    if (
        type(value) is not dict
        or set(value) != set(STOPPING_FIELDS)
        or type(value["selected_steps"]) is not int
        or type(value["evaluated_steps"]) is not int
        or type(value["criterion_met"]) is not bool
    ):
        raise ValueError("La parada regional necesita pasos enteros y un criterio booleano")
    selected, evaluated, met = (value[name] for name in STOPPING_FIELDS)
    if not episodes:
        if (selected, evaluated, met) != (0, 0, False):
            raise ValueError("Una lectura sin episodios no debe inventar trabajo adaptativo")
        return selected, evaluated
    if mode == "no_bank":
        raise ValueError("El control sin banco no admite una parada con episodios")
    maximum = recurrence.steps
    expected = maximum if recurrence.stopping.backend == "adaptive_reference_full" else selected
    if (
        selected not in (1, 2, 4)
        or selected > maximum
        or evaluated != expected
        or (not met and selected != maximum)
    ):
        raise ValueError("La parada regional no conserva sus prefijos o el trabajo del backend")
    if not any(active_views) and (selected != 1 or not met):
        raise ValueError("Un pool poblado sin lectores presentes debe elegir J1 con residual cero")
    return selected, evaluated
