"""Admitir las fases según sus observaciones y maduraciones, sin ejecutar modelos."""

import numpy as np

MAX_TRACE_ROWS = 131072
MAX_PENDING_TRACES = 32768
MAX_DECISION_ROWS = 256


def event_capacity(tables, *, decision_start):
    """Máximo pendiente del índice real, incluyendo emisiones que nunca reciben etiqueta.

    El índice valida las identidades de las observaciones. Aquí se agregan sus
    cantidades por instante. Emitir antes de madurar conserva el orden del runner.
    """
    events = {}
    for table in tables:
        times = table["event_at"].to_numpy()
        kinds = table["kind"].to_numpy()
        decisions = table["sample_at"].to_numpy()
        if not np.isin(kinds, (0, 1, 2)).all() or np.any((kinds == 1) & (times <= decisions)):
            raise ValueError("El índice contiene eventos o maduraciones incompatibles")
        for kind in (0, 1):
            selected = (kinds == kind) & (decisions >= decision_start)
            stamps, counts = np.unique(times[selected], return_counts=True)
            for stamp, count in zip(stamps, counts, strict=True):
                values = events.setdefault(int(stamp), [0, 0])
                values[kind] += int(count)
    pending = maximum = predictions = labels = largest = 0
    for observed, matured in (events[stamp] for stamp in sorted(events)):
        pending += observed
        maximum = max(maximum, pending)
        largest = max(largest, observed)
        pending -= matured
        if pending < 0:
            raise ValueError("Hay más maduraciones que predicciones emitidas")
        predictions += observed
        labels += matured
    return dict(
        observations=predictions,
        mature_labels=labels,
        max_pending=maximum,
        unresolved_at_close=pending,
        max_decision_rows=largest,
        order="emit_then_mature",
        values_used=False,
    )


def require_capacity(record, *, collection, max_scored_rows=MAX_TRACE_ROWS):
    if (
        record["max_decision_rows"] > MAX_DECISION_ROWS
        or (max_scored_rows is not None and record["mature_labels"] > max_scored_rows)
        or (collection and record["max_pending"] > MAX_PENDING_TRACES)
    ):
        raise ValueError("La fase supera la capacidad declarada. No se descartan filas")


def dataset_capacity(dataset, phases, *, collection=(), max_scored_rows=MAX_TRACE_ROWS):
    from mars_titan.memory.financial_observations import _records
    from mars_titan.memory.financial_session import FinancialPhase

    result = {}
    for name, values in phases.items():
        phase = FinancialPhase(**values) if isinstance(values, dict) else values
        measured = event_capacity(_records(dataset, phase), decision_start=phase.decision_start)
        try:
            require_capacity(
                measured, collection=name in collection, max_scored_rows=max_scored_rows
            )
        except ValueError as error:
            raise ValueError(
                f"La fase {name} no cabe en la capacidad declarada: {measured}"
            ) from error
        result[name] = dict(measured, collection=name in collection)
    return result


def capacity_audit(parent, current, *, parent_fold, policy_phases):
    from mars_titan.training.walk_forward_phases import window_phases

    return dict(
        kind="observation_queue_capacity_without_model_execution",
        parent_view_sha256=parent.identity,
        policy_view_sha256=current.identity,
        max_decision_rows=MAX_DECISION_ROWS,
        max_scored_rows=MAX_TRACE_ROWS,
        max_pending_collection_traces=MAX_PENDING_TRACES,
        parent=dataset_capacity(parent, window_phases(parent_fold, 12), max_scored_rows=None),
        policy=dataset_capacity(current, policy_phases, collection=("train", "validation")),
        numeric_targets_used_for_selection=False,
        model_executed=False,
        memory_bytes_measured=False,
    )
