"""Vincular el contexto S2 a las acciones realmente registradas por la emisión."""

import math

from mars_titan.models.titans.mechanism_catalog import MechanismAction


def emission_actions(rows, values, control):
    """Conciliar filas y puntos antes de identificar las rutas ejecutadas y emitidas."""
    if (
        type(control) is not dict
        or type(control.get("rows")) is not list
        or len(control["rows"]) != len(rows)
        or len(values) != len(rows)
        or any(
            type(record) is not dict or type(record.get("flow_id")) is not str
            for record in control["rows"]
        )
    ):
        raise ValueError("S2 necesita la traza completa de los mecanismos emitidos")
    records = {record.get("flow_id"): record for record in control["rows"]}
    if len(records) != len(rows) or set(records) != {row.flow_id for row in rows}:
        raise ValueError("La traza S2 repite o cambia los flujos emitidos")
    actions = []
    for row, value in zip(rows, values, strict=True):
        record = records[row.flow_id]
        if (
            any(record.get(name) != expected for name, expected in row.metadata().items())
            or type(record.get("point")) is not float
            or not math.isfinite(record["point"])
            or record.get("point") != value
        ):
            raise ValueError("La traza S2 no corresponde a la fila y al punto realmente emitidos")
        executed = MechanismAction(
            record.get("context"), record.get("executed_route"), record.get("calibration")
        )
        emitted = MechanismAction(
            executed.context, record.get("emitted_route"), executed.calibration
        )
        actions.append(f"mechanism/{executed.key}/emitted/{emitted.route}")
    return actions
