"""Auditoría as-of del registro de etiquetas `asof_ledger_audit` (G3, #748 y #442).

Un registro es una tabla de eventos con su disponibilidad: cada fila es una etiqueta (u otro
dato) que solo puede leerse desde `available_at`. Siguiendo la idea de Event Sourcing
(Fowler, https://martinfowler.com/eaaDev/EventSourcing.html), lo que una etapa emite en el
instante t debe poder reconstruirse a partir de los eventos disponibles en t, sin depender de
ningún evento posterior. La procedencia sigue W3C PROV-DM
(https://www.w3.org/TR/prov-dm/): cada emisión se deriva de entidades que ya existían.

La auditoría no mide rendimiento. Para una función de emisión y unos cortes declarados
comprueba tres propiedades y devuelve un recibo:

1. Reconstrucción. Las filas emitidas como mucho en t son bit a bit iguales con el registro
   completo y con el registro en el que los eventos posteriores a t tienen otros valores. Se
   prueban dos sustituciones: cero y un valor extremo de signo contrario.
2. Repetición. Dos ejecuciones sobre el mismo registro emiten lo mismo bit a bit, y lo
   emitido coincide con una emisión guardada si se aporta.
3. Rechazo. Un evento disponible en el instante de su propia emisión (`issued_at`) o antes,
   o dentro del test reservado de 2024, invalida el registro antes de auditarlo. `rejection`
   comprueba además que las etapas que reciben los dos instantes lo rechacen por sí mismas.

La función de emisión recibe un registro (diccionario de columnas alineadas) y devuelve las
emisiones con una fila por predicción y el instante de cada una. No debe modificar el
registro. Nada de este módulo ajusta modelos ni lee el test de 2024.
"""

import hashlib

import numpy as np

from mars_titan.environments.cohorts import FINAL_TEST_START_US

KIND = "asof_ledger_audit"
SCHEMA_VERSION = 1
# Valor que sustituye a los eventos posteriores al corte, con el signo contrario al original.
EXTREME = 1e6
MAX_CUTOFFS = 4096


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _instants(values, rows, name):
    values = np.asarray(values)
    _require(
        values.shape == (rows,) and values.dtype.kind in "iu",
        f"{name} necesita un instante entero por fila en microsegundos UTC",
    )
    _require(
        values.dtype.kind == "i" or int(values.max(initial=0)) <= np.iinfo(np.int64).max,
        f"{name} no cabe en int64",
    )
    return values.astype(np.int64)


def checked_ledger(ledger, columns):
    """Comprobar el registro y devolver la disponibilidad de cada evento.

    Rechaza columnas desalineadas, valores no reales, eventos disponibles en el mismo instante
    de su emisión o antes, y cualquier instante del test reservado de 2024.
    """
    _require(
        isinstance(ledger, dict) and {"issued_at", "available_at"} <= set(ledger) and columns,
        "El registro necesita `issued_at`, `available_at` y al menos una columna de valores",
    )
    rows = len(np.asarray(ledger["available_at"]))
    _require(rows >= 1, "El registro está vacío")
    for name, values in ledger.items():
        _require(len(np.asarray(values)) == rows, f"La columna {name} no está alineada")
    for name in columns:
        _require(
            name in ledger and np.asarray(ledger[name]).dtype.kind == "f",
            f"La columna de valores {name} debe ser real",
        )
    issued_at = _instants(ledger["issued_at"], rows, "La emisión")
    available_at = _instants(ledger["available_at"], rows, "La disponibilidad")
    _require(
        np.all(available_at > issued_at),
        "Un evento está disponible en el instante de su emisión o antes",
    )
    _require(
        np.all(available_at < FINAL_TEST_START_US),
        "El registro contiene eventos del test reservado de 2024",
    )
    return available_at


def _emit(emit, ledger):
    emitted, emitted_at = emit(ledger)
    emitted = np.ascontiguousarray(emitted)
    emitted_at = _instants(emitted_at, len(emitted), "Los instantes de emisión")
    return emitted, emitted_at


def _replaced(ledger, columns, future, how):
    """Copia del registro con los valores de los eventos futuros sustituidos."""
    result = {name: np.array(values, copy=True) for name, values in ledger.items()}
    for name in columns:
        values = result[name]
        if how == "zero":
            values[future] = 0.0
        else:
            values[future] = np.where(values[future] > 0, -EXTREME, EXTREME)
    return result


def _sha256(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def audit(emit, ledger, *, columns, cutoffs, stored=None):
    """Auditar una etapa sobre su registro y devolver el recibo de validez.

    `columns` son las columnas de valores que la etapa no debe leer antes de su
    disponibilidad. `cutoffs` son los instantes t auditados. `stored` es una emisión
    guardada que debe reproducirse bit a bit. Un fallo no lanza error: queda en el recibo con
    `passed=False`, para que la puerta de integridad decida.
    """
    available_at = checked_ledger(ledger, columns)
    cutoffs = sorted({int(value) for value in cutoffs})
    _require(1 <= len(cutoffs) <= MAX_CUTOFFS, f"Se auditan entre 1 y {MAX_CUTOFFS} cortes")
    snapshot = {name: np.array(values, copy=True) for name, values in ledger.items()}
    emitted, emitted_at = _emit(emit, ledger)
    again, again_at = _emit(emit, ledger)
    failures = []
    if emitted.tobytes() != again.tobytes() or not np.array_equal(emitted_at, again_at):
        failures.append(dict(check="repeat", detail="dos ejecuciones emiten valores distintos"))
    if stored is not None and np.ascontiguousarray(stored).tobytes() != emitted.tobytes():
        failures.append(dict(check="stored", detail="la emisión no reproduce la guardada"))
    if any(not np.array_equal(snapshot[name], ledger[name]) for name in ledger):
        failures.append(dict(check="read_only", detail="la etapa modificó el registro"))
    checked = []
    for cutoff in cutoffs:
        past = emitted_at <= cutoff
        future = available_at > cutoff
        entry = dict(
            cutoff=cutoff,
            emitted_rows=int(past.sum()),
            future_events=int(future.sum()),
            emitted_sha256=_sha256(emitted[past]),
        )
        for how in ("zero", "extreme"):
            if not future.any() or not past.any():
                continue
            other, other_at = _emit(emit, _replaced(ledger, columns, future, how))
            if not np.array_equal(other_at, emitted_at) or (
                other[past].tobytes() != emitted[past].tobytes()
            ):
                failures.append(dict(check="reconstruction", cutoff=cutoff, replacement=how))
        checked.append(entry)
    return dict(
        schema_version=SCHEMA_VERSION,
        kind=KIND,
        columns=list(columns),
        ledger_rows=int(len(available_at)),
        emitted_rows=int(len(emitted)),
        emitted_sha256=_sha256(emitted),
        cutoffs=checked,
        failures=failures,
        passed=not failures,
    )


def rejection(emit, ledger):
    """Comprobar que la etapa rechaza eventos imposibles en lugar de emitirlos.

    Prueba dos registros alterados en el último evento: disponible en el mismo instante de su
    emisión y disponible dentro del test reservado de 2024. Devuelve los casos que la etapa
    no rechazó con `ValueError`.
    """
    rows = len(np.asarray(ledger["available_at"]))
    issued = _instants(ledger["issued_at"], rows, "La emisión de los eventos")
    cases = {
        "available_at_issue": issued[-1],
        "available_in_2024": FINAL_TEST_START_US,
    }
    accepted = []
    for name, instant in cases.items():
        altered = {key: np.array(values, copy=True) for key, values in ledger.items()}
        altered["available_at"] = _instants(altered["available_at"], rows, "La disponibilidad")
        altered["available_at"][-1] = instant
        try:
            emit(altered)
        except ValueError:
            continue
        accepted.append(name)
    return accepted
