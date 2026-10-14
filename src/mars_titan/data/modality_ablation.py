"""Ablación de modalidades en la lectura, con el mismo contrato que una ausencia real.

Una variante declara qué modalidades se leen como ausentes en todas las filas. En la
edición con máscaras (`historical_masked_2000_v1`), una muestra sin una modalidad tiene el
bit de presencia falso, el vector con el relleno de ausencia (`missing_fill`, cero en
valores, máscaras por concepto y edades), la disponibilidad nula y una causa en
`missing_reasons`. La ablación escribe exactamente eso en la tabla de muestras, con la
causa `modality_ablation`, y las noticias pasan además a cero eventos admitidos, porque
el lector exige que presencia y recuento coincidan. Después la tabla pasa por las mismas
comprobaciones que cualquier otra, así que el modelo recibe lo mismo que ante una
ausencia real.

Solo se ablacionan noticias y fundamentales. Precios y gráficos son obligatorios y el
macro está presente en todas las filas de la edición. Las filas que ya carecían de la
modalidad conservan sus valores y su causa original.
"""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .input_policy import HISTORICAL_MASKED, MODALITIES, policy_identity

VARIANTS = {
    "mask_news": ("news",),
    "mask_fundamentals": ("fundamentals",),
    "mask_news_and_fundamentals": ("news", "fundamentals"),
}
REASON = "modality_ablation"
# Columnas que la ablación reescribe y columnas que no dependen de las modalidades
# ablacionadas. Cualquier otra columna se rechaza en lugar de quedar incoherente.
_REWRITTEN = {"presence", "news", "fundamentals", "input_availability", "news_count"}
_INDEPENDENT = {
    "prediction_at",
    "price_end_index",
    "cohort_id",
    "charts",
    "macro",
    "macro_available_at",
    "missing_reasons",
}


def ablated_modalities(variant):
    """Modalidades que una variante declarada lee como ausentes."""
    if not isinstance(variant, str) or variant not in VARIANTS:
        raise ValueError("La ablación de modalidades no está declarada")
    return VARIANTS[variant]


def ablation_identity(variant):
    """Campos que identifican la ablación en recibos e informes."""
    return dict(
        variant=variant,
        modalities=list(ablated_modalities(variant)),
        presence=False,
        fill="missing_fill",
        available_at=None,
        missing_reason=REASON,
        mask_contract=policy_identity(HISTORICAL_MASKED)["mask_contract"],
    )


def list_column(values, width, like):
    """Lista de `width` valores por fila con el tipo Arrow de la columna original."""
    if pa.types.is_fixed_size_list(like):
        array = pa.FixedSizeListArray.from_arrays(values, width)
    else:
        offsets = pa.array(np.arange(0, len(values) + 1, width, dtype=np.int32))
        array = pa.ListArray.from_arrays(offsets, values)
    return array.cast(like)


def _struct(column, replace):
    """Struct con los campos de `replace` sustituidos y la misma validez por fila."""
    children = [
        replace.get(field.name, column.field(index)) for index, field in enumerate(column.type)
    ]
    return pa.StructArray.from_arrays(children, fields=list(column.type), mask=column.is_null())


def ablate_samples(table, variant):
    """Tabla de muestras con las modalidades de la variante ausentes en todas las filas."""
    return absent_rows(table, ablated_modalities(variant), None, REASON)


def absent_rows(table, modalities, rows, reason):
    """Tabla con `modalities` ausentes en las filas marcadas en `rows`, o en todas con None.

    Las filas que ya carecían de una modalidad conservan su causa original. Las que la
    pierden reciben `reason`.
    """
    names = set(table.column_names)
    unknown = names - _REWRITTEN - _INDEPENDENT
    if unknown:
        raise ValueError(f"La ablación no sabe ausentar las columnas {sorted(unknown)}")
    if not {"presence", "news_count", "input_availability", *modalities} <= names:
        raise ValueError("La ablación necesita presencia, recuento, disponibilidad y vectores")
    count = len(table)
    if not count:
        return table
    rows = np.ones(count, dtype=bool) if rows is None else np.asarray(rows, dtype=bool)
    if rows.shape != (count,):
        raise ValueError("Las filas ausentes no corresponden a la tabla")
    selected = pa.array(rows)
    column = table["presence"].combine_chunks()
    bits = column.flatten().to_numpy(zero_copy_only=False).reshape(count, len(MODALITIES)).copy()
    observed = {name: bits[:, MODALITIES.index(name)] & rows for name in modalities}
    for name in modalities:
        bits[rows, MODALITIES.index(name)] = False
    columns = {"presence": list_column(pa.array(bits.ravel()), len(MODALITIES), column.type)}
    for name in modalities:
        vector = table[name].combine_chunks()
        lengths = pc.list_value_length(vector)
        bounds = pc.min_max(lengths)
        if lengths.null_count or bounds["min"] != bounds["max"]:
            raise ValueError(f"Los vectores de {name} no tienen una anchura común")
        width = bounds["min"].as_py()
        values = vector.flatten().to_numpy(zero_copy_only=False).reshape(count, width)
        # Las filas que no se ausentan conservan sus valores y su tipo.
        values = np.where(rows[:, None], np.zeros((), values.dtype), values)
        columns[name] = list_column(pa.array(values.ravel()), width, vector.type)
    availability = table["input_availability"].combine_chunks()
    columns["input_availability"] = _struct(
        availability,
        {
            name: pc.if_else(
                selected,
                pa.nulls(count, availability.type.field(name).type),
                availability.field(name),
            )
            for name in modalities
            if availability.type.get_field_index(name) >= 0
        },
    )
    if "news" in modalities:
        counts = table["news_count"].combine_chunks()
        columns["news_count"] = pc.if_else(selected, pa.scalar(0, counts.type), counts)
    if "missing_reasons" in names:
        reasons = table["missing_reasons"].combine_chunks()
        columns["missing_reasons"] = _struct(
            reasons,
            {
                name: pc.if_else(pa.array(observed[name]), reason, reasons.field(name))
                for name in modalities
            },
        )
    for name, values in columns.items():
        table = table.set_column(table.schema.get_field_index(name), name, values)
    return table
