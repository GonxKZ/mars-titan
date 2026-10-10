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


def _lists(values, width, like):
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
    modalities = ablated_modalities(variant)
    names = set(table.column_names)
    unknown = names - _REWRITTEN - _INDEPENDENT
    if unknown:
        raise ValueError(f"La ablación no sabe ausentar las columnas {sorted(unknown)}")
    if not {"presence", "news_count", "input_availability", *modalities} <= names:
        raise ValueError("La ablación necesita presencia, recuento, disponibilidad y vectores")
    rows = len(table)
    if not rows:
        return table
    column = table["presence"].combine_chunks()
    bits = column.flatten().to_numpy(zero_copy_only=False).reshape(rows, len(MODALITIES)).copy()
    observed = {name: bits[:, MODALITIES.index(name)].copy() for name in modalities}
    for name in modalities:
        bits[:, MODALITIES.index(name)] = False
    columns = {"presence": _lists(pa.array(bits.ravel()), len(MODALITIES), column.type)}
    for name in modalities:
        vector = table[name].combine_chunks()
        lengths = pc.list_value_length(vector)
        bounds = pc.min_max(lengths)
        if lengths.null_count or bounds["min"] != bounds["max"]:
            raise ValueError(f"Los vectores de {name} no tienen una anchura común")
        width = bounds["min"].as_py()
        columns[name] = _lists(
            pa.array(np.zeros(rows * width, dtype=np.float32)), width, vector.type
        )
    availability = table["input_availability"].combine_chunks()
    columns["input_availability"] = _struct(
        availability,
        {
            name: pa.nulls(rows, availability.type.field(name).type)
            for name in modalities
            if availability.type.get_field_index(name) >= 0
        },
    )
    if "news" in modalities:
        columns["news_count"] = pa.array(np.zeros(rows, dtype=np.int64)).cast(
            table["news_count"].type
        )
    if "missing_reasons" in names:
        reasons = table["missing_reasons"].combine_chunks()
        columns["missing_reasons"] = _struct(
            reasons,
            {
                name: pc.if_else(pa.array(observed[name]), REASON, reasons.field(name))
                for name in modalities
            },
        )
    for name, values in columns.items():
        table = table.set_column(table.schema.get_field_index(name), name, values)
    return table
