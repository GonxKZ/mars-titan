"""Contrato de ausencia para preparados y muestras, independiente del objetivo."""

import math
from numbers import Real

import numpy as np

from .temporal import aware

STRICT_INPUTS = "strict_inputs_v1"
HISTORICAL_MASKED = "historical_masked_2000_v1"
INPUT_POLICIES = (STRICT_INPUTS, HISTORICAL_MASKED)
MODALITIES = ("prices", "news", "charts", "fundamentals", "macro")


def validate_historical_vectors(vectors, presence, representation):
    """Contrastar las máscaras y rellenos NumPy antes de convertirlos a tensores."""
    if (
        not isinstance(presence, np.ndarray)
        or presence.dtype != np.bool_
        or presence.ndim != 2
        or presence.shape[1] != len(MODALITIES)
    ):
        raise ValueError("La presencia necesita cinco booleanos por muestra")
    if not presence[:, [0, 2]].all():
        raise ValueError("Los precios y gráficos causales son obligatorios")
    for name, values in vectors.items():
        if not np.isfinite(values).all():
            raise ValueError("Una modalidad histórica contiene valores no finitos")
        observed = presence[:, MODALITIES.index(name)]
        if np.any(values[~observed] != 0):
            raise ValueError("El vector de un bloque ausente debe contener solo ceros")
        if name not in {"fundamentals", "macro"}:
            continue
        catalog = "fundamental_concepts" if name == "fundamentals" else "macro_indicators"
        width = len(representation[catalog])
        if values.shape[1] != 3 * width:
            raise ValueError("El vector numérico no conserva la longitud de su catálogo")
        data, masks, ages = np.split(values, 3, axis=1)
        if not ((masks == 0) | (masks == 1)).all() or (ages < 0).any():
            raise ValueError("Las máscaras por concepto o sus edades no son válidas")
        missing = masks == 0
        if (
            not np.array_equal(observed, (~missing).any(axis=1))
            or np.any(data[missing] != 0)
            or np.any(ages[missing] != 0)
        ):
            raise ValueError("La ausencia del bloque no coincide con sus conceptos y relleno")


def masked_inputs(policy):
    if not isinstance(policy, str) or policy not in INPUT_POLICIES:
        raise ValueError("La política de entradas no está admitida")
    return policy == HISTORICAL_MASKED


def policy_identity(policy):
    if not masked_inputs(policy):
        return {}
    return dict(
        input_policy=policy,
        mask_contract=dict(
            version=1,
            order=list(MODALITIES),
            missing_fill=0.0,
            missing_available_at=None,
            numeric_layout="values_observed_log_age",
            first_prediction="2000-01-01",
            final_test_start="2024-01-01",
        ),
    )


def numeric_observations(rows, cutoff):
    """Separar valores utilizables y causas sin atribuir fechas a publicaciones desconocidas."""
    cutoff = aware(cutoff)
    values, ages, reasons, known = [], [], [], []
    for row in rows:
        value, available = row.get("value"), row.get("available_at")
        reason = row.get("missing_reason")
        if reason is not None and (not isinstance(reason, str) or not 1 <= len(reason) <= 2048):
            raise ValueError("La causa de ausencia no es un texto acotado")
        if value is not None and (isinstance(value, bool) or not isinstance(value, Real)):
            raise ValueError("Una observación numérica contiene un tipo inválido")
        if value is not None and not math.isfinite(value):
            raise ValueError("Una observación numérica contiene un valor no finito")
        if reason is None:
            if value is None:
                reason = "missing_value"
            elif available is None:
                reason = "unknown_publication"
            elif aware(available) > cutoff:
                reason = "not_yet_available"
        if reason is None:
            available = aware(available)
            values.append(float(value))
            ages.append((cutoff - available).total_seconds() / 86400)
            known.append(available)
        else:
            values.append(None)
            ages.append(0.0)
        reasons.append(reason)
    return values, ages, max(known) if known else None, reasons
