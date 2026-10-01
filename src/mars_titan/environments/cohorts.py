"""Bloques de observaciones simultáneas con disponibilidad y etiquetas separadas."""

import hashlib
import json
import re

import numpy as np

from mars_titan.models.baselines.inputs import MODALITIES

FINAL_TEST_START_US = 1_704_067_200_000_000
VALIDATION_START_US = 1_672_531_200_000_000


def shapes_contract(shapes, max_assets, max_observation_bytes):
    if (
        not isinstance(shapes, dict)
        or set(shapes) != set(MODALITIES)
        or type(max_assets) is not int
        or not 1 <= max_assets <= 4096
        or type(max_observation_bytes) is not int
        or not 1 <= max_observation_bytes <= 128 * 1024**2
    ):
        raise ValueError("Las formas y el presupuesto del entorno no son válidos")
    for name, shape in shapes.items():
        if (
            not isinstance(shape, (tuple, list))
            or len(shape) != (2 if name == "prices" else 1)
            or any(type(size) is not int or not 1 <= size <= 4096 for size in shape)
            or (name == "prices" and (shape[1] != 5 or not 2 <= shape[0] <= 512))
        ):
            raise ValueError("La forma de una modalidad no cumple el contrato")
    size = max_assets * (sum(int(np.prod(s)) for s in shapes.values()) * 4 + 1)
    if size > max_observation_bytes:
        raise ValueError("La observación rellenada supera el presupuesto")
    return {name: tuple(shape) for name, shape in shapes.items()}


def _timestamps(values, rows):
    array = np.asarray(values)
    if array.shape != (rows,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("La disponibilidad necesita microsegundos UTC enteros por fila")
    if (array < 0).any() or (array >= FINAL_TEST_START_US).any():
        raise ValueError("Las fechas están fuera del periodo de desarrollo")
    return array.astype(np.int64, copy=True)


def read_cohort(raw, shapes, max_assets, max_bytes):
    """Copiar un único bloque, ordenar activos y comprobar cada modalidad antes de usarlo."""
    keys = {"prediction_at", "asset_ids", "available_at", "target_available_at", "target", "inputs"}
    if not isinstance(raw, dict) or set(raw) != keys:
        raise ValueError("La cohorte no conserva su contrato de observaciones y etiquetas")
    at, ids = raw["prediction_at"], raw["asset_ids"]
    if (
        type(at) is not int
        or not 0 <= at < FINAL_TEST_START_US
        or not isinstance(ids, (list, tuple))
        or not 1 <= len(ids) <= max_assets
        or any(
            not isinstance(i, str) or not re.fullmatch(r"(?:US|CN)/[A-Z0-9.^_=\-]{1,64}", i)
            for i in ids
        )
        or len(set(ids)) != len(ids)
        or not isinstance(raw["inputs"], dict)
        or set(raw["inputs"]) != set(shapes)
    ):
        raise ValueError("La identidad, fecha o población de la cohorte no es válida")
    n = len(ids)
    available, maturity = (
        _timestamps(raw["available_at"], n),
        _timestamps(raw["target_available_at"], n),
    )
    if (available > at).any() or (maturity <= at).any():
        raise ValueError(
            "Una entrada procede del futuro o una etiqueta no es posterior a la decisión"
        )
    target = np.asarray(raw["target"])
    if target.shape != (n,) or np.iscomplexobj(target) or not np.isfinite(target).all():
        raise ValueError("Las etiquetas necesitan valores finitos, uno por activo")
    try:
        with np.errstate(over="raise", invalid="raise"):
            target = target.astype(np.float64, copy=False)
    except (TypeError, ValueError, FloatingPointError) as error:
        raise ValueError("Las etiquetas no se pueden representar en float64") from error
    order = np.argsort(ids)
    inputs, total = {}, 0
    for name, shape in shapes.items():
        value = np.asarray(raw["inputs"][name])
        if value.shape != (n, *shape) or np.iscomplexobj(value):
            raise ValueError("Una modalidad tiene forma o tipo incorrectos")
        total += max(value.nbytes, value.size * 4)
        if total > max_bytes:
            raise ValueError("El bloque de entrada supera el presupuesto")
        try:
            with np.errstate(over="raise", invalid="raise"):
                converted = value.astype(np.float32, copy=False)[order]
        except (TypeError, ValueError, FloatingPointError) as error:
            raise ValueError("La modalidad no se puede representar en float32") from error
        if not np.isfinite(converted).all():
            raise ValueError("La modalidad contiene valores no finitos")
        converted.setflags(write=False)
        inputs[name] = converted
    result = dict(
        prediction_at=at,
        asset_ids=[ids[i] for i in order],
        inputs=inputs,
        available_at=available[order],
        target_available_at=maturity[order],
        target=target[order],
    )
    digest = hashlib.sha256(json.dumps([at, result["asset_ids"]]).encode())
    for name in ("available_at", "target_available_at", "target"):
        digest.update(result[name].tobytes())
    for name in MODALITIES:
        digest.update(result["inputs"][name].tobytes())
    result["sha256"] = digest.hexdigest()
    return result
