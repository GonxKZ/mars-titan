"""Entradas comunes de las referencias, con las cuatro modalidades y contexto macro."""

import numpy as np

MODALITIES = ("prices", "news", "charts", "fundamentals", "macro")


def feature_vector(inputs: dict) -> np.ndarray:
    if set(inputs) != set(MODALITIES):
        raise ValueError("Se requieren las cuatro modalidades y el contexto macro")
    parts = [np.asarray(inputs[name], dtype=np.float64).reshape(-1) for name in MODALITIES]
    if any(not len(part) or not np.isfinite(part).all() for part in parts):
        raise ValueError("Las modalidades deben contener valores finitos")
    if sum(len(part) for part in parts) > 4096:
        raise ValueError("Las características superan el presupuesto de la referencia")
    return np.concatenate(parts)


def tabular_batches(records, targets: dict, partition: str, *, batch_size: int = 64):
    if type(batch_size) is not int or not 1 <= batch_size <= 4096:
        raise ValueError("El tamaño del lote tabular no es válido")
    x, y = [], []
    for row in records:
        label = targets.get(row["cursor"][0], {}).get(row["prediction_at"].isoformat())
        if label is None or label[1] != partition:
            continue
        x.append(feature_vector(row["inputs"]))
        y.append(label[0])
        if len(x) == batch_size:
            yield np.stack(x), np.asarray(y, dtype=np.float64)
            x, y = [], []
    if x:
        yield np.stack(x), np.asarray(y, dtype=np.float64)
