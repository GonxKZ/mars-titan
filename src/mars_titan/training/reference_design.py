"""Doce combinaciones fijas por familia, con niveles marginales equilibrados."""

from .selection import validate_selection


def design_cases(models, *, seed=42, epochs=30, patience=5, min_delta=0.0):
    if (
        not isinstance(models, list)
        or not models
        or len(set(models)) != len(models)
        or not set(models) <= {"rnn", "lstm", "gru", "dlinear"}
    ):
        raise ValueError("Las familias de modelos deben ser conocidas, distintas y explícitas")
    if (
        type(seed) is not int
        or not 0 <= seed < 2**32
        or type(epochs) is not int
        or not 2 <= epochs <= 30
    ):
        raise ValueError("La semilla o el presupuesto de épocas no son válidos")
    selection = dict(metric="session_mae", patience=patience, min_delta=min_delta)
    validate_selection(selection)
    cases = []
    for kind in models:
        for index in range(12):
            case = dict(
                kind=kind,
                seed=seed,
                epochs=epochs,
                huber_delta=0.01,
                loss=("mae", "mse", "huber")[(index + index // 3) % 3],
                learning_rate=(0.0001, 0.0003, 0.001)[(index + index // 4) % 3],
                architecture=dict(
                    hidden_size=(32, 64, 128)[index % 3],
                    layers=1 + index % 2,
                    dropout=(0.0, 0.1, 0.2)[index // 4],
                ),
                selection=dict(selection),
            )
            cases.append(dict(id=f"{kind}-{index:02d}-s{seed}", case=case))
    return cases
