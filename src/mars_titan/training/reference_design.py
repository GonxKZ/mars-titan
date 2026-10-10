"""Doce combinaciones fijas por familia, con niveles marginales equilibrados."""

from copy import deepcopy

from .selection import FIXED_BUDGET, validate_selection

# Lote máximo de una referencia neuronal. El ajuste, la búsqueda y la campaña comparten este
# límite, y el Transformer amplía su presupuesto de atención con el lote que declara el caso.
MAX_BATCH_SIZE = 4096


def candidate_indices(config):
    """Compartir el recuento del diseño con los lectores de registros, sin importar CUDA."""
    if config.get("schema_version", 1) == 1:
        return list(range(12))
    indices = config.get("case_indices")
    if (
        config.get("schema_version") not in {2, 3, 4}
        or not isinstance(indices, list)
        or not 1 <= len(indices) <= 3
        or any(type(i) is not int or not 0 <= i < 12 for i in indices)
        or len(set(indices)) != len(indices)
        or indices != sorted(indices)
        or len(indices) * len(config["models"]) > 10
    ):
        raise ValueError("Los candidatos deben ser índices distintos y acotados antes de comparar")
    return indices


# Mismas opciones que el codificador del núcleo Titans-MAC, declaradas en cada caso.
TRANSFORMER_OPTIONS = dict(heads=4, feedforward_multiplier=2)


def design_cases(
    models,
    *,
    seed=42,
    epochs=30,
    patience=5,
    min_delta=0.0,
    minimum_epochs=None,
    stopping=None,
):
    if (
        not isinstance(models, list)
        or not models
        or len(set(models)) != len(models)
        or not set(models) <= {"rnn", "lstm", "gru", "dlinear", "transformer"}
    ):
        raise ValueError("Las familias de modelos deben ser conocidas, distintas y explícitas")
    if (
        type(seed) is not int
        or not 0 <= seed < 2**32
        or type(epochs) is not int
        or not 2 <= epochs <= (30 if minimum_epochs is None else 100)
    ):
        raise ValueError("La semilla o el presupuesto de épocas no son válidos")
    selection = dict(metric="session_mae", patience=patience, min_delta=min_delta)
    if minimum_epochs is not None:
        selection["minimum_epochs"] = minimum_epochs
    if stopping is not None:
        selection["stopping"] = stopping
    validate_selection(selection, epochs=epochs)
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
            if kind == "transformer":
                case["architecture"]["transformer"] = dict(TRANSFORMER_OPTIONS)
            cases.append(dict(id=f"{kind}-{index:02d}-s{seed}", case=case))
    return cases


# Control de la cabeza decidido en #22: solo cambian la salida y su pérdida. Los nombres
# repiten los de models.quantile_head y data.input_policy para que el observatorio pueda
# importar este diseño sin cargar PyTorch. Una prueba comprueba que coinciden.
SCALAR_L1 = "scalar_l1"
QUANTILE_HEAD = "quantile_head_v1"
PINBALL = "pinball"
HISTORICAL_MASKED = "historical_masked_2000_v1"
HEAD_CONTROL_ARMS = {
    SCALAR_L1: dict(loss="mae"),
    QUANTILE_HEAD: dict(loss=PINBALL, head=QUANTILE_HEAD),
}
HEAD_CONTROL_COMPARISON = dict(
    partition="validation",
    metric="session_mae",
    point_prediction="median_for_quantile_head",
    contrast=dict(base=SCALAR_L1, variant=QUANTILE_HEAD, kind="delta"),
    interval="simultaneous_studentized_circular_block_bootstrap",
    level=0.95,
    family="one_contrast_per_case_index",
    fallback="option_a_if_any_simultaneous_interval_excludes_zero_against_quantile_head",
)
_HEAD_CONTROL_KEYS = {
    "schema_version",
    "kind",
    "status",
    "decision",
    "model",
    "arms",
    "case_indices",
    "seeds",
    "max_epochs",
    "patience",
    "min_delta",
    "stopping",
    "batch_size",
    "context_sessions",
    "market",
    "walk_forward",
    "input_policy",
    "prediction_retention",
    "comparison",
    "final_test_opened",
}


def head_control_cases(plan):
    """Casos emparejados del Transformer compacto con salida escalar L1 o `quantile_head_v1`.

    Cada par comparte índice de diseño, semilla, arquitectura, tasa de aprendizaje,
    presupuesto y selección. El plan solo se declara. Esta función no lee datos ni
    ejecuta ajustes y falla ante cualquier campo que permita cambiar el contraste.
    """
    if (
        not isinstance(plan, dict)
        or set(plan) != _HEAD_CONTROL_KEYS
        or plan["schema_version"] != 1
        or plan["kind"] != "quantile_head_control"
        or plan["status"] != "declared_not_executed"
        or plan["model"] != "transformer"
        or plan["arms"] != list(HEAD_CONTROL_ARMS)
        or plan["stopping"] != FIXED_BUDGET
        or plan["input_policy"] != HISTORICAL_MASKED
        or plan["prediction_retention"] != "heldout_full_train_sessions_v1"
        or plan["market"] != "US"
        or plan["context_sessions"] != 64
        or type(plan["batch_size"]) is not int
        or not 1 <= plan["batch_size"] <= 256
        or plan["comparison"] != HEAD_CONTROL_COMPARISON
        or plan["final_test_opened"] is not False
    ):
        raise ValueError("El control de la cabeza no cumple el contrato declarado")
    seeds = plan["seeds"]
    if (
        not isinstance(seeds, list)
        or not 1 <= len(seeds) <= 8
        or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Las semillas del control deben ser enteros distintos")
    indices = plan["case_indices"]
    if (
        not isinstance(indices, list)
        or not 1 <= len(indices) <= 3
        or any(type(i) is not int or not 0 <= i < 12 for i in indices)
        or indices != sorted(set(indices))
    ):
        raise ValueError("Los índices del control deben ser distintos, ordenados y del diseño")
    if not isinstance(plan["walk_forward"], str) or not isinstance(plan["decision"], str):
        raise ValueError("El control necesita su protocolo temporal y su decisión")
    cases = []
    for seed in seeds:
        design = design_cases(
            ["transformer"],
            seed=seed,
            epochs=plan["max_epochs"],
            patience=plan["patience"],
            min_delta=plan["min_delta"],
            stopping=plan["stopping"],
        )
        for index in indices:
            base = design[index]
            for arm, change in HEAD_CONTROL_ARMS.items():
                case = deepcopy(base["case"]) | change
                cases.append(dict(id=f"{base['id']}-{arm}", pair=base["id"], arm=arm, case=case))
    return cases
