"""Objetivos de salida que sustituyen la pérdida escalar del predictor (#749).

Cada objetivo es un campo opcional de `FinancialConfig` con su configuración identificada. Sin
ninguno declarado, el predictor conserva su cabeza, su identidad y su cálculo. Con uno, la
cabeza escalar sigue existiendo con los mismos pesos iniciales y el módulo del objetivo añade
sus parámetros, decide la predicción emitida y define la pérdida del ajuste cronológico. Son
excluyentes entre sí y solo sustituyen la cabeza escalar, cuyo control es la pérdida MAE.

El módulo de cada objetivo declara:

- `forward(salida_escalar, representación, mercados)`, que devuelve las salidas del ajuste
  [filas, k] y la predicción puntual;
- `loss(salidas, objetivos, sesiones)`, la pérdida media del tramo;
- `session_level`, si necesita la sesión completa en un bloque;
- `row_decomposable`, si la pérdida es una media de términos por fila.
"""

from dataclasses import fields

from mars_titan.models.baselines.multimodal import SCALAR_HEAD

from .session_ic_objective import SessionICObjective

OBJECTIVES = {"session_ic_objective": SessionICObjective}


def check_objectives(config):
    """Convertir las declaraciones JSON, comprobar tipos y exigir un solo objetivo."""
    for name, kind in OBJECTIVES.items():
        value = getattr(config, name)
        if isinstance(value, dict):
            if set(value) != {field.name for field in fields(kind)}:
                raise ValueError(f"{name} debe declarar toda su configuración")
            object.__setattr__(config, name, kind(**value))
        elif value is not None and type(value) is not kind:
            raise ValueError(f"{name} necesita su configuración, sus valores o None")
    declared = [name for name in OBJECTIVES if getattr(config, name) is not None]
    if len(declared) > 1:
        raise ValueError("Los objetivos de salida son excluyentes entre sí")
    if declared and config.head != SCALAR_HEAD:
        raise ValueError("Los objetivos de salida sustituyen solo la cabeza escalar")


def declared_objective(config):
    """La configuración del objetivo declarado, o None."""
    for name in OBJECTIVES:
        value = getattr(config, name)
        if value is not None:
            return value
    return None


def objective_identity(config):
    """Campos que el objetivo añade a la identidad del predictor. Sin objetivo, ninguno."""
    objective = declared_objective(config)
    if objective is None:
        return {}
    return dict(output=objective.name, output_objective=objective.identity())


def build_objective(config, *, dtype):
    objective = declared_objective(config)
    if objective is None:
        return None
    return objective.build(config.hidden_size, seed=config.seed, dtype=dtype)
