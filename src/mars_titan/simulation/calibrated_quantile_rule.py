"""Regla fija sobre cuantiles calibrados y no operar (G2 del estudio de variantes, #748).

G2 compara políticas financieras que leen cuantiles calibrados. PPO y KLPO (#137) necesitan
entrenar y siguen bloqueados. Este módulo implementa los dos controles que pueden
descartarlas, sin aprender nada y conectados al entorno auditado de #365:

- `calibrated_quantile_rule_v1`. La puntuación de cada activo en la cinta es la emisión de
  A2: la mediana si el intervalo calibrado excluye el cero y cero en otro caso
  (`neutral_abstention.apply`). En cada decisión la regla expone toda la
  cartera si algún activo valorado tiene una puntuación positiva y se queda en efectivo en
  otro caso. El entorno reparte la exposición por igual en el cuartil superior de las
  puntuaciones positivas, así que solo compra activos con una afirmación alcista.
- `no_trade`. Efectivo en todas las decisiones (`evaluation.fixed_policy("cash")`).

La regla lee la observación del entorno, no la cinta, y no conoce el futuro. Como el
entorno solo compra, una afirmación bajista equivale a no operar con ese activo.
"""

import numpy as np

from .environment import ACTIONS, ALLOCATIONS
from .evaluation import evaluate, fixed_policy

IDENTITY = "calibrated_quantile_rule_v1"
# Rasgos por activo de la observación de `FinancialEnv`: puntuación escalada, peso, retorno,
# volumen, orden pendiente y validez. Tras ellos van el efectivo y el resto sin invertir.
FEATURES_PER_ASSET = 6
SCORE, VALID = 0, 5
INVEST, CASH = ACTIONS.index(1.0), ACTIONS.index(0.0)


def rule_policy(observation, _step):
    """Exposición completa si algún activo válido tiene puntuación positiva, efectivo si no."""
    observation = np.asarray(observation)
    assets, rest = divmod(len(observation) - 2, FEATURES_PER_ASSET)
    if observation.ndim != 1 or assets < 1 or rest:
        raise ValueError("La observación no tiene el formato del entorno financiero")
    features = observation[: assets * FEATURES_PER_ASSET].reshape(assets, FEATURES_PER_ASSET)
    claims = (features[:, VALID] > 0) & (features[:, SCORE] > 0)
    return INVEST if claims.any() else CASH


CONTROLS = {IDENTITY: rule_policy, "no_trade": fixed_policy("cash")}


def compare(env, *, seed=42):
    """Evaluar la regla y no operar sobre el mismo entorno, sin actualizar nada.

    La regla solo tiene sentido con el reparto en el cuartil superior de puntuaciones
    positivas, el mismo que usan las referencias fijas del predictor.
    """
    if env.allocation != ALLOCATIONS[0]:
        raise ValueError("La regla necesita el reparto en el cuartil superior de puntuaciones")
    return {name: evaluate(env, policy, seed=seed) for name, policy in CONTROLS.items()}
