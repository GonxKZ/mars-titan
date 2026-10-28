"""Dos escrituras proximales con claves, pesos e hiperparámetros comunes.

Reutiliza el Gram y Cholesky de la cohorte. Mantiene separados los lados derechos y las
soluciones para conservar el recorrido numérico de las escrituras individuales.
"""

import torch

from .associative_memory import AssociativeMemory, MatureFeedback, _check_matrix


def write_proximal_pair(
    left: AssociativeMemory,
    right: AssociativeMemory,
    feedback: MatureFeedback,
    other_values: torch.Tensor,
    *,
    cutoff: int,
) -> tuple[AssociativeMemory, AssociativeMemory]:
    """Actualizar ambos estados o ninguno, sin conservar factores entre cohortes."""
    if type(left) is not AssociativeMemory or type(right) is not AssociativeMemory:
        raise ValueError("La pareja debe compartir dos estados asociativos proximales")
    a, b = left.config, right.config
    if (
        a.rule != "proximal"
        or b.rule != "proximal"
        or (a.key_size, a.rate, a.forgetting, left.writes, left.cursor)
        != (b.key_size, b.rate, b.forgetting, right.writes, right.cursor)
    ):
        raise ValueError("La pareja debe compartir regla, claves, tasa, olvido, contador y cursor")
    rows, order = left._validate(feedback, cutoff)
    _check_matrix(other_values, "Los valores del segundo estado", rows, b.value_size)
    if rows == 0:
        return left, right
    index = torch.tensor(order, dtype=torch.int64)
    keys = feedback.keys.index_select(0, index)
    values = feedback.values.index_select(0, index)
    weights = feedback.weights.index_select(0, index)
    weighted = keys * weights.unsqueeze(1)
    gram = weighted.T @ keys
    cross = weighted.T @ values
    system = torch.eye(a.key_size, dtype=torch.float64) + a.rate * gram
    factor, info = torch.linalg.cholesky_ex(system)
    if int(info) != 0:
        raise ValueError("El sistema proximal compartido no es definido positivo en FP64")
    first = torch.cholesky_solve(a.retention * left._matrix + a.rate * cross, factor)
    cross = weighted.T @ other_values.index_select(0, index)
    second = torch.cholesky_solve(b.retention * right._matrix + b.rate * cross, factor)
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError("La escritura compartida produjo un estado no finito")
    last = order[-1]
    cursor = (
        int(feedback.available_at[last]),
        int(feedback.decision_at[last]),
        int(feedback.ids[last]),
    )
    return (
        AssociativeMemory(a, first, writes=left.writes + rows, cursor=cursor),
        AssociativeMemory(b, second, writes=right.writes + rows, cursor=cursor),
    )
