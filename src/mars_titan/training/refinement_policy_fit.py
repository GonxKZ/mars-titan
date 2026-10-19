"""Ajuste acotado de utilidades con etiquetas maduras y pesos por sesión."""

import math

import torch

from mars_titan.models.titans.refinement_policy import FEATURES, RoutePolicy
from mars_titan.models.titans.state import check_finite, require_true
from mars_titan.training.learning_hold import require_learning_allowed

MAX_FIT_ROWS = 131072


def signed_gains(predictions, targets):
    """MAE de K1 menos MAE de cada continuación, sin truncar pérdidas de utilidad."""
    if (
        not isinstance(predictions, torch.Tensor)
        or not isinstance(targets, torch.Tensor)
        or predictions.ndim != 2
        or targets.ndim != 1
        or predictions.shape != (len(targets), 6)
        or predictions.dtype not in (torch.float32, torch.float64)
        or targets.dtype != predictions.dtype
        or targets.device != predictions.device
    ):
        raise ValueError("Las predicciones y etiquetas no están alineadas")
    check_finite(predictions, "Las predicciones de las rutas")
    check_finite(targets, "Las etiquetas maduras")
    errors = (predictions - targets[:, None]).abs()
    gains = errors[:, :1] - errors[:, 1:]
    check_finite(gains, "Las ganancias firmadas")
    return gains


def _check_rows(features, predictions, targets, decision_at, maturity_at, weights):
    if (
        not isinstance(features, torch.Tensor)
        or features.ndim != 2
        or features.shape[1] != len(FEATURES)
        or not 1 <= len(features) <= MAX_FIT_ROWS
        or features.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("El ajuste necesita rasgos finitos y un lote acotado")
    size = len(features)
    for value in (targets, weights):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (size,)
            or value.dtype != features.dtype
            or value.device != features.device
        ):
            raise ValueError("Las etiquetas y pesos deben compartir filas, precisión y dispositivo")
        check_finite(value, "Las etiquetas y pesos del ajuste")
    for value in (decision_at, maturity_at):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (size,)
            or value.dtype != torch.int64
        ):
            raise ValueError("El ajuste necesita instantes int64 por fila")
        if value.device != features.device:
            raise ValueError("Los instantes del ajuste deben compartir dispositivo")
    if not isinstance(predictions, torch.Tensor) or predictions.shape != (size, 6):
        raise ValueError("Las seis predicciones deben conservar las mismas filas")
    check_finite(features, "Los rasgos de ajuste")
    require_true((weights > 0).all(), "Cada fila necesita un peso positivo de su sesión")


def fit_policy(
    *,
    features,
    predictions,
    targets,
    decision_at,
    maturity_at,
    session_weights,
    parent_selected_before,
    fit_start,
    fit_end,
    ridge,
    costs,
    min_gain,
):
    """Resolver ridge sobre ganancias firmadas, con normalización ajustada solo aquí.

    `parent_selected_before` debe cubrir la selección del núcleo y del lector.
    Los recibos de sus checkpoints y de las trazas acreditan esa procedencia fuera
    de esta función. Los tiempos son microsegundos UTC. Los pesos los fija el
    contrato de mercados y sesiones del estudio, no la cantidad de rutas.
    """
    require_learning_allowed("ajuste de la política de refinamientos")
    _check_rows(features, predictions, targets, decision_at, maturity_at, session_weights)
    if (
        not isinstance(costs, torch.Tensor)
        or costs.shape != (5,)
        or costs.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("El ajuste necesita cinco costes explícitos")
    if any(type(value) is not int for value in (parent_selected_before, fit_start, fit_end)):
        raise ValueError("Los cortes del ajuste deben ser instantes enteros")
    if not 0 <= parent_selected_before <= fit_start < fit_end:
        raise ValueError("El padre y el lector deben seleccionarse antes del ajuste de la puerta")
    require_true(
        ((decision_at >= fit_start) & (decision_at < fit_end)).all(),
        "Una decisión queda fuera del tramo de ajuste",
    )
    require_true(
        ((maturity_at > decision_at) & (maturity_at < fit_end)).all(),
        "Las etiquetas deben madurar después de decidir y antes del final del ajuste",
    )
    if type(ridge) not in (int, float) or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("La regularización ridge debe ser positiva y finita")
    with torch.no_grad():
        gains = signed_gains(predictions, targets).double()
        x, weights = features.double(), session_weights.double()
        weights = weights / weights.amax()
        weights = weights / weights.sum()
        mean = (weights[:, None] * x).sum(0)
        centered = x - mean
        scale = (weights[:, None] * centered.square()).sum(0).sqrt()
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        x = torch.cat((centered / scale, torch.ones_like(x[:, :1])), dim=1)
        gram = x.T @ (weights[:, None] * x)
        # También se contrae el intercepto hacia ganancia cero, una decisión conservadora
        # explícita. No se interpreta ridge como una cota de confianza.
        gram = gram + float(ridge) * torch.eye(len(FEATURES) + 1, device=x.device, dtype=x.dtype)
        coefficients = torch.linalg.solve(gram, x.T @ (weights[:, None] * gains))
        policy = RoutePolicy(
            mean=mean,
            scale=scale,
            coefficients=coefficients,
            support_min=features.double().amin(0),
            support_max=features.double().amax(0),
            costs=costs.to(device=x.device, dtype=x.dtype),
            min_gain=min_gain,
        )
    receipt = dict(
        schema_version=1,
        target="absolute_error_k1_minus_absolute_error_route",
        rows=len(features),
        decision_sessions=int(torch.unique(decision_at).numel()),
        parent_selected_before=parent_selected_before,
        fit_start=fit_start,
        fit_end=fit_end,
        maximum_maturity_at=int(maturity_at.max().item()),
        ridge=float(ridge),
        normalization="weighted_population_moments_fit_only",
        intercept="ridge_penalized_towards_zero_gain",
        solve_dtype="torch.float64",
        policy=policy.get_extra_state(),
    )
    return policy, receipt
