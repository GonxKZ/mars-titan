"""Trazas de lo que aprende un adaptador, para el contrato común de #448.

Los entrenadores del postentrenamiento (`posttraining.run.run_case` y los entrenadores
cronológicos de Titans-MAC, los lectores y la GRU candidata) aceptan un gancho
`trace(event, modules)` que se llama después de cada validación completa, incluida la de la
época cero. `event` lleva la época, las actualizaciones aplicadas y la puntuación, y
`modules` los módulos con parámetros del recorrido. Estas funciones son las medidas propias
de los adaptadores:

- `update_statistics`: para cada tensor adaptado, la actualización efectiva
  ΔW = W' − W, con su norma de Frobenius, su norma relativa a W, su máximo absoluto y sus
  mayores valores singulares. En los adaptadores por módulo, las normas de sus pesos.
- `prediction_change`: cambio de los cinco cuantiles frente al padre congelado en las mismas
  filas, con la distancia media entre cuantiles y la fracción de medianas que cambian de
  signo.

Leen pesos y predicciones sin gradiente y no consumen números aleatorios. Con el gancho
desactivado los entrenadores no llaman a nada, así que el cálculo es el mismo bit a bit.
Las pruebas lo comprueban con el gancho activado. Ninguna traza interviene en la selección.
"""

import torch
from torch.nn.utils import parametrize

from mars_titan.models.predictive_adaptation import MODULE_ROOT
from mars_titan.models.quantile_head import LEVELS, MEDIAN_INDEX

SINGULAR_VALUES = 8


def _norms(delta, original):
    reference = torch.linalg.vector_norm(original)
    norm = torch.linalg.vector_norm(delta)
    return dict(
        frobenius=float(norm),
        relative=float(norm / reference) if float(reference) > 0 else None,
        max_abs=float(delta.abs().max()),
    )


@torch.no_grad()
def update_statistics(model, *, top=SINGULAR_VALUES):
    """Normas y espectro de la actualización efectiva de cada tensor adaptado, en float64."""
    if type(top) is not int or not 1 <= top <= 64:
        raise ValueError("El número de valores singulares de la traza no es válido")
    rows = []
    for module_name, module in model.named_modules():
        if not parametrize.is_parametrized(module):
            continue
        for tensor, chain in module.parametrizations.items():
            original = chain.original.detach().double()
            delta = getattr(module, tensor).detach().double() - original
            row = dict(module=module_name, tensor=tensor, shape=list(original.shape))
            row.update(_norms(delta, original))
            if delta.ndim == 2:
                values = torch.linalg.svdvals(delta.cpu())
                row["singular_values"] = [float(value) for value in values[:top]]
            rows.append(row)
    adapters = getattr(model, MODULE_ROOT, None)
    for name, adapter in adapters.items() if adapters is not None else ():
        for tensor, value in adapter.named_parameters():
            rows.append(
                dict(
                    module=f"{MODULE_ROOT}.{name}",
                    tensor=tensor,
                    shape=list(value.shape),
                    frobenius=float(torch.linalg.vector_norm(value.detach().double())),
                )
            )
    return rows


@torch.no_grad()
def prediction_change(adapted, parent):
    """Cambio de los cuantiles [filas, 5] del brazo frente a los del padre en las mismas filas.

    `quantile_distance` es la media entre filas de la media de |q_k − q_k^padre| sobre los
    cinco niveles: una aproximación con pesos iguales de la distancia de Wasserstein-1
    entre las dos distribuciones predictivas. No es una pérdida ni usa objetivos.
    """
    if (
        not isinstance(adapted, torch.Tensor)
        or not isinstance(parent, torch.Tensor)
        or adapted.shape != parent.shape
        or adapted.ndim != 2
        or adapted.shape[1] != len(LEVELS)
        or not len(adapted)
    ):
        raise ValueError("La traza compara cinco cuantiles por fila del brazo y del padre")
    change = (adapted.double() - parent.double()).abs()
    median = adapted[:, MEDIAN_INDEX], parent[:, MEDIAN_INDEX]
    flips = torch.sign(median[0]) != torch.sign(median[1])
    return dict(
        rows=len(adapted),
        quantile_distance=float(change.mean()),
        mean_abs_change=[float(value) for value in change.mean(dim=0)],
        max_abs_change=float(change.max()),
        median_sign_flips=float(flips.double().mean()),
    )
