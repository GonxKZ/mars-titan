"""Determinismo declarado de PyTorch para los ajustes de la segunda ola (#794).

`torch.use_deterministic_algorithms(True)` obliga a PyTorch a usar implementaciones
deterministas y lanza un error en las operaciones que no tienen ninguna, en lugar de devolver
resultados que cambian entre ejecuciones. En CUDA, cuBLAS solo es determinista con un espacio
de trabajo fijo, que se declara en `CUBLAS_WORKSPACE_CONFIG` antes de crear su primer handle, y
cuDNN necesita además `deterministic=True` y `benchmark=False`.

La política no toca la aritmética declarada. Se combina con el FP32 estricto de
`kernel_policy` y su identidad entra en la de cada ajuste que la declara. Durante el recorrido
no necesita otra comprobación: `_check_runtime` compara `_numerics`, que ya registra estas
banderas y el espacio de trabajo. Sin declararla, el proceso y la identidad quedan como estaban.
"""

import os
from pathlib import Path

import torch

TORCH_DETERMINISTIC = "torch_deterministic_algorithms_v1"
POLICIES = (TORCH_DETERMINISTIC,)
# Los dos espacios de trabajo que la documentación de cuBLAS declara deterministas.
WORKSPACES = (":4096:8", ":16:8")
WORKSPACE_VARIABLE = "CUBLAS_WORKSPACE_CONFIG"


def valid_determinism(policy):
    """None conserva el proceso. Si no, una política admitida."""
    return policy is None or (type(policy) is str and policy in POLICIES)


def _initial_workspace():
    """Espacio de trabajo del entorno con que arrancó el proceso, o None si no se puede leer.

    `/proc/self/environ` conserva el entorno del arranque, así que distingue una variable
    heredada de otra que se haya escrito después en `os.environ`.
    """
    try:
        entries = Path("/proc/self/environ").read_bytes().split(b"\0")
    except OSError:
        return None
    prefix = WORKSPACE_VARIABLE.encode() + b"="
    values = [entry[len(prefix) :].decode() for entry in entries if entry.startswith(prefix)]
    return values[-1] if values else None


def _workspace():
    """Espacio de trabajo vigente de cuBLAS, exigiendo que llegue antes de iniciar CUDA."""
    workspace = os.environ.get(WORKSPACE_VARIABLE)
    if workspace not in WORKSPACES:
        raise ValueError(f"Configura {WORKSPACE_VARIABLE}=:4096:8 antes de iniciar PyTorch")
    # Si CUDA ya está iniciada, cuBLAS pudo crear su handle antes de leer la variable. Solo vale
    # entonces la que el proceso heredó al arrancar.
    if torch.cuda.is_initialized() and _initial_workspace() != workspace:
        raise ValueError(f"{WORKSPACE_VARIABLE} llegó después de iniciar CUDA")
    return workspace


def determinism_identity(policy):
    """Identidad de la política vigente. Falla si el proceso no la cumple."""
    if (
        policy not in POLICIES
        or not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()
        or not torch.backends.cudnn.deterministic
        or torch.backends.cudnn.benchmark
    ):
        raise ValueError("El proceso no cumple el determinismo declarado")
    return dict(
        schema_version=1,
        policy=policy,
        deterministic_algorithms=True,
        warn_only=False,
        cudnn_deterministic=True,
        cudnn_benchmark=False,
        cublas_workspace=_workspace(),
    )


def apply_determinism(policy):
    """Activar el determinismo declarado en el proceso y devolver su identidad."""
    if policy not in POLICIES:
        raise ValueError(f"El determinismo debe ser {TORCH_DETERMINISTIC}")
    _workspace()
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return determinism_identity(policy)


def declared_determinism(policy):
    """Aplicar la política de una receta. Sin política declarada no cambia nada."""
    return None if policy is None else apply_determinism(policy)
