"""Precisión y núcleos comunes a los ajustes neuronales de la campaña.

`apply_kernel_policy` fija en el proceso la precisión declarada por la receta y retira las
sustituciones de `torch._native` que no compensan. La política es la misma para todos los
brazos que comparten ruta y su identidad entra en la de cada ajuste.

La única precisión admitida es `fp32_strict`: FP32 sin TF32 en cuBLAS ni en cuDNN y sin
autocast. PyTorch permite TF32 en cuDNN por defecto, así que una receta sin precisión
declarada conserva el comportamiento anterior y su identidad.

PyTorch 2.14 sustituye en CUDA `aten::bmm` por un núcleo Triton lanzado desde Python cuando
la dimensión interna vale 1 (un producto exterior). La memoria de Titans lo usa en cada
gradiente asociativo. Cada elemento es un único producto redondeado, así que cuBLAS da el
mismo resultado bit a bit en FP32, y su lanzamiento cuesta unas seis veces menos.
"""

import torch

FP32_STRICT = "fp32_strict"
PRECISIONS = (FP32_STRICT,)
# Sustituciones de `torch._native` que se retiran, por símbolo de operación.
RETIRED_NATIVE_OVERRIDES = ("bmm",)


def retire_native_overrides():
    """Retirar las sustituciones de `torch._native` de la política. Es idempotente."""
    try:
        from torch._native import registry
    except ImportError:
        return
    registry.deregister_op_overrides(disable_op_symbols=list(RETIRED_NATIVE_OVERRIDES))


def native_overrides():
    """Sustituciones activas de `torch._native` como `operación/clave/DSL`, ordenadas."""
    try:
        from torch._native import registry
    except ImportError:
        return []
    return sorted(
        f"{node.op_symbol}/{node.dispatch_key}/{node.dsl_name}"
        for nodes in registry._graphs.values()
        for node in nodes
        if registry._filter_state.check_enabled(node)
    )


def apply_kernel_policy(precision):
    """Fijar la precisión declarada y retirar las sustituciones de la política."""
    if precision not in PRECISIONS:
        raise ValueError("La precisión debe ser fp32_strict")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    retire_native_overrides()
    return kernel_policy_identity(precision)


def valid_precision(precision):
    """None conserva la configuración numérica del proceso. Si no, una precisión admitida."""
    return precision is None or (type(precision) is str and precision in PRECISIONS)


def declared_policy(precision):
    """Aplicar la política de una receta. Sin precisión declarada no cambia nada."""
    return None if precision is None else apply_kernel_policy(precision)


def require_policy(precision, identity):
    """Exigir que el proceso siga cumpliendo la política registrada al empezar."""
    if identity is not None and kernel_policy_identity(precision) != identity:
        raise ValueError("El código o la configuración numérica cambiaron durante el recorrido")


def kernel_policy_identity(precision):
    """Identidad de la política vigente. Falla si el proceso no la cumple."""
    if (
        precision not in PRECISIONS
        or torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
        or torch.get_float32_matmul_precision() != "highest"
        or any(item.split("/", 1)[0] in RETIRED_NATIVE_OVERRIDES for item in native_overrides())
    ):
        raise ValueError("El proceso no cumple la política de precisión y núcleos declarada")
    return dict(
        schema_version=1,
        precision=precision,
        matmul_allow_tf32=False,
        cudnn_allow_tf32=False,
        float32_matmul_precision="highest",
        retired_native_overrides=list(RETIRED_NATIVE_OVERRIDES),
        native_overrides=native_overrides(),
    )
