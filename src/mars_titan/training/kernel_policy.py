"""Precisión y núcleos comunes a los ajustes neuronales de la campaña.

`apply_kernel_policy` fija en el proceso la precisión declarada por la receta y retira las
sustituciones de `torch._native` que no compensan. La política es la misma para todos los
modelos que comparten ruta y su identidad entra en la de cada ajuste.

`fp32_strict` es FP32 sin TF32 en cuBLAS ni en cuDNN y sin autocast. PyTorch permite TF32 en
cuDNN por defecto, así que una receta sin precisión declarada conserva el comportamiento
anterior y su identidad.

`fp32_ieee_attention_v1` añade la atención. Con tensores FP32 en CUDA,
`scaled_dot_product_attention` elige por defecto el núcleo de memoria eficiente de CUTLASS
(`fmha_cutlassF_f32`), que no pasa por cuBLAS ni obedece a las banderas de TF32. En la GB10 su
error máximo frente a la atención en FP64 es de 5·10⁻⁷, el doble que el del núcleo matemático
(2,5·10⁻⁷), y PyTorch no declara determinista su pasada inversa. Esta precisión deja activo solo
el núcleo matemático, softmax(QKᵀ/√d)·V con productos de cuBLAS en FP32, para todo el proceso:
el núcleo, los lectores y cualquier extensión que llame a la atención. Las secuencias de MAC son
de una decena de tokens, así que la matriz de atención explícita no cuesta memoria.

PyTorch 2.14 sustituye en CUDA `aten::bmm` por un núcleo Triton lanzado desde Python cuando
la dimensión interna vale 1 (un producto exterior). La memoria de Titans lo usa en cada
gradiente asociativo. Cada elemento es un único producto redondeado, así que cuBLAS da el
mismo resultado bit a bit en FP32, y su lanzamiento cuesta unas seis veces menos.
"""

import torch

FP32_STRICT = "fp32_strict"
FP32_IEEE_ATTENTION = "fp32_ieee_attention_v1"
PRECISIONS = (FP32_STRICT, FP32_IEEE_ATTENTION)
# Núcleos de atención activos con `fp32_ieee_attention_v1`: solo el matemático.
MATH_ONLY = dict(flash=False, mem_efficient=False, cudnn=False, math=True)
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


def attention_backends():
    """Núcleos de `scaled_dot_product_attention` que el proceso tiene activos."""
    cuda = torch.backends.cuda
    return dict(
        flash=cuda.flash_sdp_enabled(),
        mem_efficient=cuda.mem_efficient_sdp_enabled(),
        cudnn=cuda.cudnn_sdp_enabled(),
        math=cuda.math_sdp_enabled(),
    )


def apply_kernel_policy(precision):
    """Fijar la precisión declarada y retirar las sustituciones de la política."""
    if precision not in PRECISIONS:
        raise ValueError("La precisión debe ser fp32_strict o fp32_ieee_attention_v1")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if precision == FP32_IEEE_ATTENTION:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
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
        or (precision == FP32_IEEE_ATTENTION and attention_backends() != MATH_ONLY)
    ):
        raise ValueError("El proceso no cumple la política de precisión y núcleos declarada")
    identity = dict(
        schema_version=1,
        precision=precision,
        matmul_allow_tf32=False,
        cudnn_allow_tf32=False,
        float32_matmul_precision="highest",
        retired_native_overrides=list(RETIRED_NATIVE_OVERRIDES),
        native_overrides=native_overrides(),
    )
    if precision == FP32_IEEE_ATTENTION:
        # Solo aparece con esta precisión, así que la identidad de fp32_strict no cambia.
        identity["attention_backends"] = dict(MATH_ONLY)
    return identity
