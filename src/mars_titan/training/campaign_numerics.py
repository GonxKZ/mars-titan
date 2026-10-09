"""Precisión numérica declarada de la campaña: FP32 estricto, sin TF32 en cuBLAS ni cuDNN.

Por orden del autor del 9 de octubre de 2026 (no inventar datos ni perder precisión), la
campaña A v2 declara en `numerics` tres indicadores de PyTorch con un único valor admitido:

- `float32_matmul_precision="highest"`,
- `cuda_matmul_allow_tf32=false`,
- `cudnn_allow_tf32=false`.

La campaña de referencias del 6 y 7 de octubre registraba `cudnn_allow_tf32=true`, el valor
por defecto de PyTorch, que permite TF32 en las convoluciones y en los RNN de cuDNN (la GRU,
la LSTM y la RNN). Aquí es un cambio de configuración declarado, no un ajuste por resultados.

El lanzador fija los indicadores antes de crear modelos y otra vez antes de cada trabajo.
Al confirmar exige que sigan igual y que el informe del ejecutor no registre otro valor con
ninguno de los nombres que usan los informes del proyecto. Un recibo sin la precisión
declarada o con otra se rechaza al reanudar.
"""

STRICT_FP32 = dict(
    float32_matmul_precision="highest",
    cuda_matmul_allow_tf32=False,
    cudnn_allow_tf32=False,
)
# Nombres con los que los informes de los ejecutores registran los mismos indicadores.
ALIASES = dict(
    float32_matmul_precision="float32_matmul_precision",
    matmul_precision="float32_matmul_precision",
    cuda_matmul_allow_tf32="cuda_matmul_allow_tf32",
    matmul_allow_tf32="cuda_matmul_allow_tf32",
    matmul_tf32="cuda_matmul_allow_tf32",
    cudnn_allow_tf32="cudnn_allow_tf32",
    cudnn_tf32="cudnn_allow_tf32",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def declared(value):
    """Validar la declaración de una campaña. Solo se admite FP32 estricto."""
    _require(
        value == STRICT_FP32 and all(type(value[k]) is type(v) for k, v in STRICT_FP32.items()),
        "La campaña declara FP32 estricto: float32_matmul_precision=highest, "
        "cuda_matmul_allow_tf32=false y cudnn_allow_tf32=false",
    )
    return dict(value)


def current():
    """Indicadores vigentes en este proceso."""
    import torch

    return dict(
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
    )


def apply(numerics):
    """Fijar los indicadores declarados y comprobar que han quedado así."""
    import torch

    torch.set_float32_matmul_precision(numerics["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = numerics["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = numerics["cudnn_allow_tf32"]
    _require(current() == numerics, "PyTorch no aceptó la precisión numérica declarada")


def recorded(report, numerics, path="informe"):
    """Valores registrados en un informe que contradicen la precisión declarada."""
    found = []
    if isinstance(report, dict):
        for key, value in report.items():
            name = ALIASES.get(key)
            if name is not None and not isinstance(value, (dict, list)):
                if value != numerics[name]:
                    found.append(f"{path}.{key}={value!r}")
                continue
            found += recorded(value, numerics, f"{path}.{key}")
    elif isinstance(report, list):
        for index, value in enumerate(report):
            found += recorded(value, numerics, f"{path}[{index}]")
    return found


def require_job(numerics, label, report):
    """Exigir al confirmar un trabajo los indicadores declarados, vigentes y registrados."""
    _require(
        current() == numerics,
        f"{label} terminó con otra precisión numérica vigente: {current()}",
    )
    found = recorded(report, numerics)
    _require(not found, f"{label} registra otra precisión numérica: {', '.join(found)}")
