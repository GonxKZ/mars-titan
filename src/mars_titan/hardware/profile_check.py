"""Comprobación técnica de una máquina frente a su perfil de hardware, con recibo JSON.

Pensada para la primera ejecución en una máquina nueva, como la DGX GB10. No ajusta nada:
no hay pasos de optimizador, ni ajustes de señales, ni lectura de datos de la campaña. Las
comprobaciones son estas, en orden:

1. `profile`: la plataforma detectada corresponde al perfil y, con memoria unificada, el
   proceso corre en un cgroup con el límite del perfil. Si falla, no se sigue.
2. `cuda_binaries`: PyTorch trae código para la capacidad de la GPU, un cubin de la misma
   versión mayor y menor igual o inferior, o PTX de una capacidad no superior.
3. `strict_fp32`: con FP32 estricto, un producto de matrices fijo en la GPU coincide con su
   referencia FP64 en CPU dentro de la tolerancia de FP32.
4. `boosting`: CuPy suma en la GPU y XGBoost se compiló con CUDA.
5. `native` (opcional): CTest de una compilación nativa, sin las pruebas con la etiqueta
   `optimizer-steps`, que aplican pasos de optimizador.

Ejemplo en la GB10, con los límites del perfil aplicados por systemd:

    systemd-run --user --scope -p MemoryMax=64G -p CPUQuota=800% \\
        uv run --no-sync python -m mars_titan.hardware.profile_check \\
        --profile dgx-gb10 --output gb10-check.json --native-build build/native/native-ppo-release
"""

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.storage import atomic_json

from . import hardware_profiles
from .platform_identity import platform_identity

KIND = "mars_titan_hardware_check"
# Error relativo admitido en la norma de Frobenius de un producto FP32 de 256 x 256 frente a
# FP64. El error de redondeo esperado ronda 1e-6.
FP32_TOLERANCE = 1e-5


def binaries_cover(arch_list, capability):
    """Si las arquitecturas de un binario CUDA cubren una capacidad de GPU.

    Un cubin `sm_XY` se ejecuta en la misma versión mayor con menor igual o superior. Las
    variantes `sm_XYa` y `sm_XYf` son específicas de su arquitectura y no se trasladan. El
    PTX `compute_XY` se compila al cargar en cualquier capacidad igual o superior.
    """
    major, minor = capability
    for arch in arch_list:
        kind, _, digits = arch.partition("_")
        if not digits.isdigit():
            continue
        arch_major, arch_minor = int(digits[:-1]), int(digits[-1])
        if kind == "sm" and arch_major == major and arch_minor <= minor:
            return True
        if kind == "compute" and (arch_major, arch_minor) <= (major, minor):
            return True
    return False


def _cuda_binaries(torch):
    capability = list(torch.cuda.get_device_capability(0))
    arch_list = torch.cuda.get_arch_list()
    covered = binaries_cover(arch_list, capability)
    return covered, dict(capability=capability, arch_list=arch_list)


def _strict_fp32(torch):
    from mars_titan.training import campaign_numerics

    campaign_numerics.apply(campaign_numerics.STRICT_FP32)
    generator = torch.Generator().manual_seed(500)
    left = torch.randn(256, 256, generator=generator, dtype=torch.float64)
    right = torch.randn(256, 256, generator=generator, dtype=torch.float64)
    reference = left @ right
    device = (left.float().cuda() @ right.float().cuda()).double().cpu()
    error = float(torch.linalg.norm(device - reference) / torch.linalg.norm(reference))
    return error <= FP32_TOLERANCE, dict(
        relative_error=error,
        tolerance=FP32_TOLERANCE,
        numerics=campaign_numerics.current(),
    )


def _profile(profile, identity):
    hardware_profiles.check_profile(profile, identity)
    return True, dict(name=profile["name"], memory=profile["memory"])


def _boosting():
    import cupy
    import xgboost

    total = int(cupy.arange(1, 101, dtype=cupy.int64).sum().get())
    info = xgboost.build_info()
    passed = total == 5050 and bool(info.get("USE_CUDA"))
    return passed, dict(
        cupy=cupy.__version__,
        cupy_sum=total,
        xgboost=xgboost.__version__,
        xgboost_cuda=bool(info.get("USE_CUDA")),
    )


def _native(build, *, run=subprocess.run):
    result = run(
        [
            "ctest",
            "--test-dir",
            str(build),
            "--label-exclude",
            "optimizer-steps",
            "--output-on-failure",
            "--no-tests=error",
        ],
        capture_output=True,
        text=True,
    )
    tail = result.stdout.strip().splitlines()[-3:]
    return result.returncode == 0, dict(build=str(build), returncode=result.returncode, tail=tail)


def _run(name, function, *args):
    try:
        passed, detail = function(*args)
    except Exception as error:  # noqa: BLE001 - el recibo debe registrar cualquier fallo
        return dict(
            name=name, status="failed", detail=dict(error=f"{type(error).__name__}: {error}")
        )
    return dict(name=name, status="passed" if passed else "failed", detail=detail)


def check_machine(profile_name, *, native_build=None, identity=None, torch=None):
    """Recibo de la comprobación. `identity` y `torch` sustituyen detecciones en las pruebas."""
    started = datetime.now(UTC).isoformat()
    profile = hardware_profiles.load_profile(profile_name)
    identity = platform_identity() if identity is None else identity
    checks = [_run("profile", _profile, profile, identity)]
    if checks[0]["status"] == "passed":
        if torch is None:
            import torch
        torch.set_num_threads(min(profile["cpu_threads"], 4))
        if torch.cuda.is_available():
            checks.append(_run("cuda_binaries", _cuda_binaries, torch))
            checks.append(_run("strict_fp32", _strict_fp32, torch))
            checks.append(_run("boosting", _boosting))
        else:
            checks.append(
                dict(name="cuda", status="failed", detail=dict(error="PyTorch no ve ninguna GPU"))
            )
        if native_build is not None:
            checks.append(_run("native", _native, Path(native_build)))
    return dict(
        schema_version=1,
        kind=KIND,
        status="passed" if all(c["status"] == "passed" for c in checks) else "failed",
        profile=dict(name=profile["name"], sha256=profile["sha256"]),
        platform=identity,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        checks=checks,
        started_at_utc=started,
        finished_at_utc=datetime.now(UTC).isoformat(),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", required=True, help="Nombre del perfil en configs/hardware")
    parser.add_argument("--output", type=Path, required=True, help="Recibo JSON de la comprobación")
    parser.add_argument("--native-build", type=Path, help="Compilación nativa para CTest")
    args = parser.parse_args(argv)
    receipt = check_machine(args.profile, native_build=args.native_build)
    atomic_json(args.output, receipt)
    for check in receipt["checks"]:
        print(f"{check['name']}: {check['status']}")
    print(f"{receipt['status']}: {args.output}")
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
