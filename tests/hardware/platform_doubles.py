"""Identidades de plataforma fijas para las pruebas, sin detectar la máquina que las ejecuta.

`laptop` reproduce lo que detecta el portátil de desarrollo. `desktop` es otra máquina x86-64
inventada para las pruebas, con otra CPU, otra GPU y más memoria, que sirve para comprobar que
una plataforma distinta se rechaza.
"""

from mars_titan.hardware.platform_identity import KIND, SCHEMA_VERSION, platform_sha256

GIB = 1024**3


def _gpu(name, capability, memory, uuid):
    return dict(
        name=name,
        compute_capability=capability,
        memory_total_mib=memory,
        driver="595.91.07",
        uuid=uuid,
        pci_bus_id="00000000:01:00.0",
    )


def identity(**changes):
    """Identidad válida con su huella recalculada tras aplicar los cambios."""
    record = dict(
        schema_version=SCHEMA_VERSION,
        kind=KIND,
        system="Linux",
        machine="x86_64",
        kernel="7.0.0-38-generic",
        python="3.12.14",
        emulated=False,
        cpu=dict(
            architecture="x86_64",
            models=["AMD Ryzen 9 8945HS w/ Radeon 780M Graphics"],
            logical=16,
            capability="AVX512",
        ),
        memory_total_bytes=30 * GIB,
        torch=dict(version="2.14.0+cu130", cuda="13.0", cudnn=92400),
        gpus=[_gpu("NVIDIA GeForce RTX 4070 Laptop GPU", [8, 9], 8188, "GPU-laptop")],
    )
    record.update(changes)
    record["platform_sha256"] = platform_sha256(record)
    return record


def laptop(**changes):
    return identity(**changes)


def desktop(**changes):
    return identity(
        **dict(
            dict(
                cpu=dict(
                    architecture="x86_64",
                    models=["AMD Ryzen 9 7950X 16-Core Processor"],
                    logical=32,
                    capability="AVX512",
                ),
                memory_total_bytes=62 * GIB,
                gpus=[_gpu("NVIDIA GeForce RTX 4090", [8, 9], 24564, "GPU-desktop")],
            ),
            **changes,
        )
    )
