"""Identidad de plataforma: lectura de lscpu y nvidia-smi, emulación y huella comparable."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mars_titan.hardware import platform_identity as identity_module
from mars_titan.hardware.platform_identity import (
    cpu_record,
    gpu_records,
    memory_total_bytes,
    platform_identity,
    platform_sha256,
    read_identity,
)
from tests.hardware.platform_doubles import gb10, laptop

# Salida de lscpu --json en la GB10: dos tipos de núcleo Arm anidados bajo su fabricante.
LSCPU_GB10 = {
    "lscpu": [
        {"field": "Architecture:", "data": "aarch64"},
        {"field": "CPU(s):", "data": "20"},
        {
            "field": "Vendor ID:",
            "data": "ARM",
            "children": [
                {"field": "Model name:", "data": "Cortex-X925"},
                {"field": "Model name:", "data": "Cortex-A725"},
                {"field": "Model name:", "data": "Cortex-X925"},
            ],
        },
    ]
}


def completed(stdout):
    return SimpleNamespace(stdout=stdout, returncode=0)


def test_lscpu_nested_models_are_kept_once_and_in_order():
    calls = []

    def run(command, **options):
        calls.append((command, options))
        return completed(json.dumps(LSCPU_GB10))

    torch = SimpleNamespace(
        backends=SimpleNamespace(cpu=SimpleNamespace(get_cpu_capability=lambda: "SVE256"))
    )
    record = cpu_record(run=run, torch=torch)
    assert record["architecture"] == "aarch64"
    assert record["models"] == ["Cortex-X925", "Cortex-A725"]
    assert record["capability"] == "SVE256"
    # Los nombres de campo dependen del idioma, así que lscpu se llama en la configuración C.
    assert calls[0][0] == ["lscpu", "--json"] and calls[0][1]["env"]["LC_ALL"] == "C"


def test_lscpu_without_architecture_or_model_is_rejected():
    output = {"lscpu": [{"field": "Architecture:", "data": "x86_64"}]}
    with pytest.raises(ValueError, match="arquitectura y modelo"):
        cpu_record(run=lambda *a, **k: completed(json.dumps(output)))


def test_nvidia_smi_rows_keep_unified_memory_as_absent():
    rows = (
        "NVIDIA GeForce RTX 4070 Laptop GPU, 8.9, 8188, 595.91.07, GPU-a, 00000000:01:00.0\n"
        "NVIDIA GB10, 12.1, [N/A], 580.95.05, GPU-b, 0000000F:01:00.0\n"
    )
    gpus = gpu_records(run=lambda *a, **k: completed(rows), which=lambda name: "/usr/bin/" + name)
    assert gpus[0]["compute_capability"] == [8, 9] and gpus[0]["memory_total_mib"] == 8188
    assert gpus[1]["compute_capability"] == [12, 1] and gpus[1]["memory_total_mib"] is None
    assert gpus[1]["driver"] == "580.95.05"


def test_no_driver_means_no_gpu_and_a_malformed_row_is_rejected():
    assert gpu_records(which=lambda name: None) == []
    with pytest.raises(ValueError, match="otra lista de campos"):
        gpu_records(run=lambda *a, **k: completed("NVIDIA X, 8.9\n"), which=lambda n: n)


def test_memory_total_comes_from_meminfo(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemFree:  10 kB\nMemTotal:       31556028 kB\n")
    assert memory_total_bytes(meminfo) == 31556028 * 1024
    meminfo.write_text("MemFree:  10 kB\n")
    with pytest.raises(ValueError):
        memory_total_bytes(meminfo)


def test_an_emulated_interpreter_is_marked(monkeypatch):
    # Bajo qemu-user Python dice aarch64, pero lscpu es del anfitrión y dice x86_64.
    monkeypatch.setattr(identity_module.platform, "machine", lambda: "aarch64")
    cpu = dict(architecture="x86_64", models=["AMD Ryzen 9"], logical=16, capability=None)
    record = platform_identity(cpu=cpu, gpus=[], torch=None, memory=1)
    assert record["emulated"] is True and record["torch"] is None
    native = platform_identity(cpu=dict(cpu, architecture="aarch64"), gpus=[], torch=None, memory=1)
    assert native["emulated"] is False
    assert native["platform_sha256"] != record["platform_sha256"]
    assert read_identity(record) == record


@pytest.mark.parametrize(
    "changes",
    [
        dict(kernel="7.1.0"),
        dict(python="3.12.15"),
        dict(memory_total_bytes=64 * 1024**3),
        dict(cpu=dict(laptop()["cpu"], logical=8)),
        dict(
            gpus=[
                dict(laptop()["gpus"][0], driver="600.1", uuid="GPU-other", memory_total_mib=8000)
            ]
        ),
    ],
)
def test_facts_that_do_not_change_numerics_keep_the_digest(changes):
    assert laptop(**changes)["platform_sha256"] == laptop()["platform_sha256"]


@pytest.mark.parametrize(
    "changes",
    [
        dict(machine="aarch64"),
        dict(system="Darwin"),
        dict(emulated=True),
        dict(cpu=dict(laptop()["cpu"], models=["Intel Core i9"])),
        dict(cpu=dict(laptop()["cpu"], capability="AVX2")),
        dict(torch=dict(version="2.15.0+cu130", cuda="13.0", cudnn=92400)),
        dict(torch=dict(version="2.14.0+cu130", cuda="13.1", cudnn=92400)),
        dict(torch=dict(version="2.14.0+cu130", cuda="13.0", cudnn=92500)),
        dict(torch=None),
        dict(gpus=[dict(laptop()["gpus"][0], name="NVIDIA GeForce RTX 4080 Laptop GPU")]),
        dict(gpus=[dict(laptop()["gpus"][0], compute_capability=[8, 6])]),
        dict(gpus=[]),
        dict(gpus=laptop()["gpus"] * 2),
    ],
)
def test_facts_that_can_change_numerics_change_the_digest(changes):
    assert laptop(**changes)["platform_sha256"] != laptop()["platform_sha256"]


def test_a_tampered_or_foreign_identity_is_rejected():
    record = gb10()
    with pytest.raises(ValueError, match="no corresponde a su contenido"):
        read_identity(dict(record, machine="x86_64"))
    with pytest.raises(ValueError, match="contrato"):
        read_identity(dict(record, kind="otra"))
    with pytest.raises(ValueError, match="incompleta"):
        read_identity({k: v for k, v in record.items() if k != "cpu"})
    assert platform_sha256(record) == record["platform_sha256"]


def test_this_machine_reports_a_consistent_identity():
    # La identidad real no crea un contexto CUDA: las GPU salen de nvidia-smi.
    record = platform_identity()
    assert read_identity(record) == record
    assert record["machine"] == identity_module.platform.machine()
    if subprocess.run(["which", "nvidia-smi"], capture_output=True).returncode == 0:
        assert record["gpus"] and all(gpu["name"] for gpu in record["gpus"])


def test_only_the_platform_identity_reads_the_processor_from_proc():
    # En aarch64 el cpuinfo de /proc no tiene `model name`, así que una lectura propia registraría
    # un nombre vacío o la arquitectura. El resto del código usa cpu_name() o cpu_record().
    root = Path(__file__).resolve().parents[2]
    needle = "/proc/" + "cpuinfo"
    allowed = root / "src/mars_titan/hardware/platform_identity.py"
    readers = [
        path.relative_to(root)
        for folder in ("src", "scripts", "benchmarks", "tests", "native")
        for pattern in ("*.py", "*.sh", "*.cpp", "*.hpp")
        for path in (root / folder).rglob(pattern)
        if path != allowed and needle in path.read_text(errors="replace")
    ]
    assert readers == []
