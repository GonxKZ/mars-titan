"""Orden de comprobación de una máquina: cobertura CUDA, parada temprana y recibo JSON.

Las pruebas sustituyen la plataforma y PyTorch. No crean contextos CUDA ni ajustan nada.
"""

import json
from types import SimpleNamespace

import pytest

from mars_titan.hardware import profile_check
from mars_titan.hardware.profile_check import binaries_cover, check_machine
from tests.hardware.platform_doubles import gb10, laptop


@pytest.mark.parametrize(
    ("arch_list", "capability", "covered"),
    [
        # Rueda cu130 de PyTorch para aarch64: el cubin sm_120 cubre la GB10 (12.1).
        (["sm_80", "sm_90", "sm_100", "sm_110", "sm_120", "compute_120"], [12, 1], True),
        (["sm_80", "sm_90", "sm_100", "sm_120"], [12, 1], True),
        (["sm_80", "sm_86", "sm_90"], [8, 9], True),
        # Un cubin de menor superior o de otra versión mayor no sirve.
        (["sm_122"], [12, 1], False),
        (["sm_90", "sm_100"], [12, 1], False),
        # Las variantes específicas de arquitectura no se trasladan.
        (["sm_90a", "sm_120a", "sm_120f"], [12, 1], False),
        # El PTX se compila al cargar en una capacidad igual o superior.
        (["compute_90"], [12, 1], True),
        (["compute_121"], [12, 1], True),
        (["compute_122"], [12, 1], False),
        ([], [8, 9], False),
    ],
)
def test_cuda_binaries_cover_a_capability(arch_list, capability, covered):
    assert binaries_cover(arch_list, capability) is covered


class FakeTorch:
    """PyTorch sustituto: registra los hilos y declara si ve una GPU."""

    def __init__(self, available, arch_list=(), capability=(8, 9)):
        self.threads = None
        self.cuda = SimpleNamespace(
            is_available=lambda: available,
            get_arch_list=lambda: list(arch_list),
            get_device_capability=lambda index: capability,
        )

    def set_num_threads(self, count):
        self.threads = count


def test_a_platform_outside_the_profile_stops_after_the_first_check():
    torch = FakeTorch(True)
    receipt = check_machine("dgx-gb10", identity=laptop(), torch=torch)
    assert receipt["status"] == "failed"
    assert [c["name"] for c in receipt["checks"]] == ["profile"]
    assert "dgx-gb10" in receipt["checks"][0]["detail"]["error"]
    assert torch.threads is None


def test_without_a_visible_gpu_the_check_fails_explicitly():
    torch = FakeTorch(False)
    receipt = check_machine("rtx4070-laptop", identity=laptop(), torch=torch)
    assert receipt["status"] == "failed"
    assert [c["name"] for c in receipt["checks"]] == ["profile", "cuda"]
    assert receipt["checks"][0]["status"] == "passed"
    assert torch.threads == 4


def test_gpu_checks_run_in_order_and_each_failure_is_recorded(monkeypatch):
    monkeypatch.setattr(profile_check, "_strict_fp32", lambda torch: (True, dict(error=0.0)))

    def broken():
        raise ImportError("No module named 'cupy'")

    monkeypatch.setattr(profile_check, "_boosting", broken)
    torch = FakeTorch(True, ["sm_86"], (8, 9))
    receipt = check_machine("rtx4070-laptop", identity=laptop(), torch=torch)
    names = [c["name"] for c in receipt["checks"]]
    assert names == ["profile", "cuda_binaries", "strict_fp32", "boosting"]
    statuses = {c["name"]: c["status"] for c in receipt["checks"]}
    assert statuses == dict(
        profile="passed", cuda_binaries="passed", strict_fp32="passed", boosting="failed"
    )
    assert "cupy" in receipt["checks"][-1]["detail"]["error"]
    assert receipt["status"] == "failed"


def test_native_tests_exclude_optimizer_steps(tmp_path):
    calls = []

    def run(command, **options):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="100% tests passed\nTotal Test time\n")

    passed, detail = profile_check._native(tmp_path, run=run)
    assert passed and detail["returncode"] == 0
    command = calls[0]
    assert command[command.index("--label-exclude") + 1] == "optimizer-steps"
    assert "--no-tests=error" in command


def test_the_receipt_is_written_and_the_exit_code_follows_it(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_check, "platform_identity", lambda: gb10())
    output = tmp_path / "check.json"
    # En esta máquina la identidad sustituta es la de la GB10, pero no corre en un cgroup
    # acotado a 64 GiB, así que el perfil se rechaza y el código de salida es 1.
    monkeypatch.setattr(
        profile_check.hardware_profiles, "cgroup_memory_limit", lambda: None, raising=True
    )
    assert profile_check.main(["--profile", "dgx-gb10", "--output", str(output)]) == 1
    receipt = json.loads(output.read_text())
    assert receipt["kind"] == "mars_titan_hardware_check" and receipt["status"] == "failed"
    assert receipt["platform"]["platform_sha256"] == gb10()["platform_sha256"]
    assert "cgroup" in receipt["checks"][0]["detail"]["error"]
