"""Pruebas TDD para la comprobación de CUDA MPS en Colab y su alternativa segura."""

from __future__ import annotations

import json
from types import SimpleNamespace

import mps_probe as mp
import pytest


def _nvidia_xml(
    *,
    name: str = "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    compute_mode: str = "Default",
    mig_current: str = "Disabled",
    total_mib: int = 97887,
    free_mib: int = 96000,
    processes_xml: str = "",
) -> str:
    return f"""<?xml version="1.0" ?>
<nvidia_smi_log>
  <gpu id="00000000:00:04.0">
    <product_name>{name}</product_name>
    <compute_mode>{compute_mode}</compute_mode>
    <mig_mode>
      <current_mig>{mig_current}</current_mig>
    </mig_mode>
    <fb_memory_usage>
      <total>{total_mib} MiB</total>
      <free>{free_mib} MiB</free>
    </fb_memory_usage>
    <processes>{processes_xml}</processes>
  </gpu>
</nvidia_smi_log>"""


def test_parse_nvidia_smi_mps_context_reads_mode_mig_memory_and_mps_server():
    procs = """
    <process_info>
      <pid>4321</pid>
      <type>M+C</type>
      <process_name>/usr/bin/nvidia-cuda-mps-server</process_name>
    </process_info>
    <process_info>
      <pid>5555</pid>
      <type>C</type>
      <process_name>python3</process_name>
    </process_info>"""
    ctx = mp.parse_nvidia_smi_mps_context(
        _nvidia_xml(compute_mode="Exclusive_Process", mig_current="Disabled", processes_xml=procs)
    )
    assert ctx["gpu_name"] == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    assert ctx["compute_mode"] == "Exclusive_Process"
    assert ctx["mig_enabled"] is False
    assert ctx["total_mib"] == 97887
    assert ctx["free_mib"] == 96000
    assert ctx["mps_server_active"] is True
    assert ctx["other_compute_pids"] == [5555]

    with pytest.raises(ValueError, match="XML"):
        mp.parse_nvidia_smi_mps_context("<not-valid")


def test_prerequisites_detect_missing_binary_prohibited_mode_and_mig_conflict(tmp_path):
    pipe = tmp_path / "mps-pipe"
    log = tmp_path / "mps-log"

    # 1. Falta binario nvidia-cuda-mps-control
    r_missing = mp.check_mps_prerequisites(
        which=lambda _cmd: None,
        run=lambda *a, **k: SimpleNamespace(returncode=0, stdout=_nvidia_xml(), stderr=""),
        geteuid=lambda: 0,
        pipe_dir=str(pipe),
        log_dir=str(log),
    )
    assert r_missing["allowed"] is False
    assert any("nvidia-cuda-mps-control" in b for b in r_missing["blockers"])

    # 2. Modo Prohibited
    r_prohib = mp.check_mps_prerequisites(
        which=lambda _cmd: "/usr/bin/nvidia-cuda-mps-control",
        run=lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=_nvidia_xml(compute_mode="Prohibited"), stderr=""
        ),
        geteuid=lambda: 0,
        pipe_dir=str(pipe),
        log_dir=str(log),
    )
    assert r_prohib["allowed"] is False
    assert any("Prohibited" in b for b in r_prohib["blockers"])

    # 3. Conflicto MIG + Exclusive_Process
    r_mig = mp.check_mps_prerequisites(
        which=lambda _cmd: "/usr/bin/nvidia-cuda-mps-control",
        run=lambda *a, **k: SimpleNamespace(
            returncode=0,
            stdout=_nvidia_xml(compute_mode="Exclusive_Process", mig_current="Enabled"),
            stderr="",
        ),
        geteuid=lambda: 0,
        pipe_dir=str(pipe),
        log_dir=str(log),
    )
    assert r_mig["allowed"] is False
    assert any("MIG" in b for b in r_mig["blockers"])

    # 4. Entorno Colab válido (root, Default, sin MIG)
    r_ok = mp.check_mps_prerequisites(
        which=lambda _cmd: "/usr/bin/nvidia-cuda-mps-control",
        run=lambda *a, **k: SimpleNamespace(
            returncode=0,
            stdout=_nvidia_xml(compute_mode="Default", mig_current="Disabled"),
            stderr="",
        ),
        geteuid=lambda: 0,
        pipe_dir=str(pipe),
        log_dir=str(log),
    )
    assert r_ok["allowed"] is True
    assert r_ok["blockers"] == []


def test_probe_mps_lifecycle_starts_checks_and_cleanly_quits(tmp_path):
    pipe = tmp_path / "pipe"
    log = tmp_path / "log"
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append((cmd, kwargs.get("input")))
        if cmd == ["nvidia-cuda-mps-control", "-d"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if kwargs.get("input") == "get_server_list\n":
            return SimpleNamespace(returncode=0, stdout="12345\n", stderr="")
        if kwargs.get("input") == "get_default_active_thread_percentage\n":
            return SimpleNamespace(returncode=0, stdout="100\n", stderr="")
        if kwargs.get("input") == "quit\n":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"Comando inesperado: {cmd} {kwargs}")

    result = mp.probe_mps_lifecycle(run=fake_run, pipe_dir=str(pipe), log_dir=str(log))
    assert result["operational"] is True
    assert result["started_cleanly"] is True
    assert result["responded_to_server_list"] is True
    assert result["stopped_cleanly"] is True
    assert result["active_thread_pct"] == 100
    assert any(inp == "quit\n" for _, inp in seen)


def test_probe_mps_lifecycle_handles_cannot_find_and_permission_denied(tmp_path):
    pipe = tmp_path / "pipe"
    log = tmp_path / "log"

    # Arranque denegado por contenedor sin permisos
    def denied_run(cmd, **kwargs):
        return SimpleNamespace(
            returncode=1, stdout="", stderr="Permission denied: cannot start daemon"
        )

    res_denied = mp.probe_mps_lifecycle(run=denied_run, pipe_dir=str(pipe), log_dir=str(log))
    assert res_denied["operational"] is False
    assert res_denied["started_cleanly"] is False
    assert "Permission denied" in (res_denied["error"] or "")

    # Arranque OK pero get_server_list dice "Cannot find" -> debe fallar y enviar quit
    calls = []

    def cannot_find_run(cmd, **kwargs):
        calls.append(kwargs.get("input"))
        if cmd == ["nvidia-cuda-mps-control", "-d"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if kwargs.get("input") == "get_server_list\n":
            return SimpleNamespace(
                returncode=0, stdout="", stderr="Cannot find MPS control daemon process\n"
            )
        if kwargs.get("input") == "quit\n":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="unexpected")

    res_cf = mp.probe_mps_lifecycle(run=cannot_find_run, pipe_dir=str(pipe), log_dir=str(log))
    assert res_cf["operational"] is False
    assert res_cf["responded_to_server_list"] is False
    assert "quit\n" in calls


def test_probe_rejects_symlink_pipe_or_log_dir(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="enlace"):
        mp.probe_mps_lifecycle(pipe_dir=str(link), log_dir=str(tmp_path / "log"))


def test_fallback_plan_caps_workers_preserves_fp32_and_rejects_intra_process_multi_model():
    plan = mp.build_fallback_plan(
        workers=6,
        vram_total_gib=96.0,
        host_ram_total_gib=96.0,
        per_job_vram_gib=12.0,
        per_job_ram_gib=17.0,
        mps_blocked_reason="daemon blocked in container",
    )
    assert plan["mode"] == "fallback_independent_processes"
    assert plan["primary_strategy"] == "independent_processes_cuda_default_timeslicing"
    assert plan["packing_strategy"] == "single_origin_page_cache_packing"
    assert plan["streams_policy"] == "intra_process_h2d_prefetch_only"
    assert plan["multi_model_same_process_allowed"] is False
    # 96 GiB RAM - 12 GiB reserve = 84 GiB -> floor(84 / 17) = 4 workers (capped below 6)
    assert plan["safe_workers"] == 4
    assert plan["bottleneck"] == "host_ram"
    env = plan["environment"]
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert env["TRITON_F32_DEFAULT"] == "ieee"
    assert "CUDA_MPS_PIPE_DIRECTORY" not in env
    assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" not in env


def test_evaluate_mps_or_fallback_switches_cleanly_and_cli_dry_run(tmp_path, capsys):
    # Cuando MPS está bloqueado, cae automáticamente a alternativa segura
    decision = mp.evaluate_mps_or_fallback(
        workers=4,
        host_ram_total_gib=96.0,
        probe_live=False,
        which=lambda _cmd: None,
        run=lambda *a, **k: SimpleNamespace(returncode=0, stdout=_nvidia_xml(), stderr=""),
        geteuid=lambda: 1000,
        pipe_dir=str(tmp_path / "p"),
        log_dir=str(tmp_path / "l"),
    )
    assert decision["selected_mode"] == "fallback_independent_processes"
    assert decision["mps_operational"] is False
    assert decision["fp32_strict_preserved"] is True

    code = mp.main(["--dry-run", "--workers", "4", "--host-ram-gib", "96"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["fp32_strict_preserved"] is True
    assert "selected_mode" in payload


def test_pass3_adversarial_mps_keep_running_success_vram_bottleneck_and_errors(tmp_path):
    pipe = tmp_path / "pipe_ok"
    log = tmp_path / "log_ok"
    inputs_seen = []

    def ok_run(cmd, **kwargs):
        inputs_seen.append((cmd, kwargs.get("input")))
        if cmd == ["nvidia-smi", "-q", "-x", "-i", "0"]:
            return SimpleNamespace(returncode=0, stdout=_nvidia_xml(), stderr="")
        if cmd == ["nvidia-cuda-mps-control", "-d"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if kwargs.get("input") == "get_server_list\n":
            return SimpleNamespace(returncode=0, stdout="7777\n", stderr="")
        if kwargs.get("input") == "get_default_active_thread_percentage\n":
            return SimpleNamespace(returncode=0, stdout="100\n", stderr="")
        if kwargs.get("input") == "quit\n":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"Inesperado: {cmd} {kwargs}")

    # keep_running=True en sonda exitosa NO debe enviar quit
    life_keep = mp.probe_mps_lifecycle(
        run=ok_run, pipe_dir=str(pipe), log_dir=str(log), keep_running=True
    )
    assert life_keep["operational"] is True
    assert life_keep["stopped_cleanly"] is False
    assert all(inp != "quit\n" for _, inp in inputs_seen)

    # evaluate_mps_or_fallback con probe_live=True exitoso selecciona mps_enabled y conserva FP32
    dec_ok = mp.evaluate_mps_or_fallback(
        workers=4,
        host_ram_total_gib=96.0,
        probe_live=True,
        keep_running=False,
        which=lambda _cmd: "/usr/bin/nvidia-cuda-mps-control",
        run=ok_run,
        geteuid=lambda: 0,
        pipe_dir=str(pipe),
        log_dir=str(log),
    )
    assert dec_ok["selected_mode"] == "mps_enabled"
    assert dec_ok["mps_operational"] is True
    assert dec_ok["fp32_strict_preserved"] is True
    assert dec_ok["active_environment"]["CUDA_MPS_PIPE_DIRECTORY"] == str(pipe)
    assert dec_ok["active_environment"]["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] == "0=12288M"

    # Cuello de botella en VRAM (24 GiB VRAM, 180 GiB RAM -> floor((24-4)/12.5) = 1 worker)
    vram_plan = mp.build_fallback_plan(
        workers=4,
        vram_total_gib=24.0,
        host_ram_total_gib=180.0,
        per_job_vram_gib=12.0,
        per_job_ram_gib=17.0,
    )
    assert vram_plan["bottleneck"] == "vram"
    assert vram_plan["safe_workers"] == 1

    # nvidia-smi lanza OSError y pipe_dir es enlace simbólico -> ambos bloquean en prerequisites
    real_dir = tmp_path / "real_p"
    real_dir.mkdir()
    sym_pipe = tmp_path / "sym_p"
    sym_pipe.symlink_to(real_dir)

    def broken_smi(*_a, **_k):
        raise OSError("nvidia-smi no encontrado")

    prereq_bad = mp.check_mps_prerequisites(
        which=lambda _cmd: "/usr/bin/nvidia-cuda-mps-control",
        run=broken_smi,
        geteuid=lambda: 0,
        pipe_dir=str(sym_pipe),
        log_dir=str(log),
    )
    assert prereq_bad["allowed"] is False
    assert any("enlace simbólico" in b for b in prereq_bad["blockers"])
    assert any("nvidia-smi" in b for b in prereq_bad["blockers"])
