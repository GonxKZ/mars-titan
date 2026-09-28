"""Admisión coherente con el supervisor y margen para los clientes gráficos."""

from types import SimpleNamespace

import pytest
import torch

from mars_titan.training import experiment_resources, gpu_supervisor


@pytest.mark.parametrize("kind,blocked", [("G", False), ("C+G", False), ("C", True)])
def test_gpu_lease_uses_supervisor_classification_and_keeps_memory_margin(
    tmp_path, monkeypatch, kind, blocked
):
    xml = f"""<nvidia_smi_log><gpu><fb_memory_usage><total>8192 MiB</total>
    <free>7680 MiB</free></fb_memory_usage><processes><process_info>
    <pid>12345</pid><type>{kind}</type><process_name>cliente</process_name>
    </process_info></processes></gpu></nvidia_smi_log>"""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(
        gpu_supervisor.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=xml)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device: (int(7.5 * 1024**3), 8 * 1024**3)
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "GPU de prueba")
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    fractions = []
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction", lambda value, device: fractions.append(value)
    )
    if blocked:
        with pytest.raises(RuntimeError, match="otra carga"):
            with experiment_resources.GpuLease():
                pass
    else:
        with experiment_resources.GpuLease() as lease:
            assert lease.record["max_vram_bytes"] == 6 * 1024**3
            assert lease.record["reserved_margin_bytes"] == 1024**3
        assert fractions == [0.75]
