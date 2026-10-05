"""Admisión y recuperación CUDA real de un trabajador en otra sesión."""

import json
import os
import sys

import pytest

from mars_titan.training import gpu_supervisor as engine


@pytest.mark.skipif(
    os.environ.get("MARS_TITAN_CUDA_INTEGRATION") != "1",
    reason="La integración CUDA requiere una ventana exclusiva declarada",
)
def test_nested_cuda_process_confirms_pause_and_resumes(tmp_path):
    import torch

    assert torch.cuda.is_available(), "La comprobación exige CUDA disponible"
    initial = engine.read_gpu()
    assert engine.memory_reason(initial, 6656) is None, (
        "La GPU tiene otra carga o margen insuficiente"
    )
    worker = tmp_path / "worker.py"
    worker.write_text(
        """
import json,os,signal,sys,time
from pathlib import Path
import torch
folder=Path(sys.argv[1])
assert torch.cuda.is_available()
stopping=False
def stop(*_):
    global stopping
    stopping=True
signal.signal(signal.SIGTERM,stop)
device=torch.device('cuda:0')
values=torch.arange(4096,dtype=torch.float64,device=device)
torch.cuda.synchronize(device)
checkpoint=folder/'checkpoint.pt'
if checkpoint.exists():
    restored=torch.load(checkpoint,weights_only=True,map_location=device)
    assert torch.equal(values,restored)
    (folder/'resumed.json').write_text(json.dumps({
        'checksum':restored.sum().item(),
        'device':str(device),
        'peak_bytes':torch.cuda.max_memory_allocated(device)}))
else:
    (folder/'ready').write_text(str(os.getpid()))
    while not stopping: time.sleep(.01)
    temporary=folder/'checkpoint.pending'
    torch.save(values.cpu(),temporary)
    os.replace(temporary,checkpoint)
    raise SystemExit(2)
""",
        encoding="utf-8",
    )
    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        """
import signal,subprocess,sys,time
stopping=False
def stop(*_):
    global stopping
    stopping=True
signal.signal(signal.SIGTERM,stop)
child=subprocess.Popen([sys.executable,sys.argv[1],sys.argv[2]],start_new_session=True)
forwarded=False
while child.poll() is None:
    if stopping and not forwarded:
        child.terminate()
        forwarded=True
    time.sleep(.01)
raise SystemExit(child.returncode)
""",
        encoding="utf-8",
    )
    observed = set()

    def probe():
        snapshot = engine.read_gpu()
        ready = tmp_path / "ready"
        if ready.exists() and not (tmp_path / "checkpoint.pt").exists():
            pid = int(ready.read_text())
            assert pid in snapshot.compute_pids
            observed.add(pid)
            return engine.GpuSnapshot(snapshot.total_mib, 0, snapshot.compute_pids)
        return snapshot

    result = engine.supervise(
        [sys.executable, str(launcher), str(worker), str(tmp_path)],
        tmp_path / "supervisor.json",
        probe=probe,
        poll_seconds=0.1,
        cooldown_seconds=0,
        pause_timeout=20,
        pause_exit_codes=(0, 2),
    )
    assert result == 0 and observed
    state = json.loads((tmp_path / "supervisor.json").read_text())
    restored = json.loads((tmp_path / "resumed.json").read_text())
    assert state["status"] == "completed" and state["starts"] == 2 and state["pauses"] == 1
    assert restored["device"] == "cuda:0"
    assert restored["checksum"] == 4095 * 4096 / 2
    assert restored["peak_bytes"] < 16 * 1024**2
    assert not (observed & set(engine.read_gpu().compute_pids))
