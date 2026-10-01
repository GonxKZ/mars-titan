"""Admisión CUDA y recuperación del proceso sin reservar memoria real."""

import json
import os
import select
import subprocess
import sys
import time

import pytest


def module():
    from mars_titan.training import gpu_supervisor

    return gpu_supervisor


def snapshot(free=7400, processes=()):
    return module().GpuSnapshot(8188, free, tuple(processes))


def test_driver_memory_units_and_compute_processes_are_checked():
    engine = module()
    xml = """<nvidia_smi_log><gpu><fb_memory_usage><total>8188 MiB</total>
    <free>7200 MiB</free></fb_memory_usage><processes><process_info>
    <pid>1234</pid><type>C</type></process_info></processes></gpu></nvidia_smi_log>"""
    result = engine.parse_gpu(xml)
    assert result.total_mib == 8188 and result.free_mib == 7200
    assert result.compute_pids == (1234,)
    for bad in (
        xml.replace("7200 MiB", "N/A"),
        xml.replace("7200 MiB", "9000 MiB"),
        xml.replace("7200 MiB", "7 GiB"),
        "<invalid>",
    ):
        with pytest.raises(ValueError):
            engine.parse_gpu(bad)


def test_desktop_graphics_do_not_masquerade_as_a_scientific_job():
    engine = module()
    xml = """<nvidia_smi_log><gpu><fb_memory_usage><total>8188 MiB</total>
    <free>7100 MiB</free></fb_memory_usage><processes>
    <process_info><pid>1234</pid><type>C+G</type></process_info>
    <process_info><pid>1235</pid><type>G</type></process_info>
    </processes></gpu></nvidia_smi_log>"""
    assert engine.parse_gpu(xml).compute_pids == ()
    assert engine.memory_reason(engine.parse_gpu(xml), 6656) is None


def test_missing_process_telemetry_is_not_treated_as_an_idle_device():
    engine = module()
    for processes in ("", "<processes>N/A</processes>"):
        xml = (
            "<nvidia_smi_log><gpu><fb_memory_usage><total>8188 MiB</total>"
            "<free>7200 MiB</free></fb_memory_usage>" + processes + "</gpu></nvidia_smi_log>"
        )
        with pytest.raises(ValueError, match="procesos"):
            engine.parse_gpu(xml)


def test_admission_rejects_low_memory_and_foreign_compute():
    engine = module()
    assert engine.memory_reason(snapshot(2000), 6656) == "memoria_insuficiente"
    assert engine.memory_reason(snapshot(7400, [os.getpid()]), 6656) == "otro_proceso_cuda"
    assert engine.memory_reason(snapshot(7400, [os.getpid()]), 768, os.getpgrp()) is None


def test_low_memory_waits_without_starting_the_command(tmp_path):
    engine = module()
    stop = engine.StopFlag()
    calls = 0

    def probe():
        nonlocal calls
        calls += 1
        if calls == 3:
            stop.requested = True
        return snapshot(100)

    marker = tmp_path / "started"
    result = engine.supervise(
        [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('started')"],
        tmp_path / "state.json",
        stop=stop,
        probe=probe,
        poll_seconds=0.01,
        cooldown_seconds=0,
    )
    assert result == 0 and not marker.exists()
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "stopped" and state["starts"] == 0


def test_probe_failure_does_not_admit_a_job(tmp_path):
    engine = module()
    stop = engine.StopFlag()

    def probe():
        stop.requested = True
        raise RuntimeError("El controlador no responde")

    marker = tmp_path / "started"
    engine.supervise(
        [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
        tmp_path / "state.json",
        stop=stop,
        probe=probe,
        poll_seconds=0.01,
    )
    assert not marker.exists()


def test_budget_larger_than_the_device_is_blocked(tmp_path):
    engine = module()
    result = engine.supervise(
        [sys.executable, "-c", "raise SystemExit(99)"],
        tmp_path / "state.json",
        probe=lambda: engine.GpuSnapshot(4096, 4096, ()),
        poll_seconds=0.01,
    )
    assert result == 2
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "blocked" and state["starts"] == 0


def test_a_failed_command_is_not_blindly_retried(tmp_path):
    engine = module()
    marker = tmp_path / "attempts"
    code = f"open({str(marker)!r}, 'a').write('attempt\\n'); raise SystemExit(9)"
    result = engine.supervise(
        [sys.executable, "-c", code], tmp_path / "state.json", probe=snapshot, poll_seconds=0.01
    )
    assert result == 9 and marker.read_text().splitlines() == ["attempt"]
    assert json.loads((tmp_path / "state.json").read_text())["status"] == "failed"


def test_low_memory_pauses_its_child_and_resumes_from_a_confirmed_marker(tmp_path):
    engine = module()
    ready, checkpoint, resumed = (tmp_path / n for n in ("ready", "checkpoint", "resumed"))
    code = f"""
import signal,time
from pathlib import Path
checkpoint=Path({str(checkpoint)!r})
if checkpoint.exists():
    Path({str(resumed)!r}).write_text(checkpoint.read_text())
else:
    def stop(*_args):
        checkpoint.write_text('confirmed')
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop)
    Path({str(ready)!r}).write_text('ready')
    while True: time.sleep(.01)
"""
    unaffected = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        result = engine.supervise(
            [sys.executable, "-c", code],
            tmp_path / "state.json",
            probe=lambda: snapshot(100 if ready.exists() and not checkpoint.exists() else 7400),
            poll_seconds=0.01,
            cooldown_seconds=0,
            pause_timeout=2,
        )
        assert result == 0 and resumed.read_text() == "confirmed"
        assert unaffected.poll() is None
        state = json.loads((tmp_path / "state.json").read_text())
        assert state["starts"] == 2 and state["pauses"] == 1
        assert state["status"] == "completed"
    finally:
        unaffected.terminate()
        unaffected.wait(timeout=3)


def test_supervisor_pause_count_is_bounded(tmp_path):
    engine = module()
    stop = engine.StopFlag()
    observed = 0
    ready = tmp_path / "ready"
    code = f"""
import signal,time
from pathlib import Path
def stop(*_):
    Path({str(ready)!r}).unlink()
    raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
Path({str(ready)!r}).write_text('ready')
while True: time.sleep(.01)
"""

    def probe():
        nonlocal observed
        if ready.exists():
            observed += 1
            if observed > 2:
                stop.requested = True
        return snapshot(100 if ready.exists() else 7400)

    result = engine.supervise(
        [sys.executable, "-c", code],
        tmp_path / "state.json",
        probe=probe,
        poll_seconds=0.01,
        cooldown_seconds=0,
        pause_timeout=2,
        max_pauses=2,
        stop=stop,
    )
    state = json.loads((tmp_path / "state.json").read_text())
    assert result == 75 and state["starts"] == 2 and state["pauses"] == 2
    assert state["status"] == "blocked" and state["reason"] == "limite_de_pausas"


def test_a_child_ignoring_sigterm_is_not_restarted(tmp_path):
    engine = module()
    ready = tmp_path / "ready"
    code = f"""
import signal,time
from pathlib import Path
signal.signal(signal.SIGTERM,signal.SIG_IGN)
Path({str(ready)!r}).write_text('ready')
while True: time.sleep(.01)
"""
    result = engine.supervise(
        [sys.executable, "-c", code],
        tmp_path / "state.json",
        probe=lambda: snapshot(100 if ready.exists() else 7400),
        poll_seconds=0.01,
        pause_timeout=0.05,
    )
    state = json.loads((tmp_path / "state.json").read_text())
    assert result == 75 and state["starts"] == 1
    assert state["status"] == "blocked" and state["reason"] == "pausa_no_recuperable"


def test_pause_waits_for_descendant_checkpoint_after_launcher_exits(tmp_path):
    engine = module()
    checkpoint = tmp_path / "confirmed"
    code = f"""
import os,signal,time
from pathlib import Path
if os.fork()==0:
    def stop(*_):
        time.sleep(.2)
        Path({str(checkpoint)!r}).write_text('confirmed')
        os._exit(0)
    signal.signal(signal.SIGTERM,stop)
    print('ready',flush=True)
    while True: time.sleep(.01)
signal.signal(signal.SIGTERM,lambda *_: os._exit(0))
while True: time.sleep(.01)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", code], start_new_session=True, stdout=subprocess.PIPE, text=True
    )
    try:
        assert select.select([child.stdout], [], [], 5)[0]
        assert child.stdout.readline().strip() == "ready"
        started = time.monotonic()
        code, forced = engine._stop_child(child, 3)
        assert code == 0 and not forced
        assert checkpoint.read_text() == "confirmed"
        assert time.monotonic() - started >= 0.2
    finally:
        try:
            os.killpg(child.pid, 9)
        except ProcessLookupError:
            pass
        child.wait(timeout=2)


def test_successful_launcher_with_a_stuck_descendant_is_not_reported_completed(tmp_path):
    engine = module()
    code = """
import os,signal,time
reader,writer=os.pipe()
if os.fork()==0:
    os.close(reader)
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    os.write(writer,b'ready')
    os.close(writer)
    while True: time.sleep(.01)
os.close(writer)
os.read(reader,5)
os.close(reader)
os._exit(0)
"""
    result = engine.supervise(
        [sys.executable, "-c", code],
        tmp_path / "state.json",
        probe=snapshot,
        poll_seconds=0.01,
        pause_timeout=0.1,
    )
    assert result == 75
    assert json.loads((tmp_path / "state.json").read_text())["status"] == "blocked"


def test_signal_stops_the_cli_while_waiting_without_launching(tmp_path):
    engine = module()
    state = tmp_path / "state.json"
    probe = tmp_path / "nvidia-smi"
    probe.write_text(
        f"#!{sys.executable}\nprint('<nvidia_smi_log><gpu><fb_memory_usage>"
        "<total>8188 MiB</total><free>100 MiB</free></fb_memory_usage>"
        "<processes/></gpu></nvidia_smi_log>')\n"
    )
    probe.chmod(0o700)
    child = subprocess.Popen(
        [
            sys.executable,
            str(engine.__file__),
            "--state",
            str(state),
            "--min-free-mib",
            "6656",
            "--poll-seconds",
            ".01",
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(99)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
    )
    try:
        assert select.select([child.stdout], [], [], 5)[0]
        assert json.loads(child.stdout.readline())["status"] == "waiting"
        child.terminate()
        output, error = child.communicate(timeout=5)
        assert child.returncode == 0, (output, error)
        record = json.loads(state.read_text())
        assert record["starts"] == 0 and record["status"] == "stopped"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=2)


@pytest.mark.parametrize(
    "options",
    [
        dict(min_free_mib=0),
        dict(reserve_mib=7000),
        dict(poll_seconds=float("nan")),
        dict(max_pauses=0),
    ],
)
def test_invalid_admission_configuration_fails_before_launch(tmp_path, options):
    with pytest.raises(ValueError):
        module().supervise([sys.executable, "-c", "pass"], tmp_path / "state.json", **options)
