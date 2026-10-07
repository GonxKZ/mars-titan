"""Admisión CUDA y recuperación del proceso sin reservar memoria real."""

import errno
import json
import os
import select
import subprocess
import sys
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest


def module():
    from mars_titan.training import gpu_supervisor

    return gpu_supervisor


def snapshot(free=7400, processes=()):
    return module().GpuSnapshot(8188, free, tuple(processes))


@pytest.mark.parametrize("error_number", [errno.ENOENT, errno.ESRCH])
def test_process_info_treats_a_disappearing_process_as_absent(monkeypatch, error_number):
    engine = module()

    def disappeared(path):
        raise OSError(error_number, os.strerror(error_number), str(path))

    monkeypatch.setattr(engine.Path, "read_text", disappeared)
    assert engine._process_info(1234) is None


@pytest.mark.parametrize("error_number", [errno.EACCES, errno.EPERM, errno.EIO])
def test_process_info_preserves_permission_and_io_errors(monkeypatch, error_number):
    engine = module()
    error = OSError(error_number, os.strerror(error_number))

    def failed(path):
        raise error

    monkeypatch.setattr(engine.Path, "read_text", failed)
    with pytest.raises(OSError) as failure:
        engine._process_info(1234)
    assert failure.value is error


def test_process_exit_between_open_and_read_returns_absent(monkeypatch):
    engine = module()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    path = engine.Path(f"/proc/{child.pid}/stat")
    original = engine.Path.open

    def open_then_exit(current, *args, **kwargs):
        stream = original(current, *args, **kwargs)
        if current == path:
            child.terminate()
            child.wait(timeout=3)
        return stream

    try:
        monkeypatch.setattr(engine.Path, "open", open_then_exit)
        assert engine._process_info(child.pid) is None
        assert child.returncode is not None
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=3)


def test_stop_child_confirms_checkpoint_when_process_exits_during_stat_read(tmp_path, monkeypatch):
    engine = module()
    checkpoint, gate = tmp_path / "checkpoint", tmp_path / "gate"
    code = f"""
import signal,time
from pathlib import Path
def stop(*_):
    Path({str(checkpoint)!r}).write_text('confirmed')
    while not Path({str(gate)!r}).exists(): time.sleep(.001)
    raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
print('ready',flush=True)
while True: time.sleep(.01)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", code], start_new_session=True, stdout=subprocess.PIPE, text=True
    )
    path = engine.Path(f"/proc/{child.pid}/stat")
    original, observed = engine.Path.open, []

    def open_then_finish(current, *args, **kwargs):
        stream = original(current, *args, **kwargs)
        if current == path and checkpoint.exists():
            gate.touch()
            child.wait(timeout=3)
            observed.append(current)
        return stream

    try:
        assert select.select([child.stdout], [], [], 5)[0]
        assert child.stdout.readline().strip() == "ready"
        monkeypatch.setattr(engine.Path, "open", open_then_finish)
        result = engine._stop_child(child, 3)
        assert result == (0, False, True)
        assert checkpoint.read_text() == "confirmed"
        assert observed == [path]
    finally:
        gate.touch()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=3)


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


def test_compute_descendant_in_a_new_session_is_owned():
    engine = module()
    code = """
import signal,subprocess,sys,time
worker=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
def stop(*_):
    worker.terminate()
    worker.wait(timeout=3)
    raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
print(worker.pid,flush=True)
while True: time.sleep(.01)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", code], start_new_session=True, stdout=subprocess.PIPE, text=True
    )
    try:
        assert select.select([child.stdout], [], [], 5)[0]
        worker = int(child.stdout.readline())
        assert os.getpgid(worker) != child.pid
        assert engine.memory_reason(snapshot(7400, [worker]), 768, child.pid) is None
        assert engine.memory_reason(snapshot(7400, [os.getpid()]), 768, child.pid) == (
            "otro_proceso_cuda"
        )
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_stale_process_generation_does_not_signal_a_live_process():
    engine = module()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        generation = engine._process_info(child.pid)[2]
        engine._signal_process(child.pid, generation + 1, 15)
        with pytest.raises(subprocess.TimeoutExpired):
            child.wait(timeout=0.1)
    finally:
        child.terminate()
        child.wait(timeout=3)


def test_recycled_group_leader_does_not_admit_an_unrelated_tree(monkeypatch):
    engine = module()
    table = {
        1234: (1, 1234, 99, "S"),
        2222: (1234, 1234, 100, "S"),
        3337: (1, 3337, 5, "S"),
    }
    monkeypatch.setattr(
        engine.os,
        "scandir",
        lambda _: nullcontext(iter(SimpleNamespace(name=str(pid)) for pid in table)),
    )
    monkeypatch.setattr(engine, "_process_info", table.get)
    assert engine._owned_processes(1234, {1234: 11, 3337: 5}) == {3337: 5}


def test_live_roots_wait_for_owned_ancestors_and_ignore_recycled_generations(monkeypatch):
    engine = module()
    table = {
        100: (1, 100, 10, "S"),
        200: (100, 200, 20, "S"),
        300: (200, 300, 30, "S"),
        400: (1, 400, 40, "S"),
        500: (400, 500, 51, "S"),
    }
    monkeypatch.setattr(engine, "_process_info", table.get)
    owned = {100: 10, 200: 20, 300: 30, 400: 40, 500: 50}
    assert engine._live_roots(owned) == {100: 10, 400: 40}
    table[100] = (1, 100, 10, "Z")
    assert engine._live_roots(owned) == {200: 20, 400: 40}
    table[100] = (1, 100, 10, "S")
    assert engine._live_roots({100: 10, 300: 30}) == {100: 10}


def test_spontaneous_exit_during_probe_is_not_restarted_as_a_pause(tmp_path):
    engine = module()
    ready, gate = tmp_path / "ready", tmp_path / "gate"
    code = f"""
import os,time
from pathlib import Path
ready=Path({str(ready)!r})
gate=Path({str(gate)!r})
if gate.exists(): raise SystemExit(2)
ready.write_text(str(os.getpid()))
while not gate.exists(): time.sleep(.001)
raise SystemExit(2)
"""
    triggered = False

    def probe():
        nonlocal triggered
        if ready.exists() and not triggered:
            triggered = True
            pid = int(ready.read_text())
            gate.write_text("salida espontánea")
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                info = engine._process_info(pid)
                if info is None or info[3] == "Z":
                    break
                time.sleep(0.001)
            return snapshot(100)
        return snapshot()

    result = engine.supervise(
        [sys.executable, "-c", code],
        tmp_path / "state.json",
        probe=probe,
        poll_seconds=0.01,
        cooldown_seconds=0,
        pause_timeout=0.1,
        pause_exit_codes=(0, 2),
    )
    state = json.loads((tmp_path / "state.json").read_text())
    assert result == 2 and state["status"] == "failed"
    assert state["starts"] == 1 and state["pauses"] == 0


def test_declared_pause_exit_code_resumes_only_after_requested_pause(tmp_path):
    engine = module()
    ready, checkpoint, resumed = (tmp_path / n for n in ("ready", "checkpoint", "resumed"))
    code = f"""
import signal,time
from pathlib import Path
checkpoint=Path({str(checkpoint)!r})
if checkpoint.exists():
    Path({str(resumed)!r}).write_text(checkpoint.read_text())
else:
    def stop(*_):
        checkpoint.write_text('confirmed')
        raise SystemExit(2)
    signal.signal(signal.SIGTERM,stop)
    Path({str(ready)!r}).write_text('ready')
    while True: time.sleep(.01)
"""
    result = engine.supervise(
        [sys.executable, "-c", code],
        tmp_path / "state.json",
        probe=lambda: snapshot(100 if ready.exists() and not checkpoint.exists() else 7400),
        poll_seconds=0.01,
        cooldown_seconds=0,
        pause_timeout=2,
        pause_exit_codes=(0, 2),
    )
    assert result == 0 and resumed.read_text() == "confirmed"
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["starts"] == 2 and state["pauses"] == 1
    assert state["status"] == "completed"
    result = engine.supervise(
        [sys.executable, "-c", "raise SystemExit(2)"],
        tmp_path / "failed.json",
        probe=snapshot,
        poll_seconds=0.01,
        pause_exit_codes=(0, 2),
    )
    assert result == 2
    assert json.loads((tmp_path / "failed.json").read_text())["status"] == "failed"


def test_pause_waits_for_checkpoint_in_a_descendant_session(tmp_path):
    engine = module()
    checkpoint = tmp_path / "confirmed"
    code = f"""
import os,signal,time
from pathlib import Path
if os.fork()==0:
    os.setsid()
    def stop(*_):
        time.sleep(.2)
        Path({str(checkpoint)!r}).write_text('confirmed')
        os._exit(0)
    signal.signal(signal.SIGTERM,stop)
    print(os.getpid(),flush=True)
    while True: time.sleep(.01)
signal.signal(signal.SIGTERM,lambda *_: os._exit(0))
while True: time.sleep(.01)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", code], start_new_session=True, stdout=subprocess.PIPE, text=True
    )
    worker = None
    try:
        assert select.select([child.stdout], [], [], 5)[0]
        worker = int(child.stdout.readline())
        started = time.monotonic()
        code, forced, signalled = engine._stop_child(child, 3)
        assert code == 0 and not forced and signalled
        assert checkpoint.read_text() == "confirmed"
        assert time.monotonic() - started >= 0.2
    finally:
        for pid in (child.pid, worker):
            if pid:
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass
        child.wait(timeout=2)


@pytest.mark.parametrize("orphaned_launcher", [False, True])
def test_cooperative_launcher_does_not_receive_a_duplicate_term_for_its_worker(
    tmp_path, monkeypatch, orphaned_launcher
):
    import signal

    engine = module()
    ready, closing, checkpoint, gate = (
        tmp_path / name for name in ("ready", "closing", "checkpoint", "gate")
    )
    result_file = tmp_path / "worker-result"
    worker_file = tmp_path / "worker.py"
    worker_file.write_text(f"""
import os,signal,time
from pathlib import Path
stopping=False
def stop(*_):
    global stopping
    stopping=True
signal.signal(signal.SIGTERM,stop)
Path({str(ready)!r}).write_text(str(os.getpid()))
while not stopping: time.sleep(.001)
Path({str(checkpoint)!r}).write_text('confirmed')
signal.signal(signal.SIGTERM,signal.SIG_DFL)
Path({str(closing)!r}).write_text('closing')
while not Path({str(gate)!r}).exists(): time.sleep(.001)
raise SystemExit(2)
""")
    launcher = f"""
import signal,subprocess,sys,time
from pathlib import Path
stopping=False
def stop(*_):
    global stopping
    stopping=True
signal.signal(signal.SIGTERM,stop)
worker=subprocess.Popen([sys.executable,{str(worker_file)!r}],start_new_session=True)
while not Path({str(ready)!r}).exists(): time.sleep(.001)
print(worker.pid,flush=True)
forwarded=False
while worker.poll() is None:
    if stopping and not forwarded:
        worker.terminate()
        forwarded=True
    time.sleep(.001)
Path({str(result_file)!r}).write_text(str(worker.returncode))
raise SystemExit(worker.returncode)
"""
    if orphaned_launcher:
        launcher_file = tmp_path / "launcher.py"
        launcher_file.write_text(launcher)
        launcher = f"""
import os,signal,subprocess,sys,time
signal.signal(signal.SIGTERM,lambda *_: os._exit(0))
child=subprocess.Popen([sys.executable,{str(launcher_file)!r}],start_new_session=True,stdout=subprocess.PIPE,text=True)
print(child.pid,child.stdout.readline().strip(),flush=True)
while True: time.sleep(.001)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", launcher], start_new_session=True, stdout=subprocess.PIPE, text=True
    )
    signal_process, owned_processes = engine._signal_process, engine._owned_processes
    generations = {}
    try:
        assert select.select([child.stdout], [], [], 5)[0]
        descendants = [int(pid) for pid in child.stdout.readline().split()]
        launcher_pid, worker = descendants if orphaned_launcher else (child.pid, descendants[0])
        generations = {
            pid: engine._process_info(pid)[2] for pid in {child.pid, launcher_pid, worker}
        }

        def acknowledged_signal(pid, generation, sig):
            delivered = signal_process(pid, generation, sig)
            if pid in {launcher_pid, worker} and sig == signal.SIGTERM and delivered:
                deadline = time.monotonic() + 2
                while not closing.exists() and time.monotonic() < deadline:
                    time.sleep(0.001)
                assert closing.exists(), "El lanzador no propagó la señal de cierre"
            return delivered

        def release_after_signal_pass(*args, **kwargs):
            if closing.exists():
                gate.touch()
            return owned_processes(*args, **kwargs)

        monkeypatch.setattr(engine, "_signal_process", acknowledged_signal)
        monkeypatch.setattr(engine, "_owned_processes", release_after_signal_pass)
        code, forced, signalled = engine._stop_child(child, 3)
        assert checkpoint.read_text() == "confirmed"
        assert code == (0 if orphaned_launcher else 2) and not forced and signalled
        assert int(result_file.read_text()) == 2
    finally:
        gate.touch()
        for pid, generation in generations.items():
            signal_process(pid, generation, signal.SIGKILL)
        child.wait(timeout=3)


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
        code, forced, signalled = engine._stop_child(child, 3)
        assert code == 0 and not forced and signalled
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
        dict(pause_exit_codes=()),
        dict(pause_exit_codes=(True,)),
        dict(pause_exit_codes=(-1,)),
        dict(pause_exit_codes=(256,)),
        dict(pause_exit_codes=2),
    ],
)
def test_invalid_admission_configuration_fails_before_launch(tmp_path, options):
    with pytest.raises(ValueError):
        module().supervise([sys.executable, "-c", "pass"], tmp_path / "state.json", **options)
