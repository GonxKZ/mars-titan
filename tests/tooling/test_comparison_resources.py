"""Admisión por trabajo y cierre de sus procesos sin cambiar el experimento."""

import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_comparison_resources.py"


@pytest.fixture
def engine():
    path = Path(os.environ.get("COMPARISON_RESOURCES_TEST_SOURCE", SCRIPT))
    if not path.is_file():
        pytest.fail("Falta el lanzador que reserva RAM por trabajo")
    spec = importlib.util.spec_from_file_location("comparison_resources_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("runner", ["fit", "reader", "ridge", "rule"])
def test_neural_and_small_controls_do_not_reserve_lightgbm_memory(engine, runner):
    assert engine.memory_limit(runner) == 8


def test_lightgbm_keeps_its_measured_memory_margin(engine):
    assert engine.memory_limit("tabular") == 12
    with pytest.raises(ValueError):
        engine.memory_limit("unknown")


def test_available_memory_uses_reclaimable_memory_and_rejects_missing_data(engine, tmp_path):
    path = tmp_path / "meminfo"
    path.write_text("MemFree: 123 kB\nMemAvailable: 12582912 kB\n")
    assert engine.available_memory(path) == 12 * 1024**3
    path.write_text("MemFree: 123 kB\n")
    with pytest.raises(ValueError):
        engine.available_memory(path)


def test_admission_keeps_four_gib_for_the_system(engine):
    assert not engine.memory_ready(8, 12 * 1024**3 - 1)
    assert engine.memory_ready(8, 12 * 1024**3)
    assert not engine.memory_ready(12, 15 * 1024**3)
    assert engine.memory_ready(12, 16 * 1024**3)


def test_admission_of_a_shared_job_uses_its_own_measured_memory(engine):
    assert not engine.memory_ready(3, 7 * 1024**3 - 1)
    assert engine.memory_ready(3, 7 * 1024**3)
    assert engine.memory_ready(1, 5 * 1024**3) and engine.memory_ready(12, 16 * 1024**3)


@pytest.mark.parametrize(
    "limit,available", [(0, 10), (13, 10), (8.0, 10), (True, 10), (8, -1), (8, 1.5)]
)
def test_invalid_admission_values_fail_closed(engine, limit, available):
    with pytest.raises(ValueError):
        engine.memory_ready(limit, available)


def test_running_status_requires_an_owned_process(engine, tmp_path):
    path = tmp_path / "record.json"
    assert engine.running_pid(path, {}) is None
    path.write_text(json.dumps(dict(status="running", attempts=[dict(pid=123)])))
    assert engine.running_pid(path, {123: 1}) == 123
    assert engine.running_pid(path, {124: 1}) is None
    path.write_text(json.dumps(dict(status="completed", attempts=[dict(pid=123)])))
    assert engine.running_pid(path, {123: 1}) is None
    with path.open("wb") as stream:
        stream.truncate(16 * 1024**2 + 1)
    with pytest.raises(ValueError, match="límite"):
        engine.running_pid(path, {})


@pytest.fixture
def launcher(tmp_path):
    path = tmp_path / "resources"
    path.write_text('#!/bin/sh\nshift 4\nexec "$@"\n')
    path.chmod(0o700)
    return path


def run_worker(engine, tmp_path, launcher, code, *, stop=None, probe=None):
    from mars_titan.training.gpu_supervisor import StopFlag

    return engine.run_worker(
        launcher,
        8,
        [sys.executable, "-c", code],
        status=tmp_path / "status.json",
        record=tmp_path / "record.json",
        job="fixture/2021/seed-42",
        stop=stop or StopFlag(),
        probe=probe or (lambda: 20 * 1024**3),
        poll_seconds=0.02,
        stop_timeout=3,
    )


def test_child_failure_is_not_reported_as_success(engine, tmp_path, launcher):
    assert run_worker(engine, tmp_path, launcher, "raise SystemExit(7)") == 7
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["phase"] == "failed" and status["returncode"] == 7


@pytest.mark.parametrize("choice", ["default", "custom", "disabled"])
def test_worker_shares_file_digests_within_the_state(
    engine, tmp_path, launcher, monkeypatch, choice
):
    name = "MARS_TITAN_DIGEST_CACHE"
    if choice == "default":
        monkeypatch.delenv(name, raising=False)
        expected = str(tmp_path / "file-digests.json")
    else:
        expected = "" if choice == "disabled" else str(tmp_path / "custom.json")
        monkeypatch.setenv(name, expected)
    output = tmp_path / "child-environment.json"
    code = (
        "import json, os, pathlib\n"
        f"pathlib.Path({str(output)!r}).write_text(json.dumps(os.environ.get({name!r})))"
    )
    assert run_worker(engine, tmp_path, launcher, code) == 0
    assert json.loads(output.read_text()) == expected


def test_worker_runs_after_memory_becomes_available(engine, tmp_path, launcher):
    readings = iter([11 * 1024**3, 12 * 1024**3])
    output = tmp_path / "executed"
    code = f"from pathlib import Path\nPath({str(output)!r}).write_text('yes')"
    result = run_worker(
        engine, tmp_path, launcher, code, probe=lambda: next(readings, 12 * 1024**3)
    )
    assert result == 0 and output.read_text() == "yes"
    assert json.loads((tmp_path / "status.json").read_text())["limit_gib"] == 8


def test_stop_while_waiting_never_launches_a_job(engine, tmp_path, launcher, monkeypatch):
    from mars_titan.training.gpu_supervisor import StopFlag

    stop = StopFlag()
    output = tmp_path / "unexpected"
    launched = []
    original = engine.subprocess.Popen

    def observe_launch(*args, **kwargs):
        child = original(*args, **kwargs)
        launched.append(child.pid)
        return child

    monkeypatch.setattr(engine.subprocess, "Popen", observe_launch)

    def probe():
        stop.request()
        return 2 * 1024**3

    code = f"from pathlib import Path\nPath({str(output)!r}).touch()"
    assert run_worker(engine, tmp_path, launcher, code, stop=stop, probe=probe) == 3
    assert launched == []
    assert not output.exists()
    assert json.loads((tmp_path / "status.json").read_text())["phase"] == "paused"


def test_stop_confirms_checkpoint_before_returning(engine, tmp_path, launcher):
    from mars_titan.training.gpu_supervisor import StopFlag

    ready, checkpoint = tmp_path / "ready", tmp_path / "checkpoint"
    code = f"""
import signal,time
from pathlib import Path
def stop(*_):
    Path({str(checkpoint)!r}).write_text('confirmed')
    raise SystemExit(3)
signal.signal(signal.SIGTERM, stop)
Path({str(ready)!r}).touch()
while True: time.sleep(.01)
"""
    stop = StopFlag()

    def request():
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        stop.request()

    thread = threading.Thread(target=request)
    thread.start()
    try:
        assert run_worker(engine, tmp_path, launcher, code, stop=stop) == 3
    finally:
        thread.join(timeout=6)
    assert checkpoint.read_text() == "confirmed"


def test_parent_exit_stops_the_job_and_confirms_its_state(engine, tmp_path, launcher, monkeypatch):
    ready, checkpoint = tmp_path / "ready", tmp_path / "checkpoint"
    code = f"""
import signal,time
from pathlib import Path
def stop(*_):
    Path({str(checkpoint)!r}).write_text('confirmed')
    raise SystemExit(3)
signal.signal(signal.SIGTERM, stop)
Path({str(ready)!r}).touch()
while True: time.sleep(.01)
"""
    monkeypatch.setattr(engine.os, "getppid", lambda: 20 if ready.exists() else 10)
    assert run_worker(engine, tmp_path, launcher, code) == 3
    assert checkpoint.read_text() == "confirmed"


@pytest.mark.parametrize("runner,limit", [("fit", "8"), ("reader", "8"), ("tabular", "12")])
def test_queue_wraps_the_original_command_without_changing_its_arguments(
    engine, tmp_path, runner, limit
):
    original = [sys.executable, "-m", "mars_titan.training.comparison_campaign", "job"]

    class Queue:
        def __init__(self, *args, **kwargs):
            assert kwargs["workers"] == 1

        def _command(self, job, *, preflight=False):
            return original + (["--preflight"] if preflight else [])

    layout = SimpleNamespace(state=tmp_path, record=lambda key: tmp_path / "record.json")
    campaign = SimpleNamespace(
        load_declaration=lambda path: ({}, "digest"),
        Layout=lambda *args: layout,
        code_commit=lambda: "commit",
        code_changes=lambda: None,
        authorized=lambda *args, **kwargs: True,
        Queue=Queue,
    )
    args = SimpleNamespace(
        declaration=None,
        state=None,
        views=None,
        machine="references",
        runtime=tmp_path / "runtime",
        resource_launcher=tmp_path / "resources",
    )
    queue = engine.make_queue(campaign, args)
    command = queue._command(SimpleNamespace(runner=runner, key="job"), preflight=True)
    assert command[command.index("--limit-gib") + 1] == limit
    assert command[command.index("--") + 1 :] == [*original, "--preflight"]
    assert command[command.index("--runtime") + 1] == str(args.runtime)


def test_authorization_is_checked_before_constructing_the_queue(engine, tmp_path):
    calls = []
    campaign = SimpleNamespace(
        load_declaration=lambda path: ({}, "digest"),
        Layout=lambda *args: SimpleNamespace(state=tmp_path),
        code_commit=lambda: "commit",
        code_changes=lambda: None,
        authorized=lambda *args, **kwargs: False,
        Queue=lambda *args, **kwargs: calls.append(True),
    )
    args = SimpleNamespace(declaration=None, state=None, views=None, machine="references")
    with pytest.raises(ValueError, match="permiso"):
        engine.make_queue(campaign, args)
    assert calls == []


def test_missing_runtime_fails_before_importing_training_code(engine, tmp_path):
    with pytest.raises(ValueError, match="runtime"):
        engine.load_campaign(tmp_path)


def test_non_executable_resource_launcher_is_rejected_before_the_queue(
    engine, tmp_path, monkeypatch
):
    launcher = tmp_path / "not-executable"
    launcher.write_text("contenido")
    monkeypatch.setattr(engine, "load_campaign", lambda runtime: SimpleNamespace())
    monkeypatch.setattr(
        engine, "make_queue", lambda *args: pytest.fail("Se construyó la cola sin ejecutable")
    )
    with pytest.raises(ValueError, match="ejecutable"):
        engine.main(
            [
                "--runtime",
                str(tmp_path),
                "--resource-launcher",
                str(launcher),
                "--machine",
                "references",
            ]
        )


def test_imported_controller_must_belong_to_the_requested_runtime(engine, tmp_path, monkeypatch):
    path = tmp_path / "src/mars_titan/training/comparison_campaign.py"
    path.parent.mkdir(parents=True)
    path.write_text('"""Runtime de la prueba."""\n')
    monkeypatch.setattr(engine.sys, "path", list(sys.path))
    monkeypatch.setenv("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    monkeypatch.setattr(
        engine.importlib, "import_module", lambda name: SimpleNamespace(__file__=__file__)
    )
    with pytest.raises(ValueError, match="distinto"):
        engine.load_campaign(tmp_path)


def test_cli_help_does_not_import_a_scientific_runtime():
    if not SCRIPT.is_file():
        pytest.fail("Falta el lanzador que reserva RAM por trabajo")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0
    assert "--runtime" in result.stdout


@pytest.mark.parametrize("completed", [True, False])
def test_cli_only_exports_a_complete_queue_and_resolves_the_destination(
    engine, tmp_path, launcher, monkeypatch, completed
):
    exports = []
    queue = SimpleNamespace(
        run=lambda: {},
        request_stop=lambda *args: None,
        layout=SimpleNamespace(state=tmp_path / "state"),
        jobs=["job"],
    )
    campaign = SimpleNamespace(
        _print_summary=lambda result: None,
        job_state=lambda layout, job: "completed" if completed else "paused",
        DONE="completed",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "load_campaign", lambda runtime: campaign)
    monkeypatch.setattr(engine, "make_queue", lambda campaign, args: queue)
    monkeypatch.setattr(engine.signal, "signal", lambda *args: None)

    def export(command, **kwargs):
        exports.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(engine.subprocess, "run", export)
    result = engine.main(
        [
            "--runtime",
            "runtime",
            "--resource-launcher",
            str(launcher),
            "--machine",
            "references",
            "--results",
            "result.zip",
        ]
    )
    assert result == (0 if completed else 3)
    assert len(exports) == int(completed)
    if completed:
        command, kwargs = exports[0]
        assert command[command.index("--destination") + 1] == str(tmp_path / "result.zip")
        assert command[command.index("--state") + 1] == str(tmp_path / "state")
        assert kwargs["cwd"] == tmp_path / "runtime"


def budgets(**changes):
    document = dict(
        schema_version=1,
        device_budget_mib=7000,
        max_jobs=2,
        reserve_mib=200,
        settle_seconds=120,
        max_celsius=84,
        admit_gib=dict(fit=3, reader=3),
        vram_default_mib=dict(fit=4000, reader=1500),
        vram_mib=dict(large=3800, small=3000, reader_model=800),
    )
    document.update(changes)
    return document


def work(model, year=2019, runner="fit", seed=43):
    return SimpleNamespace(model=model, year=year, runner=runner, key=f"{model}/{year}/seed-{seed}")


def seen(**changes):
    values = dict(since_launch=120.0, celsius=84.0, available=7 * 1024**3, free_mib=7000, used={})
    values.update(changes)
    return SimpleNamespace(**values)


def test_an_empty_gpu_admits_any_job_even_without_measurements(engine):
    packing = engine.Packing(budgets())
    blind = seen(since_launch=0.0, celsius=None, available=0, free_mib=0)
    assert packing.admits(work("lightgbm_gpu", runner="tabular"), [], blind)
    assert packing.admits(work("large"), [], blind)


def test_two_measured_jobs_share_the_gpu_at_the_exact_limits(engine):
    packing = engine.Packing(budgets(device_budget_mib=6800))
    running = [work("large")]
    # 3800 + 3000 agotan el presupuesto y la VRAM libre cubre lo que falta más la reserva.
    exact = seen(free_mib=3200, used={running[0].key: 3800})
    assert packing.admits(work("small"), running, exact)
    assert not packing.admits(work("small"), running, seen(**{**vars(exact), "free_mib": 3199}))
    assert not engine.Packing(budgets(device_budget_mib=6799)).admits(work("small"), running, exact)


def test_a_job_that_has_not_reached_its_peak_keeps_its_budget(engine):
    packing = engine.Packing(budgets())
    running = [work("large")]
    # El trabajo en marcha ocupa 1000 MiB de sus 3800: los 2800 restantes siguen reservados.
    assert packing.admits(work("small"), running, seen(free_mib=6000, used={running[0].key: 1000}))
    assert not packing.admits(
        work("small"), running, seen(free_mib=5999, used={running[0].key: 1000})
    )
    # Sin lectura de su proceso se reserva el presupuesto completo.
    assert packing.admits(work("small"), running, seen(free_mib=7000))
    assert not packing.admits(work("small"), running, seen(free_mib=6999))
    # Un proceso que supera su presupuesto no devuelve VRAM a los demás.
    assert not packing.admits(
        work("small"), running, seen(free_mib=3199, used={running[0].key: 5000})
    )


def test_unmeasured_models_use_the_default_of_their_runner_without_a_cap(engine):
    packing = engine.Packing(budgets())
    assert packing.vram(work("new_network")) == 4000 and packing.cap(work("new_network")) is None
    assert packing.vram(work("new_reader", runner="reader")) == 1500
    assert packing.cap(work("small")) == 3000
    # 3800 + 4000 no caben en 7000, y un lector sin medir sí cabe junto al modelo grande.
    assert not packing.admits(work("new_network"), [work("large")], seen())
    assert packing.admits(work("new_reader", runner="reader"), [work("large")], seen())


def test_only_models_with_a_cap_get_the_allocator_their_cap_was_measured_with(engine):
    conf = "max_split_size_mb:128,expandable_segments:True"
    packing = engine.Packing(budgets(cuda_alloc_conf=conf))
    assert packing.cuda_alloc_conf == conf and packing.alloc_conf(work("small")) == conf
    # Sin tope no hay medida hecha con ese asignador, y un ejecutor que va solo no lo necesita.
    assert packing.alloc_conf(work("new_network")) is None
    assert packing.alloc_conf(work("small", runner="tabular")) is None
    plain = engine.Packing(budgets())
    assert plain.cuda_alloc_conf is None and plain.alloc_conf(work("small")) is None
    assert engine.Packing(budgets(cuda_alloc_conf=None)).alloc_conf(work("small")) is None


@pytest.mark.parametrize("runner", ["tabular", "ridge", "rule"])
def test_runners_without_a_shared_budget_run_alone(engine, runner):
    packing = engine.Packing(budgets())
    alone = work("control", runner=runner)
    assert not packing.shares(alone) and packing.cap(alone) is None
    assert not packing.admits(alone, [work("small")], seen())
    assert not packing.admits(work("small"), [alone], seen())


def test_a_new_job_waits_for_the_previous_start_to_settle(engine):
    packing = engine.Packing(budgets())
    assert not packing.admits(work("small"), [work("large")], seen(since_launch=119.9))
    assert packing.admits(work("small"), [work("large")], seen(since_launch=120.0))


def test_heat_or_a_missing_reading_keeps_a_single_job(engine):
    packing = engine.Packing(budgets())
    assert not packing.admits(work("small"), [work("large")], seen(celsius=84.1))
    assert not packing.admits(work("small"), [work("large")], seen(celsius=None))
    free = engine.Packing(budgets(max_celsius=None))
    assert free.admits(work("small"), [work("large")], seen(celsius=None))
    assert free.admits(work("small"), [work("large")], seen(celsius=99.0))


def test_jobs_of_another_origin_wait_for_the_prepared_inputs(engine):
    packing = engine.Packing(budgets())
    assert not packing.admits(work("small", year=2022), [work("large", year=2019)], seen())
    assert not packing.admits(
        work("reader_model", year=2022, runner="reader"), [work("large", year=2019)], seen()
    )
    assert packing.admits(work("reader_model", runner="reader"), [work("large")], seen())


def test_shared_admission_keeps_the_system_memory_reserve(engine):
    packing = engine.Packing(budgets())
    assert not packing.admits(work("small"), [work("large")], seen(available=7 * 1024**3 - 1))
    wide = engine.Packing(budgets(admit_gib=dict(fit=4, reader=3)))
    assert not wide.admits(work("small"), [work("large")], seen())
    assert wide.admits(work("reader_model", runner="reader"), [work("large")], seen())


@pytest.mark.parametrize(
    "changes",
    [
        dict(schema_version=2),
        dict(device_budget_mib=0),
        dict(device_budget_mib=7000.0),
        dict(max_jobs=0),
        dict(max_jobs=None),
        dict(reserve_mib=-1),
        dict(settle_seconds=-1),
        dict(settle_seconds="120"),
        dict(max_celsius="84"),
        dict(admit_gib={}),
        dict(admit_gib=dict(fit=13)),
        dict(admit_gib=dict(fit=0)),
        dict(admit_gib=dict(tabular=3)),
        dict(vram_mib=dict(large=0)),
        dict(vram_mib=[3800]),
        dict(vram_default_mib=dict(fit=True, reader=1500)),
        dict(vram_default_mib=None),
        dict(cuda_alloc_conf=""),
        dict(cuda_alloc_conf="expandable_segments"),
        dict(cuda_alloc_conf="expandable_segments:"),
        dict(cuda_alloc_conf=":True"),
        dict(cuda_alloc_conf="expandable_segments:True,"),
        dict(cuda_alloc_conf="expandable_segments:True max_split_size_mb:128"),
        dict(cuda_alloc_conf="expandable_segments:True\n"),
        dict(cuda_alloc_conf="expandable_segments:$HOME"),
        dict(cuda_alloc_conf=True),
        dict(cuda_alloc_conf=["expandable_segments:True"]),
    ],
)
def test_invalid_budgets_are_rejected_before_the_queue_starts(engine, changes):
    with pytest.raises(ValueError, match="presupuestos"):
        engine.Packing(budgets(**changes))


def test_budgets_must_be_a_small_json_object(engine, tmp_path):
    path = tmp_path / "budgets.json"
    path.write_text(json.dumps(budgets()))
    packing = engine.read_packing(path)
    assert (packing.device_budget_mib, packing.reserve_mib, packing.settle_seconds) == (
        7000,
        200,
        120,
    )
    assert packing.max_celsius == 84 and packing.admit_gib == dict(fit=3, reader=3)
    path.write_text("[]")
    with pytest.raises(ValueError, match="presupuestos"):
        engine.read_packing(path)
    with path.open("wb") as stream:
        stream.truncate(1024**2 + 1)
    with pytest.raises(ValueError, match="límite"):
        engine.read_packing(path)


def test_device_memory_reads_free_memory_and_adds_the_processes(engine):
    calls = []

    def run(command, **options):
        calls.append((command, options))
        text = "7000\n" if "--query-gpu=memory.free" in command else "10, 3000\n11, 40\n10, 5\n"
        return SimpleNamespace(stdout=text)

    assert engine.device_memory(run) == (7000, {10: 3005, 11: 40})
    assert all(command[:2] == ["nvidia-smi", "--id=0"] for command, _ in calls)
    assert all(options["check"] and options["timeout"] == 10 for _, options in calls)
    assert engine.device_memory(
        lambda *a, **k: SimpleNamespace(stdout="512\n" if len(a[0][2]) < 25 else "")
    ) == (512, {})


@pytest.mark.parametrize("free", ["", "7000\n6000\n", "sin dato\n"])
def test_device_memory_rejects_an_unreadable_gpu(engine, free):
    def run(command, **options):
        return SimpleNamespace(stdout=free if "--query-gpu=memory.free" in command else "")

    with pytest.raises(ValueError):
        engine.device_memory(run)


def test_temperature_is_read_in_celsius_only_when_a_zone_is_given(engine, tmp_path):
    zone = tmp_path / "temp"
    zone.write_text("84500\n")
    assert engine.celsius(zone) == 84.5
    assert engine.celsius(None) is None
    with pytest.raises(OSError):
        engine.celsius(tmp_path / "missing")


def test_mps_needs_the_control_socket_and_the_live_daemon_of_the_declared_folder(engine, tmp_path):
    import socket

    declared = dict(CUDA_MPS_PIPE_DIRECTORY=str(tmp_path))
    pid_file = tmp_path / "nvidia-cuda-mps-control.pid"
    pid_file.write_text(f"{os.getpid()}\n")
    assert not engine.mps_ready({})
    assert not engine.mps_ready(dict(CUDA_MPS_PIPE_DIRECTORY=""))
    assert not engine.mps_ready(declared)
    (tmp_path / "control").write_text("no es un socket")
    assert not engine.mps_ready(declared)
    (tmp_path / "control").unlink()
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(str(tmp_path / "control"))
        assert engine.mps_ready(declared)
        # Un demonio que ha muerto deja el socket: sin su proceso no hay servicio.
        finished = subprocess.Popen([sys.executable, "-c", "pass"])
        finished.wait()
        for stale in (str(finished.pid), "0", "-1", "sin número", ""):
            pid_file.write_text(stale)
            assert not engine.mps_ready(declared), stale
        pid_file.unlink()
        assert not engine.mps_ready(declared)
        pid_file.write_text(str(os.getpid()))
        assert engine.mps_ready(declared)


def test_shared_worker_is_admitted_with_its_own_memory_and_gets_its_vram_cap(
    engine, tmp_path, launcher, monkeypatch
):
    from mars_titan.training.gpu_supervisor import StopFlag

    names = ["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT", "PYTORCH_CUDA_ALLOC_CONF"]
    for name in names:
        monkeypatch.delenv(name, raising=False)
    output = tmp_path / "cap.json"
    code = (
        "import json, os, pathlib\n"
        f"seen = [os.environ.get(name) for name in {names!r}]\n"
        f"pathlib.Path({str(output)!r}).write_text(json.dumps(seen))"
    )

    def run(**options):
        return engine.run_worker(
            launcher,
            8,
            [sys.executable, "-c", code],
            status=tmp_path / "status.json",
            record=tmp_path / "record.json",
            job="fixture/2021/seed-42",
            stop=stop,
            probe=probe,
            poll_seconds=0.02,
            stop_timeout=3,
            **options,
        )

    stop, readings = StopFlag(), []

    def probe():
        # Una admisión que siguiera esperando se corta aquí en vez de bloquear la prueba.
        readings.append(7 * 1024**3)
        if len(readings) > 2000:
            stop.request()
        return readings[-1]

    # Con 7 GiB disponibles el límite de 8 GiB esperaría: la admisión propia de 3 GiB no.
    assert run(admit_gib=3, vram_cap_mib=3000) == 0
    assert json.loads(output.read_text()) == ["0=3000M", None]
    assert json.loads((tmp_path / "status.json").read_text())["limit_gib"] == 8
    assert run(admit_gib=3) == 0
    assert json.loads(output.read_text()) == [None, None]
    conf = "expandable_segments:True"
    assert run(admit_gib=3, vram_cap_mib=2900, cuda_alloc_conf=conf) == 0
    assert json.loads(output.read_text()) == ["0=2900M", conf]
    # Un trabajo sin configuración propia conserva la que ya tuviera el entorno.
    monkeypatch.setenv(names[1], "max_split_size_mb:64")
    assert run(admit_gib=3) == 0
    assert json.loads(output.read_text()) == [None, "max_split_size_mb:64"]
    assert run(admit_gib=3, cuda_alloc_conf=conf) == 0
    assert json.loads(output.read_text()) == [None, conf]


def test_shared_worker_still_waits_below_its_own_admission(engine, tmp_path, launcher):
    from mars_titan.training.gpu_supervisor import StopFlag

    stop = StopFlag()
    phases = []

    def probe():
        status = tmp_path / "status.json"
        if status.exists():
            phases.append(json.loads(status.read_text())["phase"])
            stop.request()
        return 7 * 1024**3 - 1

    result = engine.run_worker(
        launcher,
        8,
        [sys.executable, "-c", "raise SystemExit(9)"],
        status=tmp_path / "status.json",
        record=tmp_path / "record.json",
        job="fixture/2021/seed-42",
        stop=stop,
        probe=probe,
        poll_seconds=0.01,
        stop_timeout=3,
        admit_gib=3,
    )
    assert result == 3 and phases == ["waiting_memory"]


def test_a_queued_shared_worker_reports_memory_by_its_own_admission(engine, tmp_path, launcher):
    from mars_titan.training.gpu_supervisor import StopFlag

    gib, stop, phases = 1024**3, StopFlag(), []
    status = tmp_path / "status.json"
    # Admisión, una espera de recursos con RAM para sus 3 GiB y otra sin ella.
    readings = iter([8 * gib, 8 * gib, 7 * gib - 1])

    def probe():
        phases.append(json.loads(status.read_text())["phase"] if status.exists() else None)
        value = next(readings, None)
        if value is None:
            stop.request()
            return 8 * gib
        return value

    result = engine.run_worker(
        launcher,
        8,
        [sys.executable, "-c", "import time\ntime.sleep(30)"],
        status=status,
        record=tmp_path / "record.json",
        job="fixture/2021/seed-42",
        stop=stop,
        probe=probe,
        poll_seconds=0.01,
        stop_timeout=3,
        admit_gib=3,
    )
    # Con 8 GiB el límite del grupo aún esperaría memoria: la admisión propia ya no.
    assert result == 3 and phases == [None, None, "queued_resources", "waiting_memory"]
    assert json.loads(status.read_text())["phase"] == "paused"


def shared_queue(engine, tmp_path, monkeypatch, jobs, *, packing, workers=2, launch=None):
    """Cola de pruebas con la interfaz del controlador que usa el lanzador."""
    records = {job.key: dict(status="pending", attempts=[]) for job in jobs}
    launched = []

    class Queue:
        def __init__(self, declaration, layout, machine, *, workers, declaration_path=None):
            self.layout, self.jobs, self.workers = layout, list(jobs), workers
            self.stop, self.guard = threading.Event(), threading.Lock()
            self.taken, self.poll_seconds = set(), 0.01

        def _command(self, job, *, preflight=False):
            return ["original", job.key]

        def _deadline_reached(self):
            return False

        def _disk_exhausted(self):
            return False

        def _finished(self):
            return all(records[job.key]["status"] == "completed" for job in self.jobs)

        def _launch(self, job, *, preflight=False):
            with self.guard:
                launched.append((job.key, sorted(self.taken - {job.key})))
            records[job.key] = dict(status="running", attempts=[dict(pid=hash(job.key) % 9999)])
            if launch is not None:
                launch(job)
            records[job.key]["status"] = "completed"

    def claim(layout, candidates, taken):
        for job in candidates:
            if job.key not in taken and records[job.key]["status"] == "pending":
                return job, SimpleNamespace(close=lambda: None)
        return None, None

    layout = SimpleNamespace(state=tmp_path, record=lambda key: tmp_path / f"{key}.json")
    campaign = SimpleNamespace(
        load_declaration=lambda path: ({}, "digest"),
        Layout=lambda *args: layout,
        code_commit=lambda: "commit",
        code_changes=lambda: None,
        authorized=lambda *args, **kwargs: True,
        Queue=Queue,
        _claim=claim,
        read_record=lambda layout, key: records[key],
    )
    args = SimpleNamespace(
        declaration=None,
        state=None,
        views=None,
        machine="references",
        runtime=tmp_path / "runtime",
        resource_launcher=tmp_path / "resources",
        workers=workers,
        packing=packing,
        thermal_zone=tmp_path / "zone",
    )
    args.thermal_zone.write_text("50000\n")
    monkeypatch.setattr(engine, "mps_ready", lambda *a: True)
    monkeypatch.setattr(engine, "available_memory", lambda *a: 20 * 1024**3)
    monkeypatch.setattr(engine, "device_memory", lambda *a: (7000, {}))
    return engine.make_queue(campaign, args), records, launched


def test_several_workers_need_measured_budgets_and_a_live_mps_daemon(engine, tmp_path, monkeypatch):
    jobs = [work("large")]
    with pytest.raises(ValueError, match="presupuestos"):
        shared_queue(engine, tmp_path, monkeypatch, jobs, packing=None, workers=2)
    packing = engine.Packing(budgets())
    queue, _, _ = shared_queue(engine, tmp_path, monkeypatch, jobs, packing=packing)
    assert queue.workers == 2
    monkeypatch.setattr(engine, "mps_ready", lambda *a: False)
    args = SimpleNamespace(
        declaration=None, state=None, views=None, machine="references", workers=1, packing=packing
    )
    campaign = SimpleNamespace(
        load_declaration=lambda path: ({}, "digest"),
        Layout=lambda *a: SimpleNamespace(state=tmp_path),
        code_commit=lambda: "commit",
        code_changes=lambda: None,
        authorized=lambda *a, **k: True,
        Queue=lambda *a, **k: pytest.fail("Se construyó la cola sin MPS"),
    )
    with pytest.raises(ValueError, match="MPS"):
        engine.make_queue(campaign, args)


def test_shared_commands_carry_the_admission_and_the_cap_of_each_job(engine, tmp_path, monkeypatch):
    jobs = [work("small"), work("new_network"), work("lightgbm_gpu", runner="tabular")]
    packing = engine.Packing(budgets())
    queue, _, _ = shared_queue(engine, tmp_path, monkeypatch, jobs, packing=packing)

    def options(job):
        command = queue._command(job)
        assert command[command.index("--") + 1 :] == ["original", job.key]
        assert command[command.index("--job") + 1] == job.key
        head = command[: command.index("--status")]
        return head[head.index("--limit-gib") :]

    assert options(jobs[0]) == ["--limit-gib", "8", "--admit-gib", "3", "--vram-cap-mib", "3000"]
    assert options(jobs[1]) == ["--limit-gib", "8", "--admit-gib", "3"]
    assert options(jobs[2]) == ["--limit-gib", "12"]
    conf = "expandable_segments:True"
    packing = engine.Packing(budgets(cuda_alloc_conf=conf))
    queue, _, _ = shared_queue(engine, tmp_path, monkeypatch, jobs, packing=packing)
    capped = ["--limit-gib", "8", "--admit-gib", "3", "--vram-cap-mib", "3000"]
    assert options(jobs[0]) == [*capped, "--cuda-alloc-conf", conf]
    assert options(jobs[1]) == ["--limit-gib", "8", "--admit-gib", "3"]
    assert options(jobs[2]) == ["--limit-gib", "12"]


def test_admissible_jobs_follow_the_plan_and_the_vram_each_process_already_holds(
    engine, tmp_path, monkeypatch
):
    jobs = [work("large", seed=42), work("large"), work("small"), work("small", year=2022)]
    packing = engine.Packing(budgets(settle_seconds=0))
    queue, records, _ = shared_queue(engine, tmp_path, monkeypatch, jobs, packing=packing)
    assert queue._admissible() == jobs
    queue.taken.add(jobs[0].key)
    # El trabajo en marcha aún no ocupa nada: su presupuesto entero sigue reservado.
    assert queue._admissible() == [jobs[2]]
    monkeypatch.setattr(engine, "device_memory", lambda *a: (6999, {}))
    assert queue._admissible() == []
    records[jobs[0].key] = dict(status="running", attempts=[dict(pid=1), dict(pid=77)])
    # Cuenta la VRAM del último intento: con la del primero faltarían 3700 MiB por reservar.
    monkeypatch.setattr(engine, "device_memory", lambda *a: (3200, {77: 3800, 1: 100}))
    assert queue._admissible() == [jobs[2]]
    # Un registro que ya no está en marcha no identifica ningún proceso.
    records[jobs[0].key]["status"] = "pending"
    assert queue._admissible() == []


def test_without_measurements_or_mps_no_job_is_added(engine, tmp_path, monkeypatch, capsys):
    jobs = [work("large"), work("small")]
    packing = engine.Packing(budgets(settle_seconds=0))
    queue, _, _ = shared_queue(engine, tmp_path, monkeypatch, jobs, packing=packing)
    queue.taken.add(jobs[0].key)
    assert queue._admissible() == [jobs[1]]

    def broken(*args):
        raise subprocess.SubprocessError("nvidia-smi no responde")

    monkeypatch.setattr(engine, "device_memory", broken)
    assert queue._admissible() == [] and queue._admissible() == []
    # El mismo impedimento se avisa una vez, no en cada consulta.
    assert capsys.readouterr().out.count("nvidia-smi no responde") == 1
    monkeypatch.setattr(engine, "device_memory", lambda *a: (7000, {}))
    monkeypatch.setattr(engine, "mps_ready", lambda *a: False)
    assert queue._admissible() == [] and queue._admissible() == []
    assert capsys.readouterr().out.count("MPS") == 1
    monkeypatch.setattr(engine, "mps_ready", lambda *a: True)
    assert queue._admissible() == [jobs[1]]
    monkeypatch.setattr(engine, "mps_ready", lambda *a: False)
    assert queue._admissible() == []
    # Tras recuperarse, un fallo nuevo se vuelve a avisar.
    assert capsys.readouterr().out.count("MPS") == 1
    monkeypatch.setattr(engine, "mps_ready", lambda *a: True)
    (tmp_path / "zone").write_text("84001\n")
    assert queue._admissible() == []
    (tmp_path / "zone").unlink()
    assert queue._admissible() == []
    assert "zone" in capsys.readouterr().out
    # Con la GPU vacía la cola no depende de las medidas para empezar un trabajo.
    queue.taken.clear()
    assert queue._admissible() == jobs
    # Pero sin el demonio no empieza ninguno: fallaría al iniciar CUDA y gastaría un intento.
    monkeypatch.setattr(engine, "mps_ready", lambda *a: False)
    assert queue._admissible() == [] and queue._admissible() == []
    assert capsys.readouterr().out.count("no se lanza ningún trabajo") == 1
    monkeypatch.setattr(engine, "mps_ready", lambda *a: True)
    assert queue._admissible() == jobs
    monkeypatch.setattr(engine, "mps_ready", lambda *a: False)
    assert queue._admissible() == []
    assert capsys.readouterr().out.count("no se lanza ningún trabajo") == 1


def test_the_settle_time_counts_from_the_last_start(engine, tmp_path, monkeypatch):
    jobs = [work("large"), work("small")]
    packing = engine.Packing(budgets(settle_seconds=30))
    queue, _, _ = shared_queue(engine, tmp_path, monkeypatch, jobs, packing=packing)
    queue.taken.add(jobs[0].key)
    clock = [1000.0]
    monkeypatch.setattr(engine.time, "monotonic", lambda: clock[0])
    queue.launched = 971.0
    assert queue._admissible() == []
    queue.launched = 970.0
    assert queue._admissible() == [jobs[1]]


def test_workers_pack_a_small_job_next_to_a_large_one_and_log_the_overlap(
    engine, tmp_path, monkeypatch
):
    jobs = [work("large", seed=42), work("large"), work("small")]
    first, second, small = (job.key for job in jobs)
    started, release = threading.Event(), threading.Event()

    def launch(job):
        if job.key == first:
            started.set()
            assert release.wait(5)

    packing = engine.Packing(budgets(settle_seconds=0))
    queue, records, launched = shared_queue(
        engine, tmp_path, monkeypatch, jobs, packing=packing, launch=launch
    )
    one = threading.Thread(target=queue._work, daemon=True)
    one.start()
    assert started.wait(5)
    two = threading.Thread(target=queue._work, daemon=True)
    two.start()
    deadline = time.monotonic() + 5
    while records[small]["status"] != "completed" and time.monotonic() < deadline:
        time.sleep(0.01)
    # El segundo modelo grande no cabe junto al primero y espera sin ocupar un proceso.
    assert launched == [(first, []), (small, [first])]
    assert two.is_alive()
    release.set()
    one.join(5)
    two.join(5)
    assert not one.is_alive() and not two.is_alive()
    assert [key for key, _ in launched] == [first, small, second]
    assert launched[2][1] == [] and queue.taken == set()
    lines = [json.loads(line) for line in (tmp_path / "packing-log.jsonl").read_text().splitlines()]
    events = [(line["event"], line["job"]) for line in lines]
    assert events[:2] == [("start", first), ("start", small)]
    assert sorted(events) == sorted(
        [(name, key) for name in ("start", "end") for key in (first, second, small)]
    )
    starts = {line["job"]: line for line in lines if line["event"] == "start"}
    assert starts[small]["together"] == [first] and starts[small]["vram_cap_mib"] == 3000
    assert starts[first]["together"] == [] and starts[first]["vram_cap_mib"] == 3800
    assert all(line["cuda_alloc_conf"] is None for line in starts.values())
    assert all(set(line) == {"at_utc", "event", "job"} for line in lines if line["event"] == "end")


def test_a_single_worker_without_budgets_keeps_the_original_order_and_writes_no_log(
    engine, tmp_path, monkeypatch
):
    jobs = [work("large", seed=42), work("small", year=2022), work("large")]
    queue, _, launched = shared_queue(engine, tmp_path, monkeypatch, jobs, packing=None, workers=1)
    monkeypatch.setattr(engine, "device_memory", lambda *a: pytest.fail("Sin empaquetado no mide"))
    queue._work()
    assert launched == [(job.key, []) for job in jobs]
    assert not (tmp_path / "packing-log.jsonl").exists()


def test_a_queue_without_the_mps_daemon_ends_without_launching_or_spending_attempts(
    engine, tmp_path, monkeypatch
):
    jobs = [work("large"), work("small")]
    packing = engine.Packing(budgets(settle_seconds=0))
    queue, records, launched = shared_queue(engine, tmp_path, monkeypatch, jobs, packing=packing)
    monkeypatch.setattr(engine, "mps_ready", lambda *a: False)
    queue._work()
    assert launched == [] and queue.taken == set()
    assert all(record == dict(status="pending", attempts=[]) for record in records.values())
    assert not (tmp_path / "packing-log.jsonl").exists()
    # Sin presupuestos la cola no usa MPS y no lo consulta.
    plain, _, started = shared_queue(engine, tmp_path, monkeypatch, jobs, packing=None, workers=1)
    monkeypatch.setattr(engine, "mps_ready", lambda *a: pytest.fail("Sin empaquetado no usa MPS"))
    plain._work()
    assert [key for key, _ in started] == [job.key for job in jobs]


def test_the_overlap_log_records_the_cap_and_the_allocator_of_each_start(
    engine, tmp_path, monkeypatch
):
    conf = "expandable_segments:True"
    jobs = [work("small"), work("new_network", year=2022)]
    packing = engine.Packing(budgets(cuda_alloc_conf=conf))
    queue, _, _ = shared_queue(engine, tmp_path, monkeypatch, jobs, packing=packing, workers=1)
    queue._work()
    lines = [json.loads(line) for line in (tmp_path / "packing-log.jsonl").read_text().splitlines()]
    starts = [line for line in lines if line["event"] == "start"]
    assert [(line["job"], line["vram_cap_mib"], line["cuda_alloc_conf"]) for line in starts] == [
        (jobs[0].key, 3000, conf),
        (jobs[1].key, None, None),
    ]


def test_a_stopped_queue_starts_nothing_and_a_failed_launch_frees_its_place(
    engine, tmp_path, monkeypatch
):
    jobs = [work("large"), work("small")]

    def launch(job):
        raise RuntimeError("fallo del proceso")

    packing = engine.Packing(budgets(settle_seconds=0))
    queue, _, launched = shared_queue(
        engine, tmp_path, monkeypatch, jobs, packing=packing, launch=launch
    )
    queue.stop.set()
    queue._work()
    assert launched == []
    queue.stop.clear()
    with pytest.raises(RuntimeError, match="fallo del proceso"):
        queue._work()
    assert queue.taken == set()
    lines = [json.loads(line) for line in (tmp_path / "packing-log.jsonl").read_text().splitlines()]
    assert [(line["event"], line["job"]) for line in lines] == [
        ("start", jobs[0].key),
        ("end", jobs[0].key),
    ]


def test_cli_reads_the_budgets_before_importing_the_runtime(
    engine, tmp_path, launcher, monkeypatch
):
    path = tmp_path / "budgets.json"
    path.write_text(json.dumps(budgets(schema_version=3)))
    monkeypatch.setattr(
        engine, "load_campaign", lambda runtime: pytest.fail("Se cargó el runtime antes")
    )
    arguments = [
        "--runtime",
        str(tmp_path),
        "--resource-launcher",
        str(launcher),
        "--machine",
        "references",
        "--workers",
        "2",
        "--packing",
        str(path),
    ]
    with pytest.raises(ValueError, match="presupuestos"):
        engine.main(arguments)
    path.write_text(json.dumps(budgets()))
    seen_args = []
    monkeypatch.setattr(engine, "load_campaign", lambda runtime: SimpleNamespace())

    def make_queue(campaign, args):
        seen_args.append(args)
        raise KeyboardInterrupt

    monkeypatch.setattr(engine, "make_queue", make_queue)
    with pytest.raises(KeyboardInterrupt):
        engine.main([*arguments, "--thermal-zone", str(tmp_path / "temp")])
    assert seen_args[0].workers == 2
    assert seen_args[0].packing.current().device_budget_mib == 7000
    assert seen_args[0].thermal_zone == tmp_path / "temp"


def test_worker_cli_passes_the_shared_admission_to_the_worker(engine, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(engine, "runtime_path", lambda runtime: None)
    monkeypatch.setattr(engine.signal, "signal", lambda *args: None)

    def run_worker(launcher, limit, command, **options):
        calls.append((limit, command, options))
        return 0

    monkeypatch.setattr(engine, "run_worker", run_worker)
    base = [
        "worker",
        "--runtime",
        str(tmp_path),
        "--resource-launcher",
        str(tmp_path / "resources"),
        "--limit-gib",
        "8",
    ]
    tail = [
        "--status",
        str(tmp_path / "status.json"),
        "--record",
        str(tmp_path / "record.json"),
        "--job",
        "fixture/2021/seed-42",
        "--",
        "python",
        "-m",
        "job",
    ]
    conf = "expandable_segments:True"
    shared = ["--admit-gib", "3", "--vram-cap-mib", "3000", "--cuda-alloc-conf", conf]
    assert engine.main([*base, *shared, *tail]) == 0
    assert engine.main([*base, *tail]) == 0
    assert calls[0][0] == 8 and calls[0][1] == ["python", "-m", "job"]
    options = [
        (call[2]["admit_gib"], call[2]["vram_cap_mib"], call[2]["cuda_alloc_conf"])
        for call in calls
    ]
    assert options == [(3, 3000, conf), (None, None, None)]


def test_the_number_of_shared_jobs_is_bounded_by_the_budgets(engine):
    readers = [work(f"reader_{index}", runner="reader") for index in range(3)]
    assert not engine.Packing(budgets()).admits(readers[2], readers[:2], seen())
    assert engine.Packing(budgets(max_jobs=3)).admits(readers[2], readers[:2], seen())
    assert not engine.Packing(budgets(max_jobs=1)).admits(readers[1], readers[:1], seen())


def test_budgets_corrected_on_disk_apply_to_the_next_decision(engine, tmp_path, capsys):
    path = tmp_path / "budgets.json"
    path.write_text(json.dumps(budgets()))
    source = engine.PackingFile(path)
    first = source.current()
    assert first.max_jobs == 2 and source.current() is first
    assert capsys.readouterr().out == ""
    path.write_text(json.dumps(budgets(max_jobs=3, vram_mib=dict(large=3900))))
    second = source.current()
    assert (second.max_jobs, second.vram_mib) == (3, dict(large=3900))
    assert "releídos" in capsys.readouterr().out and source.current() is second
    # Un archivo que deja de ser válido o desaparece no cambia los presupuestos vigentes.
    path.write_text(json.dumps(budgets(max_jobs=0)))
    assert source.current() is second
    assert "anteriores" in capsys.readouterr().out
    path.unlink()
    assert source.current() is second
    path.write_text(json.dumps(budgets(max_jobs=1)))
    assert source.current().max_jobs == 1
    with pytest.raises(ValueError, match="presupuestos"):
        engine.PackingFile(
            path.with_name("budgets-bad.json").write_text("{}") and path.parent / "budgets-bad.json"
        )
    with pytest.raises(OSError):
        engine.PackingFile(tmp_path / "missing.json")


def test_the_queue_uses_the_budgets_in_force_at_each_decision(engine, tmp_path, monkeypatch):
    path = tmp_path / "budgets.json"
    path.write_text(json.dumps(budgets(settle_seconds=0, max_jobs=1)))
    jobs = [work("large"), work("small")]
    queue, _, _ = shared_queue(
        engine, tmp_path, monkeypatch, jobs, packing=engine.PackingFile(path), workers=3
    )

    def cap():
        command = queue._command(jobs[1])
        return command[command.index("--vram-cap-mib") + 1]

    queue.taken.add(jobs[0].key)
    assert queue._admissible() == [] and cap() == "3000"
    measured = dict(large=3800, small=2900)
    path.write_text(json.dumps(budgets(settle_seconds=0, max_jobs=2, vram_mib=measured)))
    assert queue._admissible() == [jobs[1]] and cap() == "2900"


def test_a_runner_that_runs_alone_never_gets_a_cap_even_if_its_model_was_measured(engine):
    packing = engine.Packing(budgets(vram_mib=dict(ridge=200, small=3000)))
    assert packing.cap(work("ridge", runner="ridge")) is None
    assert packing.cap(work("small")) == 3000


def test_each_start_restarts_the_settle_time(engine, tmp_path, monkeypatch):
    jobs = [work("large")]
    packing = engine.Packing(budgets(settle_seconds=30))
    queue, _, launched = shared_queue(engine, tmp_path, monkeypatch, jobs, packing=packing)
    assert queue.launched == float("-inf")
    monkeypatch.setattr(engine.time, "monotonic", lambda: 500.0)
    queue._work()
    assert launched == [(jobs[0].key, [])] and queue.launched == 500.0


@pytest.mark.parametrize("barrier", ["_deadline_reached", "_disk_exhausted"])
def test_the_deadline_or_a_full_disk_stop_new_starts(engine, tmp_path, monkeypatch, barrier):
    jobs = [work("large")]
    packing = engine.Packing(budgets())
    queue, _, launched = shared_queue(engine, tmp_path, monkeypatch, jobs, packing=packing)
    monkeypatch.setattr(queue, barrier, lambda: True)
    queue._work()
    assert launched == [] and queue.taken == set()
