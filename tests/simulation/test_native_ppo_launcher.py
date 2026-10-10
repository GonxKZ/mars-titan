"""Admisión y señales del lanzador nativo sin consultar una GPU real."""

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mars_titan.training.learning_hold import HOLD_ENV

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_native_ppo.py"
LOCK_NAME = "mars-titan-scientific-gpu.lock"
HARNESS = """
import importlib.abc
import os
import runpy
import sys

class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            raise AssertionError('El lanzador ha intentado importar PyTorch')

sys.meta_path.insert(0, NoTorch())
os.environ['LAUNCHER_PID'] = str(os.getpid())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""


def executable(path, body, shebang):
    path.write_text(shebang + body)
    path.chmod(0o700)
    return path


def gpu_xml(free=7400, total=8188, *, compute=False):
    process = (
        f"<process_info><pid>{os.getpid()}</pid><type>C</type></process_info>" if compute else ""
    )
    return (
        "<nvidia_smi_log><gpu><fb_memory_usage>"
        f"<total>{total} MiB</total><free>{free} MiB</free>"
        f"</fb_memory_usage><processes>{process}</processes></gpu></nvidia_smi_log>"
    )


@pytest.fixture
def setup(tmp_path, learning_doubles, python_shebang):
    # El hijo es un arnés sin aprendizaje, así que el lanzador puede superar la protección.
    bin_dir, runtime = tmp_path / "bin", tmp_path / "runtime"
    bin_dir.mkdir()
    runtime.mkdir(mode=0o700)
    executable(
        bin_dir / "nvidia-smi",
        "import os\nfrom pathlib import Path\n"
        "Path(os.environ['GPU_PROBED']).write_text('probed')\n"
        "print(os.environ['GPU_XML'])\n",
        python_shebang,
    )
    native = executable(
        bin_dir / "native ppo",
        """import ctypes
import fcntl
import json
import os
import signal
import sys
import time
from pathlib import Path

if '--parent-pid' in sys.argv:
    parent = int(sys.argv[sys.argv.index('--parent-pid') + 1])
    if ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent:
        raise SystemExit(1)

record = {'args': sys.argv[1:], 'pid': os.getpid(),
          'launcher_pid': int(os.environ['LAUNCHER_PID'])}
output = Path(os.environ['NATIVE_RECORD'])
mode = os.environ.get('NATIVE_MODE', 'exit')

def stop(signum, frame):
    record['signal'] = signum
    output.write_text(json.dumps(record))
    time.sleep(0.15)
    Path(str(output) + '.stopped').write_text('paused')
    raise SystemExit(2)

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
if mode == 'ignore':
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if '--gpu-lease-fd' in sys.argv:
    fd = int(sys.argv[sys.argv.index('--gpu-lease-fd') + 1])
    lock = Path(os.environ['XDG_RUNTIME_DIR']) / 'mars-titan-scientific-gpu.lock'
    record['inherited'] = os.get_inheritable(fd)
    record['same_inode'] = os.fstat(fd).st_ino == lock.stat().st_ino
    with lock.open('a') as other:
        try:
            fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            record['exclusive'] = False
        except BlockingIOError:
            record['exclusive'] = True
    if os.environ.get('NATIVE_CLOSE_LEASE'):
        os.close(fd)
output.write_text(json.dumps(record))
if mode in ('wait', 'ignore'):
    while True:
        signal.pause()
raise SystemExit(int(os.environ.get('NATIVE_EXIT', '0')))
""",
        python_shebang,
    )
    config = tmp_path / "parameters.json"
    config.write_text("{}")
    args = [
        "--config",
        str(config),
        "--output",
        str(tmp_path / "output ; untouched"),
        "--train-tape",
        str(tmp_path / "train a"),
        "--validation-tape",
        str(tmp_path / "validation a"),
        "--binary",
        str(native),
    ]
    env = dict(
        os.environ,
        PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        XDG_RUNTIME_DIR=str(runtime),
        GPU_XML=gpu_xml(),
        GPU_PROBED=str(tmp_path / "gpu-probed"),
        NATIVE_RECORD=str(tmp_path / "native.json"),
    )
    return args, env, tmp_path


def command(args, harness=HARNESS):
    return [sys.executable, "-c", harness, str(SCRIPT), *args]


def execute(setup, *extra):
    args, env, _ = setup
    return subprocess.run(
        command([*args, *extra]), env=env, capture_output=True, text=True, timeout=8
    )


def record(setup):
    return json.loads(Path(setup[1]["NATIVE_RECORD"]).read_text())


def test_diagnostic_is_explicit_and_preserves_paths_and_pause_code(setup):
    setup[1]["NATIVE_EXIT"] = "2"
    result = execute(setup, "--diagnostic", "--resume", "--stop-after", "0")
    assert result.returncode == 2, result.stderr
    args = record(setup)["args"]
    assert args[args.index("--device") + 1] == "cpu"
    assert "--diagnostic" in args and "--resume" in args
    assert args[args.index("--stop-after") + 1] == "0"
    assert args[args.index("--output") + 1] == str(setup[2] / "output ; untouched")
    assert "--gpu-lease-fd" not in args
    assert not Path(setup[1]["GPU_PROBED"]).exists()
    assert not (Path(setup[1]["XDG_RUNTIME_DIR"]) / LOCK_NAME).exists()
    assert "CPU" in result.stderr and "32" in result.stderr


@pytest.mark.parametrize(
    ("free", "budget"), [(7400, 6439305216), (5000, 4169138176), (1280, 268435456)]
)
def test_cuda_inherits_exclusive_lease_and_the_measured_memory_budget(setup, free, budget):
    setup[1]["GPU_XML"] = gpu_xml(free)
    result = execute(setup)
    assert result.returncode == 0, result.stderr
    observed = record(setup)
    assert observed["inherited"] and observed["same_inode"] and observed["exclusive"]
    args = observed["args"]
    assert args[args.index("--device") + 1] == "cuda:0"
    assert "--diagnostic" not in args
    assert int(args[args.index("--vram-budget-bytes") + 1]) == budget
    assert int(args[args.index("--vram-total-bytes") + 1]) == 8585740288
    assert budget / int(args[args.index("--vram-total-bytes") + 1]) <= 0.75
    with (Path(setup[1]["XDG_RUNTIME_DIR"]) / LOCK_NAME).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_multiple_sources_are_forwarded_as_repeated_options(setup):
    result = execute(
        setup, "--diagnostic", "--train-tape", "train b", "--validation-tape", "validation b"
    )
    assert result.returncode == 0, result.stderr
    args = record(setup)["args"]
    assert [args[i + 1] for i, value in enumerate(args) if value == "--train-tape"] == [
        str(setup[2] / "train a"),
        "train b",
    ]
    assert [args[i + 1] for i, value in enumerate(args) if value == "--validation-tape"] == [
        str(setup[2] / "validation a"),
        "validation b",
    ]


@pytest.mark.parametrize("problem", ["memory", "process", "invalid_xml"])
def test_unavailable_gpu_is_rejected_without_starting_a_cpu_fallback(setup, problem):
    setup[1]["GPU_XML"] = (
        "not xml" if problem == "invalid_xml" else gpu_xml(1279 if problem == "memory" else 7400)
    )
    if problem == "process":
        setup[1]["GPU_XML"] = gpu_xml(compute=True)
    result = execute(setup)
    assert result.returncode == (1 if problem == "invalid_xml" else 3)
    assert result.stderr.startswith("Error: " if problem == "invalid_xml" else "Espera: ")
    assert not Path(setup[1]["NATIVE_RECORD"]).exists()


def test_existing_scientific_lease_rejects_before_gpu_probe(setup):
    with (Path(setup[1]["XDG_RUNTIME_DIR"]) / LOCK_NAME).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = execute(setup)
    assert result.returncode == 3 and result.stderr.startswith("Espera: ")
    assert not Path(setup[1]["GPU_PROBED"]).exists()
    assert not Path(setup[1]["NATIVE_RECORD"]).exists()


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_lock_must_be_a_regular_file_without_symlinks(setup, kind):
    lock = Path(setup[1]["XDG_RUNTIME_DIR"]) / LOCK_NAME
    if kind == "symlink":
        target = setup[2] / "target"
        target.write_text("original")
        lock.symlink_to(target)
    else:
        os.mkfifo(lock)
    result = execute(setup)
    assert result.returncode != 0 and result.stderr.startswith("Error: ")
    assert not Path(setup[1]["GPU_PROBED"]).exists()
    if kind == "symlink":
        assert target.read_text() == "original"


@pytest.mark.parametrize("option", ["--train-tape", "--validation-tape"])
def test_source_lists_are_bounded_before_admission(setup, option):
    result = execute(setup, *[value for _ in range(12) for value in (option, "another")])
    assert result.returncode == 2 and "12 fuentes" in result.stderr
    assert not Path(setup[1]["GPU_PROBED"]).exists()
    assert not Path(setup[1]["NATIVE_RECORD"]).exists()


@pytest.mark.parametrize("mode", ["missing", "not_executable"])
def test_invalid_binary_is_rejected_before_admission(setup, mode):
    native = Path(setup[0][setup[0].index("--binary") + 1])
    if mode == "missing":
        native.unlink()
    else:
        native.chmod(0o600)
    result = execute(setup)
    assert result.returncode != 0 and result.stderr.startswith("Error: ")
    assert not Path(setup[1]["GPU_PROBED"]).exists()


def test_failed_child_releases_the_lease_and_preserves_its_exit_code(setup):
    setup[1]["NATIVE_EXIT"] = "7"
    result = execute(setup)
    assert result.returncode == 7, result.stderr
    with (Path(setup[1]["XDG_RUNTIME_DIR"]) / LOCK_NAME).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def wait_until_ready(child, path):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                pass
        assert child.poll() is None, child.communicate()
        time.sleep(0.01)
    pytest.fail("El proceso de prueba no confirmó su inicio")


@pytest.mark.parametrize("requested", [signal.SIGINT, signal.SIGTERM])
def test_signal_waits_for_recoverable_child_stop_and_keeps_lease_until_exit(setup, requested):
    args, env, _ = setup
    env["NATIVE_MODE"] = "wait"
    env["NATIVE_CLOSE_LEASE"] = "1"
    path = Path(env["NATIVE_RECORD"])
    child = subprocess.Popen(command(args), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        observed = wait_until_ready(child, path)
        lock = Path(env["XDG_RUNTIME_DIR"]) / LOCK_NAME
        with lock.open("a") as handle:
            with pytest.raises(BlockingIOError):
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.kill(observed["launcher_pid"], requested)
        _, error = child.communicate(timeout=5)
        assert child.returncode == 2, error
        assert Path(str(path) + ".stopped").read_text() == "paused"
        assert record(setup)["signal"] == requested
        with pytest.raises(ProcessLookupError):
            os.kill(observed["pid"], 0)
        with lock.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)


def test_child_that_ignores_pause_is_killed_and_reaped_within_the_bound(setup):
    args, env, _ = setup
    env["NATIVE_MODE"] = "ignore"
    path = Path(env["NATIVE_RECORD"])
    harness = HARNESS.replace(
        "runpy.run_path(sys.argv[0], run_name='__main__')",
        "namespace = runpy.run_path(sys.argv[0])\n"
        "namespace['main'].__globals__['PAUSE_TIMEOUT'] = 0.05\n"
        "raise SystemExit(namespace['main']())",
    )
    child = subprocess.Popen(
        command(args, harness), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        observed = wait_until_ready(child, path)
        os.kill(observed["launcher_pid"], signal.SIGTERM)
        _, error = child.communicate(timeout=5)
        assert child.returncode == 75, error
        assert "no confirmó la pausa" in error.decode()
        with pytest.raises(ProcessLookupError):
            os.kill(observed["pid"], 0)
        with (Path(env["XDG_RUNTIME_DIR"]) / LOCK_NAME).open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)


def test_owned_legacy_lock_is_hardened_only_after_exclusive_admission(setup):
    lock = Path(setup[1]["XDG_RUNTIME_DIR"]) / LOCK_NAME
    lock.touch(mode=0o664)
    lock.chmod(0o664)
    with lock.open("a") as occupied:
        fcntl.flock(occupied, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert execute(setup).returncode != 0
        assert lock.stat().st_mode & 0o777 == 0o664
    result = execute(setup)
    assert result.returncode == 0, result.stderr
    assert lock.stat().st_mode & 0o777 == 0o600


def test_gpu_pressure_pauses_the_owned_group_and_preserves_code_two(setup, python_shebang):
    setup[1]["NATIVE_MODE"] = "wait"
    probe = Path(setup[1]["PATH"].split(os.pathsep)[0]) / "nvidia-smi"
    executable(
        probe,
        "import os\nfrom pathlib import Path\n"
        "from time import sleep\n"
        "sleep(0.02)\n"
        "started = Path(os.environ['NATIVE_RECORD']).exists()\n"
        "xml = os.environ['GPU_XML']\n"
        "print(xml.replace('7400 MiB','512 MiB') if started else xml)\n",
        python_shebang,
    )
    result = execute(setup, "--poll-seconds", "0.02", "--watch-state", str(setup[2] / "watch.json"))
    assert result.returncode == 2, result.stderr
    assert record(setup)["signal"] == signal.SIGTERM
    assert Path(setup[1]["NATIVE_RECORD"] + ".stopped").exists()
    status = json.loads((setup[2] / "watch.json").read_text())
    assert status["status"] == "paused" and status["reason"] == "memoria_insuficiente"
    assert status["returncode"] == 2


def test_watcher_does_not_classify_the_native_child_as_foreign_cuda(setup, python_shebang):
    setup[1]["NATIVE_MODE"] = "wait"
    probe = Path(setup[1]["PATH"].split(os.pathsep)[0]) / "nvidia-smi"
    executable(
        probe,
        """import json,os
from pathlib import Path
path=Path(os.environ['NATIVE_RECORD'])
xml=os.environ['GPU_XML']
if path.exists():
    child=json.loads(path.read_text())['pid']
    xml=xml.replace('<processes></processes>',f'<processes><process_info><pid>{child}</pid><type>C</type></process_info></processes>')
print(xml)
""",
        python_shebang,
    )
    child = subprocess.Popen(
        command([*setup[0], "--poll-seconds", "0.02"]),
        env=setup[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        wait_until_ready(child, Path(setup[1]["NATIVE_RECORD"]))
        time.sleep(0.15)
        assert child.poll() is None
        child.send_signal(signal.SIGTERM)
        _, error = child.communicate(timeout=5)
        assert child.returncode == 2, error
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


@pytest.mark.parametrize("location", ["configuration", "output", "source"])
def test_watch_state_cannot_overwrite_scientific_inputs_or_output(setup, location):
    args, _, directory = setup
    path = {
        "configuration": Path(args[args.index("--config") + 1]),
        "output": Path(args[args.index("--output") + 1]) / "watch.json",
        "source": Path(args[args.index("--train-tape") + 1]) / "watch.json",
    }[location]
    result = execute(setup, "--watch-state", str(path))
    assert result.returncode == 2
    assert not Path(setup[1]["NATIVE_RECORD"]).exists()


def test_busy_gpu_has_a_distinct_waiting_status_without_active_time(setup):
    setup[1]["GPU_XML"] = gpu_xml(compute=True)
    watch = setup[2] / "watch.json"
    result = execute(setup, "--watch-state", str(watch))
    assert result.returncode == 3, result.stderr
    status = json.loads(watch.read_text())
    assert status["status"] == "waiting" and status["child_active"] is False
    assert status["active_seconds"] == status["budget_seconds"] == 0
    assert not Path(setup[1]["NATIVE_RECORD"]).exists()


def test_short_budget_does_not_start_a_child(setup):
    watch = setup[2] / "watch.json"
    result = execute(setup, "--watch-state", str(watch), "--active-seconds-limit", "1")
    assert result.returncode == 4, result.stderr
    status = json.loads(watch.read_text())
    assert status["status"] == "budget_exhausted"
    assert status["starts"] == 0 and status["budget_seconds"] == 0
    assert not Path(setup[1]["NATIVE_RECORD"]).exists()


def test_watch_accumulates_completed_attempts_and_ignores_resume_flag(setup):
    watch = setup[2] / "watch.json"
    first = execute(setup, "--watch-state", str(watch), "--active-seconds-limit", "100")
    assert first.returncode == 0, first.stderr
    before = json.loads(watch.read_text())
    second = execute(setup, "--watch-state", str(watch), "--resume", "--active-seconds-limit", "90")
    assert second.returncode == 0, second.stderr
    after = json.loads(watch.read_text())
    assert after["identity_sha256"] == before["identity_sha256"]
    assert after["starts"] == 2 and after["child_active"] is False
    assert after["active_seconds"] > before["active_seconds"] > 0
    assert after["budget_seconds"] == after["active_seconds"]


def test_watch_rejects_changed_configuration_before_starting_another_child(setup):
    watch = setup[2] / "watch.json"
    assert execute(setup, "--watch-state", str(watch)).returncode == 0
    prior_record = Path(setup[1]["NATIVE_RECORD"]).read_bytes()
    config = Path(setup[0][setup[0].index("--config") + 1])
    config.write_text('{"schema_version":2}')
    result = execute(setup, "--watch-state", str(watch), "--resume")
    assert result.returncode == 1 and "identidad" in result.stderr
    assert Path(setup[1]["NATIVE_RECORD"]).read_bytes() == prior_record


def test_interrupted_previous_boot_charges_a_bounded_reserve_not_downtime(setup):
    watch = setup[2] / "watch.json"
    assert execute(setup, "--watch-state", str(watch)).returncode == 0
    previous = json.loads(watch.read_text())
    previous.update(child_active=True, boot_id="previous-boot", observed_at="2000-01-01T00:00:00Z")
    watch.write_text(json.dumps(previous))
    result = execute(setup, "--watch-state", str(watch), "--resume")
    assert result.returncode == 0, result.stderr
    actual = json.loads(watch.read_text())
    assert actual["uncertain_attempts"] == 1
    assert actual["active_seconds"] < 8
    assert actual["budget_seconds"] - actual["active_seconds"] == pytest.approx(
        previous["unobserved_reserve_seconds"]
    )


@pytest.mark.parametrize("previous_boot", [False, True])
def test_unobserved_child_without_parent_guard_blocks_without_a_new_launch(setup, previous_boot):
    watch = setup[2] / "watch.json"
    assert execute(setup, "--watch-state", str(watch)).returncode == 0
    previous = json.loads(watch.read_text())
    previous.update(child_active=True, status="running", parent_death_signal=None)
    if previous_boot:
        previous["boot_id"] = "previous-boot"
    watch.write_text(json.dumps(previous))
    prior_record = Path(setup[1]["NATIVE_RECORD"]).read_bytes()
    result = execute(setup, "--watch-state", str(watch), "--resume")
    assert result.returncode == 1 and "observación" in result.stderr
    actual = json.loads(watch.read_text())
    assert actual["status"] == "blocked" and actual["budget_complete"] is False
    assert actual["starts"] == previous["starts"]
    assert Path(setup[1]["NATIVE_RECORD"]).read_bytes() == prior_record


def test_killing_wrapper_ends_only_its_guarded_child_and_charges_uncertainty_once(setup):
    args, env, directory = setup
    env["NATIVE_MODE"] = "wait"
    watch = directory / "watch.json"
    wrapper = subprocess.Popen(
        command([*args, "--diagnostic", "--watch-state", str(watch)]),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    native_pid = None
    try:
        observed = wait_until_ready(wrapper, Path(env["NATIVE_RECORD"]))
        native_pid = observed["pid"]
        before = json.loads(watch.read_text())
        assert before["child_active"] and before["parent_death_signal"] == "SIGKILL"
        wrapper.kill()
        wrapper.communicate(timeout=5)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = Path(f"/proc/{native_pid}/stat")
            try:
                stopped = status.read_text().rsplit(")", 1)[1].split()[0] == "Z"
            except (FileNotFoundError, ProcessLookupError):
                stopped = True
            if stopped:
                break
            time.sleep(0.01)
        else:
            pytest.fail("El hijo protegido sobrevivió a la muerte del lanzador")
        assert unrelated.poll() is None
        assert not Path(env["NATIVE_RECORD"] + ".stopped").exists()
        env["NATIVE_MODE"] = "exit"
        result = execute(setup, "--diagnostic", "--watch-state", str(watch), "--resume")
        assert result.returncode == 0, result.stderr
        recovered = json.loads(watch.read_text())
        assert recovered["uncertain_attempts"] == 1
        assert recovered["reserved_seconds"] == before["unobserved_reserve_seconds"]
        assert recovered["starts"] == 2 and recovered["budget_complete"] is True
        assert (
            execute(setup, "--diagnostic", "--watch-state", str(watch), "--resume").returncode == 0
        )
        assert json.loads(watch.read_text())["reserved_seconds"] == recovered["reserved_seconds"]
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
            wrapper.wait(timeout=5)
        if native_pid is not None:
            try:
                os.kill(native_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        unrelated.terminate()
        unrelated.wait(timeout=5)


CATALOGS = [
    dict(schema_version=2),
    dict(schema_version=3),
    dict(schema_version=4),
    dict(schema_version=1, kind="native_klpo_terminal"),
]


@pytest.mark.parametrize("document", CATALOGS)
def test_adaptive_versions_forward_large_catalog_without_changing_version_one_limit(
    setup, document
):
    config = Path(setup[0][setup[0].index("--config") + 1])
    config.write_text(json.dumps(document))
    additional = [value for index in range(255) for value in ("--train-tape", f"train-{index}")]
    result = execute(setup, "--diagnostic", *additional)
    assert result.returncode == 0, result.stderr
    assert record(setup)["args"].count("--train-tape") == 256


def test_active_budget_requests_pause_with_shutdown_time_reserved(setup):
    setup[1]["NATIVE_MODE"] = "wait"
    watch = setup[2] / "watch.json"
    harness = HARNESS.replace(
        "runpy.run_path(sys.argv[0], run_name='__main__')",
        "namespace = runpy.run_path(sys.argv[0])\n"
        "namespace['main'].__globals__['PAUSE_TIMEOUT'] = 0.5\n"
        "namespace['main'].__globals__['KILL_TIMEOUT'] = 0.1\n"
        "namespace['main'].__globals__['GPU_PROBE_TIMEOUT'] = 0.01\n"
        "raise SystemExit(namespace['main']())",
    )
    result = subprocess.run(
        command(
            [
                *setup[0],
                "--watch-state",
                str(watch),
                "--poll-seconds",
                "0.02",
                "--active-seconds-limit",
                "1.2",
            ],
            harness,
        ),
        env=setup[1],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 2, result.stderr
    status = json.loads(watch.read_text())
    assert status["reason"] == "active_budget_exhausted"
    assert status["recoverable_checkpoint"] is True
    assert 0 < status["active_seconds"] < 1.7
    assert status["budget_seconds"] == status["active_seconds"]
    assert record(setup)["signal"] == signal.SIGTERM


@pytest.mark.parametrize("document", CATALOGS)
def test_audit_uses_the_same_launcher_without_training_arguments(setup, document):
    args = setup[0]
    config = Path(args[args.index("--config") + 1])
    config.write_text(json.dumps(document))
    native = args[args.index("--binary") + 1]
    setup[0][:] = [
        *args[:4],
        "--binary",
        native,
        "--audit-run",
        str(setup[2] / "frozen-run"),
        "--audit-tape",
        str(setup[2] / "audit-a"),
    ]
    result = execute(setup, "--diagnostic")
    assert result.returncode == 0, result.stderr
    actual = record(setup)["args"]
    assert "--audit-run" in actual and "--audit-tape" in actual
    assert "--train-tape" not in actual and "--validation-tape" not in actual


def test_relative_command_paths_do_not_reuse_watch_from_another_working_directory(setup):
    args, env, directory = setup
    watch = directory / "watch.json"
    relative = [
        "--config",
        "parameters.json",
        "--output",
        "run",
        "--train-tape",
        "train",
        "--validation-tape",
        "validation",
        "--binary",
        args[args.index("--binary") + 1],
        "--diagnostic",
        "--watch-state",
        str(watch),
    ]
    first = subprocess.run(
        command(relative), env=env, cwd=directory, capture_output=True, text=True, timeout=8
    )
    assert first.returncode == 0, first.stderr
    prior = Path(env["NATIVE_RECORD"]).read_bytes()
    elsewhere = directory / "another"
    elsewhere.mkdir()
    (elsewhere / "parameters.json").write_bytes((directory / "parameters.json").read_bytes())
    second = subprocess.run(
        command([*relative, "--resume"]),
        env=env,
        cwd=elsewhere,
        capture_output=True,
        text=True,
        timeout=8,
    )
    assert second.returncode == 1 and "identidad" in second.stderr
    assert Path(env["NATIVE_RECORD"]).read_bytes() == prior


@pytest.mark.parametrize(
    ("document", "blocked"),
    [
        (dict(schema_version=2), False),
        (dict(schema_version=4), True),
        (dict(schema_version=1, kind="native_klpo_terminal"), True),
    ],
)
def test_audits_on_reconstructed_tapes_stop_on_the_hold_before_launching(setup, document, blocked):
    args, env, directory = setup
    config = Path(args[args.index("--config") + 1])
    config.write_text(json.dumps(document))
    hold = directory / "blocking-hold.json"
    hold.write_text(json.dumps({"training_allowed": False}))
    env[HOLD_ENV] = str(hold)
    setup[0][:] = [
        *args[:4],
        "--binary",
        args[args.index("--binary") + 1],
        "--audit-run",
        str(directory / "frozen-run"),
        "--audit-tape",
        str(directory / "audit-a"),
    ]
    result = execute(setup, "--diagnostic")
    # La auditoría sintética sigue sin bloqueo. Evaluar sobre cintas reales es un uso
    # científico del histórico y se detiene antes de lanzar el binario.
    assert (result.returncode != 0) is blocked, result.stderr
    assert Path(env["NATIVE_RECORD"]).exists() is not blocked
    if blocked:
        assert "la evaluación nativa sobre cintas reconstruidas" in result.stderr


def test_klpo_training_names_its_algorithm_in_the_hold(setup):
    args, env, directory = setup
    config = Path(args[args.index("--config") + 1])
    config.write_text(json.dumps(dict(schema_version=1, kind="native_klpo_terminal")))
    hold = directory / "blocking-hold.json"
    hold.write_text(json.dumps({"training_allowed": False}))
    env[HOLD_ENV] = str(hold)
    result = execute(setup, "--diagnostic")
    assert result.returncode != 0 and "el entrenamiento KLPO nativo" in result.stderr
    assert not Path(env["NATIVE_RECORD"]).exists()
