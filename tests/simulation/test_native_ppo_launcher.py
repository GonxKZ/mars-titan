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


def executable(path, body):
    path.write_text(f"#!{sys.executable}\n{body}")
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
def setup(tmp_path):
    bin_dir, runtime = tmp_path / "bin", tmp_path / "runtime"
    bin_dir.mkdir()
    runtime.mkdir(mode=0o700)
    executable(
        bin_dir / "nvidia-smi",
        "import os\nfrom pathlib import Path\n"
        "Path(os.environ['GPU_PROBED']).write_text('probed')\n"
        "print(os.environ['GPU_XML'])\n",
    )
    native = executable(
        bin_dir / "native ppo",
        """import fcntl
import json
import os
import signal
import sys
import time
from pathlib import Path

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
    assert result.returncode != 0
    assert result.stderr.startswith("Error: ")
    assert not Path(setup[1]["NATIVE_RECORD"]).exists()


def test_existing_scientific_lease_rejects_before_gpu_probe(setup):
    with (Path(setup[1]["XDG_RUNTIME_DIR"]) / LOCK_NAME).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = execute(setup)
    assert result.returncode != 0 and result.stderr.startswith("Error: ")
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


def test_gpu_pressure_pauses_the_owned_group_and_preserves_code_two(setup):
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
    )
    result = execute(setup, "--poll-seconds", "0.02", "--watch-state", str(setup[2] / "watch.json"))
    assert result.returncode == 2, result.stderr
    assert record(setup)["signal"] == signal.SIGTERM
    assert Path(setup[1]["NATIVE_RECORD"] + ".stopped").exists()
    status = json.loads((setup[2] / "watch.json").read_text())
    assert status["status"] == "paused" and status["reason"] == "memoria_insuficiente"
    assert status["returncode"] == 2


def test_watcher_does_not_classify_the_native_child_as_foreign_cuda(setup):
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
