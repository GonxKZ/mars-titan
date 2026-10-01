"""Medición reproducible mediante ejecutables de prueba sin aprendizaje ni GPU."""

import json
import os
import runpy
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest_plugins = ["test_adaptive_campaign"]

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/benchmark_adaptive_rl.py"


@pytest.fixture
def instrument(monkeypatch):
    assert SCRIPT.is_file(), "Falta el benchmark público de adaptación"
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return runpy.run_path(str(SCRIPT))


@pytest.fixture
def benchmark(catalog, tmp_path, monkeypatch):
    source, binary, _ = catalog
    binary.write_text(
        f"#!{sys.executable}\n"
        + """import ctypes, hashlib, json, os, signal, sys
from pathlib import Path

def option(name):
    return sys.argv[sys.argv.index(name)+1]

if '--parent-pid' in sys.argv:
    assert ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0) == 0
    assert os.getppid() == int(option('--parent-pid'))
if os.environ.get('BENCHMARK_READY'):
    Path(os.environ['BENCHMARK_READY']).write_text(str(os.getpid()))
    while True:
        signal.pause()
if os.environ.get('BENCHMARK_FAIL'):
    raise SystemExit(int(os.environ['BENCHMARK_FAIL']))
config = json.loads(Path(option('--config')).read_text())
output = Path(option('--output'))
output.mkdir(parents=True)
identity = dict(configuration=config, torch_version='fixture', native_build_sha256='a'*64)
def seal(value):
    encoded = json.dumps(value,sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(encoded).hexdigest()
def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True,separators=(',',':')))
write(output/'identity.json',dict(identity=identity,sha256=seal(identity)))
report = dict(kind='native_ppo', status='completed', device='cuda:0', diagnostic=False,
    seed=config['training']['seed'], agent_variant=config['agent']['variant'],
    transitions=config['training']['total_transitions'], observed_transitions=1024,
    optimizer_steps=2, parameters=64, identity_sha256=seal(identity), invocation_seconds=.01,
    timings=dict(training_seconds=.004,evaluation_seconds=.003,checkpoint_seconds=.002,
                 setup_seconds=.001), resources=dict(ram_peak_bytes=67108864,
                 ram_peak_method='procfs_VmHWM',vram_budget_bytes=268435456,
                 vram_total_bytes=8585740288,vram_peak_bytes=None),
    best=dict(mean_log_growth=999,validation_metrics=['calidad que no debe exportarse']))
write(output/'run.json',report)
payloads = {'policy.pt':b'fixture-policy','rollout.pt':b'fixture-rollout','metadata.json':b'{}'}
manifest = dict(files={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
                      for name,data in payloads.items()})
digest=seal(manifest)
folder=output/('ppo-'+digest)
folder.mkdir()
write(folder/'manifest.json',manifest)
for name,data in payloads.items():
    (folder/name).write_bytes(data)
record=dict(bundle=folder.name,sha256=digest)
index=dict(recent=[record],best=record)
write(output/'ppo-index.json',dict(payload=index,sha256=seal(index)))
Path(os.environ['BENCHMARK_CALLS']).open('a').write(json.dumps(dict(args=sys.argv,config=config))+'\\n')
if os.environ.get('BENCHMARK_MUTATE'):
    Path(sys.argv[0]).write_text('# cambiado durante la prueba')
"""
    )
    binary.chmod(0o700)
    reference = binary.with_name("reference")
    reference.write_bytes(binary.read_bytes())
    reference.chmod(0o700)
    tools = tmp_path / "tools"
    tools.mkdir()
    probe = tools / "nvidia-smi"
    probe.write_text(
        f"#!{sys.executable}\nimport sys\n"
        "print('fixture GPU, fixture driver, 8188' if "
        "'--query-gpu=name,driver_version,memory.total' "
        "in sys.argv else '<nvidia_smi_log><gpu><fb_memory_usage><total>8188 MiB</total>"
        "<free>7400 MiB</free></fb_memory_usage><processes></processes></gpu></nvidia_smi_log>')\n"
    )
    probe.chmod(0o700)
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    monkeypatch.setenv("PATH", f"{tools}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("BENCHMARK_CALLS", str(tmp_path / "calls.jsonl"))
    return [
        "--scenarios",
        str(source),
        "--binary",
        str(binary),
        "--output",
        str(tmp_path / "result"),
        "--reference-binary",
        str(reference),
        "--variants",
        "ppo",
        "--workers",
        "1",
        "2",
        "--repetitions",
        "2",
        "--transitions",
        "32",
    ]


def test_paired_benchmark_keeps_warmup_separate_alternates_order_and_omits_quality(
    instrument, benchmark, tmp_path
):
    assert instrument["main"](benchmark) == 0
    output = tmp_path / "result"
    report = json.loads((output / "benchmark.json").read_text())
    assert report["status"] == "completed"
    assert len(report["records"]) == 12
    for workers in (1, 2):
        rows = [row for row in report["records"] if row["workers"] == workers]
        assert [row["method"] for row in rows] == [
            "reference",
            "candidate",
            "reference",
            "candidate",
            "candidate",
            "reference",
        ]
        assert [row["warmup"] for row in rows] == [True, True, False, False, False, False]
        assert all(row["disk"]["retained_bundles"] == 1 for row in rows)
        assert all(row["resources"]["ram_peak_bytes"] == 67108864 for row in rows)
    text = (output / "benchmark.json").read_text()
    assert str(tmp_path) not in text and "validation_metrics" not in text and '"best"' not in text
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == 12
    first = calls[0]["args"]
    assert first.count("--train-tape") == 256 and first.count("--validation-tape") == 128
    assert "--audit-tape" not in first
    names = [
        Path(first[index + 1]).name.split("-train-", 1)[0]
        for index, value in enumerate(first)
        if value == "--train-tape"
    ][:16]
    assert len(set(names[:8])) == 8 and names[:8] == names[8:]
    assert all(row["configuration_sha256"] for row in report["records"])
    assert all(row["count"] == 2 for row in report["summaries"])
    assert not any(
        path.is_absolute() for path in map(Path, [row["path"] for row in report["records"]])
    )


@pytest.mark.parametrize(
    "option,value",
    [
        ("--repetitions", "0"),
        ("--workers", "3"),
        ("--transitions", "1048592"),
        ("--timeout", "nan"),
    ],
)
def test_invalid_limits_are_rejected_before_starting_processes(
    instrument, benchmark, tmp_path, option, value
):
    with pytest.raises(SystemExit):
        instrument["main"]([*benchmark, option, value])
    assert not (tmp_path / "result").exists()
    assert not (tmp_path / "calls.jsonl").exists()


def test_existing_output_is_preserved(instrument, benchmark, tmp_path):
    output = tmp_path / "result"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("anterior")
    assert instrument["main"](benchmark) == 1
    assert marker.read_text() == "anterior" and not (tmp_path / "calls.jsonl").exists()


@pytest.mark.parametrize("mode", ["failure", "mutation"])
def test_failure_or_changed_binary_stops_without_retry(
    instrument, benchmark, tmp_path, monkeypatch, mode
):
    monkeypatch.setenv("BENCHMARK_FAIL" if mode == "failure" else "BENCHMARK_MUTATE", "1")
    assert instrument["main"](benchmark) == 1
    report = json.loads((tmp_path / "result/benchmark.json").read_text())
    assert report["status"] == "failed" and report["records"] == []
    assert len(list((tmp_path / "result").glob("**/process.log"))) == 1


def test_timeout_reaps_the_owned_process(instrument, tmp_path):
    pid_file = tmp_path / "pid"
    code = (
        "import os,time,sys; from pathlib import Path; "
        "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
    )
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        instrument["run_process"](
            [sys.executable, "-c", code, str(pid_file)],
            tmp_path / "process.log",
            0.5,
            dict(os.environ),
        )
    assert time.monotonic() - started < 3
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), signal.SIGTERM)


def process_finished(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"
    except (FileNotFoundError, ProcessLookupError):
        return True


def test_hard_timeout_closes_owned_legacy_child_and_preserves_foreign_process(
    instrument, tmp_path, monkeypatch
):
    monkeypatch.setitem(instrument["run_process"].__globals__, "STOP_SECONDS", 0.05)
    child_record = tmp_path / "child"
    native = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"
    wrapper = (
        "import signal,subprocess,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "child=subprocess.Popen([sys.executable,'-c',sys.argv[2]],start_new_session=True); "
        "Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(30)"
    )
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            instrument["run_process"](
                [sys.executable, "-c", wrapper, str(child_record), native],
                tmp_path / "process.log",
                0.8,
                dict(os.environ),
            )
        deadline = time.monotonic() + 2
        while not process_finished(int(child_record.read_text())) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert process_finished(int(child_record.read_text()))
        assert foreign.poll() is None
    finally:
        if child_record.exists() and not process_finished(int(child_record.read_text())):
            os.kill(int(child_record.read_text()), signal.SIGKILL)
        foreign.terminate()
        foreign.wait(timeout=3)


def test_hmm_copy_keeps_source_bytes_and_one_repetition_has_no_invented_deviation(
    instrument, benchmark, tmp_path
):
    assert (
        instrument["main"](
            [*benchmark, "--variants", "ppo_hmm", "--workers", "1", "--repetitions", "1"]
        )
        == 0
    )
    output = tmp_path / "result"
    assert (output / "hmm.json").read_bytes() == (tmp_path / "scenarios/hmm.json").read_bytes()
    config = json.loads((output / "ppo_hmm-w1.json").read_text())
    assert config["agent"]["markov_fields"] == [4, 5, 6]
    report = json.loads((output / "benchmark.json").read_text())
    assert all(row["count"] == 1 and row["stdev_sample"] is None for row in report["summaries"])


def test_sigterm_stops_the_active_wrapper_and_records_failure(benchmark, tmp_path, monkeypatch):
    ready = tmp_path / "ready"
    monkeypatch.setenv("BENCHMARK_READY", str(ready))
    process = subprocess.Popen(
        [sys.executable, str(SCRIPT), *benchmark], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            assert process.poll() is None, process.communicate()
            time.sleep(0.01)
        assert ready.exists()
        process.terminate()
        process.communicate(timeout=5)
        assert process.returncode != 0
        assert process_finished(int(ready.read_text()))
        assert json.loads((tmp_path / "result/benchmark.json").read_text())["status"] == "failed"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        if ready.exists() and not process_finished(int(ready.read_text())):
            os.kill(int(ready.read_text()), signal.SIGKILL)


@pytest.mark.parametrize("corruption", ["cpu", "nan_timing", "index_seal", "payload"])
def test_measure_rejects_incoherent_receipts_or_unconfirmed_disk_bytes(
    instrument, benchmark, tmp_path, corruption
):
    assert instrument["main"]([*benchmark, "--workers", "1", "--repetitions", "1"]) == 0
    output = tmp_path / "result"
    run = output / "ppo-w1/1-candidate/run"
    config = json.loads((output / "ppo-w1.json").read_text())
    report = json.loads((run / "run.json").read_text())
    index = json.loads((run / "ppo-index.json").read_text())
    if corruption == "cpu":
        report["device"] = "cpu"
    elif corruption == "nan_timing":
        report["timings"]["training_seconds"] = float("nan")
    elif corruption == "index_seal":
        index["sha256"] = "0" * 64
    else:
        (run / index["payload"]["best"]["bundle"] / "policy.pt").write_bytes(b"changed")
    (run / "run.json").write_text(json.dumps(report))
    (run / "ppo-index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError):
        instrument["measure"](run, config)


@pytest.mark.parametrize("filename", ["market.parquet", "context.parquet"])
def test_parquet_mutation_during_process_blocks_the_measurement(
    instrument, benchmark, tmp_path, monkeypatch, filename
):
    run_process = instrument["run_process"]
    calls = []
    index_path = Path(benchmark[benchmark.index("--scenarios") + 1])
    index = json.loads(index_path.read_text())
    source = next(row for row in index["records"] if row["split"] == "train")
    parquet = index_path.parent / source["path"] / filename

    def mutate(*args):
        result = run_process(*args)
        calls.append(result)
        parquet.write_bytes(parquet.read_bytes() + b"changed after native read")
        return result

    monkeypatch.setitem(instrument["main"].__globals__, "run_process", mutate)
    assert instrument["main"]([*benchmark, "--workers", "1", "--repetitions", "1"]) == 1
    report = json.loads((tmp_path / "result/benchmark.json").read_text())
    assert report["status"] == "failed" and report["records"] == []
    assert len(calls) == 1
