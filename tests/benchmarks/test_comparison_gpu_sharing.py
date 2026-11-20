"""Cotejo bit a bit y entorno de las pruebas de trabajos que comparten la GPU."""

import json
import math
import os
import signal
import subprocess
import sys
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks.comparison_gpu_sharing_20261120 import gemelos

KEY = "network/2019/seed-43"


def rows(values, *, step=32, metric="loss"):
    return [
        dict(step=step, phase="train", metric=metric, group=f"g{index}", stat="mean", value=value)
        for index, value in enumerate(values)
    ]


def write_run(root, trace, *, history=(0.5,), step=64, kind=None, **fit):
    """Trabajo mínimo: trazas repartidas en dos partes y el estado de su ajuste."""
    folder = root / "runs" / KEY
    (folder / "traces/parts").mkdir(parents=True)
    (folder / "fit").mkdir()
    schema = pa.schema(
        [
            ("step", pa.int64()),
            ("phase", pa.string()),
            ("metric", pa.string()),
            ("group", pa.string()),
            ("stat", pa.string()),
            ("value", kind or pa.float64()),
        ]
    )
    half = len(trace) // 2
    for index, part in enumerate((trace[:half], trace[half:])):
        table = pa.Table.from_pylist(part, schema=schema)
        pq.write_table(table, folder / f"traces/parts/part-{index:05d}.parquet")
    document = dict(
        status="paused",
        global_step=step,
        cursor=dict(epoch=len(history), phase="validation"),
        identity="identity",
        run_id="run",
        history=[
            dict(epoch=epoch, validation=dict(mae=value, counts=[3, 4]), note=None)
            for epoch, value in enumerate(history)
        ],
    )
    document.update(fit)
    (folder / "fit/run.json").write_text(json.dumps(document))
    return folder


def test_bits_separate_values_that_a_tolerance_would_accept():
    above = math.nextafter(1.0, math.inf)
    assert gemelos.bits(1.0) == "000000000000f03f" and gemelos.bits(1) == gemelos.bits(1.0)
    assert gemelos.bits(above) != gemelos.bits(1.0)
    assert gemelos.bits(0.0) != gemelos.bits(-0.0)
    assert gemelos.bits(None) is None


def test_trace_rows_keep_every_value_of_a_key_up_to_the_requested_step(tmp_path):
    trace = [*rows([0.25, 0.5]), *rows([0.75]), *rows([1.0], step=64), *rows([2.0], step=96)]
    folder = write_run(tmp_path, trace)
    read = gemelos.trace_rows(folder, 64)
    first = (32, "train", "loss", "g0", "mean")
    assert read[first] == [gemelos.bits(0.25), gemelos.bits(0.75)]
    assert set(read) == {
        first,
        (32, "train", "loss", "g1", "mean"),
        (64, "train", "loss", "g0", "mean"),
    }
    assert (96, "train", "loss", "g0", "mean") in gemelos.trace_rows(folder, 96)
    assert gemelos.trace_rows(tmp_path / "sin trazas", 64) == {}


def test_single_precision_traces_are_compared_without_rounding(tmp_path):
    folder = write_run(tmp_path, rows([0.1, 0.2]), kind=pa.float32())
    stored = pq.read_table(folder / "traces/parts/part-00000.parquet")["value"][0].as_py()
    assert stored != 0.1
    assert gemelos.trace_rows(folder, 64)[(32, "train", "loss", "g0", "mean")] == [
        gemelos.bits(stored)
    ]


def test_flat_lists_every_leaf_with_its_path_and_real_numbers_by_their_bits():
    document = dict(b=[1.5, dict(c="x")], a=None, n=3)
    assert list(gemelos.flat(document)) == [
        ("/a", None),
        ("/b[0]", gemelos.bits(1.5)),
        ("/b[1]/c", "x"),
        ("/n", 3),
    ]


def test_a_twin_equal_to_the_original_until_its_step_is_identical(tmp_path):
    trace = [*rows([0.25, 0.5]), *rows([1.0, 1.5], step=64)]
    twin = write_run(tmp_path / "twin", trace, history=(0.5,))
    # El original siguió más allá: sus filas y épocas posteriores no cuentan.
    longer = [*trace, *rows([3.0, 3.5], step=64, metric="later"), *rows([2.0, 2.5], step=96)]
    original = write_run(tmp_path / "original", longer, history=(0.5, 0.4), step=128)
    entry = gemelos.compare_run(twin, original)
    assert entry["identical"] and entry["identity_equal"] and entry["run_id_equal"]
    assert (entry["trace_rows"], entry["trace_rows_compared"]) == (4, 4)
    assert entry["trace_rows_reference_until_step"] == 6 and entry["trace_rows_only_reference"] == 2
    assert entry["trace_rows_different"] == 0 and entry["first_trace_differences"] == []
    assert entry["history_epochs_compared"] == [0] and entry["history_differences"] == 0
    assert (entry["global_step"], entry["status"]) == (64, "paused")
    assert "best_checkpoint_equal" not in entry


def test_one_unit_in_the_last_place_of_a_trace_is_a_difference(tmp_path):
    above = math.nextafter(0.5, math.inf)
    twin = write_run(tmp_path / "twin", rows([0.25, above]))
    original = write_run(tmp_path / "original", rows([0.25, 0.5]))
    entry = gemelos.compare_run(twin, original)
    assert not entry["identical"] and entry["trace_rows_different"] == 1
    assert entry["first_trace_differences"] == [
        [32, "train", "loss", "g1", "mean", [gemelos.bits(above)], [gemelos.bits(0.5)]]
    ]
    assert entry["trace_rows_compared"] == 2 and entry["history_differences"] == 0


def test_one_unit_in_the_last_place_of_an_epoch_metric_is_a_difference(tmp_path):
    above = math.nextafter(0.5, math.inf)
    twin = write_run(tmp_path / "twin", rows([0.25, 0.5]), history=(0.5, above))
    original = write_run(tmp_path / "original", rows([0.25, 0.5]), history=(0.5, 0.5))
    entry = gemelos.compare_run(twin, original)
    assert not entry["identical"] and entry["trace_rows_different"] == 0
    assert entry["history_differences"] == 1
    assert entry["first_history_differences"] == [
        (1, "/validation/mae", gemelos.bits(above), gemelos.bits(0.5))
    ]


@pytest.mark.parametrize(
    "twin,original,field",
    [
        (
            dict(trace=rows([0.25, 0.5, 0.75])),
            dict(trace=rows([0.25, 0.5])),
            "trace_rows_only_twin",
        ),
        (dict(trace=rows([0.25], metric="a")), dict(trace=rows([0.25], metric="b")), None),
        (dict(trace=[]), dict(trace=[]), None),
        (dict(history=(0.5, 0.4)), dict(history=(0.5,)), "history_epochs_missing_in_reference"),
        (dict(identity="other"), {}, "identity_equal"),
        (dict(run_id="other"), {}, "run_id_equal"),
    ],
)
def test_anything_that_cannot_be_matched_with_the_original_is_not_identity(
    tmp_path, twin, original, field
):
    base = dict(trace=rows([0.25, 0.5]))
    mine = write_run(tmp_path / "twin", **{**base, **twin})
    theirs = write_run(tmp_path / "original", **{**base, **original})
    entry = gemelos.compare_run(mine, theirs)
    assert not entry["identical"]
    if field == "trace_rows_only_twin":
        assert entry[field] == 1
    elif field == "history_epochs_missing_in_reference":
        assert entry[field] == [1]
    elif field is not None:
        assert entry[field] is False


def test_a_completed_twin_must_also_select_the_same_checkpoint(tmp_path):
    done = dict(status="completed", best_checkpoint=dict(sha256="aa"))
    twin = write_run(tmp_path / "twin", rows([0.25, 0.5]), **done)
    original = write_run(tmp_path / "original", rows([0.25, 0.5]), **done)
    entry = gemelos.compare_run(twin, original)
    assert entry["identical"] and entry["best_checkpoint_equal"]
    other = dict(status="completed", best_checkpoint=dict(sha256="bb"))
    changed = write_run(tmp_path / "changed", rows([0.25, 0.5]), **other)
    entry = gemelos.compare_run(changed, original)
    assert not entry["identical"] and entry["best_checkpoint_equal"] is False


def test_a_run_without_its_fit_state_cannot_be_compared(tmp_path):
    twin = write_run(tmp_path / "twin", rows([0.25, 0.5]))
    with pytest.raises(ValueError, match="estado del ajuste"):
        gemelos.compare_run(twin, tmp_path / "missing")
    with pytest.raises(ValueError, match="estado del ajuste"):
        gemelos.compare_run(tmp_path / "missing", twin)


def test_compare_reports_every_state_and_fails_if_any_differs(tmp_path, capsys):
    campaign = tmp_path / "campaign"
    write_run(campaign, rows([0.25, 0.5]))
    write_run(tmp_path / "equal", rows([0.25, 0.5]))
    write_run(tmp_path / "other", rows([0.25, 0.75]))
    out = tmp_path / "report.json"
    base = ["compare", "--key", KEY, "--campaign", str(campaign), "--out", str(out)]
    assert gemelos.main([*base, "--state", str(tmp_path / "equal")]) == 0
    report = json.loads(out.read_text())
    assert report["key"] == KEY and [twin["state"] for twin in report["twins"]] == ["equal"]
    assert report["twins"][0]["identical"] and json.loads(capsys.readouterr().out) == report
    both = [*base, "--state", str(tmp_path / "equal"), "--state", str(tmp_path / "other")]
    assert gemelos.main(both) == 1
    report = json.loads(out.read_text())
    assert [twin["identical"] for twin in report["twins"]] == [True, False]
    # Sin archivo de salida, el informe solo se imprime.
    out.unlink()
    assert gemelos.main(["compare", "--key", KEY, "--campaign", str(campaign), *both[-2:]]) == 1
    assert not out.exists()


def test_trial_environment_only_changes_what_the_trial_declares(tmp_path):
    base = dict(
        PATH="/usr/bin",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        CUDA_MPS_PIPE_DIRECTORY="/mps/pipe",
        CUDA_MPS_LOG_DIRECTORY="/mps/log",
        CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="0=1M",
        PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64",
    )
    plain = gemelos.trial_environment(base, tmp_path / "runtime", tmp_path / "state", mps=False)
    assert plain == dict(
        PATH="/usr/bin",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTHONPATH=str(tmp_path / "runtime/src"),
        MARS_TITAN_DIGEST_CACHE=str(tmp_path / "state/file-digests.json"),
    )
    shared = gemelos.trial_environment(base, tmp_path / "runtime", tmp_path / "state", mps=True)
    assert shared == dict(
        plain, CUDA_MPS_PIPE_DIRECTORY="/mps/pipe", CUDA_MPS_LOG_DIRECTORY="/mps/log"
    )
    capped = gemelos.trial_environment(
        base,
        tmp_path / "runtime",
        tmp_path / "state",
        mps=True,
        vram_limit_mib=3100,
        alloc_conf="expandable_segments:True",
    )
    assert capped == dict(
        shared,
        CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="0=3100M",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    assert base["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] == "0=1M"


def test_trial_environment_rejects_options_that_would_not_apply(tmp_path):
    runtime, state = tmp_path / "runtime", tmp_path / "state"
    with pytest.raises(ValueError, match="MPS"):
        gemelos.trial_environment(
            dict(CUDA_MPS_PIPE_DIRECTORY="/mps/pipe"), runtime, state, mps=True
        )
    with pytest.raises(ValueError, match="MPS"):
        gemelos.trial_environment(
            dict(CUDA_MPS_PIPE_DIRECTORY="/p", CUDA_MPS_LOG_DIRECTORY=""), runtime, state, mps=True
        )
    with pytest.raises(ValueError, match="tope"):
        gemelos.trial_environment({}, runtime, state, mps=False, vram_limit_mib=3000)


def test_group_memory_and_missing_files_are_reported_as_unknown(tmp_path):
    assert gemelos.group_anonymous(-1) is None
    assert gemelos.read_json(tmp_path / "missing.json") is None
    (tmp_path / "broken.json").write_text("{")
    assert gemelos.read_json(tmp_path / "broken.json") is None
    (tmp_path / "state.json").write_text('{"global_step": 7}')
    assert gemelos.read_json(tmp_path / "state.json") == dict(global_step=7)


def test_group_memory_adds_the_anonymous_and_shared_pages_of_the_process_group(tmp_path):
    proc, groups = tmp_path / "proc", tmp_path / "groups"
    (proc / "77").mkdir(parents=True)
    (proc / "77/cgroup").write_text("0::/user.slice/run-77.scope\n")
    folder = groups / "user.slice/run-77.scope"
    folder.mkdir(parents=True)
    (folder / "memory.stat").write_text("anon 3000\nfile 900000\nshmem 500\n")
    assert gemelos.group_anonymous(77, proc=proc, groups=groups) == 3500
    (folder / "memory.stat").write_text("anon 3000\nfile 900000\n")
    assert gemelos.group_anonymous(77, proc=proc, groups=groups) == 3000
    # Sin la memoria anónima del grupo o sin el proceso no hay medida.
    (folder / "memory.stat").write_text("file 900000\n")
    assert gemelos.group_anonymous(77, proc=proc, groups=groups) is None
    assert gemelos.group_anonymous(78, proc=proc, groups=groups) is None


def test_the_job_runs_in_its_own_memory_bounded_scope():
    command = gemelos.scope_command(KEY, "6G")
    assert command[:4] == ["systemd-run", "--user", "--scope", "--quiet"]
    assert "MemoryMax=6G" in command and "MemorySwapMax=256M" in command
    assert command[command.index("choom") :][:3] == ["choom", "-n", "900"]
    tail = ["-m", "mars_titan.training.comparison_campaign", "job", "--machine", "references"]
    assert command[-7:] == [*tail, "--key", KEY]
    assert command[-8] == sys.executable


def test_advance_records_each_barrier_once_and_asks_for_a_single_stop(tmp_path, capsys):
    twin = dict(state=tmp_path / "twin-a", timeline=[], stopped=False)
    assert not gemelos.advance(twin, {}, 1.0, 512)
    assert twin["timeline"] == [] and "mark" not in twin
    first = dict(global_step=256, cursor=dict(epoch=0, phase="train"))
    assert not gemelos.advance(twin, first, 10.004, 512)
    assert not gemelos.advance(twin, first, 11.0, 512)
    assert twin["timeline"] == [(10.0, 256, dict(epoch=0, phase="train"))]
    # El mismo paso con otro cursor es otra barrera, por ejemplo al empezar la validación.
    assert not gemelos.advance(
        twin, dict(first, cursor=dict(epoch=1, phase="validation")), 12.0, 512
    )
    assert len(twin["timeline"]) == 2
    last = dict(global_step=512, cursor=None)
    assert gemelos.advance(twin, last, 20.0, 512)
    assert twin["stopped"] and twin["timeline"][-2:] == [
        (20.0, 512, None),
        (20.0, 512, "stop_requested"),
    ]
    assert not gemelos.advance(twin, dict(global_step=600, cursor=None), 30.0, 512)
    assert twin["timeline"][-1] == (30.0, 600, None)
    printed = capsys.readouterr().out.splitlines()
    assert len(printed) == 4 and "twin-a: paso 256" in printed[0]


def test_outcome_joins_the_record_of_the_trial_with_its_observed_peaks(tmp_path):
    twin = dict(
        state=tmp_path / "twin-a",
        process=SimpleNamespace(returncode=3),
        timeline=[(0.0, None, "launched")],
        anonymous=3 * 2**30 + 2**29,
        vram=2856,
    )
    attempts = [dict(status="failed"), dict(status="paused", seconds=9.5, resources=dict(x=1))]
    entry = gemelos.outcome(twin, dict(status="paused", attempts=attempts))
    assert entry == dict(
        state="twin-a",
        exit_code=3,
        record_status="paused",
        attempt_status="paused",
        attempt_seconds=9.5,
        error=None,
        resources=dict(x=1),
        timeline=[(0.0, None, "launched")],
        group_anonymous_peak_gib=3.5,
        process_vram_peak_mib=2856,
    )
    empty = gemelos.outcome(twin, dict(status="pending", attempts=[]))
    assert empty["attempt_status"] is None and empty["record_status"] == "pending"


def test_device_sample_reads_the_gpu_and_the_memory_of_each_process(tmp_path, monkeypatch):
    zone = tmp_path / "zone"
    zone.write_text("65500\n")
    answers = {
        "--query-gpu": ["38, 41.8, 74, 1890, 7405"],
        "--query-compute-apps": ["4242, 2856", "77, 3938"],
    }
    monkeypatch.setattr(gemelos, "query", lambda field: answers[field.split("=")[0]])
    sample, used = gemelos.device_sample(zone)
    assert sample["gpu"] == [38.0, 41.8, 74.0, 1890.0, 7405.0] and sample["board_celsius"] == 65.5
    assert used == {4242: 2856, 77: 3938}
    assert sample["available_gib"] > 0 and len(sample["frequency_top4_ghz"]) <= 4

    def broken(field):
        raise subprocess.SubprocessError("nvidia-smi no responde")

    monkeypatch.setattr(gemelos, "query", broken)
    sample, used = gemelos.device_sample(None)
    assert sample["gpu"] is None and used == {} and sample["board_celsius"] is None


def fake_campaign(records):
    return SimpleNamespace(
        load_declaration=lambda path: ({}, "digest"),
        plan=lambda declaration, machine: [SimpleNamespace(key=KEY), SimpleNamespace(key="other")],
        Layout=lambda state, views: SimpleNamespace(
            state=state, output=lambda key: state / "runs" / key
        ),
        _environment=lambda layout, declaration: dict(os.environ, SEEN_STATE=layout.state.name),
        read_record=lambda layout, key: records[layout.state.name],
    )


class FakeProcess:
    """Proceso que sigue en marcha hasta que recibe la señal de parada.

    Si nadie se la envía termina solo con otro código, para que la prueba falle sin bloquearse.
    """

    def __init__(self, pid):
        self.pid, self.signals, self.polls = pid, [], 0

    @property
    def returncode(self):
        if self.signals:
            return 3
        return 0 if self.polls > 400 else None

    def poll(self):
        self.polls += 1
        return self.returncode

    def send_signal(self, number):
        self.signals.append(number)


def test_run_launches_every_state_and_stops_each_trial_at_its_step(tmp_path, monkeypatch):
    runtime, states = tmp_path / "runtime", [tmp_path / "twin-a", tmp_path / "twin-b"]
    for index, state in enumerate(states):
        write_run(state, rows([0.25, 0.5]), step=512 + index)
    read, reads = gemelos.read_json, {}

    def progress(path):
        # La primera lectura de cada prueba aún no ha llegado al paso pedido.
        reads[path] = reads.get(path, 0) + 1
        fit = read(path)
        return dict(fit, global_step=500) if reads[path] == 1 else fit

    monkeypatch.setattr(gemelos, "read_json", progress)
    records = {
        state.name: dict(status="paused", attempts=[dict(status="paused", seconds=4.0)])
        for state in states
    }
    launched = []

    def popen(command, **options):
        launched.append((command, options))
        return FakeProcess(4000 + len(launched))

    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/mps/pipe")
    monkeypatch.setenv("CUDA_MPS_LOG_DIRECTORY", "/mps/log")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
    before = dict(os.environ)
    monkeypatch.setattr(gemelos, "campaign_module", lambda path: fake_campaign(records))
    monkeypatch.setattr(gemelos.subprocess, "Popen", popen)
    monkeypatch.setattr(gemelos, "group_anonymous", lambda pid: (pid - 4000) * 2**30)
    samples = []

    def device(zone):
        samples.append(zone)
        return dict(gpu=[40.0], board_celsius=60.0), {4001: 2856, 4002: 3000}

    monkeypatch.setattr(gemelos, "device_sample", device)
    out = tmp_path / "result.json"
    arguments = ["run", "--key", KEY, "--runtime", str(runtime), "--until-step", "512"]
    arguments += ["--out", str(out), "--mps", "--vram-limit-mib", "3100", "--memory-max", "6G"]
    arguments += ["--alloc-conf", "expandable_segments:True", "--poll-seconds", "0"]
    arguments += ["--thermal-zone", str(tmp_path / "zone")]
    for state in states:
        arguments += ["--state", str(state)]
    assert gemelos.main(arguments) == 0
    assert dict(os.environ) == before
    assert [command for command, _ in launched] == [gemelos.scope_command(KEY, "6G")] * 2
    for (_, options), state in zip(launched, states, strict=True):
        environment = options["env"]
        assert options["cwd"] == runtime and options["start_new_session"] is True
        assert environment["SEEN_STATE"] == state.name
        assert environment["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] == "0=3100M"
        assert environment["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
        assert environment["MARS_TITAN_DIGEST_CACHE"] == str(state / "file-digests.json")
        assert (state / "logs/job.log").is_file()
    result = json.loads(out.read_text())
    assert result["states"] == ["twin-a", "twin-b"] and result["key"] == KEY
    assert (result["mps"], result["vram_limit_mib"], result["until_step"]) == (True, 3100, 512)
    assert result["alloc_conf"] == "expandable_segments:True" and result["elapsed_seconds"] >= 0
    assert samples == [tmp_path / "zone"] and result["samples"][0]["gpu"] == [40.0]
    first, second = result["twins"]
    assert (first["exit_code"], first["record_status"], first["attempt_seconds"]) == (
        3,
        "paused",
        4.0,
    )
    assert (first["process_vram_peak_mib"], second["process_vram_peak_mib"]) == (2856, 3000)
    assert (first["group_anonymous_peak_gib"], second["group_anonymous_peak_gib"]) == (1.0, 2.0)
    assert first["timeline"][0] == [0.0, None, "launched"]
    # La muestra del equipo solo llega en la primera vuelta: el pico no se pierde después.
    assert all(count == 2 for count in reads.values()) and len(reads) == 2
    assert [entry[1:] for entry in second["timeline"][1:]] == [
        [500, dict(epoch=1, phase="validation")],
        [513, dict(epoch=1, phase="validation")],
        [513, "stop_requested"],
    ]


def test_run_staggers_the_launches_and_leaves_a_finished_trial_alone(tmp_path, monkeypatch):
    runtime, states = tmp_path / "runtime", [tmp_path / "twin-a", tmp_path / "twin-b"]
    for state in states:
        write_run(state, rows([0.25, 0.5]), step=512)
    records = {state.name: dict(status="paused", attempts=[]) for state in states}

    class Ended(FakeProcess):
        """Proceso que ya había terminado antes de la primera consulta."""

        @property
        def returncode(self):
            return 0

    processes, pauses = [], []

    def popen(command, **options):
        processes.append((FakeProcess if processes else Ended)(4001 + len(processes)))
        return processes[-1]

    monkeypatch.setattr(gemelos, "campaign_module", lambda path: fake_campaign(records))
    monkeypatch.setattr(gemelos.subprocess, "Popen", popen)
    monkeypatch.setattr(gemelos, "group_anonymous", lambda pid: 2**30)
    used = {4001: 900, 4002: 2856}
    monkeypatch.setattr(gemelos, "device_sample", lambda zone: (dict(gpu=None), used))
    monkeypatch.setattr(
        gemelos.time, "sleep", lambda seconds: pauses.append((seconds, len(processes)))
    )
    out = tmp_path / "result.json"
    arguments = ["run", "--key", KEY, "--runtime", str(runtime), "--until-step", "512"]
    arguments += ["--out", str(out), "--stagger", "0.5", "--poll-seconds", "0"]
    for state in states:
        arguments += ["--state", str(state)]
    assert gemelos.main(arguments) == 0
    # La espera separa los arranques y no se repite después del último.
    assert [pause for pause in pauses if pause[0]] == [(0.5, 1)]
    ended, alive = processes
    # A una prueba que ya terminó no se le leen pasos ni picos y no se le pide que pare.
    assert ended.signals == [] and alive.signals == [signal.SIGTERM]
    first, second = json.loads(out.read_text())["twins"]
    assert (first["exit_code"], first["timeline"]) == (0, [[0.0, None, "launched"]])
    assert (first["process_vram_peak_mib"], first["group_anonymous_peak_gib"]) == (0, 0.0)
    assert (second["exit_code"], second["process_vram_peak_mib"]) == (3, 2856)


def test_run_restores_the_environment_when_the_runtime_rejects_the_state(tmp_path, monkeypatch):
    campaign = fake_campaign({})

    def refuse(layout, declaration):
        raise ValueError("El runtime y la declaración no tienen un permiso coincidente")

    campaign._environment = refuse
    monkeypatch.setattr(gemelos, "campaign_module", lambda path: campaign)
    monkeypatch.setattr(gemelos.subprocess, "Popen", lambda *a, **k: pytest.fail("Se lanzó"))
    before = dict(os.environ)
    arguments = ["run", "--key", KEY, "--runtime", str(tmp_path), "--until-step", "1"]
    arguments += ["--out", str(tmp_path / "result.json"), "--state", str(tmp_path / "twin")]
    with pytest.raises(ValueError, match="permiso"):
        gemelos.main(arguments)
    assert dict(os.environ) == before and not (tmp_path / "result.json").exists()


def test_prepare_copies_the_inputs_of_the_origin_and_registers_its_own_permission(
    tmp_path, monkeypatch
):
    campaign, runtime, state = tmp_path / "campaign", tmp_path / "runtime", tmp_path / "twin"
    for folder in ("cohorts/2019", "cohorts/2021", "indices"):
        (campaign / folder).mkdir(parents=True)
        (campaign / folder / "data.bin").write_text(folder)
    for name in gemelos.COPIED:
        (campaign / name).write_text(name)
    calls = []
    monkeypatch.setattr(
        gemelos.subprocess, "run", lambda command, **options: calls.append((command, options))
    )
    monkeypatch.setenv("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT", "0=1M")
    arguments = ["prepare", "--key", KEY, "--runtime", str(runtime), "--campaign", str(campaign)]
    arguments += ["--state", str(state), "--note", "Prueba fuera de la campaña"]
    assert gemelos.main(arguments) == 0
    assert (state / "cohorts/2019/data.bin").read_text() == "cohorts/2019"
    assert not (state / "cohorts/2021").exists()
    assert (state / "indices/data.bin").is_file()
    assert all((state / name).read_text() == name for name in gemelos.COPIED)
    ((command, options),) = calls
    assert command[:4] == [
        sys.executable,
        "-m",
        "mars_titan.training.comparison_campaign",
        "authorize",
    ]
    assert command[4:] == ["--state", str(state), "--note", "Prueba fuera de la campaña"]
    assert options["cwd"] == runtime and options["check"] is True
    assert options["env"]["PYTHONPATH"] == str(runtime / "src")
    assert "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT" not in options["env"]
    # Un estado de prueba nunca reutiliza una carpeta existente, aunque esté vacía.
    with pytest.raises(FileExistsError):
        gemelos.main(arguments)
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileExistsError):
        gemelos.main([*arguments[:-4], "--state", str(tmp_path / "empty"), "--note", "Otra"])
    assert len(calls) == 1 and list((tmp_path / "empty").iterdir()) == []


def test_the_campaign_controller_must_come_from_the_requested_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    elsewhere = SimpleNamespace(__file__=str(tmp_path / "other/comparison_campaign.py"))
    monkeypatch.setattr(gemelos.importlib, "import_module", lambda name: elsewhere)
    with pytest.raises(ValueError, match="controlador distinto"):
        gemelos.campaign_module(tmp_path / "runtime")
    assert sys.path[0] == str(tmp_path / "runtime/src")
    expected = tmp_path / "runtime/src/mars_titan/training/comparison_campaign.py"
    inside = SimpleNamespace(__file__=str(expected))
    monkeypatch.setattr(gemelos.importlib, "import_module", lambda name: inside)
    assert gemelos.campaign_module(tmp_path / "runtime") is inside


def test_query_asks_the_first_gpu_and_drops_empty_lines(monkeypatch):
    calls = []

    def run(command, **options):
        calls.append((command, options))
        return SimpleNamespace(stdout="38, 41.8\n\n   \n4242, 2856\n")

    monkeypatch.setattr(gemelos.subprocess, "run", run)
    assert gemelos.query("--query-gpu=utilization.gpu,power.draw") == ["38, 41.8", "4242, 2856"]
    ((command, options),) = calls
    assert command == [
        "nvidia-smi",
        "--id=0",
        "--query-gpu=utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    assert options == dict(capture_output=True, text=True, timeout=5, check=True)
