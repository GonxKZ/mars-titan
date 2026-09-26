"""Transiciones de salud sin usar CUDA ni modificar servicios reales."""

import copy
import json
import os
import subprocess
import sys

import pytest

from mars_titan.training import campaign_health as health


def sample(now=1000, **changes):
    value = dict(
        monotonic=now,
        timestamp=1_800_000_000 + now,
        boot_id="boot-a",
        service=dict(
            name="campaign.service",
            active="active",
            sub="running",
            result="success",
            pid=10,
            invocation="invocation-a",
            started_monotonic=500,
            exit_status=0,
        ),
        resources=dict(
            cpu_usec=100_000_000,
            read_bytes=100,
            write_bytes=100,
            memory_bytes=1000,
            processes={"20": dict(start_ticks=100, cpu_ticks=200, io_bytes=300)},
        ),
        guard={"status": "running", "observed_at": 1_800_000_000 + now},
        summary={"status": "running", "completed_runs": 0, "mtime": 1_800_000_000 + 500},
        preparation={"status": "running", "partitions": 0, "mtime": 1_800_000_000 + 500},
        activities=[],
        gpu_compute_pids=[20],
        errors=[],
    )
    value.update(changes)
    return value


def advance(previous, seconds=300):
    value = copy.deepcopy(previous["sample"])
    value["monotonic"] += seconds
    value["timestamp"] += seconds
    value["guard"]["observed_at"] = value["timestamp"]
    return value


def test_explicit_service_failure_wins_over_completed_reports_and_alerts_once():
    observed = sample(summary={"status": "completed"})
    observed["service"].update(active="failed", sub="failed", result="exit-code", pid=0)
    state = health.assess(observed)
    assert (state["phase"], state["health"]) == ("service_failed", "failed")
    assert state["notify"] is True
    assert health.assess(advance(state), state)["notify"] is False


def test_initial_inactive_service_does_not_report_an_unexpected_stop():
    observed = sample(guard=None, preparation=None)
    observed["service"].update(active="inactive", pid=0)
    state = health.assess(observed)
    assert state["phase"] == "not_started"
    assert state["notify"] is False


def test_stop_after_observed_run_alerts_unless_final_summary_is_completed():
    previous = health.assess(sample())
    stopped = advance(previous)
    stopped["service"].update(active="inactive", pid=0)
    stopped["guard"] = {"status": "stopped", "observed_at": stopped["timestamp"]}
    state = health.assess(stopped, previous)
    assert state["phase"] == "stopped"
    assert state["notify"] is True
    assert health.assess(advance(state), state)["notify"] is False
    stopped["summary"]["status"] = "completed"
    state = health.assess(stopped, previous)
    assert state["phase"] == "completed" and not state["notify"]


def test_cpu_preparation_wins_over_an_idle_cuda_context():
    state = health.assess(sample())
    assert state["phase"] == "preparing_cpu"
    later = advance(state, 1200)
    later["resources"]["processes"]["20"]["cpu_ticks"] += 500
    result = health.assess(later, state)
    assert result["phase"] == "preparing_cpu"
    assert result["health"] == "ok" and not result["notify"]
    assert result["activity"]["cpu_ticks"] == 500


def test_gpu_context_alone_does_not_prove_training():
    observed = sample(preparation={"status": "completed"})
    assert health.assess(observed)["phase"] == "running_unknown"
    observed["activities"] = [
        {"status": "running", "training": True, "mtime": observed["timestamp"]}
    ]
    assert health.assess(observed)["phase"] == "training"


def test_recent_gpu_wait_is_expected_and_stale_or_previous_guard_is_ignored():
    observed = sample()
    observed["guard"]["status"] = "waiting"
    initial = health.assess(observed)
    assert initial["phase"] == "waiting_gpu" and not initial["notify"]
    later = advance(initial, 3600)
    assert health.assess(later, initial)["health"] == "waiting"
    later["guard"]["observed_at"] -= 120
    assert health.assess(later, initial)["phase"] == "preparing_cpu"
    observed["guard"].update(status="failed", observed_at=observed["timestamp"] - 600)
    assert health.assess(observed)["health"] != "failed"


def test_heartbeat_and_supervisor_cgroup_cpu_do_not_masquerade_as_work():
    previous = health.assess(sample())
    observed = advance(previous, 900)
    observed["resources"]["cpu_usec"] += 2_000_000
    observed["preparation"]["mtime"] = observed["timestamp"]
    state = health.assess(observed, previous)
    assert state["health"] == "possible_inactivity"
    assert state["notify"] is True
    assert health.assess(advance(state), state)["notify"] is False


@pytest.mark.parametrize("counter", ["cpu_ticks", "io_bytes"])
def test_each_workload_counter_can_clear_inactivity(counter):
    initial = health.assess(sample())
    idle = health.assess(advance(initial, 900), initial)
    observed = advance(idle)
    observed["resources"]["processes"]["20"][counter] += 10
    recovered = health.assess(observed, idle)
    assert recovered["health"] == "ok" and recovered["alert_key"] is None
    again = health.assess(advance(recovered, 900), recovered)
    assert again["notify"] is True


def test_semantic_progress_clears_inactivity_without_using_file_timestamps():
    initial = health.assess(sample())
    observed = advance(initial, 900)
    observed["preparation"]["partitions"] += 1
    state = health.assess(observed, initial)
    assert state["health"] == "ok" and state["activity"]["semantic_progress"]
    observed = advance(state, 900)
    observed["summary"]["completed_runs"] += 1
    observed["preparation"]["status"] = "completed"
    state = health.assess(observed, state)
    assert state["phase"] == "training" and state["health"] == "ok"


@pytest.mark.parametrize("field,value", [("invocation", "new"), ("pid", 50)])
def test_service_restart_resets_the_reference_without_inventing_progress(field, value):
    initial = health.assess(sample())
    observed = advance(initial, 3600)
    observed["service"][field] = value
    observed["resources"]["processes"]["20"]["cpu_ticks"] = 1
    state = health.assess(observed, initial)
    assert state["activity"]["baseline_reset"] and state["health"] == "ok"
    assert not state["activity"]["semantic_progress"]


def test_reused_child_pid_and_counter_reset_do_not_invent_activity():
    initial = health.assess(sample())
    observed = advance(initial, 300)
    observed["resources"]["processes"]["20"].update(start_ticks=200, cpu_ticks=9000)
    state = health.assess(observed, initial)
    assert state["activity"]["cpu_ticks"] == 0
    assert state["activity"]["baseline_reset"]
    observed = advance(state, 300)
    observed["resources"]["processes"]["20"]["cpu_ticks"] = 1
    reset = health.assess(observed, state)
    assert reset["activity"]["cpu_ticks"] == 0 and reset["activity"]["baseline_reset"]


def test_missing_telemetry_is_unknown_instead_of_inactivity():
    initial = health.assess(sample())
    observed = advance(initial, 900)
    observed["resources"]["processes"] = {}
    observed["errors"] = ["No se puede leer el cgroup"]
    state = health.assess(observed, initial)
    assert state["health"] == "unknown"
    assert state["phase"] == "preparing_cpu"


def test_failed_observation_is_distinct_from_a_failed_service():
    observed = sample(service=None, errors=["systemctl no responde"])
    state = health.assess(observed)
    assert state["phase"] == "observation_failed" and state["notify"]


def test_report_reads_are_bounded_and_skip_heartbeat_fields(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"status": "running", "completed_runs": 4, "heartbeat_at": "x"}))
    record = health.read_report(path)
    assert record["completed_runs"] == 4 and "heartbeat_at" not in record
    path.write_bytes(b"x" * (health.MAX_JSON_BYTES + 1))
    with pytest.raises(ValueError, match="límite"):
        health.read_report(path)
    path.write_text("[]")
    with pytest.raises(ValueError, match="objeto"):
        health.read_report(path)


def test_fifo_is_rejected_without_a_blocking_read(tmp_path):
    path = tmp_path / "fifo"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular"):
        health.read_report(path)


def test_atomic_state_replacement_keeps_previous_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    health.atomic_json(path, {"value": 1})
    original = path.read_bytes()

    def reject(*args):
        raise OSError("No se puede sustituir")

    monkeypatch.setattr(os, "replace", reject)
    with pytest.raises(OSError):
        health.atomic_json(path, {"value": 2})
    assert path.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.json"]


def test_cli_runs_as_a_standalone_stdlib_file_and_deduplicates(tmp_path):
    executable = tmp_path / "systemctl"
    executable.write_text(
        "#!/bin/sh\ncat <<'PROPERTIES'\nActiveState=failed\nSubState=failed\n"
        "Result=exit-code\nMainPID=0\nInvocationID=attempt-a\nExecMainStatus=1\n"
        "ExecMainStartTimestampMonotonic=0\nControlGroup=\nPROPERTIES\n"
    )
    executable.chmod(0o755)
    state = tmp_path / "state.json"
    env = os.environ | {"PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]}
    command = [sys.executable, "-S", health.__file__, "--state", str(state)]
    first = subprocess.run(command, capture_output=True, text=True, env=env, timeout=5)
    assert first.returncode == 0, first.stderr
    assert "AVISO" in first.stdout
    second = subprocess.run(command, capture_output=True, text=True, env=env, timeout=5)
    assert second.returncode == 0, second.stderr
    assert "AVISO" not in second.stdout
    assert json.loads(state.read_text())["phase"] == "service_failed"


def test_inactive_completed_guard_is_terminal_evidence_even_after_poll_interval():
    previous = health.assess(sample())
    observed = advance(previous, 300)
    observed["service"].update(active="inactive", pid=0)
    observed["guard"] = {"status": "completed", "observed_at": observed["timestamp"] - 200}
    state = health.assess(observed, previous)
    assert state["phase"] == "completed" and not state["notify"]


def test_command_limits_output_time_and_checks_exit_status():
    assert health._command([sys.executable, "-c", "print('ok')"]).strip() == "ok"
    with pytest.raises(ValueError, match="límite"):
        health._command([sys.executable, "-c", "print('x'*100000)"])
    with pytest.raises(subprocess.TimeoutExpired):
        health._command([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.05)
    with pytest.raises(subprocess.CalledProcessError):
        health._command([sys.executable, "-c", "raise SystemExit(2)"])


def write_process(root, pid, name="python", *, cpu=20, start=100, script="worker.py"):
    folder = root / str(pid)
    folder.mkdir(parents=True)
    fields = ["S"] + ["0"] * 30
    fields[11], fields[12], fields[19] = str(cpu), "10", str(start)
    (folder / "stat").write_text(f"{pid} ({name}) " + " ".join(fields))
    (folder / "io").write_text("rchar: 100\nwchar: 20\nread_bytes: 30\nwrite_bytes: 40\n")
    (folder / "cmdline").write_bytes(f"python\0{script}\0--\0worker.py\0".encode())


def test_cgroup_collection_bounds_processes_and_excludes_guard_overhead(tmp_path):
    group, proc = tmp_path / "cgroup/unit", tmp_path / "proc"
    group.mkdir(parents=True)
    (group / "cpu.stat").write_text("usage_usec 100\nuser_usec 80\nsystem_usec 20\n")
    (group / "memory.current").write_text("4096\n")
    (group / "io.stat").write_text("259:0 rbytes=150 wbytes=80 rios=2 wios=1\n")
    (group / "cgroup.procs").write_text("10\n20\n21\n30\n40\n")
    write_process(proc, 10)
    write_process(proc, 20, "worker (data)")
    write_process(proc, 21, script="/local/version/gpu_supervisor.py")
    write_process(proc, 30, "nvidia-smi")
    service = sample()["service"] | {"cgroup": "/unit"}
    resources, errors = health.read_resources(
        service, exclude_supervisor=True, cgroup_root=group.parent, proc_root=proc
    )
    assert not errors
    assert set(resources["processes"]) == {"20"}
    assert resources["processes"]["20"] == {"start_ticks": 100, "cpu_ticks": 30, "io_bytes": 190}
    assert (resources["cpu_usec"], resources["memory_bytes"], resources["read_bytes"]) == (
        100,
        4096,
        150,
    )
    (group / "io.stat").unlink()
    resources, errors = health.read_resources(
        service, exclude_supervisor=True, cgroup_root=group.parent, proc_root=proc
    )
    assert set(resources["processes"]) == {"20"}
    assert any("io.stat" in error for error in errors)
    (group / "cgroup.procs").write_text("20\n" * 129)
    resources, errors = health.read_resources(service, cgroup_root=group.parent, proc_root=proc)
    assert not resources["processes"] and any("128" in error for error in errors)
    with pytest.raises(ValueError, match="raíz"):
        health.read_resources(service | {"cgroup": "/../escape"}, cgroup_root=group.parent)


def test_collect_handles_missing_and_malformed_optional_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "read_service", lambda name: sample()["service"])
    monkeypatch.setattr(health, "read_resources", lambda *a, **k: (sample()["resources"], []))
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    state = health.collect("campaign.service", summary=tmp_path / "missing", preparation=bad)
    assert state["summary"] is None and state["preparation"] is None
    assert len(state["errors"]) == 1


def test_new_activity_file_does_not_hide_progress_in_an_existing_file():
    initial = health.assess(sample(activities=[{"path": "b.json", "global_step": 1}]))
    observed = advance(initial)
    observed["activities"] = [
        {"path": "a.json", "global_step": 0},
        {"path": "b.json", "global_step": 2},
    ]
    result = health.assess(observed, initial)
    assert result["activity"]["semantic_progress"]


def test_reboot_clears_seen_run_and_does_not_alert_before_service_starts():
    initial = health.assess(sample())
    observed = advance(initial)
    observed["boot_id"] = "boot-b"
    observed["service"].update(active="inactive", pid=0)
    assert health.assess(observed, initial)["phase"] == "not_started"


def test_cli_recovers_from_an_incompatible_previous_sample(tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"schema_version": 1, "sample": {"boot_id": "boot-a"}}))
    monkeypatch.setattr(health, "collect", lambda *args, **kwargs: sample())
    args = health.argparse.Namespace(
        state=state,
        service="campaign.service",
        guard=None,
        summary=None,
        preparation=None,
        activity=[],
        probe_gpu=False,
        notify=False,
        idle_seconds=900,
    )
    result = health.check(args)
    assert result["phase"] == "preparing_cpu"
    assert any("Estado anterior" in error for error in result["sample"]["errors"])


def test_notification_is_attempted_once_and_failure_is_visible(tmp_path, monkeypatch, capsys):
    observed = sample()
    observed["service"].update(active="failed", pid=0, result="exit-code")
    monkeypatch.setattr(health, "collect", lambda *args, **kwargs: copy.deepcopy(observed))
    calls = []

    def missing_notifier(*args, **kwargs):
        calls.append(args[0])
        raise FileNotFoundError("notify-send no está instalado")

    monkeypatch.setattr(subprocess, "run", missing_notifier)
    args = health.argparse.Namespace(
        state=tmp_path / "state.json",
        service="campaign.service",
        guard=None,
        summary=None,
        preparation=None,
        activity=[],
        probe_gpu=False,
        notify=True,
        idle_seconds=900,
    )
    health.check(args)
    health.check(args)
    output = capsys.readouterr().out
    assert len(calls) == 1 and calls[0][0] == "notify-send"
    assert output.count("AVISO") == 1 and "No se pudo mostrar" in output


@pytest.mark.parametrize("arguments", [["--service", "--bad"], ["--idle-seconds", "0"]])
def test_cli_rejects_invalid_configuration(tmp_path, arguments):
    result = subprocess.run(
        [sys.executable, "-S", health.__file__, "--state", str(tmp_path / "out.json"), *arguments],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 2


def test_json_nan_and_nonfinite_counter_cannot_be_saved_as_activity(tmp_path):
    path = tmp_path / "report.json"
    path.write_text('{"status":"running","global_step":NaN,"completed_steps":true}')
    with pytest.raises(ValueError, match="contador"):
        health.read_report(path)


def test_unknown_systemd_unit_is_an_observation_error(monkeypatch):
    monkeypatch.setattr(
        health, "_command", lambda command: "LoadState=not-found\nActiveState=inactive\n"
    )
    with pytest.raises(ValueError, match="no existe"):
        health.read_service("missing.service")


def test_different_service_does_not_inherit_a_previous_run():
    initial = health.assess(sample())
    observed = advance(initial)
    observed["service"].update(name="other.service", active="inactive", pid=0)
    assert health.assess(observed, initial)["phase"] == "not_started"


def test_observation_failure_preserves_run_history_and_deduplicates_its_alert():
    initial = health.assess(sample())
    failure = advance(initial)
    failure["service"] = None
    failure["service_name"] = "campaign.service"
    failed = health.assess(failure, initial)
    assert failed["has_run"] and failed["notify"]
    repeated = health.assess(advance(failed), failed)
    assert repeated["has_run"] and not repeated["notify"]
    stopped = sample(now=1900)
    stopped["service"].update(active="inactive", pid=0)
    final = health.assess(stopped, repeated)
    assert final["phase"] == "stopped" and final["notify"]


@pytest.mark.parametrize("invalid", [{"phase": {}}, {"completed_runs": 10**400}])
def test_invalid_report_fields_remain_visible_without_aborting_collection(
    tmp_path, monkeypatch, invalid
):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(invalid))
    monkeypatch.setattr(health, "read_service", lambda name: sample()["service"])
    monkeypatch.setattr(
        health, "read_resources", lambda *args, **kwargs: (sample()["resources"], [])
    )
    result = health.collect("campaign.service", summary=path)
    assert result["summary"] is None
    assert result["errors"]


@pytest.mark.parametrize(
    "failure", [PermissionError(13, "Acceso denegado"), "rchar: no\n", "wchar: 4\n"]
)
def test_unavailable_process_io_preserves_cpu_and_reports_the_error(tmp_path, monkeypatch, failure):
    group, proc = tmp_path / "cgroup/unit", tmp_path / "proc"
    group.mkdir(parents=True)
    for name, content in {
        "cpu.stat": "usage_usec 100\n",
        "memory.current": "4096\n",
        "io.stat": "",
        "cgroup.procs": "20\n",
    }.items():
        (group / name).write_text(content)
    write_process(proc, 20)
    original_read = health._read

    def limited_read(path, limit):
        if path == proc / "20/io":
            if isinstance(failure, str):
                return failure.encode(), None
            raise failure
        return original_read(path, limit)

    monkeypatch.setattr(health, "_read", limited_read)
    service = sample()["service"] | {"cgroup": "/unit"}
    before, errors = health.read_resources(service, cgroup_root=group.parent, proc_root=proc)
    assert before["processes"]["20"] == {"start_ticks": 100, "cpu_ticks": 30, "io_bytes": None}
    assert any("PID 20" in error and "io" in error for error in errors)
    initial = health.assess(sample(resources=before, errors=errors))
    stat_path = proc / "20/stat"
    stat_path.write_text(stat_path.read_text().replace("20 10", "70 10"))
    after, errors = health.read_resources(service, cgroup_root=group.parent, proc_root=proc)
    observed = advance(initial, 1200)
    observed.update(resources=after, errors=errors)
    result = health.assess(observed, initial)
    assert result["activity"]["cpu_ticks"] == 50
    assert result["activity"]["io_bytes"] == 0
    assert result["health"] == "ok" and not result["notify"]


@pytest.mark.parametrize("before,after", [(None, 500), (500, None), (None, None)])
def test_unknown_io_values_never_invent_an_increment(before, after):
    observed = sample()
    observed["resources"]["processes"]["20"]["io_bytes"] = before
    initial = health.assess(observed)
    observed = advance(initial)
    observed["resources"]["processes"]["20"]["io_bytes"] = after
    result = health.assess(observed, initial)
    assert result["activity"]["io_bytes"] == 0
    assert result["activity"]["comparable"]
    assert not result["activity"]["baseline_reset"]


@pytest.mark.parametrize("failure", [FileNotFoundError(), ProcessLookupError()])
def test_disappeared_process_is_still_skipped(tmp_path, monkeypatch, failure):
    write_process(tmp_path, 20)
    original_read = health._read

    def vanished_io(path, limit):
        if path == tmp_path / "20/io":
            raise failure
        return original_read(path, limit)

    monkeypatch.setattr(health, "_read", vanished_io)
    with pytest.raises(type(failure)):
        health._process(20, tmp_path)
