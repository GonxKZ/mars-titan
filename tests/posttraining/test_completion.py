"""Ordenar las dependencias y recuperar una campaña sin duplicar trabajo confirmado."""

import json
import sys

import pytest

from mars_titan.posttraining.completion import run_child, stage_plan, wait_dependency
from mars_titan.training.checkpoints import StopRequest


def test_stage_order_completes_every_training_before_opening_evaluation():
    stages = stage_plan(["fold-000", "fold-001"], tabular_runs=17, adjustment_runs=132)
    assert [(s["fold"], s["stage"]) for s in stages] == [
        ("fold-000", "tabular"),
        ("fold-000", "posttraining"),
        ("fold-001", "tabular"),
        ("fold-001", "posttraining"),
        ("fold-000", "evaluation"),
        ("fold-001", "evaluation"),
    ]
    assert sum(s["planned_runs"] for s in stages if s["stage"] != "evaluation") == 298


def test_dependency_does_not_admit_a_partial_or_changed_count(tmp_path):
    path = tmp_path / "predecessor.json"
    path.write_text(
        json.dumps(
            dict(status="completed", planned_runs=216, completed_runs=215, final_test_opened=False)
        )
    )
    with pytest.raises(ValueError, match="recuento"):
        wait_dependency(path, StopRequest(), lambda state: None)
    path.write_text(
        json.dumps(
            dict(status="failed", planned_runs=216, completed_runs=75, final_test_opened=False)
        )
    )
    with pytest.raises(RuntimeError, match="previa"):
        wait_dependency(path, StopRequest(), lambda state: None)
    path.write_text(
        json.dumps(
            dict(status="running", planned_runs=216, completed_runs=75, final_test_opened=False)
        )
    )
    seen = []
    stop = StopRequest()

    def observe(state):
        seen.append(state)
        stop.requested = True

    with pytest.raises(InterruptedError):
        wait_dependency(path, stop, observe, poll_seconds=0.001)
    assert seen == ["waiting_dependency"]


def test_child_must_confirm_its_result_even_with_zero_exit_code(tmp_path):
    output = tmp_path / "summary.json"
    with pytest.raises(ValueError, match="resultado"):
        run_child(
            [sys.executable, "-c", "pass"],
            output,
            3,
            StopRequest(),
            lambda _: None,
            poll_seconds=0.01,
        )
    command = [
        sys.executable,
        "-c",
        "import json,sys\nfrom pathlib import Path\n"
        "Path(sys.argv[1]).write_text(json.dumps(dict(status='completed',"
        "completed_runs=3,planned_runs=3,final_test_opened=False)))",
        str(output),
    ]
    progress = []
    result = run_child(command, output, 3, StopRequest(), progress.append, poll_seconds=0.01)
    assert result["completed_runs"] == 3
    assert progress[-1]["status"] == "completed"
    with pytest.raises(ValueError, match="recuento"):
        run_child(command, output, 4, StopRequest(), progress.append, poll_seconds=0.01)


def test_completed_stage_and_hash_are_confirmed_together_after_a_cut(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from mars_titan.data.storage import atomic_json
    from mars_titan.posttraining import analysis, completion

    args = SimpleNamespace(
        reference=tmp_path / "references",
        encoded=tmp_path / "encoded/manifest.json",
        tabular_config=tmp_path / "tabular.json",
        post_config=tmp_path / "post.json",
        output=tmp_path / "output",
        after=None,
    )
    stages = [dict(fold="fold-000", stage="tabular", planned_runs=3)]
    monkeypatch.setattr(completion, "_inputs", lambda args: ({"test": True}, stages))
    analysed = []

    def analysis_result(path):
        analysed.append(path)
        atomic_json(path / "analysis.json", dict(status="completed"))
        (path / "analysis.md").write_text("Evaluación de prueba terminada.\n")

    monkeypatch.setattr(analysis, "analyse_completion", analysis_result)
    result = dict(status="completed", completed_runs=3, planned_runs=3, final_test_opened=False)
    calls = 0

    def child(command, receipt, expected, stop, observe):
        nonlocal calls
        calls += 1
        receipt.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(receipt, result)
        observe(result)
        if calls == 1:
            raise InterruptedError("Corte después de confirmar el hijo")
        return result

    monkeypatch.setattr(completion, "run_child", child)
    paused = completion.run_completion(args, StopRequest())
    assert paused["status"] == "paused"
    assert paused["stages"][0]["status"] == "running"
    assert not analysed
    finished = completion.run_completion(args, StopRequest())
    assert finished["status"] == "completed"
    assert len(finished["stages"][0]["sha256"]) == 64
    assert completion.run_completion(args, StopRequest())["status"] == "completed"
    assert calls == 2
    assert len(analysed) == 1


def test_stop_reaps_only_its_unresponsive_child_after_the_grace_period(tmp_path):
    import os

    output = tmp_path / "summary.json"
    code = (
        "import json,os,signal,sys,time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "Path(sys.argv[1]).write_text(json.dumps(dict(status='running',completed_runs=0,"
        "planned_runs=3,final_test_opened=False,pid=os.getpid())))\n"
        "while True: time.sleep(1)\n"
    )
    stop = StopRequest()

    def observe(result):
        stop.requested = True

    with pytest.raises(InterruptedError):
        run_child(
            [sys.executable, "-c", code, str(output)],
            output,
            3,
            stop,
            observe,
            poll_seconds=0.02,
            grace_seconds=0.05,
        )
    pid = json.loads(output.read_text())["pid"]
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_completed_dependency_retains_its_exact_receipt_hash(tmp_path):
    from mars_titan.data.storage import sha256

    path = tmp_path / "complete.json"
    path.write_text(
        json.dumps(
            dict(status="completed", planned_runs=216, completed_runs=216, final_test_opened=False)
        )
    )
    result = wait_dependency(path, StopRequest(), lambda _: pytest.fail("No debe esperar"))
    assert result == dict(path=str(path.resolve()), sha256=sha256(path))


@pytest.mark.parametrize("stage", [None, "tabular", "posttraining", "evaluation"])
def test_cli_routes_each_phase_and_preserves_paused_exit_status(monkeypatch, stage):
    from mars_titan.posttraining import completion

    args = [
        "completion",
        "--reference",
        "reference",
        "--encoded",
        "encoded",
        "--tabular-config",
        "tabular.json",
        "--post-config",
        "post.json",
        "--output",
        "output",
    ]
    if stage:
        args.extend(["--stage", stage, "--fold", "fold-003"])
    monkeypatch.setattr(sys, "argv", args)
    received = []

    def child(options):
        received.append((options.stage, options.fold))
        return 2

    monkeypatch.setattr(completion, "_stage", child)
    monkeypatch.setattr(completion, "run_completion", lambda args, stop: dict(status="paused"))
    assert completion.main() == 2
    assert received == ([(stage, "fold-003")] if stage else [])
    if stage:
        monkeypatch.setattr(sys, "argv", args[:-1] + ["../outside"])
        with pytest.raises(SystemExit, match="2"):
            completion.main()


@pytest.mark.parametrize("stage", ["tabular", "posttraining", "evaluation"])
def test_stage_worker_calls_one_backend_and_takes_no_nested_gpu_lease(tmp_path, monkeypatch, stage):
    from types import SimpleNamespace

    from mars_titan.posttraining import completion, heldout, queue
    from mars_titan.training import baseline_queue, experiment_resources

    calls = []
    result = dict(status="completed", completed_runs=1, planned_runs=1)

    def execute(*args, **kwargs):
        calls.append("backend")
        assert kwargs["stop"].requested is False
        return result

    class Lease:
        def __enter__(self):
            calls.append("lease")
            return self

        def __exit__(self, *args):
            calls.append("release")

        def check(self):
            calls.append("check")

    monkeypatch.setattr(experiment_resources, "GpuLease", Lease)
    monkeypatch.setattr(baseline_queue, "run_queue", execute)
    monkeypatch.setattr(queue, "run_queue", execute)
    monkeypatch.setattr(heldout, "run_evaluation", execute)
    args = SimpleNamespace(
        stage=stage,
        fold="fold-000",
        reference=tmp_path / "reference",
        output=tmp_path / "output",
        encoded=tmp_path / "encoded",
        tabular_config=tmp_path / "tabular.json",
        post_config=tmp_path / "post.json",
    )
    assert completion._stage(args) == 0
    assert calls == (
        ["lease", "backend", "check", "release"] if stage == "tabular" else ["backend"]
    )
