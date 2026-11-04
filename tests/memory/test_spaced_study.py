"""Configuración E2 y rechazos tempranos de su orden, sin aprendizaje."""

import copy
import json
import runpy
from pathlib import Path

import pytest

from mars_titan.training.learning_hold import LearningHoldError

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/spaced-utility-study-v1.json"


def test_saved_study_is_independent_and_keeps_all_original_controls():
    from mars_titan.memory.spaced_study import load_study

    study, digest = load_study(CONFIG)
    assert len(digest) == 64
    assert study.arms == (
        "baseline",
        "reservoir",
        "utility",
        "permuted_utility",
        "uniform_frontier",
        "recent",
    )
    assert study.retention.capacity == 1024
    assert study.retention.seed == 73
    assert study.max_seconds == 900.0
    assert study.max_phase_rows == 1_000_000
    assert study.retention_for("baseline") is None
    assert study.retention_for("reservoir").policy == "reservoir"


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(schema_version=True),
        lambda d: d.update(default_enabled=True),
        lambda d: d.update(final_test_access=True),
        lambda d: d.update(unknown=1),
        lambda d: d["retention"].update(capacity=0),
        lambda d: d["retention"].update(policy="recent"),
        lambda d: d.update(arms=["baseline", "baseline"]),
        lambda d: d.update(arms=["utility"]),
        lambda d: d["compute_budget"].update(max_seconds=0),
        lambda d: d["compute_budget"].update(max_phase_rows=True),
    ],
)
def test_invalid_study_is_rejected_without_changing_the_saved_config(tmp_path, change):
    from mars_titan.memory.spaced_study import load_study

    original = json.loads(CONFIG.read_text())
    value = copy.deepcopy(original)
    change(value)
    path = tmp_path / "study.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_study(path)
    assert json.loads(CONFIG.read_text()) == original


def test_window_guard_precedes_every_input_read_and_output_creation(tmp_path, monkeypatch):
    from mars_titan.training import spaced_window
    from mars_titan.training.spaced_window import run_spaced_window

    monkeypatch.setattr(
        spaced_window, "GpuLease", lambda: pytest.fail("Reservó GPU antes de la guarda")
    )
    with pytest.raises(LearningHoldError):
        run_spaced_window(
            tmp_path / "view",
            tmp_path / "anchor",
            tmp_path / "source",
            tmp_path / "study",
            arm="utility",
            output=tmp_path / "output",
        )
    assert not (tmp_path / "output").exists()


def test_cli_guard_precedes_every_input_read_and_gpu_reservation(tmp_path, monkeypatch):
    from mars_titan.training import spaced_window

    path = ROOT / "scripts/run_spaced_window.py"
    assert path.is_file(), "Falta la orden del estudio congelado E2"
    main = runpy.run_path(str(path), run_name="spaced_cli_test")["main"]
    monkeypatch.setattr(
        spaced_window,
        "run_spaced_window",
        lambda *_a, **_k: pytest.fail("Entró al controlador antes de la guarda CLI"),
    )
    args = []
    for name in ("view", "anchor", "source-manifest", "study", "output"):
        args += ["--" + name, str(tmp_path / name)]
    args += ["--arm", "utility"]
    with pytest.raises(LearningHoldError):
        main(args)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("device,status,code", [("cpu", "completed", 0), ("cuda:0", "paused", 3)])
def test_cli_passes_the_declared_options_to_the_protected_controller(
    tmp_path, monkeypatch, capsys, device, status, code
):
    from mars_titan.training import spaced_window

    main = runpy.run_path(
        str(ROOT / "scripts/run_spaced_window.py"), run_name="spaced_cli_fixture"
    )["main"]
    calls = []

    def execute(view, anchor, source, study, **options):
        assert (view, anchor, source, study) == tuple(
            tmp_path / name for name in ("view", "anchor", "source-manifest", "study")
        )
        assert options["arm"] == "utility" and options["device"] == device
        assert options["output"] == tmp_path / "output" and options["indices"] is None
        assert options["stop"].requested is False
        calls.append("run")
        return dict(status=status, run_id="fixture_without_execution")

    monkeypatch.setitem(
        main.__globals__, "require_learning_allowed", lambda *_: calls.append("hold")
    )
    monkeypatch.setattr(spaced_window, "run_spaced_window", execute)
    args = []
    for name in ("view", "anchor", "source-manifest", "study", "output"):
        args += ["--" + name, str(tmp_path / name)]
    args += ["--arm", "utility", "--device", device]
    assert main(args) == code
    assert calls == ["hold", "run"]
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_session_metrics_average_assets_before_sessions():
    import pyarrow as pa

    from mars_titan.training.spaced_window import _session_table

    table = pa.table(
        dict(
            market=["US", "US", "US"],
            prediction_at=pa.array([1, 1, 2], type=pa.timestamp("us", tz="UTC")),
            prediction=[2.0, 0.0, 9.0],
            target=[0.0, 0.0, 0.0],
        )
    )
    metrics = dict(labels=3, session_count=2, session_mae=5.0, session_mse=41.5)
    result = _session_table([table], metrics)
    assert result["samples"].to_pylist() == [2, 1]
    assert result["mae"].to_pylist() == [1.0, 9.0]


def test_unknown_retention_arm_is_rejected():
    from mars_titan.memory.spaced_study import load_study

    study, _ = load_study(CONFIG)
    with pytest.raises(ValueError):
        study.retention_for("unknown")


def test_budget_checks_the_supplied_resource_lease_without_running_cuda():
    from types import SimpleNamespace

    from mars_titan.training.spaced_window import _Budget

    checks = []
    budget = _Budget(
        SimpleNamespace(requested=False),
        60,
        lease=SimpleNamespace(check=lambda: checks.append("check")),
    )
    assert budget.requested is False and checks == ["check"]
