"""Determinismo declarado de PyTorch en las recetas y los recorridos (#794).

Cada prueba empieza sin el modo determinista y restaura al final las banderas globales que
encontró. No se ejecuta ningún paso de optimizador.
"""

import os
import subprocess
import sys

import pytest
import torch

from mars_titan.training import determinism
from mars_titan.training.candidate_run import CandidateRecipe
from mars_titan.training.financial_run import ChronologicalRecipe
from mars_titan.training.mars_titan_run import ReadoutRecipe
from tests.training import test_candidate_run as candidate
from tests.training import test_mars_titan_run as readout
from tests.training.test_financial_run import (
    explicit_fastpath,  # noqa: F401
    shared,  # noqa: F401
    trainer,
)

RECIPES = (ChronologicalRecipe, ReadoutRecipe, CandidateRecipe)
POLICY = determinism.TORCH_DETERMINISTIC
MODULE = "mars_titan.training.determinism"


@pytest.fixture(autouse=True)
def restore_flags(monkeypatch):
    flags = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = False, False
    monkeypatch.setenv(determinism.WORKSPACE_VARIABLE, ":4096:8")
    yield
    torch.use_deterministic_algorithms(flags[0], warn_only=flags[1])
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = flags[2:]


def test_the_policy_turns_on_the_deterministic_algorithms_and_cudnn():
    torch.backends.cudnn.benchmark = True
    identity = determinism.apply_determinism(POLICY)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    assert identity == dict(
        schema_version=1,
        policy=POLICY,
        deterministic_algorithms=True,
        warn_only=False,
        cudnn_deterministic=True,
        cudnn_benchmark=False,
        cublas_workspace=":4096:8",
    )
    assert determinism.determinism_identity(POLICY) == identity
    determinism.require_determinism(POLICY, identity)


@pytest.mark.parametrize("workspace", [None, "", ":4096:2", "4096:8", ":16:8 "])
def test_the_policy_needs_a_deterministic_cublas_workspace(monkeypatch, workspace):
    if workspace is None:
        monkeypatch.delenv(determinism.WORKSPACE_VARIABLE)
    else:
        monkeypatch.setenv(determinism.WORKSPACE_VARIABLE, workspace)
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG=:4096:8"):
        determinism.apply_determinism(POLICY)
    # El proceso no cambia si falta el espacio de trabajo.
    assert not torch.are_deterministic_algorithms_enabled()


def test_both_documented_workspaces_are_accepted(monkeypatch):
    for workspace in determinism.WORKSPACES:
        monkeypatch.setenv(determinism.WORKSPACE_VARIABLE, workspace)
        assert determinism.apply_determinism(POLICY)["cublas_workspace"] == workspace


def test_a_workspace_written_after_cuda_started_is_refused(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(determinism, "_initial_workspace", lambda: None)
    with pytest.raises(ValueError, match="después de iniciar CUDA"):
        determinism.apply_determinism(POLICY)
    monkeypatch.setattr(determinism, "_initial_workspace", lambda: ":16:8")
    with pytest.raises(ValueError, match="después de iniciar CUDA"):
        determinism.apply_determinism(POLICY)
    # Heredada desde el arranque, la misma variable sí vale con CUDA iniciada.
    monkeypatch.setattr(determinism, "_initial_workspace", lambda: ":4096:8")
    assert determinism.apply_determinism(POLICY)["cublas_workspace"] == ":4096:8"


def test_the_initial_workspace_is_the_one_inherited_at_start():
    code = (
        "import os\n"
        "from mars_titan.training.determinism import _initial_workspace as initial\n"
        "before = initial()\n"
        "os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'\n"
        "print(before, initial())\n"
    )
    environment = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
    for inherited, expected in ((True, ":4096:8"), (False, "None")):
        if not inherited:
            environment.pop("CUBLAS_WORKSPACE_CONFIG")
        result = subprocess.run(
            [sys.executable, "-c", code], env=environment, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.split() == [expected, expected]


def test_the_initial_workspace_is_unknown_without_proc(monkeypatch):
    def unreadable(self):
        raise OSError("sin /proc")

    monkeypatch.setattr(determinism.Path, "read_bytes", unreadable)
    assert determinism._initial_workspace() is None


@pytest.mark.parametrize(
    "leave",
    [
        lambda: torch.use_deterministic_algorithms(False),
        lambda: torch.use_deterministic_algorithms(True, warn_only=True),
        lambda: setattr(torch.backends.cudnn, "deterministic", False),
        lambda: setattr(torch.backends.cudnn, "benchmark", True),
    ],
    ids=["algorithms", "warn_only", "cudnn_deterministic", "cudnn_benchmark"],
)
def test_the_identity_fails_when_the_process_leaves_the_policy(leave):
    identity = determinism.apply_determinism(POLICY)
    leave()
    with pytest.raises(ValueError, match="determinismo declarado"):
        determinism.determinism_identity(POLICY)
    with pytest.raises(ValueError, match="determinismo declarado"):
        determinism.require_determinism(POLICY, identity)


def test_a_changed_workspace_changes_the_identity(monkeypatch):
    identity = determinism.apply_determinism(POLICY)
    monkeypatch.setenv(determinism.WORKSPACE_VARIABLE, ":16:8")
    with pytest.raises(ValueError, match="cambió durante el recorrido"):
        determinism.require_determinism(POLICY, identity)


@pytest.mark.parametrize("policy", ["deterministic", POLICY.upper(), True, 1, ""])
def test_only_the_declared_policy_is_accepted(policy):
    assert not determinism.valid_determinism(policy)
    with pytest.raises(ValueError, match="determinismo"):
        determinism.apply_determinism(policy)
    with pytest.raises(ValueError, match="determinismo declarado"):
        determinism.determinism_identity(policy)
    for recipe in RECIPES:
        with pytest.raises(ValueError, match="determinismo de la receta"):
            recipe(determinism=policy)
    assert not torch.are_deterministic_algorithms_enabled()


def test_undeclared_determinism_keeps_the_process_and_the_previous_identity():
    assert determinism.valid_determinism(None) and determinism.valid_determinism(POLICY)
    assert determinism.declared_determinism(None) is None
    assert not torch.are_deterministic_algorithms_enabled()
    determinism.require_determinism(None, None)
    for recipe in RECIPES:
        assert "determinism" not in recipe().identity()
        declared = recipe(determinism=POLICY).identity()
        assert declared["determinism"] == POLICY
        assert {k: v for k, v in declared.items() if k != "determinism"} == recipe().identity()


def test_declared_determinism_enters_the_run_identity_and_is_checked(shared, tmp_path):  # noqa: F811
    _, streams = shared
    plain = trainer(streams, tmp_path / "plain")
    assert "determinism" not in plain.identity
    assert MODULE not in plain.identity["implementation"]
    assert plain.identity["numerics"]["deterministic"] is False
    fixed = trainer(streams, tmp_path / "fixed", determinism=POLICY)
    assert fixed.identity["determinism"] == determinism.determinism_identity(POLICY)
    assert fixed.identity["recipe"]["determinism"] == POLICY
    assert fixed.identity["numerics"]["deterministic"] is True
    assert MODULE in fixed.identity["implementation"]
    assert fixed.run_id != plain.run_id
    fixed._check_runtime()
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.raises(ValueError, match="configuración numérica"):
        fixed._check_runtime()


def test_the_readout_run_declares_and_checks_its_determinism(shared, tmp_path):  # noqa: F811
    _, streams = shared
    plan = readout.recipe(determinism=POLICY)
    plain = readout.build(streams, tmp_path / "plain", None, admission="m0")
    assert "determinism" not in plain.identity
    assert MODULE not in plain.identity["implementation"]
    fixed = readout.build(streams, tmp_path / "fixed", None, admission="m0", plan=plan)
    assert fixed.identity["determinism"] == determinism.determinism_identity(POLICY)
    assert MODULE in fixed.identity["implementation"]
    fixed._check_runtime()
    torch.backends.cudnn.benchmark = True
    with pytest.raises(ValueError, match="configuración numérica"):
        fixed._check_runtime()


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_the_candidate_run_declares_and_checks_its_determinism(shared, tmp_path):  # noqa: F811
    _, streams = shared
    plain = candidate.trainer(streams, tmp_path / "plain")
    assert "determinism" not in plain.identity
    assert MODULE not in plain.identity["implementation"]
    fixed = candidate.trainer(streams, tmp_path / "fixed", determinism=POLICY)
    assert fixed.identity["determinism"] == determinism.determinism_identity(POLICY)
    assert MODULE in fixed.identity["implementation"]
    fixed._check_runtime()
    torch.backends.cudnn.deterministic = False
    with pytest.raises(ValueError, match="configuración numérica"):
        fixed._check_runtime()
