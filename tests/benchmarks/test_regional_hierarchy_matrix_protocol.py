"""Comprobar condiciones y pruebas de igualdad sin inicializar CUDA."""

import copy
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/regional_hierarchy_matrix_cuda.py"


def api():
    assert SCRIPT.is_file(), "Falta el benchmark de la jerarquía regional"
    return runpy.run_path(str(SCRIPT), run_name="hierarchy_matrix_protocol_test")


def test_protocol_has_eight_controls_and_four_explicit_strata():
    module = api()
    plan = module["protocol"]()
    assert len(plan["cells"]) == 16
    assert len({row["cell_id"] for row in plan["cells"]}) == 16
    assert plan["condition_records"] == 128
    assert plan["prefix_prepare_calls"] == 256
    assert plan["timed_prepare_calls"] == 896
    assert plan["total_prepare_calls"] == 1152
    assert plan["optimizer_steps"] == 0
    assert plan["scientific_evaluation"] is False
    assert plan["final_test_access"] is False
    assert plan["device"] == "cuda:0"
    assert set(plan["dtypes"]) == {"float32", "float64"}
    assert plan["batches"] == [1, 256]
    assert all(cell["variant"] == "mac_online" for cell in plan["cells"])
    assert {cell["context"] for cell in plan["cells"]} == {None, "market_memory_v1"}
    assert sum(cell["arm"] == "legacy_self_uniform_interleaved" for cell in plan["cells"]) == 2
    assert len(module["execution_order"](plan["cells"], 0)) == 16
    assert module["execution_order"](plan["cells"], 1) == list(
        reversed(module["execution_order"](plan["cells"], 0))
    )


def test_per_cell_cache_drops_only_derived_hierarchy_geometry():
    module = api()
    known = {
        "result/qkv.weight": object(),
        "result/buffer/regional_fusion.edges": object(),
        "result/buffer/regional_execution.geometry": object(),
    }
    original = dict(known)
    module["discard_geometry_proof"](known)
    assert set(known) == {"result/qkv.weight", "result/buffer/regional_fusion.edges"}
    assert all(known[name] is original[name] for name in known)


def test_protocol_preserves_each_registered_topology_partition_and_readout():
    plan = api()["protocol"]()
    base = dict(topology="tree", groups=[[0, 2], [1, 3, 4]], feedback=True, readout="local_only")
    assert plan["arm_configurations"] == {
        "legacy_self_uniform_interleaved": dict(
            selection="fixed", program="summary_interleaved_v1", context_threshold=None
        ),
        "hierarchy_tree_local": base,
        "hierarchy_star_local": dict(base, topology="star"),
        "hierarchy_permuted_tree_local": dict(base, groups=[[0, 1], [2, 3, 4]]),
        "hierarchy_tree_no_feedback_local": dict(base, feedback=False),
        "hierarchy_flat_local": dict(base, topology="flat_control"),
        "hierarchy_tree_common_skip": dict(base, readout="common_skip"),
        "hierarchy_flat_common_skip": dict(base, topology="flat_control", readout="common_skip"),
    }


def test_execution_sets_and_records_the_blas_and_cublas_environment(monkeypatch):
    module = api()
    for name in (
        "CUBLAS_WORKSPACE_CONFIG",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "wrong")
    observed = module["configure_environment"]()
    assert observed == dict(
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    assert all(os.environ[name] == value for name, value in observed.items())


def test_closure_records_guard_and_hold_errors_without_hiding_the_primary_failure():
    module = api()

    def broken():
        raise RuntimeError("fallo de cierre")

    report = dict(status="failed", error="fallo numérico", hold_before="expected")
    module["finalize_report"](
        report,
        SimpleNamespace(remove=broken),
        SimpleNamespace(check_hold=broken, HOLD_SHA256="expected"),
        SimpleNamespace(now=lambda: "2026-10-22T00:00:00Z"),
        time.monotonic() - 2,
        1,
    )
    assert report["status"] == "failed" and report["error"] == "fallo numérico"
    assert "fallo de cierre" in report["cleanup_error"]
    assert "fallo de cierre" in report["hold_error"]
    assert report["budget_exceeded"] is True
    assert report["hold_unchanged"] is False


@pytest.mark.parametrize("bad", [True, -1, 2, 0.0])
def test_invalid_execution_order_is_rejected(bad):
    module = api()
    with pytest.raises(ValueError):
        module["execution_order"](module["protocol"]()["cells"], bad)


def test_shared_parameters_require_the_same_names_and_shapes():
    module = api()
    records = [
        {"parameter_shapes": {"w": [2, 3], "buffer/regional_execution.geometry": [8, 8]}},
        {"parameter_shapes": {"w": [2, 3]}},
    ]
    before = copy.deepcopy(records)
    module["verify_parameter_shapes"](records)
    assert records == before
    for mutation in ({"v": [2, 3]}, {"w": [3, 2]}):
        with pytest.raises(AssertionError):
            module["verify_parameter_shapes"]([records[0], {"parameter_shapes": mutation}])


def test_default_cli_neither_imports_torch_nor_creates_a_report(tmp_path):
    assert SCRIPT.is_file(), "Falta el benchmark de la jerarquía regional"
    source = """
import importlib.abc
import runpy
import sys
class RejectTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise RuntimeError('El protocolo no debe importar Torch')
sys.meta_path.insert(0, RejectTorch())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    output = tmp_path / "unused-output"
    process = subprocess.run(
        [sys.executable, "-I", "-B", "-c", source, str(SCRIPT), "--output", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["condition_records"] == 128
    assert not output.exists()
