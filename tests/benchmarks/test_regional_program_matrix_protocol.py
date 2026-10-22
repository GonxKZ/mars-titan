"""Contrastar el protocolo finito sin medir modelos ni abrir el test reservado."""

import itertools
import json
import runpy
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from test_financial_adapter import specification

from mars_titan.models.titans.financial import FinancialConfig
from mars_titan.training.titans_walk_forward import predictor_options

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/regional_program_matrix_cuda.py"
DECLARATION = ROOT / "configs/experiments/regional-program-v1.json"
VARIANTS = ("transformer_direct", "mac_disabled", "mac_frozen", "mac_online")
CONTEXTS = (None, "market_memory_v1")
MIB = 1024**2
CLI_RUNNER = """
import importlib.abc
import runpy
import sys

class RejectTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise RuntimeError('PROTOCOL_IMPORTED_TORCH')
        return None

assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
sys.meta_path.insert(0, RejectTorch())
sys.argv = sys.argv[1:]
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as error:
    if error.code not in (None, 0):
        raise
assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
"""


def matrix_api():
    assert SCRIPT.is_file(), "Falta el guion del protocolo de la matriz regional"
    return runpy.run_path(str(SCRIPT), run_name="regional_matrix_protocol_test")


def constructor_results(cells):
    document = json.loads(DECLARATION.read_text())
    recipe = json.loads((ROOT / document["base_recipe"]).read_text())
    result = {}
    for cell in cells:
        common = dict(recipe, predictor=recipe["predictor"] | {"session_context": cell["context"]})
        changes = document["arms"][cell["arm"]]["predictor_changes"]
        options = predictor_options(common, {"predictor_changes": changes})
        try:
            FinancialConfig(specification(), variant=cell["variant"], **options)
        except ValueError as error:
            message = str(error)
            if "session_context" in message and "mac_online" in message:
                rejection = "context_requires_mac_online"
            elif "padre PT4" in message:
                rejection = "contextual_program_requires_context"
            else:
                pytest.fail(f"Rechazo no previsto de {cell['cell_id']}: {message}")
            result[cell["cell_id"]] = False, rejection
        else:
            result[cell["cell_id"]] = True, None
    return result


def run_cli(tmp_path, arguments):
    assert SCRIPT.is_file(), "Falta el guion del protocolo de la matriz regional"
    return subprocess.run(
        [sys.executable, "-I", "-B", "-c", CLI_RUNNER, str(SCRIPT), *arguments],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


@pytest.mark.parametrize("with_output_argument", [False, True])
def test_default_cli_prints_protocol_without_torch_or_creating_output(
    tmp_path, with_output_argument
):
    destination = tmp_path / "unused-output"
    arguments = []
    if with_output_argument:
        arguments += ["--output", str(destination)]
    process = run_cli(tmp_path, arguments)
    assert process.returncode == 0, process.stderr
    protocol = json.loads(process.stdout)
    assert protocol["planned_cells"] == 48
    assert protocol["optimizer_steps"] == 0
    assert protocol["scientific_evaluation"] is False
    assert protocol["final_test_access"] is False
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_all_48_cells_and_their_rejections_match_the_real_configuration_api():
    protocol = matrix_api()["protocol"](ROOT)
    cells = protocol["cells"]
    arms = tuple(json.loads(DECLARATION.read_text())["arms"])
    expected = list(itertools.product(VARIANTS, CONTEXTS, arms))
    actual = [(cell["variant"], cell["context"], cell["arm"]) for cell in cells]
    assert actual == expected
    assert len({cell["cell_id"] for cell in cells}) == len(expected) == 48
    admitted = constructor_results(cells)
    for cell in cells:
        context = cell["context"] or "none"
        assert cell["cell_id"] == f"{cell['variant']}/{context}/{cell['arm']}"
        assert cell["baseline_cell_id"] == f"{cell['variant']}/{context}/legacy_post_fusion"
        assert type(cell["expected_admissible"]) is bool
        assert (cell["expected_admissible"], cell["expected_rejection"]) == admitted[
            cell["cell_id"]
        ]
    assert sum(value[0] for value in admitted.values()) == protocol["admissible_cells"] == 26
    assert sum(not value[0] for value in admitted.values()) == protocol["rejected_cells"] == 22


def test_second_pass_reverses_all_26_admitted_cells_without_loss_or_duplicates():
    api = matrix_api()
    cells = api["protocol"](ROOT)["cells"]
    admitted = constructor_results(cells)
    expected = [cell["cell_id"] for cell in cells if admitted[cell["cell_id"]][0]]
    forward = api["execution_order"](cells, 0)
    inverse = api["execution_order"](cells, 1)
    assert forward == expected
    assert inverse == list(reversed(expected))
    assert len(forward) == len(inverse) == len(set(forward)) == len(set(inverse)) == 26
    assert Counter(forward) == Counter(inverse) == Counter(expected)
    for invalid in (-1, 2):
        with pytest.raises(ValueError):
            api["execution_order"](cells, invalid)


def test_work_totals_follow_admitted_cells_and_keep_the_declared_resource_limits():
    api = matrix_api()
    protocol = api["protocol"](ROOT)
    admitted = sum(value[0] for value in constructor_results(protocol["cells"]).values())
    assert protocol["dtypes"] == ["float32", "float64"]
    assert protocol["batches"] == [1, 256]
    assert protocol["warmup_per_record"] == 2
    assert protocol["repetitions_per_record"] == 5
    records = admitted * len(protocol["dtypes"]) * len(protocol["batches"]) * 2
    prefixes = records * 2
    timed = records * (protocol["warmup_per_record"] + protocol["repetitions_per_record"])
    assert records == protocol["condition_records"] == 208
    assert prefixes == protocol["prefix_prepare_calls"] == 416
    assert timed == protocol["timed_prepare_calls"] == 1456
    assert prefixes + timed == protocol["total_prepare_calls"] == 1872
    assert protocol["no_summary_pairs"] == 40
    assert protocol["child_budget_seconds"] == 240
    assert protocol["external_job_budget_seconds"] == 300
    assert protocol["external_phase_budget_seconds"] == 1500
    assert protocol["allocator_cap_bytes"] == 512 * MIB
    assert protocol["free_vram_headroom_bytes"] == 256 * MIB
    assert protocol["cpu_proof_tensor_cap_bytes"] == 128 * MIB
    assert protocol["optimizer_steps"] == 0
    assert protocol["scientific_evaluation"] is False
    assert protocol["final_test_access"] is False
    fields = (
        "dtypes",
        "batches",
        "condition_records",
        "prefix_prepare_calls",
        "timed_prepare_calls",
        "total_prepare_calls",
        "no_summary_pairs",
    )
    assert api["execution_scope"](protocol) == {name: protocol[name] for name in fields}
    for dtype, batch in (
        ("float32", None),
        (None, 1),
        ("float16", 1),
        ("float32", 128),
        ("float32", True),
    ):
        with pytest.raises(ValueError):
            api["execution_scope"](protocol, dtype=dtype, batch=batch)


@pytest.mark.parametrize("dtype,batch", list(itertools.product(("float32", "float64"), (1, 256))))
def test_each_stratum_covers_both_global_orders_and_its_complete_call_budget(dtype, batch):
    api = matrix_api()
    protocol = api["protocol"](ROOT)
    admitted = constructor_results(protocol["cells"])
    expected = [cell["cell_id"] for cell in protocol["cells"] if admitted[cell["cell_id"]][0]]
    scope = api["execution_scope"](protocol, dtype=dtype, batch=batch)
    assert scope["dtypes"] == [dtype] and scope["batches"] == [batch]
    records = len(expected) * 2
    assert records == scope["condition_records"] == 52
    assert 2 * records == scope["prefix_prepare_calls"] == 104
    assert (
        records * (protocol["warmup_per_record"] + protocol["repetitions_per_record"])
        == scope["timed_prepare_calls"]
        == 364
    )
    assert (
        scope["prefix_prepare_calls"] + scope["timed_prepare_calls"]
        == scope["total_prepare_calls"]
        == 468
    )
    assert scope["no_summary_pairs"] == 10
    assert api["execution_order"](protocol["cells"], 0) == expected
    assert api["execution_order"](protocol["cells"], 1) == list(reversed(expected))
    assert 4 * scope["condition_records"] == protocol["condition_records"]
    assert 4 * scope["prefix_prepare_calls"] == protocol["prefix_prepare_calls"]
    assert 4 * scope["timed_prepare_calls"] == protocol["timed_prepare_calls"]
    assert 4 * scope["total_prepare_calls"] == protocol["total_prepare_calls"]
    assert 4 * scope["no_summary_pairs"] == protocol["no_summary_pairs"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--dtype", "float32"],
        ["--batch", "1"],
        ["--dtype", "float16", "--batch", "1"],
        ["--dtype", "float32", "--batch", "128"],
    ],
)
def test_invalid_cli_stratum_is_rejected_before_torch_or_output(tmp_path, arguments):
    destination = tmp_path / "unused-output"
    process = run_cli(tmp_path, ["--run", "--output", str(destination), *arguments])
    assert process.returncode == 2, process.stderr
    assert "usage:" in process.stderr
    assert "PROTOCOL_IMPORTED_TORCH" not in process.stderr
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []
