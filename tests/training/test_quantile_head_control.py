"""Control declarado de la cabeza: Transformer compacto con salida escalar L1 o cuantiles.

Solo se comprueba la declaración. No se leen datos ni se ejecuta ningún ajuste.
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mars_titan.data import input_policy
from mars_titan.models import quantile_head
from mars_titan.training import reference_design
from mars_titan.training.reference_design import head_control_cases

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs/baselines/quantile-head-control-us.json"
SEARCH = ROOT / "configs/baselines/historical-masked-reference-search-us.json"


def plan():
    return json.loads(PLAN.read_text())


def test_declared_pairs_differ_only_in_output_and_loss():
    cases = head_control_cases(plan())
    assert len(cases) == 3 * 2 * 2
    assert len({case["id"] for case in cases}) == len(cases)
    pairs = {}
    for item in cases:
        pairs.setdefault(item["pair"], {})[item["arm"]] = item["case"]
    assert len(pairs) == 6
    for pair in pairs.values():
        scalar, quantile = pair["scalar_l1"], pair["quantile_head_v1"]
        assert scalar["kind"] == quantile["kind"] == "transformer"
        assert scalar["loss"] == "mae" and "head" not in scalar
        assert quantile["loss"] == "pinball" and quantile["head"] == "quantile_head_v1"
        shared = {k: v for k, v in scalar.items() if k != "loss"}
        assert {k: v for k, v in quantile.items() if k not in {"loss", "head"}} == shared
        assert scalar["selection"]["stopping"] == "fixed_budget"
        assert scalar["selection"]["metric"] == "session_mae"
    assert {case["case"]["seed"] for case in cases} == {42, 43, 44}


def test_design_matches_the_reference_search_indices_seeds_and_budget():
    declared, search = plan(), json.loads(SEARCH.read_text())
    assert declared["case_indices"] == search["case_indices"]
    assert declared["seeds"] == search["finalist_seeds"]
    for key in ("patience", "min_delta", "stopping", "batch_size", "input_policy"):
        assert declared[key] == search[key]
    assert declared["max_epochs"] == search["max_epochs"]
    assert declared["prediction_retention"] == search["prediction_retention"]
    assert (ROOT / declared["walk_forward"]).is_file()
    base = reference_design.design_cases(
        ["transformer"],
        seed=42,
        epochs=search["max_epochs"],
        patience=search["patience"],
        min_delta=search["min_delta"],
        stopping=search["stopping"],
    )
    first = next(c for c in head_control_cases(declared) if c["arm"] == "scalar_l1")
    assert first["pair"] == base[declared["case_indices"][0]]["id"]
    assert {k: v for k, v in first["case"].items() if k != "loss"} == {
        k: v for k, v in base[0]["case"].items() if k != "loss"
    }


def test_each_declared_case_passes_the_runner_validation(monkeypatch):
    from mars_titan.training import reference_run

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    declared = plan()
    for item in head_control_cases(declared):
        reference_run._options(
            item["case"],
            declared["batch_size"],
            300,
            0,
            input_policy=declared["input_policy"],
            prediction_retention=declared["prediction_retention"],
        )


def test_names_repeat_the_head_and_policy_constants():
    assert reference_design.QUANTILE_HEAD == quantile_head.QUANTILE_HEAD
    assert reference_design.PINBALL == quantile_head.PINBALL
    assert reference_design.HISTORICAL_MASKED == input_policy.HISTORICAL_MASKED


def test_design_module_still_loads_without_pytorch():
    code = (
        "import sys, mars_titan.training.reference_design as d; "
        "assert 'torch' not in sys.modules; print(len(d.HEAD_CONTROL_ARMS))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "2"


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(status="executed"),
        lambda p: p.update(final_test_opened=True),
        lambda p: p.update(model="gru"),
        lambda p: p.update(arms=["quantile_head_v1", "scalar_l1"]),
        lambda p: p.update(arms=["scalar_mse", "quantile_head_v1"]),
        lambda p: p.update(stopping="validation_plateau"),
        lambda p: p.update(input_policy="strict_inputs_v1"),
        lambda p: p.update(case_indices=[10, 0]),
        lambda p: p.update(case_indices=[0, 0]),
        lambda p: p.update(case_indices=[12]),
        lambda p: p.update(case_indices=[]),
        lambda p: p.update(seeds=[42, 42]),
        lambda p: p.update(seeds=[]),
        lambda p: p.update(batch_size=512),
        lambda p: p["comparison"].update(partition="evaluation"),
        lambda p: p["comparison"].update(metric="session_mse"),
        lambda p: p["comparison"].update(fallback="keep_quantile_head"),
        lambda p: p["comparison"]["contrast"].update(base="quantile_head_v1"),
        lambda p: p.update(extra=True),
        lambda p: p.pop("decision"),
        lambda p: p.update(walk_forward=None),
        lambda p: p.update(patience=0),
        lambda p: p.update(max_epochs=1),
    ],
)
def test_any_change_to_the_declared_contrast_is_rejected(change):
    declared = copy.deepcopy(plan())
    change(declared)
    with pytest.raises(ValueError):
        head_control_cases(declared)
