"""La medida de coste de E1 también recorre la declaración de R2 (#786) sin optimizador."""

import json
import sys

import pytest

from benchmarks import multiscale_mixture_cost


def test_the_r2_declaration_closes_a_week_in_each_measured_cycle(monkeypatch, tmp_path):
    output = tmp_path / "cost.json"
    arguments = ["--output", str(output), "--warmup", "0", "--repeats", "1"]
    declaration = ["--declaration", "configs/experiments/mature-cms-v1.json"]
    monkeypatch.setattr(sys, "argv", ["multiscale_mixture_cost", *arguments, *declaration])
    multiscale_mixture_cost.main()
    receipt = json.loads(output.read_text())
    assert receipt["declaration"] == "configs/experiments/mature-cms-v1.json"
    cases = receipt["cases"]
    assert set(cases) == {
        f"{arm}/{rows}" for arm in ("mature_cms", "same_frequency") for rows in (64, 256, 1024)
    }
    for rows in (64, 256, 1024):
        # El estado medido es el del día 0: la semana y el mes esperan esa primera cohorte.
        assert cases[f"mature_cms/{rows}"]["open_period_rows"] == [0, rows, rows]
        assert cases[f"same_frequency/{rows}"]["open_period_rows"] == [0, 0, 0]
        extra = cases[f"mature_cms/{rows}"]["numeric_state_bytes"]
        assert extra - cases[f"same_frequency/{rows}"]["numeric_state_bytes"] == 2 * rows * 69 * 8
    assert all(case["recovery_bitwise"] for case in cases.values())
    with pytest.raises(FileExistsError):
        multiscale_mixture_cost.main()
