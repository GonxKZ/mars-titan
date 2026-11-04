"""La medida E3 conserva cupos y estado, sin acceso a datos científicos."""

import json

import pytest

from benchmarks.salience_budget_cost import cost_case


@pytest.mark.parametrize("rule", ("standardized_error", "absolute_error", "uniform"))
def test_cost_cycle_preserves_selection_and_exact_json_recovery(rule):
    result = cost_case(8, rule, repeats=1, warmup=0)
    assert result["selected"] == 8 and result["exact_recovery"]
    assert result["median_seconds"] > 0 and result["serialized_state_bytes"] > 0
    assert len(result["samples_seconds"]) == 1
    assert result["state_sha256"] == cost_case(8, rule, repeats=1, warmup=0)["state_sha256"]


@pytest.mark.parametrize(
    "options",
    [
        {"rows": 0},
        {"rows": 8193},
        {"rows": True},
        {"repeats": 0},
        {"warmup": 6},
    ],
)
def test_cost_cycle_rejects_unbounded_or_empty_work(options):
    with pytest.raises(ValueError):
        cost_case(**dict(rows=8, rule="uniform") | options)


def test_cost_command_reports_all_controls_and_its_scope(monkeypatch, capsys):
    from benchmarks.salience_budget_cost import main

    monkeypatch.setattr("sys.argv", ["salience_budget_cost", "--sizes", "8", "--repeats", "1"])
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["scientific_data"] is False and report["device"] == "cpu"
    assert {case["rule"] for case in report["cases"]} == {
        "standardized_error",
        "absolute_error",
        "uniform",
    }
    assert {case["selected"] for case in report["cases"]} == {8}
    assert report["source_sha256"] and report["process_peak_rss_kib"] > 0


def test_cost_command_rejects_duplicate_sizes(monkeypatch, capsys):
    from benchmarks.salience_budget_cost import main

    monkeypatch.setattr("sys.argv", ["salience_budget_cost", "--sizes", "8", "8"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "tamaños distintos" in capsys.readouterr().err
