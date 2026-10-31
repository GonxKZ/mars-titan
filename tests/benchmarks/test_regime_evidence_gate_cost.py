"""La medida E4 repite la misma escritura y estado, sin acceso a datos científicos."""

import json

import pytest

from benchmarks.regime_evidence_gate_cost import MODES, cost_case


def test_cost_cycles_repeat_the_write_and_the_gate_changes_it():
    cases = {mode: cost_case(8, mode, repeats=1, warmup=0) for mode in MODES}
    assert all(case["exact_repetition"] and case["median_seconds"] > 0 for case in cases.values())
    assert cases["none"]["event_gates"] is None and cases["none"]["serialized_state_bytes"] == 0
    assert cases["constant"]["event_gates"] == {"CN": 0.5, "US": 0.5}
    assert all(0.25 < gate < 1.0 for gate in cases["cusum"]["event_gates"].values())
    assert len({case["memory_sha256"] for case in cases.values()}) == 3
    assert (
        cases["cusum"]["memory_sha256"]
        == cost_case(8, "cusum", repeats=1, warmup=0)["memory_sha256"]
    )


@pytest.mark.parametrize(
    "options",
    [{"rows": 0}, {"rows": 8193}, {"rows": True}, {"mode": "other"}, {"repeats": 0}, {"warmup": 6}],
)
def test_cost_cycle_rejects_unbounded_or_empty_work(options):
    with pytest.raises(ValueError):
        cost_case(**dict(rows=8, mode="cusum") | options)


def test_cost_command_reports_all_modes_and_its_scope(monkeypatch, capsys):
    from benchmarks.regime_evidence_gate_cost import main

    monkeypatch.setattr("sys.argv", ["regime_evidence_gate_cost", "--sizes", "8", "--repeats", "1"])
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["scientific_data"] is False and report["device"] == "cpu"
    assert [case["mode"] for case in report["cases"]] == list(MODES)
    assert len(report["source_sha256"]) == 2 and report["process_peak_rss_kib"] > 0


def test_cost_command_rejects_duplicate_sizes(monkeypatch, capsys):
    from benchmarks.regime_evidence_gate_cost import main

    monkeypatch.setattr("sys.argv", ["regime_evidence_gate_cost", "--sizes", "8", "8"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "tamaños distintos" in capsys.readouterr().err
