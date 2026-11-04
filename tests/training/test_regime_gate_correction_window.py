"""Ventanas E4 sobre pesos literales de fixture, sin entrenador ni pasos de optimizador."""

import json

import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.memory.regime_evidence_gate import RegimeGateConfig
from mars_titan.memory.salience_budget import SalienceBudgetConfig
from mars_titan.training import mars_titan_correction as mc
from tests.memory.spaced_selected_fixture import selected_fixture  # noqa: F401
from tests.memory.spaced_window_fixture import study_fixture

GATE = RegimeGateConfig(threshold=1.0, drift=0.0, floor_gate=0.25, min_history=1)
PARTS = ("validation", "calibration", "evaluation")


@pytest.fixture(scope="module", autouse=True)
def frozen_backend():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(autouse=True)
def only_manual_frozen_parameters(learning_doubles):
    """El padre lo guarda study_fixture con pesos literales y sin entrenador."""


@pytest.fixture(scope="module")
def window(tmp_path_factory):
    return study_fixture(tmp_path_factory.mktemp("regime-gate-window"), assets=4, rich=True)


def run(window, output, **factors):
    return mc.run_correction_window(
        window["view"],
        window["anchor"].parent / "parent",
        "configs/titans/mature-correction-historical-masked.json",
        components={"associative_memory": {"rule": "proximal", "key": "codec"}},
        seed=42,
        output=output,
        search_case="eta25e-2",
        device="cpu",
        indices=window["indices"],
        **factors,
    )


def table(output, part):
    return pq.read_table(output / f"{part}-predictions.parquet")


def test_window_persists_the_gate_and_preserves_all_rows_and_parent(window, tmp_path):
    baseline = run(window, tmp_path / "baseline")
    active = run(window, tmp_path / "active", regime_gate=GATE)
    assert "regime_gate" not in baseline["request"] and "regime_gate" not in baseline["identity"]
    assert active["request"]["regime_gate"] == active["identity"]["regime_gate"]
    assert active["identity"]["regime_gate"] == GATE.identity()
    assert active["identity"]["variant"]["kind"] == "regime_gate_correction_variant_v1"
    assert baseline["identity"]["variant_sha256"] != active["identity"]["variant_sha256"]
    assert baseline["identity"]["parent"] == active["identity"]["parent"]
    selected = json.loads((tmp_path / "active/selected.json").read_text())
    assert selected["regime_gate"] == GATE.identity()
    for part in PARTS:
        before, after = table(tmp_path / "baseline", part), table(tmp_path / "active", part)
        for name in ("sample_id", "prediction_at", "target"):
            assert before[name].equals(after[name])
        gate = active["predictions"][part]["metrics"]["regime_gate"]
        assert gate["publications"] > 0 and 0.25 <= gate["mean_gate"] <= 1.0


def test_a_constant_gate_of_one_reproduces_the_plain_window(window, tmp_path):
    run(window, tmp_path / "plain")
    constant = RegimeGateConfig(mode="constant", constant=1.0)
    run(window, tmp_path / "one", regime_gate=constant)
    for part in PARTS:
        assert table(tmp_path / "plain", part).equals(table(tmp_path / "one", part))


def test_the_matched_control_uses_the_validation_mean_of_the_variant(window, tmp_path):
    from mars_titan.training.regime_gate_inference import matched_control

    report = run(window, tmp_path / "variant", regime_gate=GATE)
    control = matched_control(GATE, report)
    expected = report["predictions"]["validation"]["metrics"]["regime_gate"]["mean_gate"]
    assert control.mode == "constant" and control.constant == expected
    assert control.threshold == GATE.threshold and control.drift == GATE.drift
    matched = run(window, tmp_path / "control", regime_gate=control)
    gates = [matched["predictions"][part]["metrics"]["regime_gate"] for part in PARTS]
    assert all(gate["mean_gate"] == pytest.approx(expected, abs=1e-15) for gate in gates)
    with pytest.raises(ValueError):
        matched_control(control, report)
    with pytest.raises(ValueError):
        matched_control(RegimeGateConfig(drift=0.25), report)


def test_regeneration_keeps_the_chosen_gate_and_exact_predictions(window, tmp_path):
    run(window, tmp_path / "anchor", regime_gate=GATE)
    report = mc.carry_correction(
        tmp_path / "anchor",
        window["view"],
        window["view"],
        tmp_path / "regenerated",
        device="cpu",
        regenerate=True,
    )
    assert report["regime_gate"] == GATE.identity()
    for part in PARTS:
        assert table(tmp_path / "anchor", part).equals(table(tmp_path / "regenerated", part))


def test_changed_gate_and_contradictory_anchor_are_rejected(window, tmp_path, monkeypatch):
    run(window, tmp_path / "anchor", regime_gate=GATE)
    with pytest.raises(ValueError, match="petición"):
        run(window, tmp_path / "anchor", regime_gate=RegimeGateConfig(drift=0.25))
    path = tmp_path / "anchor/run.json"
    report = json.loads(path.read_text())
    report["identity"]["regime_gate"]["drift"] = 0.25
    path.write_text(json.dumps(report))
    monkeypatch.setattr(
        mc,
        "_sources",
        lambda *_a, **_k: pytest.fail("Se abrieron datos con una identidad E4 contradictoria"),
    )
    with pytest.raises(ValueError, match="E4"):
        mc.carry_correction(
            tmp_path / "anchor", window["view"], window["view"], tmp_path / "bad", device="cpu"
        )


@pytest.mark.parametrize(
    "factors",
    [
        dict(regime_gate={}),
        dict(regime_gate=GATE, salience=SalienceBudgetConfig()),
    ],
)
def test_invalid_or_combined_factors_fail_before_the_parent(window, tmp_path, monkeypatch, factors):
    from mars_titan.training import mars_titan_walk_forward

    monkeypatch.setattr(
        mars_titan_walk_forward,
        "_parent",
        lambda *_a, **_k: pytest.fail("Se abrió el padre con factores de escritura inválidos"),
    )
    with pytest.raises(ValueError, match="E4|E3"):
        run(window, tmp_path / "invalid", **factors)
    assert not (tmp_path / "invalid").exists()
