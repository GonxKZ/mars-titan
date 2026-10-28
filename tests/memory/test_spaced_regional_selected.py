"""Carga explícita de pesos regionales de fixture, sin entrenamiento ni traslado de estados."""

import copy
from dataclasses import asdict

import pytest
import torch
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_spaced_regional_readout import options
from test_spaced_selected import load

from tests.memory.spaced_selected_fixture import (
    digest,
    replace_selected_reader,
    selected_fixture,
    write,
)


def components(kind="ring"):
    bridge = dict(
        regions=["view_1", "view_2", "view_3"],
        topology="ring",
        cycles=1,
        bridge_gain=0.25,
        seed=42,
    ) | options(kind)
    if "recurrence" in bridge:
        bridge["recurrence"] = bridge["recurrence"].options()
    for name in ("intervention", "hierarchy"):
        if name in bridge:
            bridge[name] = asdict(bridge[name])
    return dict(episodic_bank="m1", evidence_bridge=bridge)


@pytest.mark.parametrize("kind", ["ring", "j4", "adaptive_compact", "tree", "masked"])
@pytest.mark.parametrize("quantile", [False, True])
def test_selected_regional_reader_preserves_all_weights_and_requires_explicit_contract(
    tmp_path, four_flow_source, kind, quantile
):
    fixture = selected_fixture(
        tmp_path, dict(four_flow_source, components=components(kind)), quantiles=quantile
    )
    with pytest.raises(ValueError):
        load(fixture)
    before = torch.get_rng_state().clone()
    loaded = load(fixture, reader_contract="regional")
    assert loaded.consumer.readout.config.evidence_bridge is not None
    for module, expected in (
        (loaded.consumer.predictor, fixture["core"]),
        (loaded.consumer.readout, fixture["reader"]),
    ):
        for name, value in module.state_dict().items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, expected[name])
            else:
                assert value == expected[name]
        assert all(not p.requires_grad for p in module.parameters())
    assert torch.equal(before, torch.get_rng_state())
    assert loaded.identity()["reader_contract"] == "regional"


def test_regional_loader_rejects_standard_state_before_gpu_or_parent_loading(
    tmp_path, four_flow_source, monkeypatch
):
    from mars_titan.training import spaced_selected

    fixture = selected_fixture(tmp_path, four_flow_source)
    monkeypatch.setattr(
        spaced_selected, "_cuda", lambda *_: pytest.fail("Reservó GPU para arquitectura equivocada")
    )
    with pytest.raises(ValueError):
        load(fixture, reader_contract="regional")


@pytest.mark.parametrize("value", [None, True, "auto", []])
def test_invalid_contract_fails_before_opening_an_anchor(tmp_path, four_flow_source, value):
    from mars_titan.training.spaced_selected import load_selected_reader

    with pytest.raises(ValueError):
        load_selected_reader(
            tmp_path / "missing",
            tmp_path / "view",
            four_flow_source["spec"],
            device="cpu",
            reader_contract=value,
        )


def test_json_normalization_does_not_allow_a_different_variant(tmp_path, four_flow_source):
    fixture = selected_fixture(tmp_path, dict(four_flow_source, components=components()))
    report = copy.deepcopy(fixture["report"])
    report["request"] = copy.deepcopy(report["request"])
    report["identity"]["request"] = copy.deepcopy(report["request"])
    report["identity"]["variant"] = copy.deepcopy(report["identity"]["variant"])
    report["identity"]["variant"]["components"]["evidence_bridge"]["topology"] = "complete"
    report["run_id"] = digest(report["identity"])
    write(fixture["anchor"] / "run.json", report)
    with pytest.raises(ValueError, match="variante"):
        load(fixture, reader_contract="regional")


def test_changed_regional_payload_is_rejected_before_copying_any_reader_parameter(
    tmp_path, four_flow_source, monkeypatch
):
    from mars_titan.training import spaced_selected

    fixture = selected_fixture(tmp_path, dict(four_flow_source, components=components("tree")))
    state = copy.deepcopy(fixture["reader"])
    state["_extra_state"]["configuration"]["evidence_bridge"]["regions"] = ["forged"]
    replace_selected_reader(
        fixture, dict(model=state, readout_sha256=fixture["selected"]["readout"]["readout_sha256"])
    )
    monkeypatch.setattr(
        spaced_selected,
        "_carried_readout",
        lambda *_: pytest.fail("Copió parámetros con un contrato falso"),
    )
    with pytest.raises(ValueError):
        load(fixture, reader_contract="regional")
