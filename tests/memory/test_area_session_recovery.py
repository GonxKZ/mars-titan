"""Paridad entre ventana y sesión con recuperación de los estados por áreas y por episodios."""

import pytest
import torch
from test_area_correction_window import AREAS, BRIDGED, KERNEL, SHUFFLED
from test_financial_session_associative import four_flow_source as four_flow_source
from test_financial_session_associative import frozen_consumer as frozen_consumer
from test_financial_session_associative import module_backend as module_backend
from test_financial_session_associative import no_target_estimation as no_target_estimation
from test_financial_session_associative import options, trajectory
from test_financial_session_associative import shared_native as shared_native
from test_mars_titan_correction_parity import corrected_pass, emitted

from mars_titan.memory import area_memory, area_query_bridge, kernel_memory
from mars_titan.memory.financial_session import FinancialSession


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize(
    "correction", [AREAS, SHUFFLED, KERNEL, BRIDGED], ids=["areas", "shuffled", "kernel", "bridge"]
)
def test_session_recovers_the_same_emissions_as_the_window(
    shared_native, four_flow_source, frozen_consumer, tmp_path, correction
):
    direct = trajectory(
        shared_native, four_flow_source, frozen_consumer, tmp_path / "direct", correction=correction
    )
    recovered = trajectory(
        shared_native,
        four_flow_source,
        frozen_consumer,
        tmp_path / "recovered",
        correction=correction,
        recover=True,
    )
    expected = [
        (asset, at, value) for e in direct["events"] for _, asset, at, value in e["predictions"]
    ]
    actual = [
        (asset, at, value) for e in recovered["events"] for _, asset, at, value in e["predictions"]
    ]
    assert actual == expected and len(actual) == 20
    window, _ = corrected_pass(four_flow_source, correction)
    assert emitted(window) == expected
    assert direct["snapshot"] == recovered["snapshot"]
    state = correction.restore(direct["events"][-1]["memory"])
    torch.testing.assert_close(state.matrix, window.memory.matrix, rtol=0, atol=0)


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize(
    "module,correction",
    [(area_memory, AREAS), (kernel_memory, KERNEL), (area_query_bridge, BRIDGED)],
    ids=["areas", "kernel", "bridge"],
)
def test_session_identity_pins_the_source_of_the_enabled_memory(
    shared_native, four_flow_source, frozen_consumer, tmp_path, monkeypatch, module, correction
):
    settings = options(shared_native, four_flow_source, frozen_consumer, associative=correction)
    with FinancialSession(tmp_path / "before", **settings) as run:
        original = run.model_id
    changed_source = tmp_path / "changed-source.py"
    changed_source.write_text("# Una revisión distinta del módulo habilitado.\n")
    monkeypatch.setattr(module, "__file__", str(changed_source))
    with FinancialSession(tmp_path / "after", **settings) as run:
        assert run.model_id != original
