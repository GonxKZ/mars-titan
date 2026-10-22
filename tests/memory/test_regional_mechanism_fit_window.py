"""Fuentes regionales identificadas, con resultados maduros y sin ejecutar el ajuste."""

from dataclasses import replace

import pytest
from test_bridge_mechanism_session import engines
from test_financial_mechanisms import assert_same
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_mechanism_fit_window import config
from test_native_episode_backend import native as native
from test_native_mechanism_collection import session

from mars_titan.models.titans.mechanism_catalog import MechanismBudget, MechanismCatalog
from mars_titan.training.mechanism_fit_window import MechanismFitWindow, audit_fit_windows

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


def test_fit_contract_requires_explicit_regional_catalogue_and_preserves_legacy_identity():
    catalog = MechanismCatalog(reader_contract="shared_evidence_bridge_v1")
    protocol = config(catalog_id=catalog.fingerprint())
    protocol.check_catalog(catalog)
    assert protocol.identity()["catalog_id"] == catalog.fingerprint()
    assert "catalog_id" not in config().identity()
    config().check_catalog(MechanismCatalog())
    config().check_catalog(MechanismCatalog(MechanismBudget(max_reader_steps=2)))
    for candidate in (
        MechanismCatalog(),
        replace(catalog, budget=MechanismBudget(max_reader_steps=2)),
    ):
        with pytest.raises(ValueError, match="catálogo"):
            protocol.check_catalog(candidate)
    with pytest.raises(ValueError, match="catálogo"):
        config().check_catalog(catalog)
    with pytest.raises(ValueError, match="catálogo"):
        protocol.check_catalog(catalog.identity())
    for value in (False, "invalid", 3):
        with pytest.raises(ValueError):
            config(catalog_id=value)


@pytest.mark.parametrize("device", DEVICES)
def test_regional_native_collections_preserve_emissions_mature_prefix_and_recovery(
    native, four_flow_source, tmp_path, device
):
    data = engines(four_flow_source, device)
    path = tmp_path / "regional"
    with (
        session(native, path, data) as run,
        session(native, tmp_path / "off", data, enabled=False) as control,
    ):
        for target in (run, control):
            target.step(four_flow_source["batches"][125], [], kind="warmup")
        first = run.step(four_flow_source["batches"][126], [])
        fixed = control.step(four_flow_source["batches"][126], [])
        assert [(p.asset, p.value) for p in first.predictions] == [
            (p.asset, p.value) for p in fixed.predictions
        ]
        assert run.mechanism_collection(2)["supervision"]["outcomes"] == [None] * 4
        for target, issued in ((run, first), (control, fixed)):
            labels = [native.Feedback(p.id, 0, moment(127), 0.125) for p in issued.predictions]
            target.step([], labels, kind="settlement", cutoff=moment(127))
        later = run.step(four_flow_source["batches"][128], [])
        expected = control.step(four_flow_source["batches"][128], [])
        assert [(p.asset, p.value) for p in later.predictions] == [
            (p.asset, p.value) for p in expected.predictions
        ]
        flows = tuple(sorted(run.fast_state()["references"]))
        assert_same(
            data[3].predictor.export_state(run._fast_store.gather(run.fast_state(), flows)),
            data[3].predictor.export_state(control._fast_store.gather(control.fast_state(), flows)),
        )
        assert run.diagnostics() == control.diagnostics()
        protocol = config(
            data[3].model_id, catalog_id=data[3].catalog.fingerprint(), fit_end=moment(129)
        )
        request = MechanismFitWindow(run, (2, 4), through_generation=4)
        audit = audit_fit_windows((request,), config=protocol)
        assert audit["observed_cohorts"] == 2
        assert audit["mature_rows"] == audit["pending_rows"] == 4
        assert audit["excluded_rows"] == audit["censored_rows"] == 0
        current = run.snapshot()
        labels = [native.Feedback(p.id, 0, moment(129), 99.0) for p in later.predictions]
        run.step([], labels, kind="settlement", cutoff=moment(129))
        run.step([], [], kind="settlement", cutoff=moment(131), close_phase=True)
        assert audit_fit_windows((request,), config=protocol) == audit
        assert current["cursor"] == 4
    with session(native, path, data, resume=True) as restored:
        assert (
            audit_fit_windows(
                (MechanismFitWindow(restored, (2, 4), through_generation=4),), config=protocol
            )
            == audit
        )


@pytest.mark.parametrize("binding", [None, "a" * 64])
def test_unbound_or_wrong_regional_fit_rejects_before_opening_window(
    native, four_flow_source, tmp_path, monkeypatch, binding
):
    import mars_titan.training.mechanism_fit_window as module

    data = engines(four_flow_source)
    with session(native, tmp_path / "unbound", data) as run:
        request = MechanismFitWindow(run, (1,), through_generation=1)
        protocol = config(data[3].model_id, catalog_id=binding)

        def forbidden(*args, **options):
            raise AssertionError("Se abrió una ventana con otro contrato regional")

        monkeypatch.setattr(module, "_read", forbidden)
        with pytest.raises(ValueError, match="catálogo"):
            audit_fit_windows((request,), config=protocol)
