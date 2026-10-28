"""Diagnósticos B4 publicados en la misma generación que la predicción nativa."""

from types import SimpleNamespace

import pyarrow as pa
import pytest
from test_financial_session import moment
from test_financial_session_associative import (
    EVENTS,
    label,
    options,
    trajectory,
)
from test_financial_session_associative import (
    four_flow_source as four_flow_source,
)
from test_financial_session_associative import (
    frozen_consumer as frozen_consumer,
)
from test_financial_session_associative import (
    module_backend as module_backend,
)
from test_financial_session_associative import (
    no_target_estimation as no_target_estimation,
)
from test_financial_session_associative import (
    shared_native as shared_native,
)
from test_lesion_inference import CASES, PROTOCOL
from test_mars_titan_correction_parity import RECIPE, core
from test_mars_titan_session_parity import PHASE, events
from test_regime_routing import daily_routes

from mars_titan.memory.financial_session import FinancialSession
from mars_titan.memory.regimes import RegimeRule
from mars_titan.training.mars_titan_correction import CorrectionInference, prediction_rows


def diagnostic_run(native, source, consumer, output, correction, *, recover=False, close=False):
    settings = options(native, source, consumer, associative=correction, lesions=PROTOCOL)
    run = FinancialSession(output, **settings)
    pending, emitted, packets = [], [], []
    try:
        for index in EVENTS:
            kwargs = dict(kind="settlement" if index == 129 else "decision", cutoff=moment(index))
            batches = [] if index == 129 else source["batches"][index]
            feedback = [
                native.Feedback(row.id, 0, moment(index), label(row.asset, index))
                for row in pending
            ]
            if recover and index == 128:
                before = run.snapshot()

                def cut(boundary):
                    if boundary == native.Boundary.before_commit:
                        raise RuntimeError("Corte B4 antes de publicar")

                with pytest.raises(RuntimeError, match="Corte B4"):
                    run.step(batches, feedback, fault=cut, **kwargs)
                run.close()
                run = FinancialSession(output, **settings, resume=True)
                assert run.snapshot() == before
            commit = run.step(batches, feedback, **kwargs)
            pending = commit.predictions
            emitted.extend((row.asset, row.decision_at, row.value) for row in pending)
            packets.append(run.lesion_event())
        if close:
            run.step([], [], kind="settlement", cutoff=moment(201), close_phase=True)
            packets.append(run.lesion_event())
        bundle = run._bundle(run.snapshot()["state"])
        memory = run._associative_state(bundle["associative"])[0].export()
        return dict(emitted=emitted, packets=packets, metrics=run.lesion_metrics(), memory=memory)
    finally:
        run.close()


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize("correction", CASES, ids=["areas", "bridge", "e1_roles"])
def test_session_window_and_recovery_keep_identical_paired_diagnostics(
    shared_native, four_flow_source, frozen_consumer, tmp_path, monkeypatch, correction
):
    if correction.routing is not None:
        monkeypatch.setattr(RegimeRule, "routes", daily_routes)
    plain = trajectory(
        shared_native, four_flow_source, frozen_consumer, tmp_path / "plain", correction=correction
    )
    direct = diagnostic_run(
        shared_native, four_flow_source, frozen_consumer, tmp_path / "direct", correction
    )
    resumed = diagnostic_run(
        shared_native,
        four_flow_source,
        frozen_consumer,
        tmp_path / "resumed",
        correction,
        recover=True,
    )
    expected = [(a, t, v) for event in plain["events"] for _, a, t, v in event["predictions"]]
    assert direct["emitted"] == resumed["emitted"] == expected
    assert direct["metrics"] == resumed["metrics"]
    assert direct["memory"]["matrix_sha256"] == plain["events"][-1]["memory"]["matrix_sha256"]
    inference = CorrectionInference(
        core(four_flow_source), RECIPE, correction, four_flow_source["codec"], lesions=PROTOCOL
    )
    rows = prediction_rows(inference)
    inference._pass(SimpleNamespace(phase=PHASE), events(four_flow_source), rows=rows)
    table = pa.concat_tables(rows.finish())
    resolved = {}
    for packet in direct["packets"]:
        for key, predictions in zip(
            packet["resolved"]["keys"], packet["resolved"]["predictions"], strict=True
        ):
            resolved[tuple(key)] = predictions.tolist()
    for index, name in enumerate(inference.lesion_plan.names):
        values = [
            resolved[asset, at][index]
            for asset, at in zip(
                table["asset_id"].to_pylist(),
                table["prediction_at"].cast(pa.int64()).to_pylist(),
                strict=True,
            )
        ]
        assert values == table["lesion_" + name].to_pylist()
    assert direct["metrics"] == inference.lesion_ledger.metrics()


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_phase_finalization_censors_without_inventing_diagnostic_labels(
    shared_native, four_flow_source, frozen_consumer, tmp_path
):
    result = diagnostic_run(
        shared_native, four_flow_source, frozen_consumer, tmp_path / "closed", CASES[0], close=True
    )
    assert result["metrics"]["resolved"] == 16
    assert result["metrics"]["censored"] == 4 and result["metrics"]["pending"] == 0
    assert len(result["packets"][-1]["censored"]) == 4
    assert result["packets"][-1]["resolved"]["keys"] == []


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_session_pins_diagnostic_sources_and_rejects_another_protocol(
    shared_native, four_flow_source, frozen_consumer, tmp_path
):
    import hashlib
    from pathlib import Path

    from mars_titan.memory import lesion_session
    from mars_titan.memory.lesion_protocol import LesionProtocol

    settings = options(
        shared_native, four_flow_source, frozen_consumer, associative=CASES[0], lesions=PROTOCOL
    )
    with FinancialSession(tmp_path / "pinned", **settings) as run:
        expected = hashlib.sha256(Path(lesion_session.__file__).read_bytes()).hexdigest()
        assert run._executor.cursor == 0
        original = run._runtime_contract
        run._lesion_plan = LesionProtocol(seeds=(1,), fractions=((1, 2),)).bind(CASES[0])
        with pytest.raises(ValueError, match="configuración efectiva"):
            run.lesion_event()
        run._lesion_plan = PROTOCOL.bind(CASES[0])
        assert run._runtime_contract == original
        assert lesion_session.code()["lesion_session"] == expected
    changed = settings | dict(lesions=LesionProtocol(seeds=(1,), fractions=((1, 2),)))
    with pytest.raises((ValueError, RuntimeError)):
        FinancialSession(tmp_path / "pinned", **changed, resume=True)


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_diagnostics_reject_misaligned_pending_and_a_changed_original_forecast(
    shared_native, four_flow_source, frozen_consumer, tmp_path
):
    from mars_titan.memory import lesion_session

    settings = options(
        shared_native, four_flow_source, frozen_consumer, associative=CASES[0], lesions=PROTOCOL
    )
    with FinancialSession(tmp_path / "alignment", **settings) as run:
        commit = run.step(four_flow_source["batches"][125], [], cutoff=moment(125))
        bundle = run._bundle(run.snapshot()["state"])
        with pytest.raises(ValueError, match="pendientes"):
            lesion_session.verify(run, bundle, [])
        ledger = lesion_session.begin(run, bundle, moment(126))
        prediction = commit.predictions[0]
        ledger.pending[prediction.asset, prediction.decision_at][0] += 1.0
        proposal = dict(lesions=lesion_session.stage(run, ledger))
        outcome = SimpleNamespace(
            prediction=prediction, label=SimpleNamespace(available_at=moment(126), value=0.1)
        )
        with pytest.raises(ValueError, match="control intacto"):
            lesion_session.resolve(run, proposal, [outcome], [], [])
