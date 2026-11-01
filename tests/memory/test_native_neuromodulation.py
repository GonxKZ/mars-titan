"""N1 con predicciones y resultados del ejecutor real, sin aprendizaje."""

from dataclasses import replace

import pytest
import torch
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native

from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.memory.neuromodulation_source import (
    NeuromodulationSource,
    NeuromodulationSourceConfig,
)
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.memory.write_policy import CompositeScoreConfig, MatureErrorConfig
from mars_titan.models.titans.dual_neuromodulation import DualNeuromodulationConfig
from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.state import mac_tensors
from tests.memory.test_financial_session import moment
from tests.memory.test_mars_titan_session_parity import SCALERS

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")


def consumer(source, *, neutral=False, admission="m1", source_kind="observed"):
    gains = DualNeuromodulationConfig(
        **(dict(write_expected=0, write_unexpected=0, forget_unexpected=0) if neutral else {})
    )
    config = FinancialConfig(
        source["spec"], variant="mac_online", hidden_size=32, dual_neuromodulation=gains
    )
    model = FinancialPredictor(config, dtype=torch.float64).eval().requires_grad_(False)
    reader = (
        EpisodicReadout(
            EpisodicReadoutConfig(
                source["codec"].fingerprint(),
                hidden_size=32,
                mode="no_bank" if admission == "m0" else "bank",
            ),
            dtype=torch.float64,
        )
        .eval()
        .requires_grad_(False)
    )
    source_config = NeuromodulationSourceConfig(
        window=4, min_events=2, expected_scale=0.1, source=source_kind
    )
    return FrozenFinancialConsumer(model, readout=reader, neuromodulation_source=source_config)


def session(
    native, source, model, path, *, resume=False, block_rows=1, admission="m1", decision_start=125
):
    return FinancialSession(
        path,
        native=native,
        consumer=model,
        codec=source["codec"],
        prefixes=source["prefixes"],
        retention=MatureErrorConfig(capacity=8)
        if admission == "m2"
        else CompositeScoreConfig(SCALERS, capacity=8)
        if admission == "m3"
        else RetentionConfig(capacity=8, frontier=2, new_candidates=4),
        phase=FinancialPhase(
            "validation", moment(125), moment(decision_start), moment(200), moment(201)
        ),
        admission=admission,
        world="n1_native_fixture",
        fold="0",
        resume=resume,
        block_rows=block_rows,
    )


def feedback(native, predictions, at, target=0.125):
    return [native.Feedback(p.id, 0, moment(at), target) for p in predictions]


def test_uninitialized_session_exposes_empty_source_and_no_decision(
    native, four_flow_source, tmp_path
):
    source = four_flow_source
    with session(native, source, consumer(source), tmp_path / "empty") as run:
        initial = run.neuromodulation_state()
        assert initial["generation"] == initial["applied_total"] == 0
        assert initial["rows"] == [] and initial["cutoff"] is None
        assert run.neuromodulation_snapshot() is None


def test_source_advances_after_prediction_and_uses_actual_native_outcomes(
    native, four_flow_source, tmp_path
):
    source = four_flow_source
    model = consumer(source)
    with session(native, source, model, tmp_path / "run") as run:
        first = run.step(source["batches"][125], [])
        assert all(tuple(row) == (0.0, 0.0) for row in run.neuromodulation_snapshot()["signals"])
        second = run.step(source["batches"][126], feedback(native, first.predictions, 126))
        assert all(tuple(row) == (0.0, 0.0) for row in run.neuromodulation_snapshot()["signals"])
        third = run.step(source["batches"][127], feedback(native, second.predictions, 127))
        assert run.neuromodulation_state()["applied_total"] == run.snapshot()["applied"] == 8
        assert all(tuple(row) == (0.0, 0.0) for row in run.neuromodulation_snapshot()["signals"])
        run.step(source["batches"][128], feedback(native, third.predictions, 128, target=0.5))
        assert all(row[0] > 0 for row in run.neuromodulation_snapshot()["signals"])
        assert run.neuromodulation_snapshot()["generation"] == 3
        following = NeuromodulationSource.restore(
            model.neuromodulation_source, run.neuromodulation_state()
        )
        assert all(
            row[1] > 0
            for row in following.snapshot(
                tuple(sorted(run.fast_state()["references"])), prediction_at=moment(129)
            ).signals
        )


@pytest.mark.parametrize("source_kind", ["observed", "shuffled_past_prefix"])
def test_blocks_reordered_assets_and_resume_keep_the_same_next_prediction(
    native, four_flow_source, tmp_path, source_kind
):
    source = four_flow_source
    model = consumer(source, source_kind=source_kind)
    expected = []
    for block_rows, reopen in ((4, False), (1, True)):
        path = tmp_path / str(block_rows)
        run = session(native, source, model, path, block_rows=block_rows)
        previous = []
        for at in (125, 126, 127):
            result = run.step(
                list(reversed(source["batches"][at])) if reopen else source["batches"][at],
                feedback(native, previous, at),
            )
            previous = result.predictions
        before = run.snapshot()
        if reopen:
            run.close()
            run = session(native, source, model, path, block_rows=block_rows, resume=True)
            assert run.snapshot() == before
        result = run.step(source["batches"][128], feedback(native, previous, 128))
        flows = tuple(sorted(run.fast_state()["references"]))
        state = run._fast_store.gather(run.fast_state(), flows)
        expected.append(
            (
                [(p.asset, p.value) for p in result.predictions],
                tuple(x.cpu().clone() for x in mac_tensors(state.mac)),
                run.neuromodulation_state(),
            )
        )
        run.close()
    assert [a for a, _ in expected[0][0]] == [a for a, _ in expected[1][0]]
    torch.testing.assert_close(
        torch.tensor([v for _, v in expected[0][0]], dtype=torch.float64),
        torch.tensor([v for _, v in expected[1][0]], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    for a, b in zip(expected[0][1], expected[1][1], strict=True):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_zero_coefficients_preserve_the_previous_native_predictions_and_fast_state(
    native, four_flow_source, tmp_path
):
    source = four_flow_source
    model = consumer(source, neutral=True)
    reference = (
        FinancialPredictor(
            replace(model.predictor.config, dual_neuromodulation=None), dtype=torch.float64
        )
        .eval()
        .requires_grad_(False)
    )
    plain = FrozenFinancialConsumer(reference, readout=model.readout)
    with (
        session(native, source, model, tmp_path / "n1") as n1,
        session(native, source, plain, tmp_path / "plain") as baseline,
    ):
        previous = [], []
        for at in (125, 126, 127, 128):
            left = n1.step(source["batches"][at], feedback(native, previous[0], at))
            right = baseline.step(source["batches"][at], feedback(native, previous[1], at))
            assert [(p.asset, p.value) for p in left.predictions] == [
                (p.asset, p.value) for p in right.predictions
            ]
            previous = left.predictions, right.predictions
        assert baseline.neuromodulation_state() is None
        assert baseline.neuromodulation_snapshot() is None
        flows = tuple(sorted(n1.fast_state()["references"]))
        a = n1._fast_store.gather(n1.fast_state(), flows)
        b = baseline._fast_store.gather(baseline.fast_state(), flows)
        assert all(
            torch.equal(x, y) for x, y in zip(mac_tensors(a.mac), mac_tensors(b.mac), strict=True)
        )


def test_mature_signal_changes_fast_weights_without_changing_shared_parameters(
    native, four_flow_source, tmp_path
):
    source = four_flow_source
    active, neutral = consumer(source), consumer(source, neutral=True)
    shared = {name: value.clone() for name, value in active.predictor.named_parameters()}
    with (
        session(native, source, active, tmp_path / "active") as left,
        session(native, source, neutral, tmp_path / "neutral") as right,
    ):
        previous = [], []
        for at in (125, 126, 127, 128):
            a = left.step(source["batches"][at], feedback(native, previous[0], at))
            b = right.step(source["batches"][at], feedback(native, previous[1], at))
            previous = a.predictions, b.predictions
        flows = tuple(sorted(left.fast_state()["references"]))
        a = left._fast_store.gather(left.fast_state(), flows)
        b = right._fast_store.gather(right.fast_state(), flows)
        assert any(
            not torch.equal(x, y)
            for x, y in zip(mac_tensors(a.mac), mac_tensors(b.mac), strict=True)
        )
        assert all(
            torch.equal(value, dict(active.predictor.named_parameters())[name])
            for name, value in shared.items()
        )


@pytest.mark.parametrize("admission", ["m0", "m1", "m2", "m3"])
def test_each_admission_uses_all_applied_results_and_recovers(
    native, four_flow_source, tmp_path, admission
):
    source = four_flow_source
    model = consumer(source, admission=admission)
    path = tmp_path / admission
    run = session(native, source, model, path, admission=admission)
    first = run.step(source["batches"][125], [])
    run.step(source["batches"][126], feedback(native, first.predictions, 126))
    before = run.snapshot()
    run.close()
    with session(native, source, model, path, admission=admission, resume=True) as run:
        assert run.snapshot() == before
        assert run.neuromodulation_state()["applied_total"] == run.snapshot()["applied"] == 4
        run.step(source["batches"][127], [])


def test_warmup_and_settlement_never_invent_a_decision_snapshot_or_duplicate_results(
    native, four_flow_source, tmp_path
):
    source = four_flow_source
    model = consumer(source)
    with session(native, source, model, tmp_path / "settle", decision_start=126) as run:
        run.step(source["batches"][125], [], kind="warmup")
        assert run.neuromodulation_state()["applied_total"] == 0
        first = run.step(source["batches"][126], [])
        labels = feedback(native, first.predictions, 127)
        run.step([], labels, kind="settlement", cutoff=moment(127))
        assert run.neuromodulation_snapshot() is None
        assert run.neuromodulation_state()["applied_total"] == 4
        before = run.snapshot()
        with pytest.raises(ValueError):
            run.step([], labels, kind="settlement", cutoff=moment(128))
        with session(
            native, source, model, tmp_path / "settle", decision_start=126, resume=True
        ) as resumed:
            assert resumed.snapshot() == before
            assert (
                resumed.neuromodulation_state()["applied_total"]
                == resumed.snapshot()["applied"]
                == 4
            )


def test_interrupted_publication_keeps_both_sources_and_repeats_the_same_event(
    native, four_flow_source, tmp_path
):
    source = four_flow_source
    model = consumer(source)
    path = tmp_path / "rollback"
    run = session(native, source, model, path)
    first = run.step(source["batches"][125], [])
    run.step(source["batches"][126], feedback(native, first.predictions, 126))
    before, signal = run.snapshot(), run.neuromodulation_state()

    def fail(boundary):
        if boundary == native.Boundary.before_commit:
            raise RuntimeError("fallo N1 del fixture")

    with pytest.raises(RuntimeError, match="fallo N1"):
        run.step(source["batches"][127], [], fault=fail)
    run.close()
    with session(native, source, model, path, resume=True) as run:
        assert run.snapshot() == before and run.neuromodulation_state() == signal
        assert len(run.step(source["batches"][127], []).predictions) == 4


def test_rehashed_current_source_must_match_native_issued_predictions_and_targets(
    native, four_flow_source, tmp_path
):
    import json

    from mars_titan.memory import neuromodulation_session as link
    from mars_titan.memory.neuromodulation_source import _digest

    source = four_flow_source
    model = consumer(source)
    with session(native, source, model, tmp_path / "tamper") as run:
        first = run.step(source["batches"][125], [])
        run.step(source["batches"][126], feedback(native, first.predictions, 126))
        snap = run.snapshot()
        bundle = run._bundle(snap["state"])
        record = json.loads(run._executor.record_json(snap["cursor"]))
        link.verify_confirmed(run, bundle, snap, record)
        payload = run.neuromodulation_state()
        payload["rows"][0]["issued"] += 1
        payload["sha256"] = _digest({k: v for k, v in payload.items() if k != "sha256"})
        NeuromodulationSource.restore(model.neuromodulation_source, payload)
        forged = dict(bundle, neuromodulation=run._stage(payload, "neuromodulation"))
        with pytest.raises(ValueError, match="N1"):
            link.verify_confirmed(run, forged, snap, record)
        assert run.snapshot() == snap


def test_source_must_match_the_native_applied_count(native, four_flow_source, tmp_path):
    import json

    from mars_titan.memory import neuromodulation_session as link

    source = four_flow_source
    model = consumer(source)
    with session(native, source, model, tmp_path / "count") as run:
        first = run.step(source["batches"][125], [])
        run.step(source["batches"][126], feedback(native, first.predictions, 126))
        snap = run.snapshot()
        bundle = run._bundle(snap["state"])
        record = json.loads(run._executor.record_json(snap["cursor"]))
        link.verify_confirmed(run, bundle, snap, record)
        with pytest.raises(ValueError):
            link.verify_confirmed(run, bundle, dict(snap, applied=snap["applied"] + 1), record)


def test_previous_generation_cannot_be_relabelled_even_with_a_matching_trace(
    native, four_flow_source, tmp_path
):
    from mars_titan.memory import neuromodulation_session as link

    source = four_flow_source
    model = consumer(source)
    with session(native, source, model, tmp_path / "generation") as run:
        first = run.step(source["batches"][125], [])
        run.step(source["batches"][126], feedback(native, first.predictions, 126))
        bundle = run._bundle(run.snapshot()["state"])
        control = run._read(bundle["control"], "control")
        flows = tuple(run.neuromodulation_snapshot()["flow_ids"])
        link.verify(run, bundle, control, flows)
        before = link.state(run, bundle, previous=True)
        before.generation += 1
        forged_bundle = dict(
            bundle, neuromodulation_before=run._stage(before.export(), "neuromodulation")
        )
        forged_control = dict(
            control,
            neuromodulation_snapshot=before.snapshot(
                flows, prediction_at=bundle["cutoff"], context_id=control["context_id"]
            ).record(),
        )
        with pytest.raises(ValueError, match="generación"):
            link.verify(run, forged_bundle, forged_control, flows)


def test_decision_trace_must_match_the_previous_source(native, four_flow_source, tmp_path):
    from mars_titan.memory import neuromodulation_session as link

    source = four_flow_source
    model = consumer(source)
    with session(native, source, model, tmp_path / "trace") as run:
        previous = []
        for at in (125, 126, 127, 128):
            previous = run.step(source["batches"][at], feedback(native, previous, at)).predictions
        bundle = run._bundle(run.snapshot()["state"])
        control = run._read(bundle["control"], "control")
        flows = tuple(run.neuromodulation_snapshot()["flow_ids"])
        link.verify(run, bundle, control, flows)
        forged_trace = dict(run.neuromodulation_snapshot(), state_id="0" * 64)
        with pytest.raises(ValueError, match="snapshot N1"):
            link.verify(run, bundle, dict(control, neuromodulation_snapshot=forged_trace), flows)
        with pytest.raises(ValueError, match="generación"):
            link.verify(
                run, dict(bundle, neuromodulation_before=bundle["neuromodulation"]), control, flows
            )
