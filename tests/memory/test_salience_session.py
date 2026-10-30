"""Publicación durable E3 sobre emisiones, etiquetas y fallos del ejecutor real."""

from dataclasses import replace

import pytest
import torch
from test_financial_session import moment
from test_financial_session_associative import (
    DELTA,
    associative_state,
    options,
)
from test_financial_session_associative import four_flow_source as four_flow_source
from test_financial_session_associative import frozen_consumer as frozen_consumer
from test_financial_session_associative import module_backend as module_backend
from test_financial_session_associative import no_target_estimation as no_target_estimation
from test_financial_session_associative import shared_native as shared_native
from test_multiscale_session import BASE, BRIDGE

from mars_titan.memory import salience_session
from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.memory.salience_budget import SalienceBudgetConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer


def session(native, source, consumer, output, salience=None, correction=DELTA, **extra):
    return FinancialSession(
        output,
        **(options(native, source, consumer, associative=correction) | extra),
        salience=salience,
    )


def feedback(native, predictions, at, errors=None):
    errors = errors or [float(index + 1) for index in range(len(predictions))]
    return [
        native.Feedback(p.id, 0, moment(at), p.value + error)
        for p, error in zip(predictions, errors, strict=True)
    ]


def values(commit):
    return [(p.asset, p.decision_at, p.value) for p in commit.predictions]


def details(run):
    state = run.salience_state()
    memory, _ = associative_state(run)
    assert state["statistics"]["selected"] == memory.writes
    assert (
        sum(value[0] for value in state["statistics"]["records"].values())
        == run.snapshot()["applied"]
    )
    assert state["generation"] == run.snapshot()["cursor"]
    return state, memory


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize("correction", (DELTA, BASE, BRIDGE))
def test_large_quota_matches_the_original_points_and_full_memory(
    shared_native, four_flow_source, frozen_consumer, tmp_path, correction
):
    def factory(path, factor):
        return session(shared_native, four_flow_source, frozen_consumer, path, factor, correction)

    with (
        factory(tmp_path / "old", None) as plain,
        factory(tmp_path / "full", SalienceBudgetConfig(writes_per_market=8192)) as full,
    ):
        previous_a, previous_b = [], []
        for at in (125, 126, 127):
            a = plain.step(four_flow_source["batches"][at], feedback(shared_native, previous_a, at))
            b = full.step(four_flow_source["batches"][at], feedback(shared_native, previous_b, at))
            assert values(a) == values(b)
            x, y = associative_state(plain)[0].export(), associative_state(full)[0].export()
            for name in ("configuration", "writes", "cursor", "matrix_sha256"):
                assert x[name] == y[name]
            if "state_sha256" in x:
                assert x["state_sha256"] == y["state_sha256"]
            details(full)
            previous_a, previous_b = a.predictions, b.predictions
        assert plain.salience_state() is None
        assert "salience" not in plain._bundle(plain.snapshot()["state"])


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_zero_quota_keeps_all_mature_errors_without_writing_the_memory(
    shared_native, four_flow_source, frozen_consumer, tmp_path
):
    with session(
        shared_native,
        four_flow_source,
        frozen_consumer,
        tmp_path / "zero",
        SalienceBudgetConfig(writes_per_market=0),
    ) as run:
        first = run.step(four_flow_source["batches"][125], [])
        run.step(four_flow_source["batches"][126], feedback(shared_native, first.predictions, 126))
        state, memory = details(run)
        assert memory.writes == 0 and memory.cursor is None
        assert torch.count_nonzero(memory.matrix) == 0
        assert len(state["statistics"]["records"]) == state["last"]["observed"] == 4
        assert state["last"]["selected"] == []


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_priority_uses_the_final_emission_and_the_write_value_still_uses_the_core(
    shared_native, four_flow_source, frozen_consumer, tmp_path
):
    factor = SalienceBudgetConfig(rule="absolute_error", writes_per_market=1)
    with session(
        shared_native, four_flow_source, frozen_consumer, tmp_path / "issued", factor
    ) as run:
        first = run.step(four_flow_source["batches"][125], [])
        run.step(
            four_flow_source["batches"][126],
            feedback(shared_native, first.predictions, 126, [50.0, 0.0, 0.0, 0.0]),
        )
        issued = run.step(four_flow_source["batches"][127], []).predictions
        bundle = run._bundle(run.snapshot()["state"])
        pending = run._read(bundle["pending"], "pending")
        old_memory, core, _ = run._associative_state(bundle["associative"])
        positions = {(r["flow_id"], r["prediction_at"]): i for i, r in enumerate(pending["rows"])}
        indexes = [positions[p.asset, p.decision_at] for p in issued]
        offsets = [p.value - float(core[i]) for p, i in zip(issued, indexes, strict=True)]
        a = max(range(4), key=lambda i: abs(offsets[i]))
        b = (a + 1) % 4
        scale = abs(offsets[a])
        assert scale > 1e-5
        errors = [0.0] * 4
        errors[a] = (1 if offsets[a] > 0 else -1) * scale / 4
        errors[b] = -(1 if offsets[b] >= 0 else -1) * scale / 2
        labels = feedback(shared_native, issued, 128, errors)
        chosen = indexes[b]
        expected = old_memory.write(
            DELTA.feedback(
                ids=[old_memory.writes + 1],
                decision_at=[issued[b].decision_at],
                available_at=[moment(128)],
                keys=DELTA.keys(pending["key_inputs"][chosen : chosen + 1]),
                values=[labels[b].value - float(core[chosen])],
            ),
            cutoff=moment(128),
        )
        commit = run.step(four_flow_source["batches"][128], labels)
        assert len(commit.applied) == 4
        state, memory = details(run)
        assert state["last"]["selected"] == [[issued[b].asset, issued[b].decision_at]]
        assert torch.equal(memory.matrix, expected.matrix)
        assert memory.writes == 2


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_partial_publications_at_one_cutoff_share_its_quota_and_replays_are_rejected(
    shared_native, four_flow_source, frozen_consumer, tmp_path
):
    factor = SalienceBudgetConfig(writes_per_market=1)
    with session(
        shared_native, four_flow_source, frozen_consumer, tmp_path / "partial", factor
    ) as run:
        first = run.step(four_flow_source["batches"][125], [])
        labels = feedback(shared_native, first.predictions, 126)
        run.step(four_flow_source["batches"][126], labels[:2])
        run.step([], labels[2:], kind="settlement", cutoff=moment(126))
        state, memory = details(run)
        assert state["generation"] == 3 and state["cutoff"] == moment(126)
        assert state["statistics"]["selected"] == memory.writes == 1
        assert state["instant_writes"] == {"US": 1} and state["last"]["selected"] == []
        assert run.snapshot()["applied"] == 4
        before = (run.output / "latest.json").read_bytes()
        with pytest.raises(ValueError):
            run.step([], labels[2:], kind="settlement", cutoff=moment(127))
        assert (run.output / "latest.json").read_bytes() == before


def test_the_contract_declares_the_quota_of_each_publication_instant():
    contract = salience_session.contract(SalienceBudgetConfig(writes_per_market=1))
    assert contract["kind"] == "salience_committed_publication_v2"
    assert contract["quota"] == (
        "per_market_and_publication_instant_shared_by_the_generations_of_that_cutoff"
    )
    assert "calendar_session_quota" not in contract


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_late_earlier_decisions_are_observed_and_missing_labels_are_finalized_without_errors(
    shared_native, four_flow_source, frozen_consumer, tmp_path
):
    factor = SalienceBudgetConfig(writes_per_market=1, min_history=1)
    with session(
        shared_native, four_flow_source, frozen_consumer, tmp_path / "late", factor
    ) as run:
        first = run.step(four_flow_source["batches"][125], [])
        second = run.step(four_flow_source["batches"][126], [])
        run.step(four_flow_source["batches"][127], feedback(shared_native, second.predictions, 127))
        run.step(four_flow_source["batches"][128], feedback(shared_native, first.predictions, 128))
        state, _ = details(run)
        assert all(v[0] == 2 for v in state["statistics"]["records"].values())
        assert all(v[2] == moment(125) for v in state["statistics"]["records"].values())
        closed = run.step([], [], kind="settlement", cutoff=moment(201), close_phase=True)
        after, _ = details(run)
        assert len(closed.finalized) == 8 and not closed.applied
        assert after["statistics"] == state["statistics"] and after["last"]["observed"] == 0


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize("boundary", ("before_commit", "committed"))
def test_recovery_preserves_the_budget_and_next_prediction(
    shared_native, four_flow_source, frozen_consumer, tmp_path, boundary
):
    factor = SalienceBudgetConfig(writes_per_market=1, rule="uniform", seed=17)
    args = (shared_native, four_flow_source, frozen_consumer)
    reference = session(*args, tmp_path / "reference", factor)
    failed = session(*args, tmp_path / "failed", factor)
    try:
        a = reference.step(four_flow_source["batches"][125], [])
        b = failed.step(four_flow_source["batches"][125], [])
        reference.step(
            four_flow_source["batches"][126], feedback(shared_native, a.predictions, 126)
        )

        def fault(point):
            if point == getattr(shared_native.Boundary, boundary):
                raise RuntimeError("Corte del fixture E3")

        with pytest.raises(RuntimeError, match="Corte del fixture"):
            failed.step(
                four_flow_source["batches"][126],
                feedback(shared_native, b.predictions, 126),
                fault=fault,
            )
        failed.close()
        failed = session(*args, tmp_path / "failed", factor, resume=True)
        if boundary == "before_commit":
            failed.step(
                four_flow_source["batches"][126], feedback(shared_native, b.predictions, 126)
            )
        assert failed.salience_state() == reference.salience_state()
        assert values(failed.step(four_flow_source["batches"][127], [])) == values(
            reference.step(four_flow_source["batches"][127], [])
        )
        details(failed)
    finally:
        reference.close()
        failed.close()


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_changed_runtime_or_resume_policy_is_rejected(
    shared_native, four_flow_source, frozen_consumer, tmp_path
):
    factor = SalienceBudgetConfig(writes_per_market=1)
    args = (shared_native, four_flow_source, frozen_consumer, tmp_path / "contract")
    with session(*args, factor) as run:
        run.step(four_flow_source["batches"][125], [])
        before = run.snapshot()
        run._salience_config = replace(factor, writes_per_market=2)
        with pytest.raises(ValueError, match="configuración"):
            run.step(four_flow_source["batches"][126], [])
        assert run.snapshot() == before
    with pytest.raises(ValueError):
        session(*args, replace(factor, writes_per_market=2), resume=True)


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize("correction", (BASE, BRIDGE))
def test_reordering_assets_and_feedback_keeps_e1_state_and_does_not_consume_rng(
    shared_native, four_flow_source, frozen_consumer, tmp_path, correction
):
    factor = SalienceBudgetConfig(writes_per_market=1, rule="uniform", seed=73)
    args = (shared_native, four_flow_source, frozen_consumer)
    with (
        session(*args, tmp_path / "normal", factor, correction) as normal,
        session(*args, tmp_path / "reverse", factor, correction) as reverse,
    ):
        rng = torch.get_rng_state()
        previous_a, previous_b = [], []
        for at in (125, 126, 127):
            a = normal.step(
                four_flow_source["batches"][at], feedback(shared_native, previous_a, at)
            )
            b = reverse.step(
                four_flow_source["batches"][at][::-1],
                feedback(shared_native, previous_b, at)[::-1],
            )
            assert values(a) == values(b)
            assert normal.salience_state() == reverse.salience_state()
            x, y = details(normal)[1], details(reverse)[1]
            assert x.export()["state_sha256"] == y.export()["state_sha256"]
            assert x.writes == max(0, at - 125)
            previous_a, previous_b = a.predictions, b.predictions
        assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_future_feedback_does_not_change_the_confirmed_generation(
    shared_native, four_flow_source, frozen_consumer, tmp_path
):
    with session(
        shared_native,
        four_flow_source,
        frozen_consumer,
        tmp_path / "future",
        SalienceBudgetConfig(writes_per_market=1),
    ) as run:
        first = run.step(four_flow_source["batches"][125], [])
        before = (run.output / "latest.json").read_bytes()
        with pytest.raises(ValueError, match="futuro"):
            run.step(
                four_flow_source["batches"][126], feedback(shared_native, first.predictions, 127)
            )
        assert (run.output / "latest.json").read_bytes() == before


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize("factor,correction", ((False, DELTA), (SalienceBudgetConfig(), None)))
def test_invalid_session_configuration_fails_before_creating_output(
    shared_native, four_flow_source, frozen_consumer, tmp_path, factor, correction
):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="E3"):
        session(shared_native, four_flow_source, frozen_consumer, output, factor, correction)
    assert not output.exists()


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_warmup_and_prefix_exclusions_advance_the_clock_without_fabricating_labels(
    shared_native, tmp_path, monkeypatch
):
    import test_financial_session_controls as fixture

    monkeypatch.setattr(fixture, "DECISIONS", (63, 64, 125))
    source = fixture.flow_source(tmp_path / "early-source")
    predictor = FinancialPredictor(
        FinancialConfig(source["spec"], variant="mac_online", hidden_size=32),
        dtype=torch.float64,
        device="cpu",
    )
    consumer = FrozenFinancialConsumer(predictor.eval().requires_grad_(False))
    phase = FinancialPhase("validation", moment(63), moment(64), moment(200), moment(201))
    with session(
        shared_native,
        source,
        consumer,
        tmp_path / "early",
        SalienceBudgetConfig(writes_per_market=1),
        phase=phase,
    ) as run:
        warm = run.step(source["batches"][63], [], kind="warmup")
        excluded = run.step(source["batches"][64], [])
        assert not warm.predictions and len(excluded.excluded) == 4
        state, memory = details(run)
        assert state["generation"] == 2 and state["statistics"]["records"] == {}
        assert memory.writes == run.snapshot()["applied"] == 0
        assert state["last"]["observed"] == 0


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_lesion_diagnostics_observe_rejected_writes_without_changing_e3_predictions(
    shared_native, four_flow_source, frozen_consumer, tmp_path
):
    from test_lesion_inference import PROTOCOL

    args = (shared_native, four_flow_source, frozen_consumer)
    factor = SalienceBudgetConfig(writes_per_market=1)
    with (
        session(*args, tmp_path / "plain-b4", factor, BRIDGE) as plain,
        session(*args, tmp_path / "active-b4", factor, BRIDGE, lesions=PROTOCOL) as diagnostic,
    ):
        previous_a, previous_b = [], []
        for at in (125, 126, 127):
            a = plain.step(four_flow_source["batches"][at], feedback(shared_native, previous_a, at))
            b = diagnostic.step(
                four_flow_source["batches"][at], feedback(shared_native, previous_b, at)
            )
            assert values(a) == values(b)
            assert plain.salience_state() == diagnostic.salience_state()
            assert (
                details(plain)[1].export()["state_sha256"]
                == details(diagnostic)[1].export()["state_sha256"]
            )
            previous_a, previous_b = a.predictions, b.predictions
        metrics = diagnostic.lesion_metrics()
        assert metrics["resolved"] == diagnostic.snapshot()["applied"] == 8
        assert metrics["issued"] == 12 and metrics["pending"] == 4
        assert len(diagnostic.lesion_event()["resolved"]["keys"]) == 4
        assert details(diagnostic)[1].writes == 2
