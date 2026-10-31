"""Caché S2 limitada al evento, con copias aisladas y archivos siempre verificados."""

from dataclasses import replace

import pytest
from test_failure_session import api, book, consumer, open_session, policy
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native

from mars_titan.memory.failure_ledger import FailureLedger

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")


@pytest.mark.parametrize("budget", [-1, True, 0.5, 64 * 1024**2 + 1])
def test_decode_budget_is_closed(four_flow_source, tmp_path, budget):
    engine = consumer(four_flow_source)
    config = policy(four_flow_source, engine, tmp_path)
    with pytest.raises(ValueError, match="caché"):
        replace(config, decoded_cache_bytes=budget)


def test_decode_cache_reduces_reconstructions_without_changing_the_next_state(
    native, four_flow_source, tmp_path, monkeypatch
):
    source = four_flow_source
    engine = consumer(source)
    base = policy(source, engine, tmp_path)
    original = FailureLedger.restore
    calls = []

    def tracked(cls, *args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(FailureLedger, "restore", classmethod(tracked))
    outcomes = []
    for budget in (0, 1, 8 * 1024**2):
        config = replace(base, decoded_cache_bytes=budget)
        start = len(calls)
        with open_session(native, source, engine, tmp_path / f"budget-{budget}", config) as run:
            previous, points = [], []
            for at in (125, 126, 127):
                labels = [native.Feedback(p.id, 0, moment(at), 0.125) for p in previous]
                result = run.step(source["batches"][at], labels)
                points.append([(p.asset, p.value) for p in result.predictions])
                previous = result.predictions
                assert run._inflight is None
            outcomes.append((points, book(run).export(), len(calls) - start))
    assert outcomes[0][:2] == outcomes[1][:2] == outcomes[2][:2]
    assert outcomes[0][2] == outcomes[1][2]
    assert outcomes[2][2] < outcomes[0][2]


def test_cached_book_copies_do_not_share_mutable_records(native, four_flow_source, tmp_path):
    source = four_flow_source
    engine = consumer(source)
    config = replace(policy(source, engine, tmp_path), decoded_cache_bytes=8 * 1024**2)
    checked = []
    with open_session(native, source, engine, tmp_path / "copies", config) as run:
        first = run.step(source["batches"][125], [])
        labels = [native.Feedback(p.id, 0, moment(126), 0.125) for p in first.predictions]

        def inspect(boundary):
            if boundary != native.Boundary.before_commit:
                return
            left = api().restore(run, run._proposal)
            expected = left.export()
            emission = next(iter(left._pending.values()))
            object.__setattr__(emission.context, "sample_id", "tampered")
            object.__setattr__(emission, "prediction", 99.0)
            object.__setattr__(left._retained[0], "target", 99.0)
            left._pending.clear()
            left._last.clear()
            right = api().restore(run, run._proposal)
            assert right.export() == expected
            right.censor(list(right._pending))
            alternative = dict(run._proposal, failure_memory=api().stage(run, right))
            assert api().restore(run, alternative).counts()["pending"] == 0
            assert api().restore(run, run._proposal).export() == expected
            checked.append(True)

        run.step(source["batches"][126], labels, fault=inspect)
        assert checked == [True]
        assert book(run).counts()["admitted"] == 4


def test_cache_hit_still_checks_the_file_hash(native, four_flow_source, tmp_path):
    source = four_flow_source
    engine = consumer(source)
    config = replace(policy(source, engine, tmp_path), decoded_cache_bytes=8 * 1024**2)
    checked = []
    with open_session(native, source, engine, tmp_path / "integrity", config) as run:

        def inspect(boundary):
            if boundary != native.Boundary.before_commit:
                return
            expected = api().restore(run, run._proposal).export()
            ref = run._proposal["failure_memory"]
            path = run.artifacts.directory / ref["name"]
            raw = path.read_bytes()
            try:
                path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
                with pytest.raises(ValueError, match="cambiado"):
                    api().restore(run, run._proposal)
            finally:
                path.write_bytes(raw)
            assert api().restore(run, run._proposal).export() == expected
            checked.append(True)

        run.step(source["batches"][125], [], fault=inspect)
        assert checked == [True]


def test_budget_charges_the_graph_actually_retained(native, four_flow_source, tmp_path):
    from mars_titan.memory import failure_cache

    source = four_flow_source
    engine = consumer(source)
    config = replace(policy(source, engine, tmp_path), decoded_cache_bytes=8 * 1024**2)
    checked = []
    with open_session(native, source, engine, tmp_path / "accounting", config) as run:

        def inspect(boundary):
            if boundary != native.Boundary.before_commit:
                return
            candidate = api().restore(run, run._proposal)
            budget = failure_cache.retained_bytes(candidate, 64 * 1024**2)
            original = run._failure_config
            run._inflight.pop(failure_cache._SLOT, None)
            try:
                run._failure_config = replace(config, decoded_cache_bytes=budget)
                failure_cache.remember(run, run._proposal["failure_memory"], candidate)
                entry = run._inflight.get(failure_cache._SLOT)
                if entry is not None:
                    assert (
                        failure_cache.retained_bytes(entry[1], 64 * 1024**2, reference_key=entry[0])
                        <= budget
                    )
                checked.append(True)
            finally:
                run._failure_config = original
                run._inflight.pop(failure_cache._SLOT, None)

        run.step(source["batches"][125], [], fault=inspect)
        assert checked == [True]


def test_failed_event_discards_the_cache_and_recovers_only_confirmed_labels(
    native, four_flow_source, tmp_path
):
    source = four_flow_source
    engine = consumer(source)
    config = replace(policy(source, engine, tmp_path), decoded_cache_bytes=8 * 1024**2)
    path = tmp_path / "rollback"
    with open_session(native, source, engine, path, config) as run:
        first = run.step(source["batches"][125], [])
        saved = book(run).export()
        labels = [native.Feedback(p.id, 0, moment(126), 0.125) for p in first.predictions]

        def fail(boundary):
            if boundary == native.Boundary.before_commit:
                api().restore(run, run._proposal)
                raise RuntimeError("Corte técnico con caché temporal")

        with pytest.raises(RuntimeError, match="Corte técnico"):
            run.step(source["batches"][126], labels, fault=fail)
        assert run._inflight is None
    with open_session(native, source, engine, path, config, resume=True) as run:
        assert book(run).export() == saved
        run.step(source["batches"][126], labels)
        assert book(run).counts()["admitted"] == 4
        assert run._inflight is None


def test_runtime_budget_cannot_change_silently(native, four_flow_source, tmp_path):
    source = four_flow_source
    engine = consumer(source)
    config = replace(policy(source, engine, tmp_path), decoded_cache_bytes=8 * 1024**2)
    with open_session(native, source, engine, tmp_path / "runtime-budget", config) as run:
        run.step(source["batches"][125], [])
        before = run.snapshot()
        try:
            object.__setattr__(config, "decoded_cache_bytes", 0)
            with pytest.raises(ValueError, match="configuración"):
                run.step(source["batches"][126], [])
        finally:
            object.__setattr__(config, "decoded_cache_bytes", 8 * 1024**2)
        assert run.snapshot() == before
