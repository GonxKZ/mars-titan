"""Cursores por flujo sin estado neural, con lotes manuales."""

from types import SimpleNamespace

import pytest

from mars_titan.memory.flow_cursors import MAX_FLOWS, FlowCursors

MODEL = "a" * 64


def batch(*rows):
    flows, moments = zip(*rows, strict=True)
    return SimpleNamespace(
        flow_ids=flows,
        sample_ids=tuple(f"{flow}/{at}" for flow, at in rows),
        prediction_at=moments,
    )


def test_advance_counts_observations_and_keeps_absent_flows():
    store = FlowCursors(SimpleNamespace(model_id=MODEL))
    first = store.advance(store.empty(), batch(("US/AAA", 10), ("CN/BBB", 10)))
    second = store.advance(first, batch(("US/AAA", 11)))
    assert second["references"] == {
        "US/AAA": dict(observed_steps=2, last_prediction_at=11, last_sample_id="US/AAA/11"),
        "CN/BBB": dict(observed_steps=1, last_prediction_at=10, last_sample_id="CN/BBB/10"),
    }
    assert first["references"]["US/AAA"]["observed_steps"] == 1
    assert store.live_references(second) == [] and not store.needs_compaction(second)


@pytest.mark.parametrize("moment", [10, 9])
def test_repeated_or_earlier_observations_are_rejected(moment):
    store = FlowCursors(SimpleNamespace(model_id=MODEL))
    state = store.advance(store.empty(), batch(("US/AAA", 10)))
    with pytest.raises(ValueError, match="repite o retrocede"):
        store.advance(state, batch(("US/AAA", moment)))


def test_verification_rejects_foreign_or_malformed_manifests():
    store = FlowCursors(SimpleNamespace(model_id=MODEL))
    valid = store.advance(store.empty(), batch(("US/AAA", 10)))
    row = valid["references"]["US/AAA"]
    invalid = [
        dict(valid, model_id="b" * 64),
        dict(valid, blocks={"x": {}}),
        dict(valid, schema_version=2),
        dict(valid, references={"XX/AAA": row}),
        dict(valid, references={"US/AAA": dict(row, observed_steps=0)}),
        dict(valid, references={"US/AAA": dict(row, last_sample_id="US/AAA/9")}),
        dict(valid, references={"US/AAA": dict(row, last_prediction_at=10.0)}),
        dict(valid, references={"US/AAA": dict(row, extra=1)}),
        dict(valid, references={f"US/A{i}": row for i in range(MAX_FLOWS + 1)}),
    ]
    for manifest in invalid:
        with pytest.raises(ValueError):
            store.verify(manifest)
