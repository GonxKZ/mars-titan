"""Snapshots N1 tipados, ligados al evento y a su configuración de fuente."""

from dataclasses import replace

import pytest
import torch
from test_financial_adapter import setup
from test_frozen_financial import frozen_backend as frozen_backend

from mars_titan.memory.neuromodulation_source import (
    NeuromodulationSource,
    NeuromodulationSourceConfig,
)
from mars_titan.models.titans.dual_neuromodulation import DualNeuromodulationConfig
from mars_titan.models.titans.financial import FinancialPredictor
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer


def data():
    reference, batch = setup()
    model = (
        FinancialPredictor(
            replace(reference.config, dual_neuromodulation=DualNeuromodulationConfig()),
            dtype=torch.float64,
        )
        .eval()
        .requires_grad_(False)
    )
    config = NeuromodulationSourceConfig()
    consumer = FrozenFinancialConsumer(model, neuromodulation_source=config)
    snapshot = NeuromodulationSource(config).snapshot(
        batch.flow_ids, prediction_at=batch.prediction_at[0], context_id="a" * 64
    )
    return consumer, batch, snapshot


def test_initial_snapshot_matches_the_explicit_zero_signal_and_can_select_a_block():
    consumer, batch, snapshot = data()
    model = consumer.predictor
    state = model.initial_state(batch.flow_ids)
    expected = model.prepare(batch, state, neuromodulation=torch.zeros((2, 2), dtype=torch.float64))
    actual = consumer.prepare(batch, state, context_id="a" * 64, neuromodulation_snapshot=snapshot)
    assert torch.equal(expected.point_predictions, actual.point_predictions)
    assert snapshot.select(batch.flow_ids[:1]).signals == snapshot.signals[:1]
    assert "mars_titan.models.titans.dual_neuromodulation" in consumer.identity()["implementation"]


@pytest.mark.parametrize(
    "kind", ["missing", "context", "state", "configuration", "time", "flows", "values"]
)
def test_foreign_or_malformed_snapshots_fail_before_prediction(kind):
    consumer, batch, snapshot = data()
    if kind == "missing":
        snapshot = None
    else:
        changes = dict(
            context=dict(context_id="b" * 64),
            state=dict(state_id="bad"),
            configuration=dict(configuration_id="f" * 64),
            time=dict(prediction_at=snapshot.prediction_at + 1),
            flows=dict(flow_ids=tuple(reversed(snapshot.flow_ids))),
            values=dict(signals=((2.0, 0.0),) * 2),
        )
        snapshot = replace(snapshot, **changes[kind])
    with pytest.raises(ValueError, match="N1"):
        consumer.prepare(
            batch,
            consumer.predictor.initial_state(batch.flow_ids),
            context_id="a" * 64,
            neuromodulation_snapshot=snapshot,
        )


def test_source_type_and_identity_cannot_change_after_registration():
    consumer, _, _ = data()
    consumer.neuromodulation_source = NeuromodulationSourceConfig(seed=1)
    with pytest.raises(ValueError, match="N1"):
        consumer.verify()
    consumer.neuromodulation_source = object()
    with pytest.raises(ValueError, match="N1"):
        consumer.verify()


def test_snapshot_selection_refuses_ambiguous_or_foreign_flows():
    _, _, snapshot = data()
    for flows in ([], [[]], ["unknown"], [snapshot.flow_ids[0]] * 2):
        with pytest.raises(ValueError):
            snapshot.select(flows)


def test_snapshot_views_keep_market_specific_signals_in_requested_order():
    _, _, snapshot = data()
    snapshot = replace(
        snapshot,
        flow_ids=("US/A", "CN/B", "US/C"),
        signals=((0.1, 0.2), (0.3, 0.4), (0.5, 0.6)),
    )
    view = snapshot.select(("US/C", "CN/B"))
    assert view.flow_ids == ("US/C", "CN/B")
    assert view.signals == ((0.5, 0.6), (0.3, 0.4))
    assert view.state_id == snapshot.state_id and view.context_id == snapshot.context_id


def test_source_cannot_be_attached_to_a_predictor_without_n1():
    consumer, _, _ = data()
    model = (
        FinancialPredictor(
            replace(consumer.predictor.config, dual_neuromodulation=None), dtype=torch.float64
        )
        .eval()
        .requires_grad_(False)
    )
    with pytest.raises(ValueError, match="N1"):
        FrozenFinancialConsumer(model, neuromodulation_source=consumer.neuromodulation_source)


def test_source_requires_a_registered_plain_session_binding():
    from mars_titan.memory import neuromodulation_session

    consumer, _, _ = data()
    with pytest.raises(ValueError, match="N1"):
        neuromodulation_session.configure(consumer, object())
