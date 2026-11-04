"""Consumo congelado de las variantes por sesión y rechazo de combinaciones sin pérdida propia."""

import pytest
import torch
from test_episodic_readout import CONTEXT, reader
from test_session_rank_objective import OBJECTIVE, model, session_batch

from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.session_ranks import SESSION_RANK_INPUTS


@pytest.fixture(autouse=True)
def explicit_backend():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.mark.parametrize("objective", [None, OBJECTIVE])
def test_frozen_complete_session_preserves_the_parent_prediction(objective):
    predictor = model(session_rank_inputs=SESSION_RANK_INPUTS, session_ic_objective=objective)
    predictor.eval().requires_grad_(False)
    consumer = FrozenFinancialConsumer(predictor)
    batch = session_batch(8)
    state = predictor.initial_state(batch.flow_ids)
    expected = predictor.prepare(batch, state, session_complete=True)
    actual = consumer.prepare(batch, state, context_id=CONTEXT, session_complete=True)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    assert actual.next_state.observed_steps.tolist() == [1] * 8
    with pytest.raises(ValueError, match="sesión completa"):
        consumer.prepare(batch, state, context_id=CONTEXT)


def test_frozen_consumer_does_not_reinterpret_an_ic_score_as_a_return_in_the_reader():
    predictor = model(session_ic_objective=OBJECTIVE).eval().requires_grad_(False)
    extension = reader(refinements=1).eval().requires_grad_(False)
    with pytest.raises(ValueError, match="objetivo de salida"):
        FrozenFinancialConsumer(predictor, readout=extension)


def test_frozen_signature_covers_the_output_objective_configuration():
    predictor = model(session_ic_objective=OBJECTIVE).eval().requires_grad_(False)
    consumer = FrozenFinancialConsumer(predictor)
    predictor.output_objective.config = type(OBJECTIVE)(minimum_rows=5)
    with pytest.raises(ValueError, match="cálculo"):
        consumer.verify()
