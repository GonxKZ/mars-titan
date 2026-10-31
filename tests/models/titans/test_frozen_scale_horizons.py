"""Padres congelados con F1, F5 o A4 (#749): el consumidor lee el punto como en la referencia."""

from dataclasses import replace

import pytest
import torch
from test_episodic_readout import CONTEXT, reader
from test_financial_adapter import setup
from test_frozen_financial import bank, consumer
from test_frozen_financial import frozen_backend as frozen_backend

from mars_titan.models.titans.auxiliary_horizons import AuxiliaryHorizons
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.laplace_scale import LaplaceScaleHead
from mars_titan.models.titans.opponent_heads import OpponentHeads

CHANGES = [
    dict(laplace_scale_head=LaplaceScaleHead()),
    dict(auxiliary_horizons=AuxiliaryHorizons()),
    dict(opponent_heads=OpponentHeads()),
]


@pytest.mark.parametrize(
    "change", CHANGES, ids=["laplace_scale", "auxiliary_horizons", "opponent_heads"]
)
@pytest.mark.parametrize("refinements", [None, 1])
def test_frozen_consumer_reads_the_median_like_any_other_parent(change, refinements):
    control, batch = setup()
    model = type(control)(replace(control.config, **change), dtype=torch.float64)
    model.eval().requires_grad_(False)
    extension = None
    if refinements is not None:
        extension = reader(refinements=refinements).eval().requires_grad_(False)
    engine = consumer(model, extension)
    engine.verify()
    state = model.initial_state(batch.flow_ids)
    snapshot = None if extension is None else bank(batch)
    result = engine.prepare(batch, state, snapshot=snapshot, context_id=CONTEXT)
    with torch.no_grad():
        if extension is None:
            expected = model.prepare(batch, state).point_predictions
            # La mediana del padre es la de la referencia con la misma semilla.
            reference = control.prepare(batch, control.initial_state(batch.flow_ids))
            torch.testing.assert_close(
                result.point_predictions, reference.point_predictions, rtol=0, atol=0
            )
        else:
            # El lector refina el estado y lo lee con la cabeza de la mediana, no con la escala.
            expected = model.head(result.readout.state).squeeze(-1)
    torch.testing.assert_close(result.point_predictions, expected, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_frozen_signature_separates_the_opponent_variant_from_its_control():
    control, _ = setup()
    model = type(control)(
        replace(control.config, opponent_heads=OpponentHeads()), dtype=torch.float64
    )
    consumer = FrozenFinancialConsumer(model.eval().requires_grad_(False))
    # Mismos pesos con el nivel del control: el cálculo congelado ya no es el declarado.
    model.output_objective.config = OpponentHeads(level=0.5)
    with pytest.raises(ValueError, match="cálculo"):
        consumer.verify()
