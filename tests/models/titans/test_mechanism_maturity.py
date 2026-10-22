"""Un calendario de etiquetas ausente no elimina una predicción financiera válida."""

from dataclasses import replace

import numpy as np
import pytest
from test_context_selection import DEVICES, batch_on
from test_episodic_readout import CONTEXT
from test_frozen_financial import frozen_backend as frozen_backend
from test_mechanism_catalog import available
from test_mechanism_consumer import calibration, consumer

from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismCatalog


def test_unknown_label_calendar_is_distinct_from_a_pending_capacity_failure():
    availability = replace(available(), online_maturity=False)
    plan = MechanismCatalog().resolve(
        MechanismAction("parent", "k4cc", "online"), availability, rows=2
    )
    assert plan.applied == MechanismAction("parent", "k1", "none")
    assert plan.reasons == ("unavailable_label_calendar",)


@pytest.mark.parametrize("device", DEVICES)
def test_mixed_market_calendar_preserves_rows_and_admits_only_the_known_online_cohort(device):
    engine = consumer(
        device,
        reader=True,
        calibrator=calibration(max_rows=1),
        action=MechanismAction(route="k4cc", calibration="online"),
    )
    batch = batch_on(engine.predictor, ("US/AAA", "CN/600000.SS"), 0)
    at = batch.prediction_at[0]
    event = engine.begin(
        engine.initial_state(),
        flow_ids=batch.flow_ids,
        cutoff=at,
        context_id=CONTEXT,
        matures_at=(None, at + 10),
    )
    prepared = engine.prepare(batch, engine.predictor.initial_state(batch.flow_ids), event)
    result = engine.finish(prepared.event)
    assert result.flow_ids == tuple(sorted(batch.flow_ids))
    assert np.isfinite(result.point_predictions).all()
    us, cn = (
        {row["flow_id"].split("/")[0]: row for row in result.rows}[name] for name in ("US", "CN")
    )
    assert us["reasons"] == ("unavailable_label_calendar",)
    assert us["emitted_route"] == "k1" and us["calibration"] == "none"
    assert cn["emitted_route"] == "k4cc" and cn["calibration"] == "online" and not cn["reasons"]
    assert result.state.calibration.pending_rows == 1


@pytest.mark.parametrize("mode", ["none", "static"])
def test_non_online_actions_keep_their_route_without_a_future_label_calendar(mode):
    engine = consumer(
        reader=True,
        calibrator=calibration(),
        action=MechanismAction(route="k4cc", calibration=mode),
    )
    batch = batch_on(engine.predictor, ("US/AAA",), 0)
    event = engine.begin(
        engine.initial_state(),
        flow_ids=batch.flow_ids,
        cutoff=batch.prediction_at[0],
        context_id=CONTEXT,
        matures_at=(None,),
    )
    prepared = engine.prepare(batch, engine.predictor.initial_state(batch.flow_ids), event)
    result = engine.finish(prepared.event)
    assert result.rows[0]["emitted_route"] == "k4cc"
    assert not result.rows[0]["reasons"]
