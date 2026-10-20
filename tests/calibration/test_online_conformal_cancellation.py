"""Cancelar cohortes incompletas sin convertir ausencias en resultados financieros."""

import pytest
from test_online_conformal import cohort, start

from mars_titan.calibration.online_conformal import OnlineConformal


def test_cancel_discards_only_the_oldest_cohort_without_learning_from_missing_labels():
    model = start(groups=("US", "CN"))
    model.emit("US", 10, cohort(2), 20)
    model.emit("US", 11, cohort(1), 21)
    before = model.export()
    result = model.cancel("US", 10, at=20)
    assert result == dict(
        status="cancelled", group="US", prediction_at=10, matures_at=20, at=20, rows=2
    )
    after = model.export()
    assert after["markets"]["US"]["intervals"] == before["markets"]["US"]["intervals"]
    assert after["markets"]["US"]["matured"] == 0
    assert after["markets"]["US"]["clock"] == 20
    assert after["markets"]["CN"] == before["markets"]["CN"]
    assert model.pending("US") == [(11, 21)]
    restored = OnlineConformal.restore(after)
    assert restored.state_sha256() == model.state_sha256()
    assert restored.mature("US", 11, [0.0], at=21) == model.mature("US", 11, [0.0], at=21)


@pytest.mark.parametrize(
    "group,prediction_at,at",
    [("US", 10, 19), ("US", 11, 21), ("US", 9, 21), ("CN", 10, 21), ("US", True, 21)],
)
def test_invalid_cancellation_keeps_the_complete_state(group, prediction_at, at):
    model = start()
    model.emit("US", 10, cohort(2), 20)
    model.emit("US", 11, cohort(1), 21)
    before = model.state_sha256()
    with pytest.raises(ValueError):
        model.cancel(group, prediction_at, at=at)
    assert model.state_sha256() == before


def test_cancel_cannot_be_repeated_or_move_the_market_clock_backwards():
    model = start()
    model.emit("US", 10, cohort(1), 20)
    model.emit("US", 11, cohort(1), 21)
    model.cancel("US", 10, at=30)
    before = model.state_sha256()
    for prediction_at, at in ((10, 30), (11, 29)):
        with pytest.raises(ValueError):
            model.cancel("US", prediction_at, at=at)
        assert model.state_sha256() == before
