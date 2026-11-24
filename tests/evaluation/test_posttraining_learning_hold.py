"""El bloqueo de aprendizaje detiene los ajustes del post-entrenamiento antes de abrir o calcular.

Cada prueba demuestra una guarda del inventario (`tests/training/learning_hold_inventory.py`): con
la protección vigente, la función rechaza con `LearningHoldError` aunque sus argumentos no existan.
"""

import numpy as np
import pytest

from mars_titan.evaluation import enhanced_posttraining as enhanced
from mars_titan.evaluation import factor_neutralization as neutralization
from mars_titan.evaluation import online_aggregator, reliability, tool_arbiters
from mars_titan.training.learning_hold import LearningHoldError

ORIGIN = dict(wait_hours=0, poll=1, log=lambda text: None)


def _origin(tmp_path, organs):
    return (
        tmp_path / "no-existe.json",
        tmp_path / "estado",
        tmp_path / "referencia",
        organs,
        2013,
        tmp_path / "salida",
    )


def test_the_hold_stops_the_reliability_measure_before_reading_or_writing(tmp_path, learning_hold):
    learning_hold(False)
    with pytest.raises(LearningHoldError):
        reliability.measure(*_origin(tmp_path, tmp_path / "organos"), **ORIGIN)
    assert not (tmp_path / "salida").exists()


def test_the_hold_stops_the_tool_arbiters_before_reading_or_writing(tmp_path, learning_hold):
    learning_hold(False)
    with pytest.raises(LearningHoldError):
        tool_arbiters.measure(*_origin(tmp_path, {42: tmp_path / "organos"}), **ORIGIN)
    assert not (tmp_path / "salida").exists()


def test_the_hold_stops_the_online_aggregation_before_loading_any_model(tmp_path, learning_hold):
    learning_hold(False)
    with pytest.raises(LearningHoldError):
        online_aggregator.run_aggregation({"modelo": tmp_path / "no-existe"})


def test_the_hold_stops_the_multiscale_asset_memory_fit(learning_hold):
    learning_hold(False)
    with pytest.raises(LearningHoldError):
        enhanced.MultiScaleAssetMemory.fit_calibration(
            np.zeros(4), np.zeros(4), np.zeros((4, 2)), [(0, 4)]
        )


def test_the_hold_stops_the_shrinkage_icir_weights(learning_hold):
    learning_hold(False)
    with pytest.raises(LearningHoldError):
        enhanced.shrinkage_icir_weights(np.zeros((3, 2)), np.zeros((3, 2, 2)))


def test_the_hold_stops_the_factor_neutralizer_fit(learning_hold):
    learning_hold(False)
    with pytest.raises(LearningHoldError):
        neutralization.CausalFactorNeutralizer.fit_calibration(
            np.zeros(4), np.zeros(4), np.zeros((4, 1)), [(0, 4)]
        )
