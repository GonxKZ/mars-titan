"""Declaración y entradas del control LightGBM con las mismas entradas de la criba (#746).

El ajuste, la etiqueta y los puntos de control son los del LightGBM de Qlib común y se prueban
en `test_qlib_lightgbm.py` con un LightGBM doble. Aquí solo se comprueban la declaración del
control y las características que recibe.
"""

import copy

import numpy as np
import pytest

from mars_titan.data.price_windows import CONTEXT_SESSIONS, PRICE_WINDOW_CHANNELS
from mars_titan.models.baselines import qlib_lightgbm as qlib
from mars_titan.models.baselines import same_inputs_lightgbm as control

WIDTHS = dict(news=3, charts=2, fundamentals=4, macro=5)
PRICES = CONTEXT_SESSIONS * len(PRICE_WINDOW_CHANNELS)


CONFIG = dict(
    identity=control.IDENTITY,
    source=qlib.QLIB_SOURCE,
    parameters=qlib.PARAMETERS,
    max_rounds=qlib.QLIB_ROUNDS,
    checkpoint_rounds=10,
    features=control.FEATURES,
    label=control.LABEL,
)


def declared():
    """Copia de la declaración que registra el protocolo, para cambiarla sin tocar otras."""
    return copy.deepcopy(CONFIG)


def batch(rows, *, seed=1):
    generator = np.random.default_rng(seed)
    inputs = {name: generator.normal(size=(rows, width)) for name, width in WIDTHS.items()}
    inputs["prices"] = generator.normal(size=(rows, CONTEXT_SESSIONS, len(PRICE_WINDOW_CHANNELS)))
    presence = generator.random((rows, 5)) > 0.3
    return inputs, presence


def test_the_declaration_keeps_the_qlib_parameters_and_their_changes():
    config = control.checked_config(declared())
    kept = {key: value for key, value in config["parameters"].items() if key not in qlib.CHANGES}
    assert kept == {
        key: value for key, value in qlib.QLIB_PARAMETERS.items() if key not in qlib.CHANGES
    }
    assert qlib.QLIB_PARAMETERS["num_threads"] == 20 and config["parameters"]["num_threads"] == 8
    assert config["parameters"]["deterministic"] is True
    assert config["max_rounds"] == 1000 and config["checkpoint_rounds"] == 10


@pytest.mark.parametrize(
    "change",
    [
        dict(identity="same_inputs_lightgbm_v2"),
        dict(max_rounds=500),
        dict(checkpoint_rounds=7),
        dict(checkpoint_rounds=0),
        dict(checkpoint_rounds=10.0),
        dict(features="alpha158"),
        dict(label="raw_residual"),
        dict(extra=True),
    ],
)
def test_a_changed_declaration_is_refused(change):
    with pytest.raises(ValueError, match="parámetros de Qlib"):
        control.checked_config(dict(declared(), **change))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("learning_rate", 0.1),
        ("lambda_l1", 0.0),
        ("num_threads", 20),
        ("deterministic", 1),
        ("max_depth", 8.0),
        ("seed", 42),
    ],
)
def test_a_changed_parameter_or_type_is_refused(key, value):
    config = declared()
    config["parameters"] = dict(config["parameters"], **{key: value})
    with pytest.raises(ValueError, match="parámetros de Qlib"):
        control.checked_config(config)


def test_a_changed_source_is_refused():
    config = declared()
    config["source"] = dict(config["source"], commit="0" * 40)
    with pytest.raises(ValueError, match="parámetros de Qlib"):
        control.checked_config(config)
    with pytest.raises(ValueError, match="parámetros de Qlib"):
        control.checked_config("same_inputs_lightgbm_v1")


def test_the_features_are_the_price_window_the_modalities_and_the_presence_in_order():
    inputs, presence = batch(4)
    matrix = control.features(inputs, presence)
    assert matrix.dtype == np.float32 and matrix.shape == (4, PRICES + sum(WIDTHS.values()) + 5)
    np.testing.assert_array_equal(
        matrix[:, :PRICES], inputs["prices"].reshape(4, -1).astype(np.float32)
    )
    start = PRICES
    for name in ("news", "charts", "fundamentals", "macro"):
        width = WIDTHS[name]
        np.testing.assert_array_equal(
            matrix[:, start : start + width], inputs[name].astype(np.float32)
        )
        start += width
    np.testing.assert_array_equal(matrix[:, start:], presence.astype(np.float32))
    # El aplanado sigue la sesión y después el canal.
    assert matrix[1, len(PRICE_WINDOW_CHANNELS) + 2] == np.float32(inputs["prices"][1, 1, 2])


def test_features_outside_the_contract_are_refused():
    inputs, presence = batch(3)
    with pytest.raises(ValueError, match="cuatro modalidades"):
        control.features({k: v for k, v in inputs.items() if k != "macro"}, presence)
    with pytest.raises(ValueError, match="64 sesiones"):
        control.features(dict(inputs, prices=inputs["prices"][:, 1:]), presence)
    with pytest.raises(ValueError, match="un bit por modalidad"):
        control.features(inputs, presence.astype(np.float32))
    with pytest.raises(ValueError, match="un bit por modalidad"):
        control.features(inputs, presence[:, :4])
    with pytest.raises(ValueError, match="una fila por decisión"):
        control.features(dict(inputs, news=inputs["news"][:2]), presence)
    broken = dict(inputs, charts=inputs["charts"].copy())
    broken["charts"][0, 0] = np.inf
    with pytest.raises(ValueError, match="no finitos"):
        control.features(broken, presence)
