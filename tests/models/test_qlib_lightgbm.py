"""LightGBM de Qlib común a los controles tabulares, con un LightGBM doble que no ajusta nada.

`lightgbm.train` se sustituye por un doble que recorre las rondas, llama a las devoluciones de
llamada y devuelve un booster cuya predicción es una función fija de las características y de
las rondas. Así se comprueban los parámetros, las filas, las etiquetas y la puntuación de cada
punto de control sin estimar ningún parámetro.
"""

import copy
import dataclasses
import inspect
import sys
import types

import numpy as np
import pytest

from mars_titan.evaluation import scale_diagnostics as sd
from mars_titan.evaluation.forecast_panel import DAY_MICROSECONDS, ForecastPanel
from mars_titan.models.baselines import qlib_lightgbm as qlib

# Un control técnico con su propia identidad. El ajuste no mira la identidad, solo los campos
# que usa, y cada control comprueba la suya con `checked_config`.
CONFIG = dict(
    identity="technical_control_v1",
    source=qlib.QLIB_SOURCE,
    parameters=qlib.PARAMETERS,
    max_rounds=qlib.QLIB_ROUNDS,
    checkpoint_rounds=10,
    features="technical_features_v1",
    label="session_zscore_of_the_target",
)
DECLARED = dict(
    identity="technical_control_v1",
    features="technical_features_v1",
    label="session_zscore_of_the_target",
)


def declared():
    return copy.deepcopy(CONFIG)


def test_a_control_is_accepted_only_with_its_own_declaration():
    assert qlib.checked_config(declared(), **DECLARED) == CONFIG
    for key in DECLARED:
        with pytest.raises(ValueError, match="parámetros de Qlib"):
            qlib.checked_config(declared(), **dict(DECLARED, **{key: "otro"}))


@pytest.mark.parametrize(
    "change",
    [
        dict(max_rounds=500),
        dict(checkpoint_rounds=7),
        dict(checkpoint_rounds=0),
        dict(checkpoint_rounds=2000),
        dict(checkpoint_rounds=10.0),
        dict(source=dict(qlib.QLIB_SOURCE, commit="0" * 40)),
        dict(parameters=dict(qlib.PARAMETERS, learning_rate=0.1)),
        dict(parameters=dict(qlib.PARAMETERS, max_depth=8.0)),
        dict(extra=True),
    ],
)
def test_a_changed_training_field_is_refused_by_the_declaration_and_by_the_fit(change, lightgbm):
    config = dict(declared(), **change)
    with pytest.raises(ValueError, match="parámetros de Qlib"):
        qlib.checked_config(config, **DECLARED)
    with pytest.raises(ValueError, match="parámetros de Qlib"):
        qlib.fit(None, None, config, 42, min_assets=3)
    assert lightgbm["trains"] == []


def test_the_pinned_lightgbm_offers_what_the_control_uses():
    """API real del LightGBM fijado, leída sin construir ningún conjunto ni ajustar nada."""
    library = pytest.importorskip("lightgbm")
    assert library.__version__ == "4.7.0"
    assert {"params", "train_set", "num_boost_round", "callbacks"} <= set(
        inspect.signature(library.train).parameters
    )
    assert {"model", "iteration"} <= {
        field.name for field in dataclasses.fields(library.callback.CallbackEnv)
    }
    assert "num_iteration" in inspect.signature(library.Booster.predict).parameters
    assert "num_iteration" in inspect.signature(library.Booster.model_to_string).parameters
    assert "model_str" in inspect.signature(library.Booster).parameters
    assert {"label", "free_raw_data"} <= set(inspect.signature(library.Dataset).parameters)
    aliases = library.basic._ConfigAliases._get_all_param_aliases()
    known = set(aliases) | {alias for names in aliases.values() for alias in names}
    assert set(qlib.PARAMETERS) | {"seed"} <= known
    # `subsample` es un alias de `bagging_fraction`, que no actúa sin `bagging_freq`.
    assert "subsample" in aliases["bagging_fraction"] and "bagging_freq" not in qlib.PARAMETERS


def panel(sessions, assets, *, seed=0, constant=()):
    """Panel de `sessions` días con `assets` activos y residuos aleatorios."""
    generator = np.random.default_rng(seed)
    days = 1_700_000_000_000_000 + DAY_MICROSECONDS * np.arange(sessions)
    times = np.repeat(days, assets)
    names = np.tile([f"US/A{i}" for i in range(assets)], sessions)
    target = generator.normal(scale=0.02, size=len(times))
    for session in constant:
        target[session * assets : (session + 1) * assets] = 0.01
    ids = [f"{name}/{at}" for name, at in zip(names, times, strict=True)]
    return ForecastPanel.from_columns(
        ids, np.full(len(times), "US"), times, target, np.zeros(len(times))
    )


def test_the_labels_are_the_qlib_cross_sectional_zscore_of_the_residual():
    data = panel(6, 5, constant=(2,))
    labels, kept = qlib.session_labels(data)
    assert kept.tolist() == [True] * 10 + [False] * 5 + [True] * 15
    for start in (0, 5, 15, 20, 25):
        target = data.target[start : start + 5]
        # La zscore de Qlib: (x − media) / desviación de pandas, con ddof = 1.
        np.testing.assert_allclose(
            labels[start : start + 5], (target - target.mean()) / target.std(ddof=1), rtol=1e-12
        )
    assert np.all(labels[~kept] == 0.0)
    single = panel(3, 1)
    labels, kept = qlib.session_labels(single)
    assert not kept.any() and np.all(labels == 0.0)


class Booster:
    """Predicción fija: la primera característica por un factor que crece con las rondas."""

    def __init__(self, weights):
        self.weights = weights
        self.calls = []

    def predict(self, values, num_iteration=None):
        self.calls.append(num_iteration)
        return values @ self.weights * (1.0 + 0.01 * num_iteration)


@pytest.fixture
def lightgbm(monkeypatch, learning_doubles):
    """Módulo `lightgbm` doble que registra lo que recibe y nunca ajusta."""
    calls = dict(datasets=[], trains=[])
    module = types.ModuleType("lightgbm")

    def dataset(values, label=None, free_raw_data=True):
        calls["datasets"].append((np.array(values), np.array(label)))
        return ("dataset", len(values))

    def train(params, data, num_boost_round, callbacks):
        weights = np.linspace(-1.0, 1.0, len(calls["datasets"][-1][0][0]))
        booster = Booster(weights)
        calls["trains"].append((params, data, num_boost_round, booster))
        for iteration in range(num_boost_round):
            for callback in callbacks:
                callback(types.SimpleNamespace(model=booster, iteration=iteration))
        return booster

    module.Dataset, module.train = dataset, train
    module.Booster = lambda model_str: ("booster", model_str)
    monkeypatch.setitem(sys.modules, "lightgbm", module)
    return calls


class Rows:
    def __init__(self, data, seed):
        self.panel = data
        self.features = np.random.default_rng(seed).normal(size=(data.rows, 6)).astype(np.float32)


def test_the_fit_passes_the_qlib_parameters_and_scores_every_checkpoint(lightgbm):
    train = Rows(panel(8, 6, seed=3, constant=(4,)), 4)
    validation = Rows(panel(5, 6, seed=5), 6)
    booster, series = qlib.fit(train, validation, declared(), 43, min_assets=3)
    ((params, data, rounds, recorded),) = lightgbm["trains"]
    assert params == dict(qlib.PARAMETERS, seed=43) and rounds == 1000 and booster is recorded
    # Las filas de la sesión constante no tienen etiqueta y no entran.
    labels, kept = qlib.session_labels(train.panel)
    ((values, label),) = lightgbm["datasets"]
    np.testing.assert_array_equal(values, train.features[kept])
    np.testing.assert_array_equal(label, labels[kept])
    assert data == ("dataset", 42)
    # Cada diez rondas se puntúa la validación con la predicción que emitirá `predict`.
    assert booster.calls == list(range(10, 1001, 10))
    assert len(series) == 100
    for position in (0, 49, 99):
        expected = qlib.predict(booster, validation.features, 10 * (position + 1))
        values_, defined = sd.session_correlations(validation.panel, expected, min_assets=3)
        np.testing.assert_array_equal(series[position][0], values_)
        np.testing.assert_array_equal(series[position][1], defined)


def test_the_fit_refuses_an_invalid_seed_or_a_fit_without_cross_sections(lightgbm):
    validation = Rows(panel(5, 6, seed=5), 6)
    with pytest.raises(ValueError, match="semilla de LightGBM"):
        qlib.fit(Rows(panel(4, 6), 1), validation, declared(), 2**31, min_assets=3)
    with pytest.raises(ValueError, match="semilla de LightGBM"):
        qlib.fit(Rows(panel(4, 6), 1), validation, declared(), 42.0, min_assets=3)
    with pytest.raises(ValueError, match="sección cruzada con desviación"):
        qlib.fit(Rows(panel(4, 1), 1), validation, declared(), 42, min_assets=3)
    assert lightgbm["trains"] == []


def test_the_fit_fails_if_lightgbm_skips_a_checkpoint(lightgbm, monkeypatch):
    original = sys.modules["lightgbm"].train

    def short(params, data, num_boost_round, callbacks):
        return original(params, data, num_boost_round - 1, callbacks)

    monkeypatch.setattr(sys.modules["lightgbm"], "train", short)
    with pytest.raises(ValueError, match="todos sus puntos de control"):
        qlib.fit(Rows(panel(4, 6), 1), Rows(panel(4, 6, seed=2), 2), declared(), 42, min_assets=3)


def test_a_booster_is_saved_with_every_round_and_loaded_without_fitting(lightgbm):
    class Fitted:
        def model_to_string(self, num_iteration=None):
            return f"rondas {num_iteration}"

    # Con num_iteration ≤ 0, LightGBM guarda todas las rondas.
    assert qlib.model_text(Fitted()) == "rondas -1"
    assert qlib.load("rondas -1") == ("booster", "rondas -1")
    assert lightgbm["datasets"] == [] and lightgbm["trains"] == []


def test_a_prediction_that_is_not_finite_or_misses_rows_is_refused():
    values = np.ones((4, 2), dtype=np.float32)
    booster = Booster(np.array([1.0, 1.0]))
    assert qlib.predict(booster, values, 10).dtype == np.float64
    booster.weights = np.array([np.nan, 1.0])
    with pytest.raises(ValueError, match="predicción finita por fila"):
        qlib.predict(booster, values, 10)

    class Short(Booster):
        def predict(self, values, num_iteration=None):
            return np.zeros(len(values) - 1)

    with pytest.raises(ValueError, match="predicción finita por fila"):
        qlib.predict(Short(None), values, 10)
