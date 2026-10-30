"""Control LightGBM de la criba (#746) con un LightGBM doble que no ajusta nada.

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

from mars_titan.data.price_windows import CONTEXT_SESSIONS, PRICE_WINDOW_CHANNELS
from mars_titan.evaluation import scale_diagnostics as sd
from mars_titan.evaluation.forecast_panel import DAY_MICROSECONDS, ForecastPanel
from mars_titan.models.baselines import same_inputs_lightgbm as control

WIDTHS = dict(news=3, charts=2, fundamentals=4, macro=5)
PRICES = CONTEXT_SESSIONS * len(PRICE_WINDOW_CHANNELS)


CONFIG = dict(
    identity=control.IDENTITY,
    source=control.QLIB_SOURCE,
    parameters=control.PARAMETERS,
    max_rounds=control.QLIB_ROUNDS,
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
    qlib = {key: value for key, value in config["parameters"].items() if key not in control.CHANGES}
    assert qlib == {
        key: value for key, value in control.QLIB_PARAMETERS.items() if key not in control.CHANGES
    }
    assert control.QLIB_PARAMETERS["num_threads"] == 20 and config["parameters"]["num_threads"] == 8
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
    assert set(control.PARAMETERS) | {"seed"} <= known
    # `subsample` es un alias de `bagging_fraction`, que no actúa sin `bagging_freq`.
    assert "subsample" in aliases["bagging_fraction"] and "bagging_freq" not in control.PARAMETERS


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
    labels, kept = control.session_labels(data)
    assert kept.tolist() == [True] * 10 + [False] * 5 + [True] * 15
    for start in (0, 5, 15, 20, 25):
        target = data.target[start : start + 5]
        # La zscore de Qlib: (x − media) / desviación de pandas, con ddof = 1.
        np.testing.assert_allclose(
            labels[start : start + 5], (target - target.mean()) / target.std(ddof=1), rtol=1e-12
        )
    assert np.all(labels[~kept] == 0.0)
    single = panel(3, 1)
    labels, kept = control.session_labels(single)
    assert not kept.any() and np.all(labels == 0.0)


@pytest.mark.parametrize("target", [(-1e308, 1e308), (1e308, 1e308)])
def test_label_normalization_refuses_statistics_that_overflow(target):
    base = panel(1, 2)
    data = ForecastPanel.from_columns(
        base.row_id, np.array(["US", "US"]), base.prediction_at, np.array(target), np.zeros(2)
    )
    with pytest.raises(ValueError, match="normalización.*finitas"):
        control.session_labels(data)


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
    calls = dict(datasets=[], trains=[], iterations=[])
    module = types.ModuleType("lightgbm")

    def dataset(values, label=None, free_raw_data=True):
        calls["datasets"].append((np.array(values), np.array(label)))
        return ("dataset", len(values))

    def train(params, data, num_boost_round, callbacks):
        weights = np.linspace(-1.0, 1.0, len(calls["datasets"][-1][0][0]))
        booster = Booster(weights)
        calls["trains"].append((params, data, num_boost_round, booster))
        for iteration in range(num_boost_round):
            calls["iterations"].append(iteration)
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
    booster, series = control.fit(train, validation, declared(), 43, min_assets=3)
    ((params, data, rounds, recorded),) = lightgbm["trains"]
    assert params == dict(control.PARAMETERS, seed=43) and rounds == 1000 and booster is recorded
    # Las filas de la sesión constante no tienen etiqueta y no entran.
    labels, kept = control.session_labels(train.panel)
    ((values, label),) = lightgbm["datasets"]
    np.testing.assert_array_equal(values, train.features[kept])
    np.testing.assert_array_equal(label, labels[kept])
    assert data == ("dataset", 42)
    # Cada diez rondas se puntúa la validación con la predicción que emitirá `predict`.
    assert booster.calls == list(range(10, 1001, 10))
    assert len(series) == 100
    for position in (0, 49, 99):
        expected = control.predict(booster, validation.features, 10 * (position + 1))
        values_, defined = sd.session_correlations(validation.panel, expected, min_assets=3)
        np.testing.assert_array_equal(series[position][0], values_)
        np.testing.assert_array_equal(series[position][1], defined)


def test_the_fit_refuses_an_invalid_seed_or_a_fit_without_cross_sections(lightgbm):
    validation = Rows(panel(5, 6, seed=5), 6)
    with pytest.raises(ValueError, match="semilla de LightGBM"):
        control.fit(Rows(panel(4, 6), 1), validation, declared(), 2**31, min_assets=3)
    with pytest.raises(ValueError, match="semilla de LightGBM"):
        control.fit(Rows(panel(4, 6), 1), validation, declared(), 42.0, min_assets=3)
    with pytest.raises(ValueError, match="sección cruzada con desviación"):
        control.fit(Rows(panel(4, 1), 1), validation, declared(), 42, min_assets=3)
    assert lightgbm["trains"] == []


def test_the_fit_checks_its_budget_before_the_dataset_and_after_each_round(lightgbm):
    calls = []

    def check_stop():
        calls.append(lightgbm["iterations"][-1] if lightgbm["iterations"] else None)
        if len(calls) == 4:
            raise RuntimeError("parada técnica")

    with pytest.raises(RuntimeError, match="parada técnica"):
        control.fit(
            Rows(panel(4, 6), 1),
            Rows(panel(4, 6, seed=2), 2),
            declared(),
            42,
            min_assets=3,
            check_stop=check_stop,
        )
    assert calls == [None, 0, 1, 2] and len(lightgbm["trains"]) == 1
    assert lightgbm["trains"][0][3].calls == []


def test_an_expired_budget_prevents_constructing_the_dataset(lightgbm):
    def check_stop():
        raise RuntimeError("presupuesto agotado")

    with pytest.raises(RuntimeError, match="presupuesto agotado"):
        control.fit(
            Rows(panel(4, 6), 1),
            Rows(panel(4, 6, seed=2), 2),
            declared(),
            42,
            min_assets=3,
            check_stop=check_stop,
        )
    assert not lightgbm["datasets"] and not lightgbm["trains"]


def test_the_fit_fails_if_lightgbm_skips_a_checkpoint(lightgbm, monkeypatch):
    original = sys.modules["lightgbm"].train

    def short(params, data, num_boost_round, callbacks):
        return original(params, data, num_boost_round - 1, callbacks)

    monkeypatch.setattr(sys.modules["lightgbm"], "train", short)
    with pytest.raises(ValueError, match="todos sus puntos de control"):
        control.fit(
            Rows(panel(4, 6), 1), Rows(panel(4, 6, seed=2), 2), declared(), 42, min_assets=3
        )


def test_a_booster_is_saved_with_every_round_and_loaded_without_fitting(lightgbm):
    class Fitted:
        def model_to_string(self, num_iteration=None):
            return f"rondas {num_iteration}"

    # Con num_iteration ≤ 0, LightGBM guarda todas las rondas.
    assert control.model_text(Fitted()) == "rondas -1"
    assert control.load("rondas -1") == ("booster", "rondas -1")
    assert lightgbm["datasets"] == [] and lightgbm["trains"] == []


def test_a_prediction_that_is_not_finite_or_misses_rows_is_refused():
    values = np.ones((4, 2), dtype=np.float32)
    booster = Booster(np.array([1.0, 1.0]))
    assert control.predict(booster, values, 10).dtype == np.float64
    booster.weights = np.array([np.nan, 1.0])
    with pytest.raises(ValueError, match="predicción finita por fila"):
        control.predict(booster, values, 10)

    class Short(Booster):
        def predict(self, values, num_iteration=None):
            return np.zeros(len(values) - 1)

    with pytest.raises(ValueError, match="predicción finita por fila"):
        control.predict(Short(None), values, 10)
