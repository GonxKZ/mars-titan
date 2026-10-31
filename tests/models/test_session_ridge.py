"""Ridge del techo de señal (#747) con un Ridge doble de scikit-learn que no ajusta nada.

`sklearn.linear_model.Ridge` se sustituye por un doble cuya predicción es una columna fija de
las características elegida por su alpha. Así se comprueban las filas, la etiqueta, la elección
del alpha y la protección sin estimar ningún parámetro.
"""

import copy

import numpy as np
import pytest

from mars_titan.evaluation import scale_diagnostics as sd
from mars_titan.evaluation.forecast_panel import DAY_MICROSECONDS, ForecastPanel
from mars_titan.models.baselines import qlib_lightgbm, session_ridge
from mars_titan.training.learning_hold import LearningHoldError

CONFIG = dict(
    identity="session_ridge_v1",
    alphas=[1.0, 10.0, 100.0],
    solver="cholesky",
    features="technical_features_v1",
    label="session_zscore_of_the_target",
)
DECLARED = dict(features="technical_features_v1", label="session_zscore_of_the_target")


def declared(**changes):
    return dict(copy.deepcopy(CONFIG), **changes)


def test_the_declaration_is_accepted_only_with_its_own_inputs():
    assert session_ridge.checked_config(declared(), **DECLARED) == CONFIG
    for key in DECLARED:
        with pytest.raises(ValueError, match="solución de Cholesky"):
            session_ridge.checked_config(declared(), **dict(DECLARED, **{key: "otro"}))


@pytest.mark.parametrize(
    "change",
    [
        dict(identity="session_ridge_v2"),
        dict(alphas=[]),
        dict(alphas=[10.0, 1.0]),
        dict(alphas=[1.0, 1.0]),
        dict(alphas=[0.0, 1.0]),
        dict(alphas=[1, 10.0]),
        dict(alphas=[1.0, float("inf")]),
        dict(alphas="1.0"),
        dict(solver="svd"),
        dict(extra=True),
    ],
)
def test_a_changed_declaration_is_refused_by_the_declaration_and_by_the_fit(change, ridge):
    with pytest.raises(ValueError, match="solución de Cholesky"):
        session_ridge.checked_config(declared(**change), **DECLARED)
    with pytest.raises(ValueError, match="solución de Cholesky"):
        session_ridge.fit(None, None, declared(**change), min_assets=3)
    assert ridge == []


def panel(sessions, assets, *, seed=0, constant=()):
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


class Rows:
    """Tres columnas: ruido, el objetivo con poco ruido y el objetivo con el signo cambiado."""

    def __init__(self, data, seed):
        generator = np.random.default_rng(seed)
        noise = generator.normal(size=(data.rows, 3))
        self.panel = data
        self.features = np.column_stack(
            [noise[:, 0], data.target + 0.001 * noise[:, 1], -data.target + 0.001 * noise[:, 2]]
        )


@pytest.fixture
def ridge(monkeypatch, learning_doubles):
    """`Ridge` doble que predice una columna según su alpha: 1, 10 y 100 dan la 0, la 1 y la 2."""
    import sklearn.linear_model

    calls = []
    column = {1.0: 0, 10.0: 1, 100.0: 2, 1000.0: 1}

    class Ridge:
        def __init__(self, *, alpha, fit_intercept, solver):
            self.options = dict(alpha=alpha, fit_intercept=fit_intercept, solver=solver)

        def fit(self, values, labels):
            calls.append((self.options, np.array(values), np.array(labels)))
            return self

        def predict(self, values):
            return np.asarray(values)[:, column[self.options["alpha"]]]

    monkeypatch.setattr(sklearn.linear_model, "Ridge", Ridge)
    return calls


def test_the_fit_uses_the_session_labels_and_keeps_the_best_validation_alpha(ridge):
    train = Rows(panel(6, 5, seed=1, constant=(2,)), 2)
    validation = Rows(panel(4, 5, seed=3), 4)
    model, chosen = session_ridge.fit(train, validation, declared(), min_assets=3)
    labels, kept = qlib_lightgbm.session_labels(train.panel)
    assert [options for options, _, _ in ridge] == [
        dict(alpha=alpha, fit_intercept=True, solver="cholesky") for alpha in (1.0, 10.0, 100.0)
    ]
    for _, values, label in ridge:
        # Las filas de la sesión constante no tienen etiqueta y no entran.
        np.testing.assert_array_equal(values, train.features[kept])
        np.testing.assert_array_equal(label, labels[kept])
    assert chosen["alpha"] == 10.0 and model.options["alpha"] == 10.0
    assert chosen["fit_rows"] == int(kept.sum()) == 25
    for row, (alpha, column) in zip(
        chosen["scores"], ((1.0, 0), (10.0, 1), (100.0, 2)), strict=True
    ):
        values, defined = sd.session_correlations(
            validation.panel, validation.features[:, column], min_assets=3
        )
        assert row["alpha"] == alpha and row["sessions"] == 4
        assert row["validation_ic"] == pytest.approx(values[defined].mean(), rel=1e-12)
    assert chosen["validation_ic"] == chosen["scores"][1]["validation_ic"] > 0.99


def test_a_tie_keeps_the_largest_alpha(ridge):
    train, validation = Rows(panel(4, 5, seed=1), 2), Rows(panel(3, 5, seed=3), 4)
    # Con 10 y 1000 el doble predice la misma columna, así que empatan.
    _, chosen = session_ridge.fit(
        train, validation, declared(alphas=[1.0, 10.0, 1000.0]), min_assets=3
    )
    assert chosen["alpha"] == 1000.0
    assert chosen["scores"][1]["validation_ic"] == chosen["scores"][2]["validation_ic"]


def test_a_fit_without_cross_sections_or_validation_sessions_is_refused(ridge):
    validation = Rows(panel(3, 5, seed=3), 4)
    with pytest.raises(ValueError, match="sección cruzada con desviación"):
        session_ridge.fit(Rows(panel(4, 1), 1), validation, declared(), min_assets=3)
    assert ridge == []
    with pytest.raises(ValueError, match="no define el IC"):
        session_ridge.fit(
            Rows(panel(4, 5), 1), Rows(panel(3, 2, seed=3), 4), declared(), min_assets=3
        )


def test_the_hold_stops_the_fit_before_reading_anything():
    with pytest.raises(LearningHoldError, match="Bloqueo de aprendizaje vigente"):
        session_ridge.fit(None, None, None, min_assets=3)


def test_a_prediction_that_is_not_finite_or_misses_rows_is_refused():
    class Model:
        def __init__(self, result):
            self.result = result

        def predict(self, values):
            return self.result

    values = np.ones((3, 2))
    assert session_ridge.predict(Model(np.arange(3)), values).dtype == np.float64
    for result in (np.array([1.0, np.nan, 2.0]), np.zeros(2), np.zeros((3, 1))):
        with pytest.raises(ValueError, match="predicción finita por fila"):
            session_ridge.predict(Model(result), values)
