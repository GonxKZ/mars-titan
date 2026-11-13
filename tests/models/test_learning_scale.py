"""Escalas de aprendizaje de #836: rendimientos por sesión, estadísticas del ajuste y escalador.

Las pruebas usan arreglos técnicos en CPU. No construyen optimizadores ni ajustan nada.
"""

import math

import numpy as np
import pytest
import torch

from mars_titan.models.learning_scale import (
    INITIAL_INTERVAL,
    MIN_ROWS,
    MODALITIES,
    RELATIVE_FLOOR,
    STD_FRACTION,
    TRANSFORM,
    LearningScale,
    LearningScaler,
    fit_statistics,
    parse_learning_scale,
    price_returns,
    statistics_digest,
)

DIMENSIONS = dict(prices=6, news=4, charts=3, fundamentals=6, macro=9)
ROWS = 400


def anchored(rows=3, sessions=8, *, channels=6, seed=5):
    """Ventana anclada coherente: cierres acumulados y apertura, máximo y mínimo alrededor."""
    generator = np.random.default_rng(seed)
    close = np.cumsum(generator.normal(scale=0.02, size=(rows, sessions)), axis=1)
    close -= close[:, :1]
    opening = close + generator.normal(scale=0.005, size=close.shape)
    high = np.maximum(opening, close) + np.abs(generator.normal(scale=0.004, size=close.shape))
    low = np.minimum(opening, close) - np.abs(generator.normal(scale=0.004, size=close.shape))
    volume = np.log1p(np.abs(generator.normal(size=close.shape)))
    parts = [opening, high, low, close, volume]
    if channels == 6:
        parts.append(np.ones_like(close))
    return torch.tensor(np.stack(parts, axis=-1), dtype=torch.float64)


def fit_rows(rows=ROWS, *, seed=11):
    generator = np.random.default_rng(seed)
    inputs = dict(prices=anchored(rows, 16, seed=seed).numpy().astype(np.float32))
    for name in ("news", "charts"):
        inputs[name] = generator.normal(3.0, 2.0, size=(rows, DIMENSIONS[name])).astype(np.float32)
    presence = np.ones((rows, 5), dtype=np.bool_)
    presence[::4, 1] = False
    inputs["news"][::4] = 0
    for name, level in (("fundamentals", 15.0), ("macro", -4.0)):
        third = DIMENSIONS[name] // 3
        values = generator.normal(level, 3.0, size=(rows, third))
        observed = generator.random(size=(rows, third)) < 0.7
        ages = np.abs(generator.normal(4.0, 1.0, size=(rows, third)))
        inputs[name] = np.concatenate(
            [values * observed, observed, ages * observed], axis=1
        ).astype(np.float32)
    targets = generator.normal(scale=0.02, size=rows + 7)
    return inputs, presence, targets


def test_declaration_validates_its_three_options_and_names_them_in_its_identity():
    default = LearningScale()
    assert (default.clip, default.price_returns, default.zero_median) == (5.0, True, True)
    identity = default.identity()
    assert identity["name"] == TRANSFORM and identity["clip"] == 5.0
    assert identity["price_window"] == "session_returns_v1" and identity["initial_median"] == "zero"
    other = LearningScale(clip=3, price_returns=False, zero_median=False).identity()
    assert (other["clip"], other["price_window"], other["initial_median"]) == (
        3.0,
        "anchored_logs",
        "sampled",
    )
    for options in (
        dict(clip=0.5),
        dict(clip=101),
        dict(clip=math.inf),
        dict(clip=True),
        dict(price_returns=1),
        dict(zero_median=None),
    ):
        with pytest.raises(ValueError, match="clip entre 1 y 100"):
            LearningScale(**options)
    assert LearningScale(clip=1).clip == 1 and LearningScale(clip=100).clip == 100


def test_parse_accepts_the_complete_json_declaration_only():
    declared = dict(clip=4.0, price_returns=True, zero_median=False)
    assert parse_learning_scale(declared) == LearningScale(**declared)
    assert parse_learning_scale(None) is None
    config = LearningScale()
    assert parse_learning_scale(config) is config
    for value in (dict(clip=4.0), {**declared, "extra": 1}, "learning_scale_v1", [5.0]):
        with pytest.raises(ValueError, match="clip, price_returns y zero_median"):
            parse_learning_scale(value)


@pytest.mark.parametrize("channels", [5, 6])
def test_session_returns_rebuild_the_anchored_window(channels):
    window = anchored(channels=channels)
    result = price_returns(window)
    assert result.shape == window.shape and result.dtype == window.dtype
    overnight, intraday, upper, lower = result[..., :4].unbind(-1)
    opening, high, low, close = window[..., :4].unbind(-1)
    # La primera sesión no tiene cierre anterior y el ancla es su propio cierre.
    assert torch.equal(overnight[:, 0], torch.zeros_like(overnight[:, 0]))
    torch.testing.assert_close(torch.cumsum(overnight + intraday, dim=1) - intraday[:, :1], close)
    torch.testing.assert_close(overnight[:, 1:], opening[:, 1:] - close[:, :-1])
    torch.testing.assert_close(upper, high - torch.maximum(opening, close))
    torch.testing.assert_close(lower, torch.minimum(opening, close) - low)
    assert (upper >= 0).all() and (lower >= 0).all()
    # El volumen y el bit de presencia pasan sin cambios.
    assert torch.equal(result[..., 4:], window[..., 4:])


def test_an_absent_session_stays_at_zero_and_the_next_return_uses_the_last_present_close():
    window = anchored(rows=2, sessions=6)
    window[0, 2] = 0  # Sesión sin mercado: cinco canales y bit a cero.
    window[1, 0] = 0  # También puede faltar la primera sesión de la ventana.
    result = price_returns(window)
    assert torch.equal(result[0, 2], torch.zeros(6, dtype=window.dtype))
    assert result[0, 3, 0] == window[0, 3, 0] - window[0, 1, 3]
    assert torch.equal(result[1, 0], torch.zeros(6, dtype=window.dtype))
    # Sin ningún cierre presente antes, la sesión no tiene rendimiento nocturno.
    assert result[1, 1, 0] == 0 and result[1, 1, 1] == window[1, 1, 3] - window[1, 1, 0]
    assert result[1, 2, 0] == window[1, 2, 0] - window[1, 1, 3]


@pytest.mark.parametrize(
    "value",
    [np.zeros((2, 8, 6)), torch.zeros(8, 6), torch.zeros(2, 8, 4), torch.zeros(2, 8, 7)],
)
def test_session_returns_reject_other_shapes(value):
    with pytest.raises(ValueError, match="5 o 6 canales"):
        price_returns(value)


def test_statistics_use_the_observed_entries_of_each_column_and_the_label_interval():
    config = LearningScale()
    inputs, presence, targets = fit_rows()
    result = fit_statistics(config, inputs, presence, targets)
    assert result["schema_version"] == 1 and result["transform"] == config.identity()
    assert result["rows"] == ROWS and result["labels"] == ROWS + 7
    lower, upper = np.quantile(targets, (0.1, 0.9))
    assert result["label_scale"] == pytest.approx((upper - lower) / INITIAL_INTERVAL, rel=1e-12)
    assert set(result["modalities"]) == set(MODALITIES)
    # Noticias: solo las filas con su bit de presencia. Las ausentes valen cero y no cuentan.
    news = inputs["news"][presence[:, 1]].astype(np.float64)
    low, center, high = np.quantile(news, (0.16, 0.5, 0.84), axis=0)
    np.testing.assert_allclose(result["modalities"]["news"]["center"], center, rtol=1e-6)
    np.testing.assert_allclose(result["modalities"]["news"]["scale"], (high - low) / 2, rtol=1e-6)
    assert abs(np.median(inputs["news"][:, 0]) - center[0]) > 0.05
    # Numéricas: cada columna con sus entradas observadas. Los bits no se centran ni escalan.
    third = DIMENSIONS["fundamentals"] // 3
    values, observed = (
        inputs["fundamentals"][:, :third],
        inputs["fundamentals"][:, third : 2 * third],
    )
    record = result["modalities"]["fundamentals"]
    for column in range(third):
        chosen = values[observed[:, column] > 0, column].astype(np.float64)
        low, center, high = np.quantile(chosen, (0.16, 0.5, 0.84))
        assert record["center"][column] == pytest.approx(center, rel=1e-6)
        assert record["scale"][column] == pytest.approx((high - low) / 2, rel=1e-6)
    assert record["center"][third : 2 * third] == [0.0] * third
    assert record["scale"][third : 2 * third] == [1.0] * third
    # Precios: canales de la ventana de rendimientos, no de la ventana anclada.
    returns = price_returns(torch.from_numpy(inputs["prices"])).numpy()[..., :5].reshape(-1, 5)
    np.testing.assert_allclose(
        result["modalities"]["prices"]["center"], np.median(returns, axis=0), rtol=1e-5, atol=1e-7
    )
    anchored_statistics = fit_statistics(
        LearningScale(price_returns=False), inputs, presence, targets
    )
    assert anchored_statistics["modalities"]["prices"] != result["modalities"]["prices"]
    assert statistics_digest(result) == statistics_digest(fit_statistics(config, *fit_rows()))
    assert statistics_digest(result) != statistics_digest(anchored_statistics)


def test_constant_sparse_and_never_observed_columns_are_not_amplified():
    inputs, presence, targets = fit_rows()
    inputs["charts"][:, 0] = 7.0
    third = DIMENSIONS["macro"] // 3
    inputs["macro"][:, 0], inputs["macro"][:, third], inputs["macro"][:, 2 * third] = 0, 0, 0
    result = fit_statistics(LearningScale(), inputs, presence, targets)
    charts, macro = result["modalities"]["charts"], result["modalities"]["macro"]
    # Sin dispersión, la escala es el suelo relativo del bloque, no una división por cero.
    assert charts["center"][0] == 7.0
    assert charts["scale"][0] == pytest.approx(RELATIVE_FLOOR * np.median(charts["scale"][1:]))
    assert macro["center"][0] == 0.0
    assert macro["scale"][0] == pytest.approx(RELATIVE_FLOOR * np.median(macro["scale"][1:third]))
    assert all(value > 0 for record in result["modalities"].values() for value in record["scale"])
    # Con una cuarta parte de valores distintos de cero, la semidistancia entre cuantiles
    # todavía los separa y es la escala.
    sparse = np.zeros(ROWS, dtype=np.float32)
    sparse[: ROWS // 4] = 40.0
    inputs["charts"][:, 1] = sparse
    scale = fit_statistics(LearningScale(), inputs, presence, targets)["modalities"]["charts"]
    assert scale["scale"][1] == pytest.approx(20.0) and STD_FRACTION * sparse.std() < 20.0
    # Con casi todo ceros los dos cuantiles coinciden. La escala es entonces la mitad de la
    # desviación de la columna y un valor distinto de cero no se amplifica sin límite.
    sparse[:] = 0.0
    sparse[: ROWS // 10] = 40.0
    inputs["charts"][:, 1] = sparse
    scale = fit_statistics(LearningScale(), inputs, presence, targets)["modalities"]["charts"]
    assert np.quantile(sparse, 0.84) == np.quantile(sparse, 0.16) == 0.0
    assert scale["scale"][1] == pytest.approx(STD_FRACTION * sparse.std(), rel=1e-6)
    # Un bloque sin ninguna columna con dispersión conserva la unidad.
    inputs["charts"][:] = 3.0
    flat = fit_statistics(LearningScale(), inputs, presence, targets)["modalities"]["charts"]
    assert flat["scale"] == [1.0] * DIMENSIONS["charts"] and flat["center"] == [3.0] * 3


@pytest.mark.parametrize(
    "change",
    ["few_rows", "few_labels", "nan_input", "nan_label", "constant_labels", "missing", "presence"],
)
def test_statistics_reject_rows_that_cannot_estimate_them(change):
    inputs, presence, targets = fit_rows()
    config = LearningScale()
    if change == "few_rows":
        inputs = {name: value[: MIN_ROWS - 1] for name, value in inputs.items()}
        presence = presence[: MIN_ROWS - 1]
    elif change == "few_labels":
        targets = targets[: MIN_ROWS - 1]
    elif change == "nan_input":
        inputs["charts"][3, 1] = np.nan
    elif change == "nan_label":
        targets[5] = np.inf
    elif change == "constant_labels":
        targets[:] = 0.01
    elif change == "missing":
        inputs.pop("macro")
    else:
        presence = presence.astype(np.float32)
    with pytest.raises(ValueError):
        fit_statistics(config, inputs, presence, targets)
    with pytest.raises(ValueError, match="declaración LearningScale"):
        fit_statistics(dict(clip=5.0), *fit_rows())
    exact = {name: value[:MIN_ROWS] for name, value in fit_rows()[0].items()}
    assert fit_statistics(config, exact, fit_rows()[1][:MIN_ROWS], fit_rows()[2][:MIN_ROWS])


def scaler(config=None, *, dtype=torch.float64):
    config = LearningScale() if config is None else config
    inputs, presence, targets = fit_rows()
    statistics = fit_statistics(config, inputs, presence, targets)
    module = LearningScaler(config, DIMENSIONS, dtype=dtype)
    tensors = {name: torch.tensor(value, dtype=dtype) for name, value in inputs.items()}
    return module, statistics, tensors, torch.tensor(presence)


def test_scaler_refuses_to_transform_before_it_receives_its_statistics():
    module, statistics, tensors, presence = scaler()
    assert not bool(module.fitted)
    with pytest.raises(ValueError, match="Faltan las estadísticas"):
        module(tensors, presence)
    module.load(statistics)
    assert bool(module.fitted) and set(module(tensors, presence)) == set(MODALITIES)
    assert not list(module.parameters())


def test_scaled_fit_rows_are_centred_with_unit_robust_scale_and_absences_stay_at_zero():
    module, statistics, tensors, presence = scaler()
    module.load(statistics)
    result = module(tensors, presence)
    for name, value in result.items():
        assert value.shape == tensors[name].shape and value.dtype == tensors[name].dtype
    present = presence[:, 1]
    news = result["news"][present]
    low, center, high = torch.quantile(news, torch.tensor([0.16, 0.5, 0.84], dtype=news.dtype), 0)
    torch.testing.assert_close(center, torch.zeros_like(center), atol=1e-5, rtol=0)
    torch.testing.assert_close((high - low) / 2, torch.ones_like(low), atol=1e-5, rtol=0)
    # El bloque denso ausente sigue en cero, aunque su centro no lo sea.
    assert torch.equal(result["news"][~present], torch.zeros_like(result["news"][~present]))
    assert (module.news_center != 0).all()
    third = DIMENSIONS["fundamentals"] // 3
    observed = tensors["fundamentals"][:, third : 2 * third]
    levels, ages = result["fundamentals"][:, :third], result["fundamentals"][:, 2 * third :]
    assert torch.equal(result["fundamentals"][:, third : 2 * third], observed)
    assert torch.equal(levels[observed == 0], torch.zeros_like(levels[observed == 0]))
    assert torch.equal(ages[observed == 0], torch.zeros_like(ages[observed == 0]))
    # `torch.quantile` interpola como NumPy. `median` devolvería el valor central inferior.
    assert abs(float(torch.quantile(levels[:, 0][observed[:, 0] > 0], 0.5))) < 1e-5
    # Precios: rendimientos escalados, con el bit de presencia intacto.
    assert torch.equal(result["prices"][..., 5], tensors["prices"][..., 5])
    flat = result["prices"][..., :5].reshape(-1, 5)
    assert float(torch.quantile(flat, 0.5, dim=0).abs().max()) < 1e-4


def test_scaler_clips_drift_and_keeps_absent_sessions_at_zero():
    config = LearningScale(clip=2.0)
    module, statistics, tensors, presence = scaler(config)
    module.load(statistics)
    drifted = {name: value.clone() for name, value in tensors.items()}
    drifted["charts"] += 1000.0
    third = DIMENSIONS["macro"] // 3
    drifted["macro"][:, :third] -= 1000.0
    drifted["prices"][0, 3] = 0  # Sesión sin mercado.
    result = module(drifted, presence)
    assert torch.equal(result["charts"], torch.full_like(result["charts"], 2.0))
    observed = drifted["macro"][:, third : 2 * third] > 0
    assert torch.equal(
        result["macro"][:, :third][observed],
        torch.full_like(result["macro"][:, :third][observed], -2.0),
    )
    assert float(max(value.abs().max() for value in result.values())) <= 2.0
    assert torch.equal(result["prices"][0, 3], torch.zeros(6, dtype=torch.float64))
    assert (module.prices_center != 0).any()


def test_scaler_keeps_the_anchored_window_when_returns_are_not_declared():
    config = LearningScale(price_returns=False, clip=100.0)
    module, statistics, tensors, presence = scaler(config)
    module.load(statistics)
    result = module(tensors, presence)["prices"][..., :5]
    expected = (tensors["prices"][..., :5] - module.prices_center) * module.prices_inverse
    torch.testing.assert_close(result, expected.clamp(-100, 100))


@pytest.mark.parametrize(
    "change", ["version", "transform", "modality", "width", "negative", "nan", "not_dict"]
)
def test_scaler_rejects_statistics_of_another_declaration_or_shape(change):
    module, statistics, _, _ = scaler()
    if change == "version":
        statistics["schema_version"] = 2
    elif change == "transform":
        statistics["transform"] = LearningScale(clip=4.0).identity()
    elif change == "modality":
        statistics["modalities"].pop("news")
    elif change == "width":
        statistics["modalities"]["charts"]["center"].append(0.0)
    elif change == "negative":
        statistics["modalities"]["macro"]["scale"][0] = 0.0
    elif change == "nan":
        statistics["modalities"]["prices"]["center"][1] = math.nan
    else:
        statistics = [statistics]
    with pytest.raises(ValueError):
        module.load(statistics)
    assert not bool(module.fitted)


def test_scaler_needs_its_declaration_the_five_modalities_and_numeric_thirds():
    with pytest.raises(ValueError, match="cinco modalidades"):
        LearningScaler(dict(clip=5.0), DIMENSIONS)
    with pytest.raises(ValueError, match="cinco modalidades"):
        LearningScaler(LearningScale(), {**DIMENSIONS, "extra": 2})
    with pytest.raises(ValueError, match="valores, presencia y antigüedad"):
        LearningScaler(LearningScale(), {**DIMENSIONS, "macro": 10})


def test_statistics_travel_in_the_state_of_the_scaler():
    module, statistics, tensors, presence = scaler()
    module.load(statistics)
    twin = LearningScaler(LearningScale(), DIMENSIONS, dtype=torch.float64)
    twin.load_state_dict(module.state_dict())
    assert bool(twin.fitted)
    for name, value in module(tensors, presence).items():
        assert torch.equal(value, twin(tensors, presence)[name])
