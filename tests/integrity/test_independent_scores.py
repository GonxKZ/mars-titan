"""Segunda implementación de las métricas por sesión frente a la especificación y al código.

Los paneles son pequeños y se escriben en cada prueba. No proceden de ningún modelo
ajustado y no se ejecuta ningún paso de optimizador.
"""

import numpy as np
import pandas as pd
import pytest

from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.forecast_scores import implied_up_probability, score_sessions
from mars_titan.integrity.independent_scores import (
    interval_pairs,
    session_mean,
    session_scores,
    up_probability,
)
from mars_titan.integrity.score_recheck import compare_sessions
from mars_titan.models.quantile_head import LEVELS, QUANTILE_COLUMNS


def frame(markets, times, targets, predictions, quantiles=None):
    data = pd.DataFrame(
        dict(market=markets, prediction_at=times, target=targets, prediction=predictions)
    )
    if quantiles is not None:
        for name, column in zip(QUANTILE_COLUMNS, np.asarray(quantiles).T, strict=True):
            data[name] = column
    return data


def test_session_mae_follows_the_specification_example():
    # Errores 1 y 3 en una sesión y 6 en otra: MAE por sesión 4 y por fila 10/3.
    scores = session_scores(frame(["US"] * 3, [1, 1, 2], [0.0, 0.0, 0.0], [1.0, -3.0, 6.0]))
    assert scores["mae"].tolist() == [2.0, 6.0]
    assert session_mean(scores["mae"], [True, True], scores["market"], "session") == 4.0
    assert float(np.dot(scores["samples"], scores["mae"]) / 3) == pytest.approx(10 / 3)


def test_sign_precision_and_recall_follow_the_specification_example():
    targets = [1.0, 2.0, -1.0, -2.0, 0.0, 3.0]
    predictions = [1.0, -1.0, -1.0, 0.0, 5.0, 2.0]
    row = session_scores(frame(["US"] * 6, [7] * 6, targets, predictions)).iloc[0]
    assert (row["up_hits"], row["up_calls"], row["positive_targets"]) == (2, 2, 3)
    assert (row["down_hits"], row["down_calls"], row["negative_targets"]) == (1, 2, 2)
    # La predicción 5 sobre objetivo cero no se juzga y la predicción 0 sobre −2 es fallo.
    assert (row["direction_eligible"], row["direction_calls"], row["direction_hits"]) == (5, 4, 3)


def test_rank_ic_reasons_follow_the_declared_priority():
    data = frame(
        ["US"] * 2 + ["CN"] * 3 + ["CN"] * 3 + ["US"] * 3,
        [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        [1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0, 2.0, 1.0, 5.0, 5.0, 5.0, 3.0, 2.0, 1.0],
    )
    scores = session_scores(data).set_index(["market", "prediction_at"])
    assert scores.loc[("US", 1), "rank_ic_status"] == "insufficient_assets"
    assert scores.loc[("CN", 2), "rank_ic_status"] == "constant_target"
    assert scores.loc[("CN", 3), "rank_ic_status"] == "constant_prediction"
    assert scores.loc[("US", 4), "rank_ic"] == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("quantiles", "expected"),
    [
        ([0.1, 0.2, 0.3, 0.4, 0.5], 0.975),  # Todo por encima de cero: F(0) = 0,025.
        ([-0.5, -0.4, -0.3, -0.2, -0.1], 0.025),  # Todo por debajo: F(0) = 0,975.
        ([-2.0, -1.0, 1.0, 2.0, 3.0], 1.0 - (0.1 + 0.4 * 0.5)),  # Interpolación lineal.
        ([-2.0, -1.0, 0.0, 0.0, 3.0], 0.1),  # Empate en cero: límite por la derecha (0,9).
    ],
)
def test_up_probability_applies_the_declared_rule(quantiles, expected):
    assert up_probability([quantiles], LEVELS)[0] == pytest.approx(expected)


def test_interval_pairs_are_the_declared_central_intervals():
    assert interval_pairs(LEVELS) == [(0.1, 0.9), (0.025, 0.975)]


def random_panel(rng, sessions=40, quantile=True):
    """Panel con empates, ceros, sesiones pequeñas y dos mercados, para cotejar código."""
    parts = []
    for index in range(sessions):
        market = "US" if index % 3 else "CN"
        size = int(rng.integers(1, 9))
        target = np.round(rng.normal(0, 1, size), 1)
        target[rng.random(size) < 0.15] = 0.0
        prediction = np.round(0.4 * target + rng.normal(0, 0.6, size), 1)
        prediction[rng.random(size) < 0.15] = 0.0
        part = dict(
            sample_id=[f"{market}/{index}/{i}" for i in range(size)],
            market=[market] * size,
            prediction_at=[1_700_000_000_000_000 + index * 86_400_000_000] * size,
            target=target,
            prediction=prediction,
        )
        if quantile:
            scale = np.abs(rng.normal(0.5, 0.3, size))[:, None]
            offsets = np.array([-1.96, -1.28, 0.0, 1.28, 1.96])
            part["quantiles"] = np.round(prediction[:, None] + scale * offsets, 2)
            part["quantiles"][:, 2] = prediction
        parts.append(part)
    return parts


@pytest.mark.parametrize("quantile", [True, False])
def test_independent_scores_match_the_comparison_code_on_edge_cases(quantile):
    rng = np.random.default_rng(20261010)
    parts = random_panel(rng, quantile=quantile)
    concat = {
        key: np.concatenate([np.asarray(part[key]) for part in parts])
        for key in ("sample_id", "market", "prediction_at", "target", "prediction")
    }
    quantiles = np.vstack([part["quantiles"] for part in parts]) if quantile else None
    panel = ForecastPanel.from_columns(
        concat["sample_id"],
        concat["market"],
        concat["prediction_at"],
        concat["target"],
        concat["prediction"],
        quantiles=quantiles,
        levels=LEVELS if quantile else None,
        markets=("US", "CN"),
    )
    published = score_sessions(panel).to_table().to_pandas()
    independent = session_scores(
        frame(
            concat["market"],
            concat["prediction_at"],
            concat["target"],
            concat["prediction"],
            quantiles,
        ),
        levels=LEVELS if quantile else None,
        quantile_columns=QUANTILE_COLUMNS if quantile else None,
    )
    outcome = compare_sessions(independent, published)
    assert outcome["passed"], outcome
    if quantile:
        assert {"interval_score_0.8", "interval_score_0.95", "sign_brier"} <= set(
            outcome["columns"]
        )
        probability = implied_up_probability(quantiles, LEVELS)
        assert np.allclose(up_probability(quantiles, LEVELS), probability, rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    ("column", "change"),
    [
        ("mae", lambda v: v * (1 + 1e-6)),
        ("direction_hits", lambda v: v + 1),
        ("rank_ic", lambda v: -v),
        ("interval_score_0.95", lambda v: v + 1e-9),
        ("sign_brier", lambda v: v * 0.5),
        ("pinball_0.1", lambda v: v + 1e-8),
    ],
)
def test_a_corrupted_published_value_is_detected(column, change):
    rng = np.random.default_rng(7)
    parts = random_panel(rng, sessions=12)
    data = frame(
        np.concatenate([p["market"] for p in parts]),
        np.concatenate([p["prediction_at"] for p in parts]),
        np.concatenate([p["target"] for p in parts]),
        np.concatenate([p["prediction"] for p in parts]),
        np.vstack([p["quantiles"] for p in parts]),
    )
    reference = session_scores(data, levels=LEVELS, quantile_columns=QUANTILE_COLUMNS)
    corrupted = reference.copy()
    target = corrupted[column].notna().idxmax()
    corrupted.loc[target, column] = change(corrupted.loc[target, column])
    outcome = compare_sessions(reference, corrupted)
    assert not outcome["passed"]
    assert outcome["columns"][column]["mismatched_sessions"] == 1


def test_missing_or_extra_sessions_are_counted():
    data = frame(["US"] * 4, [1, 1, 2, 2], [1.0, -1.0, 2.0, -2.0], [0.5, -0.5, 1.0, 1.0])
    reference = session_scores(data)
    outcome = compare_sessions(reference, reference.iloc[:1])
    assert not outcome["passed"]
    assert outcome["sessions_only_recomputed"] == 1


def test_market_weighting_averages_markets_after_sessions():
    values, markets = [1.0, 3.0, 10.0], ["US", "US", "CN"]
    assert session_mean(values, [True] * 3, markets, "session") == pytest.approx(14 / 3)
    assert session_mean(values, [True] * 3, markets, "market") == pytest.approx(6.0)
    assert session_mean(values, [False] * 3, markets, "session") is None
