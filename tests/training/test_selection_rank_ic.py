"""Selección por IC de rangos por sesión: regla pura, puntuación y enmienda de la campaña."""

import math

import numpy as np
import pyarrow as pa
import pytest
from scipy.stats import spearmanr

from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.training.common_epoch import EpochValidation, rank_record, selected_score
from mars_titan.training.selection import (
    CHRONOLOGICAL_MODES,
    COMMON_EPOCH,
    FIXED_BUDGET,
    MAE_DEFINITION,
    METRICS,
    RANK_IC_DEFINITION,
    SESSION_MAE,
    SESSION_RANK_IC,
    advance_selection,
    campaign_rule,
    initial_selection,
    metric_definition,
    selection_score,
    validate_selection,
)

RANKED = dict(metric=SESSION_RANK_IC, min_assets=3, patience=3, min_delta=0.0)
MAE = dict(metric=SESSION_MAE, patience=3, min_delta=0.0)
DAY_US = 86_400_000_000
FIRST = 1_546_380_000_000_000


def panel_table(sessions):
    """Tabla puntual con una lista de sesiones, cada una con sus pares (predicción, objetivo)."""
    samples, times, predictions, targets = [], [], [], []
    for day, rows in enumerate(sessions):
        time = FIRST + day * DAY_US
        for asset, (prediction, target) in enumerate(rows):
            samples.append(f"US/A{asset}/{time}")
            times.append(time)
            predictions.append(float(prediction))
            targets.append(float(target))
    return pa.table(
        dict(
            sample_id=samples,
            market=["US"] * len(samples),
            prediction_at=pa.array(times, type=pa.timestamp("us", tz="UTC")),
            target=targets,
            prediction=predictions,
        )
    )


def loss(ic):
    return (1.0 - ic) / 2.0


def test_the_rank_ic_is_a_declared_metric_with_its_minimum_assets():
    assert METRICS == (SESSION_MAE, SESSION_RANK_IC)
    validate_selection(RANKED)
    validate_selection(dict(RANKED, stopping=FIXED_BUDGET, minimum_epochs=1), epochs=4)
    validate_selection(dict(RANKED, initial_state="reported"), initial_reported=True)
    assert metric_definition(RANKED) == RANK_IC_DEFINITION != MAE_DEFINITION
    assert metric_definition(MAE) == MAE_DEFINITION
    without = {key: value for key, value in RANKED.items() if key != "min_assets"}
    for options in (without, dict(MAE, min_assets=3), dict(RANKED, metric="session_ic")):
        with pytest.raises(ValueError, match="La selección necesita"):
            validate_selection(options)
    for assets in (2, 100_001, True, 3.0, "3", None):
        with pytest.raises(ValueError, match="activos mínimos"):
            validate_selection(dict(RANKED, min_assets=assets))
    with pytest.raises(ValueError, match="pérdida máxima"):
        validate_selection(dict(RANKED, min_delta=1))
    with pytest.raises(ValueError, match="La época común puntúa con su propio IC"):
        validate_selection(dict(RANKED, stopping=COMMON_EPOCH), modes=CHRONOLOGICAL_MODES)


def test_the_score_is_the_mae_or_the_bounded_loss_of_the_rank_ic():
    assert selection_score(MAE, {SESSION_MAE: 0.25, SESSION_RANK_IC: 0.9}) == 0.25
    for ic in (1.0, 0.04, 0.0, -0.3, -1.0):
        score = selection_score(RANKED, {SESSION_MAE: 0.25, SESSION_RANK_IC: ic})
        assert score == loss(ic) and 0 <= score <= 1
    # Sin ninguna sesión definida no hay orden: la pérdida es la de un IC nulo.
    assert selection_score(RANKED, {SESSION_RANK_IC: None}) == 0.5 == loss(0.0)
    with pytest.raises(ValueError, match="no se puntuó"):
        selection_score(RANKED, {SESSION_MAE: 0.25})
    for invalid in (1.5, -1.0000001, math.nan, math.inf, "0.1", 1, True):
        with pytest.raises(ValueError, match="entre −1 y 1"):
            selection_score(RANKED, {SESSION_RANK_IC: invalid})


def test_the_plateau_keeps_the_epoch_with_the_highest_rank_ic():
    state = None
    for epoch, ic in enumerate((0.010, 0.030, 0.020, 0.029, 0.030), 1):
        score = selection_score(RANKED, {SESSION_RANK_IC: ic})
        state = advance_selection(state, score, epoch, RANKED)
    # La época 5 empata con la 2 y un empate no es una mejora estricta: tres épocas sin mejorar.
    assert state["best_epoch"] == 2 and state["best_score"] == loss(0.030)
    assert state["stale_epochs"] == 3 and state["should_stop"]
    # Un lector conserva a su padre como candidato: solo lo sustituye un IC mayor.
    parent = initial_selection(loss(0.035), RANKED)
    assert parent["best_epoch"] == 0 and parent["best_score"] == loss(0.035)
    worse = advance_selection(parent, loss(0.034), 1, RANKED)
    assert worse["best_epoch"] == 0 and not worse["last_improved"]
    better = advance_selection(worse, loss(0.036), 2, RANKED)
    assert better["best_epoch"] == 2 and better["best_score"] == loss(0.036)
    # Con un estado inicial medido y no emitido, la primera época lo sustituye siempre.
    reported = dict(RANKED, initial_state="reported")
    start = initial_selection(loss(0.9), reported)
    assert start["best_score"] is None
    assert advance_selection(start, loss(-0.2), 1, reported)["best_epoch"] == 1
    # La mejora mínima se mide en la pérdida: 0,001 de pérdida son 0,002 de IC.
    strict = dict(RANKED, min_delta=0.001)
    first = advance_selection(None, loss(0.030), 1, strict)
    assert not advance_selection(first, loss(0.0315), 2, strict)["last_improved"]
    assert advance_selection(first, loss(0.0325), 2, strict)["last_improved"]


def test_a_campaign_amends_the_metric_only_towards_the_rank_ic():
    protocol = dict(MAE, max_epochs=8)
    amended = dict(RANKED, max_epochs=12)
    assert campaign_rule(protocol, None) == protocol
    assert campaign_rule(protocol, amended, modes=CHRONOLOGICAL_MODES) == amended
    assert campaign_rule(protocol, dict(protocol, patience=5))["patience"] == 5
    for override in (dict(amended, metric="session_ic"), dict(MAE), dict(amended, min_assets=2)):
        with pytest.raises(ValueError):
            campaign_rule(protocol, override)
    # Un protocolo que ya declara el IC de rangos no vuelve al MAE ni cambia sus activos mínimos.
    ranked_protocol = dict(RANKED, max_epochs=8)
    assert campaign_rule(ranked_protocol, dict(ranked_protocol, patience=5))["patience"] == 5
    for override in (dict(protocol), dict(ranked_protocol, min_assets=4)):
        with pytest.raises(ValueError, match="conserva la métrica"):
            campaign_rule(ranked_protocol, override)


def test_the_epoch_scorer_ranks_each_session_like_spearman():
    rng = np.random.default_rng(7)
    noisy = [tuple(pair) for pair in rng.standard_normal((6, 2))]
    tied = [(1.0, 0.3), (1.0, -0.2), (2.0, 0.5), (3.0, 0.1), (3.0, 0.9)]
    constant_prediction = [(0.5, value) for value in (0.1, -0.4, 0.7, 0.2)]
    constant_target = [(value, 0.0) for value in (0.1, -0.4, 0.7, 0.2)]
    short = [(0.1, 0.2), (0.3, -0.1)]
    table = panel_table([noisy, tied, constant_prediction, constant_target, short])
    record = rank_record(ForecastPanel.from_arrow(table), 3)
    assert record["session_rank_ic_defined"] == [True, True, False, False, False]
    for values, rows in zip(record["session_rank_ic_values"][:2], (noisy, tied), strict=True):
        expected = spearmanr(*zip(*rows, strict=True)).statistic
        assert values == pytest.approx(expected, abs=1e-12)
    assert record["session_rank_ic_values"][2:] == [0.0, 0.0, 0.0]
    defined = record["session_rank_ic_values"][:2]
    assert record["session_rank_ic"] == float(np.asarray(defined).mean())
    assert len(record["session_time"]) == 5 and len(record["cohort_sha256"]) == 64
    scorer = EpochValidation(lambda: None, lambda rows, metrics: [table], min_assets=3, rank=True)
    assert scorer.score(1, None, {}) == record
    # Con más activos mínimos solo queda la primera sesión, y sin ninguna el IC no existe.
    assert rank_record(ForecastPanel.from_arrow(table), 6)["session_rank_ic_defined"][0]
    assert sum(rank_record(ForecastPanel.from_arrow(table), 6)["session_rank_ic_defined"]) == 1
    assert rank_record(ForecastPanel.from_arrow(table), 7)["session_rank_ic"] is None


def test_the_rank_scorer_keeps_no_epoch_predictions(tmp_path):
    with pytest.raises(ValueError, match="no guarda predicciones"):
        EpochValidation(
            lambda: None, lambda rows, metrics: [], min_assets=3, directory=tmp_path, rank=True
        )
    with pytest.raises(ValueError, match="no guarda predicciones"):
        EpochValidation(lambda: None, lambda rows, metrics: [], min_assets=3, rank="yes")
    with pytest.raises(ValueError, match="al menos tres activos"):
        EpochValidation(lambda: None, lambda rows, metrics: [], min_assets=2, rank=True)


def test_the_selected_score_recomputes_the_metric_from_the_emitted_tables():
    rng = np.random.default_rng(11)
    sessions = [[tuple(pair) for pair in rng.standard_normal((5, 2))] for _ in range(4)]
    table = panel_table(sessions)
    assert selected_score(MAE, {SESSION_MAE: 0.25}, [table]) == 0.25
    ic = rank_record(ForecastPanel.from_arrow(table), 3)["session_rank_ic"]
    assert selected_score(RANKED, {SESSION_MAE: 0.25}, [table]) == loss(ic)
    # Las tablas llegan por bloques y se concilian antes de puntuar.
    assert selected_score(RANKED, {}, [table.slice(0, 7), table.slice(7)]) == loss(ic)
    undefined = panel_table([[(0.5, 0.1), (0.5, 0.2), (0.5, 0.3)]])
    assert selected_score(RANKED, {}, [undefined]) == 0.5
