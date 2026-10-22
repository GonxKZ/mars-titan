"""Contrato temporal y bloqueo del ajuste del catálogo, sin ejecutar regresiones."""

import importlib
from dataclasses import replace

import pytest

from mars_titan.models.titans.financial_inputs import HISTORICAL_START_US
from mars_titan.training.learning_hold import LearningHoldError

DAY = 86_400_000_000
START = HISTORICAL_START_US + 7000 * DAY


def api():
    try:
        return importlib.import_module("mars_titan.training.mechanism_policy_fit")
    except ModuleNotFoundError:
        pytest.fail("Falta el ajuste protegido de las ganancias del catálogo")


def configuration(**options):
    return api().MechanismFitConfig(
        parent_consumer_id="a" * 64,
        parent_selected_before=START - DAY,
        fit_start=START,
        fit_end=START + 100 * DAY,
        ridge=0.01,
        interval_weight=0.0,
        min_gain=0.0,
        market_weights=(("US", 1.0),),
        costs=(0.0,) * 36,
        **options,
    )


@pytest.mark.parametrize("entry", ["fit_policy", "_cohorts"])
def test_fit_refuses_before_inspecting_sources_or_constructing_targets(
    learning_hold, tmp_path, entry
):
    learning_hold(False)
    touched = []

    class Forbidden:
        def __iter__(self):
            touched.append(True)
            raise AssertionError("Se han abierto fuentes bajo el bloqueo")

    with pytest.raises(LearningHoldError):
        getattr(api(), entry)(Forbidden(), config=None, device="invalid")
    assert touched == []
    assert list(tmp_path.iterdir()) == []


def test_configuration_identifies_objectives_scales_and_budgets_without_fitting():
    config = configuration()
    record = config.identity()
    assert record["point_objective"] == "mae_baseline_minus_mae_candidate"
    assert record["interval_objective"] == "mean_interval_score_80_95_baseline_minus_candidate"
    assert record["normalization"] == "fit_cohorts_only_weighted_population"
    assert record["missing_targets"] == "masked_not_observed_zero"
    assert record["intercept"] == "ridge_penalized_towards_zero_gain"
    assert config.max_cohorts == 4096 and config.min_cohorts == 32
    assert config.max_source_bytes == 512 * 1024**2
    assert len(config.fingerprint()) == 64


@pytest.mark.parametrize(
    "field,value",
    [
        ("parent_consumer_id", "invalid"),
        ("parent_selected_before", START + 1),
        ("fit_start", True),
        ("fit_end", START),
        ("ridge", 0.0),
        ("ridge", float("nan")),
        ("ridge", True),
        ("interval_weight", -1.0),
        ("min_gain", float("inf")),
        ("min_cohorts", 1),
        ("max_cohorts", 8193),
        ("max_source_bytes", 0),
        ("max_sources", 17),
        ("max_condition", 0.5),
        ("market_weights", (("US", 0.0),)),
        ("market_weights", (("US", True),)),
        ("market_weights", (("US", 1.0), ("CN", 1.0))),
        ("market_weights", (("US", 1.0), ("US", 1.0))),
        ("costs", (0.0,) * 35),
        ("costs", (1.0,) * 36),
    ],
)
def test_invalid_fit_protocol_is_rejected_without_touching_any_data(field, value):
    config = configuration()
    with pytest.raises(ValueError):
        replace(config, **{field: value})


def test_final_test_boundary_and_historical_start_remain_exclusive():
    from mars_titan.models.titans.financial_inputs import FINAL_TEST_US

    config = configuration()
    with pytest.raises(ValueError):
        replace(config, fit_start=HISTORICAL_START_US - 1, parent_selected_before=0)
    with pytest.raises(ValueError):
        replace(config, fit_end=FINAL_TEST_US + 1)
    assert replace(config, fit_end=FINAL_TEST_US).fit_end == FINAL_TEST_US


def test_hold_stops_fit_before_configuration_validation(learning_hold, monkeypatch):
    learning_hold(False)
    module = api()
    config = configuration()
    inspected = []

    def forbidden(*args):
        inspected.append(True)
        raise AssertionError("Se ha inspeccionado la configuración antes de la guarda")

    monkeypatch.setattr(module.MechanismFitConfig, "__post_init__", forbidden)
    with pytest.raises(LearningHoldError):
        module.fit_policy((), config=config, device="invalid")
    assert inspected == []


def test_hold_stops_signal_reader_before_budget_inspection(learning_hold):
    learning_hold(False)
    inspected = []

    class Forbidden:
        @property
        def max_sources(self):
            inspected.append(True)
            raise AssertionError("Se han inspeccionado fuentes antes de la guarda")

    with pytest.raises(LearningHoldError):
        api()._cohorts([None], config=Forbidden(), device="invalid")
    assert inspected == []
