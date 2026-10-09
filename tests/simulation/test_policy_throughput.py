"""Coste del entorno y de la red de las políticas, sin pasos de optimizador, y sus horas.

La estimación se comprueba con caudales fijados sobre las etapas del repositorio. La
medición recorre en CPU una etapa reducida: el entorno con un ciclo fijo de acciones y la
red con inferencia y gradientes sin optimizador. No se usa la GPU ni se registran
pérdidas.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.optim import optimizer as optim_module

from mars_titan.simulation import policy_plan, policy_throughput
from mars_titan.simulation.native_runtime import library_path

STAGES = {v: Path(f"configs/simulation/historical-masked-rl-stage-{v.lower()}.json") for v in "AB"}
RATES = dict(
    stepping={
        "US": dict(python=dict(steps_per_second=500.0), native=dict(steps_per_second=1000.0)),
        "CN": dict(python=dict(steps_per_second=400.0), native=dict(status="unavailable")),
    },
    network=dict(
        inference_transitions_per_second={"1": 2000.0, "16": 16000.0},
        gradient_minibatches_per_second=100.0,
    ),
)


def sessions(stage, scope, window):
    start, end = stage["campaign"]["comparison_config"]["resolved_scopes"][scope]["windows"][
        window
    ]["evaluation"]
    return int(np.busday_count(start, end))


def fit_seconds(stage, job, step, single):
    """Ajuste: presupuesto con 16 entornos, minilotes, 17 validaciones y tres costes."""
    seconds = 262_144 * (step + 1 / 16000) + 256 * 4 * 16 / 100
    seconds += 17 * sessions(stage, job["scope"], job["validation"]) * (step + single)
    return seconds + 3 * sessions(stage, job["scope"], job["window"]) * (step + single)


@pytest.mark.parametrize("variant", "AB")
def test_policy_hours_follow_budget_validations_costs_and_backends(variant):
    stage = policy_plan.load_stage(STAGES[variant])
    estimate = policy_throughput.policy_hours(stage, RATES)
    assert estimate["status"] == "approximate" and estimate["not_measured"]
    expected_jobs = dict(A=dict(fit=1368, reference=792), B=dict(fit=456, carry=912, reference=792))
    assert estimate["jobs"] == expected_jobs[variant]
    levels = dict(
        A=dict(all_predictors=dict(fit=792, reference=792), algorithms=dict(fit=576)),
        B=dict(
            all_predictors=dict(fit=264, carry=528, reference=792),
            algorithms=dict(fit=192, carry=384),
        ),
    )
    assert {name: value["jobs"] for name, value in estimate["levels"].items()} == levels[variant]
    assert estimate["hours"] == pytest.approx(
        math.fsum(value["hours"] for value in estimate["levels"].values())
    )
    # 262 144 transiciones en recorridos de 1024, cuatro épocas de 16 minilotes de 64 y
    # 17 validaciones: al inicio y cada 16 384 transiciones.
    assert estimate["per_fit"] == dict(
        transitions=262_144, gradient_minibatches=256 * 4 * 16, validations=17
    )
    # EE. UU. usa la contabilidad nativa medida y China la Python, sin reglas nativas.
    assert estimate["stepping_backend"] == dict(US="native", CN="python")
    plan = policy_plan.plan_stage(stage)
    single = 1 / 2000
    china = [job for job in plan if job["scope"] == "CN" and job["arm"] == "klpo_terminal"]
    expected = 0.0
    for job in china:
        evaluated = 3 * sessions(stage, "CN", job["window"]) * (1 / 400 + single)
        expected += fit_seconds(stage, job, 1 / 400, single) if job["kind"] == "fit" else evaluated
    assert estimate["scopes"]["CN"]["arms"]["klpo_terminal"] * 3600 == pytest.approx(expected)
    cash = [job for job in plan if job["scope"] == "US" and job["arm"] == "cash"]
    expected = math.fsum(3 * sessions(stage, "US", job["window"]) / 1000 for job in cash)
    assert estimate["scopes"]["US"]["arms"]["cash"] * 3600 == pytest.approx(expected)
    total = math.fsum(scope["hours"] for scope in estimate["scopes"].values())
    assert estimate["hours"] == pytest.approx(total)
    # Cada predictor del nivel completo tiene sus horas. Los comparados suman además PPO y DQN.
    us_hours = estimate["scopes"]["US"]["predictors"]
    assert len(us_hours) == 11 and us_hours["titans_mac_online"] > us_hours["titans_mac_frozen"]
    assert us_hours["rnn"] == pytest.approx(us_hours["lstm"])
    assert math.fsum(us_hours.values()) == pytest.approx(estimate["scopes"]["US"]["hours"])


def test_variant_b_costs_less_than_a_with_the_same_rates():
    hours = {
        v: policy_throughput.policy_hours(policy_plan.load_stage(STAGES[v]), RATES)["hours"]
        for v in "AB"
    }
    assert 0 < hours["B"] < hours["A"]


def test_without_native_accounting_every_market_uses_python():
    stepping = {
        market: dict(values, native=dict(status="unavailable", reason="x"))
        for market, values in RATES["stepping"].items()
    }
    stage = policy_plan.load_stage(STAGES["A"])
    estimate = policy_throughput.policy_hours(stage, dict(RATES, stepping=stepping))
    assert estimate["stepping_backend"] == dict(US="python", CN="python")
    assert estimate["hours"] > policy_throughput.policy_hours(stage, RATES)["hours"]


@pytest.fixture
def cpu(monkeypatch):
    import mars_titan.data.embeddings as embeddings

    monkeypatch.setattr(embeddings, "require_cuda", lambda **_: torch.device("cpu"))
    for name, value in dict(
        synchronize=None, reset_peak_memory_stats=None, max_memory_allocated=0
    ).items():
        monkeypatch.setattr(torch.cuda, name, lambda *_, value=value: value)


def reduced():
    stage = policy_plan.load_stage(STAGES["A"])
    policies = dict(stage["policies"], universe=dict(stage["policies"]["universe"], max_assets=8))
    policies["budget"] = dict(policies["budget"], environments=4)
    policies["hyperparameters"] = dict(policies["hyperparameters"], minibatch_size=16)
    return dict(stage, policies=policies)


def test_measurement_steps_the_environment_and_the_network_without_changing_weights(
    cpu, monkeypatch
):
    created = []
    monkeypatch.setattr(
        torch.optim.Optimizer, "__init__", lambda *args, **kwargs: created.append(args)
    )
    hooks = dict(optim_module._global_optimizer_pre_hooks)
    rates = policy_throughput.measure_policies(
        reduced(), steps=8, warmup=2, library="/nonexistent/library.so"
    )
    assert dict(optim_module._global_optimizer_pre_hooks) == hooks and created == []
    assert rates["optimizer_steps"] == 0 and rates["tape"]["assets"] == 8
    for market in ("CN", "US"):
        measured = rates["stepping"][market]
        assert measured["python"]["steps_per_second"] > 0
        assert measured["python"]["measured_steps"] == 8
        assert measured["native"]["status"] == "unavailable" and measured["native"]["reason"]
    network = rates["network"]
    assert network["observation_size"] == 6 * 8 + 2 and network["minibatch_size"] == 16
    assert set(network["inference_transitions_per_second"]) == {"1", "4"}
    assert network["gradient_minibatches_per_second"] > 0
    assert not {"loss", "reward", "return"} & set(network)


@pytest.mark.skipif(library_path() is None, reason="Falta la biblioteca nativa de simulación")
def test_native_accounting_is_measured_where_the_engine_admits_the_market(cpu):
    rates = policy_throughput.measure_policies(reduced(), steps=8, warmup=2)
    assert rates["stepping"]["US"]["native"]["steps_per_second"] > 0
    china = rates["stepping"]["CN"]["native"]
    # Mientras el motor nativo no aplique las reglas A, China se mide solo en Python.
    assert china.get("steps_per_second", 0) > 0 or "reglas" in china["reason"]


def test_a_measurement_that_changes_weights_is_rejected(cpu, monkeypatch):
    clip = torch.nn.utils.clip_grad_norm_

    def tampering(parameters, *args, **kwargs):
        parameters = list(parameters)
        result = clip(parameters, *args, **kwargs)
        with torch.no_grad():
            parameters[0].add_(1.0)
        return result

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", tampering)
    with pytest.raises(RuntimeError, match="modificado parámetros"):
        policy_throughput.measure_policies(reduced(), steps=2, warmup=0, library="/none.so")


def test_synthetic_tapes_follow_the_market_rules_of_each_scope():
    from mars_titan.simulation.campaign_stage import market_rules

    china = policy_throughput.synthetic_tape("CN", 4, sessions=5)
    assert china.currency == "CNY" and china.assets[0] == "CN/600000.SS"
    assert {rule.lot for rule in market_rules(china, "CN").values()} == {100}
    us = policy_throughput.synthetic_tape("US", 4, sessions=5)
    assert us.currency == "USD" and market_rules(us, "US") is None
    assert np.isfinite(us.prices).all() and us.prices.shape == (5, 4, 5)
