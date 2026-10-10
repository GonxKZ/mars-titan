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
    """Ajuste: presupuesto con 16 entornos, minilotes, 17 validaciones y cuatro costes."""
    seconds = 262_144 * (step + 1 / 16000) + 256 * 4 * 16 / 100
    seconds += 17 * sessions(stage, job["scope"], job["validation"]) * (step + single)
    return seconds + 4 * sessions(stage, job["scope"], job["window"]) * (step + single)


@pytest.mark.parametrize("variant", "AB")
def test_policy_hours_follow_budget_validations_costs_and_backends(variant):
    stage = policy_plan.load_stage(STAGES[variant])
    estimate = policy_throughput.policy_hours(stage, RATES)
    assert estimate["status"] == "approximate" and estimate["not_measured"]
    expected_jobs = dict(
        A=dict(fit=1368, reference=1221), B=dict(fit=456, carry=912, reference=1221)
    )
    assert estimate["jobs"] == expected_jobs[variant]
    levels = dict(
        A=dict(all_predictors=dict(fit=792, reference=1221), algorithms=dict(fit=576)),
        B=dict(
            all_predictors=dict(fit=264, carry=528, reference=1221),
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
    # En estas medidas ficticias China no tiene contabilidad nativa y se estima con la Python.
    assert estimate["stepping_backend"] == dict(US="native", CN="python")
    plan = policy_plan.plan_stage(stage)
    single = 1 / 2000
    china = [job for job in plan if job["scope"] == "CN" and job["arm"] == "klpo_terminal"]
    expected = 0.0
    for job in china:
        evaluated = 4 * sessions(stage, "CN", job["window"]) * (1 / 400 + single)
        expected += fit_seconds(stage, job, 1 / 400, single) if job["kind"] == "fit" else evaluated
    assert estimate["scopes"]["CN"]["arms"]["klpo_terminal"] * 3600 == pytest.approx(expected)
    cash = [job for job in plan if job["scope"] == "US" and job["arm"] == "cash"]
    expected = math.fsum(4 * sessions(stage, "US", job["window"]) / 1000 for job in cash)
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
    # China se mide con las reglas A nativas. Una biblioteca anterior a esas reglas
    # queda sin medida nativa y declara el motivo.
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


NATIVE = dict(
    ppo_tick_seconds=8e-4,
    double_dqn_tick_seconds=9e-4,
    klpo_transition_seconds=5e-5,
    klpo_objective_transition_seconds=1e-5,
    minibatch_seconds=2e-3,
    forward_seconds=3e-4,
    gae_seconds=1e-3,
    evaluation_step_seconds=1e-4,
    reference_step_seconds=1e-5,
)


def test_native_work_follows_each_engine():
    stage = policy_plan.load_stage(STAGES["A"])
    plan = policy_plan.plan_stage(stage)
    first = {job["arm"]: job for job in plan if job["kind"] == "fit"}
    ppo = policy_throughput.native_fit_work(stage, first["ppo_clip_full_kl"])
    # 256 recorridos de 1024 transiciones con cuatro épocas de 16 minilotes de 64.
    assert ppo == dict(
        engine="ppo",
        collected=262_144,
        ticks=16_384,
        rollouts=256,
        adam_steps=16_384,
        validations=17,
    )
    # Double DQN: un minilote por transición desde el calentamiento de 256.
    dqn = policy_throughput.native_fit_work(stage, first["double_dqn"])
    assert dqn["engine"] == "double_dqn" and dqn["adam_steps"] == 262_144 - 256
    job = first["klpo_terminal"]
    klpo = policy_throughput.native_fit_work(stage, job)
    wave = sum(sessions(stage, job["scope"], job["train"][lane % 3]) - 1 for lane in range(16))
    waves = 262_144 // wave
    every = max(1, 16_384 // wave)
    assert klpo == dict(
        engine="klpo",
        collected=waves * wave,
        waves=waves,
        adam_steps=waves,
        validations=1 + waves // every + (1 if waves % every else 0),
    )
    # Nunca supera el presupuesto de PPO y se queda por debajo en menos de una oleada.
    assert 262_144 - wave < klpo["collected"] <= 262_144


def test_native_hours_add_collection_updates_and_evaluations_per_job():
    stage = policy_plan.load_stage(STAGES["A"])
    estimate = policy_throughput.native_policy_hours(stage, dict(US=NATIVE, CN=NATIVE))
    assert estimate["status"] == "lower_bound_without_adam" and estimate["not_measured"]
    assert estimate["jobs"] == dict(fit=1368, reference=1221)
    plan = policy_plan.plan_stage(stage)
    costs = len(stage["policies"]["evaluation_costs_bps"])
    dqn = [j for j in plan if j["arm"] == "double_dqn" and j["kind"] == "fit"]
    assert len(dqn) == 144

    def evaluated(job, validations):
        days = validations * sessions(stage, job["scope"], job["validation"])
        return (days + costs * sessions(stage, job["scope"], job["window"])) * 1e-4

    # Recogida en pasos de 16 entornos y un forward y backward más dos forwards por minilote.
    total = math.fsum(
        16_384 * 9e-4 + (262_144 - 256) * (2e-3 + 2 * 3e-4) + evaluated(j, 17) for j in dqn
    )
    assert estimate["arms"]["double_dqn"] * 3600 == pytest.approx(total)
    ppo = [j for j in plan if j["arm"] == "ppo_clip_full_kl" and j["kind"] == "fit"]
    total = math.fsum(16_384 * 8e-4 + 256 * 1e-3 + 16_384 * 2e-3 + evaluated(j, 17) for j in ppo)
    assert estimate["arms"]["ppo_clip_full_kl"] * 3600 == pytest.approx(total)
    klpo = [j for j in plan if j["arm"] == "klpo_terminal"]
    total = 0.0
    for job in klpo:
        work = policy_throughput.native_fit_work(stage, job)
        total += work["collected"] * 6e-5 + evaluated(job, work["validations"])
    assert estimate["arms"]["klpo_terminal"] * 3600 == pytest.approx(total)
    references = [j for j in plan if j["arm"] == "cash"]
    cash = math.fsum(costs * sessions(stage, j["scope"], j["window"]) * 1e-5 for j in references)
    assert estimate["arms"]["cash"] * 3600 == pytest.approx(cash)
    assert estimate["hours"] == pytest.approx(math.fsum(estimate["arms"].values()))
    assert estimate["hours"] == pytest.approx(math.fsum(estimate["scopes"].values()))
    assert estimate["adam_steps"]["double_dqn"] == 144 * (262_144 - 256)
    assert estimate["adam_steps"]["ppo_clip_full_kl"] == 144 * 16_384
    assert estimate["adam_steps"]["cash"] == 0
    # Double DQN hace 16 veces más minilotes que PPO con el mismo presupuesto.
    assert estimate["arms"]["double_dqn"] > 10 * estimate["arms"]["ppo_clip_full_kl"]


@pytest.mark.parametrize("value", [None, -1.0, float("nan"), "1"])
def test_native_hours_reject_missing_or_invalid_times(value):
    stage = policy_plan.load_stage(STAGES["A"])
    damaged = dict(NATIVE, gae_seconds=value)
    with pytest.raises(ValueError, match="gae_seconds"):
        policy_throughput.native_policy_hours(stage, dict(US=NATIVE, CN=damaged))


def test_native_carry_evaluates_each_cost_without_fitting():
    stage = policy_plan.load_stage(STAGES["B"])
    estimate = policy_throughput.native_policy_hours(stage, dict(US=NATIVE, CN=NATIVE))
    assert estimate["jobs"] == dict(fit=456, carry=912, reference=1221)
    plan = policy_plan.plan_stage(stage)
    costs = len(stage["policies"]["evaluation_costs_bps"])
    carried = [j for j in plan if j["arm"] == "klpo_terminal" and j["kind"] == "carry"]
    fitted = [j for j in plan if j["arm"] == "klpo_terminal" and j["kind"] == "fit"]
    total = math.fsum(costs * sessions(stage, j["scope"], j["window"]) * 1e-4 for j in carried)
    for job in fitted:
        work = policy_throughput.native_fit_work(stage, job)
        days = work["validations"] * sessions(stage, job["scope"], job["validation"])
        days += costs * sessions(stage, job["scope"], job["window"])
        total += work["collected"] * 6e-5 + days * 1e-4
    assert estimate["arms"]["klpo_terminal"] * 3600 == pytest.approx(total)
