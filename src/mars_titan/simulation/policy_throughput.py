"""Medir sin aprender el coste del entorno y de la red de las políticas y estimar horas.

La medición recorre una cinta sintética con el diseño máximo de activos que admite la
regla del universo y un año de sesiones, con la contabilidad Python y con la nativa cuando
el motor la admite para el mercado. Las acciones siguen un ciclo fijo de las seis
acciones, así que no hay política ni aprendizaje. La red es `FinancialNetwork`, la
referencia Python de dos capas de 64 unidades. En `cuda:0` se mide su inferencia por
lotes, con la copia de observaciones y acciones, y el forward y backward de un minilote
del objetivo PPO con recorte de gradientes, sin optimizador. Al terminar se exige que los
pesos no hayan cambiado.

La estimación aplica esos caudales al presupuesto de transiciones, a las validaciones de
la selección y a la evaluación por coste de cada trabajo. Es orientativa: el motor nativo
recorre sus entornos en C++ sin el control de Python, que aquí sí se mide, y no se miden
el paso de Adam, las oleadas de KLPO, el replay de Double DQN ni los puntos de control.

`native_policy_hours` usa en su lugar los tiempos del motor nativo que mide
`mars-titan-policy-benchmark` sobre cintas reconstruidas y cuenta el trabajo de cada motor:
minilotes por recorrido en PPO, un minilote por transición en Double DQN y oleadas completas
con una actualización en KLPO.
"""

import math
import time
from collections import Counter

import numpy as np

from . import window_tapes
from .policy_plan import FIT, REFERENCE, VALUE_VARIANTS, WAVE_ENGINES, plan_stage

SESSIONS = 253
UNAVAILABLE = "unavailable"
ARCHITECTURE = "financial_network_tanh_64x64"
ASSUMPTIONS = [
    "Cada ajuste recorre todo el presupuesto de transiciones y valida al inicio y cada "
    "`evaluation_transitions`, con un episodio completo sobre la cinta de validación",
    "Las sesiones de cada tramo se aproximan con días hábiles, sin festivos",
    "La cinta sintética tiene el diseño máximo: universos disjuntos en cada tramo de la "
    "política. La unión medida en la edición es menor y cuesta menos",
    "El entorno nativo se usa si el motor lo admite para el mercado. Si no, la "
    "contabilidad Python hace de aproximación",
    "La red medida es la referencia Python. El motor nativo usa libtorch y la misma forma",
    "Double DQN y KLPO se estiman con los minilotes de PPO: épocas por minilotes de cada recorrido",
    "Entorno, inferencia y gradientes se suman en serie, sin solapamiento",
]
NOT_MEASURED = [
    "Paso de Adam",
    "Oleadas y referencias de KLPO",
    "Muestreo y reposición del replay de Double DQN",
    "Puntos de control, lectura de cintas y recibos",
]


def _markets(stage):
    return sorted({job["market"] for job in plan_stage(stage)})


def synthetic_tape(market, assets, sessions=SESSIONS, seed=0):
    """Cinta sintética con precios en paseo aleatorio, volumen fijo y puntuaciones aleatorias."""
    from .market import MarketTape

    rng = np.random.default_rng(seed)
    closes = 20 * np.exp(np.cumsum(rng.normal(0, 0.01, (sessions, assets)), axis=0))
    prices = np.stack(
        [closes, closes * 1.005, closes * 0.995, closes, np.full_like(closes, 1e6)], axis=2
    )
    day = 86_400_000_000
    times = 1_672_704_000_000_000 + day * np.arange(sessions, dtype=np.int64)
    if market == "CN":
        names, currency = [f"CN/{600000 + i}.SS" for i in range(assets)], "CNY"
    else:
        names, currency = [f"{market}/S{i:04d}" for i in range(assets)], "USD"
    scores = rng.normal(0, 0.01, (sessions, assets))
    return MarketTape(prices, times, names, scores, domain="synthetic", currency=currency)


def _step_rate(env, *, steps, warmup):
    env.reset(seed=42)
    episodes, start = 0, None
    for index in range(warmup + steps):
        if index == warmup:
            start = time.perf_counter()
        _, _, terminated, truncated, _ = env.step(index % 6)
        if terminated or truncated:
            episodes += 1
            env.reset(seed=42 + episodes)
    return dict(
        steps_per_second=steps / (time.perf_counter() - start),
        measured_steps=steps,
        episodes=episodes,
    )


def _layout(stage):
    policies = stage["policies"]
    return window_tapes.layout_bound(policies["universe"], policies["train_windows"])


def measure_stepping(stage, *, steps, warmup, library=None):
    """Transiciones por segundo del entorno de cada mercado con cada contabilidad."""
    from .campaign_stage import market_rules
    from .environment import FinancialEnv

    environment = {
        key: value
        for key, value in stage["policies"]["environment"].items()
        if key != "dividend_payment_lag_sessions"
    }
    assets = _layout(stage)
    result = {}
    for market in _markets(stage):
        tape = synthetic_tape(market, assets)
        result[market] = {}
        for backend in ("python", "native"):
            try:
                env = FinancialEnv(
                    tape,
                    backend=backend,
                    native_library=library if backend == "native" else None,
                    instruments=market_rules(tape, market),
                    **environment,
                )
            except (ValueError, OSError) as error:
                result[market][backend] = dict(status=UNAVAILABLE, reason=str(error))
                continue
            result[market][backend] = _step_rate(env, steps=steps, warmup=warmup)
    return result, int(6 * assets + 2)


def _timed(steps, warmup, run):
    from mars_titan.training.campaign_throughput import _synchronize

    start = None
    for index in range(warmup + steps):
        if index == warmup:
            _synchronize()
            start = time.perf_counter()
        run()
    _synchronize()
    return time.perf_counter() - start


def measure_network(stage, observations, *, steps, warmup, device):
    """Inferencia por lotes y gradientes de un minilote de la red, sin optimizador."""
    import torch
    from torch.nn import functional as F

    from .algorithms import FinancialNetwork, ppo_objective

    budget, hyper = stage["policies"]["budget"], stage["policies"]["hyperparameters"]
    torch.manual_seed(42)
    network = FinancialNetwork(observations, value_head=True).to(device)
    initial = [value.detach().clone() for value in network.parameters()]
    rng = np.random.default_rng(42)
    sampling = torch.Generator(device=device).manual_seed(42)
    torch.cuda.reset_peak_memory_stats(0)
    inference = {}
    for size in sorted({1, budget["environments"]}):
        inputs = rng.normal(size=(size, observations)).astype(np.float32)

        def act(inputs=inputs):
            with torch.inference_mode():
                logits, _ = network(torch.from_numpy(inputs).to(device))
                torch.multinomial(logits.softmax(-1), 1, generator=sampling).cpu()

        inference[str(size)] = size * steps / _timed(steps, warmup, act)
    size = hyper["minibatch_size"]
    batch = dict(
        observations=rng.normal(size=(size, observations)).astype(np.float32),
        actions=rng.integers(0, 6, size),
        advantages=rng.normal(size=size).astype(np.float32),
        returns=rng.normal(size=size).astype(np.float32),
    )
    batch = {key: torch.from_numpy(value).to(device) for key, value in batch.items()}
    old = torch.full((size,), -math.log(6), device=device)

    def gradient():
        logits, value = network(batch["observations"])
        logp = torch.log_softmax(logits, dim=-1)
        selected = logp.gather(1, batch["actions"][:, None]).squeeze(1)
        entropy = -(logp.exp() * logp).sum(1).mean()
        loss = ppo_objective(selected, old, batch["advantages"], clip=hyper["clip"])
        loss = loss + hyper["value_weight"] * F.mse_loss(value, batch["returns"])
        (loss - hyper["entropy"] * entropy).backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), hyper["gradient_norm"])
        # Sin optimizador: los gradientes se liberan sin tocar ningún peso.
        network.zero_grad(set_to_none=True)

    gradients = steps / _timed(steps, warmup, gradient)
    current = list(network.parameters())
    if not all(torch.equal(a, b.detach()) for a, b in zip(initial, current, strict=True)):
        raise RuntimeError("La medición de las políticas ha modificado parámetros")
    return dict(
        architecture=ARCHITECTURE,
        device=str(device),
        observation_size=observations,
        inference_transitions_per_second=inference,
        gradient_minibatches_per_second=gradients,
        minibatch_size=size,
        measured_steps=steps,
        peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(0),
    )


def measure_policies(stage, *, steps=2048, warmup=64, library=None):
    """Medir entorno y red de la etapa una vez, con el gancho que rechaza pasos."""
    from mars_titan.data.embeddings import require_cuda
    from mars_titan.training.campaign_throughput import _forbid_steps

    device = require_cuda()
    guard = _forbid_steps()
    try:
        stepping, observations = measure_stepping(
            stage, steps=steps, warmup=warmup, library=library
        )
        network = measure_network(stage, observations, steps=steps, warmup=warmup, device=device)
    finally:
        guard.remove()
    return dict(
        tape=dict(
            domain="synthetic",
            assets=_layout(stage),
            sessions=SESSIONS,
            actions="cycle_of_six",
        ),
        stepping=stepping,
        network=network,
        optimizer_steps=0,
    )


# Lo que la estimación con el motor nativo no puede medir mientras rige el bloqueo.
NATIVE_NOT_MEASURED = [
    "Paso de Adam: se cuentan los pasos de cada brazo, sin tiempo",
    "Muestreo del replay de Double DQN y sincronización de su red objetivo",
    "Puntos de control, lectura de cintas, recibos y arranque de cada proceso",
    "Diseños de activos mayores que el universo: los tiempos se miden con las cintas del "
    "benchmark y el diseño por fechas une los universos de todos los tramos de la política",
]
NATIVE_FIELDS = (
    "ppo_tick_seconds",
    "double_dqn_tick_seconds",
    "klpo_transition_seconds",
    "klpo_objective_transition_seconds",
    "minibatch_seconds",
    "forward_seconds",
    "gae_seconds",
    "evaluation_step_seconds",
    "reference_step_seconds",
)
# Tiempos de la cabeza cuantílica, exigidos solo cuando la etapa declara qr_dqn o qr_dqn_cvar.
# Ambas comparten red y coste, porque el CVaR solo cambia qué niveles se promedian.
QUANTILE_FIELDS = ("qr_dqn_tick_seconds", "qr_dqn_minibatch_seconds", "qr_dqn_forward_seconds")
# Objetivo de grupo por transición recogida, exigido solo si la etapa declara esos brazos. La
# recogida es la de KLPO y usa su tiempo.
GROUP_FIELDS = ("group_objective_transition_seconds",)
DQN_WARMUP = 256


def _sessions(stage, scope, window):
    """Días hábiles del tramo de evaluación de una ventana, sin festivos."""
    resolved = stage["campaign"]["comparison_config"]["resolved_scopes"]
    start, end = resolved[scope]["windows"][window]["evaluation"]
    return int(np.busday_count(start, end))


def native_fit_work(stage, job):
    """Unidades de trabajo de un ajuste nativo según su motor, sin tiempos.

    PPO recoge el presupuesto en pasos de `environments` entornos y hace épocas de
    minilotes por recorrido. Double DQN hace un minilote por transición elegible a partir
    del calentamiento, también con cabeza cuantílica. KLPO y los objetivos de grupo recogen
    oleadas completas, un episodio por entorno que recorre en ciclo las ventanas de ajuste, y
    hacen una actualización por oleada.
    """
    policies = stage["policies"]
    budget, hyper = policies["budget"], policies["hyperparameters"]
    transitions, environments = budget["transitions"], budget["environments"]
    evaluation = budget["evaluation_transitions"]
    if job["engine"] in WAVE_ENGINES:
        wave = sum(
            _sessions(stage, job["scope"], job["train"][lane % len(job["train"])]) - 1
            for lane in range(environments)
        )
        waves = transitions // wave
        every = max(1, evaluation // wave)
        return dict(
            engine="klpo" if job["engine"] == "native_klpo" else "group",
            collected=waves * wave,
            waves=waves,
            adam_steps=waves,
            validations=1 + waves // every + (1 if waves % every else 0),
        )
    rollouts = math.ceil(transitions / budget["rollout_transitions"])
    variant = policies["policies"][job["arm"]]["variant"]
    minibatches = math.ceil(budget["rollout_transitions"] / hyper["minibatch_size"])
    updates = (
        transitions - DQN_WARMUP
        if variant in VALUE_VARIANTS
        else rollouts * hyper["epochs"] * minibatches
    )
    return dict(
        engine=variant,
        collected=transitions,
        ticks=math.ceil(transitions / environments),
        rollouts=rollouts,
        adam_steps=updates,
        validations=1 + math.ceil(transitions / evaluation),
    )


def native_policy_hours(stage, measured):
    """Horas de la etapa en serie con los tiempos del motor nativo medidos sin aprendizaje.

    `measured` contiene, por mercado, los segundos de `NATIVE_FIELDS` en un dispositivo:
    un paso de recogida con todos los entornos, la recogida y el objetivo de KLPO por
    transición, el forward y backward y el forward sin gradiente de un minilote, la GAE de
    un recorrido y un paso de un carril con y sin red. Un minilote de Double DQN cuenta un
    forward y backward y dos forwards sin gradiente. QR-DQN sigue la misma cuenta con los
    tiempos de `QUANTILE_FIELDS`, que solo se exigen si la etapa declara esas variantes. Sin el
    paso de Adam, las horas son una cota inferior: el resultado cuenta esos pasos por brazo
    para completarla después.
    """
    for market, values in measured.items():
        _require_fields(values, market)
    costs = len(stage["policies"]["evaluation_costs_bps"])
    totals = dict(
        hours=0.0, adam_steps=Counter(), arms=Counter(), levels=Counter(), scopes=Counter()
    )

    def seconds(job):
        rate = measured[job["market"]]
        evaluated = costs * _sessions(stage, job["scope"], job["window"])
        if job["kind"] == REFERENCE:
            return evaluated * rate["reference_step_seconds"], 0
        total = evaluated * rate["evaluation_step_seconds"]
        if job["kind"] != FIT:
            return total, 0
        work = native_fit_work(stage, job)
        total += (
            work["validations"]
            * _sessions(stage, job["scope"], job["validation"])
            * rate["evaluation_step_seconds"]
        )
        if work["engine"] == "klpo":
            per_transition = (
                rate["klpo_transition_seconds"] + rate["klpo_objective_transition_seconds"]
            )
            return total + work["collected"] * per_transition, work["adam_steps"]
        if work["engine"] == "group":
            _require_fields(rate, job["market"], GROUP_FIELDS)
            per_transition = (
                rate["klpo_transition_seconds"] + rate["group_objective_transition_seconds"]
            )
            return total + work["collected"] * per_transition, work["adam_steps"]
        if work["engine"] == "double_dqn":
            update = rate["minibatch_seconds"] + 2 * rate["forward_seconds"]
            total += work["ticks"] * rate["double_dqn_tick_seconds"]
            return total + work["adam_steps"] * update, work["adam_steps"]
        if work["engine"] in VALUE_VARIANTS:
            _require_fields(rate, job["market"], QUANTILE_FIELDS)
            update = rate["qr_dqn_minibatch_seconds"] + 2 * rate["qr_dqn_forward_seconds"]
            total += work["ticks"] * rate["qr_dqn_tick_seconds"]
            return total + work["adam_steps"] * update, work["adam_steps"]
        total += work["ticks"] * rate["ppo_tick_seconds"] + work["rollouts"] * rate["gae_seconds"]
        return total + work["adam_steps"] * rate["minibatch_seconds"], work["adam_steps"]

    jobs = plan_stage(stage)
    for job in jobs:
        value, steps = seconds(job)
        hours = value / 3600
        totals["hours"] += hours
        totals["arms"][job["arm"]] += hours
        totals["levels"][job["level"]] += hours
        totals["scopes"][job["scope"]] += hours
        totals["adam_steps"][job["arm"]] += steps
    return dict(
        status="lower_bound_without_adam",
        hours=totals["hours"],
        jobs=dict(Counter(job["kind"] for job in jobs)),
        arms=dict(totals["arms"]),
        levels=dict(totals["levels"]),
        scopes=dict(totals["scopes"]),
        adam_steps=dict(totals["adam_steps"]),
        not_measured=NATIVE_NOT_MEASURED,
    )


def _require_fields(values, market, fields=NATIVE_FIELDS):
    missing = [
        name
        for name in fields
        if type(values.get(name)) not in (int, float)
        or not math.isfinite(values[name])
        or values[name] < 0
    ]
    if missing:
        raise ValueError(f"Faltan tiempos nativos de {market}: {', '.join(missing)}")


def policy_hours(stage, rates):
    """Horas orientativas de cada ajuste, traslado y referencia de la etapa, por nivel.

    Un ajuste recorre el presupuesto con `environments` entornos por lote, calcula los
    minilotes de cada recorrido, valida al inicio y cada `evaluation_transitions` y evalúa
    cada coste. Un traslado evalúa cada coste con inferencia y una referencia sin ella.
    """
    policies = stage["policies"]
    budget, hyper = policies["budget"], policies["hyperparameters"]
    network = rates["network"]
    acting = 1 / network["inference_transitions_per_second"][str(budget["environments"])]
    single = 1 / network["inference_transitions_per_second"]["1"]
    rollouts = math.ceil(budget["transitions"] / budget["rollout_transitions"])
    minibatches = rollouts * hyper["epochs"]
    minibatches *= math.ceil(budget["rollout_transitions"] / hyper["minibatch_size"])
    validations = 1 + math.ceil(budget["transitions"] / budget["evaluation_transitions"])
    costs = len(policies["evaluation_costs_bps"])
    step, backends = {}, {}
    for market, measured in rates["stepping"].items():
        backends[market] = "native" if "steps_per_second" in measured["native"] else "python"
        step[market] = 1 / measured[backends[market]]["steps_per_second"]

    def sessions(scope, window):
        return _sessions(stage, scope, window)

    def seconds(job):
        per = step[job["market"]]
        evaluated = costs * sessions(job["scope"], job["window"])
        if job["kind"] == REFERENCE:
            return evaluated * per
        total = evaluated * (per + single)
        if job["kind"] == FIT:
            total += budget["transitions"] * (per + acting)
            total += minibatches / network["gradient_minibatches_per_second"]
            total += validations * sessions(job["scope"], job["validation"]) * (per + single)
        return total

    jobs = plan_stage(stage)
    scopes, levels = {}, {}
    for job in jobs:
        hours = seconds(job) / 3600
        scope = scopes.setdefault(job["scope"], dict(hours=0.0, arms={}, predictors={}))
        scope["hours"] += hours
        scope["arms"][job["arm"]] = scope["arms"].get(job["arm"], 0.0) + hours
        predictor = job["predictor"]
        scope["predictors"][predictor] = scope["predictors"].get(predictor, 0.0) + hours
        level = levels.setdefault(job["level"], dict(hours=0.0, jobs=Counter()))
        level["hours"] += hours
        level["jobs"][job["kind"]] += 1
    return dict(
        status="approximate",
        hours=math.fsum(scope["hours"] for scope in scopes.values()),
        jobs=dict(Counter(job["kind"] for job in jobs)),
        levels={name: dict(value, jobs=dict(value["jobs"])) for name, value in levels.items()},
        scopes=scopes,
        per_fit=dict(
            transitions=budget["transitions"],
            gradient_minibatches=minibatches,
            validations=validations,
        ),
        stepping_backend=backends,
        assumptions=ASSUMPTIONS,
        not_measured=NOT_MEASURED,
    )
