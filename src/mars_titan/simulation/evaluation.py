"""Evaluación sin actualizaciones sobre periodos completos y referencias fijadas."""

import math
import time

from .environment import ALLOCATIONS

# La venta hipotética al último cierre paga `cost_bps` y el impuesto de venta vigente en la
# fecha de ese cierre, como `liquidated_nav` en el motor nativo.
LIQUIDATION_BASIS = "final_close_minus_cost_bps_and_sell_taxes"
# La serie de patrimonio valora cada cierre con la contabilidad de la cartera. El motor
# nativo de políticas la reconstruye con sus recompensas logarítmicas y declara otra base.
EQUITY_BASIS = "close_valuation"


def _exit_costs(env):
    """Coste de vender todas las posiciones al último cierre con su impuesto de venta."""
    at = int(env.tape.close_times[env.cursor])
    held, taxes = [], []
    for asset, quantity in env.book.positions.items():
        value = quantity * env.tape.prices[env.cursor, env.tape.assets.index(asset), 3]
        held.append(value)
        rate = env.instruments[asset].tax(at, "sell")
        if rate:
            taxes.append(value * rate)
    # Mismo orden de operaciones que el motor nativo: el coste sobre la suma y después el timbre.
    return math.fsum(held) * env.cost_bps / 10000 + math.fsum(taxes)


def evaluate(env, policy, *, seed=42, check_resources=None):
    """Valorar posiciones al cierre, conservando las órdenes que no se ejecutaron."""
    started = time.perf_counter()
    observation, _ = env.reset(seed=seed)
    initial = env.book.nav[env.tape.currency]
    peak, drawdown, steps, unfilled = initial, 0.0, 0, 0
    nav, reason, done = initial, None, False
    # Patrimonio contable en cada cierre: None si falta una valoración, cero tras la ruina.
    series = [initial]
    while not done:
        action = policy(observation.copy(), steps)
        observation, _, terminated, truncated, info = env.step(action)
        steps += 1
        nav = info["nav"][env.tape.currency]
        series.append(nav)
        unfilled += len(info["unfilled"])
        reason = info["reason"]
        if nav is not None:
            peak = max(peak, nav)
            drawdown = max(drawdown, 1 - nav / peak)
        done = terminated or truncated
        if check_resources:
            check_resources()
    state = env.book.snapshot()["state"]
    completed = nav is not None
    net_return = nav / initial - 1 if completed else None
    # La recompensa valora al cierre sin pagar la salida. Se informa aparte, sin cambiarla.
    exit_costs = _exit_costs(env) if completed else None
    if completed and (not math.isfinite(net_return) or net_return < -1 or not 0 <= drawdown <= 1):
        raise ValueError("La valoración no conserva los límites de una cartera sin deuda")
    return dict(
        schema_version=1,
        activity="evaluation",
        domain=env.tape.domain,
        partition=env.tape.partition,
        identity=env.identity,
        seed=seed,
        parent_frozen=True,
        final_test_opened=False,
        financial_validation=dict(
            net_return=net_return,
            max_drawdown=drawdown if completed else None,
            costs=state["costs"][env.tape.currency],
            turnover=state["turnover"][env.tape.currency] / initial,
            steps=steps,
            completed=completed,
            invalid_reason=reason if not completed else "ruined" if reason == "ruin" else None,
        ),
        currency=env.tape.currency,
        cost_bps=env.cost_bps,
        unfilled_order_observations=unfilled,
        ending_positions=env.book.positions,
        pending_orders=env.book.orders,
        ruin_reward_penalty=env.ruin_penalty if reason == "ruin" else None,
        terminal_liquidation=dict(
            basis=LIQUIDATION_BASIS,
            estimated_costs=exit_costs,
            net_return=(nav - exit_costs) / initial - 1 if completed else None,
        ),
        equity=dict(
            basis=EQUITY_BASIS,
            close_times=[int(value) for value in env.tape.close_times[: steps + 1]],
            nav=series,
        ),
        elapsed_seconds=time.perf_counter() - started,
    )


# Sesiones entre dos reequilibrios de la cartera 1/N, unas cuatro semanas de mercado.
REBALANCE_SESSIONS = 21
# Regla de composición de cada referencia. Efectivo, comprar y mantener y la regla fija del
# 50 % usan el cuartil superior de puntuaciones del predictor. La cartera 1/N y el índice de
# mercado reparten por igual entre los activos valorados de su cinta y no usan predicciones.
REFERENCE_ALLOCATIONS = dict(
    cash=ALLOCATIONS[0],
    hold_initial=ALLOCATIONS[0],
    rebalance_50=ALLOCATIONS[0],
    equal_weight_monthly=ALLOCATIONS[1],
    market_index=ALLOCATIONS[1],
)


def fixed_policy(name):
    if name == "cash":
        return lambda _observation, _step: 1
    if name in ("hold_initial", "market_index"):
        return lambda _observation, step: 5 if step == 0 else 0
    if name == "rebalance_50":
        return lambda _observation, _step: 3
    if name == "equal_weight_monthly":
        return lambda _observation, step: 5 if step % REBALANCE_SESSIONS == 0 else 0
    raise ValueError("La referencia financiera no está definida")


def learned_policy(trainer):
    import torch

    trainer.network.eval()

    def policy(observation, _step):
        with torch.no_grad():
            output = trainer.network(trainer._tensor(observation)[None])
            scores = output[0] if trainer.algorithm == "ppo" else output
            return int(scores.argmax(1).item())

    return policy
