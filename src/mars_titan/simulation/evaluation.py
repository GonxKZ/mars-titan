"""Evaluación sin actualizaciones sobre periodos completos y referencias fijadas."""

import math
import time


def evaluate(env, policy, *, seed=42, check_resources=None):
    """Valorar posiciones al cierre, conservando las órdenes que no se ejecutaron."""
    started = time.perf_counter()
    observation, _ = env.reset(seed=seed)
    initial = env.book.nav[env.tape.currency]
    peak, drawdown, steps, unfilled = initial, 0.0, 0, 0
    nav, reason, done = initial, None, False
    while not done:
        action = policy(observation.copy(), steps)
        observation, _, terminated, truncated, info = env.step(action)
        steps += 1
        nav = info["nav"][env.tape.currency]
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
            invalid_reason="missing_close"
            if not completed
            else "ruined"
            if reason == "ruin"
            else None,
        ),
        currency=env.tape.currency,
        cost_bps=env.cost_bps,
        unfilled_order_observations=unfilled,
        ending_positions=env.book.positions,
        pending_orders=env.book.orders,
        ruin_reward_penalty=env.ruin_penalty if reason == "ruin" else None,
        elapsed_seconds=time.perf_counter() - started,
    )


def fixed_policy(name):
    if name == "cash":
        return lambda _observation, _step: 1
    if name == "hold_initial":
        return lambda _observation, step: 5 if step == 0 else 0
    if name == "rebalance_50":
        return lambda _observation, _step: 3
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
