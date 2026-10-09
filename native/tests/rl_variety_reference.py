"""Referencia FP64 independiente de los objetivos de grupo y de QR-DQN, sin entorno ni optimizador.

Calcula con NumPy, a partir de las ecuaciones de cada fuente, las ventajas de grupo, los pesos
de agregación, la pérdida por episodio, su gradiente analítico respecto a los logits del actor y
los diagnósticos de traza. Para QR-DQN calcula la pérdida cuantílica de Huber, su gradiente
respecto a los cuantiles predichos, los objetivos con selección doble y las puntuaciones de riesgo.
No usa autograd de ATen ni de PyTorch, así que contrasta el núcleo C++ con una derivación aparte.
Los tensores son constantes escritas aquí y no entrenan nada.

Desde la raíz del repositorio se regenera con `--output` apuntando al archivo versionado
`native/tests/fixtures/rl_variety_reference.json`.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

ACTIONS = 6


def log_softmax(z):
    shifted = z - z.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def group_advantages(returns, groups, normalize, epsilon):
    """Ventaja de resultado de DeepSeekMath (media y desviación) o de Dr. GRPO (solo media)."""
    result = np.empty_like(returns)
    for label in sorted(set(groups.tolist())):
        members = [i for i, g in enumerate(groups.tolist()) if g == label]
        total = 0.0
        for i in members:
            total += returns[i]
        mean = total / len(members)
        spread = 0.0
        for i in members:
            spread += (returns[i] - mean) ** 2
        deviation = math.sqrt(spread / (len(members) - 1))
        for i in members:
            centered = returns[i] - mean
            result[i] = centered / (deviation + epsilon) if normalize else centered
    return result


def episode_weights(lengths, aggregation, length_normalizer):
    episodes = len(lengths)
    if aggregation == "sequence_mean_token_mean":
        return np.array([1.0 / (episodes * n) if n > 0 else 0.0 for n in lengths])
    if aggregation == "sequence_mean_token_sum_constant":
        return np.array([1.0 / (episodes * length_normalizer) if n > 0 else 0.0 for n in lengths])
    if aggregation == "token_mean":
        total = float(sum(lengths))
        return np.array([1.0 / total if n > 0 else 0.0 for n in lengths])
    if aggregation == "sequence_mean":
        return np.array([1.0 / episodes if n > 0 else 0.0 for n in lengths])
    raise ValueError(aggregation)


def clipped_branch(ratio, advantage, lower, upper):
    """Verdadero cuando el mínimo elige la rama recortada y el gradiente se anula."""
    return (advantage > 0 and ratio > upper) or (advantage < 0 and ratio < lower)


def group_case(name, config, seed, lengths, groups, returns, noise):
    rng = np.random.default_rng(seed)
    batch, horizon = len(lengths), max(lengths)
    logits = rng.normal(0.0, 0.6, size=(batch, horizon, ACTIONS))
    # Desplazamiento entre el actor y el muestreador q por episodio, para cubrir las dos ramas.
    scale = np.asarray(noise, dtype=np.float64).reshape(batch, 1, 1)
    behavior_logits = logits + scale * rng.normal(0.0, 1.0, size=logits.shape)
    behavior = np.exp(log_softmax(behavior_logits))
    actions = rng.integers(0, ACTIONS, size=(batch, horizon))
    lengths = list(lengths)
    groups = np.array(groups, dtype=np.int64)
    returns = np.array(returns, dtype=np.float64)
    kind = config["kind"]
    lower, upper = 1 - config["clip_low"], 1 + config["clip_high"]
    normalize = kind != "dr_grpo"
    advantages = group_advantages(returns, groups, normalize, config["advantage_epsilon"])
    aggregation = {
        "grpo": "sequence_mean_token_mean",
        "dr_grpo": "sequence_mean_token_sum_constant",
        "dapo": "token_mean",
        "gspo": "sequence_mean",
    }[kind]
    weights = episode_weights(lengths, aggregation, config["length_normalizer"])
    logp = log_softmax(logits)
    logq = np.log(behavior)
    p = np.exp(logp)
    loss = np.zeros(batch)
    gradient = np.zeros_like(logits)
    ratios, clipped, k3_values = [], [], []
    entropy, full_kl, decisions = 0.0, 0.0, 0
    for i in range(batch):
        n = lengths[i]
        chosen = np.array([logp[i, t, actions[i, t]] for t in range(n)])
        old = np.array([logq[i, t, actions[i, t]] for t in range(n)])
        log_ratio = chosen - old
        for t in range(n):
            entropy += -(p[i, t] * logp[i, t]).sum()
            full_kl += (p[i, t] * (logp[i, t] - logq[i, t])).sum()
            delta = old[t] - chosen[t]
            k3_values.append(math.exp(delta) - delta - 1)
            decisions += 1
        a = advantages[i]
        # g[t] es la derivada de la pérdida de la oleada respecto a log pi(a_t).
        g = np.zeros(n)
        if kind == "gspo":
            ratio = math.exp(log_ratio.mean())
            surrogate = min(ratio * a, min(max(ratio, lower), upper) * a)
            loss[i] = -weights[i] * surrogate
            branch = clipped_branch(ratio, a, lower, upper)
            ratios.append(ratio)
            clipped.append(branch)
            if not branch:
                g[:] = -weights[i] * ratio * a / n
        else:
            total = 0.0
            for t in range(n):
                ratio = math.exp(log_ratio[t])
                surrogate = min(ratio * a, min(max(ratio, lower), upper) * a)
                branch = clipped_branch(ratio, a, lower, upper)
                ratios.append(ratio)
                clipped.append(branch)
                term = surrogate
                derivative = 0.0 if branch else ratio * a
                if config["kl_beta"] > 0:
                    delta = old[t] - chosen[t]
                    term -= config["kl_beta"] * (math.exp(delta) - delta - 1)
                    derivative -= config["kl_beta"] * (1 - math.exp(delta))
                total += term
                g[t] = -weights[i] * derivative
            loss[i] = -weights[i] * total
        for t in range(n):
            gradient[i, t] = g[t] * (np.eye(ACTIONS)[actions[i, t]] - p[i, t])
    for ratio in ratios:
        for boundary in (lower, upper):
            assert abs(ratio - boundary) > 1e-9, "El fixture cae en una frontera del recorte"
    deviation = float(np.sqrt(np.mean((advantages - advantages.mean()) ** 2)))
    trace = dict(
        episodes=batch,
        decisions=decisions,
        entropy=entropy / decisions,
        full_kl=full_kl / decisions,
        k3_kl=sum(k3_values) / decisions,
        ratio_mean=float(np.mean(ratios)),
        ratio_min=float(np.min(ratios)),
        ratio_max=float(np.max(ratios)),
        clip_fraction=float(np.mean(clipped)),
        advantage_mean=float(advantages.mean()),
        advantage_std=deviation,
        advantage_min=float(advantages.min()),
        advantage_max=float(advantages.max()),
    )
    return dict(
        name=name,
        config=config,
        lengths=lengths,
        groups=groups.tolist(),
        returns=returns.tolist(),
        logits=logits.tolist(),
        behavior=behavior.tolist(),
        actions=actions.tolist(),
        advantages=advantages.tolist(),
        weights=weights.tolist(),
        loss=loss.tolist(),
        gradient=gradient.tolist(),
        trace=trace,
    )


GROUP_RETURNS = [0.031, -0.012, 0.047, 0.005, -0.21, -0.18, -0.205, -0.19]
GROUP_LABELS = [0, 0, 0, 0, 1, 1, 1, 1]
GROUP_LENGTHS = [5, 3, 4, 5, 2, 5, 5, 1]
TOKEN_NOISE = [0.25] * 8
# GSPO recorta el cociente de secuencia con rangos de 1e-4: mitad de episodios casi en política.
SEQUENCE_NOISE = [0.00002, 0.3, 0.00001, 0.2, 0.00003, 0.25, 0.00002, 0.4]


def group_cases():
    base = dict(advantage_epsilon=1e-6, length_normalizer=256, kl_beta=0.0, group_size=4)
    return [
        group_case(
            "grpo",
            dict(base, kind="grpo", clip_low=0.2, clip_high=0.2, kl_beta=0.04),
            11,
            GROUP_LENGTHS,
            GROUP_LABELS,
            GROUP_RETURNS,
            TOKEN_NOISE,
        ),
        group_case(
            "dr_grpo",
            dict(base, kind="dr_grpo", clip_low=0.2, clip_high=0.2, advantage_epsilon=0.0),
            12,
            GROUP_LENGTHS,
            GROUP_LABELS,
            GROUP_RETURNS,
            TOKEN_NOISE,
        ),
        group_case(
            "dapo",
            dict(base, kind="dapo", clip_low=0.2, clip_high=0.28),
            13,
            GROUP_LENGTHS,
            GROUP_LABELS,
            GROUP_RETURNS,
            TOKEN_NOISE,
        ),
        group_case(
            "gspo",
            dict(base, kind="gspo", clip_low=0.0003, clip_high=0.0004),
            14,
            GROUP_LENGTHS,
            GROUP_LABELS,
            GROUP_RETURNS,
            SEQUENCE_NOISE,
        ),
    ]


def midpoints(count):
    return np.array([(2 * i + 1) / (2 * count) for i in range(count)])


def risk_scores(quantiles, alpha):
    """Media de los alpha*N primeros niveles tau (CVaR por punto medio). Con alpha=1 es la media."""
    atoms = int(round(alpha * quantiles.shape[-1]))
    return quantiles[..., :atoms].mean(axis=-1)


def quantile_case(name, seed, quantiles, alpha, kappa, gamma):
    rng = np.random.default_rng(seed)
    batch = 4
    online = rng.normal(0.0, 0.05, size=(batch, ACTIONS, quantiles))
    target = rng.normal(0.0, 0.05, size=(batch, ACTIONS, quantiles))
    online.sort(axis=-1)
    target.sort(axis=-1)
    current = rng.normal(0.0, 0.05, size=(batch, ACTIONS, quantiles))
    current.sort(axis=-1)
    # Un cuantil cruzado para que la puntuación use el nivel tau y no el valor ordenado.
    current[0, 2, [0, 1]] = current[0, 2, [1, 0]]
    rewards = rng.normal(0.0, 0.01, size=batch)
    terminated = np.array([False, True, False, False])
    actions = rng.integers(0, ACTIONS, size=batch)
    chosen = risk_scores(online, alpha).argmax(axis=1)
    targets = np.empty((batch, quantiles))
    for b in range(batch):
        alive = 0.0 if terminated[b] else 1.0
        targets[b] = rewards[b] + gamma * alive * target[b, chosen[b]]
    predicted = np.stack([current[b, actions[b]] for b in range(batch)])
    tau = midpoints(quantiles)
    loss = np.zeros(batch)
    gradient = np.zeros_like(predicted)
    for b in range(batch):
        for i in range(quantiles):
            for j in range(quantiles):
                u = targets[b, j] - predicted[b, i]
                assert abs(abs(u) - kappa) > 1e-9, "El fixture cae en el cambio de rama de Huber"
                huber = 0.5 * u * u if abs(u) <= kappa else kappa * (abs(u) - 0.5 * kappa)
                derivative = u if abs(u) <= kappa else kappa * math.copysign(1.0, u)
                weight = abs(tau[i] - (1.0 if u < 0 else 0.0))
                loss[b] += weight * huber / quantiles
                # du/dtheta_i = -1.
                gradient[b, i] -= weight * derivative / quantiles
    return dict(
        name=name,
        quantiles=quantiles,
        alpha=alpha,
        kappa=kappa,
        gamma=gamma,
        online=online.tolist(),
        target=target.tolist(),
        current=current.tolist(),
        rewards=rewards.tolist(),
        terminated=terminated.tolist(),
        actions=actions.tolist(),
        midpoints=tau.tolist(),
        scores=risk_scores(current, alpha).tolist(),
        chosen=chosen.tolist(),
        targets=targets.tolist(),
        predicted=predicted.tolist(),
        loss=loss.tolist(),
        gradient=gradient.tolist(),
    )


def quantile_cases():
    return [
        quantile_case("qr_mean", 21, 8, 1.0, 1.0, 0.99),
        quantile_case("qr_cvar", 22, 8, 0.25, 1.0, 0.99),
        # kappa pequeño para recorrer la rama lineal de Huber.
        quantile_case("qr_linear", 23, 8, 0.5, 0.01, 0.9),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = dict(
        schema_version=1,
        kind="rl_variety_fp64_reference",
        note="Constantes de prueba sin entorno, muestreo ni pasos de optimizador",
        group=group_cases(),
        quantile=quantile_cases(),
    )
    text = json.dumps(document, indent=1, sort_keys=True, allow_nan=False) + "\n"
    args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
