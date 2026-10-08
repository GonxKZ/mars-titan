"""Oráculo Decimal de trayectorias finitas, sin entorno, muestreo ni optimizador."""

import argparse
import json
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path

parser = argparse.ArgumentParser(description="Oráculo terminal KLPO sin aprendizaje")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
ZERO, ONE = D(0), D(1)
BETA, GAMMA = D("0.3"), D("0.83")


def table():
    q = {0: tuple(D(i) / 21 for i in (1, 2, 3, 4, 5, 6))}
    first = tuple(map(D, ("-0.5", "0.15", "0.4", "-0.8", "0.6", "-0.1")))
    paths = []
    for a in range(6):
        if a < 2:
            for noise, mass in ((D("-0.2"), D(1) / 3), (D("0.1"), D(2) / 3)):
                paths.append((q[0][a] * mass, ((0, a),), (first[a] + noise,)))
            continue
        for event, mass in ((0, D(1) / 3), (1, D(2) / 3)):
            state = 1 + 2 * (a - 2) + event
            weights = (6, 4, 2, 5, 3, 1)
            q[state] = tuple(D(weights[(b + state) % 6]) / 21 for b in range(6))
            for b in range(6):
                second = D((3 * a + 5 * b + 7 * event) % 17 - 8) / 5 + D(a * b) / 20
                for noise, ending_mass in ((D("-0.3"), D(1) / 4), (D("0.1"), D(3) / 4)):
                    paths.append(
                        (
                            q[0][a] * mass * q[state][b] * ending_mass,
                            ((0, a), (state, b)),
                            (first[a], second + noise),
                        )
                    )
    return q, paths


def features(state, action):
    return (
        D(((state + 2) * (action + 1)) % 11 - 5) / 4,
        D(((state + 3) * (action + 2)) % 13 - 6) / 5,
    )


def policies(theta, q):
    logs = {}
    for state in q:
        logits = [
            sum(t * f for t, f in zip(theta, features(state, a), strict=True))
            + D((state + action_offset(a)) % 7 - 3) / 10
            for a in range(6)
        ]
        normalizer = sum(value.exp() for value in logits).ln()
        logs[state] = tuple(value - normalizer for value in logits)
    return logs


def action_offset(action):
    return 3 * action


def reward(rewards, gamma):
    return sum(gamma**u * r for u, r in enumerate(rewards))


def population_values(q, paths, gamma):
    mass = {(state, a): ZERO for state in q for a in range(6)}
    returns = dict(mass)
    for weight, steps, rewards in paths:
        for key in steps:
            mass[key] += weight
            returns[key] += weight * reward(rewards, gamma)
    values = {key: returns[key] / mass[key] for key in mass}
    occupancy = {state: sum(mass[state, a] for a in range(6)) for state in q}
    assert abs(sum(weight for weight, _, _ in paths) - ONE) < D("1e-50")
    for state in q:
        assert max(abs(mass[state, a] - occupancy[state] * q[state][a]) for a in range(6)) < D(
            "1e-50"
        )
    return values, occupancy


def objective(theta, q, paths, gamma):
    logs = policies(theta, q)
    values, occupancy = population_values(q, paths, gamma)
    result = ZERO
    # Var(X) = sum_ab q_a*q_b*(X_a-X_b)^2 / 2.
    for state in q:
        centered_targets = [
            values[state, a] - BETA * (logs[state][a] - q[state][a].ln()) for a in range(6)
        ]
        result += (
            occupancy[state]
            * sum(
                q[state][a] * q[state][b] * (centered_targets[a] - centered_targets[b]) ** 2
                for a in range(6)
                for b in range(6)
            )
            / (4 * BETA)
        )
    return result


def sequence_objective(theta, q, paths, gamma):
    logs = policies(theta, q)
    means = {
        state: sum(q[state][a] * (logs[state][a] - q[state][a].ln()) for a in range(6))
        for state in q
    }
    return sum(
        weight
        * (
            reward(rewards, gamma)
            - BETA * sum(logs[s][a] - q[s][a].ln() - means[s] for s, a in steps)
        )
        ** 2
        for weight, steps, rewards in paths
    ) / (2 * BETA)


def surrogate_gradient(theta, q, paths, gamma, variant="terminal"):
    logs = policies(theta, q)
    mean_features = {
        state: tuple(sum(q[state][a] * features(state, a)[j] for a in range(6)) for j in range(2))
        for state in q
    }
    result = [ZERO, ZERO]
    total_mass = sum(weight for weight, _, _ in paths)
    for weight, steps, rewards in paths:
        terminal = reward(rewards, gamma)
        for u, (state, a) in enumerate(steps):
            signal = terminal
            if variant == "immediate":
                signal = rewards[u]
            if variant == "scaled_remaining":
                signal = sum(gamma**v * rewards[v] for v in range(u, len(rewards)))
            if variant == "remaining":
                signal = sum(gamma ** (v - u) * rewards[v] for v in range(u, len(rewards)))
            coefficient = signal - BETA * (logs[state][a] - q[state][a].ln())
            factor = D(len(steps)) if variant == "per_length" else ONE
            for j in range(2):
                result[j] -= (
                    weight
                    / total_mass
                    / factor
                    * coefficient
                    * (features(state, a)[j] - mean_features[state][j])
                )
    return tuple(result)


def finite_difference(function, theta):
    epsilon = D("1e-12")
    result = []
    for j in range(2):
        upper, lower = list(theta), list(theta)
        upper[j] += epsilon
        lower[j] -= epsilon
        result.append((function(upper) - function(lower)) / (2 * epsilon))
    return tuple(result)


def floats(values):
    return [float(value) for value in values]


with localcontext() as context:
    context.prec = 60
    q, paths = table()
    records = []
    for theta in (
        (D("-0.4"), D("0.7")),
        (ZERO, ZERO),
        (D("0.37"), D("-0.2")),
        (D("1.1"), D("0.3")),
    ):
        exact = finite_difference(lambda t: objective(t, q, paths, GAMMA), theta)
        sequence = finite_difference(lambda t: sequence_objective(t, q, paths, GAMMA), theta)
        terminal = surrogate_gradient(theta, q, paths, GAMMA)
        remaining = surrogate_gradient(theta, q, paths, GAMMA, "scaled_remaining")
        immediate = surrogate_gradient(theta, q, paths, GAMMA, "immediate")
        unscaled = surrogate_gradient(theta, q, paths, GAMMA, "remaining")
        length = surrogate_gradient(theta, q, paths, GAMMA, "per_length")
        accepted = [row for row in paths if reward(row[2], GAMMA) > 0]
        filtered = surrogate_gradient(theta, q, accepted, GAMMA)
        assert max(abs(a - b) for a, b in zip(exact, terminal, strict=True)) < D("1e-20")
        assert max(abs(a - b) for a, b in zip(exact, sequence, strict=True)) < D("1e-20")
        assert max(abs(a - b) for a, b in zip(terminal, remaining, strict=True)) < D("1e-50")
        for altered in (immediate, unscaled, length, filtered):
            assert max(abs(a - b) for a, b in zip(terminal, altered, strict=True)) > D("1e-4")
        records.append(
            dict(
                theta=floats(theta),
                population_fd=floats(exact),
                terminal_full=floats(terminal),
                sequence_fd=floats(sequence),
                gamma_u_remaining=floats(remaining),
                immediate=floats(immediate),
                unscaled_remaining=floats(unscaled),
                divided_by_length=floats(length),
                outcome_filtered=floats(filtered),
                max_terminal_fd_error_decimal=str(
                    max(abs(a - b) for a, b in zip(exact, terminal, strict=True))
                ),
                max_sequence_fd_error_decimal=str(
                    max(abs(a - b) for a, b in zip(exact, sequence, strict=True))
                ),
            )
        )
    q_altered = {
        state: tuple(D(1) / 6 for _ in range(6)) if state else values for state, values in q.items()
    }
    theta = (D("0.37"), D("-0.2"))
    mismatched = surrogate_gradient(theta, q_altered, paths, GAMMA)
    true = surrogate_gradient(theta, q, paths, GAMMA)
    assert max(abs(a - b) for a, b in zip(true, mismatched, strict=True)) > D("1e-4")
    report = dict(
        status="passed",
        precision_decimal_digits=60,
        actions=6,
        parameter_coordinates=2,
        prefixes=len(q),
        complete_paths=len(paths),
        lengths=sorted({len(steps) for _, steps, _ in paths}),
        terminal_noise=True,
        intermediate_random_outcomes=True,
        beta=float(BETA),
        gamma=float(GAMMA),
        fixed_points=records,
        matched_collection=dict(reference=floats(true), wrong_conditional_q=floats(mismatched)),
        computation=(
            "Enumeración de una tabla finita y diferencias finitas Decimal. "
            "Sin llamar a la implementación propuesta ni usar autograd."
        ),
        no_environment=True,
        no_sampling=True,
        no_fit=True,
        no_optimizer=True,
        no_gpu=True,
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
