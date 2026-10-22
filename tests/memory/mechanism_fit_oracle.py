"""Referencia de pérdidas y ridge desde cuantiles sellados y etiquetas manuales."""

import math

import numpy as np


def interval_score(quantiles, target):
    scores = []
    for lower, upper, alpha in ((1, 3, 0.2), (0, 4, 0.05)):
        left, right = quantiles[lower], quantiles[upper]
        score = right - left
        if target < left:
            score += 2 / alpha * (left - target)
        if target > right:
            score += 2 / alpha * (target - right)
        scores.append(score)
    return math.fsum(scores) / 2


def cohort(packet, labels):
    """No utiliza el lector de ganancias ni la normalización de producción."""
    observation = packet["observation"]["observation"]
    assert len(observation["groups"]) == len(packet["groups"]) == 1
    rows = packet["rows"]
    indices = [index for index, row in enumerate(rows) if row["flow_id"] in labels]
    assert indices
    point, interval = [None] * 12, [None] * 36
    losses = []
    for candidate in packet["groups"][0]["candidates"]:
        if candidate["status"] != "observed":
            losses.append(None)
            continue
        quantiles = candidate["quantiles"]
        absolute, scores = [], []
        for index in indices:
            target = labels[rows[index]["flow_id"]]
            absolute.append(abs(target - quantiles[index][2]))
            scores.append(interval_score(quantiles[index], target))
        losses.append((math.fsum(absolute) / len(indices), math.fsum(scores) / len(indices)))
    assert losses[0] is not None
    for index, loss in enumerate(losses):
        if loss is not None:
            if index % 3 == 0:
                point[index // 3] = losses[0][0] - loss[0]
            interval[index] = losses[0][1] - loss[1]
    return dict(features=observation["groups"][0]["values"], targets=point + interval)


def ridge(rows, *, penalty, minimum, interval_weight):
    """Resolver ecuaciones normales, distintas del sistema aumentado CUDA."""
    features = np.array([row["features"] for row in rows], dtype=np.float64)
    weights = np.full(len(rows), 1 / len(rows))
    lower, upper = features.min(0), features.max(0)
    constant = lower == upper
    mean = np.array([math.fsum(column) / len(column) for column in features.T])
    mean[constant] = lower[constant]
    scale = np.sqrt(np.sum(weights[:, None] * (features - mean) ** 2, axis=0))
    scale[constant | (scale == 0)] = 1.0
    design = np.column_stack(((features - mean) / scale, np.ones(len(rows))))
    coefficients = np.zeros((22, 48))
    counts = []
    for column in range(48):
        indices = [index for index, row in enumerate(rows) if row["targets"][column] is not None]
        counts.append(len(indices))
        if column in (0, 12) or len(indices) < minimum:
            continue
        matrix = design[indices]
        target = np.array([rows[index]["targets"][column] for index in indices])
        objective_weights = weights[indices] / weights[indices].sum()
        gram = matrix.T @ (objective_weights[:, None] * matrix) + penalty * np.eye(22)
        rhs = matrix.T @ (objective_weights * target)
        coefficients[:, column] = np.linalg.solve(gram, rhs)
    available = [
        index == 0
        or (
            counts[index // 3] >= minimum
            and (interval_weight == 0 or counts[12 + index] >= minimum)
        )
        for index in range(36)
    ]
    return dict(
        mean=mean,
        scale=scale,
        support_min=lower,
        support_max=upper,
        point_coefficients=coefficients[:, :12],
        interval_coefficients=coefficients[:, 12:],
        action_mask=np.array(available),
        counts=counts,
    )
