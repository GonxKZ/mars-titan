"""Evaluación congelada de centros, cuantización, política y controles predictivos."""

import numpy as np
import pyarrow as pa
import torch

from mars_titan.data.batches import atomic_parquet_batches
from mars_titan.environments.actions import ActionGrid
from mars_titan.evaluation.session_metrics import SessionErrors
from mars_titan.models.klpo import behavior_log_probabilities, token_loss
from mars_titan.models.predictive_adaptation import gaussian_log_probabilities


def evaluate_predictive(
    model, dataset, batch_size, *, partition, destination=None, stop=None, klpo=None
):
    model.eval()
    grid = ActionGrid.from_dict(dataset.metadata["grid"])
    values = torch.tensor(grid.values, dtype=torch.float64, device="cuda:0")
    accumulators = {
        name: SessionErrors() for name in ("center", "nearest", "median", "parent", "zero")
    }
    totals = dict(
        samples=0,
        expected_absolute_error=0.0,
        entropy=0.0,
        extreme_mass=0.0,
        centers_outside_grid=0,
        targets_outside_grid=0,
        near_one_hot=0,
        behavior_kl_nats=0.0,
        klpo_variance=0.0,
    )

    def tables():
        with torch.inference_mode():
            for batch in dataset.batches(
                partition=partition, batch_size=batch_size, epoch=0, seed=0
            ):
                if stop is not None and stop.requested:
                    raise InterruptedError(
                        "Evaluación interrumpida antes de completar la partición"
                    )
                parent = torch.tensor(batch["parent"], dtype=torch.float64, device="cuda:0")
                centers_gpu = model(torch.as_tensor(batch["features"], device="cuda:0"), parent)
                logp = gaussian_log_probabilities(centers_gpu, values, grid.scale)
                p = logp.exp()
                target_gpu = torch.as_tensor(batch["target"], dtype=torch.float64, device="cuda:0")
                extra_columns = {}
                if klpo is not None:
                    logq = behavior_log_probabilities(
                        parent, values, grid.scale, klpo["behavior_epsilon"]
                    )
                    divergence = (logq.exp() * (logq - logp)).sum(1)
                    rewards = -(values[None, :] - target_gpu[:, None]).abs() / grid.scale
                    variance, _ = token_loss(logp, logq, rewards, klpo["beta"], "klpo_exact")
                    metrics = torch.stack((divergence, variance), dim=1).cpu().numpy()
                    extra_columns = dict(
                        behavior_kl_nats=metrics[:, 0], klpo_variance=metrics[:, 1]
                    )
                    for key, column in extra_columns.items():
                        totals[key] += float(column.sum())
                expected = (p * (values[None, :] - target_gpu[:, None]).abs()).sum(dim=1)
                entropy = -(p * logp).sum(dim=1)
                edge_mass = p[:, 0] + p[:, -1]
                tensors = torch.stack(
                    (centers_gpu, expected, entropy, edge_mass, p.max(dim=1).values), dim=1
                )
                statistics = tensors.cpu().numpy()
                centers = statistics[:, 0]
                probability = p.cpu().numpy()
                median = grid.median(probability)
                # argmin elige el índice inferior si las distancias calculadas son iguales.
                nearest = grid.values[np.abs(centers[:, None] - grid.values).argmin(axis=1)]
                target = batch["target"]
                predictions = dict(
                    center=centers,
                    nearest=nearest,
                    median=median,
                    parent=batch["parent"],
                    zero=np.zeros(len(target)),
                )
                for name, prediction in predictions.items():
                    accumulators[name].update(
                        batch["market"], batch["prediction_at"], prediction - target
                    )
                outside = (centers < grid.values[0]) | (centers > grid.values[-1])
                target_outside = (target < grid.values[0]) | (target > grid.values[-1])
                totals["samples"] += len(target)
                totals["expected_absolute_error"] += float(statistics[:, 1].sum())
                totals["entropy"] += float(statistics[:, 2].sum())
                totals["extreme_mass"] += float(statistics[:, 3].sum())
                totals["centers_outside_grid"] += int(outside.sum())
                totals["targets_outside_grid"] += int(target_outside.sum())
                totals["near_one_hot"] += int(np.sum(statistics[:, 4] >= 1 - 1e-12))
                yield pa.table(
                    dict(
                        sample_id=batch["sample_ids"],
                        asset_id=[key.rsplit("/", 1)[0] for key in batch["sample_ids"]],
                        market=batch["market"],
                        prediction_at=pa.array(
                            batch["prediction_at"], type=pa.timestamp("us", tz="UTC")
                        ),
                        target=target,
                        prediction=median,
                        center=centers,
                        nearest=nearest,
                        parent=batch["parent"],
                        zero=np.zeros(len(target)),
                        expected_absolute_error=statistics[:, 1],
                        policy_entropy=statistics[:, 2],
                        extreme_action_mass=statistics[:, 3],
                        center_outside_grid=outside,
                        target_outside_grid=target_outside,
                        **extra_columns,
                    )
                )

    if destination is None:
        for _ in tables():
            pass
    else:
        atomic_parquet_batches(destination, tables())
    count = totals["samples"]
    if count != dataset.counts[partition] or not count:
        raise ValueError("La evaluación no conserva todas las muestras de la partición")
    result = {}
    for name, accumulator in accumulators.items():
        summary = accumulator.summary()
        result[name] = dict(
            summary, mae=summary["absolute_error"] / count, mse=summary["squared_error"] / count
        )
    result["policy"] = dict(
        samples=count,
        expected_mae=totals["expected_absolute_error"] / count,
        mean_entropy_nats=totals["entropy"] / count,
        mean_extreme_action_mass=totals["extreme_mass"] / count,
        center_saturation_rate=totals["centers_outside_grid"] / count,
        target_saturation_rate=totals["targets_outside_grid"] / count,
        near_one_hot_rate=totals["near_one_hot"] / count,
        calibrated_uncertainty=False,
    )
    if klpo is not None:
        result["policy"].update(
            mean_behavior_kl_nats=totals["behavior_kl_nats"] / count,
            mean_klpo_variance=totals["klpo_variance"] / count,
        )
    return result
