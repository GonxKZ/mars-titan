"""Validación real de centros, política discreta y padre congelado."""

import pyarrow as pa
import torch

from mars_titan.data.batches import atomic_parquet_batches
from mars_titan.evaluation.session_metrics import SessionErrors
from mars_titan.models.predictive_adaptation import gaussian_log_probabilities


def centers(model, batch, *, neural, device):
    if neural:
        result = model({k: torch.as_tensor(v, device=device) for k, v in batch["inputs"].items()})
    else:
        result = model(
            torch.as_tensor(batch["features"], device=device),
            torch.as_tensor(batch["parent"], dtype=torch.float64, device=device),
        )
    if result.shape != batch["target"].shape or not torch.isfinite(result).all():
        raise ValueError("Las predicciones no son finitas o no corresponden al lote")
    return result.double()


def evaluate(model, dataset, grid, *, batch_size, neural, device, stop, destination=None):
    model.eval()
    accumulators = {name: SessionErrors() for name in ("center", "median", "parent")}
    values = torch.tensor(grid.values, dtype=torch.float64, device=device)

    def tables():
        with torch.inference_mode():
            for batch in dataset.batches(
                partition="validation", condition="real", batch_size=batch_size
            ):
                if stop.requested:
                    raise InterruptedError(
                        "Validación interrumpida, sin confirmar una puntuación parcial"
                    )
                prediction = centers(model, batch, neural=neural, device=device)
                probability = (
                    gaussian_log_probabilities(prediction, values, grid.scale).exp().cpu().numpy()
                )
                predictions = dict(
                    center=prediction.cpu().numpy(),
                    median=grid.median(probability),
                    parent=batch["parent"],
                )
                for name, output in predictions.items():
                    accumulators[name].update(
                        batch["market"], batch["prediction_at"], output - batch["target"]
                    )
                yield pa.table(
                    dict(
                        sample_id=batch["sample_ids"],
                        market=batch["market"],
                        prediction_at=pa.array(
                            batch["prediction_at"], type=pa.timestamp("us", tz="UTC")
                        ),
                        target=batch["target"],
                        prediction=predictions["median"],
                        **predictions,
                    )
                )

    if destination is None:
        for _ in tables():
            pass
    else:
        atomic_parquet_batches(destination, tables())
    summaries = {name: value.summary() for name, value in accumulators.items()}
    count = summaries["median"]["samples"]
    if count != dataset.counts["validation"] or not count:
        raise ValueError("La validación no conserva todas sus filas reales")
    for summary in summaries.values():
        summary.update(mae=summary["absolute_error"] / count, mse=summary["squared_error"] / count)
    return dict(
        **summaries["median"],
        center=summaries["center"],
        parent=summaries["parent"],
        primary="median",
    )
