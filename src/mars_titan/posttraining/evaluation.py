"""Validación real de centros, política discreta, cuantiles y padre congelado."""

import pyarrow as pa
import torch

from mars_titan.data.batches import atomic_parquet_batches
from mars_titan.evaluation.session_metrics import SessionErrors
from mars_titan.models.predictive_adaptation import gaussian_log_probabilities
from mars_titan.models.quantile_head import LEVELS, MEDIAN_INDEX, QUANTILE_COLUMNS, pinball_loss


def centers(model, batch, *, neural, device):
    """Centro escalar por fila o, si el modelo emite cuantiles, sus cinco niveles."""
    if neural:
        inputs = {k: torch.as_tensor(v, device=device) for k, v in batch["inputs"].items()}
        # Los lotes con máscaras llevan bits de presencia. Los estrictos, no.
        presence = batch.get("presence")
        extra = () if presence is None else (torch.as_tensor(presence, device=device),)
        result = model(inputs, *extra)
    else:
        result = model(
            torch.as_tensor(batch["features"], device=device),
            torch.as_tensor(batch["parent"], dtype=torch.float64, device=device),
        )
    shape = batch["target"].shape
    if getattr(model, "emits_quantiles", False):
        shape = (*shape, len(LEVELS))
    if result.shape != shape or not torch.isfinite(result).all():
        raise ValueError("Las predicciones no son finitas o no corresponden al lote")
    return result.double()


def _summaries(accumulators, dataset):
    summaries = {name: value.summary() for name, value in accumulators.items()}
    count = next(iter(summaries.values()))["samples"]
    if count != dataset.counts["validation"] or not count:
        raise ValueError("La validación no conserva todas sus filas reales")
    for summary in summaries.values():
        summary.update(mae=summary["absolute_error"] / count, mse=summary["squared_error"] / count)
    return summaries, count


def _write(destination, tables):
    if destination is None:
        for _ in tables:
            pass
    else:
        atomic_parquet_batches(destination, tables)


def _evaluate_quantiles(model, dataset, *, batch_size, device, stop, destination):
    """Validar cinco cuantiles con su mediana como predicción principal, como la campaña."""
    accumulators = {name: SessionErrors() for name in ("center", "parent")}
    pinball = []

    def tables():
        with torch.inference_mode():
            for batch in dataset.batches(
                partition="validation", condition="real", batch_size=batch_size
            ):
                if stop.requested:
                    raise InterruptedError(
                        "Validación interrumpida, sin confirmar una puntuación parcial"
                    )
                levels = centers(model, batch, neural=True, device=device)
                target = torch.as_tensor(batch["target"], dtype=torch.float64, device=device)
                pinball.append(float(pinball_loss(levels, target, reduction="none").sum()))
                levels = levels.cpu().numpy()
                point = levels[:, MEDIAN_INDEX]
                for name, output in (("center", point), ("parent", batch["parent"])):
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
                        prediction=point,
                        center=point,
                        parent=batch["parent"],
                        **dict(zip(QUANTILE_COLUMNS, levels.T, strict=True)),
                    )
                )

    _write(destination, tables())
    summaries, count = _summaries(accumulators, dataset)
    return dict(
        **summaries["center"],
        center=summaries["center"],
        parent=summaries["parent"],
        pinball=sum(pinball) / count,
        primary="quantile_median",
    )


def evaluate(model, dataset, grid, *, batch_size, neural, device, stop, destination=None):
    model.eval()
    if getattr(model, "emits_quantiles", False):
        return _evaluate_quantiles(
            model, dataset, batch_size=batch_size, device=device, stop=stop, destination=destination
        )
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

    _write(destination, tables())
    summaries, _ = _summaries(accumulators, dataset)
    return dict(
        **summaries["median"],
        center=summaries["center"],
        parent=summaries["parent"],
        primary="median",
    )
