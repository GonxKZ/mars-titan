"""Medir el caudal de las referencias neuronales sin pasos de optimizador y estimar horas.

La medición recorre lotes reales de la primera ventana con forward, pinball y
backward, libera los gradientes y no crea ningún optimizador. Un gancho global
rechaza cualquier paso de optimizador durante la medición. Solo se registran tiempos
y memoria, nunca pérdidas ni errores, así que no es una evaluación. Las horas se
estiman aplicando esos caudales a las filas de cada ventana del plan A o B.

Ridge y XGBoost no se miden: medir una ronda o una solución ya sería ajustar. Su
coste queda como no medido en el informe.
"""

import argparse
import json
import resource
import time
from datetime import UTC, datetime
from pathlib import Path

from .campaign_plan import FIT, NEURAL, load_campaign, plan_campaign

TRAINING_PARTITIONS = ("validation", "calibration", "evaluation", "train")


def neural_job_seconds(job, counts, rates, epochs):
    """Segundos de un trabajo neuronal con caudales en filas por segundo.

    Un ajuste recorre todas las épocas de ajuste, valida en cada época y predice
    al final validación, calibración, evaluación y el resumen del ajuste. Un traslado
    solo predice calibración y evaluación. Sin caso conocido (finalista o traslado)
    se toma el candidato más lento, una cota prudente.
    """
    candidates = [job["candidate"]] if job["candidate"] else list(rates)
    seconds = []
    for candidate in candidates:
        train, inference = rates[candidate]["train"], rates[candidate]["inference"]
        if job["kind"] == FIT:
            per_epoch = counts["train"] / train + counts["validation"] / inference
            final = sum(counts[name] for name in TRAINING_PARTITIONS) / inference
            seconds.append(epochs * per_epoch + final)
        else:
            seconds.append((counts["calibration"] + counts["evaluation"]) / inference)
    return max(seconds)


def estimate_hours(campaign, counts, rates):
    """Horas previstas por ámbito y brazo neuronal. Los tabulares quedan sin medir.

    `counts` asigna a cada ámbito y ventana sus filas por tramo y `rates` a cada brazo
    y candidato los caudales de entrenamiento e inferencia medidos.
    """
    epochs = campaign["rule"]["max_epochs"]
    scopes = {}
    for job in plan_campaign(campaign):
        if job["family"] != NEURAL:
            continue
        window = counts[job["scope"]][job["window"]]
        seconds = neural_job_seconds(job, window, rates[job["arm"]], epochs)
        scope = scopes.setdefault(job["scope"], dict(hours=0.0, arms={}))
        scope["hours"] += seconds / 3600
        scope["arms"][job["arm"]] = scope["arms"].get(job["arm"], 0.0) + seconds / 3600
    return dict(
        variant=campaign["variant"],
        retrain_every_months=campaign["retrain_every_months"],
        neural_hours=sum(scope["hours"] for scope in scopes.values()),
        scopes=scopes,
        tabular="not_measured",
        assumptions=[
            "El caudal de la primera ventana se aplica a todas las ventanas y ámbitos",
            "Finalistas y traslados usan el candidato más lento",
            "No incluye esperas de disco, reanudaciones ni otras cargas en la GPU",
        ],
    )


def _forbid_steps():
    from torch.optim.optimizer import register_optimizer_step_pre_hook

    def forbid(optimizer, args, kwargs):
        raise RuntimeError("La medición de caudal no admite pasos de optimizador")

    return register_optimizer_step_pre_hook(forbid)


def _rate(model, dataset, partition, *, train, size, seed, warmup, batches, device):
    """Filas por segundo de lotes completos tras el calentamiento, sin guardar pérdidas."""
    import torch

    from mars_titan.models.quantile_head import pinball_loss

    from .reference_run import _forward

    model.train(train)
    source = dataset.batches(partition=partition, batch_size=size, epoch=0, seed=seed)
    rows, start = 0, None
    for index, batch in enumerate(source):
        if index == warmup:
            torch.cuda.synchronize(0)
            start = time.perf_counter()
        if index == warmup + batches:
            break
        with torch.set_grad_enabled(train):
            emitted = _forward(model, batch, device)
            if train:
                target = torch.from_numpy(batch["target"]).to(device, dtype=torch.float32)
                pinball_loss(emitted, target).backward()
                # Sin optimizador: los gradientes se liberan sin tocar ningún peso.
                model.zero_grad(set_to_none=True)
        rows += len(batch["target"]) if index >= warmup else 0
    torch.cuda.synchronize(0)
    if start is None or not rows:
        raise ValueError(f"{partition} no tiene lotes suficientes para medir")
    return rows / (time.perf_counter() - start), rows


def measure_rates(campaign, view, *, batches=50, warmup=5):
    """Medir filas por segundo de cada brazo y candidato sobre una vista, sin aprender."""
    import torch

    from mars_titan.budget_training import seed_run
    from mars_titan.data.embeddings import require_cuda
    from mars_titan.models.baselines.multimodal import PRESENCE_FUSION, MultimodalReference
    from mars_titan.models.quantile_head import QUANTILE_HEAD

    from .reference_run import configured_corpus

    if (
        type(batches) is not int
        or not 1 <= batches <= 10_000
        or type(warmup) is not int
        or not 0 <= warmup <= 1000
    ):
        raise ValueError("Los lotes medidos y de calentamiento deben ser enteros acotados")
    device = require_cuda()
    dataset = configured_corpus(Path(view), input_policy=campaign["input_policy"])
    size = campaign["neural"]["batch_size"]
    rates, guard = {}, _forbid_steps()
    try:
        for arm, candidates in campaign["neural"]["candidates"].items():
            rates[arm] = {}
            for name, case in candidates:
                seed_run(case["seed"])
                first = next(
                    dataset.batches(partition="train", batch_size=1, epoch=0, seed=case["seed"])
                )
                model = MultimodalReference(
                    case["kind"],
                    {key: value.shape[-1] for key, value in first["inputs"].items()},
                    context=dataset.context,
                    mask_fusion=PRESENCE_FUSION,
                    head=QUANTILE_HEAD,
                    **case["architecture"],
                ).to(device)
                torch.cuda.reset_peak_memory_stats(0)
                options = dict(
                    size=size, seed=case["seed"], warmup=warmup, batches=batches, device=device
                )
                train, train_rows = _rate(model, dataset, "train", train=True, **options)
                inference, rows = _rate(model, dataset, "validation", train=False, **options)
                rates[arm][name] = dict(
                    train=train,
                    inference=inference,
                    measured_train_rows=train_rows,
                    measured_inference_rows=rows,
                    peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(0),
                )
    finally:
        guard.remove()
    return rates


def measure_campaigns(paths, views, first_view, *, batches=50, warmup=5):
    """Medir una vez y estimar las horas de cada variante declarada."""
    from .experiment_resources import GpuLease
    from .masked_campaign import scope_views

    campaigns = [load_campaign(path) for path in paths]
    reference = campaigns[0]
    if any(c["neural"]["candidates"] != reference["neural"]["candidates"] for c in campaigns):
        raise ValueError("Las variantes comparadas deben declarar los mismos candidatos")
    started = time.perf_counter()
    with GpuLease() as lease:
        rates = measure_rates(reference, first_view, batches=batches, warmup=warmup)
        resources = lease.record
    estimates = []
    for campaign in campaigns:
        counts = {
            scope: {
                w: v["counts"]
                for w, v in scope_views(views[scope], scope, campaign)["windows"].items()
            }
            for scope in campaign["scopes"]
        }
        estimates.append(estimate_hours(campaign, counts, rates))
    return dict(
        kind="masked_campaign_throughput",
        measured_at_utc=datetime.now(UTC).isoformat(),
        view=str(Path(first_view).resolve()),
        batches=batches,
        warmup=warmup,
        rates=rates,
        estimates=estimates,
        resources=resources,
        seconds=time.perf_counter() - started,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        optimizer_steps=0,
        scientific_training_started=False,
        final_test_opened=False,
    )


def main(argv=None):
    from .masked_campaign import _views_argument

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", type=Path, action="append", required=True)
    parser.add_argument("--views", action="append", required=True)
    parser.add_argument("--first-view", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args(argv)
    report = measure_campaigns(
        args.campaign,
        _views_argument(args.views),
        args.first_view,
        batches=args.batches,
        warmup=args.warmup,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
