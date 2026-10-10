"""Medir el caudal de las familias de la campaña sin pasos de optimizador y estimar horas.

La medición recorre datos reales de la primera ventana. Las referencias neuronales y los
adaptadores de la matriz de postentrenamiento calculan forward, pinball y backward por
lotes. Titans-MAC, la GRU candidata, el lector episódico de MARS-TITAN y los núcleos y
lectores de CM-v1 recorren su ajuste cronológico real por tramos, con un optimizador que
solo cuenta los pasos pedidos y libera los gradientes, y su inferencia por eventos con los
parámetros congelados. Un gancho global rechaza cualquier paso de un optimizador de
PyTorch y al terminar se exige que los pesos no hayan cambiado, también los del padre
congelado de un lector. Solo se registran tiempos, memoria y contadores, nunca pérdidas
ni errores, así que no es una evaluación.

Titans-MAC y los núcleos de CM-v1, que comparten su receta y también acumulan con la
penalización C, comparan `accumulation_rows`, y la GRU candidata `accumulation_rows` y
`recompute`, con la misma medida. Los lectores no tienen opciones de memoria y su medida
vale para cada opción de su familia. Una opción que no cabe en la memoria reservada queda
registrada como tal. Las horas se estiman aplicando los caudales a las filas de cada
ventana de las variantes A y B, por familia y por opción, incluida la etapa de la matriz
de adaptadores. Con una declaración preparada (`campaign_extensions`), las familias que A
y B todavía no declaran se miden y se estiman como si lo estuvieran.

Con las etapas de políticas, `simulation.policy_throughput` mide además el entorno
financiero y la red de las políticas por lotes, sin pasos de optimizador, y añade a cada
variante una estimación orientativa de esa etapa por nivel, ámbito, brazo y predictor,
separada de las horas de GPU.

Con las etapas de ablación de modalidades no se mide nada más. Cada predicción
enmascarada recorre la evaluación de su ventana, con el calentamiento en las familias
cronológicas, al caudal de inferencia ya medido de su brazo. Es otra estimación separada.

Ridge y XGBoost no se miden: medir una ronda o una solución ya sería ajustarlos. Su
coste queda como no medido en el informe.
"""

import argparse
import gc
import json
import os
import resource
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.input_policy import HISTORICAL_MASKED

from . import campaign_numerics
from .campaign_plan import (
    CM,
    CM_ARMS,
    CM_CORES,
    EPISODIC,
    FIT,
    MARS,
    NEURAL,
    TITANS,
    _arm_specs,
    extend_campaign,
    load_campaign,
    plan_campaign,
)

TRAINING_PARTITIONS = ("validation", "calibration", "evaluation", "train")
POSTTRAINING = "posttraining_adapter_matrix"
POLICY_STAGE = "rl_policy_comparison"
ABLATION_STAGE = "modality_ablation"
NOT_MEASURED = "not_measured"
OUT_OF_MEMORY = "out_of_memory"
# Opciones de memoria comparadas con la misma medida. Se mide además la de la receta.
TITANS_OPTIONS = ({"accumulation_rows": None}, {"accumulation_rows": 128})
CANDIDATE_OPTIONS = (
    {"accumulation_rows": None, "recompute": False},
    {"accumulation_rows": 128, "recompute": False},
    {"accumulation_rows": None, "recompute": True},
    {"accumulation_rows": 128, "recompute": True},
)
# Única opción de los lectores: la de su receta.
RECIPE_ONLY = ({},)
# Familias que recorren el ajuste cronológico y sus fases con calentamiento.
CHRONOLOGICAL = (TITANS, EPISODIC, MARS, CM)
# Familias que la orden puede añadir para medir con `with_candidate` o `campaign_extensions`.
PREPARED = (EPISODIC, MARS, CM)
# Contadores de la ventana medida: flujos de C en el núcleo y episodios en el banco del lector.
CONTROL_COUNTERS = ("control_groups", "control_flows")
BANK_COUNTERS = ("admitted",)
_LIMITS = dict(
    batches=(1, 10_000),
    warmup=(0, 1000),
    segments=(1, 10_000),
    segment_warmup=(0, 1000),
    events=(1, 100_000),
    event_warmup=(0, 10_000),
    policy_steps=(1, 100_000),
    policy_warmup=(0, 10_000),
)
SETTINGS = dict(
    batches=50,
    warmup=5,
    segments=8,
    segment_warmup=2,
    events=64,
    event_warmup=8,
    policy_steps=2048,
    policy_warmup=64,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bounded(**values):
    _require(
        all(
            type(value) is int and _LIMITS[name][0] <= value <= _LIMITS[name][1]
            for name, value in values.items()
        ),
        "Los lotes, tramos y eventos medidos y de calentamiento deben ser enteros acotados",
    )


def option_name(option):
    """Nombre estable de una opción de memoria, por ejemplo `accumulation_rows=128`.

    La opción vacía es la de la receta, sin alternativas que comparar.
    """
    return ",".join(f"{key}={json.dumps(value)}" for key, value in option.items()) or "recipe"


def _options(recipe, alternatives):
    """La opción de la receta primero y después las alternativas distintas."""
    declared = {key: getattr(recipe, key) for key in alternatives[0]}
    return [declared, *(option for option in alternatives if option != declared)]


# Estimación de horas


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


def validated_job_seconds(job, rows, rate, epochs):
    """Segundos de un trabajo que valida antes de la primera época y tras cada una.

    Es la forma de Titans-MAC, de la GRU candidata y de los casos de la matriz. Un ajuste
    recorre `epochs` veces el tramo de ajuste, valida `epochs + 1` veces, vuelve a
    predecir la validación con el estado elegido y predice calibración y evaluación. Un
    traslado predice calibración y evaluación. `rows` incluye el calentamiento si existe.
    """
    inference = rate["inference"]
    if job["kind"] != FIT:
        return (rows["calibration"] + rows["evaluation"]) / inference
    if rate["train"] is None:
        # Un trabajo sin ajuste, como la corrección B6, predice una sola vez cada tramo medido,
        # sin épocas ni validaciones repetidas.
        return (rows["validation"] + rows["calibration"] + rows["evaluation"]) / inference
    predicted = (epochs + 2) * rows["validation"] + rows["calibration"] + rows["evaluation"]
    return epochs * rows["train"] / rate["train"] + predicted / inference


def titans_rows(counts, fold, warmup_months):
    """Filas que recorre Titans-MAC en cada tramo de una ventana.

    Validación, calibración y evaluación observan antes sus `warmup_months` previos sin
    etiquetas. Esas filas se estiman con la densidad del propio tramo en el tiempo.
    """
    from .titans_walk_forward import PREDICTED, window_phases

    phases = window_phases(fold, warmup_months)
    rows = dict(counts)
    for name in PREDICTED:
        phase = phases[name]
        share = (phase.decision_start - phase.warmup_start) / (
            phase.decision_end - phase.decision_start
        )
        rows[name] = counts[name] + round(counts[name] * share)
    return rows


def _slowest(records):
    """Caudales más lentos de varios candidatos. Es la cota prudente de un caso sin elegir."""
    return {key: min(record[key] for record in records) for key in ("train", "inference")}


def _hours(jobs, seconds):
    scopes = {}
    for job in jobs:
        hours = seconds(job) / 3600
        scope = scopes.setdefault(job["scope"], dict(hours=0.0, arms={}))
        scope["hours"] += hours
        scope["arms"][job["arm"]] = scope["arms"].get(job["arm"], 0.0) + hours
    return dict(
        hours=sum(scope["hours"] for scope in scopes.values()),
        training_jobs=sum(job["kind"] == FIT for job in jobs),
        prediction_jobs=sum(job["kind"] != FIT for job in jobs),
        scopes=scopes,
    )


def _chronological_rows(campaign, family, counts):
    """Filas por ventana de una familia cronológica, con el calentamiento de su receta Titans.

    Los lectores de MARS-TITAN usan las fases de su padre Titans-MAC y los núcleos y
    lectores de CM-v1 las de la receta del núcleo. La GRU candidata no calienta.
    """
    if family == EPISODIC:
        return counts
    from .financial_run import load_recipe
    from .titans_walk_forward import walk_forward_options

    path = campaign[CM]["recipes"]["core_recipe"] if family == CM else campaign[TITANS]["path"]
    months = walk_forward_options(load_recipe(path)[1])["warmup_months"]
    resolved = campaign["comparison_config"]["resolved_scopes"]
    return {
        scope: {
            window: titans_rows(rows, resolved[scope]["windows"][window], months)
            for window, rows in windows.items()
        }
        for scope, windows in counts.items()
    }


def _option_hours(campaign, family, jobs, counts, measured, epochs):
    """Horas de una familia cronológica con cada opción de memoria medida.

    Un brazo medido solo con su receta, como un lector de CM-v1, usa esa medida en cada
    opción de los brazos de su familia que comparan opciones, como sus núcleos.
    """
    rows = _chronological_rows(campaign, family, counts)
    compared = {
        arm: record for arm, record in measured.items() if list(record["options"]) != ["recipe"]
    } or measured
    declared = {record["declared_option"] for record in compared.values()}
    _require(len(declared) == 1, "Los brazos de una familia deben declarar la misma opción")
    result = dict(declared=declared.pop(), options={})
    for name in next(iter(compared.values()))["options"]:
        records = {
            arm: record["options"][name if arm in compared else "recipe"]
            for arm, record in measured.items()
        }
        peak = max(record["peak_vram_allocated_bytes"] for record in records.values())
        if any(record.get("status") == OUT_OF_MEMORY for record in records.values()):
            result["options"][name] = dict(status=OUT_OF_MEMORY, peak_vram_allocated_bytes=peak)
            continue

        def seconds(job, records=records):
            rate = dict(
                train=records[job["arm"]]["train"], inference=measured[job["arm"]]["inference"]
            )
            return validated_job_seconds(job, rows[job["scope"]][job["window"]], rate, epochs)

        result["options"][name] = dict(_hours(jobs, seconds), peak_vram_allocated_bytes=peak)
    return result


def _posttraining_hours(stage, counts, rates):
    """Horas de la etapa de adaptadores, con la caché de cada padre ajustado.

    En el walk-forward por etapas cada caso ajusta solo las filas nuevas de su ventana. Se
    estiman con los recuentos como el tramo de ajuste de la ventana menos ajuste,
    validación y calibración de la anterior, una cota algo mayor porque las filas que la
    purga quitó en las fronteras de la ventana anterior sí están en el ajuste de la nueva.
    Cada padre predice una vez esas filas y la validación para su caché, y el padre
    congelado predice validación, calibración y evaluación, ambos con la inferencia neuronal
    más lenta del brazo base. En el plan anclado de B el ajuste recorre todo su tramo.

    La medida de la matriz solo cubre las redes de referencia. Los brazos de Titans-MAC y la
    cadena trivial de Ridge y XGBoost quedan en `without_estimate`, sin sumar sus horas.
    """
    from mars_titan.posttraining.campaign_stage import FROZEN, plan_stage

    measured, neural = rates[POSTTRAINING], rates[NEURAL]
    epochs = stage["matrix"]["budget"]["epochs"]
    jobs, missing = [], set()
    for job in plan_stage(stage):
        if job["base_arm"] in measured and job["base_arm"] in neural:
            jobs.append(job)
        else:
            missing.add(job["base_arm"])
    parents = {}
    for job in jobs:
        if job["kind"] == FIT:
            parents.setdefault((job["scope"], job["window"], job["base_arm"], job["seed"]), job)
    first = {job["id"] for job in parents.values()}

    def rows_of(job):
        rows = counts[job["scope"]][job["window"]]
        if job.get("parent_window") is None:
            return rows
        previous = counts[job["scope"]][job["parent_window"]]
        fresh = rows["train"] - sum(
            previous[name] for name in ("train", "validation", "calibration")
        )
        _require(fresh > 0, f"{job['id']} no tiene filas nuevas en los recuentos")
        return dict(rows, train=fresh)

    def seconds(job):
        rows = rows_of(job)
        cache = _slowest(neural[job["base_arm"]].values())["inference"]
        if job["kind"] == FROZEN:
            return (rows["validation"] + rows["calibration"] + rows["evaluation"]) / cache
        rate = _slowest([points[job["point"]] for points in measured[job["base_arm"]].values()])
        total = validated_job_seconds(job, rows, rate, epochs)
        if job["id"] in first:
            total += (rows["train"] + rows["validation"]) / cache
        return total

    return dict(_hours(jobs, seconds), parent_caches=len(first), without_estimate=sorted(missing))


def _ablation_hours(campaign, stage, counts, rates):
    """Horas orientativas de la ablación de modalidades, que no ajusta nada.

    Cada trabajo predice una vez la evaluación de su ventana con el estado elegido. Las
    familias cronológicas recorren además el calentamiento de esa evaluación. Se usa la
    inferencia más lenta medida del brazo. Tabulares y familias sin medir quedan sin estimar.
    """
    from .modality_ablation_stage import plan_stage

    _require(
        stage["campaign"]["path"] == campaign["path"],
        "La etapa de ablación no parte de esta campaña",
    )
    rows = {
        family: _chronological_rows(campaign, family, counts)
        for family in CHRONOLOGICAL
        if family in rates
    }
    jobs, missing = [], set()
    for job in plan_stage(stage):
        if job["family"] in rows or (job["family"] == NEURAL and NEURAL in rates):
            # Una predicción enmascarada nunca es un ajuste.
            jobs.append(dict(job, kind="ablation"))
        else:
            missing.add(job["arm"])

    def seconds(job):
        if job["family"] == NEURAL:
            inference = _slowest(rates[NEURAL][job["arm"]].values())["inference"]
            return counts[job["scope"]][job["window"]]["evaluation"] / inference
        inference = rates[job["family"]][job["arm"]]["inference"]
        return rows[job["family"]][job["scope"]][job["window"]]["evaluation"] / inference

    return dict(_hours(jobs, seconds), without_estimate=sorted(missing))


def _totals(families):
    """Horas de GPU con las opciones declaradas y con las más rápidas que caben en memoria.

    Una familia con brazos sin estimar suma las horas de los demás y también queda en
    `without_estimate`, para que el total no parezca completo.
    """
    declared, fastest, missing = 0.0, 0.0, []
    for name, family in families.items():
        if "hours" in family:
            declared += family["hours"]
            fastest += family["hours"]
            if family.get("without_estimate"):
                missing.append(name)
            continue
        fitted = [
            option["hours"] for option in family.get("options", {}).values() if "hours" in option
        ]
        if not fitted:
            missing.append(name)
            continue
        fastest += min(fitted)
        chosen = family["options"][family["declared"]].get("hours")
        declared = None if declared is None or chosen is None else declared + chosen
    return dict(declared_options=declared, fastest_options=fastest, without_estimate=missing)


def estimate_hours(
    campaign, counts, rates, *, stage=None, policy_stage=None, ablation_stage=None, epochs=None
):
    """Horas previstas por familia, opción, ámbito y brazo de una variante.

    `counts` asigna a cada ámbito y ventana sus filas por tramo y `rates` a cada familia
    medida sus caudales. Las familias declaradas sin medir y los tabulares quedan como no
    medidos. Con `stage`, añade la etapa de la matriz de adaptadores de esa variante. Con
    `policy_stage`, añade aparte la estimación orientativa de la etapa de políticas y con
    `ablation_stage`, la de la ablación de modalidades. `epochs` sustituye las épocas de la
    regla de parada, por ejemplo con las épocas efectivas previstas de una parada temprana.
    """
    epochs = campaign["rule"]["max_epochs"] if epochs is None else epochs
    jobs = plan_campaign(campaign)
    families = {}
    for family in (NEURAL, *CHRONOLOGICAL):
        selected = [job for job in jobs if job["family"] == family]
        if not selected:
            continue
        measured = rates.get(family)
        if measured is None:
            families[family] = dict(status=NOT_MEASURED)
        elif family == NEURAL:
            families[family] = _hours(
                selected,
                lambda job, measured=measured: neural_job_seconds(
                    job, counts[job["scope"]][job["window"]], measured[job["arm"]], epochs
                ),
            )
        else:
            families[family] = _option_hours(campaign, family, selected, counts, measured, epochs)
    specs = {spec["arm"]: spec for spec in _arm_specs(campaign)}
    # Familias que A y B todavía no declaran: el informe dice de dónde viene su sección.
    for family in PREPARED:
        if campaign.get(family) and family in families:
            families[family]["declared_in_campaign"] = campaign[family].get(
                "declared_in_campaign", True
            )
            # Cada lector parte del padre elegido en su ventana. Sus horas se suman en la
            # familia del padre: Titans-MAC para MARS-TITAN y los núcleos dentro de CM-v1.
            parents = {
                arm: dict(parent=spec["parent"], counted_in=specs[spec["parent"]]["family"])
                for arm, spec in specs.items()
                if spec["family"] == family and spec["parent"]
            }
            if parents:
                families[family]["parents"] = parents
    if stage is not None:
        _require(
            stage["campaign"]["path"] == campaign["path"],
            "La etapa de adaptadores no parte de esta campaña",
        )
        families[POSTTRAINING] = (
            _posttraining_hours(stage, counts, rates)
            if POSTTRAINING in rates and NEURAL in rates
            else dict(status=NOT_MEASURED)
        )
    extra = {}
    if policy_stage is not None:
        from mars_titan.simulation.policy_throughput import policy_hours

        _require(
            policy_stage["campaign"]["path"] == campaign["path"],
            "La etapa de políticas no parte de esta campaña",
        )
        extra[POLICY_STAGE] = (
            policy_hours(policy_stage, rates[POLICY_STAGE])
            if POLICY_STAGE in rates
            else dict(status=NOT_MEASURED)
        )
    if ablation_stage is not None:
        extra[ABLATION_STAGE] = _ablation_hours(campaign, ablation_stage, counts, rates)
    return dict(
        variant=campaign["variant"],
        retrain_every_months=campaign["retrain_every_months"],
        families=families,
        **extra,
        tabular=NOT_MEASURED,
        total_gpu_hours=_totals(families),
        assumptions=[
            "El caudal de la primera ventana se aplica a todas las ventanas y ámbitos",
            "Finalistas, traslados y padres sin elegir usan el candidato más lento",
            "Los casos de búsqueda de Titans-MAC solo cambian hiperparámetros del optimizador "
            "y comparten el caudal de su control",
            "El calentamiento de Titans-MAC se estima con la densidad del tramo que precede",
            "Los lectores de MARS-TITAN y CM-v1 se miden sobre un padre con pesos iniciales, "
            "con la misma arquitectura y el mismo cálculo que el padre elegido",
            "Las horas de un lector no incluyen el ajuste de su padre, que se cuenta en la "
            "familia del padre, y los núcleos de CM-v1 se cuentan en CM-v1",
            "La corrección B6 no ajusta: cada trabajo predice una vez validación, calibración "
            "y evaluación con el caudal medido sobre eventos de ajuste, que siempre corrigen",
            "Con presupuesto fijo, cada ajuste recorre todas sus épocas",
            "No incluye esperas de disco, índices, normalizadores, reanudaciones ni otras "
            "cargas en la GPU",
        ],
    )


# Medición


def _forbid_steps():
    from torch.optim.optimizer import register_optimizer_step_pre_hook

    def forbid(optimizer, args, kwargs):
        raise RuntimeError("La medición de caudal no admite pasos de optimizador")

    return register_optimizer_step_pre_hook(forbid)


class _NoStepOptimizer:
    """Optimizador de la medición: cuenta los pasos pedidos y solo libera los gradientes.

    No hereda de `torch.optim.Optimizer`, así que el recorrido llega hasta el paso sin
    modificar ningún peso. `unchanged` lo comprueba frente a una copia inicial de los
    parámetros ajustables y de los `frozen`, como el padre congelado de un lector.
    """

    def __init__(self, groups, frozen=()):
        self.param_groups = [dict(group, params=list(group["params"])) for group in groups]
        self.calls = 0
        self.initial = [value.detach().clone() for value in self.parameters()]
        self.frozen = [(value, value.detach().clone()) for value in frozen]

    def parameters(self):
        return [value for group in self.param_groups for value in group["params"]]

    def step(self):
        self.calls += 1

    def zero_grad(self, set_to_none=True):
        for value in self.parameters():
            value.grad = None

    def unchanged(self):
        import torch

        current = self.parameters()
        return all(
            torch.equal(a, b.detach()) for a, b in zip(self.initial, current, strict=True)
        ) and all(torch.equal(value.detach(), initial) for value, initial in self.frozen)


class _Budget:
    """Parada que cronometra entre dos consultas de `requested` y después detiene.

    El ajuste cronológico consulta la parada en la barrera de cada paso y la inferencia
    antes de cada evento. `rows(consulta)` devuelve las filas recorridas hasta entonces.
    Tras la medida, la parada sigue pedida: el ajuste vuelve a consultarla después de su
    guardado y solo entonces se detiene.
    """

    def __init__(self, warmup, measured, rows, synchronize):
        self.warmup, self.measured = warmup, measured
        self.rows, self.synchronize = rows, synchronize
        self.calls, self.start, self.result = 0, None, None

    @property
    def requested(self):
        if self.result is not None:
            return True
        index, self.calls = self.calls, self.calls + 1
        if index == self.warmup:
            self.synchronize()
            self.start = (time.perf_counter(), self.rows(index))
        if index == self.warmup + self.measured:
            self.synchronize()
            began, first = self.start
            self.result = (self.rows(index) - first, time.perf_counter() - began)
            return True
        return False

    def rate(self, label):
        _require(
            self.result is not None and self.result[0] > 0,
            f"{label} no tiene tramos o eventos suficientes para medir",
        )
        rows, seconds = self.result
        return rows / seconds, rows


def _synchronize():
    import torch

    torch.cuda.synchronize(0)


def _no_save(*_args, **_kwargs):
    """La medición no confirma checkpoints."""


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


def _reference(case, dataset, batch_size):
    """Referencia neuronal del caso con los pesos iniciales de su semilla.

    El Transformer admite el lote medido como en `reference_run`, sin cambiar sus pesos.
    """
    from mars_titan.budget_training import seed_run
    from mars_titan.models.baselines.multimodal import (
        PRESENCE_FUSION,
        MultimodalReference,
        transformer_batch_options,
    )
    from mars_titan.models.quantile_head import QUANTILE_HEAD

    seed_run(case["seed"])
    first = next(dataset.batches(partition="train", batch_size=1, epoch=0, seed=case["seed"]))
    model = MultimodalReference(
        case["kind"],
        {key: value.shape[-1] for key, value in first["inputs"].items()},
        context=dataset.context,
        mask_fusion=PRESENCE_FUSION,
        head=QUANTILE_HEAD,
        **case["architecture"],
        **transformer_batch_options(case["kind"], batch_size),
    )
    return model, {key: value.shape[1:] for key, value in first["inputs"].items()}


def _batch_rates(model, dataset, *, size, seed, warmup, batches, device):
    import torch

    torch.cuda.reset_peak_memory_stats(0)
    options = dict(size=size, seed=seed, warmup=warmup, batches=batches, device=device)
    train, train_rows = _rate(model, dataset, "train", train=True, **options)
    inference, rows = _rate(model, dataset, "validation", train=False, **options)
    return dict(
        train=train,
        inference=inference,
        measured_train_rows=train_rows,
        measured_inference_rows=rows,
        peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(0),
    )


def measure_rates(campaign, view, *, batches=50, warmup=5):
    """Medir filas por segundo de cada brazo y candidato neuronal, sin aprender."""
    from mars_titan.data.embeddings import require_cuda

    from .reference_run import configured_corpus

    _bounded(batches=batches, warmup=warmup)
    device = require_cuda()
    dataset = configured_corpus(Path(view), input_policy=campaign["input_policy"])
    size = campaign["neural"]["batch_size"]
    rates, guard = {}, _forbid_steps()
    try:
        for arm, candidates in campaign["neural"]["candidates"].items():
            rates[arm] = {}
            for name, case in candidates:
                model, _ = _reference(case, dataset, size)
                rates[arm][name] = _batch_rates(
                    model.to(device),
                    dataset,
                    size=size,
                    seed=case["seed"],
                    warmup=warmup,
                    batches=batches,
                    device=device,
                )
    finally:
        guard.remove()
    return rates


def measure_posttraining(stage, view, *, batches=50, warmup=5):
    """Medir cada caso de la matriz sobre cada padre candidato con pesos iniciales.

    El padre es la referencia del brazo base con los pesos iniciales de su caso, porque
    los padres elegidos todavía no existen. Cada brazo de la matriz y la continuación
    completa se construyen como en `posttraining.run.build_model`. Los casos de una
    semilla bastan: las demás solo cambian la inicialización.
    """
    from mars_titan.data.embeddings import require_cuda
    from mars_titan.models.quantile_head import CONTRACT, QUANTILE_HEAD
    from mars_titan.posttraining import adapter_matrix
    from mars_titan.posttraining.parents import FrozenParent
    from mars_titan.posttraining.run import build_model

    from .reference_run import configured_corpus

    _bounded(batches=batches, warmup=warmup)
    device = require_cuda()
    campaign, matrix = stage["campaign"], stage["matrix"]
    dataset = configured_corpus(Path(view), input_policy=campaign["input_policy"])
    budget = matrix["budget"]
    seed = budget["seeds"][0]
    identity = dict(input_policy=campaign["input_policy"], output_head=CONTRACT)
    rates, guard = {}, _forbid_steps()
    try:
        for arm, family in stage["families"].items():
            rates[arm] = {}
            for name, case in campaign["neural"]["candidates"][arm]:
                model, shapes = _reference(case, dataset, budget["batch_size"])
                parent = FrozenParent(model, dict(identity, model=family), shapes, device)
                points = rates[arm][name] = {}
                for item in adapter_matrix.cases(
                    matrix, stage["matrix_sha256"], family, head=QUANTILE_HEAD
                ):
                    if item["case"]["seed"] != seed:
                        continue
                    adapted = build_model(parent, item["case"], None, None).to(device)
                    record = _batch_rates(
                        adapted,
                        dataset,
                        size=budget["batch_size"],
                        seed=seed,
                        warmup=warmup,
                        batches=batches,
                        device=device,
                    )
                    record["trainable_parameters"] = sum(
                        value.numel() for value in adapted.parameters() if value.requires_grad
                    )
                    points[item["id"].split("/", 1)[1]] = record
    finally:
        guard.remove()
    return rates


def _measured_cases(stage):
    """Lo que mide `measure_posttraining`: brazos de las redes, sus casos y el presupuesto.

    Los casos se comparan sin la huella de la matriz que los declara, porque un mismo caso
    cuesta lo mismo en la v2 y en la v3.
    """
    from mars_titan.models.quantile_head import QUANTILE_HEAD
    from mars_titan.posttraining import adapter_matrix

    cases = {}
    for arm, family in stage["families"].items():
        cases[arm] = {}
        for item in adapter_matrix.cases(
            stage["matrix"], stage["matrix_sha256"], family, head=QUANTILE_HEAD
        ):
            # La continuación completa no tiene adaptador ni, por tanto, huella de la matriz.
            case = dict(item["case"])
            if "adapter" in case:
                case["adapter"] = dict(case["adapter"], matrix_sha256=None)
            cases[arm][item["id"]] = json.dumps(case, sort_keys=True)
    return dict(cases=cases, budget=json.dumps(stage["matrix"]["budget"], sort_keys=True))


def _covers(measured, other):
    """True si la medida de una etapa sirve para otra: mismo presupuesto y sus casos dentro."""
    return measured["budget"] == other["budget"] and all(
        arm in measured["cases"]
        and all(measured["cases"][arm].get(key) == case for key, case in cases.items())
        for arm, cases in other["cases"].items()
    )


def covering_stage(stages):
    """Etapa de adaptadores que se mide, porque contiene los casos de todas las demás.

    A declara la matriz v3, con los casos de Titans-MAC y los brazos de la variedad de
    adaptadores, y B la v2. Los casos de las redes de B están todos en A con el mismo
    presupuesto, así que una sola medida de A estima las dos etapas.
    """
    measured = [_measured_cases(stage) for stage in stages]
    for stage, cases in zip(stages, measured, strict=True):
        if all(_covers(cases, other) for other in measured):
            return stage
    raise ValueError("Ninguna etapa de adaptadores contiene los casos medidos de las demás")


def _view_fold(dataset):
    """Ventana walk-forward de la vista, común a todos sus mercados."""
    from .temporal_contract import temporal_contracts

    contracts = temporal_contracts(dataset.manifest, input_policy=HISTORICAL_MASKED)
    folds = [contract["fold"] for contract in contracts.values()]
    _require(
        folds and all(fold == folds[0] for fold in folds),
        "La vista no declara una única ventana walk-forward",
    )
    return folds[0]


def _event_inputs(source):
    """Observaciones de cada evento del índice, en el orden en que se recorren."""
    import numpy as np
    import pyarrow.parquet as pq

    path = source.path.parent / source.metadata["events_path"]
    table = pq.read_table(path, columns=["event_at", "kind"])
    stamps = table["event_at"].to_numpy()[table["kind"].to_numpy() == 0]
    unique, counts = np.unique(stamps, return_counts=True)
    found = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    return [found.get(at, 0) for at, _ in source.metadata["groups"]]


def _train_option(trainer, run, paused, settings, counters=()):
    """Filas por segundo del ajuste entre dos barreras posteriores a un paso.

    `counters` son contadores del recorrido que se registran al empezar y al terminar la
    ventana medida, por ejemplo los flujos de C, para saber qué cálculo recorrió la medida.
    """
    import torch

    marks = []

    def rows(_):
        marks.append({key: run.counters.get(key, 0) for key in counters})
        return run.counters["observations"]

    budget = _Budget(settings["segment_warmup"], settings["segments"], rows, _synchronize)
    cursor = dict(epoch=0, phase="train", event=0, stage="start")
    try:
        trainer._train_pass(run, cursor, budget, _no_save)
    except paused:
        pass
    except torch.cuda.OutOfMemoryError as error:
        return dict(status=OUT_OF_MEMORY, error=str(error).splitlines()[0])
    rate, measured = budget.rate("El tramo de ajuste")
    result = dict(train=rate, measured_train_rows=measured)
    if counters:
        start, end = marks
        result["window_counters"] = dict(start=start, end=end)
    return result


def _inference(trainer, inputs, paused, settings):
    """Filas por segundo de la inferencia congelada sobre los eventos del ajuste."""
    import torch

    prefix = [0]
    for count in inputs:
        prefix.append(prefix[-1] + count)
    budget = _Budget(settings["event_warmup"], settings["events"], prefix.__getitem__, _synchronize)
    torch.cuda.reset_peak_memory_stats(0)
    try:
        trainer.evaluate(trainer.train, stop=budget)
    except paused:
        pass
    rate, rows = budget.rate("La inferencia")
    return dict(
        inference=rate,
        measured_inference_rows=rows,
        inference_peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(0),
    )


def _chronological(build, fresh, paused, options, inputs, settings, counters=()):
    """Medir el ajuste con cada opción y la inferencia de un control cronológico.

    `build(opción)` construye el entrenador con `_NoStepOptimizer` y `fresh(entrenador)` el
    estado vacío de su recorrido de ajuste. La inferencia no depende de la opción de
    memoria y se mide una vez, con la de la receta y sobre los mismos eventos de ajuste.
    Una opción que no cabe en la memoria reservada se registra y la medición sigue.
    """
    import torch

    record = dict(declared_option=option_name(options[0]), options={})
    for index, option in enumerate(options):
        trainer = build(option)
        torch.cuda.reset_peak_memory_stats(0)
        result = _train_option(trainer, fresh(trainer), paused, settings, counters)
        gc.collect()
        result.update(
            option,
            step_calls_without_update=trainer.optimizer.calls,
            peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(0),
        )
        record["options"][option_name(option)] = result
        if index == 0:
            record.update(_inference(trainer, inputs, paused, settings))
        if not trainer.optimizer.unchanged():
            raise RuntimeError("La medición de caudal ha modificado parámetros")
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
    return record


def _chronological_sources(view, document, work):
    """Ventana, fuentes de ajuste y validación y observaciones por evento de una receta Titans.

    Las fases llevan el calentamiento de la receta. Titans-MAC, los lectores sobre su padre
    y los núcleos de CM-v1 comparten así las fases y reutilizan el mismo índice por fase.
    """
    from . import titans_walk_forward as titans
    from .corpus_inputs import CorpusDataset

    dataset = CorpusDataset(Path(view), input_policy=HISTORICAL_MASKED)
    fold = _view_fold(dataset)
    phases = titans.window_phases(fold, titans.walk_forward_options(document)["warmup_months"])
    sources = titans._sources(
        dataset,
        {name: phases[name] for name in ("train", "validation")},
        Path(work) / "titans-indices",
    )
    return fold, sources, _event_inputs(sources["train"])


def _core_builder(document, recipe, sources, work, *, variant, seed, device, local_control=None):
    """Constructor del ajuste cronológico de Titans-MAC con el optimizador de la medición.

    `local_control` es el contrato del control C de un núcleo de CM-v1, como en
    `run_titans_window`. Con `penalty`, cada tramo suma los flujos medidos de C.
    """
    from mars_titan.budget_training import seed_run

    from . import titans_walk_forward as titans
    from .financial_run import ChronologicalTrainer

    specification = sources["train"].specification()

    def build(option):
        seed_run(seed)
        predictor, pairing = titans._predictor(
            document, specification, variant, seed, device, local_control
        )
        return ChronologicalTrainer(
            predictor,
            replace(recipe, **option),
            train=sources["train"],
            validation=sources["validation"],
            output=Path(work) / "titans-unused",
            optimizer_factory=_NoStepOptimizer,
            pairing=pairing,
        )

    return build


def _shared(candidates, inputs):
    """Caso medido de un brazo y los casos que comparten su medida."""
    return dict(
        measured_case=candidates[0][0],
        shared_by_cases=[name for name, _ in candidates],
        max_event_inputs=max(inputs),
    )


def measure_titans(
    campaign, view, work, *, segments=8, segment_warmup=2, events=64, event_warmup=8
):
    """Medir cada control de Titans-MAC con la receta de la campaña y sus opciones de memoria.

    El ajuste recorre `ChronologicalTrainer._train_pass` desde el inicio del tramo de
    ajuste de la vista, con fastpath desactivado como en `titans_fit`. Cada control se
    mide con su primer caso de búsqueda: los casos solo cambian hiperparámetros del
    optimizador, que no intervienen en forward ni backward. `work` guarda los índices.
    """
    from mars_titan.data.embeddings import require_cuda

    from . import titans_walk_forward as titans
    from .financial_run import Paused, _Pass, load_recipe

    settings = dict(
        segments=segments, segment_warmup=segment_warmup, events=events, event_warmup=event_warmup
    )
    _bounded(**settings)
    section = campaign.get(TITANS)
    _require(section, "La campaña no declara Titans-MAC")
    device = str(require_cuda())
    _, document = load_recipe(section["path"])
    _, sources, inputs = _chronological_sources(view, document, work)
    rates, guard = {}, _forbid_steps()
    try:
        with titans.unfused_attention():
            for arm, candidates in section["candidates"].items():
                _, case = candidates[0]
                recipe = titans.case_recipe(document, case["search_case"])
                build = _core_builder(
                    document,
                    recipe,
                    sources,
                    work,
                    variant=case["variant"],
                    seed=case["seed"],
                    device=device,
                )
                record = _chronological(
                    build,
                    lambda _: _Pass(),
                    Paused,
                    _options(recipe, TITANS_OPTIONS),
                    inputs,
                    settings,
                )
                rates[arm] = dict(record, variant=case["variant"], **_shared(candidates, inputs))
    finally:
        guard.remove()
    return rates


def _readout_record(
    family, parent, document, case, window, work, device, settings, counters=BANK_COUNTERS
):
    """Medir el lector de una familia sobre un padre `mac_online` con pesos iniciales.

    Los padres elegidos todavía no existen y su coste no depende de sus pesos. El padre se
    construye como en `_frozen_parent`: congelado y, si su núcleo se ajustó con C en
    `penalty`, como su gemelo `disabled`, porque C no interviene al predecir. `parent` da la
    receta del padre, su huella y su caso de búsqueda, que solo entran en la identidad de la
    variante. `window` es la ventana, sus fuentes y sus observaciones por evento. El recorrido
    es `ReadoutTrainer._train_pass` hasta el paso, con el banco de la escritura declarada y
    su retención, y se exige que no cambien ni el lector ni el padre. M3 estima antes sus
    escalas con la regla de la campaña, sobre el tramo de entrenamiento de la ventana medida,
    y el registro guarda su huella y la duración de ese recorrido, que no entra en el caudal.
    """
    from functools import partial

    from mars_titan.budget_training import seed_run
    from mars_titan.memory.episodic_codec import FrozenEpisodeCodec

    from . import mars_titan_walk_forward as readouts
    from . import titans_walk_forward as titans
    from .financial_run import Paused
    from .mars_titan_run import ReadoutTrainer, case_recipe

    fold, sources, inputs = window
    recipe = case_recipe(document, case["search_case"])
    specification = sources["train"].specification()
    control = titans.control_config(family.control)
    if control is not None and control.mode != "disabled":
        control = replace(control, mode="disabled", weight=0.0)
    request = dict(
        recipe_sha256=parent["sha256"],
        search_case=parent["search_case"],
        window=fold["id"],
        local_control=family.control,
    )
    scalers = {}

    def retention(variant):
        if variant.admission != "m3":
            return family.retention(recipe, variant)
        if not scalers:
            start = time.perf_counter()
            value = readouts.window_scalers(sources["train"], recipe)
            scalers.update(value=value, seconds=time.perf_counter() - start)
        return family.retention(recipe, variant, scalers=scalers["value"])

    def build(option):
        seed_run(case["seed"])
        predictor, _ = titans._predictor(
            parent["document"],
            specification,
            "mac_online",
            case["seed"],
            device,
            None if control is None else asdict(control),
        )
        predictor = predictor.eval().requires_grad_(False)
        variant = family.variant(predictor, dict(request=request))
        codec = FrozenEpisodeCodec(specification)
        readout = readouts._readout(variant, codec, predictor, recipe, case["seed"])
        return ReadoutTrainer(
            predictor,
            readout,
            replace(recipe, **option),
            admission=variant.admission,
            retention=retention(variant),
            native=readouts._native(variant.admission),
            codec=codec,
            train=sources["train"],
            validation=sources["validation"],
            output=Path(work) / "readout-unused",
            world=family.world,
            fold=fold["id"],
            optimizer_factory=partial(_NoStepOptimizer, frozen=list(predictor.parameters())),
        )

    record = _chronological(
        build,
        lambda trainer: trainer._new_pass("train"),
        Paused,
        RECIPE_ONLY,
        inputs,
        settings,
        counters,
    )
    if scalers:
        value = scalers["value"]
        record["write_scalers"] = dict(
            sha256=value.fingerprint(),
            source_sha256=value.source_sha256,
            decisions=value.decisions,
            labels=value.labels,
            seconds=scalers["seconds"],
        )
    return dict(record, bank_capacity=recipe.bank_capacity)


def _correction_record(parent, case, window, device, settings):
    """Medir la corrección B6 sobre un padre `mac_online` con pesos iniciales.

    B6 no tiene ajuste. Se mide la inferencia cronológica del padre con la lectura y la
    escritura de A sobre los eventos del tramo de ajuste, que no tienen calentamiento, así
    que todos los eventos medidos corrigen. El caso solo cambia η y comparte la medida.
    """
    import torch

    from mars_titan.budget_training import seed_run
    from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
    from mars_titan.memory.mars_titan_variant import check_components, load_declaration

    from . import mars_titan_correction as correction
    from . import titans_walk_forward as titans
    from .financial_run import Paused

    fold, sources, inputs = window
    document = correction.load_correction_recipe(case["recipe"])
    values = correction.case_values(document, case["search_case"])
    _, mature = check_components(
        load_declaration(), correction.arm_components(case["components"], values)
    )
    specification = sources["train"].specification()
    seed_run(case["seed"])
    predictor, _ = titans._predictor(
        parent["document"], specification, "mac_online", case["seed"], device
    )
    predictor = predictor.eval().requires_grad_(False)
    initial = [value.detach().clone() for value in predictor.parameters()]
    chronological = titans.case_recipe(parent["document"], parent["search_case"])
    inference = correction.CorrectionInference(
        predictor, chronological, mature, FrozenEpisodeCodec(specification)
    )
    prefix = [0]
    for count in inputs:
        prefix.append(prefix[-1] + count)
    budget = _Budget(settings["event_warmup"], settings["events"], prefix.__getitem__, _synchronize)
    torch.cuda.reset_peak_memory_stats(0)
    events = sources["train"].batched_events(block_rows=chronological.block_rows)
    try:
        inference._pass(sources["train"], events, stop=budget)
    except Paused:
        pass
    rate, rows = budget.rate("La corrección B6")
    peak = torch.cuda.max_memory_allocated(0)
    if not all(
        torch.equal(a, b.detach()) for a, b in zip(initial, predictor.parameters(), strict=True)
    ):
        raise RuntimeError("La medición de caudal ha modificado parámetros")
    return dict(
        declared_option="recipe",
        options={"recipe": dict(train=None, peak_vram_allocated_bytes=peak)},
        inference=rate,
        measured_inference_rows=rows,
        inference_peak_vram_allocated_bytes=peak,
        associative_writes=inference.memory.writes,
        window=fold["id"],
    )


def measure_mars_titan(
    campaign, view, work, *, segments=8, segment_warmup=2, events=64, event_warmup=8
):
    """Medir el lector de cada brazo de MARS-TITAN sobre su padre `titans_mac_online`.

    Cada brazo se mide por separado porque la escritura (M0, M1, M2 o M3) y K cambian el
    cálculo. Los casos de búsqueda solo cambian la tasa de aprendizaje y comparten la
    medida. Las fases son las del padre, con su calentamiento. `work` guarda los índices.
    M3 registra además los cambios de su índice selectivo en la ventana medida. Un brazo de
    corrección B6 no tiene lector y solo mide su inferencia (`_correction_record`).
    """
    from mars_titan.data.embeddings import require_cuda

    from . import mars_titan_walk_forward as readouts
    from . import titans_walk_forward as titans
    from .financial_run import load_recipe
    from .mars_titan_run import load_recipe as load_readout
    from .mars_titan_run import m3_counters

    settings = dict(
        segments=segments, segment_warmup=segment_warmup, events=events, event_warmup=event_warmup
    )
    _bounded(**settings)
    section = campaign.get(MARS)
    _require(section, "La campaña no declara MARS-TITAN")
    device = str(require_cuda())
    titans_section = campaign[TITANS]
    _, document = load_recipe(titans_section["path"])
    parent = dict(
        document=document,
        sha256=titans_section["sha256"],
        search_case=titans_section["candidates"][section["parent_arm"]][0][0],
    )
    readout = load_readout(section["path"])
    window = _chronological_sources(view, document, work)
    rates, guard = {}, _forbid_steps()
    try:
        with titans.unfused_attention():
            for arm, candidates in section["candidates"].items():
                _, case = candidates[0]
                if "associative_memory" in case["components"]:
                    record = _correction_record(parent, case, window, device, settings)
                    rates[arm] = dict(
                        record,
                        components=case["components"],
                        parent_arm=section["parent_arm"],
                        **_shared(candidates, window[2]),
                    )
                    continue
                family = readouts._mars_family(case["components"])
                counters = BANK_COUNTERS
                if case["components"].get("episodic_bank") == "m3":
                    counters += tuple(m3_counters())
                record = _readout_record(
                    family, parent, readout, case, window, work, device, settings, counters
                )
                rates[arm] = dict(
                    record,
                    components=case["components"],
                    parent_arm=section["parent_arm"],
                    **_shared(candidates, window[2]),
                )
    finally:
        guard.remove()
    return rates


def _check_cm_v1(campaign, segments):
    """Exigir antes de medir que la ventana medida alcance los flujos que mide C.

    C mide un flujo cuando su número de observaciones es múltiplo de `frequency`, así que
    la ventana necesita al menos `frequency` instantes: `segments` tramos de `truncation`.
    """
    section = campaign.get(CM)
    if not section:
        return
    declaration, _ = read_manifest(Path(section["path"]), 64 * 1024)
    core, _ = read_manifest(Path(section["recipes"]["core_recipe"]), 64 * 1024)
    frequency = declaration["control"]["frequency"]
    instants = segments * core["recipe"]["truncation"]
    _require(
        instants >= frequency,
        f"La medición de C recorre {instants} instantes y necesita al menos frequency="
        f"{frequency} para incluir flujos medidos. Aumenta los tramos medidos",
    )


def measure_cm_v1(campaign, view, work, *, segments=8, segment_warmup=2, events=64, event_warmup=8):
    """Medir los dos núcleos de CM-v1 y el lector de cada brazo, sin pasos ni cambios de pesos.

    Los núcleos recorren `ChronologicalTrainer._train_pass` con la receta del núcleo:
    `cm_v1_core_b` con C en `disabled` (SDPA Math) y `cm_v1_core_c` con la penalización,
    que en cada evento elige los flujos medidos y calcula su término de RᵀJR con su JVP. Se
    registran los grupos y flujos de C de la ventana medida. Los núcleos comparten la receta
    de Titans-MAC y comparan sus mismas opciones de `accumulation_rows`, que C admite. Cada
    brazo mide su lector M1 con K = 1 sobre el gemelo `disabled` de su núcleo, con la
    retención reservoir en B y B+C y con centros fijos en B+M y B+C+M.
    """
    from mars_titan.data.embeddings import require_cuda

    from . import cm_v1_factorial as cm
    from . import titans_walk_forward as titans
    from .financial_run import Paused, _Pass, load_recipe
    from .mars_titan_run import load_recipe as load_readout

    settings = dict(
        segments=segments, segment_warmup=segment_warmup, events=events, event_warmup=event_warmup
    )
    _bounded(**settings)
    section = campaign.get(CM)
    _require(section, "La campaña no declara CM-v1")
    _check_cm_v1(campaign, segments)
    declaration = cm.load_declaration(section["path"])
    _require(
        declaration["sha256"] == section["sha256"],
        "La declaración de CM-v1 cambió después de planificar la campaña",
    )
    device = str(require_cuda())
    _, document = load_recipe(declaration["recipes"]["core_recipe"])
    readout = load_readout(declaration["recipes"]["readout_recipe"])
    window = _chronological_sources(view, document, work)
    _, sources, inputs = window
    candidates = section["candidates"]
    rates, guard = {}, _forbid_steps()
    try:
        with titans.unfused_attention():
            for core in CM_CORES:
                _, case = candidates[core][0]
                contract = cm.control_contract(declaration, cm.CORES[core])
                recipe = titans.case_recipe(document, case["search_case"])
                build = _core_builder(
                    document,
                    recipe,
                    sources,
                    work,
                    variant="mac_online",
                    seed=case["seed"],
                    device=device,
                    local_control=contract,
                )
                record = _chronological(
                    build,
                    lambda _: _Pass(),
                    Paused,
                    _options(recipe, TITANS_OPTIONS),
                    inputs,
                    settings,
                    CONTROL_COUNTERS if contract["mode"] == "penalty" else (),
                )
                rates[core] = dict(
                    record,
                    control_mode=contract["mode"],
                    accumulation_rows=recipe.accumulation_rows,
                    **_shared(candidates[core], inputs),
                )
            for arm, core in CM_ARMS.items():
                _, case = candidates[arm][0]
                parent = dict(
                    document=document,
                    sha256=candidates[core][0][1]["recipe_sha256"],
                    search_case=candidates[core][0][0],
                )
                record = _readout_record(
                    cm.readout_family(declaration, arm),
                    parent,
                    readout,
                    case,
                    window,
                    work,
                    device,
                    settings,
                )
                rates[arm] = dict(
                    record,
                    parent_arm=core,
                    control=cm.ARMS[arm][0],
                    consolidation=cm.ARMS[arm][1],
                    **_shared(candidates[arm], inputs),
                )
    finally:
        guard.remove()
    return rates


def measure_candidate(
    campaign, view, work, *, segments=8, segment_warmup=2, events=64, event_warmup=8
):
    """Medir la GRU candidata con su receta, su variante y sus opciones de memoria.

    El ajuste recorre `CandidateChronologicalTrainer._train_pass` con el módulo nativo
    y un optimizador que no modifica pesos. `work` guarda los índices de observaciones.
    """
    from mars_titan.budget_training import seed_run
    from mars_titan.data.embeddings import require_cuda
    from mars_titan.models.candidate.input_adapter import CandidateInputAdapter

    from .candidate_run import CandidateChronologicalTrainer, _Pass, _Pause
    from .candidate_walk_forward import WORLD, _dtype, campaign_case, window_sources
    from .corpus_inputs import CorpusDataset

    settings = dict(
        segments=segments, segment_warmup=segment_warmup, events=events, event_warmup=event_warmup
    )
    _bounded(**settings)
    section = campaign.get(EPISODIC)
    _require(section, "La campaña no declara la GRU candidata")
    device = str(require_cuda())
    work = Path(work)
    dataset = CorpusDataset(Path(view), input_policy=HISTORICAL_MASKED)
    fold = _view_fold(dataset)
    rates, guard = {}, _forbid_steps()
    try:
        for arm, candidates in section["candidates"].items():
            # Los casos solo cambian hiperparámetros del optimizador, como en Titans-MAC.
            _, case = candidates[0]
            recipe, model, warmup_months = campaign_case(case)
            sources = window_sources(
                dataset, work / "candidate-indices", ("train", "validation"), warmup_months
            )
            inputs = _event_inputs(sources["train"])
            specification = sources["train"].specification()

            def build(
                option,
                case=case,
                recipe=recipe,
                model=model,
                sources=sources,
                specification=specification,
            ):
                seed_run(case["seed"])
                adapter = CandidateInputAdapter(
                    specification,
                    dtype=_dtype(model["dtype"]),
                    device=device,
                    parameter_seed=case["seed"],
                    feature_seed=model["feature_seed"],
                    key_seed=model["key_seed"],
                )
                return CandidateChronologicalTrainer(
                    adapter,
                    replace(recipe, **option),
                    train=sources["train"],
                    validation=sources["validation"],
                    output=work / "candidate-unused",
                    world=WORLD,
                    fold=fold["id"],
                    optimizer_factory=_NoStepOptimizer,
                )

            record = _chronological(
                build,
                lambda trainer: _Pass(bank=trainer._new_bank("train")),
                _Pause,
                _options(recipe, CANDIDATE_OPTIONS),
                inputs,
                settings,
            )
            rates[arm] = dict(record, variant=case["variant"], **_shared(candidates, inputs))
    finally:
        guard.remove()
    return rates


def with_candidate(campaign, recipe, variant=None):
    """Añadir la GRU candidata que la campaña todavía no declara, solo para medir y estimar.

    La sección se valida con las reglas de una sección declarada. Por defecto usa la
    variante principal de la receta y la semilla de búsqueda neuronal. No cambia la
    configuración de la campaña ni sus límites.
    """
    _require(not campaign.get(EPISODIC), "La campaña ya declara la GRU candidata")
    recipe = Path(recipe).resolve()
    document, _ = read_manifest(recipe, 64 * 1024)
    arms = campaign["comparison_config"]["arms"]
    section = dict(
        recipe=str(recipe),
        arms={
            name: variant or document.get("principal")
            for name, arm in arms.items()
            if arm["family"] == EPISODIC
        },
        search_seed=campaign["neural"]["search_seed"],
    )
    _require(section["arms"], "La comparación no declara la GRU candidata")
    return extend_campaign(campaign, {EPISODIC: section})


def _candidates(campaign, family):
    if family == NEURAL:
        return campaign["neural"]["candidates"]
    return (campaign.get(family) or {}).get("candidates")


def _window_counts(campaign, views):
    from .masked_campaign import scope_views

    return {
        scope: {
            window: value["counts"]
            for window, value in scope_views(views[scope], scope, campaign)["windows"].items()
        }
        for scope in campaign["scopes"]
    }


def _comparison(estimates):
    """Horas de GPU de A y B y su cociente, la entrada de la decisión de #363."""
    totals = {estimate["variant"]: estimate["total_gpu_hours"] for estimate in estimates}
    if set(totals) != {"A", "B"}:
        return None
    ratio = {
        key: None
        if not totals["A"][key] or totals["B"][key] is None
        else totals["B"][key] / totals["A"][key]
        for key in ("declared_options", "fastest_options")
    }
    return dict(A=totals["A"], B=totals["B"], b_over_a=ratio)


def measure_campaigns(
    paths,
    views,
    first_view,
    *,
    stages=(),
    rl_stages=(),
    ablation_stages=(),
    candidate=None,
    extensions=None,
    work=None,
    output=None,
    **settings,
):
    """Medir una vez en `cuda:0` y estimar las horas de cada variante declarada.

    Mide las referencias neuronales y cada familia cronológica que la campaña declara
    (Titans-MAC, GRU candidata, MARS-TITAN y CM-v1), la matriz de adaptadores si se pasan
    sus etapas y el entorno y la red de las políticas si se pasan `rl_stages`. Las etapas de
    `ablation_stages` solo se estiman con los caudales ya medidos. `candidate`
    añade solo la GRU candidata y `extensions`, la declaración preparada de
    `campaign_extensions` con sus tres familias y los límites de sus etapas de políticas.
    `work` guarda los índices de la primera vista y `output`, el informe.
    """
    from mars_titan.data.storage import atomic_json
    from mars_titan.posttraining.campaign_stage import load_stage
    from mars_titan.simulation import policy_plan, policy_throughput

    from . import modality_ablation_stage
    from .experiment_resources import GpuLease

    settings = SETTINGS | settings
    _bounded(**settings)
    _require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") in {":4096:8", ":16:8"},
        "Configura CUBLAS_WORKSPACE_CONFIG antes de iniciar PyTorch",
    )
    campaigns = [load_campaign(path) for path in paths]
    prepared = None
    if extensions is not None:
        from .campaign_extensions import extended_campaign, extended_policies, load_extensions

        _require(candidate is None, "La declaración preparada ya incluye la GRU candidata")
        prepared = load_extensions(extensions)
        campaigns = [extended_campaign(prepared, c) for c in campaigns]
    if candidate is not None:
        campaigns = [c if c.get(EPISODIC) else with_candidate(c, **candidate) for c in campaigns]
    reference = campaigns[0]
    _require(
        all(
            _candidates(c, family) == _candidates(reference, family)
            for c in campaigns
            for family in (NEURAL, *CHRONOLOGICAL)
        ),
        "Las variantes comparadas deben declarar los mismos candidatos",
    )
    _check_cm_v1(reference, settings["segments"])
    loaded = [load_stage(path) for path in stages]
    by_campaign = {stage["campaign"]["path"]: stage for stage in loaded}
    _require(
        len(by_campaign) == len(loaded) and set(by_campaign) <= {c["path"] for c in campaigns},
        "Cada etapa de adaptadores parte de una campaña medida distinta",
    )
    measured_stage = covering_stage(loaded) if loaded else None
    policies = [policy_plan.load_stage(path) for path in rl_stages]
    by_policies = {stage["campaign"]["path"]: stage for stage in policies}
    _require(
        len(by_policies) == len(policies)
        and set(by_policies) <= {c["path"] for c in campaigns}
        and len({stage["policies"]["sha256"] for stage in policies}) <= 1,
        "Cada etapa de políticas parte de una campaña medida, con las mismas políticas",
    )
    ablations = [modality_ablation_stage.load_stage(path) for path in ablation_stages]
    by_ablation = {stage["campaign"]["path"]: stage for stage in ablations}
    _require(
        len(by_ablation) == len(ablations) and set(by_ablation) <= {c["path"] for c in campaigns},
        "Cada etapa de ablación parte de una campaña medida distinta",
    )
    if prepared is not None:
        # Con la declaración preparada, la etapa resuelve sus predictores en la campaña ampliada.
        measured = {c["path"]: c for c in campaigns}
        by_policies = {
            path: extended_policies(prepared, stage, measured[path])
            for path, stage in by_policies.items()
        }
        policies = list(by_policies.values())
    batched = {key: settings[key] for key in ("batches", "warmup")}
    stepped = dict(steps=settings["policy_steps"], warmup=settings["policy_warmup"])
    chronological = {
        key: value
        for key, value in settings.items()
        if key not in batched and not key.startswith("policy_")
    }
    _require(
        work is not None or not any(reference.get(family) for family in CHRONOLOGICAL),
        "Las familias cronológicas necesitan un directorio de trabajo para sus índices",
    )
    declared = {json.dumps(c.get("numerics"), sort_keys=True) for c in campaigns}
    _require(len(declared) == 1, "Las variantes medidas deben declarar la misma precisión")
    numerics = reference.get("numerics")
    if numerics:
        # Se mide con la precisión con la que se entrenará.
        campaign_numerics.apply(numerics)
    started = time.perf_counter()
    with GpuLease() as lease:
        rates = {NEURAL: measure_rates(reference, first_view, **batched)}
        if reference.get(TITANS):
            rates[TITANS] = measure_titans(reference, first_view, work, **chronological)
        if reference.get(EPISODIC):
            rates[EPISODIC] = measure_candidate(reference, first_view, work, **chronological)
        if reference.get(MARS):
            rates[MARS] = measure_mars_titan(reference, first_view, work, **chronological)
        if reference.get(CM):
            rates[CM] = measure_cm_v1(reference, first_view, work, **chronological)
        if loaded:
            rates[POSTTRAINING] = measure_posttraining(measured_stage, first_view, **batched)
        if policies:
            rates[POLICY_STAGE] = policy_throughput.measure_policies(policies[0], **stepped)
        resources = lease.record
    estimates = [
        estimate_hours(
            campaign,
            _window_counts(campaign, views),
            rates,
            stage=by_campaign.get(campaign["path"]),
            policy_stage=by_policies.get(campaign["path"]),
            ablation_stage=by_ablation.get(campaign["path"]),
        )
        for campaign in campaigns
    ]
    report = dict(
        schema_version=2,
        kind="masked_campaign_throughput",
        measured_at_utc=datetime.now(UTC).isoformat(),
        view=str(Path(first_view).resolve()),
        settings=settings,
        extensions=None
        if prepared is None
        else dict(path=prepared["path"], sha256=prepared["sha256"], status=prepared["status"]),
        rates=rates,
        numerics=None if numerics is None else campaign_numerics.current(),
        estimates=estimates,
        comparison=_comparison(estimates),
        resources=resources,
        seconds=time.perf_counter() - started,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        optimizer_steps=0,
        scientific_training_started=False,
        final_test_opened=False,
    )
    if output is not None:
        atomic_json(Path(output), report)
    return report


def main(argv=None):
    from .masked_campaign import _views_argument

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", type=Path, action="append", required=True)
    parser.add_argument("--views", action="append", required=True)
    parser.add_argument("--first-view", type=Path, required=True)
    parser.add_argument("--stage", type=Path, action="append", default=[])
    parser.add_argument("--rl-stage", type=Path, action="append", default=[])
    parser.add_argument("--ablation-stage", type=Path, action="append", default=[])
    parser.add_argument("--candidate-recipe", type=Path)
    parser.add_argument("--candidate-variant")
    parser.add_argument("--extensions", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--output", type=Path)
    for name, value in SETTINGS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=value)
    args = parser.parse_args(argv)
    candidate = None
    if args.candidate_recipe is not None:
        candidate = dict(recipe=args.candidate_recipe, variant=args.candidate_variant)
    elif args.candidate_variant is not None:
        parser.error("--candidate-variant necesita --candidate-recipe")
    if candidate is not None and args.extensions is not None:
        parser.error("--extensions ya incluye la GRU candidata. Usa una de las dos")
    report = measure_campaigns(
        args.campaign,
        _views_argument(args.views),
        args.first_view,
        stages=args.stage,
        rl_stages=args.rl_stage,
        ablation_stages=args.ablation_stage,
        candidate=candidate,
        extensions=args.extensions,
        work=args.work,
        output=args.output,
        **{name: getattr(args, name) for name in SETTINGS},
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
