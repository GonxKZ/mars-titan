"""Coste medido de las trazas de aprendizaje con las formas reales de la campaña (#448).

Cada familia se recorre sin pasos de optimizador en tres condiciones: sin trazas, con
registro en cada paso (`every=1`, la cota superior del coste) y con la cadencia declarada.
Las referencias neuronales y los adaptadores de la matriz de postentrenamiento recorren
lotes reales de ajuste con forward, pinball, backward, la norma del gradiente y el paso
nulo, en el orden de `reference_run` y `run_case`, cronometrados por bloques con una
sincronización al principio y otra al final. Titans-MAC, la GRU candidata, los lectores de
MARS-TITAN, los núcleos y lectores de CM-v1 y la corrección B6 recorren su ajuste o su
inferencia cronológica real con `training.campaign_throughput` y las trazas enganchadas.
Ridge y XGBoost no se pueden recorrer sin ajustar: se mide su evento de traza con la anchura
real de la vista. La etapa RL nativa no se mide aquí, porque sin pasos de optimizador sus
binarios solo registran validaciones (sus pruebas lo comprueban).

Las condiciones se alternan en cada repetición para repartir la deriva de una máquina
compartida, y el informe guarda la carga concurrente de la GPU. El coste relativo es el
cociente entre el tiempo por fila con trazas y sin ellas en la misma repetición, resumido
con su mediana. Un gancho global rechaza cualquier paso de optimizador y cada recorrido
exige al final que ningún peso haya cambiado.
"""

import argparse
import gc
import itertools
import json
import math
import os
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from .hooks import open_recorder, open_traces, trace_identity, traced_update
from .recorder import TraceConfig

SCHEMA_VERSION = 1
KIND = "learning_traces_overhead"
CONDITIONS = ("off", "every_step", "declared")
FAMILIES = ("neural", "posttraining", "titans_mac", "episodic_gru", "mars_titan", "cm_v1")
MIB = 1024**2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _config(condition, declared):
    """Configuración de las trazas en cada condición, o None sin trazas."""
    if condition == "off":
        return None
    if condition == "every_step":
        return dict(every=1, max_bytes=declared["max_bytes"])
    return dict(declared)


def _orders(repeats):
    """Orden de las condiciones en cada repetición: se rota para repartir la deriva."""
    return [CONDITIONS[index:] + CONDITIONS[:index] for index in range(repeats)]


def _gpu_load():
    """Procesos de cómputo en la GPU y memoria usada, tal como los ve nvidia-smi."""
    query = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(query, capture_output=True, text=True, check=False, timeout=30)
    rows = [line.split(",") for line in result.stdout.strip().splitlines() if line.strip()]
    others = [dict(pid=int(pid), used_mib=int(used)) for pid, used in rows]
    return [row for row in others if row["pid"] != os.getpid()]


def _summary(times, baseline):
    """Mediana y extremos del coste relativo por repetición frente a la condición sin trazas."""
    ratios = [value / base - 1 for value, base in zip(times, baseline, strict=True)]
    return dict(
        median=statistics.median(ratios),
        minimum=min(ratios),
        maximum=max(ratios),
        repeats=len(ratios),
    )


# Referencias neuronales y adaptadores


def _neural_block(model, batches, optimizer, traces, device, first_step):
    """Segundos de un bloque de lotes con el paso de `reference_run`, sin actualizar pesos."""
    import torch

    from mars_titan.models.quantile_head import pinball_loss
    from mars_titan.training.reference_run import _forward

    torch.cuda.synchronize(0)
    started = time.perf_counter()
    for offset, batch in enumerate(batches):
        optimizer.zero_grad(set_to_none=True)
        emitted = _forward(model, batch, device)
        target = torch.from_numpy(batch["target"]).to(device, dtype=torch.float32)
        loss = pinball_loss(emitted, target)
        loss.backward()
        with traced_update(traces, first_step + offset, optimizer, loss) as probe:
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                math.inf,
                error_if_nonfinite=True,
            )
            probe.clip(norm)
            optimizer.step()
    torch.cuda.synchronize(0)
    return time.perf_counter() - started


def _batched_family(name, model, batches, folder, declared, settings, observers=()):
    """Coste de las trazas en un modelo por lotes con las tres condiciones alternadas."""
    import torch

    from mars_titan.training.campaign_throughput import _NoStepOptimizer

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = _NoStepOptimizer([dict(params=trainable, lr=1e-3, role="predictor")])
    warmup, steps = settings["warmup"], settings["steps"]
    rows = sum(len(batch["target"]) for batch in batches[warmup : warmup + steps])
    seconds = {condition: [] for condition in CONDITIONS}
    peaks, volume, validation = {}, {}, {}
    for repeat, order in enumerate(_orders(settings["repeats"])):
        for condition in order:
            config = _config(condition, declared)
            traces = None
            if config is not None:
                traces = open_traces(
                    folder / f"{condition}-{repeat}",
                    config,
                    trace_identity(name, dict(measure=name, condition=condition)),
                    dict(model=model),
                    observers=observers,
                )
            torch.cuda.reset_peak_memory_stats(0)
            _neural_block(model, batches[:warmup], optimizer, traces, model_device(model), 1)
            measured = _neural_block(
                model,
                batches[warmup : warmup + steps],
                optimizer,
                traces,
                model_device(model),
                warmup + 1,
            )
            seconds[condition].append(measured / rows)
            peaks.setdefault(condition, []).append(torch.cuda.max_memory_allocated(0))
            if traces is not None:
                torch.cuda.synchronize(0)
                started = time.perf_counter()
                traces.validation(dict(global_step=warmup + steps, score=0.0))
                cursor = traces.checkpoint()
                torch.cuda.synchronize(0)
                validation.setdefault(condition, []).append(time.perf_counter() - started)
                volume[condition] = dict(
                    bytes=cursor["bytes"],
                    parts=cursor["sequence"],
                    traced_steps=(warmup + steps) // config["every"],
                )
    _require(optimizer.unchanged(), f"{name}: la medida ha modificado parámetros")
    return dict(
        trainable_parameters=sum(p.numel() for p in trainable),
        rows_per_block=rows,
        seconds_per_row={c: statistics.median(v) for c, v in seconds.items()},
        overhead={c: _summary(seconds[c], seconds["off"]) for c in CONDITIONS[1:]},
        peak_vram_allocated_bytes={c: max(v) for c, v in peaks.items()},
        validation_seconds={c: statistics.median(v) for c, v in validation.items()},
        volume=volume,
    )


def model_device(model):
    return next(model.parameters()).device


def _batches(dataset, size, seed, count):
    """Primeros lotes reales de ajuste de la vista, cargados una vez en memoria."""
    source = dataset.batches(partition="train", batch_size=size, epoch=0, seed=seed)
    batches = list(itertools.islice(source, count))
    _require(len(batches) == count, "La vista no tiene lotes de ajuste suficientes")
    return batches


def measure_neural(campaign, view, work, declared, settings, device):
    """Cada referencia neuronal de la campaña con su primer candidato y su lote."""
    import torch

    from mars_titan.training.campaign_throughput import _reference
    from mars_titan.training.reference_run import configured_corpus

    dataset = configured_corpus(Path(view), input_policy=campaign["input_policy"])
    size = campaign["neural"]["batch_size"]
    results = {}
    for arm, candidates in campaign["neural"]["candidates"].items():
        name, case = candidates[0]
        model, _ = _reference(case, dataset, size)
        model = model.to(device)
        batches = _batches(dataset, size, case["seed"], settings["warmup"] + settings["steps"])
        results[arm] = dict(
            case=name,
            **_batched_family(
                f"neural_{arm}", model, batches, Path(work) / "neural" / arm, declared, settings
            ),
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return results


def measure_posttraining(stage, view, work, declared, settings, device, arms=None):
    """Cada caso de la matriz de adaptadores sobre el padre con pesos iniciales.

    Igual que `campaign_throughput.measure_posttraining`, con los casos de la primera
    semilla. Las validaciones añaden el observador de adaptadores, que calcula los valores
    singulares de cada ΔW.
    """
    import torch

    from mars_titan.models.quantile_head import CONTRACT, QUANTILE_HEAD
    from mars_titan.posttraining import adapter_matrix
    from mars_titan.posttraining.adapter_traces import AdapterObserver
    from mars_titan.posttraining.parents import FrozenParent
    from mars_titan.posttraining.run import build_model
    from mars_titan.training.campaign_throughput import _reference
    from mars_titan.training.reference_run import configured_corpus

    campaign, matrix = stage["campaign"], stage["matrix"]
    dataset = configured_corpus(Path(view), input_policy=campaign["input_policy"])
    budget = matrix["budget"]
    seed = budget["seeds"][0]
    identity = dict(input_policy=campaign["input_policy"], output_head=CONTRACT)
    batches = _batches(dataset, budget["batch_size"], seed, settings["warmup"] + settings["steps"])
    results = {}
    for arm, family in stage["families"].items():
        if arms is not None and arm not in arms:
            continue
        name, case = campaign["neural"]["candidates"][arm][0]
        model, shapes = _reference(case, dataset, budget["batch_size"])
        parent = FrozenParent(model, dict(identity, model=family), shapes, device)
        results[arm] = {}
        for item in adapter_matrix.cases(
            matrix, stage["matrix_sha256"], family, head=QUANTILE_HEAD
        ):
            if item["case"]["seed"] != seed:
                continue
            adapted = build_model(parent, item["case"], None, None).to(device)
            label = item["id"].split("/", 1)[1]
            results[arm][label] = _batched_family(
                f"posttraining_{arm}",
                adapted,
                batches,
                Path(work) / "posttraining" / arm / label.replace("/", "_"),
                declared,
                settings,
                observers=[AdapterObserver()],
            )
            del adapted
            gc.collect()
            torch.cuda.empty_cache()
    return results


# Familias cronológicas


def _attacher(root, family, config):
    """Enganche de las trazas a cada entrenador (o a la corrección B6) que mide la familia."""
    from mars_titan.training.mars_titan_correction import CorrectionInference

    from .trainers import attach

    counter = itertools.count()

    def attach_traces(target):
        folder = Path(root) / f"{next(counter):03d}"
        if isinstance(target, CorrectionInference):
            target.trace_recorder = open_recorder(
                folder,
                config,
                trace_identity("b6", dict(measure=str(folder))),
                step_unit="write_event",
            )
        else:
            attach(target, folder, config, family)

    return attach_traces


def _trimmed(campaign, family, keep):
    """La campaña con solo los modelos `keep` de una familia, para acotar la medida."""
    section = dict(campaign[family])
    section["candidates"] = {arm: section["candidates"][arm] for arm in keep}
    return dict(campaign, **{family: section})


def _rates(record):
    """Filas por segundo del ajuste o de la inferencia de cada modelo y opción."""
    result = {}
    for arm, entry in record.items():
        for option, values in entry["options"].items():
            rate = values.get("train") or entry.get("inference")
            result[f"{arm}/{option}"] = dict(
                rate=rate,
                peak_vram_allocated_bytes=values.get("peak_vram_allocated_bytes"),
                learning_traces=values.get("learning_traces") or entry.get("learning_traces"),
            )
    return result


def measure_chronological(name, measure, campaign, view, work, declared, settings):
    """Medir una familia cronológica con `campaign_throughput` en las tres condiciones.

    Cada medición de `campaign_throughput` instala su propio gancho contra los pasos de
    optimizador y comprueba que los pesos no cambian.
    """
    chronological = dict(
        segments=settings["segments"],
        segment_warmup=settings["segment_warmup"],
        events=settings["events"],
        event_warmup=settings["event_warmup"],
    )
    runs = {condition: [] for condition in CONDITIONS}
    for repeat, order in enumerate(_orders(settings["chronological_repeats"])):
        for condition in order:
            config = _config(condition, declared)
            traces = None
            if config is not None:
                root = Path(work) / "traces" / name / f"{condition}-{repeat}"
                traces = _attacher(root, name, config)
            record = measure(campaign, view, work, traces=traces, **chronological)
            runs[condition].append(_rates(record))
    result = {}
    for key in runs["off"][0]:
        per = {c: [run[key]["rate"] for run in runs[c]] for c in CONDITIONS}
        seconds = {c: [1 / rate for rate in values] for c, values in per.items()}
        result[key] = dict(
            rows_per_second={c: statistics.median(v) for c, v in per.items()},
            overhead={c: _summary(seconds[c], seconds["off"]) for c in CONDITIONS[1:]},
            peak_vram_allocated_bytes={
                c: runs[c][-1][key]["peak_vram_allocated_bytes"] for c in CONDITIONS
            },
            volume={c: runs[c][-1][key]["learning_traces"] for c in CONDITIONS[1:]},
        )
    return result


# Eventos de las familias que no se pueden recorrer sin ajustar


def measure_events(view, campaign, work, declared, repeats=200):
    """Coste de los eventos de Ridge (uno por ajuste) y XGBoost (uno por ronda).

    Ridge registra la puntuación, el término independiente y las estadísticas de sus
    coeficientes, con la anchura aplanada de la entrada de la vista. XGBoost registra tres
    escalares por ronda y vuelca al confirmar cada punto de control.
    """
    import numpy as np
    import torch

    from mars_titan.training.reference_run import configured_corpus

    from .recorder import TraceRecorder

    dataset = configured_corpus(Path(view), input_policy=campaign["input_policy"])
    first = next(dataset.batches(partition="train", batch_size=1, epoch=0, seed=0))
    width = int(sum(np.prod(value.shape[1:]) for value in first["inputs"].values()))
    coefficient = torch.randn(
        width, dtype=torch.float64, generator=torch.Generator().manual_seed(0)
    )
    config = TraceConfig(1, declared["max_bytes"])
    ridge = TraceRecorder(
        Path(work) / "events/ridge", config, dict(family="ridge"), step_unit="fit"
    )
    started = time.perf_counter()
    with ridge.at(0, "validation"):
        ridge.scalar("validation.score", 0.5)
        ridge.tensor("ridge.coefficient", coefficient)
        ridge.scalar("ridge.intercept", 0.0)
    ridge.flush()
    ridge_seconds = time.perf_counter() - started
    xgboost = TraceRecorder(
        Path(work) / "events/xgboost", config, dict(family="xgboost"), step_unit="boosting_round"
    )
    started = time.perf_counter()
    for count in range(1, repeats + 1):
        with xgboost.at(count, "validation"):
            xgboost.scalar("validation.score", 0.5)
            xgboost.scalar("validation.best_score", 0.5)
            xgboost.scalar("validation.rounds_without_improvement", 0)
        if count % 50 == 0:
            xgboost.flush()
    xgboost.flush()
    return dict(
        ridge=dict(coefficients=width, seconds=ridge_seconds, bytes=ridge.bytes),
        xgboost=dict(
            rounds=repeats,
            seconds_per_round=(time.perf_counter() - started) / repeats,
            bytes_per_round=xgboost.bytes / repeats,
        ),
    )


def measure(
    campaign_path,
    view,
    work,
    *,
    declared,
    stage_path=None,
    extensions=None,
    families=FAMILIES,
    settings,
):
    """Medir las familias pedidas y devolver el informe."""
    import torch

    from mars_titan.data.embeddings import require_cuda
    from mars_titan.posttraining.campaign_stage import load_stage
    from mars_titan.training import campaign_numerics
    from mars_titan.training import campaign_throughput as throughput
    from mars_titan.training.campaign_plan import load_campaign

    _require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") in {":4096:8", ":16:8"},
        "Configura CUBLAS_WORKSPACE_CONFIG antes de iniciar PyTorch",
    )
    _require(set(families) <= set(FAMILIES), f"Familias conocidas: {FAMILIES}")
    TraceConfig(declared["every"], declared["max_bytes"])
    device = require_cuda()
    campaign = load_campaign(campaign_path)
    if extensions is not None:
        from mars_titan.training.campaign_extensions import extended_campaign, load_extensions

        campaign = extended_campaign(load_extensions(extensions), campaign)
    if campaign.get("numerics"):
        campaign_numerics.apply(campaign["numerics"])
    work = Path(work)
    load = dict(before=_gpu_load())
    started = time.perf_counter()
    results = {}
    guard = throughput._forbid_steps()
    try:
        if "neural" in families:
            results["neural"] = measure_neural(campaign, view, work, declared, settings, device)
        if "posttraining" in families:
            _require(stage_path is not None, "El postentrenamiento necesita su etapa")
            results["posttraining"] = measure_posttraining(
                load_stage(stage_path),
                view,
                work,
                declared,
                settings,
                device,
                settings.get("arms"),
            )
    finally:
        guard.remove()
    chronological = dict(
        titans_mac=(throughput.measure_titans, throughput.TITANS),
        episodic_gru=(throughput.measure_candidate, throughput.EPISODIC),
        mars_titan=(throughput.measure_mars_titan, throughput.MARS),
        cm_v1=(throughput.measure_cm_v1, throughput.CM),
    )
    for name, (function, section) in chronological.items():
        if name not in families:
            continue
        _require(campaign.get(section), f"La campaña no declara {section}")
        results[name] = measure_chronological(
            name, function, _measured(campaign, section), view, work, declared, settings
        )
    results["events"] = measure_events(view, campaign, work, declared)
    load["after"] = _gpu_load()
    return dict(
        schema_version=SCHEMA_VERSION,
        kind=KIND,
        measured_at_utc=datetime.now(UTC).isoformat(),
        campaign=str(campaign_path),
        view=str(Path(view).resolve()),
        declared=declared,
        settings=settings,
        device=dict(
            name=torch.cuda.get_device_name(device),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            numerics=campaign_numerics.current(),
        ),
        concurrent_gpu_load=load,
        results=results,
        seconds=time.perf_counter() - started,
        optimizer_steps=0,
        scientific_training_started=False,
        final_test_opened=False,
    )


def _measured(campaign, section):
    """Modelos medidos de una familia cronológica.

    Titans-MAC y la GRU candidata miden su primer modelo, porque los demás comparten
    entrenador y ganchos. MARS-TITAN mide su primer lector y su primera corrección B6. CM-v1
    recorre siempre sus dos núcleos y sus cuatro lectores.
    """
    from mars_titan.training import campaign_throughput as throughput

    candidates = campaign[section]["candidates"]
    if section in (throughput.TITANS, throughput.EPISODIC):
        return _trimmed(campaign, section, [next(iter(candidates))])
    if section == throughput.MARS:
        corrections = [
            a for a, c in candidates.items() if "associative_memory" in c[0][1]["components"]
        ]
        readouts = [a for a in candidates if a not in corrections]
        return _trimmed(campaign, section, readouts[:1] + corrections[:1])
    return campaign


def main(argv=None):
    from mars_titan.data.storage import atomic_json

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--view", type=Path, required=True, help="Manifiesto de la vista")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", type=Path, help="Etapa de adaptadores")
    parser.add_argument("--extensions", type=Path, help="Declaración preparada de familias")
    parser.add_argument("--families", nargs="+", default=list(FAMILIES))
    parser.add_argument("--arms", nargs="+", help="Modelos base del postentrenamiento medidos")
    parser.add_argument("--every", type=int, required=True, help="Cadencia declarada")
    parser.add_argument("--max-mib", type=int, default=64)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chronological-repeats", type=int, default=2)
    parser.add_argument("--segments", type=int, default=6)
    parser.add_argument("--segment-warmup", type=int, default=1)
    parser.add_argument("--events", type=int, default=16)
    parser.add_argument("--event-warmup", type=int, default=2)
    args = parser.parse_args(argv)
    settings = dict(
        steps=args.steps,
        warmup=args.warmup,
        repeats=args.repeats,
        chronological_repeats=args.chronological_repeats,
        segments=args.segments,
        segment_warmup=args.segment_warmup,
        events=args.events,
        event_warmup=args.event_warmup,
        arms=args.arms,
    )
    report = measure(
        args.campaign,
        args.view,
        args.work,
        declared=dict(every=args.every, max_bytes=args.max_mib * MIB),
        stage_path=args.stage,
        extensions=args.extensions,
        families=args.families,
        settings=settings,
    )
    atomic_json(args.output, report)
    print(json.dumps(dict(output=str(args.output), seconds=report["seconds"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
