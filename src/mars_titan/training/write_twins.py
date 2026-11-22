"""Gemelos de escritura de Titans-MAC: la misma red ajustada, evaluada escribiendo o sin escribir.

Prueba causal declarada el 22 de noviembre de 2026: ¿cuánto de la ventaja de `titans_mac` sobre su
control congelado viene de escribir en la memoria durante la evaluación y cuánto de los
parámetros que dejó un ajuste con escritura? Para cada estado, origen, semilla y par declarado
(`online` ajustado con `mac_online`, `frozen` su control `mac_frozen`), en la propia ventana y
con la vista de los trabajos:

- paridad: cada ancla se regenera con su propia variante (`carry_titans(..., regenerate=True)`) y
  sus tres tramos tienen que salir bit a bit iguales a los emitidos
  (`prediction_regeneration.compare`). Si no, la orden se detiene con la primera fila distinta y
  no se interpreta nada: el montaje no reproduce lo que se mide;
- `online_fit_frozen_eval`: el estado elegido de `online` evaluado sin escribir (gemelo
  `mac_frozen` con todos sus tensores, parámetros y buffers);
- `frozen_fit_online_eval`: el estado elegido de `frozen` evaluado escribiendo, el control (a)
  del gemelo inverso, que debería empeorar o quedar igual.

No ajusta pesos, selección ni normalizadores ni abre el test final, y escribe cada trabajo en un
directorio nuevo; las emisiones de la campaña solo se leen. Un trabajo terminado se reutiliza
después de comprobar sus huellas y uno interrumpido se aparta sin borrarlo.

`analizar` (CPU) mide en cada tramo el IC de rangos por sesión (Spearman con al menos
`min_assets` activos, la métrica de la criba) de las cuatro celdas ajuste × evaluación: oo
(emitida por `online`), of (gemelo sin escribir), ff (emitida por `frozen`) y fo (gemelo
escribiendo). Por sesión, con las semillas promediadas como réplicas:

- N = IC(oo) − IC(of), la ganancia de escribir en evaluación con los parámetros del ajuste en línea;
- D = IC(oo) − IC(ff), la ventaja completa de la memoria en línea sobre su control;
- A = IC(fo) − IC(ff), escribir con los parámetros de un ajuste congelado (control a);
- I = N − A, la interacción entre ajustar con escritura y evaluar con escritura.

Cada media lleva su error HAC de Newey-West (núcleo de Bartlett, retardo automático
floor(4 (T/100)^(2/9)) por bloque de estado, origen y mercado, el de la criba) y su efecto mínimo
detectable 2,8 × EE (80 % de potencia, α = 0,05 bilateral). TTWS = N̄ / D̄ es la proporción de la
ventaja atribuible a la escritura en evaluación, con error por el método delta y el intervalo de
Fieller de la covarianza HAC conjunta de (N, D). Si D̄ no se distingue de cero con la confianza
declarada, el intervalo de Fieller no está acotado y el cociente no se interpreta.

Cada tramo lleva además el perfil de N, D, A e I por trimestre natural de sus sesiones y su
pendiente por año (mínimos cuadrados con error HAC de la puntuación), para ver si la ventaja de
escribir decae a lo largo del recorrido.

`reinicios` (GPU) añade a un par con sus gemelos hechos los brazos con reinicio de la memoria
rápida, aprobados por el coordinador el 22 de noviembre de 2026: el estado elegido de `online`
evaluado escribiendo en la evaluación, pero desde el estado inicial de la memoria al abrirse la
decisión y sin calcular el calentamiento (R0), o además volviendo a ese estado cada 63 instantes
medidos (R63). Se reinicia todo: pesos rápidos, momentos y cola de cada flujo, y el contexto de
mercado. Antes de ningún brazo, la calibración se regenera sin reinicio con el mismo código y
tiene que salir bit a bit igual a la emitida; si no, la orden se detiene. `analizar-reinicios`
(CPU) da, en la evaluación, R − oo y R − of por sesión, con su media HAC, su perfil por trimestre
y su pendiente: R0 − oo es lo que aportan las escrituras del calentamiento.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pyarrow.parquet as pq

from mars_titan.data import prediction_files
from mars_titan.data.storage import atomic_json, sha256

KIND = "posttraining_write_twins_v1"
PARTITIONS = ("validation", "calibration", "evaluation")
JOBS = {
    "parity_online": ("online", None),
    "parity_frozen": ("frozen", None),
    "online_fit_frozen_eval": ("online", "mac_frozen"),
    "frozen_fit_online_eval": ("frozen", "mac_online"),
}
CELLS = ("oo", "of", "ff", "fo")
# Brazos con reinicio: periodo en instantes medidos (None: solo al abrirse la decisión).
RESETS = {"R0": None, "R63": 63}
RESET_PARITY = "parity_online_calibration"
YEAR_US = 365.25 * 86_400_000_000
FIRST_ORIGIN, LAST_ORIGIN = 2005, 2023
_DECLARATION = {
    "schema_version",
    "kind",
    "final_test_access",
    "description",
    "market",
    "min_assets",
    "pairs",
    "runs",
    "analysis",
}
_ANALYSIS = {"primary", "secondary", "confidence"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, minimum, maximum=None):
    return type(value) is int and value >= minimum and (maximum is None or value <= maximum)


def checked_declaration(document):
    """Validar la declaración antes de abrir ningún estado."""
    _require(
        isinstance(document, dict) and set(document) == _DECLARATION,
        f"La declaración de los gemelos tiene exactamente {sorted(_DECLARATION)}",
    )
    _require(
        document["schema_version"] == 1 and document["kind"] == KIND,
        f"La declaración debe ser {KIND} con schema_version 1",
    )
    _require(document["final_test_access"] is False, "Los gemelos no abren el test final")
    _require(
        isinstance(document["description"], str) and document["description"].strip(),
        "La declaración se describe",
    )
    _require(document["market"] == "US", "Los gemelos se declaran sobre el mercado US")
    _require(_integer(document["min_assets"], 2), "min_assets es un entero mayor que uno")
    pairs = document["pairs"]
    _require(
        isinstance(pairs, list)
        and pairs
        and all(
            isinstance(pair, dict)
            and set(pair) == {"online", "frozen"}
            and all(isinstance(pair[key], str) and pair[key] for key in pair)
            and pair["online"] != pair["frozen"]
            for pair in pairs
        )
        and len({pair["online"] for pair in pairs}) == len(pairs),
        "Cada par declara un modelo online y su control frozen, distintos y sin repetir",
    )
    runs = document["runs"]
    _require(
        isinstance(runs, list)
        and runs
        and all(
            isinstance(run, dict)
            and set(run) == {"state", "origin", "seeds"}
            and isinstance(run["state"], str)
            and run["state"]
            and _integer(run["origin"], FIRST_ORIGIN, LAST_ORIGIN)
            and isinstance(run["seeds"], list)
            and run["seeds"]
            and all(_integer(seed, 0) for seed in run["seeds"])
            and run["seeds"] == sorted(set(run["seeds"]))
            for run in runs
        )
        and len({(run["state"], run["origin"]) for run in runs}) == len(runs),
        "Cada ejecución declara estado, origen anterior al test final y semillas crecientes",
    )
    analysis = document["analysis"]
    _require(
        isinstance(analysis, dict)
        and set(analysis) == _ANALYSIS
        and analysis["primary"] in PARTITIONS
        and isinstance(analysis["secondary"], list)
        and all(
            name in PARTITIONS and name != analysis["primary"] for name in analysis["secondary"]
        )
        and len(set(analysis["secondary"])) == len(analysis["secondary"])
        and type(analysis["confidence"]) is float
        and 0.5 < analysis["confidence"] < 1,
        "El análisis declara su tramo principal, los secundarios y la confianza",
    )
    return document


def _canonical(document):
    return json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _saved_declaration(output, declaration):
    """Guardar la declaración en la salida o comprobar que es la misma que ya estaba."""
    output.mkdir(parents=True, exist_ok=True)
    saved = output / "declaration.json"
    if saved.exists():
        previous = json.loads(saved.read_text(encoding="utf-8"))
        _require(
            _canonical(previous) == _canonical(declaration),
            f"{output} pertenece a otra declaración",
        )
    else:
        atomic_json(saved, declaration)
    return hashlib.sha256(_canonical(declaration).encode()).hexdigest()


def _stem(run, seed, pair):
    return f"{run['state']}--{run['origin']}--seed-{seed}--{pair['online']}"


def anchors(states, run, seed, pair):
    """Carpetas de los dos ajustes de un par y la vista con que se ajustaron, comprobadas."""
    _require(run["state"] in states, f"Falta el estado {run['state']}")
    root = Path(states[run["state"]])
    folders = {
        role: root / "runs" / pair[role] / str(run["origin"]) / f"seed-{seed}"
        for role in ("online", "frozen")
    }
    reports = {}
    for role, folder in folders.items():
        path = folder / "run.json"
        _require(path.is_file(), f"Falta el trabajo {folder}")
        reports[role] = json.loads(path.read_text(encoding="utf-8"))
        _require(
            reports[role].get("status") == "completed"
            and reports[role].get("final_test_opened") is False,
            f"El trabajo {folder} no está terminado",
        )
    expected = {"online": "mac_online", "frozen": "mac_frozen"}
    for role, report in reports.items():
        _require(
            report["request"]["variant"] == expected[role],
            f"{folders[role]} no es un ajuste {expected[role]}",
        )
        _require(
            report["request"]["seed"] == seed,
            f"{folders[role]} no es de la semilla {seed}",
        )
    _require(
        reports["online"]["request"]["view_sha256"] == reports["frozen"]["request"]["view_sha256"]
        and reports["online"]["request"]["recipe_sha256"]
        == reports["frozen"]["request"]["recipe_sha256"],
        "Los dos ajustes del par no comparten vista y receta",
    )
    view = root / "cohorts" / str(run["origin"]) / "manifest.json"
    _require(view.is_file(), f"No está la vista {view}")
    _require(
        sha256(view) == reports["online"]["request"]["view_sha256"],
        f"La vista {view} no es la de los ajustes",
    )
    return folders, reports, view


def _bits(values):
    values = np.ascontiguousarray(values, dtype=np.float64)
    return values.view(np.int64)


def first_difference(original, regenerated):
    """Primera fila distinta entre dos tablas de predicciones, por muestra, o None."""
    left = pq.read_table(original).sort_by("sample_id")
    right = pq.read_table(regenerated).sort_by("sample_id")
    if left.schema.names != right.schema.names or left.num_rows != right.num_rows:
        return dict(reason="Las tablas no tienen las mismas columnas ni filas")
    samples = left["sample_id"].to_pylist()
    if samples != right["sample_id"].to_pylist():
        return dict(reason="Las tablas no tienen las mismas muestras")
    for name in left.schema.names:
        a, b = left[name], right[name]
        if a.equals(b):
            continue
        try:
            differ = np.flatnonzero(_bits(a.to_numpy()) != _bits(b.to_numpy()))
        except (TypeError, ValueError):
            differ = [
                i
                for i, (x, y) in enumerate(zip(a.to_pylist(), b.to_pylist(), strict=True))
                if x != y
            ]
        row = int(differ[0])
        return dict(
            column=name,
            row=row,
            sample_id=samples[row],
            original=repr(a[row].as_py()),
            regenerated=repr(b[row].as_py()),
            differing_rows=len(differ),
        )
    return None


def _aside(destination):
    """Apartar sin borrar un trabajo que quedó a medias."""
    for index in range(1, 1000):
        moved = destination.with_name(f"{destination.name}.interrumpido-{index}")
        if not moved.exists():
            destination.rename(moved)
            return moved
    raise ValueError(f"Demasiados trabajos interrumpidos junto a {destination}")


def _finished(destination, evaluated, partitions=PARTITIONS, memory_reset=None):
    """Recibo de un trabajo ya terminado de esta prueba, con sus tablas comprobadas, o None."""
    path = destination / "carry.json"
    if not path.is_file():
        return None
    receipt = json.loads(path.read_text(encoding="utf-8"))
    switch = receipt.get("write_policy")
    _require(
        receipt.get("status") == "completed"
        and receipt.get("regenerated") is True
        and (switch is None if evaluated is None else switch["evaluated_variant"] == evaluated)
        and set(receipt["predictions"]) == set(partitions)
        and receipt.get("memory_reset")
        == (None if memory_reset is None else memory_reset.record()),
        f"{destination} no es un trabajo terminado de esta prueba",
    )
    for record in receipt["predictions"].values():
        prediction_files.verify(
            destination / record["path"], record["sha256"], label="Han cambiado las predicciones"
        )
    return receipt


def _reset_peak(device):
    """Pone a cero los máximos de memoria de la GPU antes de un recorrido.

    El asignador de PyTorch no los admite antes de inicializar CUDA («Invalid device argument»)
    y el primer recorrido de la orden empieza sin contexto.
    """
    if device.startswith("cuda"):
        import torch

        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(0)


def _peak(device):
    """Memoria máxima del proceso: residente en el anfitrión y reservada en la GPU."""
    peak = dict(host_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    if device.startswith("cuda"):
        import torch

        peak.update(
            cuda_max_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
            cuda_max_reserved_bytes=int(torch.cuda.max_memory_reserved(0)),
        )
    return peak


def run_twins(declaration, states, output, *, device="cuda:0", log=print, carry=None):
    """Paridad y gemelos de cada par, origen y semilla declarados, en directorios nuevos."""
    from . import prediction_regeneration as regeneration
    from . import titans_walk_forward as wf
    from .window_indices import SHARED_INDICES_ENV

    declaration = checked_declaration(declaration)
    output = Path(output)
    digest = _saved_declaration(output, declaration)
    numerics = regeneration.strict_fp32()
    carry = carry or wf.carry_titans
    # Los cuatro recorridos de una vista comparten sus índices de entrada, solo durante la orden.
    shared = os.environ.get(SHARED_INDICES_ENV)
    if shared is None:
        os.environ[SHARED_INDICES_ENV] = str(output / "indices")
    try:
        return _twins(declaration, states, output, digest, numerics, device, log, carry)
    finally:
        if shared is None:
            os.environ.pop(SHARED_INDICES_ENV, None)


def _twins(declaration, states, output, digest, numerics, device, log, carry):
    from . import prediction_regeneration as regeneration
    from . import titans_walk_forward as wf

    summaries = []
    for run in declaration["runs"]:
        for seed in run["seeds"]:
            for pair in declaration["pairs"]:
                folders, reports, view = anchors(states, run, seed, pair)
                folder = output / _stem(run, seed, pair)
                folder.mkdir(parents=True, exist_ok=True)
                jobs = {}
                for name, (role, evaluated) in JOBS.items():
                    destination = folder / name
                    receipt = _finished(destination, evaluated)
                    started, computed = time.perf_counter(), receipt is None
                    if computed:
                        if destination.exists():
                            log(f"{folder.name}/{name}: apartado en {_aside(destination).name}")
                        _reset_peak(device)
                        with wf.unfused_attention():
                            receipt = carry(
                                folders[role],
                                view,
                                view,
                                destination,
                                device=device,
                                regenerate=True,
                                evaluated_variant=evaluated,
                            )
                        log(f"{folder.name}/{name}: {time.perf_counter() - started:.0f} s")
                    else:
                        log(f"{folder.name}/{name}: ya hecho")
                    job = dict(
                        anchor=str(folders[role]),
                        evaluated_variant=evaluated or reports[role]["request"]["variant"],
                        receipt_sha256=sha256(destination / "carry.json"),
                        seconds=receipt.get("seconds"),
                        peak=_peak(device) if computed else None,
                    )
                    if evaluated is None:
                        comparison = regeneration.compare(
                            regeneration.originals(folders[role], reports[role]),
                            destination,
                            receipt,
                        )
                        job["parity"] = comparison
                        if not comparison["identical"]:
                            wrong = next(
                                partition
                                for partition, item in comparison["partitions"].items()
                                if not item["identical"]
                            )
                            item = comparison["partitions"][wrong]
                            detail = first_difference(item["original"], item["regenerated"])
                            atomic_json(folder / "parity-failure.json", dict(job, first=detail))
                            raise ValueError(
                                f"La paridad de {folders[role]} falla en {wrong}: {detail}. "
                                "El montaje no reproduce lo emitido y no se interpreta nada"
                            )
                    else:
                        job["write_policy"] = receipt["write_policy"]
                    jobs[name] = job
                summary = dict(
                    schema_version=1,
                    kind=f"{KIND}_jobs",
                    declaration_sha256=digest,
                    state=run["state"],
                    origin=run["origin"],
                    seed=seed,
                    pair=pair,
                    view_sha256=reports["online"]["request"]["view_sha256"],
                    anchors={
                        role: dict(
                            folder=str(folders[role]),
                            run_sha256=sha256(folders[role] / "run.json"),
                            checkpoint_sha256=reports[role]["checkpoint"]["sha256"],
                        )
                        for role in folders
                    },
                    jobs=jobs,
                    device=device,
                    numerics=numerics,
                    final_test_opened=False,
                    finished_at_utc=datetime.now(UTC).isoformat(),
                )
                atomic_json(folder / "twins.json", summary)
                summaries.append(summary)
    return summaries


def _pair(declaration, online):
    pairs = [pair for pair in declaration["pairs"] if pair["online"] == online]
    _require(pairs, f"El par {online} no está declarado")
    return pairs[0]


def run_resets(
    declaration, states, output, online, arms, *, device="cuda:0", log=print, carry=None
):
    """Brazos con reinicio de un par con sus gemelos hechos, tras una paridad sin reinicio."""
    from . import prediction_regeneration as regeneration
    from . import titans_walk_forward as wf
    from .window_indices import SHARED_INDICES_ENV

    declaration = checked_declaration(declaration)
    pair = _pair(declaration, online)
    _require(
        isinstance(arms, list | tuple)
        and arms
        and all(arm in RESETS for arm in arms)
        and len(set(arms)) == len(arms),
        f"Los brazos con reinicio son {sorted(RESETS)}, sin repetir",
    )
    output = Path(output)
    digest = _saved_declaration(output, declaration)
    numerics = regeneration.strict_fp32()
    carry = carry or wf.carry_titans
    # Los brazos leen los índices de entrada que ya compartieron los gemelos.
    shared = os.environ.get(SHARED_INDICES_ENV)
    if shared is None:
        os.environ[SHARED_INDICES_ENV] = str(output / "indices")
    try:
        return _resets(
            declaration, states, output, pair, arms, digest, numerics, device, log, carry
        )
    finally:
        if shared is None:
            os.environ.pop(SHARED_INDICES_ENV, None)


def _resets(declaration, states, output, pair, arms, digest, numerics, device, log, carry):
    from . import prediction_regeneration as regeneration
    from . import titans_walk_forward as wf
    from .financial_run import MemoryReset

    plan = [(RESET_PARITY, ("calibration",), None)] + [
        (f"reset_{arm}", ("evaluation",), MemoryReset(RESETS[arm])) for arm in arms
    ]
    summaries = []
    for run in declaration["runs"]:
        for seed in run["seeds"]:
            folders, reports, view = anchors(states, run, seed, pair)
            # Sin los gemelos con su paridad exacta no hay celdas oo y of con que comparar.
            _cells(states, output, run, seed, pair)
            folder = output / _stem(run, seed, pair)
            jobs = {}
            for name, partitions, reset in plan:
                destination = folder / name
                receipt = _finished(destination, None, partitions, reset)
                started, computed = time.perf_counter(), receipt is None
                if computed:
                    if destination.exists():
                        log(f"{folder.name}/{name}: apartado en {_aside(destination).name}")
                    _reset_peak(device)
                    with wf.unfused_attention():
                        receipt = carry(
                            folders["online"],
                            view,
                            view,
                            destination,
                            device=device,
                            regenerate=True,
                            evaluated_variant=None,
                            partitions=partitions,
                            memory_reset=reset,
                        )
                    log(f"{folder.name}/{name}: {time.perf_counter() - started:.0f} s")
                else:
                    log(f"{folder.name}/{name}: ya hecho")
                job = dict(
                    anchor=str(folders["online"]),
                    partitions=list(partitions),
                    memory_reset=None if reset is None else reset.record(),
                    receipt_sha256=sha256(destination / "carry.json"),
                    seconds=receipt.get("seconds"),
                    peak=_peak(device) if computed else None,
                )
                if reset is None:
                    originals = regeneration.originals(folders["online"], reports["online"])
                    comparison = regeneration.compare(
                        {partition: originals[partition] for partition in partitions},
                        destination,
                        receipt,
                    )
                    job["parity"] = comparison
                    if not comparison["identical"]:
                        item = comparison["partitions"][partitions[0]]
                        detail = first_difference(item["original"], item["regenerated"])
                        atomic_json(folder / "resets-parity-failure.json", dict(job, first=detail))
                        raise ValueError(
                            f"La paridad sin reinicio de {folders['online']} falla en "
                            f"{partitions[0]}: {detail}. No se lanza ningún brazo con reinicio"
                        )
                else:
                    metrics = receipt["predictions"]["evaluation"]["metrics"]
                    job.update(
                        {key: metrics[key] for key in ("memory_resets", "measured_instants")}
                    )
                jobs[name] = job
            summary = dict(
                schema_version=1,
                kind=f"{KIND}_resets",
                declaration_sha256=digest,
                state=run["state"],
                origin=run["origin"],
                seed=seed,
                pair=pair,
                arms=list(arms),
                view_sha256=reports["online"]["request"]["view_sha256"],
                jobs=jobs,
                device=device,
                numerics=numerics,
                final_test_opened=False,
                finished_at_utc=datetime.now(UTC).isoformat(),
            )
            atomic_json(folder / "resets.json", summary)
            summaries.append(summary)
    return summaries


def _screening():
    from mars_titan.evaluation.posttraining_pilots import screening

    return screening()


def session_ics(folder, partition, min_assets):
    """IC de rangos por sesión de un tramo, con las funciones de la criba, y sus filas."""
    module = _screening()
    data = module.load_split(Path(folder), partition)
    _require(data is not None, f"Faltan las predicciones de {partition} en {folder}")
    ics = {}
    for market, instant, start, end in data["sessions"]:
        if end - start < min_assets:
            continue
        value = module.correlation(
            module.average_ranks(data["prediction"][start:end]),
            module.average_ranks(data["target"][start:end]),
        )
        if not math.isnan(value):
            ics[(market, instant)] = value
    return ics, data["sample"], data["target"]


def long_run_covariance(values, blocks=None):
    """Covarianza HAC de las medias de varias series alineadas (Bartlett, retardo automático).

    Es la de `dgx_screening_report._newey_west_hac` extendida a varias columnas: cada bloque
    suma su propio núcleo con su retardo automático y el total se divide entre T².
    """
    module = _screening()
    values = np.asarray(values, dtype=np.float64)
    count = values.shape[0]
    _require(values.ndim == 2 and count >= 2, "La covarianza necesita al menos dos sesiones")
    centered = values - values.mean(axis=0)
    groups = {}
    for index, block in enumerate([None] * count if blocks is None else blocks):
        groups.setdefault(block, []).append(index)
    total = np.zeros((values.shape[1], values.shape[1]))
    lags = 0
    for indices in groups.values():
        u = centered[np.asarray(indices)]
        size = u.shape[0]
        lag = module.automatic_newey_west_lags(size) if size > 1 else 0
        lags = max(lags, lag)
        block = u.T @ u
        for k in range(1, lag + 1):
            gamma = u[k:].T @ u[:-k]
            block += (1.0 - k / (lag + 1.0)) * (gamma + gamma.T)
        total += block
    return total / count**2, lags


def _mean_statistics(values, blocks, z):
    module = _screening()
    t, se, lags = module._newey_west_hac(values, blocks=blocks)
    mean = float(np.mean(values))
    return dict(
        mean=mean,
        se_hac=se,
        t=t,
        hac_lags=lags,
        interval=None if se is None else [mean - z * se, mean + z * se],
        mde80=None if se is None else 2.8 * se,
        sessions=len(values),
    )


def ratio_statistics(numerator, denominator, confidence, blocks=None):
    """Cociente de medias N̄/D̄ con error delta e intervalo de Fieller de la covarianza HAC."""
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    n, d = float(np.mean(numerator)), float(np.mean(denominator))
    covariance, lags = long_run_covariance(np.column_stack([numerator, denominator]), blocks)
    vnn, vnd, vdd = covariance[0, 0], covariance[0, 1], covariance[1, 1]
    result = dict(estimate=None, delta_se=None, fieller=None, hac_lags=lags)
    if d == 0:
        result["fieller"] = dict(bounded=False, reason="La ventaja de la memoria en línea es cero")
        return result
    ratio = n / d
    gradient = np.array([1.0 / d, -n / d**2])
    result["estimate"] = ratio
    result["delta_se"] = float(math.sqrt(max(gradient @ covariance @ gradient, 0.0)))
    a = d**2 - z**2 * vdd
    b = -2.0 * (n * d - z**2 * vnd)
    c = n**2 - z**2 * vnn
    if a <= 0:
        result["fieller"] = dict(
            bounded=False,
            reason="La ventaja de la memoria en línea no se distingue de cero con esta "
            "confianza: el cociente no está acotado",
        )
        return result
    root = math.sqrt(max(b**2 - 4 * a * c, 0.0))
    result["fieller"] = dict(bounded=True, interval=[(-b - root) / (2 * a), (-b + root) / (2 * a)])
    return result


def _reading(statistics, z):
    t = statistics["t"]
    if t is None:
        return "not_estimable"
    if t >= z:
        return "positive"
    if t <= -z:
        return "negative"
    return "not_detected"


def factorial_contrasts(ic):
    """N, D, A e I sesión a sesión a partir del IC de las cuatro celdas ajuste × evaluación."""
    contrasts = dict(N=ic["oo"] - ic["of"], D=ic["oo"] - ic["ff"], A=ic["fo"] - ic["ff"])
    contrasts["I"] = contrasts["N"] - contrasts["A"]
    return contrasts


def seed_average(seeds, keys):
    """IC de cada celda en las sesiones `keys`, promediado sesión a sesión entre semillas."""
    return {
        cell: np.mean([[values[cell][key] for key in keys] for values in seeds.values()], axis=0)
        for cell in CELLS
    }


def _quarter(instant):
    """Trimestre natural (UTC) de un instante en microseconds: «2013T1»."""
    moment = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=int(instant))
    return f"{moment.year}T{(moment.month - 1) // 3 + 1}"


def trend_statistics(values, instants, blocks):
    """Pendiente por año de una serie de sesiones (MCO) con su error HAC por bloque.

    Var(b) = Var(Σ (x − x̄)·e) / Sxx², con la varianza de largo plazo de la puntuación (x − x̄)·e
    del error de Newey-West de la criba (Bartlett, retardo automático por bloque).
    """
    values = np.asarray(values, dtype=np.float64)
    years = (np.asarray(instants, dtype=np.float64) - float(np.min(instants))) / YEAR_US
    centered = years - years.mean()
    spread = float(np.sum(centered**2))
    if len(values) < 3 or spread == 0:
        return dict(slope_per_year=None, se_hac=None, t=None, hac_lags=0, sessions=len(values))
    slope = float(np.sum(centered * (values - values.mean())) / spread)
    residual = values - values.mean() - slope * centered
    _, se_mean, lags = _screening()._newey_west_hac(centered * residual, blocks=blocks)
    se = None if se_mean is None else float(se_mean) * len(values) / spread
    return dict(
        slope_per_year=slope,
        se_hac=se,
        t=None if not se else slope / se,
        hac_lags=lags,
        sessions=len(values),
    )


def profile_statistics(contrasts, blocks, instants, z):
    """Cada contraste por trimestre natural de sus sesiones y su pendiente por año."""
    labels = np.array([_quarter(instant) for instant in instants])
    quarters = {}
    for label in sorted(set(labels.tolist())):
        rows = np.flatnonzero(labels == label)
        inside = [blocks[i] for i in rows]
        quarters[label] = dict(
            sessions=len(rows),
            **{
                name: _mean_statistics(values[rows], inside, z)
                for name, values in contrasts.items()
            },
        )
    trend = {name: trend_statistics(values, instants, blocks) for name, values in contrasts.items()}
    return quarters, trend


def partition_statistics(ic, blocks, confidence, instants=None):
    """Medias de las celdas, contrastes con su error HAC, sus lecturas y el cociente TTWS.

    Con `instants` (el instante de cada sesión) añade el perfil de cada contraste por trimestre
    natural y su pendiente por año, con errores HAC.
    """
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    contrasts = factorial_contrasts(ic)
    statistics = {name: _mean_statistics(values, blocks, z) for name, values in contrasts.items()}
    readings = {name: _reading(value, z) for name, value in statistics.items()}
    ratio = ratio_statistics(contrasts["N"], contrasts["D"], confidence, blocks)
    ratio["interpretable"] = readings["D"] == "positive" and ratio["fieller"]["bounded"]
    result = dict(
        sessions=len(blocks),
        ic={cell: float(ic[cell].mean()) for cell in CELLS},
        contrasts=statistics,
        readings=readings,
        ttws=ratio,
    )
    if instants is not None:
        _require(len(instants) == len(blocks), "Cada sesión del perfil necesita su instante")
        result["quarters"], result["trend"] = profile_statistics(contrasts, blocks, instants, z)
    return result


def _cells(states, output, run, seed, pair):
    """Carpetas de las cuatro celdas de un par, después de comprobar su paridad."""
    folders, _, _ = anchors(states, run, seed, pair)
    folder = Path(output) / _stem(run, seed, pair)
    path = folder / "twins.json"
    _require(path.is_file(), f"Faltan los gemelos de {folder}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    for name in ("parity_online", "parity_frozen"):
        _require(
            summary["jobs"][name]["parity"]["identical"] is True,
            f"La paridad de {folder / name} no es exacta: no se interpreta nada",
        )
    return dict(
        oo=folders["online"],
        of=folder / "online_fit_frozen_eval",
        ff=folders["frozen"],
        fo=folder / "frozen_fit_online_eval",
    )


def analyze(declaration, states, output):
    """Informe de los gemelos: IC de las cuatro celdas y contrastes N, D, A, I y TTWS por tramo."""
    declaration = checked_declaration(declaration)
    output = Path(output)
    digest = _saved_declaration(output, declaration)
    analysis = declaration["analysis"]
    report = dict(
        schema_version=1,
        kind=f"{KIND}_report",
        declaration_sha256=digest,
        final_test_opened=False,
        min_assets=declaration["min_assets"],
        confidence=analysis["confidence"],
        pairs={},
    )
    for pair in declaration["pairs"]:
        partitions = {}
        for partition in (analysis["primary"], *analysis["secondary"]):
            per_seed, series, blocks, instants = {}, {}, [], []
            for run in declaration["runs"]:
                seeds = {}
                for seed in run["seeds"]:
                    cells = _cells(states, output, run, seed, pair)
                    values, reference = {}, None
                    for cell, folder in cells.items():
                        ics, sample, target = session_ics(
                            folder, partition, declaration["min_assets"]
                        )
                        if reference is None:
                            reference = (sample, target)
                        _require(
                            sample == reference[0] and np.array_equal(target, reference[1]),
                            f"Las filas de {partition} de {folder} no son las de las demás celdas",
                        )
                        values[cell] = ics
                    seeds[seed] = values
                common = set.intersection(*(set(v[c]) for v in seeds.values() for c in CELLS))
                _require(common, f"{run['state']}/{run['origin']} no tiene sesiones comunes")
                keys = sorted(common)
                for seed, values in seeds.items():
                    rows = {c: np.array([values[c][k] for k in keys]) for c in CELLS}
                    per_seed[f"{run['state']}/{run['origin']}/seed-{seed}"] = dict(
                        sessions=len(keys),
                        ic={c: float(rows[c].mean()) for c in CELLS},
                        **{
                            name: float(value.mean())
                            for name, value in factorial_contrasts(rows).items()
                        },
                    )
                for cell, averaged in seed_average(seeds, keys).items():
                    series.setdefault(cell, []).append(averaged)
                blocks += [(run["state"], run["origin"], key[0]) for key in keys]
                instants += [key[1] for key in keys]
            ic = {c: np.concatenate(series[c]) for c in CELLS}
            statistics = partition_statistics(ic, blocks, analysis["confidence"], instants=instants)
            partitions[partition] = dict(statistics, per_seed=per_seed)
        report["pairs"][pair["online"]] = dict(frozen=pair["frozen"], partitions=partitions)
    atomic_json(output / "report.json", report)
    return report


def reset_contrasts(ic, arms):
    """N y, por brazo con reinicio, R − oo (lo que pierde sin lo escrito antes) y R − of."""
    contrasts = dict(N=ic["oo"] - ic["of"])
    for arm in arms:
        contrasts[f"{arm}-oo"] = ic[arm] - ic["oo"]
        contrasts[f"{arm}-of"] = ic[arm] - ic["of"]
    return contrasts


def analyze_resets(declaration, states, output, online):
    """Brazos con reinicio frente a oo y of en la evaluación: medias HAC, trimestres y pendiente."""
    declaration = checked_declaration(declaration)
    pair = _pair(declaration, online)
    output = Path(output)
    digest = _saved_declaration(output, declaration)
    confidence = declaration["analysis"]["confidence"]
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    series, blocks, instants, arms = {}, [], [], None
    for run in declaration["runs"]:
        seeds = {}
        for seed in run["seeds"]:
            cells = _cells(states, output, run, seed, pair)
            folder = output / _stem(run, seed, pair)
            path = folder / "resets.json"
            _require(path.is_file(), f"Faltan los brazos con reinicio de {folder}")
            summary = json.loads(path.read_text(encoding="utf-8"))
            _require(
                summary["jobs"][RESET_PARITY]["parity"]["identical"] is True,
                f"La paridad sin reinicio de {folder} no es exacta: no se interpreta nada",
            )
            present = [arm for arm in RESETS if f"reset_{arm}" in summary["jobs"]]
            _require(
                present and (arms is None or present == arms),
                f"{folder} no tiene los mismos brazos con reinicio que las demás semillas",
            )
            arms = present
            columns = dict(
                oo=cells["oo"], of=cells["of"], **{arm: folder / f"reset_{arm}" for arm in arms}
            )
            values, reference = {}, None
            for name, source in columns.items():
                ics, sample, target = session_ics(source, "evaluation", declaration["min_assets"])
                if reference is None:
                    reference = (sample, target)
                _require(
                    sample == reference[0] and np.array_equal(target, reference[1]),
                    f"Las filas de evaluation de {source} no son las de las demás celdas",
                )
                values[name] = ics
            seeds[seed] = values
        common = set.intersection(*(set(item) for v in seeds.values() for item in v.values()))
        _require(common, f"{run['state']}/{run['origin']} no tiene sesiones comunes")
        keys = sorted(common)
        for name in seeds[run["seeds"][0]]:
            averaged = np.mean([[v[name][key] for key in keys] for v in seeds.values()], axis=0)
            series.setdefault(name, []).append(averaged)
        blocks += [(run["state"], run["origin"], key[0]) for key in keys]
        instants += [key[1] for key in keys]
    ic = {name: np.concatenate(values) for name, values in series.items()}
    contrasts = reset_contrasts(ic, arms)
    statistics = {name: _mean_statistics(values, blocks, z) for name, values in contrasts.items()}
    quarters, trend = profile_statistics(contrasts, blocks, instants, z)
    report = dict(
        schema_version=1,
        kind=f"{KIND}_resets_report",
        declaration_sha256=digest,
        final_test_opened=False,
        pair=pair,
        partition="evaluation",
        arms=arms,
        min_assets=declaration["min_assets"],
        confidence=confidence,
        sessions=len(blocks),
        ic={name: float(values.mean()) for name, values in ic.items()},
        contrasts=statistics,
        readings={name: _reading(value, z) for name, value in statistics.items()},
        quarters=quarters,
        trend=trend,
    )
    atomic_json(output / f"resets-{online}.json", report)
    return report


def _states(values):
    states = {}
    for value in values:
        name, separator, path = value.partition("=")
        _require(separator and name and path, f"--state espera NOMBRE=RUTA: {value!r}")
        _require(name not in states, f"Estado repetido: {name}")
        states[name] = path
    return states


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("gemelos", "analizar", "reinicios", "analizar-reinicios"):
        command = commands.add_parser(name)
        command.add_argument("--declaration", type=Path, required=True)
        command.add_argument("--state", action="append", required=True, help="NOMBRE=RUTA")
        command.add_argument("--output", type=Path, required=True)
        if name in ("gemelos", "reinicios"):
            command.add_argument("--device", default="cuda:0", choices=("cpu", "cuda:0"))
        if name in ("reinicios", "analizar-reinicios"):
            command.add_argument("--pair", required=True, help="modelo online del par")
        if name == "reinicios":
            command.add_argument("--arm", action="append", required=True, choices=sorted(RESETS))
    args = parser.parse_args(argv)
    declaration = json.loads(args.declaration.read_text(encoding="utf-8"))
    states = _states(args.state)
    if args.command == "gemelos":
        run_twins(declaration, states, args.output, device=args.device)
    elif args.command == "reinicios":
        run_resets(declaration, states, args.output, args.pair, args.arm, device=args.device)
    elif args.command == "analizar-reinicios":
        report = analyze_resets(declaration, states, args.output, args.pair)
        print(json.dumps(report["contrasts"], ensure_ascii=False, indent=1)[:4000])
    else:
        report = analyze(declaration, states, args.output)
        print(json.dumps(report["pairs"], ensure_ascii=False, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
