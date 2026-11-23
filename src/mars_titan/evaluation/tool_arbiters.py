"""Ideas 4 y 7 de Gemini para el núcleo agéntico: árbitros de herramientas frente a S2.

Encargo del coordinador del 23 de noviembre de 2026 (programa §5.15), con el protocolo fijado antes
de medir y la configuración en `configs/experiments/posttraining-tool-arbiters-*.json`, cuya huella
exige la orden. Es exploratorio y va fuera de la familia primaria.

- **Herramientas:** emisiones de modelos de la campaña, del árbol y de los órganos, alineadas fila a
  fila con el padre de S2. Cada sesión las tipifica por rangos, como el agregador en línea
  (`online_aggregator`), y la pérdida de una herramienta en una sesión madura es (1 − IC)/2.
- **agent:** AdaNormalHedge.TV con las pérdidas maduras (`online_arbiters.mature_mixture`), sin
  abstención. **agent_uniform:** la media uniforme de las herramientas tipificadas.
- **DODEA (idea 4):** agent con dos añadidos causales.
  - Optimismo con retraso (Flaspohler y otros, 2021). Cada sesión aún sin madurar al decidir recibe
    como pista h la media exponencial de las pérdidas maduras de cada herramienta. Con D sesiones
    pendientes, cada copia suma D·(h̄ − h_k) a su arrepentimiento y D·|h̄ − h_k| a su magnitud,
    con h̄ = p·h la pista de la mezcla base. Corrección declarada: el texto suma una sola pista;
    una por sesión pendiente es la forma del artículo que cita y coincide con la suya si D = 1.
  - Descuento de redundancia contemporánea. Con Σ_t la covarianza entre herramientas de sus
    predicciones tipificadas en la sesión (sin etiquetas), la mezcla emitida es
    Proj_Δ((Σ_t + μI)⁻¹ p_t), la proyección euclídea al símplex.
  - La semivida de la pista y μ se eligen en calibración, en la rejilla declarada.
- **Placebo de DODEA, de igual capacidad:** el mismo cálculo y la misma elección, pero la pista y la
  redundancia de cada herramienta son las de otra: una permutación sin puntos fijos, distinta en
  cada sesión y fijada por semilla. Conserva el tamaño de los ajustes y rompe su relación con cada
  herramienta. Si DODEA no le gana, lo que aporta es un regularizador más, no su mecanismo.
- **S2:** la configuración registrada (ridge_cov_active sobre titans_mac_nogain, el árbol y la señal
  de su brazo combinado) en cada semilla, sobre las mismas filas.

Cada variante se mide en la etiqueta y en lo operable (k·p + α frente a y + α, con k de cada
herramienta elegido en calibración). La serie de una variante es, en cada sesión, la media de sus
IC en las semillas; los contrastes son pareados por sesión en evaluación, con la t de Newey-West de
la criba. Nada se elige en evaluación. 2013 decide y 2024 sigue sellado.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.evaluation import agent_core as ac
from mars_titan.evaluation import holographic as holo
from mars_titan.evaluation import online_aggregator as oa
from mars_titan.evaluation import online_arbiters as arbiters
from mars_titan.evaluation import organ_aggregation as og
from mars_titan.evaluation import posttraining_pilots as pilots

KIND = "posttraining_tool_arbiters_v1"
# Huellas canónicas de las declaraciones registradas (programa §5.15).
REGISTERED = {
    "52d2946de090298d9086b6f95317626ab5c65228f71f0b9f545ac6b982259763": (
        "posttraining-tool-arbiters-hoy.json"
    ),
    "4b051af3c88acf5f0a4fda4ef85d237080d9ee66491b98c9099e26fb10f8af1b": (
        "posttraining-tool-arbiters-wp5.json"
    ),
}
SPLITS, AGGREGATED = og.SPLITS, og.AGGREGATED
SOURCES = ("state", "reference", "organs")
MAX_TOOLS = 16
_FIELDS = {
    "schema_version",
    "kind",
    "final_test_access",
    "description",
    "seeds",
    "tree_seed",
    "min_assets",
    "s2",
    "tools",
    "splits",
    "selection_split",
    "align",
    "modes",
    "operable_scales",
    "aggregator_placebo_seed",
    "arbiters",
    "rule",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_declaration(document):
    """Una declaración registrada y su huella, o un error que dice por qué no lo es."""
    _require(isinstance(document, dict), "La declaración de los árbitros es un objeto JSON")
    digest = hashlib.sha256(pilots._canonical(document).encode()).hexdigest()
    _require(
        digest in REGISTERED,
        f"La declaración no es una de las registradas para los árbitros (huella {digest[:12]}): "
        "la regla no cambia",
    )
    _require(set(document) == _FIELDS, f"La declaración tiene exactamente {sorted(_FIELDS)}")
    _require(
        document["kind"] == KIND and document["final_test_access"] is False,
        f"La declaración es {KIND} y no abre el test final",
    )
    tools = document["tools"]
    names = [tool["name"] for tool in tools]
    _require(
        2 <= len(tools) <= MAX_TOOLS
        and len(set(names)) == len(names)
        and all(set(tool) == {"name", "source"} and tool["source"] in SOURCES for tool in tools),
        f"Se declaran de 2 a {MAX_TOOLS} herramientas distintas con su fuente ({SOURCES})",
    )
    _require(
        all(tool["source"] != "organs" or "__" in tool["name"] for tool in tools),
        "Una herramienta de los órganos se llama <padre>__<brazo>",
    )
    dodea = document["arbiters"]["dodea"]
    _require(
        set(dodea) == {"hint_half_lives", "ridges", "placebo_seed"}
        and all(h > 0 for h in dodea["hint_half_lives"])
        and all(m > 0 for m in dodea["ridges"]),
        "DODEA declara semividas de la pista y μ positivos y la semilla del placebo",
    )
    return document, digest


# ---------------------------------------------------------------------------------------------
# Piezas puras


def market_seed(market):
    return int.from_bytes(hashlib.sha256(str(market).encode("utf-8")).digest()[:8], "big")


def derangement(size, rng):
    """Permutación sin puntos fijos, uniforme entre ellas (por rechazo); size ≥ 2."""
    _require(size >= 2, "Una permutación sin puntos fijos necesita al menos dos herramientas")
    identity = np.arange(size)
    while True:
        order = rng.permutation(size)
        if not np.any(order == identity):
            return order


def optimistic(state, hint, pending, base):
    """Mezcla de AdaNormalHedge.TV con una pista por sesión pendiente.

    Cada copia (τ, k) suma D·(h̄ − h_k) a su arrepentimiento y D·|h̄ − h_k| a su magnitud, con
    h̄ = base·h. Si ninguna copia tiene peso, se queda la mezcla base.
    """
    gap = float(base @ hint) - hint
    regret, magnitude = state._copies()
    regret = regret + pending * gap
    magnitude = magnitude + pending * np.abs(gap)
    mass = state.prior * (state._start_weights() * arbiters.weights(regret, magnitude)).sum(axis=0)
    total = float(mass.sum())
    return base if total <= 0.0 or not math.isfinite(total) else mass / total


def decorrelated(mixture, covariance, ridge):
    """Proj_Δ((Σ + μI)⁻¹ p): el descuento de redundancia de DODEA."""
    size = mixture.size
    return oa.project_simplex(np.linalg.solve(covariance + ridge * np.eye(size), mixture))


def dodea_weights(stream, *, half_life, ridge, placebo_seed=None, market="US"):
    """Pesos de DODEA en cada sesión del flujo de un mercado (o los de su placebo).

    Repite `online_arbiters.mature_mixture` con AdaNormalHedge.TV: antes de decidir la sesión t se
    aplican, en orden de madurez y de decisión, las sesiones con madurez ≤ t, con su pérdida si
    puntúan. La pista es la media exponencial (semivida en sesiones maduras) de esas pérdidas; las
    pendientes son las sesiones ya decididas cuya madurez es posterior a t.
    """
    times = np.asarray(stream["times"], dtype=np.int64)
    matures = stream["matures"]
    _require(matures is not None, "DODEA necesita la madurez exacta de cada sesión")
    matures = np.asarray(matures, dtype=np.int64)
    losses, scored, covs = stream["losses"], stream["scored"], stream["covs"]
    sessions, size = losses.shape
    _require(
        sessions >= 1 and bool(np.all(matures > times)) and np.unique(times).size == sessions,
        "Las sesiones tienen instantes distintos y maduran después de decidir",
    )
    state = arbiters.AdaNormalHedgeTV.start(np.ones(size))
    awake = np.ones(size, dtype=bool)
    decay = 0.5 ** (1.0 / half_life)
    total, mass = np.zeros(size), 0.0
    order, mature = np.argsort(times), np.lexsort((times, matures))
    result = np.empty((sessions, size))
    applied = 0
    for session in order:
        now = times[session]
        while applied < sessions and matures[mature[applied]] <= now:
            known = mature[applied]
            if scored[known]:
                state.update(losses[known], awake)
                total, mass = decay * total + losses[known], decay * mass + 1.0
            applied += 1
        base = state.distribution(awake)
        hint, covariance = (total / mass if mass > 0 else np.zeros(size)), covs[session]
        if placebo_seed is not None:
            rng = np.random.default_rng(
                np.random.SeedSequence([placebo_seed, market_seed(market), int(now)])
            )
            swap = derangement(size, rng)
            hint, covariance = hint[swap], covariance[np.ix_(swap, swap)]
        pending = int(np.count_nonzero((times < now) & (matures > now)))
        mixture = optimistic(state, hint, pending, base) if mass > 0 and pending else base
        result[session] = decorrelated(mixture, covariance, ridge)
    return result


def agent_weights(stream):
    """AdaNormalHedge.TV con las pérdidas maduras, sin abstención: el agente."""
    losses = stream["losses"]
    awake = np.ones(losses.shape, dtype=bool)
    scored = np.repeat(np.asarray(stream["scored"], dtype=bool)[:, None], losses.shape[1], axis=1)
    mixtures, _ = arbiters.mature_mixture(
        np.asarray(stream["times"], dtype=np.int64),
        np.asarray(stream["matures"], dtype=np.int64),
        losses,
        awake,
        np.ones(losses.shape[1]),
        scored=scored,
        hedge=arbiters.AdaNormalHedgeTV,
    )
    return mixtures


def uniform_weights(stream):
    return oa.rank_average_weights(len(stream["times"]), stream["losses"].shape[1])


def split_ics(aligned, streams, weights, split, min_assets):
    """IC por sesión de la mezcla con esos pesos en un tramo."""
    prediction, _, abstained, _ = oa._emit_split_predictions_from_weights(
        aligned, streams, weights, split
    )
    return oa.session_rank_ic(
        aligned[split], prediction, min_assets=min_assets, abstained=abstained
    )


def _mean(values):
    return float(np.mean(list(values.values()))) if values else float("-inf")


def chosen_dodea(aligned, streams, grid, *, min_assets, placebo_seed=None):
    """DODEA (o su placebo) con la semivida y μ de más IC medio en calibración.

    Empates: el primero de la rejilla declarada. Devuelve la elección, la puntuación de cada punto
    y los IC por sesión de calibración y evaluación de la elegida.
    """
    scores, best = {}, None
    for half_life, ridge in grid:
        weights = {
            market: dodea_weights(
                stream,
                half_life=half_life,
                ridge=ridge,
                placebo_seed=placebo_seed,
                market=market,
            )
            for market, stream in streams.items()
        }
        score = _mean(split_ics(aligned, streams, weights, "calibration", min_assets))
        scores[f"{half_life:g}/{ridge:g}"] = score
        if best is None or score > best[0]:
            best = (score, half_life, ridge, weights)
    _, half_life, ridge, weights = best
    return dict(
        choice=dict(half_life=half_life, ridge=ridge, calibration_ic=best[0]),
        scores=scores,
        ics={s: split_ics(aligned, streams, weights, s, min_assets) for s in AGGREGATED},
    )


# ---------------------------------------------------------------------------------------------
# Entradas


def _parent_of(name):
    return name.rsplit("__", 1)[0]


def roles(declaration):
    """Papel → (modelo, fuente, semilla): herramientas, miembros de S2 y padres de órganos."""
    found = {}
    s2 = declaration["s2"]
    for seed in declaration["seeds"]:
        own = [(tool["name"], tool["source"]) for tool in declaration["tools"]]
        own += [(s2["parent"], "state"), (s2["tree"], "reference"), (s2["organ"], "organs")]
        for tool in declaration["tools"]:
            found[f"tool/{tool['name']}/{seed}"] = (tool["name"], tool["source"], seed)
        for role, source in (("parent", "state"), ("tree", "reference"), ("organ", "organs")):
            found[f"s2/{role}/{seed}"] = (s2[role], source, seed)
        for name, source in own:
            if source == "organs":
                found[f"parent/{_parent_of(name)}/{seed}"] = (_parent_of(name), "state", seed)
    return found


def folders(declaration, state, reference, organs, origin):
    """La carpeta de cada papel."""
    result = {}
    for role, (name, source, seed) in roles(declaration).items():
        if source == "state":
            base, run_seed = Path(state), seed
        elif source == "reference":
            base, run_seed = Path(reference), declaration["tree_seed"]
        else:
            base, run_seed = Path(organs[seed]), seed
        result[role] = base / "runs" / name / str(origin) / f"seed-{run_seed}"
    return result


def _needed(declaration, role):
    source = roles(declaration)[role][1]
    return AGGREGATED if source == "organs" else SPLITS


def missing_inputs(declaration, sources, manifest):
    """Lo que falta todavía; una lista vacía cuando está todo."""
    missing = [] if manifest.is_file() else [f"el manifiesto de la cohorte {manifest}"]
    for role, folder in sources.items():
        record = folder / "run.json"
        try:
            status = json.loads(record.read_text(encoding="utf-8")).get("status")
        except (OSError, ValueError):
            status = None
        if status != "completed":
            missing.append(f"{role}: {record} (estado {status!r})")
            continue
        missing += [
            f"{role}: {folder / f'{split}-predictions.parquet'}"
            for split in _needed(declaration, role)
            if not og._readable(folder / f"{split}-predictions.parquet")
        ]
    return missing


def fingerprint(declaration, sources, manifest):
    files = {"manifest": manifest}
    for role, folder in sources.items():
        files[f"{role}/run.json"] = folder / "run.json"
        for split in _needed(declaration, role):
            files[f"{role}/{split}"] = folder / f"{split}-predictions.parquet"
    return {name: dict(path=str(path), sha256=pilots._sha256(path)) for name, path in files.items()}


def checked_records(declaration, origin, records, prints):
    """Registros terminados, sin test final, una vista y cada órgano emitido de su padre."""
    views = set()
    for role, record in records.items():
        _require(record.get("final_test_opened") is False, f"{role} abrió el test final")
        name, source, seed = roles(declaration)[role]
        if source == "state":
            views.add((record.get("request") or {}).get("view_sha256"))
        if source != "organs":
            continue
        parent = records[f"parent/{_parent_of(name)}/{seed}"]
        link = record.get("parent") or {}
        _require(
            record.get("model") == name
            and record.get("arm") == name.rsplit("__", 1)[1]
            and record.get("origin") == origin
            and record.get("seed") == seed
            and link.get("model") == _parent_of(name)
            and link.get("run_id") == parent.get("run_id"),
            f"{role} no es el brazo {name} del ajuste declarado de su padre (semilla {seed})",
        )
        emissions = (record.get("inputs") or {}).get("emissions") or {}
        for split in SPLITS:
            _require(
                emissions.get(split)
                == prints[f"parent/{_parent_of(name)}/{seed}/{split}"]["sha256"],
                f"{role} se emitió con otras predicciones de su padre ({split})",
            )
        for split in AGGREGATED:
            _require(
                (record.get("outputs") or {}).get(split) == prints[f"{role}/{split}"]["sha256"],
                f"La emisión de {split} de {role} no es la de su registro",
            )
    _require(
        len(views) == 1 and None not in views,
        f"Los modelos de la campaña no comparten una vista ({sorted(map(str, views))})",
    )
    return views.pop()


# ---------------------------------------------------------------------------------------------
# Medida


def seed_mean(per_seed):
    """En cada sesión común a todas las semillas, la media de sus IC."""
    common = set.intersection(*(set(values) for values in per_seed))
    return {key: float(np.mean([values[key] for values in per_seed])) for key in sorted(common)}


CONTRASTS = (
    ("dodea_vs_s2", "dodea", "s2"),
    ("dodea_vs_placebo", "dodea", "dodea_placebo"),
    ("dodea_vs_agent", "dodea", "agent"),
    ("dodea_vs_uniform", "dodea", "agent_uniform"),
    ("placebo_vs_agent", "dodea_placebo", "agent"),
    ("agent_vs_s2", "agent", "s2"),
    ("uniform_vs_s2", "agent_uniform", "s2"),
    ("tools_ridge_vs_s2", "tools_ridge", "s2"),
)


def _wins(entry, minimum, min_t):
    mean, t = entry.get("mean"), entry.get("t_nw")
    return mean is not None and t is not None and mean > minimum and t >= min_t


def verdict(contrasts, rule):
    """La regla declarada, leída mecánicamente sobre los contrastes de evaluación."""
    main = contrasts[rule["primary_mode"]]
    min_t = rule["min_t"]
    reasons = []
    if not _wins(main["dodea_vs_s2"], 0.0, min_t):
        reasons.append(f"no mejora a S2 con t ≥ {min_t} en {rule['primary_mode']}")
    if not _wins(main["dodea_vs_placebo"], 0.0, min_t):
        reasons.append(f"no mejora a su placebo con t ≥ {min_t} en {rule['primary_mode']}")
    author = rule["author"]
    other = contrasts[author["mode"]]
    survives = bool(
        other["dodea_vs_agent"]["mean"] is not None
        and other["dodea_vs_agent"]["mean"] >= author["vs_agent"]
        and (other["dodea_vs_agent"]["t_nw"] or 0.0) >= author["min_t"]
        and other["dodea_vs_uniform"]["mean"] is not None
        and other["dodea_vs_uniform"]["mean"] >= author["vs_uniform"]
        and (other["dodea_vs_uniform"]["t_nw"] or 0.0) >= author["min_t"]
    )
    return dict(
        enters=not reasons,
        reasons=reasons,
        author_refuted=not survives,
        reading=("DODEA entra" if not reasons else "DODEA no entra")
        + (
            "; la predicción de su autor se cumple" if survives else "; su autor la da por refutada"
        ),
    )


def _seed_variants(declaration, aggregate, inputs, seed, mode, min_assets):
    """Las variantes de una semilla y un modo: IC por sesión de evaluación y lo elegido."""
    s2 = declaration["s2"]
    s2_run = aggregate(
        {
            s2["parent"]: inputs[("s2/parent", seed)],
            s2["tree"]: inputs[("s2/tree", seed)],
            og.SIGNAL: inputs[("s2/signal", seed)],
        },
        mode,
    )
    names = [tool["name"] for tool in declaration["tools"]]
    tools_run = aggregate({name: inputs[("tool/" + name, seed)] for name in names}, mode)
    aligned = tools_run["aligned"]
    for split in AGGREGATED:
        _require(
            list(aligned[split]["sample_id"]) == list(s2_run["aligned"][split]["sample_id"]),
            f"Las filas de las herramientas y de S2 no coinciden en {mode}/{split} ({seed})",
        )
    streams = oa._build_market_streams(aligned, tuple(declaration["splits"]), min_assets=min_assets)

    def ridge_ics(run):
        prediction = run["aggregators"]["ridge_cov_active"]["split_predictions"]["evaluation"]
        return oa.session_rank_ic(run["aligned"]["evaluation"], prediction, min_assets=min_assets)

    def weighted(function):
        weights = {market: function(stream) for market, stream in streams.items()}
        return split_ics(aligned, streams, weights, "evaluation", min_assets)

    dodea = declaration["arbiters"]["dodea"]
    grid = [(h, m) for h in dodea["hint_half_lives"] for m in dodea["ridges"]]
    real = chosen_dodea(aligned, streams, grid, min_assets=min_assets)
    fake = chosen_dodea(
        aligned, streams, grid, min_assets=min_assets, placebo_seed=dodea["placebo_seed"]
    )
    series = dict(
        s2=ridge_ics(s2_run),
        agent=weighted(agent_weights),
        agent_uniform=weighted(uniform_weights),
        dodea=real["ics"]["evaluation"],
        dodea_placebo=fake["ics"]["evaluation"],
        tools_ridge=ridge_ics(tools_run),
    )
    order = list(aligned["evaluation"]["model_names"])
    for name in names:
        series[f"tool:{name}"] = oa.session_rank_ic(
            aligned["evaluation"],
            aligned["evaluation"]["predictions"][:, order.index(name)],
            min_assets=min_assets,
        )
    chosen = dict(
        dodea=dict(real["choice"], scores=real["scores"]),
        dodea_placebo=dict(fake["choice"], scores=fake["scores"]),
        operable_scales=tools_run["operable"].get("scales"),
    )
    return series, chosen


def measure(
    declaration, state, reference, organs, origin, output, *, wait_hours, poll, log, markets=None
):
    """Los árbitros de un origen, publicados en `<output>/<origen>`."""
    declaration, digest = checked_declaration(declaration)
    seeds = list(declaration["seeds"])
    _require(
        sorted(organs) == sorted(seeds),
        f"Hace falta una raíz de órganos por semilla declarada ({seeds}); hay {sorted(organs)}",
    )
    sources = folders(declaration, state, reference, organs, origin)
    manifest = Path(state) / "cohorts" / str(origin) / "manifest.json"
    deadline = time.monotonic() + wait_hours * 3600
    while True:
        missing = missing_inputs(declaration, sources, manifest)
        if not missing or time.monotonic() >= deadline:
            break
        log(f"esperando {len(missing)} entradas; la primera: {missing[0]}")
        time.sleep(poll)
    if missing:
        raise og.MissingInputs(missing)
    prints = fingerprint(declaration, sources, manifest)
    target = Path(output) / og.target_name(origin, markets)
    if target.exists():
        return og._already(target, digest, prints, log)
    started = time.monotonic()

    def say(text):
        log(f"{time.monotonic() - started:6.0f} s {text}")

    records = {
        role: json.loads((folder / "run.json").read_text(encoding="utf-8"))
        for role, folder in sources.items()
    }
    view = checked_records(declaration, origin, records, prints)
    cache = {}
    tables = {}
    for role, folder in sources.items():
        if folder not in cache:
            cache[folder] = {s: og._table(folder, s) for s in AGGREGATED}
        tables[role] = cache[folder]
    reference_role = f"s2/parent/{seeds[0]}"
    present = og.markets_of(tables[reference_role])
    measured = og.measured_markets(present, markets)
    if measured != present:
        tables = {
            role: {s: og.only(t, measured) for s, t in splits.items()}
            for role, splits in tables.items()
        }
    say(f"mercados medidos: {measured} (el estado tiene {present})")
    for role, splits in tables.items():
        for split, table in splits.items():
            for name in og.KEYS:
                _require(
                    table.column(name).equals(tables[reference_role][split].column(name)),
                    f"{sources[role]} no tiene las filas de {sources[reference_role]} en {split} "
                    f"({name})",
                )
    alpha, _, _, matures = ac.known_rows(state, origin, tables[reference_role], view)
    say("alfa conocido y madurez")

    def prediction(role, split):
        return tables[role][split].column("prediction").to_numpy().astype(np.float64)

    min_assets = declaration["min_assets"]
    Path(output).mkdir(parents=True, exist_ok=True)
    work = Path(output) / f".{origin}.{os.getpid()}.tmp"
    try:
        work.mkdir(parents=True)
        inputs = {}
        for seed in seeds:
            for tool in declaration["tools"]:
                role = f"tool/{tool['name']}/{seed}"
                inputs[("tool/" + tool["name"], seed)] = og._write_component(
                    work / "entradas" / f"{seed}" / tool["name"],
                    tables[reference_role],
                    {s: prediction(role, s) for s in AGGREGATED},
                    alpha,
                    matures,
                )
            for role in ("parent", "tree"):
                inputs[(f"s2/{role}", seed)] = og._write_component(
                    work / "entradas" / f"{seed}" / f"s2-{role}",
                    tables[reference_role],
                    {s: prediction(f"s2/{role}/{seed}", s) for s in AGGREGATED},
                    alpha,
                    matures,
                )
            inputs[("s2/signal", seed)] = og._write_component(
                work / "entradas" / f"{seed}" / "s2-signal",
                tables[reference_role],
                {
                    s: prediction(f"s2/organ/{seed}", s) - prediction(f"s2/parent/{seed}", s)
                    for s in AGGREGATED
                },
                alpha,
                matures,
            )

        def aggregate(members, mode):
            return oa.run_aggregation(
                members,
                splits=tuple(declaration["splits"]),
                selection_split=declaration["selection_split"],
                min_assets=min_assets,
                align=declaration["align"],
                operable=mode == "operable",
                alpha_col="alpha",
                maturity_col="matures_at",
                placebo_seed=declaration["aggregator_placebo_seed"],
                operable_scales=tuple(declaration["operable_scales"]),
            )

        tool = og._screening()
        levels, contrasts, chosen, rows, correlations = {}, {}, {}, [], {}
        for mode in declaration["modes"]:
            per_seed, chosen[mode] = {}, {}
            for seed in seeds:
                per_seed[seed], chosen[mode][seed] = _seed_variants(
                    declaration, aggregate, inputs, seed, mode, min_assets
                )
                say(f"{mode}: semilla {seed} medida")
            variants = list(per_seed[seeds[0]])
            mean = {v: seed_mean([per_seed[s][v] for s in seeds]) for v in variants}
            levels[mode] = {
                v: dict(
                    seeds={str(s): tool.summary(per_seed[s][v]) for s in seeds},
                    mean=tool.summary(mean[v]),
                )
                for v in variants
            }
            contrasts[mode] = {
                name: holo.paired(tool, mean[left], mean[right]) for name, left, right in CONTRASTS
            }
            correlations[mode] = {
                "dodea_s2": holo.ic_correlation(mean["dodea"], mean["s2"]),
                "agent_s2": holo.ic_correlation(mean["agent"], mean["s2"]),
            }
            for v in variants:
                for (market, instant), value in mean[v].items():
                    rows.append((mode, v, "mean", str(market), int(instant), float(value)))
                for s in seeds:
                    for (market, instant), value in per_seed[s][v].items():
                        rows.append((mode, v, str(s), str(market), int(instant), float(value)))
        decision = verdict(contrasts, declaration["rule"])
        summary = og._plain(
            dict(
                schema_version=1,
                kind=f"{KIND}_summary",
                status="completed",
                origin=origin,
                declaration_sha256=digest,
                declaration=REGISTERED[digest],
                inputs=prints,
                markets=measured,
                seeds=seeds,
                tools=[t["name"] for t in declaration["tools"]],
                levels=levels,
                contrasts=contrasts,
                correlations=correlations,
                chosen=chosen,
                verdict=decision,
                seconds=round(time.monotonic() - started),
            )
        )
        mode_c, series_c, seed_c, market_c, instant_c, value_c = zip(*rows, strict=True)
        pq.write_table(
            pa.table(
                dict(
                    mode=pa.array(mode_c, type=pa.string()),
                    series=pa.array(series_c, type=pa.string()),
                    seed=pa.array(seed_c, type=pa.string()),
                    market=pa.array(market_c, type=pa.string()),
                    instant=pa.array(instant_c, type=pa.int64()),
                    value=pa.array(value_c, type=pa.float64()),
                )
            ),
            work / "sesiones.parquet",
        )
        shutil.rmtree(work / "entradas")
        (work / "resumen.md").write_text(markdown(summary), encoding="utf-8")
        (work / "resumen.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        os.rename(work, target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    say(f"publicado en {target}")
    return summary


def _f(value, digits=4):
    return "—" if value is None else f"{value:+.{digits}f}".replace("-", "−").replace(".", ",")


def _c(entry):
    interval = entry.get("interval")
    text = f"{_f(entry.get('mean'))} (t {_f(entry.get('t_nw'), 2)})"
    return text if not interval else f"{text} [{_f(interval[0])}; {_f(interval[1])}]"


def markdown(summary):
    lines = [
        f"# Árbitros de herramientas (ideas 4 y 7) en el origen {summary['origin']}",
        "",
        f"Declaración {summary['declaration']} ({summary['declaration_sha256'][:12]}); mercados "
        f"{', '.join(summary['markets'])}; semillas {summary['seeds']}; herramientas "
        f"{', '.join(summary['tools'])}. Evaluación, media de semillas por sesión, diferencias "
        "pareadas con t de Newey-West.",
        "",
        f"**Lectura de la regla: {summary['verdict']['reading']}.** "
        + "; ".join(summary["verdict"]["reasons"]),
        "",
        "## IC medio por sesión en evaluación",
        "",
        "| Serie | " + " | ".join(summary["levels"]) + " |",
        "|---|" + "---|" * len(summary["levels"]),
    ]
    first = next(iter(summary["levels"].values()))
    for variant in first:
        cells = [_f(summary["levels"][m][variant]["mean"]["mean"]) for m in summary["levels"]]
        lines.append(f"| {variant} | " + " | ".join(cells) + " |")
    lines += ["", "## Contrastes", "", "| Contraste | " + " | ".join(summary["contrasts"]) + " |"]
    lines.append("|---|" + "---|" * len(summary["contrasts"]))
    for name, _, _ in CONTRASTS:
        cells = [_c(summary["contrasts"][m][name]) for m in summary["contrasts"]]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## Elección en calibración (semivida de la pista / μ)", ""]
    for mode, by_seed in summary["chosen"].items():
        for seed, chosen in by_seed.items():
            real, fake = chosen["dodea"], chosen["dodea_placebo"]
            lines.append(
                f"- {mode}, semilla {seed}: DODEA {real['half_life']:g}/{real['ridge']:g}; "
                f"placebo {fake['half_life']:g}/{fake['ridge']:g}"
            )
    return "\n".join(lines) + "\n"


def parser():
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = result.add_subparsers(dest="command", required=True)
    run = commands.add_parser("medir", help="Medir los árbitros de un origen")
    run.add_argument("--declaration", type=Path, required=True)
    run.add_argument("--state", type=Path, required=True, help="Estado de la campaña")
    run.add_argument("--reference", type=Path, required=True, help="Raíz de los árboles")
    run.add_argument(
        "--organs",
        type=holo._organ_root,
        action="append",
        required=True,
        metavar="SEMILLA=RAÍZ",
        help="Raíz de los órganos de una semilla (una por semilla declarada)",
    )
    run.add_argument("--origin", type=int, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--wait-hours", type=float, default=0.0)
    run.add_argument("--poll-seconds", type=float, default=300.0)
    run.add_argument("--market", action="append", dest="markets", metavar="MERCADO")
    return result


def main(argv=None):
    args = parser().parse_args(argv)

    def log(line):
        print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {line}", flush=True)

    try:
        organs = dict(args.organs)
        _require(len(organs) == len(args.organs), "Una semilla aparece dos veces en --organs")
        declaration = json.loads(args.declaration.read_text(encoding="utf-8"))
        measure(
            declaration,
            args.state,
            args.reference,
            organs,
            args.origin,
            args.output,
            wait_hours=args.wait_hours,
            poll=args.poll_seconds,
            log=log,
            markets=args.markets,
        )
    except og.MissingInputs as missing:
        print(
            f"Faltan entradas de los árbitros para el origen {args.origin} tras esperar "
            f"{args.wait_hours:g} h; no se escribe nada:",
            file=sys.stderr,
        )
        for item in missing.items:
            print(f"- {item}", file=sys.stderr)
        return og.MISSING_EXIT
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
