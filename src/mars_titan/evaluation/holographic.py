# ruff: noqa: E501
"""F6, el cerebro holográfico: red estática con órganos episódicos frente al núcleo con memoria en línea.

Encargo del coordinador del 23 de noviembre de 2026 (F6), con la regla acordada antes de medir
(programa §5.11) y la configuración fijada en `configs/experiments/posttraining-holographic.json`,
cuya huella exige la orden. Es exploratorio y va fuera de la familia primaria.

- **H, el holográfico, por semilla:** ridge_cov_active de {padre estático, señal de su brazo
  combinado (órgano − padre)}, sin árboles y con la configuración de S2. H3 es la media de rangos
  por sesión de sus semillas.
- **C3, el núcleo:** la media de rangos por sesión de titans_mac_nogain en las mismas semillas, la
  pareja exacta del estático: solo cambia la memoria en línea. T3, con titans_mac, es la lectura
  secundaria, y E3 es el estático sin órganos.
- **Conjuntos con pesos 1/2 fijos:** C3⊕H3 = (C3 + H3)/2 y el control a igual número de redes
  C3⊕E3 = (C3 + E3)/2; lo mismo con T3.
- **Regla:** H entra como segundo cerebro si IC(C3⊕H3) − IC(C3) ≥ 0,0020 con t ≥ 2 en la
  etiqueta y es > 0 en lo operable. Si C3⊕H3 no supera a C3⊕E3 con t ≥ 2, la ganancia viene de la
  segunda red y no de los órganos, y se dice así.
- **«Se equivoca en sitios distintos»:** la correlación de los IC por sesión de C3 y H3 y la
  correlación de rangos media entre sus predicciones dentro de cada sesión, sin umbral.
- **Variantes de WP5 (programa §5.14):** la declaración de F6 con el órgano
  `<estático>__dense_bank`, `__holographic_delta` o `__asset_trace` y otra descripción. Cada
  una tiene su huella registrada, la orden exige que el registro del órgano sea el de ese brazo y
  la regla no cambia.

Los contrastes son diferencias pareadas por sesión en evaluación, con la t de Newey-West de la
criba. Cada semilla puede traer sus órganos de una raíz distinta si las raíces comparten
declaración salvo la descripción y la lista de modelos. Nada se elige en evaluación. 2013 decide;
los orígenes limpios confirman con la misma regla.
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
import pyarrow.parquet as pq
from scipy.stats import rankdata

from mars_titan.evaluation import agent_core as ac
from mars_titan.evaluation import online_aggregator as oa
from mars_titan.evaluation import organ_aggregation as og
from mars_titan.evaluation import posttraining_pilots as pilots

KIND = "posttraining_holographic_v1"
# Huella canónica de configs/experiments/posttraining-holographic.json, la declaración de F6.
REGISTERED_SHA256 = "0316226e16174587f50dc993a447c63cb6f0ca2b888c000ee131e0b216ae2c63"
# Huellas de configs/experiments/posttraining-holographic-wp5-<brazo>.json: F6 con un órgano de WP5.
WP5_SHA256 = {
    "dense_bank": "9ea3c63c973337adca90a9064a9813985a816b2ec842d3cd9e40a5acc9d71b49",
    "holographic_delta": "8272fe12ee4206fc848fad0ffd8e6e5c5a93fb506d754c3774205239775bb446",
    "asset_trace": "1053e40ec9efc85db4de705879f570db65ca8abc1c74c4fb2a523e8f6e388c7a",
}
# El brazo del órgano de cada declaración registrada.
REGISTERED = {REGISTERED_SHA256: "combined", **{v: k for k, v in WP5_SHA256.items()}}
SPLITS, AGGREGATED = og.SPLITS, og.AGGREGATED
BRAINS = ("C", "H", "E", "T")
Z95 = 1.959963984540054


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_declaration(document):
    """La declaración registrada de F6 y su huella, o un error que dice por qué no lo es."""
    _require(isinstance(document, dict), "La declaración de F6 es un objeto JSON")
    digest = hashlib.sha256(pilots._canonical(document).encode()).hexdigest()
    _require(
        digest in REGISTERED,
        f"La declaración no es la registrada para F6 (huella {digest[:12]}, registrada "
        f"{REGISTERED_SHA256[:12]}) ni una de sus variantes de WP5: la regla no cambia",
    )
    _require(
        document["organ"] == f"{document['static']}__{REGISTERED[digest]}",
        f"El órgano declarado {document['organ']} no es el brazo {REGISTERED[digest]} del estático",
    )
    grid = document["aggregator"]
    _require(
        tuple(grid["lambdas"]) == tuple(oa.RIDGE_LAMBDAS)
        and tuple(grid["half_lives"]) == tuple(oa.RIDGE_HALF_LIVES),
        "El agregador del código no usa la rejilla de ridge_cov_active declarada",
    )
    return document, digest


# ---------------------------------------------------------------------------------------------
# Piezas puras


def session_ranks(values, bounds):
    """Rango medio de cada fila dentro de su sesión, dividido por el tamaño de la sesión."""
    result = np.empty(len(values), dtype=np.float64)
    for start, end in bounds:
        result[start:end] = rankdata(values[start:end]) / (end - start)
    return result


def session_ics(tool, score, target, bounds, keys, min_assets):
    """IC de rangos de cada sesión puntuable, por (mercado, instante)."""
    result = {}
    for (start, end), key in zip(bounds, keys, strict=True):
        if end - start < min_assets:
            continue
        value = tool.correlation(
            tool.average_ranks(score[start:end]), tool.average_ranks(target[start:end])
        )
        if not math.isnan(value):
            result[key] = value
    return result


def paired(tool, left, right):
    """left − right por sesión en las que puntúan las dos: media, t de Newey-West e intervalo."""
    common = sorted(set(left) & set(right))
    if not common:
        return dict(sessions=0, mean=None, t_nw=None, interval=None)
    stats = tool.summary({k: left[k] - right[k] for k in common})
    t = stats["t_nw"]
    se = abs(stats["mean"] / t) if t not in (None, 0) and math.isfinite(t) else None
    return dict(
        sessions=len(common),
        mean=stats["mean"],
        t_nw=t,
        interval=None if se is None else [stats["mean"] - Z95 * se, stats["mean"] + Z95 * se],
    )


def ic_correlation(left, right):
    """Correlación entre sesiones de dos series de IC, en las sesiones comunes."""
    common = sorted(set(left) & set(right))
    if len(common) < 3:
        return None
    return float(np.corrcoef([left[k] for k in common], [right[k] for k in common])[0, 1])


def within_correlation(tool, left, right, bounds, min_assets):
    """Correlación de rangos media, dentro de cada sesión, entre dos predicciones."""
    values = []
    for start, end in bounds:
        if end - start < min_assets:
            continue
        value = tool.correlation(
            tool.average_ranks(left[start:end]), tool.average_ranks(right[start:end])
        )
        if not math.isnan(value):
            values.append(value)
    return float(np.mean(values)) if values else None


def verdict(contrasts, rule):
    """La regla acordada, leída mecánicamente sobre los contrastes de evaluación."""
    primary = contrasts["label"]["c3h3_vs_c3"]
    operable = contrasts["operable"]["c3h3_vs_c3"]
    control = contrasts["label"]["c3h3_vs_c3e3"]
    reasons = []
    if (
        primary["mean"] is None
        or primary["mean"] < rule["min_gain"]
        or primary["t_nw"] is None
        or primary["t_nw"] < rule["min_t"]
    ):
        reasons.append(
            "en la etiqueta, IC(C3⊕H3) − IC(C3) no llega a "
            f"+{rule['min_gain']} con t ≥ {rule['min_t']}"
        )
    if operable["mean"] is None or not operable["mean"] > 0:
        reasons.append("en lo operable, IC(C3⊕H3) − IC(C3) no es positivo")
    organs = bool(
        control["mean"] is not None
        and control["mean"] > 0
        and control["t_nw"] is not None
        and control["t_nw"] >= rule["control_min_t"]
    )
    if reasons:
        reading = "H se queda como órgano del agregador"
    elif organs:
        reading = "H entra como segundo cerebro, y la ganancia es de los órganos"
    else:
        reading = "H entra, pero la ganancia viene de la segunda red y no de los órganos"
    return dict(enters=not reasons, reasons=reasons, organs_matter=organs, reading=reading)


def organ_declarations(roots, static, ignored):
    """Comprueba que las raíces de órganos comparten declaración salvo `ignored`; sus huellas."""
    documents = {}
    for root in sorted({str(Path(r)) for r in roots}):
        path = Path(root) / "declaration.json"
        _require(
            path.is_file(), f"{path} no existe: no se puede comprobar la declaración de órganos"
        )
        document = json.loads(path.read_text(encoding="utf-8"))
        _require(
            static in (document.get("models") or []), f"{root} no declara órganos para {static}"
        )
        documents[root] = document
    reference = None
    for document in documents.values():
        kept = {k: v for k, v in document.items() if k not in ignored}
        if reference is None:
            reference = kept
        _require(
            kept == reference,
            f"Las raíces de órganos {sorted(documents)} no comparten declaración fuera de "
            f"{list(ignored)}: no se mezclan sus semillas",
        )
    return {root: pilots._sha256(Path(root) / "declaration.json") for root in documents}


# ---------------------------------------------------------------------------------------------
# Entradas


def folders(declaration, state, organs, origin):
    """Carpetas de cada papel y semilla; los órganos de cada semilla, en su raíz."""
    sources = {}
    for seed in declaration["seeds"]:
        for role in ("core", "secondary", "static"):
            sources[f"{role}/{seed}"] = (
                Path(state) / "runs" / declaration[role] / str(origin) / f"seed-{seed}"
            )
        sources[f"organ/{seed}"] = (
            Path(organs[seed]) / "runs" / declaration["organ"] / str(origin) / f"seed-{seed}"
        )
    return sources


def _needed(role):
    return AGGREGATED if role.startswith("organ/") else SPLITS


def missing_inputs(sources, manifest):
    """Lo que falta todavía; una lista vacía cuando está todo."""
    missing = [] if manifest.is_file() else [f"el manifiesto de la cohorte {manifest}"]
    for role, folder in sources.items():
        record = folder / "run.json"
        if not record.is_file():
            missing.append(f"{role}: {record}")
            continue
        try:
            status = json.loads(record.read_text(encoding="utf-8")).get("status")
        except (OSError, ValueError):
            status = None
        if status != "completed":
            missing.append(f"{role}: {record} no ha terminado (estado {status!r})")
            continue
        missing += [
            f"{role}: {folder / f'{split}-predictions.parquet'}"
            for split in _needed(role)
            if not og._readable(folder / f"{split}-predictions.parquet")
        ]
    return missing


def fingerprint(sources, manifest):
    files = {"manifest": manifest}
    for role, folder in sources.items():
        files[f"{role}/run.json"] = folder / "run.json"
        for split in _needed(role):
            files[f"{role}/{split}"] = folder / f"{split}-predictions.parquet"
    return {name: dict(path=str(path), sha256=pilots._sha256(path)) for name, path in files.items()}


def organ_arm(declaration):
    """El brazo del órgano: `combined` en F6 y el órgano de WP5 en sus variantes."""
    return declaration["organ"].rsplit("__", 1)[-1]


def checked_records(declaration, origin, sources, records, prints):
    """Comprueba los registros y devuelve la vista común de los padres."""
    arm = organ_arm(declaration)
    label = "combinado" if arm == "combined" else arm
    views = set()
    for role, record in records.items():
        _require(record.get("final_test_opened") is False, f"{sources[role]} abrió el test final")
        if not role.startswith("organ/"):
            views.add((record.get("request") or {}).get("view_sha256"))
    _require(
        len(views) == 1 and None not in views,
        f"Los padres no comparten una vista ({sorted(map(str, views))}): no se alinean",
    )
    for seed in declaration["seeds"]:
        organ, static = records[f"organ/{seed}"], records[f"static/{seed}"]
        parent = organ.get("parent") or {}
        _require(
            organ.get("model") == declaration["organ"]
            and organ.get("arm") == arm
            and organ.get("origin") == origin
            and organ.get("seed") == seed
            and parent.get("model") == declaration["static"]
            and parent.get("run_id") == static.get("run_id"),
            f"{sources[f'organ/{seed}']} no es el brazo {label} de este ajuste de "
            f"{declaration['static']} (origen {origin}, semilla {seed})",
        )
        emissions = (organ.get("inputs") or {}).get("emissions") or {}
        for split in SPLITS:
            _require(
                emissions.get(split) == prints[f"static/{seed}/{split}"]["sha256"],
                f"El órgano de la semilla {seed} se emitió con otras predicciones del estático ({split})",
            )
        for split in AGGREGATED:
            _require(
                (organ.get("outputs") or {}).get(split)
                == prints[f"organ/{seed}/{split}"]["sha256"],
                f"La emisión de {split} del órgano de la semilla {seed} no es la de su registro",
            )
    return views.pop()


# ---------------------------------------------------------------------------------------------
# La orden


def measure(declaration, state, organs, origin, output, *, wait_hours, poll, log, markets=None):
    """F6 de un origen, publicado en `<output>/<origen>` (o `<origen>-<mercados>`)."""
    declaration, digest = checked_declaration(declaration)
    seeds = list(declaration["seeds"])
    _require(
        sorted(organs) == sorted(seeds),
        f"Hace falta una raíz de órganos por semilla declarada ({seeds}); hay {sorted(organs)}",
    )
    sources = folders(declaration, state, organs, origin)
    manifest = Path(state) / "cohorts" / str(origin) / "manifest.json"
    deadline = time.monotonic() + wait_hours * 3600
    while True:
        missing = missing_inputs(sources, manifest)
        if not missing or time.monotonic() >= deadline:
            break
        log(f"esperando {len(missing)} entradas; la primera: {missing[0]}")
        time.sleep(poll)
    if missing:
        raise og.MissingInputs(missing)
    prints = fingerprint(sources, manifest)
    target = Path(output) / og.target_name(origin, markets)
    if target.exists():
        return og._already(target, digest, prints, log)
    started = time.monotonic()

    def say(text):
        log(f"{time.monotonic() - started:6.0f} s {text}")

    roots = organ_declarations(
        [organs[s] for s in seeds], declaration["static"], declaration["organ_declaration_ignores"]
    )
    records = {
        role: json.loads((folder / "run.json").read_text(encoding="utf-8"))
        for role, folder in sources.items()
    }
    view = checked_records(declaration, origin, sources, records, prints)
    tool = og._screening()
    tables = {
        role: {s: og._table(folder, s) for s in AGGREGATED} for role, folder in sources.items()
    }
    reference = f"core/{seeds[0]}"
    present = og.markets_of(tables[reference])
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
                    table.column(name).equals(tables[reference][split].column(name)),
                    f"{sources[role]} no tiene las filas de {sources[reference]} en {split} ({name})",
                )
    alpha, _, _, matures = ac.known_rows(state, origin, tables[reference], view)
    say("alfa conocido y madurez")

    def prediction(role, split):
        return tables[role][split].column("prediction").to_numpy().astype(np.float64)

    names = {role: declaration[role] for role in ("core", "secondary", "static")}
    min_assets = declaration["min_assets"]
    Path(output).mkdir(parents=True, exist_ok=True)
    work = Path(output) / f".{origin}.{os.getpid()}.tmp"
    try:
        work.mkdir(parents=True)
        inputs = {}
        for seed in seeds:
            for role in ("core", "secondary", "static"):
                inputs[(role, seed)] = og._write_component(
                    work / "entradas" / f"{role}-{seed}",
                    tables[reference],
                    {s: prediction(f"{role}/{seed}", s) for s in AGGREGATED},
                    alpha,
                    matures,
                )
            inputs[("signal", seed)] = og._write_component(
                work / "entradas" / f"signal-{seed}",
                tables[reference],
                {
                    s: prediction(f"organ/{seed}", s) - prediction(f"static/{seed}", s)
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

        levels, contrasts, diversity, rows, factors = {}, {}, {}, [], {}
        for mode in declaration["modes"]:
            members = {split: {b: {} for b in BRAINS} for split in AGGREGATED}
            frames, factors[mode] = {}, {}
            for seed in seeds:
                h = aggregate(
                    {
                        names["static"]: inputs[("static", seed)],
                        og.SIGNAL: inputs[("signal", seed)],
                    },
                    mode,
                )
                c = aggregate(
                    {
                        names["core"]: inputs[("core", seed)],
                        names["secondary"]: inputs[("secondary", seed)],
                    },
                    mode,
                )
                if mode == "operable":
                    factors[mode][seed] = dict(h["operable"]["scales"], **c["operable"]["scales"])
                for split in AGGREGATED:
                    held, core = h["aligned"][split], c["aligned"][split]
                    frame = frames.setdefault(
                        split,
                        dict(
                            sample_id=list(held["sample_id"]),
                            target=np.asarray(held["target"], dtype=np.float64),
                            bounds=ac.bounds_of(held["sessions"]),
                            keys=[(str(m), int(t)) for m, t, _, _ in held["sessions"]],
                        ),
                    )
                    _require(
                        list(held["sample_id"]) == frame["sample_id"]
                        and list(core["sample_id"]) == frame["sample_id"],
                        f"Las filas alineadas de {mode}/{split} cambian entre semillas o conjuntos",
                    )
                    at_h = {n: i for i, n in enumerate(held["model_names"])}
                    at_c = {n: i for i, n in enumerate(core["model_names"])}
                    members[split]["H"][seed] = np.asarray(
                        h["aggregators"]["ridge_cov_active"]["split_predictions"][split],
                        dtype=np.float64,
                    )
                    members[split]["E"][seed] = held["predictions"][
                        :, at_h[names["static"]]
                    ].astype(np.float64)
                    members[split]["C"][seed] = core["predictions"][:, at_c[names["core"]]].astype(
                        np.float64
                    )
                    members[split]["T"][seed] = core["predictions"][
                        :, at_c[names["secondary"]]
                    ].astype(np.float64)
                say(f"{mode}: semilla {seed} agregada")
            levels[mode] = {}
            for split in AGGREGATED:
                frame = frames[split]
                bounds, keys = frame["bounds"], frame["keys"]
                brain = {
                    b: np.mean([session_ranks(members[split][b][s], bounds) for s in seeds], axis=0)
                    for b in BRAINS
                }
                series = {f"{b}_{s}": members[split][b][s] for b in BRAINS for s in seeds}
                series.update({f"{b}3": brain[b] for b in BRAINS})
                series.update(
                    C3H3=(brain["C"] + brain["H"]) / 2,
                    C3E3=(brain["C"] + brain["E"]) / 2,
                    T3H3=(brain["T"] + brain["H"]) / 2,
                    T3E3=(brain["T"] + brain["E"]) / 2,
                )
                ics = {
                    name: session_ics(tool, values, frame["target"], bounds, keys, min_assets)
                    for name, values in series.items()
                }
                levels[mode][split] = {name: tool.summary(v) for name, v in ics.items()}
                for name, values in ics.items():
                    rows += [
                        dict(
                            mode=mode,
                            split=split,
                            series=name,
                            market=k[0],
                            prediction_at=k[1],
                            ic=v,
                        )
                        for k, v in sorted(values.items())
                    ]
                if split != "evaluation":
                    continue
                contrasts[mode] = dict(
                    c3h3_vs_c3=paired(tool, ics["C3H3"], ics["C3"]),
                    c3h3_vs_c3e3=paired(tool, ics["C3H3"], ics["C3E3"]),
                    c3e3_vs_c3=paired(tool, ics["C3E3"], ics["C3"]),
                    h3_vs_c3=paired(tool, ics["H3"], ics["C3"]),
                    h3_vs_e3=paired(tool, ics["H3"], ics["E3"]),
                    t3h3_vs_t3=paired(tool, ics["T3H3"], ics["T3"]),
                    t3h3_vs_t3e3=paired(tool, ics["T3H3"], ics["T3E3"]),
                    **{f"c3_vs_c_{s}": paired(tool, ics["C3"], ics[f"C_{s}"]) for s in seeds},
                )
                diversity[mode] = dict(
                    ic_correlation_c3_h3=ic_correlation(ics["C3"], ics["H3"]),
                    ic_correlation_c3_e3=ic_correlation(ics["C3"], ics["E3"]),
                    ic_correlation_c3_t3=ic_correlation(ics["C3"], ics["T3"]),
                    within_c3_h3=within_correlation(
                        tool, brain["C"], brain["H"], bounds, min_assets
                    ),
                    within_c3_e3=within_correlation(
                        tool, brain["C"], brain["E"], bounds, min_assets
                    ),
                    within_c3_t3=within_correlation(
                        tool, brain["C"], brain["T"], bounds, min_assets
                    ),
                )
        shutil.rmtree(work / "entradas")
        code = og._code_identity()
        code["files"]["holographic"] = pilots._sha256(Path(__file__))
        summary = og._plain(
            dict(
                schema_version=1,
                kind=KIND,
                status="completed",
                origin=origin,
                declaration_sha256=digest,
                code=code,
                inputs=prints,
                organ_roots=roots,
                markets=measured,
                seeds=seeds,
                operable_factors=factors.get("operable"),
                levels=levels,
                contrasts=contrasts,
                diversity=diversity,
                verdict=verdict(contrasts, declaration["rule"]),
                seconds=round(time.monotonic() - started),
            )
        )
        pq.write_table(ac._series_table(rows), work / "sesiones.parquet")
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
    return "—" if value is None else f"{value:+.{digits}f}".replace(".", ",").replace("-", "−")


def _c(entry):
    if not entry or entry.get("mean") is None:
        return "—"
    interval = entry.get("interval")
    tail = "" if not interval else f" [{_f(interval[0])}; {_f(interval[1])}]"
    return f"{_f(entry['mean'])} (t {_f(entry['t_nw'], 2)}){tail}"


def markdown(summary):
    """Resumen legible de F6 de un origen."""
    result = summary["verdict"]
    lines = [
        f"# F6, el cerebro holográfico, en el origen {summary['origin']}",
        "",
        f"Declaración {summary['declaration_sha256'][:12]}; mercados {', '.join(summary['markets'])}; "
        f"semillas {summary['seeds']}; evaluación, diferencias pareadas por sesión con t de Newey-West.",
        "",
        f"**Lectura de la regla: {result['reading']}.**"
        + ("" if not result["reasons"] else " " + "; ".join(result["reasons"]) + "."),
        "",
        "## IC medio por sesión en evaluación",
        "",
        "| Serie | Etiqueta | Operable |",
        "|---|---|---|",
    ]
    names = ["C3", "H3", "E3", "T3", "C3H3", "C3E3", "T3H3", "T3E3"]
    names += sorted(k for k in summary["levels"]["label"]["evaluation"] if k not in names)
    for name in names:
        cells = [
            _f((summary["levels"].get(mode, {}).get("evaluation", {}).get(name) or {}).get("mean"))
            for mode in ("label", "operable")
        ]
        lines.append(f"| {name} | {cells[0]} | {cells[1]} |")
    lines += ["", "## Contrastes", "", "| Contraste | Etiqueta | Operable |", "|---|---|---|"]
    for key in summary["contrasts"]["label"]:
        lines.append(
            f"| {key} | {_c(summary['contrasts']['label'][key])} | "
            f"{_c(summary['contrasts'].get('operable', {}).get(key))} |"
        )
    lines += ["", "## Dónde se equivocan", "", "| Medida | Etiqueta | Operable |", "|---|---|---|"]
    for key in summary["diversity"]["label"]:
        lines.append(
            f"| {key} | {_f(summary['diversity']['label'][key], 3)} | "
            f"{_f(summary['diversity'].get('operable', {}).get(key), 3)} |"
        )
    return "\n".join(lines) + "\n"


def _organ_root(text):
    seed, sep, root = text.partition("=")
    if not sep or not seed.isdigit() or not root:
        raise argparse.ArgumentTypeError(f"--organs espera SEMILLA=RAÍZ, no {text!r}")
    return int(seed), Path(root)


def parser():
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = result.add_subparsers(dest="command", required=True)
    run = commands.add_parser("medir", help="Calcular F6 de un origen")
    run.add_argument("--declaration", type=Path, required=True)
    run.add_argument("--state", type=Path, required=True, help="Estado de la campaña")
    run.add_argument(
        "--organs",
        type=_organ_root,
        action="append",
        required=True,
        metavar="SEMILLA=RAÍZ",
        help="Raíz de los órganos de una semilla (una por semilla declarada)",
    )
    run.add_argument("--origin", type=int, required=True)
    run.add_argument("--output", type=Path, required=True, help="Raíz propia de F6")
    run.add_argument("--wait-hours", type=float, default=0.0)
    run.add_argument("--poll-seconds", type=float, default=300.0)
    run.add_argument(
        "--market",
        action="append",
        dest="markets",
        metavar="MERCADO",
        help="Medir solo este mercado (repetible); obligatorio en un estado con varios",
    )
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
            f"Faltan entradas de F6 para el origen {args.origin} tras esperar "
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
