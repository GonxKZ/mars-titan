#!/usr/bin/env python3
"""Contrastes con varias semillas: cada modelo frente a su control en un origen.

Las semillas de un mismo origen recorren las mismas sesiones, así que no son réplicas
independientes de una sesión. El IC de rangos de cada sesión se promedia entre semillas y el
contraste pareado con su control se hace sobre esa serie, con error con autocorrelación. La
diferencia de cada semilla se informa aparte, para ver si el signo se repite.

Además se mide el conjunto de cada modelo: el IC de la media de los rangos de sus semillas, que
es lo que se usaría al predecir con las tres. Su ganancia sobre la semilla media y el contraste
entre el conjunto de un modelo y el de su control van pareados por sesión.

Un modelo al que le falta alguna semilla no se contrasta con las que tenga: se anota como
incompleto.

Con varios orígenes (`--year` repetido) se añade el resumen entre orígenes. Cada origen cuenta
como una réplica: la media de las diferencias de los orígenes, su error entre orígenes y una t
con K - 1 grados de libertad, que no supone nada sobre la dependencia dentro de un año. Al lado
va el contraste con todas las sesiones juntas, cuya autocorrelación no cruza de un origen a
otro. Un origen en el que al modelo o a su control le falta una semilla queda fuera de ese
contraste y se nombra.

Uso:
    dgx_seed_contrasts.py --state <estado> --year 2013 --seed 42 --seed 43 --seed 44 \
        --contrasts <controles.json> --output <carpeta nueva>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

SPLITS = ("validation", "calibration", "evaluation")
DECISION = ("validation", "calibration")
# Las series que se miden con el alfa conocido, y el nivel del documento que lleva cada una con
# la clave del factor que la acompaña.
OPERABLE_KINDS = ("free", "liquid", "known", "returned")
OPERABLE_LEVELS = (
    ("free", "operable", "factor"),
    ("liquid", "operable_liquid", None),
    ("known", "label_known", None),
    ("returned", "label_returned", "back_factor"),
)


def _report_module():
    """El informe de la criba, de donde salen la lectura de emisiones y el contraste pareado."""
    name = "dgx_screening_report"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name("dgx_screening_report.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _home(state, model, year, references, reference_seed):
    """Dónde vive un modelo y qué semilla se lee allí para cada semilla pedida.

    Se busca en el estado de la campaña y después en los de referencia, y todas sus semillas se
    leen del mismo sitio. Una referencia ajustada una sola vez se lee siempre con su semilla.
    """
    place = Path(state)
    if not (place / "runs" / model / str(year)).is_dir():
        for other in references:
            if (Path(other) / "runs" / model / str(year)).is_dir():
                return Path(other), reference_seed
    return place, None


def _run(place, model, year, seed):
    return Path(place) / "runs" / model / str(year) / f"seed-{seed}"


def seed_splits(place, model, year, seed, market=None):
    """Emisiones de cada tramo de una ejecución, o None si le falta alguno.

    Con `market` solo quedan las sesiones de ese mercado. En una campaña conjunta cada sesión
    de cada mercado pesa lo mismo, y un cambio que solo sirve a uno se diluye en el total.
    """
    report = _report_module()
    run = _run(place, model, year, seed)
    result = {}
    for split in SPLITS:
        data = report.load_split(run, split)
        if data is None:
            return None
        if market is not None:
            kept = [session for session in data["sessions"] if session[0] == market]
            if not kept:
                raise ValueError(f"{run}: el tramo {split} no tiene sesiones de {market}")
            data = dict(data, sessions=kept)
        result[split] = data
    return result


def seed_sessions(splits):
    """IC de rangos por sesión de cada tramo de una ejecución."""
    report = _report_module()
    return {split: report.session_ic(splits[split]) for split in SPLITS}


def averaged(per_seed, model):
    """Media entre semillas del IC de cada sesión. Las semillas deben compartir sesiones."""
    result = {}
    for split in SPLITS:
        series = [sessions[split] for sessions in per_seed]
        keys = set(series[0])
        if any(set(other) != keys for other in series[1:]):
            raise ValueError(f"Las semillas de {model} no comparten sesiones en {split}")
        result[split] = {key: float(np.mean([one[key] for one in series])) for key in keys}
    return result


def _ranks(data):
    """Rangos medios de la predicción dentro de cada sesión, sin normalizar.

    Son múltiplos de 0,5 y su suma entre semillas es exacta en coma flotante. Normalizados
    antes de sumar, dos activos con la misma suma de rangos podrían diferir en el último bit y
    dejar de contar como empate.
    """
    report = _report_module()
    ranks = np.empty(len(data["target"]), dtype=np.float64)
    for _, _, start, end in data["sessions"]:
        ranks[start:end] = report.average_ranks(data["prediction"][start:end])
    return ranks


def ensemble(members, model):
    """IC por sesión de la media de rangos de las semillas: el conjunto de un modelo."""
    report = _report_module()
    result = {}
    for split in SPLITS:
        base = members[0][split]
        if any(other[split]["sample"] != base["sample"] for other in members[1:]):
            raise ValueError(f"Las semillas de {model} no comparten muestras en {split}")
        # Las semillas comparten filas, así que la suma de rangos ordena igual que su media.
        result[split] = report.session_ic(base, sum(_ranks(member[split]) for member in members))
    return result


def _provenance(place, model, year, seeds):
    """Los sitios donde se ajustaron las semillas de un modelo, según sus registros de la cola."""
    report = _report_module()
    places = []
    for seed in dict.fromkeys(seeds):
        path = Path(place) / "queue" / f"{model}--{year}--seed-{seed}.json"
        if not path.is_file():
            continue
        for item in report.provenance(json.loads(path.read_text(encoding="utf-8"))):
            if item not in places:
                places.append(item)
    return places


def _operable(offset, known, run, splits):
    """IC sin alfa por sesión de una ejecución, con todas las filas y en el tercio más líquido.

    La etiqueta sin alfa es el rendimiento que se puede operar y la predicción para ella es la
    del modelo, por un factor, más alfa. El factor se elige con el IC sin alfa de calibración y
    con él se miden los tres tramos. El alfa de las filas no depende del modelo ni de la
    semilla: se lee una vez por tramo y se reutiliza mientras las muestras sean las mismas.

    Un modelo ajustado para lo operable se lee además de vuelta en la etiqueta: su predicción
    de la etiqueta es factor·(p + alfa) − alfa, con otro factor elegido igual, y se mide en las
    filas con alfa conocido junto a la etiqueta con la predicción tal cual.
    """
    report = _report_module()
    alphas, volumes = {}, {}
    for split in SPLITS:
        key = (split, splits[split]["sample"])
        if key not in known:
            known[key] = offset.rows(run, split)
        alphas[split], volumes[split] = known[key]
    factor = report.best_scale(splits, alphas)

    def measure(liquidity):
        return {
            split: report.implied_sessions(
                splits[split],
                alphas[split],
                None if liquidity is None else liquidity[split],
                free=True,
                prediction=factor * splits[split]["prediction"],
            )
            for split in SPLITS
        }

    # La etiqueta en las mismas filas con alfa conocido: con la predicción tal cual, y con la
    # predicción de lo operable devuelta a la etiqueta por el factor que elige la calibración.
    back = report.best_return_scale(splits, alphas)
    return dict(
        factor=factor,
        free=measure(None),
        liquid=measure(volumes),
        known={
            split: report.implied_sessions(
                splits[split], alphas[split], free=False, prediction=splits[split]["prediction"]
            )
            for split in SPLITS
        },
        back_factor=back,
        returned={
            split: report.returned_sessions(splits[split], alphas[split], back) for split in SPLITS
        },
    )


def _operable_level(sessions, per_seed, factors=None):
    report = _report_module()
    entry = {split: report.summary(sessions[split]) for split in SPLITS}
    entry["decision"] = report.summary(_pooled(sessions, DECISION))
    # Un universo pequeño puede no tener sesiones con un tercio líquido suficiente.
    entry["per_seed"] = {
        split: [
            float(np.mean(list(one[split].values()))) if one[split] else None for one in per_seed
        ]
        for split in SPLITS
    }
    if factors is not None:
        entry["factors"] = factors
        # En un borde de la rejilla la búsqueda quedó truncada y el factor es una cota.
        edges = (min(report.SCALES), max(report.SCALES))
        entry["factor_at_edge"] = any(factor in edges for factor in factors)
    return entry


def _pooled(sessions, splits):
    return {(split, *key): value for split in splits for key, value in sessions[split].items()}


def _paired(candidate, control):
    """Contraste pareado en cada tramo y en los dos de decisión juntos."""
    report = _report_module()
    entry = {split: report.paired(candidate[split], control[split]) for split in SPLITS}
    entry["decision"] = report.paired(_pooled(candidate, DECISION), _pooled(control, DECISION))
    return entry


def _level(sessions, per_seed, combined):
    report = _report_module()
    entry = {split: report.summary(sessions[split]) for split in SPLITS}
    entry["decision"] = report.summary(_pooled(sessions, DECISION))
    # Un universo sin ninguna sesión válida no tiene media: se dice con None y no con NaN.
    entry["per_seed"] = {
        split: [
            float(np.mean(list(one[split].values()))) if one[split] else None for one in per_seed
        ]
        for split in SPLITS
    }
    entry["ensemble"] = {split: report.summary(combined[split]) for split in SPLITS}
    entry["ensemble"]["decision"] = report.summary(_pooled(combined, DECISION))
    # Lo que gana el conjunto sobre la semilla media, sesión a sesión.
    entry["ensemble_gain"] = _paired(combined, sessions)
    return entry


def build(
    state, year, seeds, controls, references=(), reference_seed=None, offset=None, market=None
):
    """Niveles de cada modelo y contraste de cada uno con su control.

    Con `offset` (el alfa conocido de cada fila) se mide además el rendimiento operable. Con
    `market` todo se mide solo en las sesiones de ese mercado.
    """
    return _origin(state, year, seeds, controls, references, reference_seed, offset, market)[0]


def _origin(
    state, year, seeds, controls, references=(), reference_seed=None, offset=None, market=None
):
    """El documento de un origen y sus series de IC por sesión, una por clase de medida."""
    report = _report_module()
    seeds = list(dict.fromkeys(seeds))
    names = sorted(set(controls) | set(controls.values()))
    per_seed, mean, combined, missing, where = {}, {}, {}, {}, {}
    known, measured_seed, measured_mean = {}, {}, {}
    for name in names:
        # Las emisiones de un modelo se sueltan al pasar al siguiente: solo queda el IC.
        place, fixed = _home(state, name, year, references, reference_seed)
        if fixed is None:
            runs = [seed_splits(place, name, year, seed, market) for seed in seeds]
        else:
            # Un solo ajuste: se lee una vez y hace de pareja de todas las semillas.
            runs = [seed_splits(place, name, year, fixed, market)] * len(seeds)
        absent = [seed for seed, run in zip(seeds, runs, strict=True) if run is None]
        if absent:
            missing[name] = absent
            continue
        where[name] = _provenance(place, name, year, seeds if fixed is None else [fixed])
        per_seed[name] = [seed_sessions(run) for run in runs]
        mean[name] = averaged(per_seed[name], name)
        combined[name] = ensemble(runs, name)
        if offset is not None:
            if fixed is None:
                measured = [
                    _operable(offset, known, _run(place, name, year, seed), run)
                    for seed, run in zip(seeds, runs, strict=True)
                ]
            else:
                measured = [_operable(offset, known, _run(place, name, year, fixed), runs[0])]
                measured = measured * len(seeds)
            measured_seed[name] = measured
            measured_mean[name] = {
                kind: averaged([item[kind] for item in measured], name) for kind in OPERABLE_KINDS
            }
    document = dict(
        schema_version=1,
        kind="dgx_seed_contrasts_v1",
        final_test_opened=False,
        year=year,
        seeds=seeds,
        models={
            name: dict(_level(mean[name], per_seed[name], combined[name]), provenance=where[name])
            for name in mean
        },
        contrasts={},
        incomplete={},
    )
    if market is not None:
        document["market"] = market
    for name, per_kind in measured_mean.items():
        seeds_of = measured_seed[name]
        for kind, key, factor in OPERABLE_LEVELS:
            document["models"][name][key] = _operable_level(
                per_kind[kind],
                [item[kind] for item in seeds_of],
                None if factor is None else [item[factor] for item in seeds_of],
            )
    for name, control in sorted(controls.items()):
        if name in missing or control in missing:
            document["incomplete"][name] = sorted(
                set(missing.get(name, [])) | set(missing.get(control, []))
            )
            continue
        entry = dict(control=control, seeds=seeds, per_seed={}, signs={})
        for split in SPLITS:
            entry[split] = report.paired(mean[name][split], mean[control][split])
            differences = [
                report.paired(one[split], other[split])["mean"]
                for one, other in zip(per_seed[name], per_seed[control], strict=True)
            ]
            entry["per_seed"][split] = differences
            entry["signs"][split] = "".join(
                "?" if value is None else "+" if value > 0 else "-" for value in differences
            )
        entry["decision"] = report.paired(
            _pooled(mean[name], DECISION), _pooled(mean[control], DECISION)
        )
        entry["ensemble"] = _paired(combined[name], combined[control])
        if offset is not None:
            one, other = measured_mean[name], measured_mean[control]
            entry["operable"] = _paired(one["free"], other["free"])
            entry["operable_liquid"] = _paired(one["liquid"], other["liquid"])
            # La etiqueta devuelta del modelo frente a la etiqueta tal cual de su control.
            entry["label_returned"] = _paired(one["returned"], other["known"])
        document["contrasts"][name] = entry
    series = dict(seeds=mean, ensemble=combined)
    if offset is not None:
        series.update(
            operable={name: per_kind["free"] for name, per_kind in measured_mean.items()},
            operable_liquid={name: per_kind["liquid"] for name, per_kind in measured_mean.items()},
        )
    return document, series


def _between(values):
    """Media de las diferencias de los orígenes con su error entre orígenes."""
    values = [value for value in values if value is not None]
    count = len(values)
    if count == 0:
        return dict(origins=0, mean=None, se=None, t=None, df=0, p=None, positive=0)
    result = dict(
        origins=count,
        mean=float(np.mean(values)),
        se=None,
        t=None,
        df=count - 1,
        p=None,
        positive=int(sum(value > 0 for value in values)),
    )
    if count < 2:
        return result
    error = float(np.std(values, ddof=1) / math.sqrt(count))
    result["se"] = error
    if error > 0:
        from scipy import stats

        result["t"] = result["mean"] / error
        result["p"] = float(2 * stats.t.sf(abs(result["t"]), count - 1))
    return result


def _split(sessions, split):
    """Las sesiones de un tramo, o las de los dos de decisión juntas."""
    return _pooled(sessions, DECISION) if split == "decision" else sessions[split]


def build_origins(
    state, years, seeds, controls, references=(), reference_seed=None, offsets=None, market=None
):
    """Los contrastes de cada origen y su resumen entre orígenes.

    `offsets` da, para cada año, el alfa conocido de sus filas; con él se resume también el
    rendimiento operable. Con `market` todo se mide solo en las sesiones de ese mercado.
    """
    report = _report_module()
    years = list(dict.fromkeys(years))
    parts = [
        _origin(
            state,
            year,
            seeds,
            controls,
            references,
            reference_seed,
            None if offsets is None else offsets(year),
            market,
        )
        for year in years
    ]
    document = dict(
        schema_version=1,
        kind="dgx_seed_contrasts_v1",
        final_test_opened=False,
        years=years,
        seeds=parts[0][0]["seeds"],
        origins=[part[0] for part in parts],
        across={},
    )
    if market is not None:
        document["market"] = market
    for name, control in sorted(controls.items()):
        present = [part for part in parts if name in part[0]["contrasts"]]
        if not present:
            continue
        inside = [part[0]["year"] for part in present]
        entry = dict(
            control=control, years=inside, missing=[year for year in years if year not in inside]
        )
        for kind in present[0][1]:
            entry[kind] = {}
            for split in ("decision", "evaluation"):
                means, left, right = [], {}, {}
                for origin, series in present:
                    contrast = origin["contrasts"][name]
                    means.append((contrast if kind == "seeds" else contrast[kind])[split]["mean"])
                    sessions = series[kind]
                    for target, model in ((left, name), (right, control)):
                        target.update(
                            {
                                (origin["year"], *key): value
                                for key, value in _split(sessions[model], split).items()
                            }
                        )
                entry[kind][split] = dict(
                    origin_means=means, between=_between(means), pooled=report.paired(left, right)
                )
        document["across"][name] = entry
    return document


def _places(document, prefix=""):
    """Las líneas de la procedencia de los modelos de un origen."""
    report = _report_module()
    models = {name: model["provenance"] for name, model in document["models"].items()}
    return report.provenance_lines(models, prefix=prefix)


PLACES = [
    "",
    "## Procedencia de los ajustes",
    "",
    "Equipo y GPU de cada ajuste, según la telemetría de sus intentos.",
    "",
]


def _number(value, digits=4, sign=False):
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def _difference(entry, sign=True):
    return f"{_number(entry['mean'], sign=sign)} ({_number(entry['t'], 1, True)})"


def _operable_table(document):
    """El rendimiento operable de un origen, si se midió."""
    rows = {name: entry for name, entry in document["contrasts"].items() if "operable" in entry}
    if not rows:
        return []
    lines = [
        "",
        "## Rendimiento operable",
        "",
        "IC de la etiqueta sin alfa con la predicción más alfa: lo que se puede operar. El factor",
        "de escala de cada semilla se elige con la calibración. El tercio líquido son las filas",
        "con más volumen en dólares de cada sesión. La etiqueta devuelta es el IC de la etiqueta",
        "con factor·(p + alfa) − alfa, lo que conserva de la etiqueta un modelo ajustado para lo",
        "operable; se compara con la etiqueta tal cual de su control en las filas con alfa.",
        "",
        "| Modelo | Control | IC operable de evaluación | Δ decisión (t) | Δ evaluación (t) "
        "| Tercio líquido | Δ evaluación en el tercio líquido (t) | Etiqueta devuelta "
        "| Δ frente a la etiqueta del control (t) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, entry in rows.items():
        model = document["models"][name]
        lines.append(
            f"| {name} | {entry['control']} "
            f"| {_number(model['operable']['evaluation']['mean'])} "
            f"| {_difference(entry['operable']['decision'])} "
            f"| {_difference(entry['operable']['evaluation'])} "
            f"| {_number(model['operable_liquid']['evaluation']['mean'])} "
            f"| {_difference(entry['operable_liquid']['evaluation'])} "
            f"| {_number(model['label_returned']['evaluation']['mean'])} "
            f"| {_difference(entry['label_returned']['evaluation'])} |"
        )
    edged = sorted(
        name
        for name in {*rows, *(entry["control"] for entry in rows.values())}
        if any(
            document["models"][name][level].get("factor_at_edge")
            for level in ("operable", "label_returned")
        )
    )
    if edged:
        lines += [
            "",
            "Factor en el borde de la rejilla (la búsqueda quedó truncada y el valor es una cota): "
            + ", ".join(edged)
            + ".",
        ]
    return lines


def _market_note(document):
    """La línea que dice que solo se leyó un mercado, si es así."""
    market = document.get("market")
    return [] if market is None else [f"Solo las sesiones de {market}.", ""]


def markdown(document):
    lines = [
        f"# Contrastes con {len(document['seeds'])} semillas, origen {document['year']}",
        "",
        *_market_note(document),
        "El IC de cada sesión es la media de las semillas. La t usa error con autocorrelación.",
        "El conjunto es el IC de la media de rangos de las semillas, y su Δ compara el conjunto",
        "del modelo con el de su control, en evaluación.",
        "",
        "| Modelo | Control | Semillas | IC de evaluación | Conjunto | Δ decisión (t) "
        "| Δ evaluación (t) | Δ conjunto (t) | Signo por semilla |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, entry in document["contrasts"].items():
        model = document["models"][name]
        level, combined = model["evaluation"]["mean"], model["ensemble"]["evaluation"]["mean"]
        decision, evaluation = entry["decision"], entry["evaluation"]
        between = entry["ensemble"]["evaluation"]
        lines.append(
            f"| {name} | {entry['control']} | {len(entry['seeds'])} | {_number(level)} "
            f"| {_number(combined)} "
            f"| {_number(decision['mean'], sign=True)} ({_number(decision['t'], 1, True)}) "
            f"| {_number(evaluation['mean'], sign=True)} ({_number(evaluation['t'], 1, True)}) "
            f"| {_number(between['mean'], sign=True)} ({_number(between['t'], 1, True)}) "
            f"| {entry['signs']['evaluation']} |"
        )
    lines += [
        "",
        "## Ganancia del conjunto sobre la semilla media",
        "",
        "| Modelo | IC de la semilla media | IC del conjunto | Ganancia en decisión (t) "
        "| Ganancia en evaluación (t) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, model in document["models"].items():
        gain = model["ensemble_gain"]
        lines.append(
            f"| {name} | {_number(model['evaluation']['mean'])} "
            f"| {_number(model['ensemble']['evaluation']['mean'])} "
            f"| {_number(gain['decision']['mean'], sign=True)} "
            f"({_number(gain['decision']['t'], 1, True)}) "
            f"| {_number(gain['evaluation']['mean'], sign=True)} "
            f"({_number(gain['evaluation']['t'], 1, True)}) |"
        )
    if document["incomplete"]:
        lines += ["", "Sin contrastar, por semillas que faltan:", ""]
        lines += [
            f"- {name}: semillas {', '.join(map(str, seeds))}"
            for name, seeds in document["incomplete"].items()
        ]
    lines += _operable_table(document)
    lines += [*PLACES, *_places(document)]
    return "\n".join(lines) + "\n"


def markdown_origins(document):
    years = document["years"]
    lines = [
        f"# Contrastes con {len(document['seeds'])} semillas en {len(years)} orígenes",
        "",
        f"Orígenes: {', '.join(map(str, years))}.",
        *_market_note(document),
        "El IC de cada sesión es la media de las semillas.",
        "",
        "## Entre orígenes",
        "",
        "Evaluación. Cada origen cuenta como una réplica: la t entre orígenes tiene K - 1 grados",
        "de libertad. La Δ por sesión junta todas las sesiones, con error con autocorrelación",
        "dentro de cada origen.",
        "",
        "| Modelo | Control | Orígenes | Δ media | Error entre orígenes | t entre orígenes (gl) "
        "| p | A favor | Δ por sesión (t) | Δ del conjunto (t entre orígenes) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, entry in document["across"].items():
        block, combined = entry["seeds"]["evaluation"], entry["ensemble"]["evaluation"]
        between, pooled, other = block["between"], block["pooled"], combined["between"]
        left_out = "".join(f" (sin {year})" for year in entry["missing"])
        lines.append(
            f"| {name} | {entry['control']} | {between['origins']}{left_out} "
            f"| {_number(between['mean'], sign=True)} | {_number(between['se'])} "
            f"| {_number(between['t'], 1, True)} ({between['df']}) | {_number(between['p'], 3)} "
            f"| {between['positive']} de {between['origins']} "
            f"| {_number(pooled['mean'], sign=True)} ({_number(pooled['t'], 1, True)}) "
            f"| {_number(other['mean'], sign=True)} ({_number(other['t'], 1, True)}) |"
        )
    lines += [
        "",
        "## Δ de evaluación de cada origen",
        "",
        f"| Modelo | {' | '.join(map(str, years))} |",
        f"| --- | {' | '.join('---' for _ in years)} |",
    ]
    for name, entry in document["across"].items():
        by_year = dict(
            zip(entry["years"], entry["seeds"]["evaluation"]["origin_means"], strict=True)
        )
        cells = " | ".join(_number(by_year.get(year), sign=True) for year in years)
        lines.append(f"| {name} | {cells} |")
    measured = {name: entry for name, entry in document["across"].items() if "operable" in entry}
    if measured:
        lines += [
            "",
            "## Entre orígenes: rendimiento operable",
            "",
            "Evaluación sobre la etiqueta sin alfa, con el factor de escala elegido en",
            "calibración.",
            "",
            "| Modelo | Control | Orígenes | Δ media | Error entre orígenes "
            "| t entre orígenes (gl) | p | A favor | Δ por sesión (t) "
            "| Tercio líquido: Δ media (t entre orígenes) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, entry in measured.items():
            block = entry["operable"]["evaluation"]
            between, liquid = block["between"], entry["operable_liquid"]["evaluation"]["between"]
            lines.append(
                f"| {name} | {entry['control']} | {between['origins']} "
                f"| {_number(between['mean'], sign=True)} | {_number(between['se'])} "
                f"| {_number(between['t'], 1, True)} ({between['df']}) "
                f"| {_number(between['p'], 3)} | {between['positive']} de {between['origins']} "
                f"| {_difference(block['pooled'])} | {_difference(liquid)} |"
            )
    lines += PLACES
    for origin in document["origins"]:
        lines += _places(origin, prefix=f"Origen {origin['year']}, ")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--year", type=int, action="append", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--contrasts", type=Path, required=True, help="JSON con 'controls'")
    parser.add_argument("--output", type=Path, required=True, help="Carpeta nueva")
    parser.add_argument(
        "--reference",
        type=Path,
        action="append",
        default=[],
        help="Otro estado donde buscar los modelos que no están en el de la campaña",
    )
    parser.add_argument(
        "--reference-seed",
        type=int,
        help="Semilla con la que se lee una referencia ajustada una sola vez",
    )
    parser.add_argument(
        "--offset",
        action="store_true",
        help="Medir también el rendimiento operable, con el alfa conocido de cada fila",
    )
    parser.add_argument(
        "--market",
        help="Medir solo las sesiones de este mercado (por ejemplo CN en una campaña conjunta)",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="MODELO:CONTROL",
        help="Contraste que se añade a los declarados",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("La carpeta de salida ya existe")
    controls = dict(json.loads(args.contrasts.read_text(encoding="utf-8"))["controls"])
    for pair in args.pair:
        name, _, control = pair.partition(":")
        if not name or not control:
            parser.error(f"--pair se escribe MODELO:CONTROL y se recibió {pair!r}")
        if controls.get(name, control) != control:
            parser.error(f"{name} ya tiene declarado otro control: {controls[name]}")
        controls[name] = control
    sources = dict(
        references=args.reference, reference_seed=args.reference_seed, market=args.market
    )

    def known(year):
        return _report_module().KnownOffset(args.state, year)

    if len(args.year) == 1:
        offset = known(args.year[0]) if args.offset else None
        document = build(args.state, args.year[0], args.seed, controls, offset=offset, **sources)
        table = markdown(document)
    else:
        offsets = known if args.offset else None
        document = build_origins(
            args.state, args.year, args.seed, controls, offsets=offsets, **sources
        )
        table = markdown_origins(document)
    args.output.mkdir(parents=True)
    text = json.dumps(document, ensure_ascii=False, indent=2, default=str) + "\n"
    (args.output / "seed-contrasts.json").write_text(text, encoding="utf-8")
    (args.output / "seed-contrasts.md").write_text(table, encoding="utf-8")
    print(table, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
