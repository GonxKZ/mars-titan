#!/usr/bin/env python3
"""Búsqueda de hiperparámetros sobre la campaña: cada ensayo es un modelo más de una criba.

Optuna propone los valores y la campaña entrena. Cada tanda de ensayos se vuelve una receta
por ensayo y una declaración de campaña con el modelo base y todos los ensayos hasta entonces,
de modo que los trabajos ya hechos se heredan y cada ajuste conserva su identidad, su
telemetría y su procedencia. La búsqueda solo mira el IC de rangos de validación del origen
declarado. La calibración confirma a los mejores y el tramo final del origen no se lee aquí.

Los primeros ensayos son un hipercubo latino del espacio, fijado por la semilla. Después propone
el estimador de Parzen en árbol (TPE) multivariante, que cuenta con los ensayos aún en marcha.

Órdenes:
    ask     pide una tanda y escribe sus recetas, su campaña y sus controles en el repositorio
    tell    lee de un estado de campaña los ensayos terminados y los modelos ya conocidos
    fail    da por fallido un ensayo que no va a terminar
    report  ensayos por validación, confirmación en calibración y candidato

Uso de una tanda:
    campaign_search.py ask --study <carpeta> --declaration <json> --repo <repositorio> --count 6
    (commit de lo escrito; plan, prepare, inherit y run --tier search_g<N> de la campaña)
    campaign_search.py tell --study <carpeta> --declaration <json> --state <estado>
    campaign_search.py report --study <carpeta> --declaration <json> --output <carpeta nueva>
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np

KIND = "campaign_search_v1"
SPLITS = ("validation", "calibration")
FIELDS = {
    "schema_version",
    "kind",
    "study",
    "tag",
    "description",
    "final_test_access",
    "seed",
    "campaign",
    "base_model",
    "year",
    "run_seed",
    "startup",
    "confirm",
    "space",
    "known",
}
# Lo que define la búsqueda. Cambiar cuántos se confirman o qué modelos se dan por conocidos
# no cambia lo que Optuna ha visto ni lo que propone el espacio.
IDENTITY = (
    "study",
    "tag",
    "seed",
    "campaign",
    "base_model",
    "year",
    "run_seed",
    "startup",
    "space",
)
_STUDY = re.compile(r"[a-z0-9][a-z0-9-]{0,47}")
_TAG = re.compile(r"[a-z0-9]{1,8}")
CLOSED = "Sin resultados publicados. El test final de 2024 permanece cerrado."


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


def _optuna():
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
    return optuna


def _inside(item, value):
    if item["type"] == "ordinal":
        return any(value == other and type(value) is type(other) for other in item["values"])
    number = isinstance(value, (int, float)) and not isinstance(value, bool)
    return number and item["low"] <= value <= item["high"]


def validate(declared):
    """La declaración de una búsqueda, o ValueError con lo que le falta."""
    unknown = sorted(set(declared) - FIELDS)
    if unknown:
        raise ValueError(
            f"Campos que no se declaran en una búsqueda: {', '.join(unknown)}. El objetivo es "
            "siempre el IC de rangos de validación y no se declara"
        )
    if declared.get("schema_version") != 1 or declared.get("kind") != KIND:
        raise ValueError(f"La declaración debe ser {KIND} con schema_version 1")
    if declared.get("final_test_access") is not False:
        raise ValueError("final_test_access debe ser false: el test final sigue cerrado")
    if not _STUDY.fullmatch(str(declared.get("study", ""))):
        raise ValueError("study debe ser un nombre en minúsculas, cifras y guiones")
    if not _TAG.fullmatch(str(declared.get("tag", ""))):
        raise ValueError("tag debe tener de una a ocho minúsculas o cifras")
    for field in ("seed", "year", "run_seed", "startup", "confirm"):
        value = declared.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field} debe ser un entero no negativo")
    if declared["startup"] < 2:
        raise ValueError("startup debe ser al menos 2: el hipercubo latino necesita dos puntos")
    if declared["confirm"] < 1:
        raise ValueError("confirm debe ser al menos 1")
    for field in ("campaign", "base_model"):
        if not isinstance(declared.get(field), str) or not declared[field]:
            raise ValueError(f"Falta {field}")
    space = declared.get("space")
    if not isinstance(space, list) or not space:
        raise ValueError("space debe ser una lista con al menos un campo")
    names = [item.get("name") for item in space]
    if len(set(names)) != len(names):
        raise ValueError("space repite un campo")
    for item in space:
        name = item.get("name")
        if not isinstance(name, str) or not name.startswith(("recipe.", "predictor.")):
            raise ValueError(f"El campo {name!r} debe empezar por recipe. o predictor.")
        if item.get("type") == "float":
            low, high = item.get("low"), item.get("high")
            numbers = all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in (low, high)
            )
            if not numbers or not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError(f"{name}: low debe ser menor que high, y los dos finitos")
            if item.get("log") and low <= 0:
                raise ValueError(f"{name}: con log, low debe ser positivo")
        elif item.get("type") == "ordinal":
            values = item.get("values")
            if not isinstance(values, list) or len(values) < 2:
                raise ValueError(f"{name}: values debe tener al menos dos valores ordenados")
        else:
            raise ValueError(f"{name}: type debe ser float u ordinal")
    by_name = {item["name"]: item for item in space}
    for item in declared.get("known") or []:
        values = item.get("values") or {}
        if not isinstance(item.get("model"), str) or set(values) != set(by_name):
            raise ValueError("known: cada modelo conocido da un valor a cada campo del espacio")
        for name, value in values.items():
            if not _inside(by_name[name], value):
                raise ValueError(f"known: {item['model']} está fuera del espacio en {name}")
    return declared


def _identity(declared):
    chosen = {field: declared[field] for field in IDENTITY}
    text = json.dumps(chosen, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _distributions(space):
    optuna = _optuna()
    result = {}
    for item in space:
        if item["type"] == "float":
            result[item["name"]] = optuna.distributions.FloatDistribution(
                item["low"], item["high"], log=bool(item.get("log"))
            )
        else:
            # El índice conserva el orden de los valores, que una categoría perdería.
            result[item["name"]] = optuna.distributions.IntDistribution(0, len(item["values"]) - 1)
    return result


def _values(space, params):
    """De los parámetros de Optuna a los valores que lleva la receta."""
    result = {}
    for item in space:
        raw = params[item["name"]]
        if item["type"] == "ordinal":
            result[item["name"]] = item["values"][int(raw)]
        else:
            value = float(f"{float(raw):.6g}")
            result[item["name"]] = min(max(value, item["low"]), item["high"])
    return result


def _params(space, values):
    """De los valores de una receta a los parámetros de Optuna."""
    result = {}
    for item in space:
        value = values[item["name"]]
        if item["type"] == "ordinal":
            result[item["name"]] = next(
                index
                for index, other in enumerate(item["values"])
                if value == other and type(value) is type(other)
            )
        else:
            result[item["name"]] = float(value)
    return result


def latin(space, count, seed):
    """Hipercubo latino: cada campo reparte los ensayos en tantos tramos como ensayos haya."""
    rng = np.random.default_rng([seed, 0x5EA7C4])
    rows = [{} for _ in range(count)]
    for item in space:
        unit = (rng.permutation(count) + rng.random(count)) / count
        for row, point in zip(rows, unit, strict=True):
            if item["type"] == "ordinal":
                size = len(item["values"])
                row[item["name"]] = min(int(point * size), size - 1)
                continue
            low, high = float(item["low"]), float(item["high"])
            if item.get("log"):
                value = math.exp(math.log(low) + point * (math.log(high) - math.log(low)))
            else:
                value = low + point * (high - low)
            row[item["name"]] = min(max(float(f"{value:.6g}"), low), high)
    return rows


def _write(path, document):
    """Escritura atómica de un JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f"{path.name}.pending-{os.getpid()}")
    pending.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(pending, path)


def _ledger(folder):
    path = folder / "trials.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return dict(schema_version=1, kind="campaign_search_trials_v1", trials=[], known={})


def _study(folder, declared):
    """El estudio de Optuna de la carpeta: lo crea con su hipercubo latino o lo reanuda."""
    optuna = _optuna()
    validate(declared)
    folder.mkdir(parents=True, exist_ok=True)
    manifest, identity = folder / "manifest.json", _identity(declared)
    if manifest.is_file():
        if json.loads(manifest.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError(
                "La declaración de la búsqueda ha cambiado respecto a la de esta carpeta: una "
                "búsqueda distinta necesita otro nombre y otra carpeta"
            )
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(folder / "optuna.journal"))
    )
    if manifest.is_file():
        study = optuna.load_study(study_name=declared["study"], storage=storage)
    else:
        study = optuna.create_study(
            study_name=declared["study"], storage=storage, direction="maximize"
        )
        for row in latin(declared["space"], declared["startup"], declared["seed"]):
            study.enqueue_trial(row)
        _write(manifest, dict(schema_version=1, identity=identity, declaration=declared))
    trials = study.get_trials(deepcopy=False)
    # Un ensayo pedido que no llegó al registro (un corte entre las dos escrituras) se cierra.
    registered = {entry["number"] for entry in _ledger(folder)["trials"]}
    for trial in trials:
        if trial.state == optuna.trial.TrialState.RUNNING and trial.number not in registered:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    # La semilla del muestreador depende de la historia: reanudar no repite propuestas y la
    # misma historia da las mismas.
    text = f"{declared['seed']}:{len(trials)}".encode()
    seed = int.from_bytes(hashlib.sha256(text).digest()[:4], "big")
    study.sampler = optuna.samplers.TPESampler(
        seed=seed, multivariate=True, constant_liar=True, n_startup_trials=declared["startup"]
    )
    return study


def _apply(recipe, values, origin):
    """La receta con los valores puestos en sus campos. Los campos padre deben existir."""
    result = copy.deepcopy(recipe)
    for name, value in values.items():
        cursor, parts = result, name.split(".")
        for part in parts[:-1]:
            if not isinstance(cursor.get(part), dict):
                raise ValueError(f"La receta {origin} no tiene el campo {part} de {name}")
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return result


def _campaign(base, digest, declared, trials, generation):
    """La campaña de una tanda: el modelo base y todos los ensayos vivos hasta ella."""
    model = base["models"][declared["base_model"]]
    models = {declared["base_model"]: model}
    recipes = {model["recipe"]: base["recipes"][model["recipe"]]}
    for entry in trials:
        key = f"search_{declared['tag']}t{entry['number']:03d}"
        models[entry["model"]] = dict(model, recipe=key, tier=f"search_g{entry['generation']}")
        recipes[key] = entry["recipe"]
    return dict(
        base,
        description=(
            f"Búsqueda {declared['study']}, tanda {generation}: el modelo {declared['base_model']} "
            "y los ensayos pedidos hasta ella, cada uno con los campos del espacio declarado en "
            "otro valor. Se elige por el IC de rangos de validación del origen "
            f"{declared['year']} con la semilla {declared['run_seed']}."
        ),
        source_declaration_sha256=digest,
        models=models,
        recipes=recipes,
        tiers=[model["tier"], *(f"search_g{index}" for index in range(1, generation + 1))],
    )


def ask(folder, declared, repo, count):
    """Pide una tanda de ensayos y escribe sus recetas y su campaña en el repositorio."""
    folder, repo = Path(folder), Path(repo)
    validate(declared)
    source = repo / declared["campaign"]
    base = json.loads(source.read_text(encoding="utf-8"))
    if base.get("final_test_access") is not False:
        raise ValueError("La campaña de partida debe tener final_test_access en false")
    model = base["models"][declared["base_model"]]
    origin = base["recipes"][model["recipe"]]
    recipe = json.loads((repo / origin).read_text(encoding="utf-8"))
    # Un campo mal escrito se ve antes de gastar ensayos.
    _apply(recipe, {item["name"]: None for item in declared["space"]}, origin)
    study = _study(folder, declared)
    ledger = _ledger(folder)
    generation = 1 + max((entry["generation"] for entry in ledger["trials"]), default=0)
    batch, distributions = [], _distributions(declared["space"])
    for _ in range(count):
        trial = study.ask(distributions)
        values = _values(declared["space"], trial.params)
        target = f"configs/titans/search/{declared['study']}/t{trial.number:03d}.json"
        entry = dict(
            number=trial.number,
            generation=generation,
            model=f"{declared['base_model']}_{declared['tag']}t{trial.number:03d}",
            recipe=target,
            values=values,
            status="asked",
        )
        ledger["trials"].append(entry)
        _write(folder / "trials.json", ledger)
        changed = ", ".join(f"{name} = {value!r}" for name, value in values.items())
        document = _apply(recipe, values, origin)
        document["pending"] = [
            f"Ensayo {trial.number} de la búsqueda {declared['study']}: es {Path(origin).name} "
            f"con {changed}.",
            CLOSED,
        ]
        _write(repo / target, document)
        batch.append(entry)
    alive = [entry for entry in ledger["trials"] if entry["status"] != "failed"]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    stem = f"configs/experiments/search/{declared['study']}-g{generation}"
    _write(repo / f"{stem}.json", _campaign(base, digest, declared, alive, generation))
    controls = dict(
        schema_version=1,
        description=(
            f"Control de cada ensayo de la búsqueda {declared['study']} para el contraste "
            "pareado: el modelo base, del que solo lo separan los campos del espacio."
        ),
        controls={entry["model"]: declared["base_model"] for entry in alive},
    )
    _write(repo / f"{stem}-contrasts.json", controls)
    return batch


def _measure(state, declared, model):
    """IC de rangos por sesión de validación y de calibración de un trabajo terminado."""
    report = _report_module()
    run = state / "runs" / model / str(declared["year"]) / f"seed-{declared['run_seed']}"
    result = {}
    for split in SPLITS:
        data = report.load_split(run, split)
        if data is None:
            return None
        result[split] = report.session_ic(data)
    return result


def tell(folder, declared, state):
    """Lee del estado los ensayos terminados y los modelos conocidos. Devuelve los nuevos."""
    optuna, report = _optuna(), _report_module()
    folder, state = Path(folder), Path(state)
    study = _study(folder, declared)
    ledger = _ledger(folder)
    done = report.completed(state, declared["year"], declared["run_seed"])
    told, measured = [], {}
    for entry in ledger["trials"]:
        if entry["status"] != "asked" or entry["model"] not in done:
            continue
        sessions = _measure(state, declared, entry["model"])
        if sessions is None:
            continue
        measured[entry["model"]] = sessions
        summary = {split: report.summary(sessions[split]) for split in SPLITS}
        study.tell(entry["number"], summary["validation"]["mean"])
        entry.update(status="told", **summary)
        told.append(entry)
        _write(folder / "trials.json", ledger)
    for item in declared.get("known") or []:
        if item["model"] in ledger["known"] or item["model"] not in done:
            continue
        sessions = _measure(state, declared, item["model"])
        if sessions is None:
            continue
        measured[item["model"]] = sessions
        summary = {split: report.summary(sessions[split]) for split in SPLITS}
        study.add_trial(
            optuna.trial.create_trial(
                params=_params(declared["space"], item["values"]),
                distributions=_distributions(declared["space"]),
                value=summary["validation"]["mean"],
                user_attrs=dict(known=item["model"]),
            )
        )
        ledger["known"][item["model"]] = dict(values=item["values"], **summary)
        _write(folder / "trials.json", ledger)
    # El contraste con el modelo base, sesión a sesión, de los ensayos que aún no lo tienen.
    base = declared["base_model"]
    lacking = [
        entry
        for entry in ledger["trials"]
        if entry["status"] == "told" and "against_base" not in entry
    ]
    if lacking and base in done:
        control = measured.get(base) or _measure(state, declared, base)
        for entry in lacking if control is not None else []:
            sessions = measured.get(entry["model"]) or _measure(state, declared, entry["model"])
            if sessions is None:
                continue
            entry["against_base"] = {
                split: report.paired(sessions[split], control[split]) for split in SPLITS
            }
        _write(folder / "trials.json", ledger)
    return told


def fail(folder, declared, number, reason):
    """Da por fallido un ensayo pedido que no va a terminar: deja de contar como pendiente."""
    optuna = _optuna()
    folder = Path(folder)
    study = _study(folder, declared)
    ledger = _ledger(folder)
    entry = next((one for one in ledger["trials"] if one["number"] == number), None)
    if entry is None or entry["status"] != "asked":
        raise ValueError(f"El ensayo {number} no está pedido y pendiente")
    study.tell(number, state=optuna.trial.TrialState.FAIL)
    entry.update(status="failed", reason=reason)
    _write(folder / "trials.json", ledger)
    return entry


def report(folder, declared):
    """Ensayos por validación, confirmación de los mejores en calibración y candidato."""
    module = _report_module()
    validate(declared)
    ledger = _ledger(Path(folder))
    told = [entry for entry in ledger["trials"] if entry["status"] == "told"]
    rows = sorted(told, key=lambda entry: (-entry["validation"]["mean"], entry["number"]))
    candidates = []
    for entry in rows[: declared["confirm"]]:
        against = (entry.get("against_base") or {}).get("calibration") or {}
        difference = against.get("mean")
        candidates.append(dict(entry, confirmed=None if difference is None else difference > 0))
    concordance = None
    if len(rows) >= 3:
        concordance = module.correlation(
            module.average_ranks(np.asarray([row["validation"]["mean"] for row in rows])),
            module.average_ranks(np.asarray([row["calibration"]["mean"] for row in rows])),
        )
    errors = [
        row["against_base"]["validation"]["se"]
        for row in rows
        if (row.get("against_base") or {}).get("validation", {}).get("se") is not None
    ]
    # Lo que el mejor de n ensayos sin efecto ganaría al modelo base solo por azar.
    chance = float(np.median(errors) * math.sqrt(2 * math.log(len(rows)))) if errors else None
    recommended = next((row["number"] for row in candidates if row["confirmed"]), None)
    return dict(
        schema_version=1,
        kind="campaign_search_report_v1",
        final_test_opened=False,
        study=declared["study"],
        base_model=declared["base_model"],
        year=declared["year"],
        run_seed=declared["run_seed"],
        space=declared["space"],
        told=len(told),
        pending=sum(entry["status"] == "asked" for entry in ledger["trials"]),
        failed=sum(entry["status"] == "failed" for entry in ledger["trials"]),
        known=ledger["known"],
        trials=rows,
        candidates=candidates,
        concordance=concordance,
        chance_maximum=chance,
        recommended=recommended,
    )


def _number(value, digits=4, sign=False):
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def _difference(entry, split):
    against = (entry.get("against_base") or {}).get(split)
    if not against or against.get("mean") is None:
        return "—"
    return f"{_number(against['mean'], sign=True)} ({_number(against['t'], 1, True)})"


def markdown(document):
    names = [item["name"] for item in document["space"]]
    short = [name.rsplit(".", 1)[-1] for name in names]
    labels = short if len(set(short)) == len(short) else names
    lines = [
        f"# Búsqueda {document['study']}",
        "",
        f"Origen {document['year']}, semilla {document['run_seed']}. Se ordena por el IC de "
        "rangos de validación, que es lo único que ve la búsqueda. La calibración confirma. "
        "Solo se leen esos dos tramos.",
        "",
        f"Ensayos medidos: {document['told']}. Pendientes: {document['pending']}. "
        f"Fallidos: {document['failed']}.",
        "",
        f"| Ensayo | Tanda | {' | '.join(labels)} | IC de validación | Δ validación (t) "
        "| IC de calibración | Δ calibración (t) |",
        f"| --- | --- | {' | '.join('---' for _ in labels)} | --- | --- | --- | --- |",
    ]
    for row in document["trials"]:
        values = " | ".join(f"{row['values'][name]:g}" for name in names)
        lines.append(
            f"| {row['number']} | {row['generation']} | {values} "
            f"| {_number(row['validation']['mean'])} | {_difference(row, 'validation')} "
            f"| {_number(row['calibration']['mean'])} | {_difference(row, 'calibration')} |"
        )
    for model, entry in document["known"].items():
        values = " | ".join(f"{entry['values'][name]:g}" for name in names)
        lines.append(
            f"| {model} | conocido | {values} | {_number(entry['validation']['mean'])} | — "
            f"| {_number(entry['calibration']['mean'])} | — |"
        )
    lines += ["", "Las Δ son frente al modelo base, pareadas por sesión, con su t.", ""]
    if document["concordance"] is not None:
        lines.append(
            "Concordancia de rangos entre validación y calibración de los ensayos: "
            f"{_number(document['concordance'], 2, True)}."
        )
    if document["chance_maximum"] is not None:
        lines.append(
            f"El mejor de {document['told']} ensayos sin efecto ganaría al modelo base en "
            f"validación unos {_number(document['chance_maximum'])} solo por azar."
        )
    lines += ["", "## Candidatos", ""]
    for row in document["candidates"]:
        verdict = {True: "confirmado", False: "no confirmado", None: "sin modelo base medido"}
        lines.append(
            f"- Ensayo {row['number']}: validación {_number(row['validation']['mean'])}, "
            f"calibración frente al base {_difference(row, 'calibration')}: "
            f"{verdict[row['confirmed']]}."
        )
    if document["recommended"] is None:
        lines += ["", "Ningún candidato confirmado en calibración: se mantiene el modelo base."]
    else:
        lines += [
            "",
            f"Candidato: el ensayo {document['recommended']}, el mejor en validación entre los "
            "confirmados en calibración. Antes de adoptarlo se repite con las otras semillas.",
        ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    orders = parser.add_subparsers(dest="order", required=True)
    for name in ("ask", "tell", "fail", "report"):
        order = orders.add_parser(name)
        order.add_argument("--study", type=Path, required=True, help="Carpeta de la búsqueda")
        order.add_argument("--declaration", type=Path, required=True)
        if name == "ask":
            order.add_argument("--repo", type=Path, required=True)
            order.add_argument("--count", type=int, required=True)
        elif name == "tell":
            order.add_argument("--state", type=Path, required=True, help="Estado de la campaña")
        elif name == "fail":
            order.add_argument("--trial", type=int, required=True)
            order.add_argument("--reason", required=True)
        else:
            order.add_argument("--output", type=Path, required=True, help="Carpeta nueva")
    args = parser.parse_args(argv)
    declared = json.loads(args.declaration.read_text(encoding="utf-8"))
    if args.order == "ask":
        if args.count < 1:
            parser.error("--count debe ser al menos 1")
        batch = ask(args.study, declared, args.repo, args.count)
        for entry in batch:
            print(f"ensayo {entry['number']}: {entry['model']} {json.dumps(entry['values'])}")
        print(f"tanda {batch[0]['generation']}: tier search_g{batch[0]['generation']}")
    elif args.order == "tell":
        told = tell(args.study, declared, args.state)
        for entry in told:
            print(f"ensayo {entry['number']}: validación {entry['validation']['mean']:.4f}")
        print(f"ensayos nuevos medidos: {len(told)}")
    elif args.order == "fail":
        entry = fail(args.study, declared, args.trial, args.reason)
        print(f"ensayo {entry['number']} dado por fallido: {entry['reason']}")
    else:
        if args.output.exists():
            parser.error("La carpeta de salida ya existe")
        document = report(args.study, declared)
        _write(args.output / "search.json", document)
        table = markdown(document)
        (args.output / "search.md").write_text(table, encoding="utf-8")
        print(table, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
