"""Órganos de memoria: la memoria por activo y por vecinos emitidas fila a fila sobre un modelo.

El estudio de memorias (`posttraining_pilots.memory_study`) mide cuánto suman la memoria por
activo, la memoria por vecinos y su combinación sobre las emisiones de un modelo. Los órganos
emiten esas lecturas como un modelo más, con dos brazos por modelo y semilla:

- `combined`: la puntuación z + κₐ·u_activo + κᵥ·u_vecinos, con la semivida y los dos pesos
  elegidos por el IC de la calibración del propio origen;
- `asset_lab`: la memoria por activo fija del laboratorio (h = 21, κ = 0,1), sin ruido de
  selección.

La lectura pasa a unidades de retorno con la dispersión de la predicción en su sesión: la
predicción y sus cinco cuantiles se mueven lo mismo (σ_sesión · lectura), así que el orden de cada
sesión es el de la puntuación que midió el estudio y los cuantiles no se cruzan. Se emiten la
calibración y la evaluación (la validación no: los pesos se eligen con la calibración, que es
posterior) en una raíz propia con la forma de un estado de campaña,
`runs/<modelo>__<brazo>/<origen>/seed-<s>/` y `queue/`, que `dgx_screening_report.py` lee como
un estado más (`--reference <raíz> --decision-split calibration`). Los placebos se miden y se
informan en `studies/`; no se emiten.

Nada se escribe dentro del estado de la campaña. Las claves de la memoria por vecinos salen de la
vista del origen (`view_keys`), se construyen una vez por vista en un proceso aparte y se
comprueban contra la vista de los trabajos y contra las huellas de su recibo.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.evaluation import posttraining_pilots as pilots
from mars_titan.memory import neighbour_residual_memory as nrm
from mars_titan.memory import view_keys as vk

KIND = "posttraining_memory_organs_v1"
RUN_KIND = "posttraining_memory_organ_run_v1"
SCHEMA_VERSION = 1
SOURCE = "view_memory_keys_v1"
ARMS = ("combined", "asset_lab")
EMITTED = ("calibration", "evaluation")
DECLARATION = Path("configs") / "experiments" / "posttraining-memory-organs-v1.json"
SYSTEMD_RUN = "/usr/bin/systemd-run"
_FIELDS = {
    "schema_version",
    "kind",
    "final_test_access",
    "description",
    "min_assets",
    "models",
    "inputs",
    "asset",
    "neighbours",
    "combination",
    "placebo_seed",
}
_LABEL = re.compile(r"[a-z0-9][a-z0-9_.-]*")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_organs(document):
    """Validar la declaración de los órganos antes de leer ninguna emisión."""
    _require(isinstance(document, dict), "La declaración de los órganos es un objeto JSON")
    _require(
        document.get("kind") == KIND and document.get("schema_version") == SCHEMA_VERSION,
        f"La declaración debe ser {KIND} con schema_version {SCHEMA_VERSION}",
    )
    _require(
        set(document) == _FIELDS, f"Los órganos declaran exactamente los campos {sorted(_FIELDS)}"
    )
    _require(document["final_test_access"] is False, "Los órganos no abren el test final")
    minimum = pilots.screening().MIN_ASSETS
    _require(
        document["min_assets"] == minimum,
        f"El mínimo de activos por sesión es el del informe de criba ({minimum})",
    )
    return pilots.checked_memories(document, SOURCE, "la vista de cada origen")


# --- estados exportados ---------------------------------------------------------------------------


def _plain(value, arrays):
    if isinstance(value, np.ndarray):
        name = f"a{len(arrays)}"
        arrays[name] = value
        return {"__array__": name}
    if isinstance(value, dict):
        _require(
            all(isinstance(key, str) and key != "__array__" for key in value),
            "Un estado exportado solo tiene claves de texto",
        )
        return {key: _plain(item, arrays) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item, arrays) for item in value]
    _require(
        value is None or isinstance(value, bool | int | float | str),
        f"Un estado exportado no puede guardar un {type(value).__name__}",
    )
    return value


def save_state(payload, path):
    """Guardar el estado exportado de una memoria (con su huella) en un .npz sin objetos."""
    path = Path(path)
    arrays = {}
    structure = json.dumps(_plain(payload, arrays), sort_keys=True, allow_nan=False)
    arrays["structure"] = np.frombuffer(structure.encode(), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    try:
        with open(temporary, "wb") as handle:
            np.savez(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_state(path):
    """Leer un estado guardado con `save_state`; la memoria comprueba su huella al restaurarlo."""
    with np.load(path, allow_pickle=False) as data:

        def build(value):
            if isinstance(value, dict):
                if set(value) == {"__array__"}:
                    return np.array(data[value["__array__"]])
                return {key: build(item) for key, item in value.items()}
            if isinstance(value, list):
                return [build(item) for item in value]
            return value

        return build(json.loads(bytes(data["structure"]).decode()))


# --- emisión --------------------------------------------------------------------------------------


def shifted(sessions, prediction, quantiles, dispersion, signal):
    """La predicción y sus cinco cuantiles movidos por la lectura del brazo en unidades de retorno.

    Cada fila se mueve σ·lectura, con σ la desviación poblacional de la predicción en su sesión:
    el orden de la sesión es el de z + lectura, el que midió el estudio. Una sesión con la
    predicción constante (σ = 0) no admite una lectura distinta de cero y falla con su nombre.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    dispersion = np.asarray(dispersion, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    for market, instant, start, end in sessions:
        _require(
            bool(np.all(dispersion[start:end] > 0)) or not np.any(signal[start:end]),
            f"La predicción es constante en la sesión {market} {instant} y el brazo la movería:"
            " su orden no cabe en unidades de retorno",
        )
    return prediction + dispersion * signal, pilots.shift_quantiles(quantiles, dispersion, signal)


def _temporary(path):
    """Un temporal propio junto a su destino: dos procesos no comparten nunca el mismo nombre."""
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


def _atomic_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    try:
        temporary.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parent(state, model, origin, seed):
    return state / "runs" / model / str(origin) / f"seed-{seed}"


def _finished(folder):
    """El recibo de un trabajo terminado con sus tres tramos emitidos, o None."""
    complete = (folder / "run.json").is_file() and all(
        (folder / f"{name}-predictions.parquet").is_file() for name in pilots.SPLITS
    )
    if not complete:
        return None
    record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    return record if record.get("status") == "completed" else None


def _claim(output, declaration, digest):
    """Una raíz de órganos pertenece a una sola declaración."""
    saved = output / "declaration.json"
    if saved.is_file():
        previous = json.loads(saved.read_text(encoding="utf-8"))
        _require(
            hashlib.sha256(pilots._canonical(previous).encode()).hexdigest() == digest,
            f"{output} pertenece a otra declaración",
        )
    else:
        _atomic_json(saved, declaration)


def _keys(manifest, view, root, log):
    """Las claves de la vista (construidas en un proceso aparte si faltan) y su recibo."""
    folder = root / view
    if not folder.exists():
        partial = folder.with_name(folder.name + ".tmp")
        if partial.exists():
            log(f"claves de {view[:8]}: se retira una construcción a medias")
            shutil.rmtree(partial)
        root.mkdir(parents=True, exist_ok=True)
        log(f"claves de {view[:8]}: se construyen desde {manifest}")
        source = str(Path(vk.__file__).resolve().parents[2])
        paths = [source, *filter(None, [os.environ.get("PYTHONPATH")])]
        command = [sys.executable, "-m", "mars_titan.memory.view_keys", "construir"]
        command += ["--manifest", str(manifest), "--output", str(folder)]
        # Un proceso aparte devuelve su memoria al terminar: la de los órganos tiene tope.
        done = subprocess.run(
            command, env=dict(os.environ, PYTHONPATH=os.pathsep.join(paths)), check=False
        )
        _require(
            done.returncode == 0,
            f"No se pudieron construir las claves de {manifest} en {folder}"
            f" (el proceso terminó con código {done.returncode})",
        )
    receipt = json.loads((folder / "keys.json").read_text(encoding="utf-8"))
    _require(
        receipt.get("kind") == vk.KIND and receipt.get("view_sha256") == view,
        f"Las claves de {folder} son de otra vista ({receipt.get('view_sha256')} ≠ {view})",
    )
    return folder, receipt


def emit(declaration, state, origin, seeds, keys, output, *, threads=1, log=print, maturity=None):
    """Emitir los dos brazos de cada modelo declarado en un origen, semilla a semilla.

    `keys` es la carpeta de claves por vista y `output` la raíz propia de los órganos; ninguna
    puede estar dentro del estado. Un modelo sin trabajo terminado se anota y se emiten los demás.
    Una semilla ya emitida (con su informe en `studies/`) no se rehace, salvo que haya terminado
    un modelo que faltaba: entonces se emite entera otra vez. `maturity` lee la
    madurez de las etiquetas de la vista (por omisión, el lector de la campaña).
    """
    declaration = checked_organs(declaration)
    state, keys, output = (Path(path).resolve() for path in (state, keys, output))
    _require(state.is_dir(), f"No está el estado {state}")
    for path, what in ((output, "La raíz de los órganos"), (keys, "La carpeta de claves")):
        _require(
            not path.is_relative_to(state),
            f"{what} {path} queda dentro del estado {state}: los órganos no escriben en él",
        )
    _require(type(origin) is int and 2000 <= origin <= 2023, "El origen es un año de 2000 a 2023")
    seeds = list(seeds)
    _require(
        seeds and all(type(seed) is int for seed in seeds) and len(set(seeds)) == len(seeds),
        "Las semillas son enteros distintos",
    )
    digest = hashlib.sha256(pilots._canonical(declaration).encode()).hexdigest()
    _claim(output, declaration, digest)
    context = dict(
        declaration=declaration,
        digest=digest,
        state=state,
        origin=origin,
        manifest=state / "cohorts" / str(origin) / "manifest.json",
        keys=keys,
        output=output,
        maturity=pilots._maturity if maturity is None else maturity,
        threads=threads,
        log=log,
        views={},
    )
    runs = [_emit_seed(context, seed) for seed in seeds]
    return dict(origin=origin, state=str(state), output=str(output), runs=runs)


def _emit_seed(context, seed):
    started = time.monotonic()
    declaration, state, origin = context["declaration"], context["state"], context["origin"]
    output, log = context["output"], context["log"]
    stem = f"{origin}--seed-{seed}"
    report_path = output / "studies" / f"{stem}.json"
    records = {}
    for model in declaration["models"]:
        record = _finished(_parent(state, model, origin, seed))
        if record is not None:
            records[model] = record
    if report_path.is_file():
        done = json.loads(report_path.read_text(encoding="utf-8"))
        finished = [model for model in done["missing_models"] if model in records]
        if not finished:
            log(f"{stem}: ya emitido")
            return dict(
                seed=seed,
                models=list(done["models"]),
                missing=done["missing_models"],
                skipped=True,
                seconds=0.0,
            )
        # Una semilla emitida a medias se emite entera otra vez en cuanto termina un modelo que
        # faltaba: la lectura por vecinos es conjunta y el informe nuevo sustituye al anterior.
        log(f"{stem}: ya terminó {', '.join(finished)}, que faltaba; se emite de nuevo")
    present = [model for model in declaration["models"] if model in records]
    missing = [model for model in declaration["models"] if model not in records]
    _require(
        present,
        f"Ningún modelo declarado tiene terminado su trabajo de {origin} con la semilla {seed}"
        f" en {state}",
    )
    if missing:
        log(f"{stem}: sin trabajo terminado {', '.join(missing)}; se anotan y se emiten los demás")
    views = {records[model]["request"]["view_sha256"] for model in present}
    _require(
        len(views) == 1,
        f"Los trabajos de {origin} con la semilla {seed} no comparten vista: {sorted(views)}",
    )
    view = views.pop()
    if view not in context["views"]:
        tables = context["maturity"](context["manifest"], view)
        context["views"][view] = (tables, *_keys(context["manifest"], view, context["keys"], log))
    tables, folder, receipt = context["views"][view]
    reference, predictions, emissions = None, {}, {}
    for model in present:
        parent = _parent(state, model, origin, seed)
        splits, emissions[model] = {}, {}
        for name in pilots.SPLITS:
            path = parent / f"{name}-predictions.parquet"
            emissions[model][name] = pilots._sha256(path)
            splits[name] = pilots.load_split(pq.read_table(path), tables[name], name)
        reference = reference or splits
        for name in pilots.SPLITS:
            same = all(
                np.array_equal(getattr(splits[name], field), getattr(reference[name], field))
                for field in ("market", "asset", "prediction_at", "target")
            )
            _require(same, f"Las filas de {name} de {model} no son las de {present[0]}")
        predictions[model] = np.concatenate([splits[name].prediction for name in pilots.SPLITS])
        log(f"{stem}: cargado {model}")
    assets = np.concatenate([reference[name].asset for name in pilots.SPLITS])
    days = np.concatenate([reference[name].prediction_at for name in pilots.SPLITS])
    raw, files = pilots.lab_inputs(folder, folder / "names.json", assets, days // pilots.DAY_US)
    changed = sorted(name for name, value in receipt["outputs"].items() if files.get(name) != value)
    _require(
        not changed and set(files) == set(receipt["outputs"]),
        f"Las claves de {folder} no coinciden con su recibo: {', '.join(changed) or 'archivos'}",
    )
    inputs = declaration["inputs"]
    _require(
        raw.shape[1] == inputs["columns"],
        f"Las claves de {folder} no tienen {inputs['columns']} columnas",
    )
    matrix = nrm.projection(raw.shape[1], inputs["projection_dim"], seed=inputs["projection_seed"])
    keys = nrm.keys_from_inputs(raw, matrix)
    del raw
    results, table, organs = pilots.memory_study(
        reference,
        predictions,
        keys,
        declaration,
        threads=context["threads"],
        log=log,
        organs=True,
    )
    del keys
    bounds = np.cumsum([0] + [reference[name].rows for name in pilots.SPLITS])
    position = {name: slice(bounds[i], bounds[i + 1]) for i, name in enumerate(pilots.SPLITS)}
    near_file = Path("states") / str(origin) / f"seed-{seed}" / "neighbours.npz"
    save_state(organs["neighbours"], output / near_file)
    near = dict(
        file=str(near_file),
        sha256=pilots._sha256(output / near_file),
        state_sha256=organs["neighbours"]["state_sha256"],
        models=list(organs["order"]),
    )
    shared = dict(
        view=view,
        keys=dict(folder=str(folder), view_sha256=view, files=files),
        near=near,
        position=position,
        reference=reference,
        results=results,
        seconds=time.monotonic() - started,
    )
    for model in present:
        for name in ARMS:
            _write_arm(context, seed, model, name, records[model], emissions[model], organs, shared)
    report = dict(
        schema_version=SCHEMA_VERSION,
        kind=f"{KIND}_report",
        final_test_opened=False,
        declaration_sha256=context["digest"],
        state=str(state),
        origin=origin,
        seed=seed,
        view_sha256=view,
        arms=list(ARMS),
        models=results,
        missing_models=missing,
        inputs=dict(emissions=emissions, keys=shared["keys"]),
        rows={name: reference[name].rows for name in pilots.SPLITS},
        sessions={name: len(reference[name].sessions) for name in pilots.SPLITS},
        neighbours=near,
        seconds=round(time.monotonic() - started, 1),
    )
    (output / "studies").mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output / "studies" / f"{stem}.parquet")
    # El informe va el último: con él la semilla queda emitida.
    _atomic_json(report_path, report)
    log(f"{stem}: emitido en {report['seconds']:.0f} s")
    return dict(
        seed=seed, models=present, missing=missing, skipped=False, seconds=report["seconds"]
    )


def _write_arm(context, seed, model, name, record, emissions, organs, shared):
    state, origin, output = context["state"], context["origin"], context["output"]
    organ = organs["models"][model]
    chosen = organ["arms"][name]
    label = f"{model}__{name}"
    queue = output / "queue" / f"{label}--{origin}--seed-{seed}.json"
    # Sin registro en la cola, el informe de criba no lee una emisión a medias.
    queue.unlink(missing_ok=True)
    folder = output / "runs" / label / str(origin) / f"seed-{seed}"
    temporary = folder.with_name(folder.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    parent = _parent(state, model, origin, seed)
    outputs = {}
    for split in EMITTED:
        path = parent / f"{split}-predictions.parquet"
        _require(pilots._sha256(path) == emissions[split], f"{path} cambió mientras se medía")
        table = pq.read_table(path)
        _require(
            all(column in table.column_names for column in pilots.QUANTILE_COLUMNS),
            f"{path} no tiene los cinco cuantiles",
        )
        order = pilots.emission_order(table)[3]
        rows = shared["position"][split]
        moved, quantiles = shifted(
            shared["reference"][split].sessions,
            table["prediction"].to_numpy()[order],
            np.column_stack([table[c].to_numpy() for c in pilots.QUANTILE_COLUMNS])[order],
            organ["dispersion"][rows],
            chosen["signal"][rows],
        )
        for column, values in (
            ("prediction", moved),
            *zip(pilots.QUANTILE_COLUMNS, quantiles.T, strict=True),
        ):
            index = table.schema.get_field_index(column)
            field = table.schema.field(index)
            restored = np.empty(len(values))
            restored[order] = values
            table = table.set_column(index, field, pa.array(restored, type=field.type))
        pq.write_table(table, temporary / f"{split}-predictions.parquet")
        outputs[split] = pilots._sha256(temporary / f"{split}-predictions.parquet")
    save_state(chosen["state"], temporary / "asset-memory.npz")
    states = dict(
        asset=dict(
            file=str(folder.relative_to(output) / "asset-memory.npz"),
            sha256=pilots._sha256(temporary / "asset-memory.npz"),
            state_sha256=chosen["state"]["state_sha256"],
        )
    )
    if name == "combined":
        states["neighbours"] = shared["near"]
    receipt = dict(
        schema_version=SCHEMA_VERSION,
        kind=RUN_KIND,
        status="completed",
        model=label,
        arm=name,
        parent=dict(model=model, state=str(state), run_id=record.get("run_id")),
        origin=origin,
        seed=seed,
        request=dict(view_sha256=shared["view"], window=record["request"].get("window"), seed=seed),
        selection=chosen["selection"],
        ic={split: shared["results"][model][split][name] for split in EMITTED},
        inputs=dict(declaration_sha256=context["digest"], emissions=emissions, keys=shared["keys"]),
        outputs=outputs,
        state=states,
        code=dict(
            memory_organs=pilots._sha256(Path(__file__)),
            posttraining_pilots=pilots._sha256(Path(pilots.__file__)),
        ),
        final_test_opened=False,
    )
    _atomic_json(temporary / "run.json", receipt)
    if folder.exists():
        context["log"](f"{label}: se rehace una emisión a medias")
        shutil.rmtree(folder)
    os.replace(temporary, folder)
    record = dict(
        status="completed",
        model=label,
        tier="memory_organ",
        year=origin,
        seed=seed,
        parent=model,
        arm=name,
        attempts=[dict(seconds=round(shared["seconds"], 1))],
        result=dict(best_epoch=None, best_score=None),
    )
    _atomic_json(queue, record)


# --- la orden del relevo -------------------------------------------------------------------------


def relay_line(
    *, label, state, origin, seeds, tree, python, keys, output, declaration=None, threads=3
):
    """La línea `orden|…` del plan del relevo que emite los órganos de un origen.

    systemd-run sin --wait arranca la unidad y vuelve enseguida: el relevo sigue con la fase
    siguiente sin esperar a la memoria y un fallo de la memoria no lo detiene (queda en el diario
    de la unidad, `journalctl --user -u mars-titan-memorias-<estado>-<origen>`). El relevo parte
    los argumentos por espacios, así que ninguna ruta puede llevarlos.
    """
    declaration = Path(tree) / DECLARATION if declaration is None else Path(declaration)
    _require(
        isinstance(label, str) and _LABEL.fullmatch(label) is not None,
        f"La etiqueta {label!r} del estado va en el nombre de la unidad: minúsculas, cifras, - _ .",
    )
    _require(type(origin) is int and 2000 <= origin <= 2023, "El origen es un año de 2000 a 2023")
    seeds = list(seeds)
    _require(
        seeds and all(type(seed) is int for seed in seeds) and len(set(seeds)) == len(seeds),
        "Las semillas son enteros distintos",
    )
    paths = dict(
        state=state, tree=tree, python=python, keys=keys, output=output, declaration=declaration
    )
    for name, path in paths.items():
        text = str(path)
        _require(
            Path(text).is_absolute() and not any(c.isspace() or c == "|" for c in text),
            f"La ruta {name} ({text}) debe ser absoluta y sin espacio ni '|': el relevo parte la"
            " línea por '|' y los argumentos por espacios",
        )
    words = ["--user", "--collect", "--unit", f"mars-titan-memorias-{label}-{origin}"]
    words += ["--slice=mars-titan.slice", "-p", "CPUWeight=10", "-p", "Nice=15"]
    words += ["-p", "MemoryMax=8G", "-p", "MemorySwapMax=0", f"--working-directory={tree}"]
    for variable in (
        f"PYTHONPATH={Path(tree) / 'src'}",
        "PYTHONUNBUFFERED=1",
        "CUDA_VISIBLE_DEVICES=",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
    ):
        words += ["-E", variable]
    words += [str(python), "-m", "mars_titan.evaluation.memory_organs", "emitir"]
    words += ["--declaration", str(declaration), "--state", str(state), "--origin", str(origin)]
    for seed in seeds:
        words += ["--seed", str(seed)]
    words += ["--keys", str(keys), "--output", str(output), "--threads", str(threads)]
    return f"orden|{SYSTEMD_RUN}|{' '.join(words)}"


def parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    emitting = commands.add_parser("emitir", help="Emitir los órganos de un origen")
    emitting.add_argument("--declaration", type=Path, required=True)
    emitting.add_argument("--state", type=Path, required=True, help="Estado de la campaña")
    emitting.add_argument("--origin", type=int, required=True)
    emitting.add_argument("--seed", type=int, action="append", required=True)
    emitting.add_argument("--keys", type=Path, required=True, help="Carpeta de claves por vista")
    emitting.add_argument("--output", type=Path, required=True, help="Raíz propia de los órganos")
    emitting.add_argument("--threads", type=int, default=3)
    line = commands.add_parser("linea", help="Línea del plan del relevo para un origen")
    line.add_argument("--label", required=True, help="Nombre corto del estado (v11)")
    line.add_argument("--state", type=Path, required=True)
    line.add_argument("--origin", type=int, required=True)
    line.add_argument("--seed", type=int, action="append", required=True)
    line.add_argument("--tree", type=Path, required=True, help="Copia fija del repositorio")
    line.add_argument("--python", type=Path, required=True)
    line.add_argument("--keys", type=Path, required=True)
    line.add_argument("--output", type=Path, required=True)
    line.add_argument("--declaration", type=Path)
    line.add_argument("--threads", type=int, default=3)
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "linea":
        options = dict(vars(args), seeds=args.seed)
        for name in ("command", "seed"):
            options.pop(name)
        print(relay_line(**options))
        return 0

    def log(line):
        print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {line}", flush=True)

    declaration = json.loads(args.declaration.read_text(encoding="utf-8"))
    summary = emit(
        declaration,
        args.state,
        args.origin,
        args.seed,
        args.keys,
        args.output,
        threads=args.threads,
        log=log,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
