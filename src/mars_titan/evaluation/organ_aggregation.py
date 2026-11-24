"""Órganos de memoria como componente del agregador causal: S1 y S2 de un origen.

Son los contrastes secundarios S1 y S2 del registro confirmatorio. Se añadieron el 22 de
noviembre de 2026 a las 20:05 UTC, antes de ajustar ningún origen limpio.
- **S1**: la media de rangos de `titans_mac_nogain` y `lightgbm_technical` frente al mejor de
  los dos. El mejor es el de más IC en los tramos de decisión del origen (validación y
  calibración), y se usa el mismo en las dos medidas.
- **S2**: `ridge_cov_active` sobre esos dos modelos y la señal del brazo combinado de los órganos
  de `titans_mac_nogain`, frente a la media de rangos de S1.
  - `ridge_cov_active` es una regresión contraída con la covarianza de la sesión y conjunto
    activo; λ y la semivida se eligen en calibración.
  - La señal es la emisión del brazo menos la predicción del padre.
  - Placebo: la misma señal barajada entre activos dentro de cada sesión.

Cada contraste se mide en la etiqueta y en lo operable. Lo operable compara k·p + α con y + α,
con k elegido en calibración para cada componente. Las diferencias son de evaluación, pareadas
por sesión, con el t HAC de la criba.

La configuración no cambia después de registrarse:
- componentes, tramos, rejillas, factores y semillas están en la declaración;
- la declaración debe tener la huella registrada;
- el agregador del código debe usar la rejilla declarada.
El agregador aprende en línea con la madurez exacta de la vista. La evaluación no interviene en
ninguna elección.

Las entradas las escriben otras unidades del relevo: el padre, los árboles y los órganos.
- Si falta alguna, la orden espera hasta `--wait-hours`. Si siguen faltando, sale con el código
  75 y la lista de lo que falta, sin escribir nada.
- Calcula en una carpeta temporal y la publica de una vez como `<output>/<origen>`.
- Un origen ya calculado con la misma declaración y las mismas entradas no se repite.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mars_titan.evaluation import online_aggregator as oa
from mars_titan.evaluation import posttraining_pilots as pilots

KIND = "posttraining_organ_aggregation_v1"
# Huella canónica de configs/experiments/posttraining-aggregator-s2.json, la declaración de S2.
REGISTERED_SHA256 = "e00daa98e23c120904aeb13fc3ef58cb9019521f49c1eed1940f05b6d83ed15f"
# Variantes registradas: la misma configuración con otros nombres de modelo. La prueba comprueba
# campo a campo que solo cambian el padre, los árboles, el órgano y la descripción.
VARIANT_SHA256 = {
    # Brazo v3 de H9 (rasgos long_horizon_v3): los modelos llevan el sufijo _v3.
    "110ea6591a5543d48b1f5adf33c1d0b01efde0e7e4311faed07445a4bf1c8032": (
        "posttraining-aggregator-s2-v3.json"
    ),
}
MISSING_EXIT = 75  # EX_TEMPFAIL: faltan entradas de otras unidades y la orden se puede repetir
SPLITS = ("validation", "calibration", "evaluation")
AGGREGATED = ("calibration", "evaluation")
ORDER = [("market", "ascending"), ("prediction_at", "ascending"), ("sample_id", "ascending")]
COLUMNS = ["sample_id", "asset_id", "market", "prediction_at", "target", "prediction"]
KEYS = ("sample_id", "asset_id", "market", "prediction_at", "target")
CODE = Path(__file__).resolve().parents[3]
SCREENING = CODE / "scripts" / "dgx_screening_report.py"
SIGNAL, PLACEBO = "senal_organo", "senal_barajada"
CONFIGS = dict(
    dos=("parent", "tree"),
    senal=("parent", "tree", "signal"),
    placebo=("parent", "tree", "placebo"),
)
MODE_NAMES = dict(label="etiqueta", operable="operable")


class MissingInputs(Exception):
    """Entradas de otras unidades que todavía no están."""

    def __init__(self, items):
        super().__init__(f"Faltan {len(items)} entradas")
        self.items = list(items)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _screening():
    _require(SCREENING.is_file(), f"Falta la criba {SCREENING} junto al código")
    spec = importlib.util.spec_from_file_location("criba_de_s2", SCREENING)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def checked_declaration(document):
    """La declaración registrada de S2 y su huella, o un error que dice por qué no lo es."""
    _require(isinstance(document, dict), "La declaración de S2 es un objeto JSON")
    digest = hashlib.sha256(pilots._canonical(document).encode()).hexdigest()
    _require(
        digest == REGISTERED_SHA256 or digest in VARIANT_SHA256,
        f"La declaración no es la registrada para S2 (huella {digest[:12]}, registrada "
        f"{REGISTERED_SHA256[:12]} o una de sus variantes con otros nombres de modelo): la "
        "configuración de un contraste confirmatorio no cambia",
    )
    grid = document["aggregator"]
    _require(
        tuple(grid["lambdas"]) == tuple(oa.RIDGE_LAMBDAS)
        and tuple(grid["half_lives"]) == tuple(oa.RIDGE_HALF_LIVES),
        "El agregador del código no usa la rejilla de ridge_cov_active declarada "
        f"(usa λ {oa.RIDGE_LAMBDAS} y semividas {oa.RIDGE_HALF_LIVES})",
    )
    return document, digest


def folders(declaration, state, reference, organs, origin):
    """Carpetas del padre, de los árboles y del brazo combinado de los órganos de un origen."""
    seed = f"seed-{declaration['seed']}"
    return dict(
        parent=Path(state) / "runs" / declaration["parent"] / str(origin) / seed,
        tree=Path(reference) / "runs" / declaration["tree"] / str(origin) / seed,
        organ=Path(organs) / "runs" / declaration["organ"] / str(origin) / seed,
    )


def _needed(role):
    return AGGREGATED if role == "organ" else SPLITS


def _readable(path):
    try:
        pq.read_metadata(path)
    except (OSError, pa.ArrowException):
        return False
    return True


def missing_inputs(declaration, sources, manifest):
    """Lo que falta todavía de las otras unidades; una lista vacía cuando está todo."""
    missing = [] if manifest.is_file() else [f"el manifiesto de la cohorte {manifest}"]
    for role, folder in sources.items():
        record = folder / "run.json"
        if not record.is_file():
            missing.append(f"{role}: {record}")
            continue
        if role != "tree":
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
            if not _readable(folder / f"{split}-predictions.parquet")
        ]
    return missing


def wait_for_inputs(declaration, sources, manifest, *, hours, poll, sleep=time.sleep, say=print):
    """Espera a que estén todas las entradas, como mucho `hours`; devuelve lo que falte."""
    deadline = time.monotonic() + hours * 3600
    while True:
        missing = missing_inputs(declaration, sources, manifest)
        if not missing or time.monotonic() >= deadline:
            return missing
        say(f"esperando {len(missing)} entradas; la primera: {missing[0]}")
        sleep(poll)


def _fingerprint(sources, manifest):
    files = {"manifest": manifest}
    for role, folder in sources.items():
        files[f"{role}/run.json"] = folder / "run.json"
        for split in _needed(role):
            files[f"{role}/{split}"] = folder / f"{split}-predictions.parquet"
    return {name: dict(path=str(path), sha256=pilots._sha256(path)) for name, path in files.items()}


def _code_identity():
    files = dict(
        organ_aggregation=Path(__file__),
        online_aggregator=Path(oa.__file__),
        posttraining_pilots=Path(pilots.__file__),
        screening=SCREENING,
        technical_features=CODE / "src" / "mars_titan" / "data" / "technical_features.py",
    )
    return dict(
        root=str(CODE),
        files={name: pilots._sha256(path) for name, path in files.items() if path.is_file()},
    )


def _table(folder, split):
    path = folder / f"{split}-predictions.parquet"
    return pq.read_table(path, columns=COLUMNS).sort_by(ORDER)


def checked_factors(manifest_path, markets):
    """Cada mercado medido declara su factor de mercado, y el fichero está y conserva su huella.

    Sin él, `KnownOffset` deja el alfa en NaN sin avisar (hallazgo D-7: el CSI 300 de China no está
    en la DGX ni en este equipo) y lo operable pierde ese mercado en silencio.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    declared = manifest.get("market_factors") or {}
    for market in sorted(markets):
        entry = declared.get(market) or {}
        path = entry.get("prices_path")
        _require(
            bool(path) and bool(entry.get("prices_sha256")),
            f"La vista {manifest_path} no declara el factor de mercado de {market}: no hay alfa "
            "conocido; mide los otros mercados con --market",
        )
        _require(
            Path(path).is_file(),
            f"No está el factor de mercado de {market} que declara la vista ({path}): no hay alfa "
            "conocido; mide los otros mercados con --market",
        )
        _require(
            pilots._sha256(Path(path)) == entry["prices_sha256"],
            f"El factor de mercado de {market} ({path}) no conserva la huella de la vista",
        )


def markets_of(tables):
    """Mercados con filas en unas tablas, ordenados."""
    found = set()
    for table in tables.values():
        found.update(table.column("market").unique().to_pylist())
    return sorted(found)


def measured_markets(present, wanted):
    """Mercados que se miden: los pedidos con --market, o el único del estado.

    En un estado conjunto la lectura es por mercado: sin --market no se mezclan.
    """
    if wanted:
        missing = sorted(set(wanted) - set(present))
        _require(not missing, f"El estado no tiene filas de {missing} (tiene {present})")
        return sorted(set(wanted))
    _require(
        len(present) == 1,
        f"El estado tiene filas de {present}: la lectura es por mercado, indica --market "
        "(por ejemplo --market US)",
    )
    return list(present)


def only(table, markets):
    """Las filas de unos mercados, en el mismo orden."""
    return table.filter(pc.is_in(table.column("market"), value_set=pa.array(list(markets))))


def target_name(origin, markets):
    """Carpeta de un origen: `<año>` con el único mercado del estado; `<año>-<mercados>` si se
    piden."""
    return str(origin) if not markets else f"{origin}-{'+'.join(sorted(set(markets)))}"


def known_rows(state, origin, parent, view, tables, markets=None):
    """Alfa conocido y madurez exacta de cada fila del padre, por tramo y en el orden de las tablas.

    El alfa es el de la etiqueta (`KnownOffset` de la criba; NaN sin modelo de mercado). La
    madurez sale de la vista con que se ajustó el padre.
    """
    markets = list(markets or markets_of(tables))
    manifest = Path(state) / "cohorts" / str(origin) / "manifest.json"
    checked_factors(manifest, markets)
    known = _screening().KnownOffset(Path(state), origin)
    maturity = pilots._maturity(manifest, view)
    # Precalentar `known.assets` en paralelo (4 hilos) si es un KnownOffset real con muchos activos.
    if hasattr(known, "assets") and hasattr(known, "prepared") and hasattr(known, "module"):
        all_assets = sorted(
            {
                asset
                for table in tables.values()
                for asset in table.column("asset_id").unique().to_pylist()
            }
        )
        if len(all_assets) > 4:
            from concurrent.futures import ThreadPoolExecutor

            names_tf = known.module.NAMES
            idx_has = names_tf.index("has_market_model")
            idx_alp = names_tf.index("alpha_percent")
            idx_liq = names_tf.index("log_dollar_volume_21")

            def _warm(asset):
                mkt, sym = asset.split("/", 1)
                prices, days = known._table(known.prepared / mkt / sym / "prices.parquet")
                feats = known.module.technical_features(
                    prices, days, *known.factors.get(mkt, (None, None))
                )
                ok = feats[:, idx_has] > 0
                vals = feats[:, idx_alp].astype(np.float64) / 100
                liq = feats[:, idx_liq].astype(np.float64)
                return asset, (days, np.where(ok, vals, np.nan), liq)

            with ThreadPoolExecutor(max_workers=4) as pool:
                for asset, entry in pool.map(_warm, all_assets):
                    known.assets[asset] = entry
    alpha, matures = {}, {}
    for split, table in tables.items():
        every = pq.read_table(
            parent / f"{split}-predictions.parquet",
            columns=["sample_id", "market", "prediction_at"],
        ).sort_by(ORDER)
        keep = np.isin(np.asarray(every.column("market").to_pylist()), markets)
        _require(
            every.column("sample_id").filter(pa.array(keep)).equals(table.column("sample_id")),
            f"Las filas de {split} no son las del padre en {markets}",
        )
        alpha[split] = known.rows(parent, split)[0][keep]
        rows = maturity[split]
        lookup = dict(
            zip(
                zip(
                    rows.column("asset_id").to_pylist(),
                    rows.column("prediction_at").cast(pa.int64()).to_pylist(),
                    strict=True,
                ),
                rows.column("matures_at").cast(pa.int64()).to_pylist(),
                strict=True,
            )
        )
        keys = zip(
            table.column("asset_id").to_pylist(),
            table.column("prediction_at").cast(pa.int64()).to_pylist(),
            strict=True,
        )
        values = [lookup.get(key) for key in keys]
        _require(
            all(value is not None for value in values),
            f"Hay filas de {split} del padre sin madurez en la vista {view[:12]}",
        )
        matures[split] = np.asarray(values, dtype=np.int64)
    return alpha, matures


def shuffled_within_sessions(values, market, instant, seed):
    """La señal barajada entre activos dentro de cada sesión, con un generador por tramo."""
    rng = np.random.default_rng(seed)
    result = np.array(values, copy=True)
    change = np.r_[True, (market[1:] != market[:-1]) | (instant[1:] != instant[:-1])]
    starts = np.flatnonzero(change)
    for start, end in zip(starts, np.r_[starts[1:], len(values)], strict=True):
        result[start:end] = values[start:end][rng.permutation(end - start)]
    return result


def _checked_sources(declaration, origin, sources, records, fingerprint):
    tree, organ, parent = records["tree"], records["organ"], records["parent"]
    _require(
        parent.get("final_test_opened") is False and organ.get("final_test_opened") is False,
        f"El padre o el órgano de {origin} abrieron el test final",
    )
    _require(
        tree.get("model") == declaration["tree"]
        and tree.get("seed") == declaration["seed"]
        and tree.get("final_test_access") is False,
        f"{sources['tree']} no son los árboles declarados ({declaration['tree']}, semilla "
        f"{declaration['seed']}, sin test final)",
    )
    declared = organ.get("parent")
    _require(
        organ.get("model") == declaration["organ"]
        and organ.get("arm") == "combined"
        and organ.get("origin") == origin
        and organ.get("seed") == declaration["seed"]
        and isinstance(declared, dict)
        and declared.get("model") == declaration["parent"]
        and declared.get("run_id") == parent.get("run_id"),
        f"{sources['organ']} no es el brazo combinado de este ajuste de {declaration['parent']} "
        f"(origen {origin}, semilla {declaration['seed']})",
    )
    emissions = (organ.get("inputs") or {}).get("emissions") or {}
    for split in SPLITS:
        _require(
            emissions.get(split) == fingerprint[f"parent/{split}"]["sha256"],
            f"El órgano de {origin} se emitió con otras predicciones del padre ({split})",
        )
    for split in AGGREGATED:
        _require(
            (organ.get("outputs") or {}).get(split) == fingerprint[f"organ/{split}"]["sha256"],
            f"La emisión de {split} del órgano de {origin} no es la de su registro",
        )


def _decision(tool, folder, splits, markets):
    ics = {
        split: {
            key: value
            for key, value in tool.session_ic(tool.load_split(folder, split)).items()
            if key[0] in markets
        }
        for split in splits
    }
    return tool.summary(tool.pooled(ics, splits))["mean"]


def _mean(values):
    return float(np.mean(list(values.values()))) if values else None


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_component(folder, tables, prediction, alpha, matures):
    folder.mkdir(parents=True)
    for split in AGGREGATED:
        base = tables[split]
        table = pa.table(
            {
                **{name: base.column(name) for name in KEYS},
                "prediction": pa.array(prediction[split].astype(np.float32)),
                "alpha": pa.array(alpha[split].astype(np.float64)),
                "matures_at": pa.array(matures[split].astype(np.int64)),
            }
        )
        pq.write_table(table, folder / f"{split}-predictions.parquet")
    return folder


def _measure(
    declaration, tool, inputs, names, best, decision, mode, work, *, raw_by_split=None, cache=None
):
    runs = {}
    is_operable = mode == "operable"
    decl_splits = tuple(declaration["splits"])
    decl_scales = tuple(declaration["operable_scales"])
    for config, members in CONFIGS.items():
        runs[config] = oa.run_aggregation(
            {names[role]: inputs[role] for role in members},
            splits=decl_splits,
            selection_split=declaration["selection_split"],
            min_assets=declaration["min_assets"],
            align=declaration["align"],
            operable=is_operable,
            alpha_col="alpha",
            maturity_col="matures_at",
            placebo_seed=declaration["aggregator_placebo_seed"],
            operable_scales=decl_scales,
            aggregators=("rank_average", "ridge_cov_active"),
            include_placebos=(config == "senal"),
            include_asset_abstention=False,
            preloaded_raw=raw_by_split,
            precomputed_cache=cache,
        )
        report = oa.render_markdown(oa._json_safe_report(runs[config]))
        (work / "informes" / f"{mode}-{config}.md").write_text(report, encoding="utf-8")
    two = runs["dos"]
    evaluation = two["aligned"]["evaluation"]
    ics = {
        name: (
            cache["individual_ics"][name]["evaluation"]
            if cache is not None
            else oa.session_rank_ic(
                evaluation, evaluation["predictions"][:, i], min_assets=declaration["min_assets"]
            )
        )
        for i, name in enumerate(two["model_names"])
    }
    ics.update(
        s1=two["aggregators"]["rank_average"]["evaluation_session_ics"],
        ridge_two=two["aggregators"]["ridge_cov_active"]["evaluation_session_ics"],
        s2=runs["senal"]["aggregators"]["ridge_cov_active"]["evaluation_session_ics"],
        s2_placebo=runs["placebo"]["aggregators"]["ridge_cov_active"]["evaluation_session_ics"],
    )
    parent, tree = names["parent"], names["tree"]
    operable = {config: run["operable"] for config, run in runs.items()}
    result = dict(
        means={name: _mean(values) for name, values in ics.items()},
        s1=dict(
            best=names[best],
            decision_ic={names[role]: value for role, value in decision.items()},
            vs_best=tool.paired(ics["s1"], ics[names[best]]),
            vs_parent=tool.paired(ics["s1"], ics[parent]),
            vs_tree=tool.paired(ics["s1"], ics[tree]),
        ),
        s2=dict(
            vs_s1=tool.paired(ics["s2"], ics["s1"]),
            vs_placebo=tool.paired(ics["s2"], ics["s2_placebo"]),
            placebo_vs_s1=tool.paired(ics["s2_placebo"], ics["s1"]),
            ridge_two_vs_s1=tool.paired(ics["ridge_two"], ics["s1"]),
            vs_ridge_two=tool.paired(ics["s2"], ics["ridge_two"]),
            weights=runs["senal"]["aggregators"]["ridge_cov_active"]["mean_evaluation_weights"],
            placebo_weights=runs["placebo"]["aggregators"]["ridge_cov_active"][
                "mean_evaluation_weights"
            ],
            aggregator_placebos={
                name: dict(
                    mean=entry["summary"]["all"]["mean"],
                    vs_ridge_cov_active=entry["contrasts"]["vs_ridge_cov_active"],
                )
                for name, entry in runs["senal"]["placebos"].items()
            },
        ),
        hyperparameters={
            config: run["selected_hyperparams"]["ridge_cov_active"] for config, run in runs.items()
        },
        factors=(
            {config: info["scales"] for config, info in operable.items()}
            if mode == "operable"
            else None
        ),
        factors_on_boundary=(
            {config: info["scales_on_boundary"] for config, info in operable.items()}
            if mode == "operable"
            else None
        ),
    )
    return result, ics


def _number(value, digits=4):
    if value is None:
        return "—"
    return f"{value:+.{digits}f}".replace(".", ",").replace("-", "−")


def _contrast(entry):
    if entry["mean"] is None:
        return "—"
    return f"{_number(entry['mean'])} (t {_number(entry['t'], 1)}; {entry['sessions']} sesiones)"


def _markdown(summary):
    lines = [
        f"# S1 y S2 del origen {summary['origin']}",
        "",
        f"Declaración {summary['declaration_sha256'][:12]}; código en {summary['code']['root']}.",
        "Diferencias de evaluación pareadas por sesión, con t HAC.",
        "",
    ]
    for mode, result in summary["modes"].items():
        s1, s2, means = result["s1"], result["s2"], result["means"]
        rows = [
            (f"S1: media de rangos − el mejor por decisión ({s1['best']})", s1["vs_best"]),
            ("S1: media de rangos − titans_mac_nogain", s1["vs_parent"]),
            ("S1: media de rangos − lightgbm_technical", s1["vs_tree"]),
            ("S2: ridge_cov_active con la señal − media de rangos", s2["vs_s1"]),
            ("S2 − su placebo (señal barajada)", s2["vs_placebo"]),
            ("Placebo − media de rangos", s2["placebo_vs_s1"]),
            ("ridge_cov_active de los dos − media de rangos", s2["ridge_two_vs_s1"]),
            ("S2 − ridge_cov_active de los dos", s2["vs_ridge_two"]),
        ]
        lines += [
            f"## Medida: {MODE_NAMES[mode]}",
            "",
            "IC medio de evaluación: "
            + ", ".join(f"{name} {_number(value)}" for name, value in means.items())
            + ".",
            "",
            "| Contraste | Δ (t HAC) |",
            "|---|---|",
            *(f"| {label} | {_contrast(entry)} |" for label, entry in rows),
            "",
            "Pesos medios de S2 en evaluación: "
            + ", ".join(f"{name} {value:.2f}" for name, value in s2["weights"].items())
            + ". Hiperparámetros elegidos en calibración: "
            + "; ".join(f"{config} {value}" for config, value in result["hyperparameters"].items())
            + ".",
        ]
        if result["factors"] is not None:
            lines.append(
                "Factores de lo operable elegidos en calibración: "
                + "; ".join(f"{config} {value}" for config, value in result["factors"].items())
                + "."
            )
            for config, names in result["factors_on_boundary"].items():
                if names:
                    lines.append(
                        f"AVISO: en {config}, el factor de {', '.join(names)} cae en un borde "
                        "de la rejilla."
                    )
        lines.append("")
    return "\n".join(lines)


def _series(ics_by_mode):
    columns = {name: [] for name in ("mode", "market", "prediction_at")}
    names = sorted({name for ics in ics_by_mode.values() for name in ics})
    columns.update({name: [] for name in names})
    for mode, ics in ics_by_mode.items():
        keys = sorted({key for values in ics.values() for key in values})
        for key in keys:
            columns["mode"].append(mode)
            columns["market"].append(str(key[0]))
            columns["prediction_at"].append(int(key[1]))
            for name in names:
                columns[name].append(float(ics.get(name, {}).get(key, math.nan)))
    return pa.table(columns)


def _already(target, digest, fingerprint, log):
    record = target / "resumen.json"
    try:
        previous = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = {}
    _require(
        previous.get("status") == "completed" and previous.get("declaration_sha256") == digest,
        f"La raíz {target} es de otra declaración o no terminó: no se mezcla con S2",
    )
    _require(
        previous.get("inputs") == fingerprint,
        f"Las entradas cambiaron desde que se calculó {target}: apártala para repetir el origen",
    )
    log(f"{target}: ya calculado con esta declaración y estas entradas")
    return previous


def aggregate(
    declaration, state, reference, organs, origin, output, *, wait_hours, poll, log, markets=None
):
    """S1 y S2 de un origen, publicados en `<output>/<origen>`; devuelve el resumen.

    `markets` limita la medida a esos mercados (`<output>/<origen>-<mercados>`). Un estado con
    varios mercados exige elegirlos: la lectura es por mercado.
    """
    declaration, digest = checked_declaration(declaration)
    sources = folders(declaration, state, reference, organs, origin)
    manifest = Path(state) / "cohorts" / str(origin) / "manifest.json"
    missing = wait_for_inputs(declaration, sources, manifest, hours=wait_hours, poll=poll, say=log)
    if missing:
        raise MissingInputs(missing)
    fingerprint = _fingerprint(sources, manifest)
    target = Path(output) / target_name(origin, markets)
    if target.exists():
        return _already(target, digest, fingerprint, log)
    started = time.monotonic()

    def say(text):
        log(f"{time.monotonic() - started:6.0f} s {text}")

    records = {
        role: json.loads((folder / "run.json").read_text(encoding="utf-8"))
        for role, folder in sources.items()
    }
    _checked_sources(declaration, origin, sources, records, fingerprint)
    tables = {role: {s: _table(folder, s) for s in AGGREGATED} for role, folder in sources.items()}
    present = markets_of(tables["parent"])
    measured = measured_markets(present, markets)
    if measured != present:
        tables = {
            role: {s: only(table, measured) for s, table in splits.items()}
            for role, splits in tables.items()
        }
    say(f"mercados medidos: {measured} (el estado tiene {present})")
    tool = _screening()
    decision = {
        role: _decision(tool, sources[role], tuple(declaration["decision_splits"]), measured)
        for role in ("parent", "tree")
    }
    best = "parent" if decision["parent"] >= decision["tree"] else "tree"
    say(f"decisión: {decision}; el mejor es {declaration[best]}")
    for role in ("tree", "organ"):
        for split in AGGREGATED:
            for name in KEYS:
                _require(
                    tables[role][split].column(name).equals(tables["parent"][split].column(name)),
                    f"{sources[role]} no tiene las filas del padre en {split} (columna {name})",
                )
    view = records["parent"]["request"]["view_sha256"]
    alpha, matures = known_rows(state, origin, sources["parent"], view, tables["parent"], measured)
    say("alfa conocido y madurez de la vista")
    own = {
        s: tables["parent"][s].column("prediction").to_numpy().astype(np.float64)
        for s in AGGREGATED
    }
    signal = {
        s: tables["organ"][s].column("prediction").to_numpy().astype(np.float64) - own[s]
        for s in AGGREGATED
    }
    placebo = {
        s: shuffled_within_sessions(
            signal[s],
            np.asarray(tables["parent"][s].column("market").to_pylist()),
            tables["parent"][s].column("prediction_at").cast(pa.int64()).to_numpy(),
            declaration["placebo"]["seed"],
        )
        for s in AGGREGATED
    }
    predictions = dict(
        parent=own,
        tree={
            s: tables["tree"][s].column("prediction").to_numpy().astype(np.float64)
            for s in AGGREGATED
        },
        signal=signal,
        placebo=placebo,
    )
    names = dict(
        parent=declaration["parent"], tree=declaration["tree"], signal=SIGNAL, placebo=PLACEBO
    )
    Path(output).mkdir(parents=True, exist_ok=True)
    work = Path(output) / f".{origin}.{os.getpid()}.tmp"
    try:
        (work / "informes").mkdir(parents=True)
        raw_base = {
            s: oa.build_raw_split_base(
                tables["parent"][s], alpha=alpha[s], matures_at=matures[s], already_sorted=True
            )
            for s in AGGREGATED
        }
        inputs = {role: names[role] for role in predictions}
        modes, series = {}, {}
        decl_splits = tuple(declaration["splits"])
        decl_scales = tuple(declaration["operable_scales"])
        for mode in declaration["modes"]:
            is_operable = mode == "operable"
            raw_by_split = {
                s: {
                    names[role]: oa.build_raw_split(
                        raw_base[s], values[s], include_alpha=is_operable
                    )
                    for role, values in predictions.items()
                }
                for s in AGGREGATED
            }
            cache = (
                oa.build_shared_panel_cache(
                    raw_by_split,
                    splits=decl_splits,
                    min_assets=declaration["min_assets"],
                    align=declaration["align"],
                    operable=is_operable,
                    operable_scales=decl_scales,
                )
                if declaration["align"] == "strict"
                else None
            )
            modes[mode], series[mode] = _measure(
                declaration,
                tool,
                inputs,
                names,
                best,
                decision,
                mode,
                work,
                raw_by_split=raw_by_split,
                cache=cache,
            )
            say(f"{MODE_NAMES[mode]}: S2 − S1 {modes[mode]['s2']['vs_s1']['mean']}")
        summary = _plain(
            dict(
                schema_version=1,
                kind=KIND,
                status="completed",
                origin=origin,
                declaration_sha256=digest,
                code=_code_identity(),
                inputs=fingerprint,
                markets=measured,
                alpha_known={s: float(np.isfinite(alpha[s]).mean()) for s in AGGREGATED},
                modes=modes,
                seconds=round(time.monotonic() - started),
            )
        )
        pq.write_table(_series(series), work / "sesiones.parquet")
        (work / "resumen.md").write_text(_markdown(summary), encoding="utf-8")
        (work / "resumen.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        os.rename(work, target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    say(f"publicado en {target}")
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = result.add_subparsers(dest="command", required=True)
    run = commands.add_parser("agregar", help="Calcular S1 y S2 de un origen")
    run.add_argument("--declaration", type=Path, required=True)
    run.add_argument("--state", type=Path, required=True, help="Estado de la campaña")
    run.add_argument("--reference", type=Path, required=True, help="Raíz de los árboles")
    run.add_argument("--organs", type=Path, required=True, help="Raíz de los órganos")
    run.add_argument("--origin", type=int, required=True)
    run.add_argument("--output", type=Path, required=True, help="Raíz propia del agregador")
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
        declaration = json.loads(args.declaration.read_text(encoding="utf-8"))
        aggregate(
            declaration,
            args.state,
            args.reference,
            args.organs,
            args.origin,
            args.output,
            wait_hours=args.wait_hours,
            poll=args.poll_seconds,
            log=log,
            markets=args.markets,
        )
    except MissingInputs as missing:
        print(
            f"Faltan entradas de S2 para el origen {args.origin} tras esperar "
            f"{args.wait_hours:g} h; no se escribe nada:",
            file=sys.stderr,
        )
        for item in missing.items:
            print(f"- {item}", file=sys.stderr)
        return MISSING_EXIT
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
