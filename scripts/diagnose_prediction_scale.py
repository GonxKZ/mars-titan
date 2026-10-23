"""Fase 0 del estudio de variantes: escala, ordenación y contracción de predicciones guardadas.

Lee las predicciones por muestra que guardaron los pilotos de desarrollo, comprueba sus
huellas, elige λ por mercado con la partición de calibración y lo aplica sin cambios a la
evaluación. No ajusta ningún modelo ni importa PyTorch (#743).

    uv run python scripts/diagnose_prediction_scale.py \\
        --declaration configs/evaluation/scale-diagnostics-phase0.json \\
        --root <estado privado de los pilotos> \\
        --output reports/research/scale-diagnostics-20261023
"""

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation import scale_diagnostics as sd
from mars_titan.evaluation.forecast_panel import ForecastPanel

KEYS = ["sample_id", "asset_id", "market", "prediction_at", "target", "prediction"]
REPOSITORY = Path(__file__).resolve().parents[1]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _variant_seed(name):
    variant, _, seed = name.rpartition("-")
    return (variant, int(seed)) if seed.isdigit() else (name, None)


def _inventory_runs(source, root):
    """Ejecuciones de un inventario publicado, con la huella de cada archivo declarada."""
    document = json.loads((REPOSITORY / source["inventory"]).read_text())
    items = document["artifacts"] if isinstance(document, dict) else document
    runs = {}
    for item in items:
        if item.get("kind") != "sample_predictions":
            continue
        path = root / source["directory"] / item["relative_path"]
        run = str(Path(item["relative_path"]).parent)
        runs.setdefault(run, {})[item["phase"]] = (path, item["sha256"])
    return [dict(run=run, name=Path(run).name, files=files) for run, files in sorted(runs.items())]


def _phase_files(folder):
    """Predicciones por partición en cualquiera de las tres estructuras que usan los pilotos."""
    files = {}
    for phase in ("validation", "calibration", "evaluation"):
        candidates = (
            folder / f"{phase}-predictions.parquet",
            folder / phase / "predictions.parquet",
            folder / "run" / f"{phase}-predictions.parquet",
        )
        present = [path for path in candidates if path.exists()]
        _require(len(present) <= 1, f"Hay varias predicciones de {phase} en {folder.name}")
        if present:
            files[phase] = (present[0], None)
    return files


def _layout_runs(source, root):
    """Ejecuciones descubiertas por su estructura de carpetas, sin huella previa publicada."""
    base = root / source["directory"]
    found = []
    if source["layout"] == "runs":
        for folder in sorted(path for path in base.iterdir() if path.is_dir()):
            found.append(dict(run=folder.name, name=folder.name, files=_phase_files(folder)))
        return found
    _require(source["layout"] == "refinement", f"Estructura desconocida: {source['layout']}")
    for evaluate in sorted(base.glob("runs/*/evaluate-k*")):
        projection, seed = _variant_seed(evaluate.parent.name)
        reader = evaluate.name.removeprefix("evaluate-")
        folders = [(f"{projection}/selector-{reader}", evaluate)] + [
            (f"{projection}/{reader}/{control.name}", control)
            for control in sorted((evaluate / "controls").iterdir())
            if control.is_dir()
        ]
        for variant, folder in folders:
            run = str(folder.relative_to(base))
            found.append(dict(run=run, name=f"{variant}-{seed}", files=_phase_files(folder)))
    return found


def _read(path, expected):
    digest = sha256(path)
    _require(expected is None or digest == expected, f"La huella no coincide: {path.name}")
    table = pq.read_table(path, columns=KEYS)
    return table, digest


def _panel(tables):
    return ForecastPanel.from_arrow(pa.concat_tables(tables) if len(tables) > 1 else tables[0])


def _prediction_key(*panels):
    """Huella de población y predicciones. Sumar 0,0 convierte −0,0 en +0,0 antes de hashear."""
    digest = hashlib.sha256()
    for panel in panels:
        digest.update(panel.cohort_sha256.encode())
        digest.update(np.ascontiguousarray(panel.prediction + 0.0).tobytes())
    return digest.hexdigest()


def _past_median_mae(tables, evaluation, parameters):
    """MAE por sesión del control de mediana pasada del mismo activo.

    ``tables`` va en orden temporal y termina con la propia evaluación, cuyas filas son las
    últimas de la concatenación.
    """
    _require(tables and tables[-1] is evaluation, "La evaluación debe ser la última partición")
    for earlier, later in zip(tables, tables[1:], strict=False):
        _require(
            pc.max(earlier["prediction_at"]).as_py() < pc.min(later["prediction_at"]).as_py(),
            "Las particiones del control pasado no están en orden temporal",
        )
    keys = pa.concat_tables([table.select(KEYS[:5]) for table in tables])
    times = keys["prediction_at"].cast(pa.int64()).to_numpy()
    markets = np.asarray(keys["market"].to_pylist())
    sessions = np.zeros(len(times), dtype=np.int64)
    for market in np.unique(markets):
        rows = markets == market
        sessions[rows] = np.unique(times[rows], return_inverse=True)[1]
    values = sd.past_median(
        np.asarray(keys["asset_id"].to_pylist()),
        sessions,
        keys["target"].to_numpy(),
        lag=parameters["lag_sessions"],
        window=parameters["window"],
        min_labels=parameters["min_labels"],
    )
    rows = len(evaluation)
    table = evaluation.set_column(
        evaluation.schema.get_field_index("prediction"),
        "prediction",
        pa.array(values[-rows:]),
    )
    panel = ForecastPanel.from_arrow(table)
    return sd.session_mae(panel, panel.prediction)


def _interval(family, index):
    return dict(
        estimate=float(family["estimate"][index]),
        interval=[float(family["lower"][index]), float(family["upper"][index])],
        simultaneous=[float(family["joint_lower"][index]), float(family["joint_upper"][index])],
    )


def _families(columns, period, bootstrap):
    """Familia principal y sensibilidades de bloque de una matriz de contrastes."""
    names = list(columns)
    values = np.column_stack([columns[name] for name in names])
    options = dict(
        replicates=bootstrap["replicates"],
        seed=bootstrap["seed"],
        confidence=bootstrap["confidence"],
    )
    main = sd.block_family(values, period, block_length=bootstrap["block_length"], **options)
    sensitivity = {
        str(length): sd.block_family(values, period, block_length=length, **options)
        for length in bootstrap["sensitivity_block_lengths"]
    }
    return dict(
        critical=main["critical"],
        members={
            name: dict(
                **_interval(main, index),
                sensitivity={
                    length: _interval(local, index)["simultaneous"]
                    for length, local in sensitivity.items()
                },
            )
            for index, name in enumerate(names)
        },
    )


def _diagnose_run(source, record, declaration):
    files = record["files"]
    needed = {*declaration["fit_partition"], declaration["evaluation_partition"]}
    _require(needed <= set(files), f"Faltan particiones en {source['name']}/{record['run']}")
    tables, digests = {}, {}
    for phase, (path, expected) in files.items():
        tables[phase], digests[phase] = _read(path, expected)
    fit = _panel([tables[phase] for phase in declaration["fit_partition"]])
    evaluation = _panel([tables[declaration["evaluation_partition"]]])
    result = sd.diagnose_predictor(
        fit, evaluation, fit_prediction=fit.prediction, prediction=evaluation.prediction
    )
    wide = declaration["sensitivity_fit_partition"]
    if set(wide) <= set(tables):
        panel = _panel([tables[phase] for phase in wide])
        sd.ensure_before(panel, evaluation)
        factors = sd.market_shrinkage(panel, panel.prediction)
        result["sensitivity_shrinkage"] = factors
        result["series"]["sensitivity"] = sd.session_mae(
            evaluation, sd.apply_shrinkage(evaluation, evaluation.prediction, factors)
        )
    variant, seed = _variant_seed(record["name"])
    return dict(
        id=f"{source['name']}/{record['run']}",
        source=source["name"],
        variant=variant,
        seed=seed,
        files={phase: digest for phase, digest in sorted(digests.items())},
        key=_prediction_key(fit, evaluation),
        cohort=evaluation.cohort_sha256,
        evaluation=evaluation,
        tables=tables,
        **result,
    )


def _summary_row(item, zero):
    series = item["series"]

    def excess(values):
        return float(values.mean() / zero - 1)

    return dict(
        id=item["id"],
        source=item["source"],
        variant=item["variant"],
        seed=item["seed"],
        aliases=item["aliases"],
        files=item["files"],
        raw_mae=float(series["raw"].mean()),
        raw_excess=excess(series["raw"]),
        shrunk_mae=float(series["shrunk"].mean()),
        shrunk_excess=excess(series["shrunk"]),
        sensitivity_excess=(excess(series["sensitivity"]) if "sensitivity" in series else None),
        bound_excess_with_future_information=excess(series["bound"]),
        shrinkage=item["shrinkage"],
        sensitivity_shrinkage=item.get("sensitivity_shrinkage"),
        shrinkage_with_future_information=item["shrinkage_with_future_information"],
        fit_slope=item["fit_slope"],
        evaluation_slope=item["evaluation_slope"],
        mean_pearson_ic=item["mean_pearson_ic"],
        mean_rank_ic=item["mean_rank_ic"],
        pearson_icir=item["pearson_icir"],
        undefined_ic_sessions=item["undefined_ic_sessions"],
        prediction_scale=item["prediction_scale"],
        target_scale=item["target_scale"],
    )


def _best_gain_power(items, zero_series, mde):
    """Potencia frente al propio diferencial de los predictores con mayor mejora sobre cero."""
    result = {}
    for kind in ("shrunk", "bound"):
        best = min(items, key=lambda item: item["series"][kind].mean())
        differential = best["series"][kind] - zero_series
        effect = sd.minimum_detectable_effect(
            differential, candidates=mde["candidates"], alpha=mde["alpha"], power=mde["power"]
        )
        gain = float(-differential.mean())
        result[kind] = dict(
            id=best["id"],
            gain=gain,
            gain_relative_to_zero=gain / float(zero_series.mean()),
            standardized_gain=gain / effect["standard_deviation"]
            if effect["standard_deviation"] > 0
            else None,
            detectable_over_gain={
                count: (value["absolute"] / gain if gain > 0 else None)
                for count, value in effect["effects"].items()
            },
        )
    return result


def _cohort_report(items, tables, evaluation, declaration):
    first = items[0]
    period = evaluation.session_period
    zero_series = first["series"]["zero"]
    zero = float(zero_series.mean())
    partitions = [p for p in declaration["past_median"]["partitions"] if p in tables]
    past = _past_median_mae(
        [tables[p] for p in partitions],
        tables[declaration["evaluation_partition"]],
        declaration["past_median"],
    )
    inventory = {source["name"] for source in declaration["sources"] if "inventory" in source}
    primary = declaration["rule"]["primary_source"]
    bootstrap = declaration["bootstrap"]
    complete = [item for item in items if item["undefined_ic_sessions"] == 0]
    ic = (
        _families({item["id"]: item["series"]["pearson"] for item in complete}, period, bootstrap)
        if complete
        else dict(critical=None, members={})
    )
    raw = _families(
        {
            "past_median": past - zero_series,
            **{item["id"]: item["series"]["raw"] - zero_series for item in items},
        },
        period,
        bootstrap,
    )
    shrunk = _families(
        {item["id"]: item["series"]["shrunk"] - zero_series for item in items}, period, bootstrap
    )
    mde = declaration["minimum_detectable_effect"]
    effects = {
        kind: [
            sd.minimum_detectable_effect(
                item["series"][kind] - zero_series,
                candidates=mde["candidates"],
                alpha=mde["alpha"],
                power=mde["power"],
            )
            for item in items
        ]
        for kind in ("raw", "shrunk")
    }
    return dict(
        cohort_sha256=first["cohort"],
        rows=evaluation.rows,
        sessions=evaluation.sessions,
        assets=len(set(tables[declaration["evaluation_partition"]]["asset_id"].to_pylist())),
        first_session=int(evaluation.session_time.min()),
        last_session=int(evaluation.session_time.max()),
        sources=sorted({item["source"] for item in items}),
        zero_mae=zero,
        past_median=dict(
            partitions=partitions,
            mae=float(past.mean()),
            excess=float(past.mean() / zero - 1),
            **raw["members"]["past_median"],
        ),
        predictors=[_summary_row(item, zero) for item in items],
        families=dict(
            pearson_ic=dict(members_without_complete_ic=len(items) - len(complete), **ic),
            # Sensibilidad añadida tras la revisión de la PR: la misma familia limitada a la
            # fuente principal o a los estudios con inventario publicado.
            pearson_ic_scopes={
                scope: _families(
                    {item["id"]: item["series"]["pearson"] for item in members}, period, bootstrap
                )
                for scope, members in (
                    ("primary_source", [i for i in complete if i["source"] == primary]),
                    ("inventory_sources", [i for i in complete if i["source"] in inventory]),
                )
                if members
            },
            raw_minus_zero=raw,
            shrunk_minus_zero=shrunk,
        ),
        minimum_detectable_effect={
            kind: {
                str(count): dict(
                    median_absolute=float(
                        np.median([e["effects"][str(count)]["absolute"] for e in values])
                    ),
                    median_relative_to_zero=float(
                        np.median([e["effects"][str(count)]["absolute"] for e in values]) / zero
                    ),
                    median_standardized=float(
                        np.median([e["effects"][str(count)]["standardized"] for e in values])
                    ),
                )
                for count in mde["candidates"]
            }
            for kind, values in effects.items()
        },
        best_gain_power=_best_gain_power(items, zero_series, mde),
        differential_autocorrelation=dict(
            raw_median=float(np.median([e["autocorrelation"] for e in effects["raw"]])),
            shrunk_median=float(np.median([e["autocorrelation"] for e in effects["shrunk"]])),
        ),
    )


def _signs(members):
    return dict(
        size=len(members),
        positive=sorted(name for name, row in members.items() if row["simultaneous"][0] > 0),
        negative=sorted(name for name, row in members.items() if row["simultaneous"][1] < 0),
    )


def _verdict(cohorts, declaration):
    """Decisión con la familia declarada (toda la población) y su sensibilidad a la familia."""
    rule = declaration["rule"]
    primary = next(cohort for cohort in cohorts if rule["primary_source"] in cohort["sources"])
    declared = _signs(primary["families"]["pearson_ic"]["members"])
    scopes = {
        scope: _signs(family["members"])
        for scope, family in primary["families"]["pearson_ic_scopes"].items()
    }
    return dict(
        primary_cohort=primary["cohort_sha256"],
        family="whole_cohort",
        ic_family_size=declared["size"],
        ic_distinguishable_positive=declared["positive"],
        ic_distinguishable_negative=declared["negative"],
        decision=rule["otherwise"] if declared["positive"] else rule["if_none_distinguishable"],
        family_sensitivity=scopes,
        robust_to_family=all(
            bool(scope["positive"]) == bool(declared["positive"]) for scope in scopes.values()
        ),
    )


def _commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, capture_output=True, text=True, check=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPOSITORY, capture_output=True, text=True, check=True
    )
    return dict(commit=result.stdout.strip(), clean=status.stdout.strip() == "")


def run(declaration_path, root, output):
    started, cpu = time.perf_counter(), time.process_time()
    declaration = json.loads(declaration_path.read_text())
    _require(declaration["kind"] == "scale_diagnostics_phase0", "Declaración inesperada")
    _require(declaration["final_test_opened"] is False, "La fase 0 no abre el test final")
    items, seen, references, skipped = [], {}, {}, []
    for source in declaration["sources"]:
        runs = (
            _inventory_runs(source, root) if "inventory" in source else _layout_runs(source, root)
        )
        _require(runs, f"La fuente {source['name']} no tiene ejecuciones")
        for record in runs:
            # Regla añadida tras la declaración: una carpeta descubierta sin la partición de
            # ajuste (dos celdas congeladas del estudio combinado y una ejecución sin
            # predicciones) se registra como omitida. En un inventario sería un error.
            needed = {*declaration["fit_partition"], declaration["evaluation_partition"]}
            if "inventory" not in source and not needed <= set(record["files"]):
                skipped.append(
                    dict(
                        id=f"{source['name']}/{record['run']}",
                        partitions=sorted(record["files"]),
                    )
                )
                continue
            item = _diagnose_run(source, record, declaration)
            # Solo se conservan las tablas de una ejecución por población, la que tenga más
            # particiones, para el control de mediana pasada. El resto se libera al momento.
            tables, evaluation = item.pop("tables"), item.pop("evaluation")
            current = references.get(item["cohort"])
            if current is None or len(tables) > len(current[0]):
                references[item["cohort"]] = (tables, evaluation)
            if item["key"] in seen:
                seen[item["key"]]["aliases"].append(item["id"])
                continue
            item["aliases"] = []
            seen[item["key"]] = item
            items.append(item)
    cohorts = {}
    for item in items:
        cohorts.setdefault(item["cohort"], []).append(item)
    reports = [
        _cohort_report(group, *references[cohort], declaration) for cohort, group in cohorts.items()
    ]
    summary = dict(
        schema_version=1,
        kind="scale_diagnostics_phase0_result",
        declaration=dict(
            path=str(declaration_path.relative_to(REPOSITORY)), sha256=sha256(declaration_path)
        ),
        source=_commit(),
        inventories={
            source["name"]: sha256(REPOSITORY / source["inventory"])
            for source in declaration["sources"]
            if "inventory" in source
        },
        environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            pyarrow=pa.__version__,
            machine=platform.machine(),
        ),
        predictors=len(items),
        aliases=sum(len(item["aliases"]) for item in items),
        skipped=skipped,
        cohorts=reports,
        verdict=_verdict(reports, declaration),
        cost=dict(
            wall_seconds=time.perf_counter() - started,
            process_seconds=time.process_time() - cpu,
            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        ),
        optimizer_steps=0,
        final_test_opened=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    # La ejecución publicada no carga PyTorch, así que no puede dar pasos de optimizador.
    _require("torch" not in sys.modules, "La fase 0 no debe cargar PyTorch")
    summary = run(args.declaration.resolve(), args.root.resolve(), args.output)
    _require("torch" not in sys.modules, "La fase 0 no debe cargar PyTorch")
    print(json.dumps(summary["verdict"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
