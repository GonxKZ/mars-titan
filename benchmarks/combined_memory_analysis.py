"""Errores por sesión, interacciones y costes del catálogo combinado fijado."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation.forecast_panel import SessionSeries
from mars_titan.evaluation.paired_comparisons import compare_series, delta, interaction

from .combined_memory_study import configuration, jobs, verify_prepared
from .memory_microstudy_analysis import score_table


def cell_means(series, fold):
    """Promediar errores de semillas completas sobre las mismas sesiones."""
    seeds = configuration()["seeds"] if fold == "fold-018" else [42]
    result = {}
    for name in [f"cell{i}" for i in range(8)] + ["pt1_init"]:
        keys = [f"{fold}/{name}-{seed}" for seed in seeds]
        if all(key in series for key in keys):
            result[name + "-mean"] = SessionSeries.average([series[key] for key in keys])
    return result


def factor_interaction(mask):
    """Diferencia de segundo o tercer orden respecto a la celda sin ampliaciones."""
    if type(mask) is not int or mask not in (3, 5, 6, 7):
        raise ValueError("La interacción necesita dos o tres factores del catálogo")
    return {
        f"cell{bits}-mean": (-1) ** (mask.bit_count() - bits.bit_count())
        for bits in range(8)
        if bits & mask == bits
    }


def averaged_factor_difference(mask):
    """Promediar una diferencia factorial sobre los niveles de los otros factores."""
    if type(mask) is not int or not 1 <= mask <= 7:
        raise ValueError("La diferencia necesita uno, dos o tres factores")
    scale = 2 ** (3 - mask.bit_count())
    return {
        f"cell{bits}-mean": (-1) ** (mask.bit_count() - (bits & mask).bit_count()) / scale
        for bits in range(8)
    }


def contrast_families(series, fold):
    """Familias fijadas por el catálogo, sin escoger pares por sus resultados."""
    families = {}

    def add(family, name, coefficients):
        if set(coefficients) <= set(series):
            families.setdefault(family, {})[name] = coefficients

    for bits in range(8):
        name = f"cell{bits}-mean"
        add("factorial", name + "_minus_zero", delta("zero", name))
        if bits:
            add("factorial", name + "_minus_cell0", delta("cell0-mean", name))
    for mask in (3, 5, 6, 7):
        add("factorial", f"interaction_{mask}", factor_interaction(mask))
    for mask in (1, 2, 4, 3, 5, 6):
        add("factorial", f"averaged_difference_{mask}", averaged_factor_difference(mask))
    for mask in (1, 2, 4):
        add("factorial", f"joint_minus_without_{mask}", delta(f"cell{7 ^ mask}-mean", "cell7-mean"))
    add("alternatives", "pt1_init_minus_base", delta("cell0-mean", "pt1_init-mean"))
    add("alternatives", "pt1_minus_initialization", delta("pt1_init-mean", "cell1-mean"))
    for job in jobs():
        if job["fold"] != fold:
            continue
        name = job["id"]
        if job["kind"] == "readout":
            parent, seed = job["parent"].rsplit("-", 1)
            frozen = parent + "_frozen-" + seed
            if frozen not in series:
                continue
            add("readers", name + "_minus_parent", delta(frozen, name))
            if job["admission"] != "m0_no_bank":
                add("readers", name + "_minus_zero", delta("zero", name))
            if job["initialization"] == "zero_refinement":
                if job["refinements"] == 1:
                    old = name.replace("_zero_refinement", "_pytorch_default")
                else:
                    old = f"{fold}/cell{job['bits']}_m1_zero_refinement-{seed}"
                if old != name:
                    add("readers", name + "_minus_reader_control", delta(old, name))
        elif job["kind"] in ("core", "candidate") and not job.get("factorial"):
            add("alternatives", name + "_minus_base", delta(f"{fold}/cell0-{job['seed']}", name))
            add("alternatives", name + "_minus_zero", delta("zero", name))
        elif job["kind"] == "post":
            parent = (
                job["parent"] if job["name"] == "full_continuation" else f"{fold}/cell0_frozen-42"
            )
            add("posttraining", name + "_minus_parent", delta(parent, name))
            add("posttraining", name + "_minus_zero", delta("zero", name))
    for bits in (0, 7):
        for initialization in ("pytorch_default", "zero_refinement"):
            for left, right in (("m0_no_bank", "m1"), ("m1", "m2"), ("m2", "m3")):
                base = f"{fold}/cell{bits}_{left}_{initialization}-42"
                variant = f"{fold}/cell{bits}_{right}_{initialization}-42"
                add("readers", variant + "_minus_" + left, delta(base, variant))
    cm = [f"{fold}/cm_v1_{name}-42" for name in ("b", "bc", "bm", "bcm")]
    for name in cm:
        add("cm", name + "_minus_zero", delta("zero", name))
        if name != cm[0]:
            add("cm", name + "_minus_b", delta(cm[0], name))
    add("cm", "cm_interaction", interaction(*cm))
    add("cm", "cm_joint_minus_c", delta(cm[1], cm[3]))
    add("cm", "cm_joint_minus_m", delta(cm[2], cm[3]))
    add("cm", "core_c_minus_core_b", delta(f"{fold}/cm_v1_core_b-42", f"{fold}/cm_v1_core_c-42"))
    return families


def read_result(work, job, count, cohort):
    folder = work / "runs" / job["id"]
    manifest = folder / ("window.json" if job["kind"] == "candidate" else "run.json")
    if not manifest.exists():
        return None, "Sin manifiesto"
    run = json.loads(manifest.read_text())
    if run.get("status") != "completed":
        return None, run.get("status", "Sin estado")
    if run.get("final_test_opened") is not False:
        raise ValueError("La ejecución no conserva la reserva final")
    record = run["predictions"]["evaluation"]
    path = (folder / record["path"]).resolve()
    if not path.is_relative_to(folder.resolve()) or sha256(path) != record["sha256"]:
        raise ValueError("Las predicciones no conservan su ruta y huella")
    table = pq.read_table(path)
    if len(table) != count or record["rows"] != count:
        raise ValueError("La evaluación no contiene todas las filas de la cohorte")
    year = 2023 if job["fold"] == "fold-018" else 2020
    moments = table["prediction_at"].to_numpy()
    if np.any(moments < np.datetime64(f"{year}-01-01")) or np.any(
        moments >= np.datetime64(f"{year + 1}-01-01")
    ):
        raise ValueError("Las predicciones salen del año de desarrollo declarado")
    if job["kind"] == "reference":
        parent = work / "runs" / job["parent"] / "run.json"
        parent_run = json.loads(parent.read_text())
        if (
            run.get("optimizer_steps") != 0
            or run.get("shared_parameters_unchanged") is not True
            or run.get("parent_sha256") != sha256(parent)
            or run.get("checkpoint_sha256") != parent_run["checkpoint"]["sha256"]
        ):
            raise ValueError("La referencia no conserva el padre congelado declarado")
    scores = score_table(table, cohort)
    fit_path = folder / ("run/run.json" if job["kind"] == "candidate" else "fit/run.json")
    fit = json.loads(fit_path.read_text()) if fit_path.exists() else {}
    result = dict(
        summary=scores.summary(),
        run_sha256=sha256(manifest),
        predictions_sha256=sha256(path),
        seed=job["seed"],
        kind=job["kind"],
        selection=fit.get("selection"),
        job=job,
    )
    return (result, scores, table), None


def stratum_scores(table, groups):
    """Desglose descriptivo por grupos fijados en el ajuste, con cobertura sin solapes."""
    members = [asset for assets in groups.values() for asset in assets]
    observed = set(table["asset_id"].to_pylist())
    if len(members) != len(set(members)) or not observed <= set(members):
        raise ValueError("Los estratos solapan activos o no cubren todas las filas")
    result = {}
    for name, assets in groups.items():
        subset = table.filter(pc.is_in(table["asset_id"], value_set=pa.array(assets)))
        if not len(subset):
            result[name] = dict(rows=0)
            continue
        scores, zero = score_table(subset), score_table(subset, zero=True)
        result[name] = dict(
            rows=len(subset),
            sessions=len(scores.series("mae").values),
            mae=scores.summary()["point"]["mae"],
            zero_mae=zero.summary()["point"]["mae"],
        )
    return result


def audit_groups(path, receipt):
    if path is None:
        return {}
    path = Path(path)
    if sha256(path) != receipt["audit_sha256"]:
        raise ValueError("La auditoría de estratos no es la preparada")
    rows = {row["symbol"]: row for row in json.loads(path.read_text())["rows"]}
    result = {}
    for fold, population in receipt["cohorts"].items():
        groups = {}
        for symbol in population["assets"]:
            name = rows[symbol]["folds"][fold]["stratum"]
            groups.setdefault(name, []).append("US/" + symbol)
        result[fold] = groups
    return result


def analyze(work, output, *, audit=None):
    work, output = Path(work).resolve(), Path(output)
    receipt = verify_prepared(work)
    measurements = [json.loads(p.read_text()) for p in (work / "measurements").glob("*.json")]
    prepared_sha = sha256(work / "prepared.json")
    if any(
        m.get("prepared_sha256") != prepared_sha for m in measurements if m["status"] == "completed"
    ):
        raise ValueError("Una ejecución completada no pertenece a esta preparación")
    groups = audit_groups(audit, receipt)
    if output.exists():
        raise FileExistsError("El análisis necesita un destino nuevo")
    output.mkdir(parents=True)
    windows, pending = {}, []
    for fold, population in receipt["cohorts"].items():
        folder = output / fold
        folder.mkdir()
        models, series, cohort = {}, {}, None
        for job in receipt["jobs"]:
            if job["fold"] != fold:
                continue
            value, reason = read_result(work, job, population["counts"]["evaluation"], cohort)
            if reason:
                item = dict(job=job["id"], reason=reason)
                attempts = [m for m in measurements if m.get("job") == job]
                last = max(attempts, key=lambda m: m.get("started_at_utc", ""), default={})
                if reason == "running" and last.get("status") == "terminated":
                    item.update(reason="terminated", manifest_status=reason)
                pending.append(item)
                continue
            if not any(m.get("job") == job and m["status"] == "completed" for m in measurements):
                raise ValueError("Falta la medición de ejecución de un resultado completado")
            result, scores, table = value
            if fold in groups:
                result["training_coverage_strata"] = stratum_scores(table, groups[fold])
            cohort = scores.cohort_sha256
            name = job["id"]
            models[name], series[name] = result, scores.series("mae")
            pq.write_table(scores.to_table(), folder / (name.split("/")[1] + "-sessions.parquet"))
            if "zero" not in models:
                zero = score_table(table, cohort, zero=True)
                models["zero"] = dict(summary=zero.summary(), kind="zero_residual")
                series["zero"] = zero.series("mae")
                pq.write_table(zero.to_table(), folder / "zero-sessions.parquet")
        if not series:
            windows[fold] = dict(status="no_completed_jobs", population=population)
            continue
        means = cell_means(series, fold)
        seeds = configuration()["seeds"] if fold == "fold-018" else [42]
        for name, mean in means.items():
            stem = name.removesuffix("-mean")
            values = [models[f"{fold}/{stem}-{seed}"]["summary"]["point"]["mae"] for seed in seeds]
            models[name] = dict(
                kind="mean_of_session_errors",
                mae=float(np.mean(values)),
                seeds=seeds,
                seed_mae=values,
                seed_standard_deviation=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                sessions=len(mean.values),
            )
        series.update(means)
        comparisons = {}
        for family, contrasts in contrast_families(series, fold).items():
            comparisons[family] = {}
            for block in [16, *configuration()["sensitivity_block_lengths"]]:
                options = configuration()["bootstrap"] | dict(block_length=block)
                comparisons[family][str(block)] = compare_series(series, contrasts, **options)
        windows[fold] = dict(
            status="analyzed",
            population=population,
            cohort_sha256=cohort,
            models=models,
            comparisons=comparisons,
        )
    result = dict(
        schema_version=1,
        kind="combined_memory_results",
        scientific_level="exploratory_development",
        recorded_at_utc=datetime.now(UTC).isoformat(),
        prepared_sha256=sha256(work / "prepared.json"),
        script_sha256=sha256(Path(__file__)),
        windows=windows,
        pending=pending,
        measurements=measurements,
        total_measured_seconds=sum(m["process_seconds"] for m in measurements),
        final_test_opened=False,
        limits=[
            "Dos épocas no acreditan convergencia ni superioridad general.",
            "La ventana secundaria está prevista con una semilla y otra población. "
            "Su estado indica si llegó a ejecutarse.",
            "Los intervalos simultáneos se refieren a cada familia declarada, no a todas juntas.",
            "El coste de lectores y adaptadores se añade al de su padre.",
        ],
    )
    atomic_json(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    result = analyze(args.work, args.output, audit=args.audit)
    print(json.dumps(dict(windows=len(result["windows"]), pending=len(result["pending"]))))


if __name__ == "__main__":
    main()
