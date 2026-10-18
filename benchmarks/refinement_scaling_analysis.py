"""Comparaciones de 64/256 activos sobre poblaciones completas y los 64 comunes."""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from mars_titan.data.batches import atomic_parquet_batches, read_bounded_table
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation.forecast_panel import SessionSeries
from mars_titan.evaluation.paired_comparisons import compare_series, delta, interaction
from mars_titan.training.learning_hold import require_learning_allowed

from .memory_microstudy_analysis import score_table
from .refinement_study import verify_prepared
from .refinement_study_policy import selection_group
from .refinement_study_traces import completed_phase

STUDIES = ("n64-original", "n64-geometry", "n256-original", "n256-geometry")
ROUTES = ("policy", "k1", "k4cc")


def score_populations(tables, common_assets):
    """Conservar la población fijada, sin intersectar filas ausentes entre modelos."""
    if set(tables) != set(STUDIES) or any(set(row) != set(ROUTES) for row in tables.values()):
        raise ValueError("Faltan combinaciones o rutas de la comparación")
    flows = {f"US/{asset}" for asset in common_assets}
    if not flows or len(flows) != len(common_assets):
        raise ValueError("La población común debe ser explícita y no repetirse")
    common = {}
    for name, cases in tables.items():
        common[name] = {}
        for route, table in cases.items():
            observed = set(table["asset_id"].to_pylist())
            if not flows <= observed or (name.startswith("n64-") and observed != flows):
                raise ValueError("Las filas no conservan los activos comunes declarados")
            common[name][route] = table.filter(
                pc.is_in(table["asset_id"], value_set=pa.array(sorted(flows)))
            )
    populations = {"common64": common}
    populations.update(
        {
            f"full{size}": {
                name: cases for name, cases in tables.items() if name.startswith(f"n{size}-")
            }
            for size in (64, 256)
        }
    )
    result = {}
    for population, sources in populations.items():
        scores, cohort = {}, None
        for name, cases in sources.items():
            for route, table in cases.items():
                scored = score_table(table, cohort)
                cohort = scored.cohort_sha256
                scores[f"{name}:{route}"] = scored.series("mae")
                if "zero" not in scores:
                    scores["zero"] = score_table(table, cohort, zero=True).series("mae")
        result[population] = scores
    return result


def compare_population(series, config):
    """Familia fijada por tamaño, representación y ruta, con sensibilidad predefinida."""
    contrasts = {}
    studies = [name for name in STUDIES if f"{name}:policy" in series]
    for route in ROUTES:
        for size in (64, 256):
            a, b = f"n{size}-original:{route}", f"n{size}-geometry:{route}"
            if a in series and b in series:
                contrasts[f"geometry_n{size}_{route}"] = delta(a, b)
        if len(studies) == 4:
            for representation in ("original", "geometry"):
                contrasts[f"scale_{representation}_{route}"] = delta(
                    f"n64-{representation}:{route}", f"n256-{representation}:{route}"
                )
            contrasts[f"interaction_{route}"] = interaction(
                *(f"{name}:{route}" for name in STUDIES)
            )
    for name in studies:
        for control in ("k1", "k4cc", "zero"):
            base = "zero" if control == "zero" else f"{name}:{control}"
            contrasts[f"{name}_policy_minus_{control}"] = delta(base, f"{name}:policy")
    return compare_series(
        series,
        contrasts,
        **config["bootstrap"],
        sensitivity_block_lengths=config["sensitivity_block_lengths"],
    )


def average_seeds(records, seeds):
    """Media de pérdidas por sesión solo con todas las semillas, nunca un ensemble."""
    if not seeds or len(set(seeds)) != len(seeds) or not set(records) <= set(seeds):
        raise ValueError("Las semillas no coinciden con el protocolo")
    missing = [seed for seed in seeds if seed not in records]
    if missing:
        return dict(status="pending", missing_seeds=missing)
    first = records[seeds[0]]
    if any(
        set(row) != set(first)
        or any(set(row[population]) != set(models) for population, models in first.items())
        for row in records.values()
    ):
        raise ValueError("Las semillas no conservan las mismas poblaciones y modelos")
    result = {}
    for population, models in first.items():
        result[population] = {}
        for name, reference in models.items():
            items = [records[seed][population][name] for seed in seeds]
            if any(
                item.markets != reference.markets
                or not np.array_equal(item.session_market, reference.session_market)
                or not np.array_equal(item.session_period, reference.session_period)
                or not np.array_equal(item.defined, reference.defined)
                for item in items
            ):
                raise ValueError("Las semillas no conservan las mismas sesiones")
            result[population][name] = SessionSeries.average(items)
    return dict(status="completed", seeds=list(seeds), series=result)


def load_tables(work, job, prepared):
    """Una evaluación incompleta no aporta filas al resumen."""
    output = work / "runs" / job["id"]
    path = output / "run.json"
    if not path.is_file():
        return None
    report = read_manifest(path)[0]
    if report.get("status") != "completed":
        return None
    selected = work / "runs" / job["parent"] / "phase.json"
    if (
        report.get("final_test_opened") is not False
        or report["job"] != job
        or report["selection_group"] != selection_group(work, job)
        or report["selection_sha256"] != sha256(selected)
    ):
        raise ValueError("La evaluación no conserva la selección confirmada")
    tables, hashes = {}, {}
    for route in ROUTES:
        folder = output if route == "policy" else output / "controls" / route
        phase = (
            report["phases"]["evaluation"]
            if route == "policy"
            else report["controls"][route]["evaluation"]
        )
        if phase["identity"].get("prepared_sha256") != sha256(work / "prepared.json"):
            raise ValueError("La fase pertenece a otra preparación")
        confirmed = completed_phase(folder / "evaluation", phase["identity"])
        if confirmed != phase:
            raise ValueError("La fase no corresponde al resultado confirmado")
        artifact = folder / "evaluation/predictions.parquet"
        table = read_bounded_table(artifact, max_rows=131072)
        if len(table) != prepared["counts"]["fold-018"]["evaluation"]:
            raise ValueError("La evaluación no conserva todas las filas preparadas")
        tables[route] = table
        hashes[route] = sha256(artifact)
    return tables, dict(
        run_sha256=sha256(path), selection_sha256=sha256(selected), predictions=hashes
    )


def preparations(works):
    if set(works) != set(STUDIES) or len(set(works.values())) != 4:
        raise ValueError("Se necesitan los cuatro estudios independientes")
    prepared = {name: verify_prepared(works[name]) for name in STUDIES}
    first = prepared[STUDIES[0]]
    config = dict(first["configuration"])
    config.pop("cohort_size")
    for name, record in prepared.items():
        candidate = dict(record["configuration"])
        size = candidate.pop("cohort_size")
        if (
            size != (64 if name.startswith("n64-") else 256)
            or candidate != config
            or record["jobs"] != first["jobs"]
            or record["code"] != first["code"]
            or record["environment"] != first["environment"]
            or record.get("final_test_opened") is not False
            or record["cohort"]["common_assets"] != first["assets"]
        ):
            raise ValueError("Los estudios no conservan protocolo, código y población común")
    for size in (64, 256):
        if prepared[f"n{size}-original"]["assets"] != prepared[f"n{size}-geometry"]["assets"]:
            raise ValueError("Las representaciones no conservan los mismos activos")
    return prepared


def write_population(output, populations, config):
    result = {}
    for name, series in populations.items():
        table = pa.concat_tables(
            [
                pa.table(
                    dict(
                        model=[model] * len(row.values),
                        market=[row.markets[int(index)] for index in row.session_market],
                        utc_day_index=row.session_period,
                        mae=row.values,
                        defined=row.defined,
                    )
                )
                for model, row in series.items()
            ]
        )
        artifact = output / f"{name}-sessions.parquet"
        atomic_parquet_batches(artifact, [table])
        result[name] = dict(
            comparison=compare_population(series, config),
            sessions=dict(path=artifact.name, sha256=sha256(artifact)),
            day_index_scope="utc_calendar_day_index_from_the_first_observed_day_of_this_population",
        )
    return result


def analyze(works, output):
    require_learning_allowed("el resumen emparejado de desarrollo de 64 y 256 activos")
    started = time.perf_counter()
    works = {name: Path(path).resolve() for name, path in works.items()}
    output = Path(output)
    safe_destination(output)
    for work in works.values():
        outside_source(work, output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("El resumen necesita una salida nueva")
    prepared = preparations(works)
    config = prepared[STUDIES[0]]["configuration"]
    common = prepared[STUDIES[0]]["assets"]
    output.mkdir(parents=True)
    results, pending, grouped = {}, [], {}
    for job in prepared[STUDIES[0]]["jobs"]:
        if job["kind"] != "evaluate":
            continue
        records = {name: load_tables(works[name], job, prepared[name]) for name in STUDIES}
        missing = [name for name, row in records.items() if row is None]
        if missing:
            pending.append(dict(job=job["id"], missing_studies=missing))
            continue
        populations = score_populations({name: row[0] for name, row in records.items()}, common)
        target = output / job["id"].replace("/", "_")
        results[job["id"]] = dict(
            populations=write_population(target, populations, config),
            provenance={name: row[1] for name, row in records.items()},
        )
        grouped.setdefault((job["projection"], job["depth"]), {})[job["seed"]] = populations
    means = {}
    for projection in config["projections"]:
        for depth in config["reader_depths"]:
            key = f"{projection}-k{depth}"
            averaged = average_seeds(grouped.get((projection, depth), {}), config["seeds"])
            if averaged["status"] == "completed":
                averaged["populations"] = write_population(
                    output / f"mean-{key}", averaged.pop("series"), config
                )
            means[key] = averaged
    report = dict(
        status="completed" if not pending else "partial",
        kind="paired_refinement_scaling_analysis",
        scientific_level="exploratory_development",
        prepared_sha256={name: sha256(work / "prepared.json") for name, work in works.items()},
        source_sha256=sha256(Path(__file__)),
        results=results,
        pending=pending,
        seed_means=means,
        final_test_opened=False,
        process_seconds=time.perf_counter() - started,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        limits=[
            "2023 es desarrollo reutilizado. No es el test final.",
            "La cohorte ampliada cambia datos de ajuste y composición del banco episódico.",
            "Las medias multisemilla promedian pérdidas por sesión, no predicciones.",
            "Los intervalos se corrigen dentro de cada familia, "
            "no entre todas las proyecciones y semillas.",
            "Los índices de día se refieren al primer día observado de cada población "
            "y no son nuevas réplicas.",
        ],
    )
    atomic_json(output / "results.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in STUDIES:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    works = {name: getattr(args, name.replace("-", "_")) for name in STUDIES}
    print(json.dumps(analyze(works, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
