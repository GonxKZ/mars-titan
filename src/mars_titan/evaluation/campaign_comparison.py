"""Comparar campañas congeladas sin volver a entrenar ni abrir la reserva final."""

import argparse
import csv
import math
import resource
import time
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mars_titan.data.cohort_files import safe_destination
from mars_titan.data.input_policy import INPUT_POLICIES, STRICT_INPUTS
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation.comparison_sources import predictive_sources
from mars_titan.evaluation.prediction_statistics import (
    InitialPolicyCache,
    block_intervals,
    review_predictions,
)

GROUP = ("partition", "family", "method")
CONTRASTS = ("zero", "parent", "initial_policy", "center")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _key(row):
    return row["fold"], row["partition"], row["id"]


def _mean(values):
    present = [value for value in values if value is not None]
    return math.fsum(present) / len(present) if present else None


def _market_population(source, result, fold):
    declared = fold.get("market_counts")
    if declared is None:
        return
    totals = (
        result["sessions"].group_by("market", use_threads=False).aggregate([("samples", "sum")])
    )
    actual = {row["market"]: row["samples_sum"] for row in totals.to_pylist()}
    _require(
        actual == {market: counts[source["partition"]] for market, counts in declared.items()},
        "Las muestras por mercado no corresponden a su cohorte temporal",
    )
    cohort = result["cohort"]
    prefixes = pc.utf8_slice_codeunits(cohort["asset_id"], start=0, stop=3)
    _require(
        pc.all(pc.equal(prefixes, pc.binary_join_element_wise(cohort["market"], "/", ""))).as_py(),
        "Un activo no pertenece al mercado de su predicción",
    )


def _market_aggregates(admitted, daily, sessions, markets, **options):
    result = {name: [] for name in ("overall", "by_fold", "intervals")}
    for market in markets:
        cases = []
        for key, original in admitted.items():
            rows = [row for (name, _), row in daily[key].items() if name == market]
            _require(rows, "Un modelo no conserva las sesiones de todos los mercados")
            cases.append(
                dict(
                    original,
                    sessions=len(rows),
                    samples=sum(row["samples"] for row in rows),
                    session_mse=_mean(row.get("mse_prediction") for row in rows),
                    mean_session_rank_ic=_mean(row.get("rank_ic") for row in rows),
                    mean_session_direction_accuracy=_mean(
                        row.get("direction_accuracy") for row in rows
                    ),
                )
            )
        selected = sessions.filter(pc.equal(sessions["market"], market))
        local = aggregate_results(cases, selected, **options)
        for name in result:
            result[name].extend(dict(row, market=market) for row in local[name])
    return result


def aggregate_results(cases, sessions, *, block_lengths=(1, 5, 10, 20), repetitions=2000, seed=42):
    """Promediar errores de semillas y dar el mismo peso descriptivo a cada ventana."""
    _require(len({_key(row) for row in cases}) == len(cases), "Hay casos duplicados")
    admitted = {_key(row): row for row in cases if row["included"]}
    _require(admitted, "No hay casos seleccionados")
    daily = defaultdict(dict)
    for row in sessions.to_pylist():
        key = _key(row)
        if key not in admitted:
            continue
        moment = (row["market"], str(row["prediction_at"]))
        _require(moment not in daily[key], "Hay sesiones duplicadas en un caso")
        daily[key][moment] = row
    _require(set(daily) == set(admitted), "Faltan pérdidas de casos seleccionados")
    grouped = defaultdict(list)
    for case in admitted.values():
        _require(len(daily[_key(case)]) == case["sessions"], "Faltan sesiones de un caso")
        grouped[(case["fold"], *(case[name] for name in GROUP))].append(case)
    markets = sorted({market for rows in daily.values() for market, _ in rows})
    _require(set(markets) <= {"US", "CN"}, "Las pérdidas contienen un mercado desconocido")
    if len(markets) > 1:
        return _market_aggregates(
            admitted,
            daily,
            sessions,
            markets,
            block_lengths=block_lengths,
            repetitions=repetitions,
            seed=seed,
        )
    fold_results, comparisons = [], defaultdict(list)
    for (fold, partition, family, method), rows in sorted(grouped.items()):
        _require(len({row["seed"] for row in rows}) == len(rows), "Hay semillas duplicadas")
        _require(len({row["primary"] for row in rows}) == 1, "Las salidas primarias no coinciden")
        keys = sorted(daily[_key(rows[0])])
        _require(
            all(sorted(daily[_key(row)]) == keys for row in rows),
            "Las semillas no comparten todas las sesiones",
        )
        _require(
            len({market for market, _ in keys}) == 1, "El bootstrap requiere un mercado por ventana"
        )
        count = len(keys)
        losses = {}
        for role in ("prediction", *CONTRASTS):
            values = [[daily[_key(row)][key].get(f"mae_{role}") for key in keys] for row in rows]
            if all(value is not None for sample in values for value in sample):
                losses[role] = np.mean(np.asarray(values, dtype=np.float64), axis=0)
        _require("prediction" in losses and "zero" in losses, "Faltan pérdidas principales")
        summary = dict(
            fold=fold,
            partition=partition,
            family=family,
            method=method,
            models=len(rows),
            sessions=count,
            session_mae=float(np.mean(losses["prediction"])),
            session_mse=_mean(row["session_mse"] for row in rows),
            selected_initial=sum(row.get("selected_initial") is True for row in rows),
            stopped_early=sum(row.get("stopped_early") is True for row in rows),
            primary=rows[0]["primary"],
            mean_rank_ic=_mean(row.get("mean_session_rank_ic") for row in rows),
            mean_direction_accuracy=_mean(
                row.get("mean_session_direction_accuracy") for row in rows
            ),
        )
        for role in CONTRASTS:
            eligible = role in losses and (
                role != "parent" or any(row["parent_id"] is not None for row in rows)
            )
            summary[f"mae_{role}"] = float(np.mean(losses[role])) if role in losses else None
            summary[f"delta_{role}"] = (
                float(np.mean(losses["prediction"] - losses[role])) if eligible else None
            )
            if eligible:
                comparisons[(fold, partition)].append(
                    (
                        dict(family=family, method=method, reference=role, models=len(rows)),
                        keys,
                        losses["prediction"] - losses[role],
                    )
                )
        fold_results.append(summary)
    intervals = []
    for (fold, partition), records in sorted(comparisons.items()):
        keys = records[0][1]
        _require(all(row[1] == keys for row in records), "Los métodos no comparten las sesiones")
        effects = np.column_stack([row[2] for row in records])
        for length in block_lengths:
            result = block_intervals(
                effects, block_length=length, repetitions=repetitions, seed=seed
            )
            for column, (identity, _, _) in enumerate(records):
                intervals.append(
                    dict(
                        fold=fold,
                        partition=partition,
                        **identity,
                        block_length=length,
                        sessions=len(keys),
                        repetitions=repetitions,
                        seed=seed,
                        estimate=result["estimate"][column],
                        lower=result["lower"][column] if result["lower"] is not None else None,
                        upper=result["upper"][column] if result["upper"] is not None else None,
                        reason=result["reason"],
                    )
                )
    across = defaultdict(list)
    for row in fold_results:
        across[tuple(row[name] for name in GROUP)].append(row)
    overall = []
    for (partition, family, method), rows in sorted(across.items()):
        _require(len({row["fold"] for row in rows}) == len(rows), "Hay ventanas duplicadas")
        overall.append(
            dict(
                partition=partition,
                family=family,
                method=method,
                folds=len(rows),
                models=sum(row["models"] for row in rows),
                primary=rows[0]["primary"],
                mean_fold_session_mae=_mean(row["session_mae"] for row in rows),
                mean_fold_session_mse=_mean(row["session_mse"] for row in rows),
                minimum_fold_session_mae=min(row["session_mae"] for row in rows),
                maximum_fold_session_mae=max(row["session_mae"] for row in rows),
                mean_fold_rank_ic=_mean(row["mean_rank_ic"] for row in rows),
                mean_fold_direction_accuracy=_mean(row["mean_direction_accuracy"] for row in rows),
                selected_initial=sum(row["selected_initial"] for row in rows),
                stopped_early=sum(row["stopped_early"] for row in rows),
                **{
                    f"mean_fold_delta_{role}": _mean(row[f"delta_{role}"] for row in rows)
                    for role in CONTRASTS
                },
                **{
                    f"mean_fold_mae_{role}": _mean(row[f"mae_{role}"] for row in rows)
                    for role in CONTRASTS
                },
            )
        )
    return dict(overall=overall, by_fold=fold_results, intervals=intervals)


def _case(source, result, seconds, parent_difference):
    metadata = source["metadata"]
    selection = metadata.get("selection") or {}
    case = metadata["case"]
    scores = result["metrics"]
    primary = scores["prediction"]
    row = {
        name: source[name]
        for name in (
            "fold",
            "id",
            "partition",
            "stage",
            "phase",
            "family",
            "method",
            "seed",
            "included",
            "primary",
            "parent_id",
            "checkpoint_sha256",
            "source_report_sha256",
        )
    }
    best, last = selection.get("best_epoch"), selection.get("last_epoch")
    maximum = case.get("epochs")
    rounds = metadata.get("completed_rounds")
    early = metadata.get("stopped_early")
    if early is None:
        if last is not None and maximum is not None:
            early = last < maximum
        elif rounds is not None and case.get("rounds") is not None:
            early = rounds < case["rounds"]
    row.update(
        samples=primary["samples"],
        sessions=primary["session_count"],
        session_mae=primary["session_mae"],
        session_mse=primary["session_mse"],
        row_mae=primary["mae"],
        row_mse=primary["mse"],
        best_epoch=best,
        last_epoch=last,
        maximum_epochs=maximum,
        selected_initial=best == 0
        if source["parent_id"] is not None and best is not None
        else None,
        stopped_early=early,
        stop_reason=metadata.get("stop_reason"),
        completed_rounds=metadata.get("completed_rounds"),
        selected_round=metadata.get("selected_round"),
        global_step=metadata.get("global_step"),
        reported_attempt_seconds=None,
        parent_prediction_max_difference=parent_difference,
        checked_seconds=seconds,
        predictions_sha256=source["sha256"],
        **result["diagnostics"],
    )
    timings = [record["seconds"] for record in metadata.get("attempts", []) if "seconds" in record]
    row["reported_attempt_seconds"] = math.fsum(timings) if timings else None
    row["reported_total_seconds"] = metadata.get("total_seconds")
    totals = [
        record["total_seconds"]
        for record in metadata.get("attempts", [])
        if "total_seconds" in record
    ]
    row["reported_attempt_total_seconds"] = math.fsum(totals) if totals else None
    for role in CONTRASTS:
        row[f"mae_{role}"] = scores[role]["session_mae"] if role in scores else None
        row[f"delta_{role}"] = (
            primary["session_mae"] - scores[role]["session_mae"]
            if role in scores and (role != "parent" or source["parent_id"] is not None)
            else None
        )
    markets = sorted(set(result["sessions"]["market"].to_pylist()))
    if len(markets) > 1:
        for market in markets:
            selected = result["sessions"].filter(pc.equal(result["sessions"]["market"], market))
            row[f"sessions_{market}"] = len(selected)
            row[f"samples_{market}"] = sum(selected["samples"].to_pylist())
    return row


def _csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare_campaigns(
    reference,
    completion,
    output,
    *,
    repetitions=2000,
    seed=42,
    initial_cache_bytes=8 * 1024**2,
    input_policy=STRICT_INPUTS,
):
    """Producir agregados independientes sin cambiar los recibos científicos.

    La política de entradas se declara y la procedencia la conserva cuando no es
    la estricta. Las vistas que no se adhieren a ella se rechazan.
    """
    started = time.perf_counter()
    reference, completion, output = map(Path, (reference, completion, output))
    safe_destination(output)
    _require(not output.exists(), "La salida debe ser nueva")
    for path in (reference, completion):
        outside_source(path, output)
        _require(not path.resolve().is_relative_to(output.resolve()), "La salida contiene fuentes")
    sources, provenance = predictive_sources(reference, completion, input_policy=input_policy)
    _require(len(sources) <= 8192, "La comparación supera el presupuesto de archivos")
    folds = {fold["id"]: fold for fold in provenance["folds"]}
    initial_cache = InitialPolicyCache(initial_cache_bytes)
    cohorts, parents, cases, tables = {}, {}, [], []
    needed = {
        (row["fold"], row["partition"], row["parent_id"])
        for row in sources
        if row["parent_id"] is not None
    }
    hashed_bytes = 0
    for source in sorted(
        sources,
        key=lambda row: (row["fold"], row["partition"], row["parent_id"] is not None, row["id"]),
    ):
        began = time.perf_counter()
        cohort_key = source["fold"], source["partition"]
        result = review_predictions(
            source, cohort=cohorts.get(cohort_key), initial_cache=initial_cache
        )
        _market_population(source, result, folds[source["fold"]])
        if (source["metadata"].get("selection") or {}).get("best_epoch") == 0:
            _require(
                result["diagnostics"]["primary_equals_initial_policy"] is not False,
                "La selección inicial no coincide con su política reconstruida",
            )
        cohorts.setdefault(cohort_key, result["cohort"])
        parent_difference = None
        if source["parent_id"] is not None:
            parent_key = (*cohort_key, source["parent_id"])
            _require(parent_key in parents, "Falta el padre anterior a la continuación")
            expected = parents[parent_key]
            parent_difference = float(np.max(np.abs(result["parent"] - expected)))
            absolute, relative = (1e-9, 1e-10) if source["family"] == "ridge" else (1e-6, 1e-5)
            _require(
                np.allclose(result["parent"], expected, atol=absolute, rtol=relative),
                "La columna padre no corresponde al modelo archivado",
            )
        if _key(source) in needed:
            parents[_key(source)] = result["prediction"]
        cases.append(_case(source, result, time.perf_counter() - began, parent_difference))
        table = result["sessions"]
        for name in ("fold", "id", "partition", "family", "method", "included"):
            table = table.append_column(name, pa.array([source[name]] * len(table)))
        tables.append(table)
        hashed_bytes += source["path"].stat().st_size
    daily = pa.concat_tables(tables, promote_options="default")
    aggregates = aggregate_results(cases, daily, repetitions=repetitions, seed=seed)
    report = dict(
        schema_version=2,
        status="completed",
        created_at_utc=datetime.now(UTC).isoformat(),
        final_test_opened=False,
        candidate_implemented=False,
        candidate_trained=False,
        provenance=provenance,
        counts=provenance["counts"],
        **aggregates,
        metadata=[
            {key: source[key] for key in ("fold", "id", "metadata")}
            for source in sources
            if source["partition"] == "evaluation"
        ],
        method=dict(
            primary="session_mae",
            aggregate="equal_fold_means_of_seed_mean_errors",
            inference="exploratory_moving_session_blocks",
            block_lengths=[1, 5, 10, 20],
            repetitions=repetitions,
            seed=seed,
            confidence=0.95,
            multiplicity_adjusted=False,
            pooled_fold_interval=False,
            numeric_tolerances=dict(absolute=1e-12, relative=1e-10),
            parent_prediction_tolerances=dict(
                neural_absolute=1e-6,
                neural_relative=1e-5,
                ridge_absolute=1e-9,
                ridge_relative=1e-10,
            ),
        ),
        resources=dict(
            hashed_prediction_bytes=hashed_bytes,
            process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            gpu_used=False,
            initial_policy_cache=dict(
                limit_bytes=initial_cache.max_bytes,
                retained_bytes=initial_cache.bytes_used,
                hits=initial_cache.hits,
                misses=initial_cache.misses,
            ),
        ),
        versions={name: version(name) for name in ("numpy", "scipy", "pyarrow")},
        analysis_source_sha256={
            f"evaluation/{name}": sha256(Path(__file__).with_name(name))
            for name in (
                "comparison_sources.py",
                "prediction_statistics.py",
                "campaign_comparison.py",
            )
        }
        | {
            "training/temporal_contract.py": sha256(
                Path(__file__).parents[1] / "training/temporal_contract.py"
            )
        },
    )
    if len(set(daily["market"].to_pylist())) > 1:
        report["method"].update(
            market_stratification="separate_markets",
            markets=sorted(set(daily["market"].to_pylist())),
        )
    output.mkdir(parents=True, exist_ok=False)
    pq.write_table(daily, output / "session-errors.parquet", compression="zstd")
    for name, records in (
        ("cases.csv", cases),
        ("methods.csv", aggregates["overall"]),
        ("folds.csv", aggregates["by_fold"]),
        ("intervals.csv", aggregates["intervals"]),
    ):
        _csv(output / name, records)
    report["artifacts"] = {
        name: sha256(output / name)
        for name in (
            "session-errors.parquet",
            "cases.csv",
            "methods.csv",
            "folds.csv",
            "intervals.csv",
        )
    }
    report["resources"]["elapsed_before_receipt_seconds"] = time.perf_counter() - started
    atomic_json(output / "comparison.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "completion", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-cache-mib", type=int, default=8)
    parser.add_argument(
        "--input-policy",
        choices=INPUT_POLICIES,
        default=STRICT_INPUTS,
        help="Política de entradas declarada por las vistas (estricta si no se indica).",
    )
    args = parser.parse_args(argv)
    result = compare_campaigns(
        args.reference,
        args.completion,
        args.output,
        repetitions=args.repetitions,
        seed=args.seed,
        initial_cache_bytes=args.initial_cache_mib * 1024**2,
        input_policy=args.input_policy,
    )
    print(f"Comparados {result['counts']['models']} modelos. Test final cerrado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
