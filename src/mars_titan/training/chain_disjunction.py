"""Verificador de disjunción por ventana del walk-forward por etapas, sin ajustar nada.

Lee las vistas de cada ámbito y, si existen, los recibos de la campaña base y de la cadena,
y demuestra con recuentos y huellas (`campaign_chain.row_fingerprint`) que:

1. las filas nuevas del posentrenamiento de la ventana k (tramo `train` de la vista k con
   decisión en `campaign_chain.posttraining_rows`) no cortan las filas de ajuste,
   validación ni calibración de la vista k-1 del padre, y que toda etiqueta del padre
   madura antes de la primera decisión nueva,
2. cada recibo de la cadena, que alimenta la cinta de su ventana en la RL, declara una
   última etiqueta usada anterior a la primera decisión de su evaluación y no menor que la
   maduración de las filas que fijaron su estado,
3. las filas de test son las mismas para todas las familias: cada evaluación tiene una sola
   huella de filas en los recibos base, la misma huella por mercado en las ventanas de
   ámbitos distintos con los mismos tramos, y el mismo número de filas en la cadena,
4. ninguna fila de 2024 aparece en las vistas ni en los recibos, y cada fila cae en su
   tramo con la etiqueta madura antes de su final.

Lee solo cuatro columnas de las etiquetas, por activo y en paralelo acotado. No carga
modelos, no reserva la GPU y no escribe más que su informe. Con `walk_forward_stages`
declarado, las etapas de adaptadores y de políticas no se ejecutan sin un informe sin
fallos que cubra las vistas y las selecciones que usan (`require_report`).
"""

import argparse
import json
import multiprocessing
import resource
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.cohort_files import read_manifest
from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.evaluation.splits import PARTITIONS

from . import campaign_chain as chain

KIND = "campaign_chain_disjunction"
COLUMNS = ["sample_row", "prediction_at", "target_available_at", "partition"]
PARENT_PARTITIONS = ("train", "validation", "calibration")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _micros(day):
    return int(np.datetime64(day, "us").astype(np.int64))


def _labels(manifest_path):
    """Lee del manifiesto de una vista la raíz de sus etiquetas y la lista de activos."""
    manifest, _ = read_manifest(Path(manifest_path), 64 * 1024**2)
    root = Path(manifest["roots"]["labels"])
    return root, [(asset["market"], asset["symbol"]) for asset in manifest["assets"]]


def check_asset(task):
    """Cuenta y resume las filas de un activo en todas las ventanas de un ámbito.

    Devuelve por ventana los recuentos, las huellas y los extremos de decisión y
    maduración. Recorre las ventanas en orden para comparar las filas nuevas de cada una
    con las que usó el padre en la anterior sin volver a leerlas.

    `task` es (mercado, activo, ventanas) con cada ventana como (id, raíz de etiquetas,
    tramos en microsegundos, intervalo de filas nuevas o None).
    """
    market, symbol, windows = task
    result, parent = {}, None
    for window, root, bounds, fresh in windows:
        path = Path(root) / market / symbol / "labels.parquet"
        if not path.is_file():
            result[window] = None
            parent = None
            continue
        table = pq.read_table(path, columns=COLUMNS)
        rows = table["sample_row"].to_numpy()
        moment = table["prediction_at"].cast(pa.int64()).to_numpy()
        maturity = table["target_available_at"].cast(pa.int64()).to_numpy()
        partition = np.asarray(table["partition"].to_pylist(), dtype=object)
        record = dict(counts={}, misplaced=0, after_cutoff=0)
        for name in PARTITIONS:
            mask = partition == name
            start, end = bounds[name]
            record["counts"][name] = int(mask.sum())
            record["misplaced"] += int(
                (mask & ((moment < start) | (moment >= end) | (maturity >= end))).sum()
            )
            record["after_cutoff"] += int(
                (mask & ((moment >= FINAL_TEST_START_US) | (maturity >= FINAL_TEST_START_US))).sum()
            )
            if mask.any():
                record[f"{name}_max_maturity"] = int(maturity[mask].max())
                record[f"{name}_min_decision"] = int(moment[mask].min())
        evaluation = np.sort(rows[partition == "evaluation"])
        record["evaluation_digest"] = chain.asset_digest(evaluation) if len(evaluation) else None
        used = np.isin(partition, PARENT_PARTITIONS)
        if fresh is not None:
            new = (partition == "train") & (moment >= fresh[0]) & (moment < fresh[1])
            record["new_rows"] = int(new.sum())
            record["new_digest"] = chain.asset_digest(rows[new]) if new.any() else None
            if new.any():
                record["new_min_decision"] = int(moment[new].min())
                record["new_max_decision"] = int(moment[new].max())
                record["new_max_maturity"] = int(maturity[new].max())
            parent_rows = np.empty(0, dtype=np.int64) if parent is None else parent[0]
            record["overlap"] = int(np.isin(rows[new], parent_rows).sum())
            record["parent_rows"] = int(len(parent_rows))
            if parent is not None and parent[1] is not None:
                record["parent_max_maturity"] = parent[1]
        parent = (rows[used], int(maturity[used].max()) if used.any() else None)
        result[window] = record
    return market, symbol, result


def _fold_micros(fold):
    return {name: (_micros(fold[name][0]), _micros(fold[name][1])) for name in PARTITIONS}


def _scan(campaign, scope, directory, workers):
    """Recorre en paralelo los activos de un ámbito y agrupa sus resúmenes por ventana."""
    from .masked_campaign import scope_views

    checked = scope_views(directory, scope, campaign)
    ordered = chain.scope_windows(campaign, scope)
    folds = dict(ordered)
    windows, assets = [], set()
    for index, (window, fold) in enumerate(ordered):
        root, listed = _labels(checked["windows"][window]["path"])
        assets.update(listed)
        fresh = None
        if index:
            start, end = chain.posttraining_rows(folds[ordered[index - 1][0]], fold)
            fresh = (_micros(start), _micros(end))
        windows.append((window, str(root), _fold_micros(fold), fresh))
    tasks = [(market, symbol, windows) for market, symbol in sorted(assets)]
    per_window = defaultdict(lambda: defaultdict(list))
    # Se arrancan procesos nuevos porque el lector de Arrow ya tiene hilos y bifurcar un
    # proceso con hilos puede dejar cerrojos tomados en el hijo.
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for market, symbol, result in pool.map(check_asset, tasks, chunksize=16):
            for window, record in result.items():
                if record is not None:
                    per_window[window][market].append((symbol, record))
    return checked, ordered, per_window


def _summary(market_records, fresh):
    """Agrega los resúmenes de los activos de una ventana por mercado y en conjunto."""
    markets, new_parts, totals = {}, {}, defaultdict(int)
    extremes = {}
    for market, records in sorted(market_records.items()):
        counts = {name: sum(r["counts"][name] for _, r in records) for name in PARTITIONS}
        digests = {
            (market, symbol): r["evaluation_digest"]
            for symbol, r in records
            if r["evaluation_digest"] is not None
        }
        evaluation = chain.combine(digests)
        first = [r["evaluation_min_decision"] for _, r in records if "evaluation_min_decision" in r]
        markets[market] = dict(
            counts=counts,
            evaluation=dict(rows=evaluation[0], sha256=evaluation[1]),
            evaluation_first_decision=min(first) if first else None,
            max_maturity={
                name: max(
                    (r[f"{name}_max_maturity"] for _, r in records if f"{name}_max_maturity" in r),
                    default=None,
                )
                for name in PARENT_PARTITIONS
            },
        )
        for _, r in records:
            totals["misplaced"] += r["misplaced"]
            totals["after_cutoff"] += r["after_cutoff"]
            if fresh is not None:
                totals["overlap"] += r["overlap"]
                totals["new_rows"] += r["new_rows"]
                totals["parent_rows"] += r["parent_rows"]
                for key, pick in (
                    ("new_min_decision", min),
                    ("new_max_decision", max),
                    ("new_max_maturity", max),
                    ("parent_max_maturity", max),
                ):
                    if key in r:
                        extremes[key] = pick(extremes.get(key, r[key]), r[key])
        if fresh is not None:
            new_parts.update(
                {
                    (market, symbol): r["new_digest"]
                    for symbol, r in records
                    if r["new_digest"] is not None
                }
            )
    result = dict(
        markets=markets, misplaced=totals["misplaced"], after_cutoff=totals["after_cutoff"]
    )
    if fresh is not None:
        rows, digest = chain.combine(new_parts)
        result["posttraining"] = dict(
            fit=list(fresh),
            new_rows=dict(rows=rows, sha256=digest),
            parent_rows=totals["parent_rows"],
            overlap=totals["overlap"],
            **extremes,
        )
    return result


def _base_receipts(campaign_output):
    """Reúne por ámbito y ventana las huellas de filas de evaluación de los recibos base.

    Solo lee recibos confirmados. Un trabajo sin predicciones de evaluación no aporta huella.
    """
    found = defaultdict(dict)
    for path in sorted(Path(campaign_output).glob("jobs/**/receipt.json")):
        receipt, _ = read_manifest(path, 64 * 1024**2)
        identity = receipt.get("identity", {})
        record = receipt.get("predictions", {}).get("evaluation")
        if record is None:
            continue
        key = identity["scope"], identity["window"]
        found[key][identity["id"]] = record["rows_sha256"]
    return found


def _chain_receipts(root, campaign, scope, ordered):
    """Lee las selecciones confirmadas de la cadena de un ámbito, agrupadas por ventana.

    Cada una debe partir de la ventana anterior, que es la única que el diseño admite como
    padre.
    """
    found = defaultdict(list)
    for window, _ in ordered:
        folder = Path(root) / "windows" / scope / window
        for selection in sorted(folder.glob(f"*{chain.CHAIN_SUFFIX}/seed-*/{chain.SELECTION}")):
            base_arm = selection.parent.parent.name.removesuffix(chain.CHAIN_SUFFIX)
            seed = int(selection.parent.name.removeprefix("seed-"))
            document = chain.read_selection(root, scope, window, base_arm, seed)
            _require(
                document["parent_window"] == chain.parent_window(campaign, scope, window),
                f"La selección de {selection.parent} no parte de la ventana anterior",
            )
            found[window].append(document)
    return found


def verify(campaign, views, *, campaign_output=None, posttraining=None, workers=4, scopes=None):
    """Comprueba las cuatro disjunciones y devuelve el informe con sus fallos.

    `scopes` limita la comprobación a algunos ámbitos de la campaña, todos por omisión.
    """
    selected = list(campaign["scopes"] if scopes is None else scopes)
    _require(set(selected) <= set(campaign["scopes"]), "Algún ámbito no es de la campaña")
    failures, scopes = [], {}
    evaluations = defaultdict(dict)
    for scope in selected:
        checked, ordered, per_window = _scan(campaign, scope, views[scope], workers)
        folds = dict(ordered)
        report = {}
        for index, (window, fold) in enumerate(ordered):
            fresh = None
            if index:
                fresh = chain.posttraining_rows(folds[ordered[index - 1][0]], fold)
            summary = _summary(per_window.get(window, {}), fresh)
            summary["manifest_sha256"] = checked["windows"][window]["sha256"]
            label = f"{scope}/{window}"
            if summary["misplaced"]:
                failures.append(f"{label}: {summary['misplaced']} filas fuera de su tramo")
            if summary["after_cutoff"]:
                failures.append(f"{label}: {summary['after_cutoff']} filas de 2024")
            post = summary.get("posttraining")
            if post is not None:
                start = _micros(fresh[0])
                if post["overlap"]:
                    failures.append(f"{label}: {post['overlap']} filas nuevas ya las usó el padre")
                if post.get("parent_max_maturity", start - 1) >= start:
                    failures.append(f"{label}: una etiqueta del padre madura con las filas nuevas")
                if post["new_rows"]["rows"] == 0:
                    failures.append(f"{label}: el posentrenamiento no tiene filas nuevas")
            bounds = tuple(tuple(fold[name]) for name in PARTITIONS)
            for market, record in summary["markets"].items():
                evaluations[bounds, market][label] = record["evaluation"]
            report[window] = summary
        scopes[scope] = report
    # 3. Las mismas filas de test por mercado en los ámbitos con los mismos tramos.
    for (_, market), found in evaluations.items():
        if len({json.dumps(v, sort_keys=True) for v in found.values()}) > 1:
            failures.append(f"{market}: evaluaciones distintas en {', '.join(sorted(found))}")
    receipts = {}
    if campaign_output is not None:
        for (scope, window), found in _base_receipts(campaign_output).items():
            distinct = sorted(set(found.values()))
            receipts[f"{scope}/{window}"] = dict(jobs=len(found), evaluation_rows_sha256=distinct)
            if len(distinct) > 1:
                failures.append(f"{scope}/{window}: los recibos base evalúan filas distintas")
    selections = {}
    if posttraining is not None:
        for scope in selected:
            ordered = chain.scope_windows(campaign, scope)
            for window, documents in _chain_receipts(
                posttraining, campaign, scope, ordered
            ).items():
                summary = scopes[scope][window]
                for document in documents:
                    label = selection_label(scope, window, document["base_arm"], document["seed"])
                    found, evidence = _check_selection(label, document, summary)
                    failures += found
                    selections[label] = dict(
                        selected=document["selected"], tapes=evidence, sha256=document["sha256"]
                    )
    return dict(
        schema_version=1,
        kind=KIND,
        campaign_sha256=campaign["sha256"],
        scopes=scopes,
        base_receipts=receipts,
        selections=selections,
        failures=failures,
        training_executed=False,
        final_test_opened=False,
    )


def selection_label(scope, window, base_arm, seed):
    """Etiqueta con la que el informe registra una selección de la cadena."""
    return f"{scope}/{window}/{chain.chain_arm(base_arm)}/seed-{seed}"


def used_views(views, scopes, pairs=None):
    """Huellas de las vistas que usará una etapa por ámbito y ventana, con `pairs` si limita.

    `views` es el resultado de `masked_campaign.scope_views` por ámbito, el mismo del que
    sale `manifest_sha256` en el informe.
    """
    return {
        scope: {
            window: record["sha256"]
            for window, record in views[scope]["windows"].items()
            if pairs is None or (scope, window) in pairs
        }
        for scope in scopes
    }


def require_report(path, campaign, views, *, selections=None):
    """Exige un informe sin fallos de esta campaña que cubra lo que va a leer una etapa.

    `views` asigna a cada ámbito las huellas de las vistas por ventana que usará la etapa y
    `selections`, a la etiqueta de cada selección de la cadena que leerá, la huella de su
    `selection.json`. El informe debe haber comprobado esas mismas vistas y selecciones. Así
    el posentrenamiento no ajusta sin haber demostrado las disjunciones sobre sus vistas y
    la RL no lee una cadena que el verificador no haya contrastado con ellas. Devuelve la
    huella del informe.
    """
    _require(
        path is not None,
        "La campaña declara el walk-forward por etapas y falta el informe de disjunción",
    )
    report, digest = read_manifest(Path(path), 256 * 1024**2)
    _require(
        isinstance(report, dict)
        and report.get("kind") == KIND
        and report.get("schema_version") == 1
        and report.get("campaign_sha256") == campaign["sha256"]
        and report.get("failures") == []
        and report.get("training_executed") is False
        and report.get("final_test_opened") is False,
        "El informe de disjunción no es de esta campaña o registra fallos",
    )
    for scope, windows in views.items():
        for window, view in windows.items():
            checked = report["scopes"].get(scope, {}).get(window, {})
            _require(
                checked.get("manifest_sha256") == view,
                f"El informe de disjunción no comprobó la vista {scope}/{window} que se usará",
            )
    for label, value in (selections or {}).items():
        _require(
            report["selections"].get(label, {}).get("sha256") == value,
            f"El informe de disjunción no comprobó la selección {label} que se leerá",
        )
    return digest


def _check_selection(label, document, summary):
    """Comprueba una selección de la cadena frente a las vistas de su ventana.

    `read_window_receipt` ya exige que la última etiqueta usada sea anterior al inicio de la
    evaluación, y toda decisión de la cinta es posterior a ese inicio. Aquí se exige además
    que no sea anterior a la calibración común de la ventana, la más tardía de las etiquetas
    que fijan el estado, que la evaluación tenga las filas de la vista y que el ajuste use
    las filas nuevas. Devuelve los fallos y, por mercado, las cifras que lo demuestran.
    """
    failures, evidence = [], {}
    until, markets = document["labels_used_until"], summary["markets"]
    calibration = [
        record["max_maturity"]["calibration"]
        for record in markets.values()
        if record["max_maturity"]["calibration"] is not None
    ]
    if calibration and until < max(calibration):
        failures.append(f"{label}: declara una última etiqueta anterior a las que usó")
    for market, receipt in document["receipts"].items():
        record = markets.get(market, dict(evaluation=dict(rows=0)))
        declared = dict(receipt.predictions).get("evaluation")
        if declared is None or declared[0] != record["evaluation"]["rows"]:
            failures.append(f"{label}: la evaluación de {market} no tiene las filas de la vista")
        evidence[market] = dict(
            labels_used_until=until,
            first_decision=record.get("evaluation_first_decision"),
            evaluation_rows=record["evaluation"]["rows"],
        )
    if document["selected"]["kind"] in ("adapter", "continuation"):
        fit, post = document["fit_rows"], summary["posttraining"]
        if (fit["rows"], fit["sha256"]) != (post["new_rows"]["rows"], post["new_rows"]["sha256"]):
            failures.append(f"{label}: las filas de ajuste no son las filas nuevas de la vista")
    return failures, evidence


def main(argv=None):
    from .campaign_plan import load_campaign
    from .masked_campaign import _views_argument

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--views", action="append", required=True, help="ÁMBITO=DIRECTORIO")
    parser.add_argument("--campaign-output", type=Path)
    parser.add_argument("--posttraining", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scope", action="append", help="Comprobar solo estos ámbitos")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    campaign = load_campaign(args.campaign)
    report = verify(
        campaign,
        _views_argument(args.views),
        campaign_output=args.campaign_output,
        posttraining=args.posttraining,
        workers=args.workers,
        scopes=args.scope,
    )
    report.update(
        checked_at_utc=datetime.now(UTC).isoformat(),
        elapsed_seconds=time.perf_counter() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    from mars_titan.data.storage import atomic_json

    atomic_json(args.output, report)
    print(json.dumps(dict(failures=report["failures"], output=str(args.output)), indent=2))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
