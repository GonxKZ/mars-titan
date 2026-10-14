"""Escáner de fugas sobre la matriz tabular real de cada ventana, solo en validación.

Recorre la validación de cada vista con el mismo lector y la misma matriz que la referencia
tabular (`tabular_corpus._matrix`), la guarda por columnas en un archivo temporal del destino
y la escanea con `leakage_scan.scan_blocks`. No ajusta nada ni lee calibración, evaluación
o el test reservado.

Con entradas con máscara, las columnas de un bloque ausente valen 0 en la matriz. Aquí se
tratan como ausentes, porque ese relleno empata a muchos activos de una sesión y diluiría una
fuga presente en las demás filas. Los bits de presencia se escanean como columnas, así que
una fuga que estuviera en la ausencia de un bloque también se vería.

Una ventana marcada se revisa por modalidad con `review_window`, que aporta evidencia para
decidir sobre la marca sin cambiar los umbrales ni el resultado del escaneo.
"""

import argparse
import json
import resource
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mars_titan.data.price_windows import PRICE_WINDOW_CHANNELS
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.integrity import leakage_scan
from mars_titan.models.baselines.inputs import MODALITIES
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.tabular_corpus import _matrix

KIND = "window_input_leakage_scan"
SUMMARY = "window_input_leakage_summary"
REVIEW = "window_input_leakage_review"
# Semilla de las permutaciones de la revisión, fijada al escribirla (14 de octubre de 2026).
REVIEW_SEED = 20261014
PARTITION = "validation"
CODE = (
    "integrity/window_leakage.py",
    "integrity/leakage_scan.py",
    "training/tabular_corpus.py",
    "training/corpus_inputs.py",
    "data/price_windows.py",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def column_layout(batch, masked):
    """Nombres de las columnas de la matriz y tramo de columnas de cada modalidad."""
    names, spans = [], {}
    for name in MODALITIES:
        shape = batch["inputs"][name].shape[1:]
        start = len(names)
        if name == "prices" and len(shape) == 2 and shape[1] == len(PRICE_WINDOW_CHANNELS):
            names += [f"prices.{s}.{c}" for s in range(shape[0]) for c in PRICE_WINDOW_CHANNELS]
        else:
            names += [f"{name}.{i}" for i in range(int(np.prod(shape)))]
        spans[name] = (start, len(names))
    if masked:
        names += [f"presence.{name}" for name in MODALITIES]
    return names, spans


def window_matrix(dataset, path, *, batch_size=1024):
    """Escribir la validación de una vista en `path` por columnas y devolver claves y nombres."""
    count = dataset.manifest["counts"][PARTITION]
    masked = dataset.masked
    keys = dict(market=[], asset_id=[], prediction_at=[], target=[])
    matrix, names, filled = None, None, 0
    for batch in dataset.batches(partition=PARTITION, batch_size=batch_size, epoch=0, seed=0):
        values = _matrix(batch, np.float32, presence=masked)
        size = len(values)
        if matrix is None:
            names, spans = column_layout(batch, masked)
            _require(len(names) == values.shape[1], "Los nombres no cubren la matriz tabular")
            matrix = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float32, shape=(count, len(names)), fortran_order=True
            )
        _require(filled + size <= count, "La validación supera su recuento declarado")
        if masked:
            for index, name in enumerate(MODALITIES):
                start, stop = spans[name]
                values[~batch["presence"][:, index], start:stop] = np.nan
        matrix[filled : filled + size] = values
        keys["market"] += list(batch["market"])
        keys["asset_id"] += ["/".join(s.split("/")[:2]) for s in batch["sample_ids"]]
        keys["prediction_at"].append(batch["prediction_at"])
        keys["target"].append(batch["target"])
        filled += size
    _require(matrix is not None and filled == count, "La validación no tiene sus filas declaradas")
    matrix.flush()
    frame = pd.DataFrame(
        dict(
            market=keys["market"],
            asset_id=keys["asset_id"],
            prediction_at=np.concatenate(keys["prediction_at"]),
            target=np.concatenate(keys["target"]),
        )
    )
    return frame, matrix, names


def scan_window(manifest, destination, *, input_policy, lag, chunk=32, batch_size=1024):
    """Escanear la validación de una vista y publicar su informe en `destination`."""
    started, clock = time.perf_counter(), time.process_time()
    dataset = CorpusDataset(Path(manifest), input_policy=input_policy)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = destination.with_name(f".{destination.stem}.matrix.npy")
    scratch.unlink(missing_ok=True)
    try:
        keys, matrix, names = window_matrix(dataset, scratch, batch_size=batch_size)
        _require(
            keys["prediction_at"].max() < np.datetime64("2024-01-01"),
            "La validación de la vista llega al test reservado",
        )
        blocks = (
            (names[start : start + chunk], matrix[:, start : start + chunk])
            for start in range(0, len(names), chunk)
        )
        result = leakage_scan.scan_blocks(keys, blocks, lag=lag, chunk=chunk)
    finally:
        scratch.unlink(missing_ok=True)
    root = Path(__file__).parents[1]
    report = dict(
        result,
        kind=KIND,
        partition=PARTITION,
        view_sha256=dataset.identity,
        input_policy=input_policy,
        features=len(names),
        checked_columns=sum(entry["sessions"] > 0 for entry in result["columns"]),
        absent_blocks="missing",
        first_prediction_at=str(keys["prediction_at"].min()),
        last_prediction_at=str(keys["prediction_at"].max()),
        code={name: sha256(root / name) for name in CODE},
        seconds=time.perf_counter() - started,
        cpu_seconds=time.process_time() - clock,
        max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        final_test_opened=False,
    )
    atomic_json(destination, report)
    return report


def review_window(
    manifest,
    report_path,
    modality,
    destination,
    *,
    input_policy,
    permutations,
    seed=REVIEW_SEED,
    batch_size=1024,
):
    """Revisar las columnas marcadas de una modalidad en la validación de una vista.

    Solo cuentan para la regla las sesiones con al menos `min_assets` filas con la modalidad
    presente. La revisión reúne esas filas y describe cada sesión: activos y vectores distintos
    de la modalidad, porque muchos activos con el mismo vector reducen la muestra efectiva. Con
    `leakage_scan.permutation_null` repite la puntuación del escáner, que debe marcar las mismas
    columnas que el informe, y mide cuántas marcaría la regla por azar con esas mismas sesiones.
    No cambia ningún umbral ni el resultado del escaneo.
    """
    started, clock = time.perf_counter(), time.process_time()
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    dataset = CorpusDataset(Path(manifest), input_policy=input_policy)
    _require(
        report.get("kind") == KIND and report.get("view_sha256") == dataset.identity,
        "El informe no es el escaneo de esta vista",
    )
    _require(dataset.masked and modality in MODALITIES, "La revisión necesita una modalidad")
    marked = [name for name in report["flagged"] if name.startswith(f"{modality}.")]
    _require(marked, f"El informe no marca ninguna columna de {modality}")
    index = MODALITIES.index(modality)
    keys = dict(market=[], asset_id=[], prediction_at=[], target=[])
    values, names = [], None
    for batch in dataset.batches(partition=PARTITION, batch_size=batch_size, epoch=0, seed=0):
        if names is None:
            names, spans = column_layout(batch, True)
            start, stop = spans[modality]
            names = names[start:stop]
        present = batch["presence"][:, index]
        if not present.any():
            continue
        values.append(_matrix(batch, np.float32, presence=True)[present, start:stop])
        keys["market"] += list(np.asarray(batch["market"])[present])
        keys["asset_id"] += [
            "/".join(s.split("/")[:2]) for s in np.asarray(batch["sample_ids"])[present]
        ]
        keys["prediction_at"].append(batch["prediction_at"][present])
        keys["target"].append(batch["target"][present])
    _require(values, f"Ninguna fila de validación tiene {modality}")
    frame = pd.DataFrame(
        dict(
            market=keys["market"],
            asset_id=keys["asset_id"],
            prediction_at=np.concatenate(keys["prediction_at"]),
            target=np.concatenate(keys["target"]),
        )
    )
    values = np.concatenate(values)
    thresholds = report["thresholds"]
    sizes = frame.groupby(list(leakage_scan.KEYS))["target"].transform("size").to_numpy()
    kept = sizes >= thresholds["min_assets"]
    _require(kept.any(), f"Ninguna sesión tiene {thresholds['min_assets']} filas con {modality}")
    frame, values = frame[kept].reset_index(drop=True), values[kept]
    result = leakage_scan.permutation_null(
        frame, values, names, permutations=permutations, seed=seed, **thresholds
    )
    _require(
        result["observed"]["flagged"] == marked,
        "La revisión no reproduce las columnas que marca el informe",
    )
    scanned = {entry["column"]: entry["mean_abs_ic"] for entry in report["columns"]}
    differences = [
        abs(entry["mean_abs_ic"] - scanned[entry["column"]])
        for entry in result["observed"]["columns"]
        if entry["mean_abs_ic"] is not None
    ]
    sessions = [
        dict(
            market=market,
            prediction_at=str(moment),
            assets=len(rows),
            distinct_vectors=len(np.unique(values[rows], axis=0)),
        )
        for (market, moment), rows in frame.groupby(list(leakage_scan.KEYS)).indices.items()
    ]
    root = Path(__file__).parents[1]
    review = dict(
        schema_version=1,
        kind=REVIEW,
        created_at_utc=datetime.now(UTC).isoformat(),
        partition=PARTITION,
        modality=modality,
        view_sha256=dataset.identity,
        report_sha256=sha256(report_path),
        input_policy=input_policy,
        session_detail=sessions,
        max_difference_from_report=max(differences),
        **result,
        code={name: sha256(root / name) for name in CODE},
        seconds=time.perf_counter() - started,
        cpu_seconds=time.process_time() - clock,
        max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        final_test_opened=False,
    )
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, review)
    return review


def _strongest(report, count=5):
    defined = [entry for entry in report["columns"] if entry["mean_abs_ic"] is not None]
    defined.sort(key=lambda entry: entry["mean_abs_ic"], reverse=True)
    return [
        {key: entry[key] for key in ("column", "mean_abs_ic", "max_abs_ic", "past_mean_abs_ic")}
        for entry in defined[:count]
    ]


def scan_scope(campaign_path, scope, views, output, *, lag, windows=None, chunk=32):
    """Escanear las ventanas de un ámbito, reanudando las que ya tienen informe de su vista."""
    from mars_titan.training.masked_campaign import load_campaign, scope_views

    campaign = load_campaign(campaign_path)
    checked = scope_views(views, scope, campaign)
    selected = list(checked["windows"]) if windows is None else list(windows)
    _require(
        selected and set(selected) <= set(checked["windows"]),
        "Las ventanas escaneadas deben ser del ámbito",
    )
    folder = Path(output) / scope
    folder.mkdir(parents=True, exist_ok=True)
    rows = {}
    for window in selected:
        record = checked["windows"][window]
        destination = folder / f"{window}.json"
        report = json.loads(destination.read_text()) if destination.exists() else None
        if report is None or report.get("view_sha256") != record["sha256"]:
            report = scan_window(
                record["path"],
                destination,
                input_policy=campaign["input_policy"],
                lag=lag,
                chunk=chunk,
            )
        _require(report["placebo_lag"] == lag, f"{window} se escaneó con otro retardo")
        rows[window] = dict(
            view_sha256=report["view_sha256"],
            report_sha256=sha256(destination),
            rows=report["rows"],
            sessions=report["sessions"],
            features=report["features"],
            checked_columns=report["checked_columns"],
            flagged=report["flagged"],
            strongest=_strongest(report),
            seconds=report["seconds"],
        )
    summary = dict(
        schema_version=1,
        kind=SUMMARY,
        created_at_utc=datetime.now(UTC).isoformat(),
        campaign_sha256=sha256(Path(campaign_path)),
        scope=scope,
        views_report_sha256=checked["report_sha256"],
        partition=PARTITION,
        placebo_lag=lag,
        complete=set(rows) == set(checked["windows"]),
        windows=rows,
        flagged={window: row["flagged"] for window, row in rows.items() if row["flagged"]},
        unchecked=[window for window, row in rows.items() if not row["checked_columns"]],
        final_test_opened=False,
    )
    # Una ventana sin ninguna sesión con correlación definida no ha comprobado nada.
    summary["passed"] = summary["complete"] and not summary["flagged"] and not summary["unchecked"]
    atomic_json(folder / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--views", type=Path, required=True, help="Carpeta de vistas del ámbito")
    parser.add_argument("--lag", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", action="append", help="Por defecto, todas las del ámbito")
    parser.add_argument("--review", help="Ventana escaneada cuya marca se revisa")
    parser.add_argument("--modality", choices=MODALITIES, help="Modalidad que se revisa")
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=REVIEW_SEED)
    args = parser.parse_args(argv)
    if args.review is not None:
        from mars_titan.training.masked_campaign import load_campaign, scope_views

        _require(args.modality is not None, "La revisión necesita --modality")
        campaign = load_campaign(args.campaign)
        record = scope_views(args.views, args.scope, campaign)["windows"][args.review]
        folder = Path(args.output) / args.scope
        review = review_window(
            record["path"],
            folder / f"{args.review}.json",
            args.modality,
            folder / f"{args.review}.review-{args.modality}.json",
            input_policy=campaign["input_policy"],
            permutations=args.permutations,
            seed=args.seed,
        )
        print(
            json.dumps(
                dict(
                    observed=len(review["observed"]["flagged"]),
                    null=review["null"]["flagged_quantiles"],
                )
            )
        )
        return 0
    summary = scan_scope(
        args.campaign, args.scope, args.views, args.output, lag=args.lag, windows=args.window
    )
    print(json.dumps({key: summary[key] for key in ("scope", "complete", "flagged", "passed")}))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
