"""Ventanas de los controles comunes de la criba (#746): la regla y LightGBM.

Cada ventana lee la vista de su origen con la edición histórica con máscaras y escribe, como un
ajuste de la criba, la predicción de validación, calibración y evaluación con exactamente las
filas y los objetivos de la vista (`check_view_rows`). Las tablas siguen el orden canónico de
`ForecastPanel` y su `sample_id` es activo/instante en µs, como las de los ajustes.

- La regla de momentum y reversión no ajusta nada. Una ventana sirve para todas las semillas.
- LightGBM se ajusta con las filas de entrenamiento decididas en los meses de ajuste de la
  criba, las mismas que usan los modelos (`window_phases`). Ajusta las cinco semillas en el
  mismo trabajo, elige sus rondas con la época común de todas y deja una carpeta por semilla.
  Cada booster se guarda con su serie de validación al terminar su ajuste, así que un
  intento interrumpido solo ajusta las semillas que faltan. Los ajustes guardados se
  borran cuando la ventana confirma sus predicciones.

Las dos comprueban la protección del aprendizaje antes de leer la vista y se detienen en la
siguiente barrera si el presupuesto del trabajo lo pide.
"""

import dataclasses
import hashlib
import importlib
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa

from mars_titan.data.batches import atomic_parquet_batches
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.evaluation import scale_diagnostics as sd
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.splits import build_folds
from mars_titan.models.baselines import momentum_reversal
from mars_titan.models.baselines import same_inputs_lightgbm as lightgbm_control
from mars_titan.models.titans.config import canonical

from . import selection
from .candidate_walk_forward import check_view_rows
from .corpus_inputs import CorpusDataset
from .learning_hold import require_learning_allowed
from .temporal_contract import temporal_contracts
from .walk_forward_phases import PREDICTED, window_phases

RULE_KIND = "common_rule_window_v1"
TABULAR_KIND = "common_tabular_window_v1"
# Filas por tanda de lectura, el máximo que admite el corpus.
BLOCK_ROWS = 4096
# Un booster de 1.000 rondas con 210 hojas ocupa unos 25 MB de texto.
FIT_BYTES = 256 * 1024**2
_CODE = (
    "mars_titan.training.common_control_windows",
    "mars_titan.models.baselines.momentum_reversal",
    "mars_titan.models.baselines.same_inputs_lightgbm",
    "mars_titan.training.corpus_inputs",
)


class Stopped(Exception):
    """El presupuesto del trabajo pidió parar en la siguiente barrera."""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclasses.dataclass(frozen=True)
class Rows:
    """Filas de un tramo en el orden canónico de su panel, con sus entradas en el mismo orden.

    `panel` lleva las claves y el residuo con la predicción a cero. `features` solo existe si
    se pidió, porque la regla solo lee la ventana de precios.
    """

    panel: ForecastPanel
    asset_id: np.ndarray
    prices: np.ndarray
    features: np.ndarray | None


def partition_rows(dataset, partition, *, with_features, since=None):
    """Filas de un tramo de la vista o, con `since`, las decididas desde ese instante en µs."""
    columns = dict(asset_id=[], market=[], prediction_at=[], target=[], prices=[], features=[])
    for batch in dataset.batches(partition=partition, batch_size=BLOCK_ROWS, epoch=0, seed=0):
        moments = batch["prediction_at"].astype("datetime64[us]").astype(np.int64)
        keep = np.ones(len(moments), dtype=bool) if since is None else moments >= since
        if not keep.any():
            continue
        # El identificador del corpus es activo/instante y el activo es mercado/símbolo.
        assets = np.asarray([sample.rsplit("/", 1)[0] for sample in batch["sample_ids"]])
        columns["asset_id"].append(assets[keep])
        columns["market"].append(np.asarray(batch["market"])[keep])
        columns["prediction_at"].append(moments[keep])
        columns["target"].append(batch["target"][keep])
        columns["prices"].append(batch["inputs"]["prices"][keep])
        if with_features:
            inputs = {name: values[keep] for name, values in batch["inputs"].items()}
            columns["features"].append(lightgbm_control.features(inputs, batch["presence"][keep]))
    _require(columns["target"], f"El tramo {partition} no tiene filas que leer")
    merged = {name: np.concatenate(parts) for name, parts in columns.items() if parts}
    ids = pa.array(
        [f"{a}/{t}" for a, t in zip(merged["asset_id"], merged["prediction_at"], strict=True)]
    )
    # La posición de cada fila como predicción provisional devuelve el orden canónico del panel.
    panel = ForecastPanel.from_columns(
        ids,
        merged["market"],
        merged["prediction_at"],
        merged["target"],
        np.arange(len(ids), dtype=np.float64),
    )
    order = panel.prediction.astype(np.int64)
    return Rows(
        panel=dataclasses.replace(panel, prediction=np.zeros(panel.rows)),
        asset_id=merged["asset_id"][order],
        prices=merged["prices"][order],
        features=merged["features"][order] if with_features else None,
    )


def _fold(dataset, protocol, window):
    """Ventana declarada del protocolo, que la vista conserva en cada mercado."""
    folds = {fold["id"]: fold for fold in build_folds(protocol)}
    _require(window in folds, "La ventana no pertenece al protocolo")
    contracts = temporal_contracts(dataset.manifest, input_policy=HISTORICAL_MASKED)
    common = {key: value for key, value in protocol.items() if key != "market"}
    _require(
        contracts
        and all(
            {key: value for key, value in contract["protocol"].items() if key != "market"} == common
            and contract["fold"] == folds[window]
            for contract in contracts.values()
        )
        and dataset.manifest.get("final_test_opened") is False,
        "La vista no conserva el protocolo y la ventana declarados o abre el test final",
    )
    return folds[window]


def _code():
    return {name: sha256(Path(importlib.import_module(name).__file__)) for name in _CODE}


def _protocol(protocol):
    document = protocol if isinstance(protocol, dict) else read_manifest(Path(protocol))[0]
    return document, hashlib.sha256(canonical(document).encode()).hexdigest()


def _records(output, report):
    """Archivos confirmados de una ventana con su huella, también los de cada semilla."""
    for record in report.get("predictions", {}).values():
        yield output / record["path"], record["sha256"]
    for member in report.get("seeds", {}).values():
        path = output / member["path"]
        yield path, member["sha256"]
        nested = read_manifest(path, 16 * 1024**2)[0]
        for record in nested["predictions"].values():
            yield path.parent / record["path"], record["sha256"]


def _start(output, request, kind):
    """Informe de la ventana: el terminado se verifica y se devuelve, el resto se repite."""
    safe_destination(output)
    path = output / "run.json"
    if path.exists():
        report = read_manifest(path, 16 * 1024**2)[0]
        _require(report.get("request") == request, "La petición de la ventana ha cambiado")
        if report.get("status") == "completed":
            for item, digest in _records(output, report):
                _require(sha256(item) == digest, f"Ha cambiado {item.name} de la ventana")
        return report
    _require(
        not output.exists() or not any(output.iterdir()),
        "La salida contiene datos de otra ejecución",
    )
    report = dict(
        schema_version=1,
        kind=kind,
        request=request,
        status="running",
        started_at_utc=datetime.now(UTC).isoformat(),
        final_test_opened=False,
    )
    atomic_json(path, report)
    return report


def _open(view, protocol, window, output):
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    fold = _fold(dataset, protocol, window)
    for protected in (*dataset.roots.values(), Path(view).parent):
        outside_source(protected, output)
    return dataset, fold


def _write(folder, partition, rows, prediction, dataset):
    """Escribir la predicción de un tramo, ya conciliada con las filas de la vista."""
    panel = rows.panel
    _require(
        prediction.shape == (panel.rows,) and np.isfinite(prediction).all(),
        f"La predicción de {partition} no tiene un valor finito por fila",
    )
    _require(
        int(panel.prediction_at.max()) < FINAL_TEST_START_US,
        "La ventana no puede predecir el test final",
    )
    table = pa.table(
        dict(
            sample_id=panel.row_id.cast(pa.string()),
            asset_id=pa.array(rows.asset_id, type=pa.string()),
            market=pa.array(np.asarray(panel.markets)[panel.market], type=pa.string()),
            prediction_at=pa.array(panel.prediction_at, type=pa.timestamp("us", tz="UTC")),
            target=pa.array(panel.target, type=pa.float64()),
            prediction=pa.array(prediction, type=pa.float64()),
            zero=pa.array(np.zeros(panel.rows), type=pa.float64()),
        )
    )
    check_view_rows(table, dataset, partition)
    path = Path(folder) / f"{partition}-predictions.parquet"
    written = atomic_parquet_batches(path, [table])
    _require(written == panel.rows, f"El Parquet de {partition} no conserva sus filas")
    return dict(path=path.name, sha256=sha256(path), rows=written, bytes=path.stat().st_size)


def _finish(output, report, **fields):
    report.update(status="completed", finished_at_utc=datetime.now(UTC).isoformat(), **fields)
    atomic_json(output / "run.json", report)
    return report


def run_rule_window(view, protocol, window, *, output, config, stop=None):
    """Predicción de la regla de momentum y reversión en los tres tramos de la ventana."""
    require_learning_allowed("la ventana de la regla de los controles comunes")
    config = momentum_reversal.checked_config(config)
    view, output = Path(view), Path(output)
    document, digest = _protocol(protocol)
    request = dict(
        view_sha256=sha256(view), protocol_sha256=digest, window=window, config=config, code=_code()
    )
    report = _start(output, request, RULE_KIND)
    if report["status"] == "completed":
        return report
    dataset, _ = _open(view, document, window, output)
    predictions = {}
    for name in PREDICTED:
        if stop is not None and stop.requested:
            report["status"] = "paused"
            atomic_json(output / "run.json", report)
            return report
        rows = partition_rows(dataset, name, with_features=False)
        values = momentum_reversal.predict(rows.prices, rows.panel.session, config)
        predictions[name] = _write(output, name, rows, values, dataset)
    return _finish(output, report, predictions=predictions)


def _checked_seeds(seeds):
    _require(
        isinstance(seeds, list)
        and seeds
        and all(type(seed) is int and 0 <= seed < 2**31 for seed in seeds)
        and len(set(seeds)) == len(seeds),
        "LightGBM necesita las semillas de la criba, distintas y válidas",
    )
    return seeds


def _saved_fit(output, seed, booster, scored):
    """Guardar el booster de una semilla y su serie puntuada, de donde parte una reanudación."""
    path = output / "fits" / f"seed-{seed}.json"
    atomic_json(
        path,
        dict(
            seed=seed,
            model=lightgbm_control.model_text(booster),
            series=[[values.tolist(), defined.tolist()] for values, defined in scored],
        ),
    )
    return dict(path=f"fits/{path.name}", sha256=sha256(path))


def _loaded_fit(output, seed, record):
    """Ajuste guardado de una semilla, leído una sola vez y comprobado con su huella."""
    document, digest = read_manifest(output / record["path"], FIT_BYTES)
    _require(
        digest == record["sha256"] and document.get("seed") == seed,
        f"Ha cambiado el ajuste guardado de la semilla {seed}",
    )
    series = [
        (np.asarray(values, dtype=np.float64), np.asarray(defined, dtype=bool))
        for values, defined in document["series"]
    ]
    return lightgbm_control.load(document["model"]), series


def _fit_seeds(train, validation, config, seeds, *, min_assets, stop, output, report):
    """Booster y serie del IC de validación de cada semilla, con la barrera entre semillas.

    Una semilla ya guardada se carga sin ajustar. Las demás se ajustan y se guardan antes de
    pasar a la siguiente, con el informe actualizado.
    """
    fits, saved = [], report.setdefault("fits", {})
    for seed in seeds:
        if str(seed) in saved:
            fits.append(_loaded_fit(output, seed, saved[str(seed)]))
            continue
        if stop is not None and stop.requested:
            raise Stopped
        booster, scored = lightgbm_control.fit(
            train, validation, config, seed, min_assets=min_assets
        )
        saved[str(seed)] = _saved_fit(output, seed, booster, scored)
        atomic_json(output / "run.json", report)
        fits.append((booster, scored))
    return fits


def run_tabular_window(
    view, protocol, window, *, seeds, fit_months, output, config, radius, min_assets, stop=None
):
    """Ajustar LightGBM con cada semilla, elegir sus rondas en común y emitir cada semilla.

    `fit_months` son los meses de ajuste de la criba, `radius` el radio de la época común y
    `min_assets` el mínimo de activos del IC por sesión. Las rondas elegidas son el punto de
    control común multiplicado por las rondas de cada punto.
    """
    require_learning_allowed("la ventana LightGBM de los controles comunes")
    config = lightgbm_control.checked_config(config)
    seeds = _checked_seeds(seeds)
    view, output = Path(view), Path(output)
    document, digest = _protocol(protocol)
    request = dict(
        view_sha256=sha256(view),
        protocol_sha256=digest,
        window=window,
        seeds=seeds,
        fit_months=fit_months,
        config=config,
        radius=radius,
        min_assets=min_assets,
        code=_code(),
    )
    report = _start(output, request, TABULAR_KIND)
    if report["status"] == "completed":
        # Un corte justo después de confirmar deja los ajustes guardados, que ya sobran.
        if (output / "fits").exists():
            shutil.rmtree(output / "fits")
        return report
    dataset, fold = _open(view, document, window, output)
    since = window_phases(fold, 0, fit_months)["train"].decision_start
    train = partition_rows(dataset, "train", with_features=True, since=since)
    predicted = {name: partition_rows(dataset, name, with_features=True) for name in PREDICTED}
    try:
        fits = _fit_seeds(
            train,
            predicted["validation"],
            config,
            seeds,
            min_assets=min_assets,
            stop=stop,
            output=output,
            report=report,
        )
    except Stopped:
        report["status"] = "paused"
        atomic_json(output / "run.json", report)
        return report
    series = [scored for _, scored in fits]
    chosen = selection.common_epoch(selection.seed_epoch_scores(series), radius=radius)
    rounds = chosen["epoch"] * config["checkpoint_rounds"]
    members = {}
    for seed, (booster, scored) in zip(seeds, fits, strict=True):
        folder = output / f"seed-{seed}"
        emitted = {
            name: lightgbm_control.predict(booster, predicted[name].features, rounds)
            for name in PREDICTED
        }
        # La validación emitida es la que eligió las rondas.
        values, defined = sd.session_correlations(
            predicted["validation"].panel, emitted["validation"], min_assets=min_assets
        )
        _require(
            np.array_equal(values, scored[chosen["epoch"] - 1][0])
            and np.array_equal(defined, scored[chosen["epoch"] - 1][1]),
            "La validación emitida no reproduce la puntuada al elegir las rondas",
        )
        record = dict(
            schema_version=1,
            kind=TABULAR_KIND,
            status="completed",
            seed=seed,
            rounds=rounds,
            predictions={
                name: _write(folder, name, predicted[name], emitted[name], dataset)
                for name in PREDICTED
            },
        )
        atomic_json(folder / "run.json", record)
        members[str(seed)] = dict(path=f"seed-{seed}/run.json", sha256=sha256(folder / "run.json"))
    # Las predicciones confirmadas sustituyen a los ajustes guardados, que se borran después.
    report.pop("fits")
    finished = _finish(
        output,
        report,
        selection=dict(chosen, rounds=rounds, checkpoint_rounds=config["checkpoint_rounds"]),
        fit_rows=int(np.count_nonzero(lightgbm_control.session_labels(train.panel)[1])),
        train_rows=train.panel.rows,
        seeds=members,
    )
    shutil.rmtree(output / "fits")
    return finished
