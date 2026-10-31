"""Techo de señal con anomalías conocidas en cada origen de la criba (#747, fase 0b).

La fase 0 (#744) no encontró ordenación robusta en los predictores neuronales de los pilotos.
Antes de gastar GPU, este estudio mide cuánta señal diaria extraen dos modelos conocidos con
anomalías publicadas, con las filas, ventanas, cohortes y semillas de la criba (#746):

- Las filas de cada tramo son las que declara la vista del origen (`view_rows`). El ajuste
  usa las de entrenamiento decididas en los meses de ajuste de la criba (`window_phases`).
- Las características son las de `anomaly_features` con los precios verificados de la vista,
  su factor de mercado declarado y las presentaciones de los fundamentales de cada activo,
  llevadas a rangos por sesión entre los activos de la cohorte con precio en esa sesión.
- Cada modelo se ajusta con el residuo de la criba y con el retorno bruto apertura-cierre de la
  sesión siguiente, el contraste de la residualización, en las mismas filas.
- Ridge (`session_ridge`) elige su alpha y LightGBM (`qlib_lightgbm`, con las semillas de la
  criba) sus rondas con el IC por sesión de la validación. La calibración elige el λ por
  mercado del MAE calibrado. La evaluación del origen, su bloque de la criba, solo se mide.
- La importancia de cada familia es la caída del IC por sesión de la evaluación al permutar sus
  columnas dentro de cada sesión, con la misma permutación para los dos modelos y objetivos.

Cada origen escribe un único informe al terminar, así que un corte repite el origen entero.
Un informe terminado se reutiliza si la petición coincide. La protección del aprendizaje se
comprueba antes de leer la vista y ningún tramo alcanza el test final.
"""

import dataclasses
import hashlib
import importlib
import importlib.metadata
from pathlib import Path

import numpy as np
import pyarrow as pa

from mars_titan.data import anomaly_features as af
from mars_titan.data.batches import read_bounded_table
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.evaluation import scale_diagnostics as sd
from mars_titan.evaluation.forecast_panel import DAY_MICROSECONDS, ForecastPanel
from mars_titan.models.baselines import qlib_lightgbm, session_ridge
from mars_titan.models.titans.config import canonical

from . import selection
from .candidate_walk_forward import view_rows
from .corpus_inputs import CorpusDataset
from .learning_hold import require_learning_allowed
from .screening_cohorts import ORIGIN_FOLDS
from .temporal_contract import temporal_fold
from .walk_forward_phases import PREDICTED, window_phases

KIND = "signal_ceiling_origin_v1"
DECLARATION_KIND = "signal_ceiling_protocol_v1"
TARGETS = ("residual", "raw_open_close")
LIGHTGBM_IDENTITY = "known_anomalies_lightgbm_v1"
LABEL = "session_zscore_of_each_target"
COLUMNS = (*af.FEATURES, *(f"{name}_missing" for name in af.FEATURES))
TABLE_ROWS = 200_000
TABLE_BYTES = 64 * 1024**2
REPORT_BYTES = 64 * 1024**2
VERSIONED = ("numpy", "scipy", "scikit-learn", "lightgbm", "pyarrow")
_CODE = (
    "mars_titan.training.signal_ceiling",
    "mars_titan.data.anomaly_features",
    "mars_titan.models.baselines.qlib_lightgbm",
    "mars_titan.models.baselines.session_ridge",
    "mars_titan.evaluation.scale_diagnostics",
    "mars_titan.training.selection",
    "mars_titan.training.corpus_inputs",
    "mars_titan.training.candidate_walk_forward",
    "mars_titan.training.temporal_contract",
    "mars_titan.training.walk_forward_phases",
    "mars_titan.training.screening_cohorts",
    "mars_titan.data.batches",
    "mars_titan.data.cohort_files",
    "mars_titan.data.input_policy",
    "mars_titan.data.storage",
    "mars_titan.environments.cohorts",
    "mars_titan.evaluation.forecast_panel",
    "mars_titan.models.titans.config",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(document):
    return hashlib.sha256(canonical(document).encode()).hexdigest()


def _code():
    return {name: sha256(Path(importlib.import_module(name).__file__)) for name in _CODE}


def _checked_inference(inference):
    """Bootstrap por bloques de días UTC de la criba con sus longitudes y réplicas."""
    sensitivity = inference.get("sensitivity_block_days") if isinstance(inference, dict) else None
    _require(
        isinstance(inference, dict)
        and type(inference.get("block_days")) is int
        and inference["block_days"] >= 1
        and isinstance(sensitivity, list)
        and all(type(length) is int and length >= 1 for length in sensitivity)
        and type(inference.get("replicates")) is int
        and inference["replicates"] >= 1
        and type(inference.get("seed")) is int
        and type(inference.get("confidence")) is float
        and 0 < inference["confidence"] < 1,
        "La inferencia declara los bloques, las réplicas, la semilla y la confianza de la criba",
    )


def _checked_screening(shared):
    """Orígenes, semillas, meses de ajuste y selección que el estudio comparte con la criba."""
    origins, seeds = shared.get("origins"), shared.get("seeds")
    _require(
        isinstance(origins, list)
        and all(isinstance(origin, dict) and set(origin) == {"year", "fold"} for origin in origins)
        and [origin["year"] for origin in origins] == sorted(ORIGIN_FOLDS)
        and {origin["year"]: origin["fold"] for origin in origins} == ORIGIN_FOLDS
        and isinstance(seeds, list)
        and seeds
        and all(type(seed) is int and 0 <= seed < 2**31 for seed in seeds)
        and len(set(seeds)) == len(seeds)
        and type(shared.get("fit_months")) is int
        and shared["fit_months"] >= 1
        and type(shared.get("smoothing_radius")) is int
        and shared["smoothing_radius"] >= 0
        and type(shared.get("minimum_assets")) is int
        and shared["minimum_assets"] >= 3,
        "El techo de señal comparte los orígenes, las semillas, los meses de ajuste y la "
        "selección de la criba",
    )
    _checked_inference(shared.get("inference"))


def checked_declaration(declaration):
    """Declaración del estudio con sus orígenes, características, modelos e importancia."""
    _require(
        isinstance(declaration, dict)
        and declaration.get("kind") == DECLARATION_KIND
        and declaration.get("final_test_access") is False
        and isinstance(declaration.get("screening"), dict)
        and isinstance(declaration.get("features"), dict)
        and declaration["features"].get("identity") == af.IDENTITY
        and declaration["features"].get("families")
        == {family: list(names) for family, names in af.FAMILIES.items()}
        and declaration["features"].get("minimum_share") == af.MINIMUM_SHARE
        and isinstance(declaration.get("targets"), dict)
        and set(declaration["targets"]) == set(TARGETS)
        and isinstance(declaration.get("models"), dict)
        and set(declaration["models"]) == {"ridge", "lightgbm"}
        and isinstance(declaration.get("importance"), dict),
        "El techo de señal declara sus características, sus modelos, su importancia y el test "
        "final cerrado",
    )
    _checked_screening(declaration["screening"])
    session_ridge.checked_config(declaration["models"]["ridge"], features=af.IDENTITY, label=LABEL)
    qlib_lightgbm.checked_config(
        declaration["models"]["lightgbm"],
        identity=LIGHTGBM_IDENTITY,
        features=af.IDENTITY,
        label=LABEL,
    )
    seed = declaration["importance"].get("seed")
    _require(
        declaration["importance"].get("metric") == "session_ic"
        and type(seed) is int
        and 0 <= seed < 2**63,
        "La importancia declara el IC por sesión y su semilla",
    )
    return declaration


@dataclasses.dataclass(frozen=True)
class Cohort:
    """Rangos por sesión de la cohorte y sus retornos apertura-cierre en el calendario declarado.

    `ranks` y `missing` tienen forma (activos, sesiones con mercado, características).
    `open_close` tiene una fila por activo y una columna por posición del calendario.
    """

    market: str
    index: dict
    decisions: np.ndarray
    absent: np.ndarray
    ranks: np.ndarray
    missing: np.ndarray
    open_close: np.ndarray
    sources: dict


def _microseconds(column):
    _require(pa.types.is_timestamp(column.type) and column.type.tz, "Un instante no tiene zona")
    values = column.cast(pa.timestamp("us", tz="UTC")).cast(pa.int64())
    _require(values.null_count == 0, "Falta un instante de disponibilidad")
    return values.to_numpy()


def _regular(path, digest, message):
    path = Path(path)
    safe_destination(path)
    _require(path.is_file() and not path.is_symlink() and sha256(path) == digest, message)
    return path


def _market_factor(dataset, market, decisions, absent, length):
    """Cierres densos del factor de mercado que declara la vista, comprobado con su huella."""
    entry = (dataset.manifest.get("market_factors") or {}).get(market)
    _require(
        isinstance(entry, dict)
        and entry.get("market") == market
        and isinstance(entry.get("prices_path"), str)
        and isinstance(entry.get("prices_sha256"), str),
        "La vista no declara el factor de mercado de su residuo",
    )
    path = _regular(
        entry["prices_path"], entry["prices_sha256"], "El factor de mercado cambió desde la vista"
    )
    table = read_bounded_table(path, max_rows=TABLE_ROWS, max_bytes=TABLE_BYTES)
    available = _microseconds(table["available_at"])
    positions = np.searchsorted(decisions, available)
    _require(
        np.all(positions < len(decisions)) and np.array_equal(decisions[positions], available),
        "El factor de mercado no sigue el calendario declarado",
    )
    close = table["close"].to_numpy().astype(np.float64)
    return af.dense_series(close, af.market_sessions(positions, absent), length), dict(
        symbol=entry.get("symbol"), sha256=entry["prices_sha256"]
    )


def _filings(dataset, asset):
    """Accesos y disponibilidades de las presentaciones del activo, con su huella comprobada.

    El manifiesto preparado del activo declara la huella de sus fundamentales y la de los
    precios que comprueba la vista, así que la cadena llega hasta la cohorte.
    """
    folder = dataset.roots["prepared"] / asset["market"] / asset["symbol"]
    manifest, digest = read_manifest(folder / "manifest.json", 4 * 1024**2)
    artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else None
    _require(
        isinstance(artifacts, dict)
        and artifacts.get("prices.parquet") == asset["prices_sha256"]
        and isinstance(artifacts.get("fundamentals.parquet"), str),
        "La preparación del activo no declara sus fundamentales con los precios de la vista",
    )
    path = _regular(
        folder / "fundamentals.parquet",
        artifacts["fundamentals.parquet"],
        "Los fundamentales del activo cambiaron desde su preparación",
    )
    table = read_bounded_table(path, max_rows=TABLE_ROWS, max_bytes=TABLE_BYTES)
    source = dict(manifest_sha256=digest, fundamentals_sha256=artifacts["fundamentals.parquet"])
    if not table.num_rows:
        return np.empty(0, dtype=object), np.empty(0, dtype=np.int64), source
    _require(table["accession"].null_count == 0, "Un hecho fundamental no tiene acceso")
    accessions = np.asarray(table["accession"].to_pylist(), dtype=object)
    return accessions, _microseconds(table["available_at"]), source


def load_cohort(dataset, market):
    """Características de todos los activos del mercado en la vista y sus rangos por sesión."""
    decisions, absent = dataset.market_calendar(market)
    length = len(decisions) - len(absent)
    factor, factor_source = _market_factor(dataset, market, decisions, absent, length)
    assets = sorted(
        (asset for asset in dataset.assets if asset["market"] == market),
        key=lambda asset: asset["symbol"],
    )
    _require(assets, "La vista no tiene activos del mercado")
    values = np.full((len(assets), length, len(af.FEATURES)), np.nan)
    present = np.zeros((len(assets), length), dtype=bool)
    open_close = np.full((len(assets), len(decisions)), np.nan)
    sources = {}
    for index, asset in enumerate(assets):
        prices, available = dataset.asset_prices(asset)
        positions, _ = dataset.calendar_positions(asset, available)
        sessions = af.market_sessions(positions, absent)
        close = af.dense_series(prices[:, 3], sessions, length)
        accessions, moments, source = _filings(dataset, asset)
        filings = af.filing_sessions(accessions, moments, decisions, absent)
        values[index] = af.asset_features(
            close, af.dense_series(prices[:, 4], sessions, length), factor, filings
        )
        present[index] = np.isfinite(close)
        open_close[index, positions] = prices[:, 3] / prices[:, 0] - 1
        sources[f"{market}/{asset['symbol']}"] = dict(
            source, prices_sha256=asset["prices_sha256"], filings=len(filings)
        )
    ranks, missing = af.session_ranks(values, present)
    return Cohort(
        market=market,
        index={f"{market}/{asset['symbol']}": k for k, asset in enumerate(assets)},
        decisions=decisions,
        absent=absent,
        ranks=ranks,
        missing=missing,
        open_close=open_close,
        sources=dict(market_factor=factor_source, assets=sources),
    )


@dataclasses.dataclass(frozen=True)
class CeilingRows:
    """Filas de un tramo en el orden canónico de su panel, con el retorno bruto y sus entradas.

    El panel lleva el residuo de la vista. `raw` es el retorno apertura-cierre de la sesión
    siguiente de cada fila y `features` sus rangos y ausencias, en el mismo orden.
    """

    panel: ForecastPanel
    raw: np.ndarray
    asset: np.ndarray
    session: np.ndarray
    features: np.ndarray

    def target(self, name):
        """Las mismas filas con el objetivo pedido en su panel."""
        _require(name in TARGETS, f"Objetivo desconocido: {name}")
        if name == "residual":
            return self
        return dataclasses.replace(self, panel=dataclasses.replace(self.panel, target=self.raw))


def partition_rows(dataset, cohort, partition, *, since=None):
    """Filas de un tramo de la vista o, con `since`, las decididas desde ese instante en µs."""
    declared = view_rows(dataset, partition)
    keep = np.ones(len(declared["target"]), dtype=bool)
    if since is not None:
        keep = declared["prediction_at"] >= since
    _require(keep.any(), f"El tramo {partition} no tiene filas que leer")
    assets = np.asarray(declared["asset_id"])[keep]
    moments = declared["prediction_at"][keep]
    ids = pa.array([f"{a}/{t}" for a, t in zip(assets, moments.tolist(), strict=True)])
    panel = ForecastPanel.from_columns(
        ids,
        np.asarray(declared["market"])[keep],
        moments,
        declared["target"][keep],
        np.arange(len(moments), dtype=np.float64),
    )
    order = panel.prediction.astype(np.int64)
    _require(
        set(np.unique(assets)) <= set(cohort.index),
        "Una fila de la vista no pertenece a la cohorte del mercado",
    )
    asset = np.asarray([cohort.index[name] for name in assets[order]], dtype=np.int64)
    positions = np.searchsorted(cohort.decisions, moments[order])
    _require(
        np.all(positions + 1 < len(cohort.decisions))
        and np.array_equal(cohort.decisions[positions], moments[order]),
        "Una decisión de la vista no pertenece al calendario declarado",
    )
    raw = cohort.open_close[asset, positions + 1]
    _require(np.isfinite(raw).all(), "Una fila no tiene el retorno de la sesión siguiente")
    session = af.market_sessions(positions, cohort.absent)
    features = np.concatenate(
        [cohort.ranks[asset, session], cohort.missing[asset, session].astype(np.float64)], axis=1
    )
    return CeilingRows(
        panel=dataclasses.replace(panel, prediction=np.zeros(panel.rows)),
        raw=raw,
        asset=asset,
        session=session,
        features=features,
    )


def _family_columns():
    """Columnas de cada familia: sus rangos y los indicadores de sus ausencias."""
    position = {name: k for k, name in enumerate(COLUMNS)}
    return {
        family: [position[name] for name in names] + [position[f"{n}_missing"] for n in names]
        for family, names in af.FAMILIES.items()
    }


def permuted_features(rows, columns, generator):
    """Características con las `columns` permutadas juntas dentro de cada sesión."""
    values = rows.features.copy()
    panel = rows.panel
    for start, count in zip(panel.session_starts, panel.session_samples, strict=True):
        order = start + generator.permutation(count)
        values[start : start + count, columns] = rows.features[order][:, columns]
    return values


def eligible_sessions(panel, min_assets):
    """Sesiones con activos suficientes y objetivo variable, las que pueden medir un orden."""
    high = np.maximum.reduceat(panel.target, panel.session_starts)
    low = np.minimum.reduceat(panel.target, panel.session_starts)
    return (panel.session_samples >= min_assets) & (high > low)


def ordering(panel, prediction, *, min_assets, rank=False):
    """IC por sesión en el que una predicción constante no ordena y vale cero.

    `session_correlations` deja sin definir la sesión de una predicción constante, como la
    criba. Aquí cuenta con IC nulo si la sesión tiene activos suficientes y objetivo variable,
    para que un modelo degenerado, como un LightGBM que no encuentra ninguna partición con la
    regularización de Qlib, mida cero en vez de interrumpir el estudio.
    """
    values, _ = sd.session_correlations(panel, prediction, rank=rank, min_assets=min_assets)
    return values, eligible_sessions(panel, min_assets)


def _seed_mean(items):
    """Media de las semillas en cada sesión. Una sesión cuenta si todas la definen."""
    defined = np.logical_and.reduce([defined for _, defined in items])
    values = np.mean([values for values, _ in items], axis=0)
    return np.where(defined, values, 0.0), defined


def _series(values, defined):
    return dict(values=np.asarray(values, dtype=np.float64).tolist(), defined=defined.tolist())


def measure(data, members, permuted, *, min_assets):
    """Series por sesión de la evaluación de un modelo, con sus semillas promediadas.

    `members` tiene la predicción de cada tramo para cada semilla. El λ del MAE calibrado se
    elige en la calibración de cada semilla. `permuted` da, por familia, las características
    permutadas de la evaluación y la predicción de cada semilla con ellas.
    """
    evaluation, calibration = data["evaluation"].panel, data["calibration"].panel
    sd.ensure_before(calibration, evaluation)
    sd.ensure_before(data["validation"].panel, calibration)
    ic, rank, raw, calibrated, validation = [], [], [], [], []
    for member in members:
        prediction = member["evaluation"]
        ic.append(ordering(evaluation, prediction, min_assets=min_assets))
        rank.append(ordering(evaluation, prediction, rank=True, min_assets=min_assets))
        raw.append(sd.session_mae(evaluation, prediction))
        factors = sd.market_shrinkage(calibration, member["calibration"])
        calibrated.append(
            sd.session_mae(evaluation, sd.apply_shrinkage(evaluation, prediction, factors))
        )
        validation.append(
            ordering(data["validation"].panel, member["validation"], min_assets=min_assets)
        )
    full = _seed_mean(ic)
    importance = {}
    for family, predictions in permuted.items():
        values, defined = _seed_mean(
            [ordering(evaluation, prediction, min_assets=min_assets) for prediction in predictions]
        )
        both = defined & full[1]
        importance[family] = _series(np.where(both, full[0] - values, 0.0), both)
    validation_mean = _seed_mean(validation)
    return dict(
        session_ic=_series(*full),
        session_rank_ic=_series(*_seed_mean(rank)),
        session_mae_raw=np.mean(raw, axis=0).tolist(),
        session_mae_calibrated=np.mean(calibrated, axis=0).tolist(),
        validation_ic=float(validation_mean[0][validation_mean[1]].mean())
        if validation_mean[1].any()
        else None,
        importance=importance,
    )


def _ridge(data, config, permuted_values, *, min_assets):
    model, chosen = session_ridge.fit(
        data["train"], data["validation"], config, min_assets=min_assets
    )
    member = {name: session_ridge.predict(model, data[name].features) for name in PREDICTED}
    permuted = {
        family: [session_ridge.predict(model, values)] for family, values in permuted_values.items()
    }
    return [member], permuted, chosen


def _lightgbm(data, config, permuted_values, *, seeds, radius, min_assets):
    fits = [
        qlib_lightgbm.fit(data["train"], data["validation"], config, seed, min_assets=min_assets)
        for seed in seeds
    ]
    # Una predicción constante en una sesión de validación cuenta como IC nulo, igual que en
    # `ordering`, así que cada punto de control se puntúa en todas las sesiones que lo admiten.
    eligible = eligible_sessions(data["validation"].panel, min_assets)
    series = [[(values, eligible) for values, _ in scored] for _, scored in fits]
    chosen = selection.common_epoch(selection.seed_epoch_scores(series), radius=radius)
    rounds = chosen["epoch"] * config["checkpoint_rounds"]
    members = [
        {name: qlib_lightgbm.predict(booster, data[name].features, rounds) for name in PREDICTED}
        for booster, _ in fits
    ]
    permuted = {
        family: [qlib_lightgbm.predict(booster, values, rounds) for booster, _ in fits]
        for family, values in permuted_values.items()
    }
    return members, permuted, dict(chosen, rounds=rounds)


def _coverage(rows):
    """Fracción de filas con cada característica definida."""
    observed = 1.0 - rows.features[:, len(af.FEATURES) :].mean(axis=0)
    return dict(zip(af.FEATURES, observed.round(6).tolist(), strict=True))


def run_origin(view, origin, declaration, output):
    """Ajustar y medir Ridge y LightGBM con los dos objetivos en un origen de la criba.

    Devuelve el informe del origen, que también se escribe en `output`. Un informe terminado
    con la misma petición se devuelve sin repetir nada.
    """
    require_learning_allowed("el techo de señal con anomalías conocidas")
    declaration = checked_declaration(declaration)
    _require(origin in declaration["screening"]["origins"], "El origen no está declarado")
    view, output = Path(view), Path(output)
    safe_destination(output)
    path = output / f"origin-{origin['year']}.json"
    request = dict(
        view_sha256=sha256(view),
        origin=origin,
        declaration_sha256=_digest(declaration),
        code=_code(),
    )
    if path.exists():
        report = read_manifest(path, REPORT_BYTES)[0]
        _require(
            report.get("kind") == KIND and report.get("request") == request,
            "La petición del origen ha cambiado",
        )
        return report
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    for protected in (*dataset.roots.values(), view.parent):
        outside_source(protected, output)
    fold = temporal_fold(dataset.manifest, input_policy=HISTORICAL_MASKED)
    _require(
        fold["id"] == origin["fold"] and dataset.manifest.get("final_test_opened") is False,
        "La vista no es la ventana del origen o abre el test final",
    )
    markets = {asset["market"] for asset in dataset.assets}
    _require(len(markets) == 1, "La vista de un origen de la criba tiene un único mercado")
    (market,) = markets
    shared = declaration["screening"]
    cohort = load_cohort(dataset, market)
    since = window_phases(fold, 0, shared["fit_months"])["train"].decision_start
    parts = dict(
        train=partition_rows(dataset, cohort, "train", since=since),
        **{name: partition_rows(dataset, cohort, name) for name in PREDICTED},
    )
    _require(
        int(parts["evaluation"].panel.prediction_at.max()) < FINAL_TEST_START_US,
        "El techo de señal no puede medir el test final",
    )
    families = _family_columns()
    generators = np.random.SeedSequence(declaration["importance"]["seed"]).spawn(len(families))
    permuted_values = {
        family: permuted_features(parts["evaluation"], columns, np.random.default_rng(seed))
        for (family, columns), seed in zip(families.items(), generators, strict=True)
    }
    minimum = shared["minimum_assets"]
    evaluation = parts["evaluation"].panel
    results = {}
    for target in TARGETS:
        data = {name: rows.target(target) for name, rows in parts.items()}
        ridge = _ridge(data, declaration["models"]["ridge"], permuted_values, min_assets=minimum)
        lightgbm = _lightgbm(
            data,
            declaration["models"]["lightgbm"],
            permuted_values,
            seeds=shared["seeds"],
            radius=shared["smoothing_radius"],
            min_assets=minimum,
        )
        results[target] = dict(
            zero_mae=sd.session_mae(data["evaluation"].panel, np.zeros(evaluation.rows)).tolist(),
            ridge=dict(measure(data, ridge[0], ridge[1], min_assets=minimum), selection=ridge[2]),
            lightgbm=dict(
                measure(data, lightgbm[0], lightgbm[1], min_assets=minimum),
                selection=lightgbm[2],
            ),
        )
    report = dict(
        schema_version=1,
        kind=KIND,
        request=request,
        status="completed",
        final_test_opened=False,
        rows={name: rows.panel.rows for name, rows in parts.items()},
        fit_start_us=int(since),
        coverage={name: _coverage(rows) for name, rows in parts.items()},
        sources=cohort.sources,
        versions={name: importlib.metadata.version(name) for name in VERSIONED},
        sessions=dict(
            time=evaluation.session_time.tolist(),
            day=(evaluation.session_time // DAY_MICROSECONDS).tolist(),
            assets=evaluation.session_samples.tolist(),
        ),
        targets=results,
    )
    atomic_json(path, report)
    return report
