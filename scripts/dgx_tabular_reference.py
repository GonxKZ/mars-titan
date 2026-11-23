#!/usr/bin/env python3
"""Referencia tabular de una campaña DGX: árboles de gradiente con los rasgos técnicos.

Para un origen de un estado de campaña, ajusta un LightGBM con los rasgos técnicos que declara
su cohorte, que son los que reciben los modelos, y escribe sus predicciones de validación,
calibración y evaluación con el formato de la campaña en un estado de referencia. El informe de
la criba lo admite con `--reference`: así cada tabla dice qué consigue un modelo sencillo con la
misma información y las combinaciones pueden contar con él.

El objetivo de ajuste es la etiqueta en unidades de retorno, con pérdida de Huber, para que su
predicción se pueda comparar con el alfa conocido de la etiqueta como la de los modelos de la
campaña. Con `--target alpha_free` se ajusta la etiqueta sin ese alfa y la predicción de la
etiqueta se lo resta. El número de árboles se elige en validación, con el IC de rangos por
sesión, entre los declarados. Solo lee las etiquetas de los tramos de la vista, y el test final
no está en ella.

    dgx_tabular_reference.py --state ESTADO --year 2013 --output ESTADO_DE_REFERENCIA
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.training.learning_hold import require_learning_allowed

SPLITS = ("validation", "calibration", "evaluation")
PARTITIONS = {"train": 0, "validation": 1, "calibration": 2, "evaluation": 3}
MIN_ASSETS = 30
# Recorte del objetivo, en puntos porcentuales de retorno.
CLIP = 10.0
PARAMETERS = dict(
    objective="huber",
    # Umbral de Huber, en puntos porcentuales: por encima el error pesa de forma lineal.
    alpha=1.0,
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=1000,
    feature_fraction=0.8,
    bagging_fraction=0.5,
    bagging_freq=1,
    lambda_l2=10.0,
    max_bin=63,
    deterministic=True,
    force_row_wise=True,
    verbose=-1,
)


def technical_module():
    """`technical_features`, junto a esta herramienta instalada o en el repositorio."""
    here = Path(__file__).resolve()
    for path in (
        here.with_name("technical_features.py"),
        here.parents[1] / "src/mars_titan/data/technical_features.py",
    ):
        if path.is_file():
            spec = importlib.util.spec_from_file_location("technical_features", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise ValueError("Falta technical_features.py para calcular los rasgos")


def price_table(path):
    table = pq.read_table(path, columns=["open", "high", "low", "close", "volume", "available_at"])
    stamps = table.column("available_at").cast("int64").to_numpy()
    columns = ("open", "high", "low", "close", "volume")
    values = np.column_stack([table.column(name).to_numpy() for name in columns])
    return values, stamps.astype("datetime64[us]").astype("datetime64[D]")


def market_factor(manifest, market):
    """Días y rendimiento intradía del factor declarado de un mercado, o (None, None) si falta.

    Si la cohorte exige el modelo de mercado (`market_model: required`), un factor ausente
    detiene la referencia en vez de dejar a cero sus rasgos.
    """
    declared = (manifest.get("market_factors") or {}).get(market) or {}
    path = Path(declared.get("prices_path") or "")
    if not path.is_file():
        if "market_model" in (manifest.get("input_extensions") or {}):
            raise ValueError(
                f"Falta en este equipo el factor de mercado de {market} que declara la vista"
            )
        return None, None
    # El factor puede ser un índice, sin volumen: solo se leen su apertura y su cierre.
    table = pq.read_table(path, columns=["open", "close", "available_at"])
    stamps = table.column("available_at").cast("int64").to_numpy()
    opened, close = table.column("open").to_numpy(), table.column("close").to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        valid = (opened > 0) & (close > 0)
        returns = np.where(valid, close / opened - 1, np.nan)
    return stamps.astype("datetime64[us]").astype("datetime64[D]"), returns


def load_rows(manifest, technical):
    """Rasgos, etiqueta, alfa conocido, sesión e identificador de cada fila con etiqueta."""
    version = (manifest.get("input_extensions") or {}).get("technical")
    if version not in technical.VERSIONS:
        raise ValueError("La cohorte no declara una versión de los rasgos técnicos")
    names = technical.VERSIONS[version]
    prepared, labels = Path(manifest["roots"]["prepared"]), Path(manifest["roots"]["labels"])
    markets = sorted({asset["market"] for asset in manifest["assets"]})
    factors = {market: market_factor(manifest, market) for market in markets}
    parts = dict(
        features=[],
        target=[],
        alpha=[],
        has_market_model=[],
        instant=[],
        market=[],
        partition=[],
        asset=[],
    )
    assets = []
    for asset in manifest["assets"]:
        market, symbol = asset["market"], asset["symbol"]
        table = pq.read_table(
            labels / market / symbol / "labels.parquet",
            columns=["prediction_at", "target", "partition"],
        )
        codes = np.array(
            [PARTITIONS.get(value, -1) for value in table.column("partition").to_pylist()],
            dtype=np.int8,
        )
        keep = codes >= 0
        if not keep.any():
            continue
        instants = table.column("prediction_at").cast("int64").to_numpy()[keep]
        target = table.column("target").to_numpy(zero_copy_only=False)[keep].astype(np.float64)
        prices, days = price_table(prepared / market / symbol / "prices.parquet")
        wanted = instants.astype("datetime64[us]").astype("datetime64[D]")
        position = np.searchsorted(days, wanted)
        inside = position < len(days)
        inside[inside] = days[position[inside]] == wanted[inside]
        if not inside.all():
            raise ValueError(f"Una etiqueta de {market}/{symbol} no tiene su sesión de precios")
        features = technical.technical_features(prices, days, *factors[market], version=version)
        rows = features[position]
        known = rows[:, names.index("has_market_model")] > 0
        alpha = np.where(known, rows[:, names.index("alpha_percent")].astype(np.float64) / 100, 0.0)
        parts["features"].append(rows)
        parts["target"].append(target)
        parts["alpha"].append(alpha)
        parts["has_market_model"].append(known.astype(np.bool_))
        parts["instant"].append(instants)
        parts["market"].append(np.full(len(target), markets.index(market), dtype=np.int8))
        parts["partition"].append(codes[keep])
        parts["asset"].append(np.full(len(target), len(assets), dtype=np.int32))
        assets.append(f"{market}/{symbol}")
    data = {key: np.concatenate(values) for key, values in parts.items()}
    order = np.lexsort((data["asset"], data["instant"], data["market"]))
    data = {key: values[order] for key, values in data.items()}
    change = (data["instant"][1:] != data["instant"][:-1]) | (
        data["market"][1:] != data["market"][:-1]
    )
    bounds = np.r_[0, np.flatnonzero(change) + 1, len(order)]
    return data, bounds, assets, markets, list(names), version


def rank_ic(prediction, target):
    from scipy.stats import rankdata

    left, right = rankdata(prediction), rankdata(target)
    left, right = left - left.mean(), right - right.mean()
    scale = np.sqrt((left**2).sum() * (right**2).sum())
    return float((left * right).sum() / scale) if scale > 0 else float("nan")


def session_ic(prediction, target, bounds, partition, code):
    values = []
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        if partition[start] == code and end - start >= MIN_ASSETS:
            value = rank_ic(prediction[start:end], target[start:end])
            if np.isfinite(value):
                values.append(value)
    return float(np.mean(values)) if values else float("nan")


def fit(data, names, bounds, *, trees, threads, seed, alpha_free, min_leaf=None, markets=None):
    """Ajustar con el tramo de ajuste y elegir el número de árboles en validación.

    El objetivo va en unidades de retorno (en porcentaje y acotado a ±CLIP), con pérdida de
    Huber: así la predicción es comparable con el alfa conocido, que también es un retorno. Con
    `alpha_free` se ajusta la etiqueta sin alfa y la predicción de la etiqueta le resta el alfa.
    """
    if alpha_free:
        has_mm = data.get("has_market_model")
        if has_mm is None:
            has_mm = data["features"][:, names.index("has_market_model")] > 0
        for market_code in np.unique(data["market"]):
            mask = data["market"] == market_code
            coverage = float(np.mean(has_mm[mask]))
            if coverage < 0.5:
                market_name = (
                    markets[int(market_code)]
                    if markets is not None and 0 <= int(market_code) < len(markets)
                    else str(int(market_code))
                )
                raise ValueError(
                    f"El objetivo alpha_free requiere modelo de mercado; el mercado {market_name} "
                    f"solo tiene has_market_model en {coverage:.1%} de las filas."
                )

    import lightgbm

    label = (data["target"] + data["alpha"] if alpha_free else data["target"]) * 100
    goal = np.clip(label, -CLIP, CLIP)
    matrix = np.ascontiguousarray(
        np.column_stack([data["features"], data["market"].astype(np.float32)])
    )
    train = data["partition"] == 0
    if not train.any() or not (data["partition"] == 1).any():
        raise ValueError("La referencia necesita filas de ajuste y de validación")
    parameters = dict(PARAMETERS, num_threads=threads, seed=seed)
    if min_leaf is not None:
        parameters.update(min_data_in_leaf=min_leaf)
    booster = lightgbm.train(
        parameters,
        lightgbm.Dataset(matrix[train], label=goal[train], feature_name=[*names, "market"]),
        num_boost_round=max(trees),
    )
    rest = ~train
    curve, predictions = {}, {}
    for count in sorted(trees):
        raw = np.zeros(len(label))
        raw[rest] = booster.predict(matrix[rest], num_iteration=count, num_threads=threads) / 100
        if alpha_free:
            raw[rest] -= data["alpha"][rest]
        curve[count] = session_ic(raw, data["target"], bounds, data["partition"], 1)
        predictions[count] = raw
    best = max(sorted(trees), key=lambda count: (curve[count], -count))
    gain = booster.feature_importance(importance_type="gain", iteration=best)
    importance = sorted(
        zip([*names, "market"], (gain / max(gain.sum(), 1e-300)).tolist(), strict=True),
        key=lambda item: -item[1],
    )
    return predictions[best], best, curve, importance, lightgbm.__version__, parameters


def write_state(output, name, year, seed, data, bounds, assets, markets, prediction, receipt):
    run = output / "runs" / name / str(year) / f"seed-{seed}"
    run.mkdir(parents=True, exist_ok=True)
    stamp = pa.timestamp("us", tz="UTC")
    for split in SPLITS:
        rows = np.flatnonzero(data["partition"] == PARTITIONS[split])
        identifiers = [assets[index] for index in data["asset"][rows]]
        table = pa.table(
            dict(
                sample_id=[
                    f"{asset}/{instant}"
                    for asset, instant in zip(
                        identifiers, data["instant"][rows].tolist(), strict=True
                    )
                ],
                asset_id=identifiers,
                market=[markets[index] for index in data["market"][rows]],
                prediction_at=pa.array(data["instant"][rows], type=pa.int64()).cast(stamp),
                target=data["target"][rows],
                prediction=prediction[rows],
            )
        )
        pq.write_table(table, run / f"{split}-predictions.parquet")
    (run / "run.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    queue = output / "queue"
    queue.mkdir(parents=True, exist_ok=True)
    record = dict(
        status="completed",
        model=name,
        tier="reference",
        year=year,
        seed=seed,
        attempts=[dict(status="completed", seconds=receipt["seconds"])],
        result=dict(best_epoch=None, best_score=None),
    )
    (queue / f"{name}--{year}--seed-{seed}.json").write_text(json.dumps(record, indent=2) + "\n")
    return run


def build(state, year, output, *, name, seed, trees, threads, alpha_free, min_leaf=None):
    start = time.time()
    manifest = json.loads((state / "cohorts" / str(year) / "manifest.json").read_text())
    if manifest.get("final_test_opened") is not False:
        raise ValueError("La vista no declara cerrado el test final")
    technical = technical_module()
    data, bounds, assets, markets, names, version = load_rows(manifest, technical)
    prediction, best, curve, importance, library, parameters = fit(
        data,
        names,
        bounds,
        trees=trees,
        threads=threads,
        seed=seed,
        alpha_free=alpha_free,
        min_leaf=min_leaf,
        markets=markets,
    )
    receipt = dict(
        kind="dgx_tabular_reference",
        model=name,
        year=year,
        seed=seed,
        state=str(state),
        technical=version,
        features=[*names, "market"],
        target="alpha_free_return_percent" if alpha_free else "label_return_percent",
        target_clip_percent=CLIP,
        parameters=parameters,
        trees=best,
        validation_ic_by_trees={str(count): value for count, value in curve.items()},
        rows={key: int((data["partition"] == code).sum()) for key, code in PARTITIONS.items()},
        ic={
            split: session_ic(prediction, data["target"], bounds, data["partition"], code)
            for split, code in PARTITIONS.items()
            if split != "train"
        },
        importance=importance[:20],
        versions=dict(lightgbm=library, numpy=np.__version__, pyarrow=pa.__version__),
        final_test_access=False,
        seconds=round(time.time() - start, 1),
    )
    run = write_state(output, name, year, seed, data, bounds, assets, markets, prediction, receipt)
    return run, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, nargs="+", default=[200, 400, 800, 1200])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--target", choices=("label", "alpha_free"), default="label")
    parser.add_argument("--min-leaf", type=int, help="Filas mínimas por hoja (1000 por defecto)")
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 32 or any(not 1 <= count <= 5000 for count in args.trees):
        parser.error("Los hilos van de 1 a 32 y los árboles de 1 a 5000")
    # El guion ajusta un modelo: respeta la protección local antes de leer el estado.
    require_learning_allowed("la referencia tabular de la campaña DGX")
    alpha_free = args.target == "alpha_free"
    name = args.name or ("lightgbm_technical_free" if alpha_free else "lightgbm_technical")
    run, receipt = build(
        args.state,
        args.year,
        args.output,
        name=name,
        seed=args.seed,
        trees=args.trees,
        threads=args.threads,
        alpha_free=alpha_free,
        min_leaf=args.min_leaf,
    )
    ic = receipt["ic"]
    print(
        f"{name} {args.year}: {receipt['trees']} árboles, IC de validación {ic['validation']:.4f}, "
        f"calibración {ic['calibration']:.4f}, evaluación {ic['evaluation']:.4f}: {run}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
