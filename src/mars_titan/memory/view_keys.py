"""Entradas de las claves de la memoria por vecinos, construidas desde la vista de un origen.

Son las 92 columnas que leyó el laboratorio tabular en el origen 2013 de EE. UU.
(`laboratorio/tab-cache-us-2013`): 46 rasgos técnicos escalados y sus rangos gaussianos por
sesión. Se calculan, como allí, con los precios preparados y las etiquetas de la vista del
origen (el manifiesto de su cohorte, cuya huella es `view_sha256`):

- rasgos en bruto de cada fila (activo, día UTC de su decisión): las 30 columnas de
  `long_horizon_v1` (`data.technical_features`: cero donde falta historia y acotadas a ±8) y 16
  con la definición del laboratorio, que deja NaN donde falta historia y no acota: las medias
  intradía y nocturnas de 5, 63 y 252 sesiones en volatilidades anuales, la etiqueta ya realizada
  de 1, 5 y 21 sesiones, el alfa de 21 y 63 sesiones, la fracción de sesiones sin negociación de
  63, la posición del cierre en el rango del día y su media de 5, y las medias intradía y nocturna
  de 252 sesiones en porcentaje. No son las 16 de `long_horizon_v2`, que ponen cero donde falta
  historia y acotan a ±8: el laboratorio las calculó antes;
- escala robusta de cada mercado con sus filas de ajuste: mediana, 1,4826 × MAD (la desviación
  típica si la MAD es nula y 1 si también lo es), recorte a ±5 y cero donde falta el valor;
- rangos gaussianos de cada columna dentro de cada sesión: Φ⁻¹((r − ½)·(1/n)) en FP32 con el
  rango estable (a igual valor, el orden de los activos en la vista), recortados a ±5. El
  laboratorio los calculó en la GPU, donde PyTorch divide por un escalar multiplicando por su
  inverso; aquí van en la CPU con la misma función de PyTorch. La Φ⁻¹ en FP32 de la CPU y la de
  la GPU difieren en unas pocas ULP en una parte de los valores (medido en nueve sesiones de la
  caché de 2013: el 5,3 % de los valores, como mucho 5 ULP); los rasgos escalados no cambian.

Cada mercado se trata como un laboratorio aparte (su escala, sus sesiones y, en la memoria, su
banco), así que en la vista conjunta EE. UU.+China cada mercado recibe lo mismo que con su vista.

Contrato causal: cada rasgo de una fila usa sus precios y los anteriores, y los precios se cortan
en el último día con etiqueta en un tramo; la escala solo usa filas de ajuste; el rango de una
sesión solo usa las filas de esa sesión. No se lee ninguna fila del test final: falla si un tramo
de la vista llega a 2024. Cada archivo leído se comprueba contra la huella de la vista.

Salida, con la forma de la caché del laboratorio que lee `posttraining_pilots.lab_inputs`:
`X.npy` y `XR.npy` (FP32, filas de validación, calibración y evaluación), `asset.npy`, `day.npy`,
`bounds.npy`, `session_part.npy`, `names.json` y el recibo `keys.json` con las huellas de
entradas y salidas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data import technical_features as tf
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.environments.cohorts import FINAL_TEST_START_US

KIND = "view_memory_keys_v1"
PARTS = {"train": 0, "validation": 1, "calibration": 2, "evaluation": 3}
LAB_EXTRA = (
    "intraday_5",
    "intraday_63",
    "intraday_252",
    "overnight_5",
    "overnight_63",
    "overnight_252",
    "residual_1",
    "residual_5",
    "residual_21",
    "alpha_21",
    "alpha_63",
    "zero_volume_63",
    "close_location_1",
    "close_location_5",
)
LAB_PERCENT = ("intraday_mean_252_percent", "overnight_mean_252_percent")
COLUMNS = (*tf.NAMES, *LAB_EXTRA, *LAB_PERCENT)
LIMIT = 5.0
OUTPUTS = ("X.npy", "XR.npy", "asset.npy", "day.npy", "bounds.npy", "session_part.npy")
DAY_US = 86_400_000_000
_PRICES = ("open", "high", "low", "close", "volume")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _window_sum(values, window):
    """Suma de las `window` últimas filas hasta cada una, con las que haya al principio."""
    total = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    index = np.arange(1, len(values) + 1)
    return total[index] - total[np.maximum(index - window, 0)]


def lab_extra(prices, sessions, calendar=None, market=None, *, _precomputed_market=None):
    """Las 16 columnas añadidas por el laboratorio, en FP64, con NaN donde falta historia.

    `prices` es la tabla OHLCV de un activo, `sessions` el día de cada fila y `calendar` y
    `market` los días del factor de mercado y su rentabilidad de apertura a cierre; sin ellos,
    la etiqueta realizada y el alfa quedan en NaN.
    """
    prices = np.asarray(prices, dtype=np.float64)
    _require(
        prices.ndim == 2 and prices.shape[1] == 5 and len(sessions) == len(prices),
        "Las columnas del laboratorio necesitan una tabla OHLCV y el día de cada fila",
    )
    opened, high, low, close, volume = prices.T
    rows = len(close)
    found = {name: np.full(rows, np.nan) for name in (*LAB_EXTRA, *LAB_PERCENT)}
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_close = np.log(close)
        daily = np.full(rows, np.nan)
        daily[1:] = log_close[1:] - log_close[:-1]
        filled = np.nan_to_num(daily)
        mean = _window_sum(filled, 252) / 252
        year = np.sqrt(np.maximum(_window_sum(filled * filled, 252) / 252 - mean * mean, 0.0))
        year[:252] = np.nan
        intraday = np.nan_to_num(np.log(close / opened))
        overnight = np.full(rows, np.nan)
        overnight[1:] = np.log(opened[1:] / close[:-1])
        overnight = np.nan_to_num(overnight)
        for window in (5, 63, 252):
            mean_intraday = _window_sum(intraday, window) / window
            mean_intraday[: window - 1] = np.nan
            mean_overnight = _window_sum(overnight, window) / window
            mean_overnight[:window] = np.nan
            found[f"intraday_{window}"] = mean_intraday / year * np.sqrt(window)
            found[f"overnight_{window}"] = mean_overnight / year * np.sqrt(window)
        found["intraday_mean_252_percent"] = mean_intraday * 100
        found["overnight_mean_252_percent"] = mean_overnight * 100
        location = np.where(high > low, (2 * close - high - low) / (high - low), 0.0)
        found["close_location_1"] = location
        found["close_location_5"] = _window_sum(np.nan_to_num(location), 5) / 5
        stale = ((volume <= 0) | ((opened == close) & (high == low))).astype(np.float64)
        untraded = _window_sum(stale, 63) / 63
        untraded[:62] = np.nan
        found["zero_volume_63"] = untraded
        if _precomputed_market is not None or (calendar is not None and market is not None):
            if _precomputed_market is not None:
                calendar, market, position, inside, aligned, alpha, beta, residual, _ = (
                    _precomputed_market
                )
            else:
                sessions = np.asarray(sessions, dtype="datetime64[D]")
                calendar, market, position, inside, aligned, alpha, beta, residual, _ = (
                    tf._aligned_market_model(opened, close, sessions, calendar, market)
                )
            # Etiqueta de la sesión anterior, ya realizada al cierre de esta, en desviaciones
            # residuales de la anterior (NaN si falta), y su suma en 5 y 21 sesiones.
            realized = np.full(len(calendar), np.nan)
            realized[1:] = aligned[1:] - alpha[:-1] - beta[:-1] * market[1:]
            scale = np.full(len(calendar), np.nan)
            scale[1:] = residual[:-1]
            unit = realized / scale
            known = np.nan_to_num(unit)
            free = np.nan_to_num(aligned - beta * market)
            at = position[inside]
            found["residual_1"][inside] = unit[at]
            for window in (5, 21):
                found[f"residual_{window}"][inside] = (
                    _window_sum(known, window) / np.sqrt(window)
                )[at]
            for window in (21, 63):
                found[f"alpha_{window}"][inside] = (_window_sum(free, window) / window * 100)[at]
    return np.column_stack([found[name] for name in (*LAB_EXTRA, *LAB_PERCENT)])


def lab_columns(prices, sessions, calendar=None, market=None):
    """Las 46 columnas en bruto del laboratorio para cada fila de precios, en FP32."""
    pre = None
    if calendar is not None and market is not None:
        prices_arr = np.asarray(prices, dtype=np.float64)
        sessions_arr = np.asarray(sessions, dtype="datetime64[D]")
        if (
            prices_arr.ndim == 2
            and prices_arr.shape[1] == 5
            and len(sessions_arr) == len(prices_arr)
        ):
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                pre = tf._aligned_market_model(
                    prices_arr[:, 0], prices_arr[:, 3], sessions_arr, calendar, market
                )
    base = tf.technical_features(prices, sessions, calendar, market, _precomputed_market=pre)
    extra = lab_extra(prices, sessions, calendar, market, _precomputed_market=pre).astype(
        np.float32
    )
    return np.hstack([base, extra])


def _fast_median_1d(arr):
    """Mediana exacta de un vector 1D finito (idéntica a np.median con np.partition O(n))."""
    n = len(arr)
    if n == 0:
        return np.float32(np.nan)
    mid = n // 2
    if n % 2 == 1:
        return np.partition(arr, mid)[mid]
    part = np.partition(arr, (mid - 1, mid))
    return (part[mid - 1] + part[mid]) * np.asarray(0.5, dtype=arr.dtype)


def robust_scale(raw, train):
    """Centro y escala de cada columna con las filas de ajuste, como el laboratorio (FP32).

    Calcula columna a columna con `np.partition` O(n) en lugar de copiar y ordenar la matriz
    completa (ahorra ~4,5 GiB de RAM y ~30x de tiempo en 6,5 M filas).
    """
    train_idx = np.flatnonzero(train) if np.asarray(train).dtype == bool else np.asarray(train)
    _require(len(train_idx) > 0, "La escala necesita filas de ajuste")
    cols = raw.shape[1]
    dtype = raw.dtype if np.issubdtype(raw.dtype, np.floating) else np.float32
    center = np.empty(cols, dtype=dtype)
    spread = np.empty(cols, dtype=np.float64)
    for j in range(cols):
        col = raw[train_idx, j]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            center[j] = np.nan
            spread[j] = 1.0
            continue
        med = _fast_median_1d(finite)
        center[j] = med
        dev = np.abs(finite - med)
        mad = 1.4826 * float(_fast_median_1d(dev))
        if mad > 0.0:
            spread[j] = mad
        else:
            wide = float(np.std(finite))
            spread[j] = wide if wide > 0.0 else 1.0
    return center, spread


def scaled(raw, center, spread):
    """Filas escaladas y recortadas a ±5, con cero donde falta el valor."""
    with np.errstate(invalid="ignore"):
        values = np.where(np.isfinite(raw), (raw - center) / spread, 0.0)
    return np.clip(values, -LIMIT, LIMIT).astype(np.float32)


def session_ranks(values, bounds, *, workers=1):
    """Rangos gaussianos de cada columna en cada sesión [inicio, fin), en FP32 y a ±5."""
    from concurrent.futures import ThreadPoolExecutor

    import torch

    values = np.asarray(values, dtype=np.float32)
    result = np.empty_like(values)
    pairs = list(zip(bounds[:-1], bounds[1:], strict=True))

    def _one(pair):
        start, end = int(pair[0]), int(pair[1])
        block = values[start:end]
        order = np.argsort(block, axis=0, kind="stable")
        ranks = np.empty(block.shape, dtype=np.float32)
        steps = np.arange(1, end - start + 1, dtype=np.float32)[:, None]
        np.put_along_axis(ranks, order, np.broadcast_to(steps, block.shape), axis=0)
        # Como el laboratorio en la GPU: PyTorch divide por un escalar multiplicando por su
        # inverso en FP32, así que el nivel es (r − ½)·(1/n) y no (r − ½)/n.
        inverse = np.float32(1.0) / np.float32(end - start)
        level = torch.from_numpy((ranks - np.float32(0.5)) * inverse)
        result[start:end] = torch.special.ndtri(level).clamp(-LIMIT, LIMIT).numpy()

    if workers > 1 and len(pairs) > 4:
        with ThreadPoolExecutor(max_workers=min(workers, len(pairs))) as pool:
            for _ in pool.map(_one, pairs):
                pass
    else:
        for pair in pairs:
            _one(pair)
    return result


def _verified(path, digest, what):
    """Contenido de un archivo comprobado contra la huella de la vista."""
    data = Path(path).read_bytes()
    _require(hashlib.sha256(data).hexdigest() == digest, f"{path} no es el {what} de la vista")
    return pq.read_table(pa.BufferReader(data))


def _days(column):
    moments = column.cast(pa.int64()).to_numpy()
    return moments, (moments // DAY_US).astype("datetime64[D]")


def _factor(declared, last_day):
    """Días y rentabilidad de apertura a cierre del factor de mercado, hasta `last_day`."""
    if not declared:
        return None, None, None
    table = _verified(declared["prices_path"], declared["prices_sha256"], "factor de mercado")
    _, days = _days(table["available_at"])
    keep = days <= last_day
    opened = table["open"].to_numpy()[keep]
    close = table["close"].to_numpy()[keep]
    valid = (opened > 0) & (close > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.where(valid, close / opened - 1, np.nan)
    return days[keep], returns, declared["prices_sha256"]


def _market_rows(manifest, market, log, *, workers=4):
    """Filas en bruto de un mercado, en el orden de los activos de la vista y de sus etiquetas."""
    from concurrent.futures import ThreadPoolExecutor

    prepared, labels_root = Path(manifest["roots"]["prepared"]), Path(manifest["roots"]["labels"])
    assets = [a for a in manifest["assets"] if a["market"] == market]

    def _read_label(asset):
        path = labels_root / market / asset["symbol"] / "labels.parquet"
        table = _verified(path, asset["labels_sha256"], "archivo de etiquetas")
        names = table["partition"].to_pylist()
        unknown = {p for p in names if p is not None and p not in PARTS}
        _require(not unknown, f"{path} tiene particiones desconocidas: {sorted(unknown)}")
        code = np.array([PARTS.get(p, -1) for p in names], dtype=np.int8)
        keep = code >= 0
        moments, days = _days(table["prediction_at"])
        newest = None
        if keep.any():
            _require(
                int(moments[keep].max()) < FINAL_TEST_START_US,
                f"La vista de {market}/{asset['symbol']} llega al test final",
            )
            newest = days[keep].max()
        return asset, days[keep], code[keep], newest

    use_pool = workers > 1 and len(assets) > 4
    if use_pool:
        with ThreadPoolExecutor(max_workers=min(workers, len(assets))) as pool:
            label_results = list(pool.map(_read_label, assets))
    else:
        label_results = [_read_label(asset) for asset in assets]

    labels, last_day = [], None
    for asset, kept_days, kept_code, newest in label_results:
        if newest is not None:
            last_day = newest if last_day is None else max(last_day, newest)
        labels.append((asset, kept_days, kept_code))
    _require(last_day is not None, f"La vista no tiene filas de {market} en sus tramos")
    calendar, factor, factor_sha = _factor(
        (manifest.get("market_factors") or {}).get(market), last_day
    )

    def _read_price(item):
        asset, at, code = item
        if not len(at):
            return None
        path = prepared / market / asset["symbol"] / "prices.parquet"
        table = _verified(path, asset["prices_sha256"], "archivo de precios")
        _, days = _days(table["available_at"])
        cut = days <= last_day
        prices = np.column_stack([table[c].to_numpy()[cut] for c in _PRICES]).astype(np.float64)
        days = days[cut]
        position = np.searchsorted(days, at)
        good = position < len(days)
        good[good] = days[position[good]] == at[good]
        if not good.any():
            return None
        values = lab_columns(prices, days, calendar, factor)
        return (
            values[position[good]],
            at[good].astype(np.int64).astype(np.int32),
            code[good],
            f"{market}/{asset['symbol']}",
        )

    raw, day, part, owner, names = [], [], [], [], []
    started = time.monotonic()
    if use_pool:
        with ThreadPoolExecutor(max_workers=min(workers, len(labels))) as pool:
            for index, result in enumerate(pool.map(_read_price, labels)):
                if result is not None:
                    vals, d_arr, p_arr, asset_name = result
                    raw.append(vals)
                    day.append(d_arr)
                    part.append(p_arr)
                    owner.append(np.full(len(d_arr), len(names), dtype=np.int32))
                    names.append(asset_name)
                if index % 500 == 0:
                    log(f"{market}: {index}/{len(labels)} activos, {time.monotonic() - started:.0f} s")
    else:
        for index, item in enumerate(labels):
            result = _read_price(item)
            if result is not None:
                vals, d_arr, p_arr, asset_name = result
                raw.append(vals)
                day.append(d_arr)
                part.append(p_arr)
                owner.append(np.full(len(d_arr), len(names), dtype=np.int32))
                names.append(asset_name)
            if index % 500 == 0:
                log(f"{market}: {index}/{len(labels)} activos, {time.monotonic() - started:.0f} s")
    return dict(
        raw=np.concatenate(raw),
        day=np.concatenate(day),
        part=np.concatenate(part),
        asset=np.concatenate(owner),
        names=names,
        factor_sha256=factor_sha,
    )


def build(manifest_path, output, *, markets=None, log=print):
    """Construir las claves de la vista en `output` (carpeta nueva) y devolver su recibo."""
    manifest_path, output = Path(manifest_path), Path(output)
    _require(
        not output.exists(), f"{output} ya existe: las claves se escriben en una carpeta nueva"
    )
    started = time.monotonic()
    document = manifest_path.read_bytes()
    manifest = json.loads(document)
    present = sorted({asset["market"] for asset in manifest["assets"]})
    markets = present if markets is None else list(markets)
    _require(markets and set(markets) <= set(present), f"Los mercados de la vista son {present}")
    blocks, names, receipt_markets = [], [], {}
    for market in markets:
        rows = _market_rows(manifest, market, log)
        day_all, part_all, asset_all = rows["day"], rows["part"], rows["asset"]
        train_mask = part_all == 0
        post_mask = part_all >= 1
        _require(
            bool(np.any(post_mask)), f"{market} no tiene filas de validación, calibración ni evaluación"
        )
        # Comprobar separación cronológica sin ordenar 5+ millones de filas de ajuste innecesariamente:
        # la escala robusta es invariante al orden de las filas de ajuste.
        train_count = int(np.sum(train_mask))
        if train_count > 0:
            max_train_day = int(day_all[train_mask].max())
            min_post_day = int(day_all[post_mask].min())
            if max_train_day >= min_post_day:
                # Respaldo exacto para detectar filas intercaladas en el mismo orden que antes.
                order_all = np.argsort(day_all, kind="mergesort")
                part_sorted = part_all[order_all]
                post_pos = np.flatnonzero(part_sorted >= 1)
                _require(
                    np.all(part_sorted[post_pos[0] :] >= 1),
                    f"Las filas posteriores al ajuste de {market} no son las últimas",
                )
        raw_all = rows.pop("raw")
        center, spread = robust_scale(raw_all, train_mask)
        post_idx = np.flatnonzero(post_mask)
        post_day = day_all[post_idx]
        # mergesort sobre post_idx (ya en orden creciente del manifiesto) da exactamente el mismo
        # orden estable por (day, orden original) que ordenar todas las filas y cortar [post[0]:].
        post_order = post_idx[np.argsort(post_day, kind="mergesort")]
        values = scaled(raw_all[post_order], center, spread)
        del raw_all
        day, part, asset = day_all[post_order], part_all[post_order], asset_all[post_order]
        change = np.flatnonzero(day[1:] != day[:-1]) + 1
        bounds = np.r_[0, change, len(day)].astype(np.int64)
        ranks = session_ranks(values, bounds, workers=4)
        blocks.append((values, ranks, asset + len(names), day, bounds, part[bounds[:-1]]))
        receipt_markets[market] = dict(
            rows=int(len(day_all)),
            train_rows=train_count,
            rows_after_fit=int(len(day)),
            sessions_after_fit=int(len(bounds) - 1),
            assets=len(rows["names"]),
            factor_prices_sha256=rows["factor_sha256"],
            center=[float(v) for v in center],
            spread=[float(v) for v in spread],
        )
        names += rows["names"]
        log(f"{market}: {len(day)} filas tras el ajuste, {time.monotonic() - started:.0f} s")
    offsets = np.cumsum([0] + [len(block[3]) for block in blocks])
    arrays = {
        "X.npy": np.concatenate([b[0] for b in blocks]),
        "XR.npy": np.concatenate([b[1] for b in blocks]),
        "asset.npy": np.concatenate([b[2] for b in blocks]).astype(np.int32),
        "day.npy": np.concatenate([b[3] for b in blocks]).astype(np.int32),
        "bounds.npy": np.concatenate(
            [b[4][:-1] + offsets[i] for i, b in enumerate(blocks)] + [offsets[-1:]]
        ).astype(np.int64),
        "session_part.npy": np.concatenate([b[5] for b in blocks]).astype(np.int8),
    }
    temporary = output.with_name(output.name + ".tmp")
    temporary.mkdir(parents=True)
    for name, value in arrays.items():
        np.save(temporary / name, value)
    names_text = json.dumps(dict(assets=names, columns=list(COLUMNS), markets=markets))
    (temporary / "names.json").write_text(names_text + "\n", encoding="utf-8")
    receipt = dict(
        schema_version=1,
        kind=KIND,
        final_test_opened=False,
        manifest=str(manifest_path),
        view_sha256=hashlib.sha256(document).hexdigest(),
        columns=list(COLUMNS),
        gaussian="torch.special.ndtri en FP32 en la CPU",
        markets=receipt_markets,
        outputs={name: sha256(temporary / name) for name in (*OUTPUTS, "names.json")},
        code_sha256=sha256(Path(__file__)),
        seconds=round(time.monotonic() - started, 1),
    )
    atomic_json(temporary / "keys.json", receipt)
    temporary.rename(output)
    return receipt


def compare(keys, lab, lab_names):
    """Diferencias con la caché del laboratorio en las filas posteriores al ajuste.

    Empareja las filas por (activo, día) y da, para X y XR, cuántos valores difieren en algún
    bit, la mayor diferencia absoluta y la mayor en unidades en la última cifra (ULP de FP32).
    """
    keys, lab = Path(keys), Path(lab)
    mine = json.loads((keys / "names.json").read_text(encoding="utf-8"))["assets"]
    theirs = json.loads(Path(lab_names).read_text(encoding="utf-8"))["assets"]
    part = np.load(lab / "session_part.npy")
    bounds = np.load(lab / "bounds.npy")
    post = np.flatnonzero(part >= 1)
    first, last = int(bounds[post[0]]), int(bounds[post[-1] + 1])
    lab_code = {name: i for i, name in enumerate(theirs)}
    my_asset = np.load(keys / "asset.npy")
    translated = np.array([lab_code.get(name, -1) for name in mine], dtype=np.int64)[my_asset]
    _require(bool(np.all(translated >= 0)), "Hay activos de las claves que el laboratorio no tiene")
    my_key = translated * 1_000_000 + np.load(keys / "day.npy").astype(np.int64)
    lab_asset = np.asarray(np.load(lab / "asset.npy", mmap_mode="r")[first:last]).astype(np.int64)
    lab_day = np.asarray(np.load(lab / "day.npy", mmap_mode="r")[first:last]).astype(np.int64)
    lab_key = lab_asset * 1_000_000 + lab_day
    order = np.argsort(lab_key)
    position = np.searchsorted(lab_key[order], my_key)
    found = position < len(order)
    found[found] = lab_key[order][position[found]] == my_key[found]
    report = dict(rows=int(len(my_key)), lab_rows=int(len(lab_key)), matched=int(found.sum()))
    index = order[position[found]]
    for name in ("X.npy", "XR.npy"):
        ours = np.load(keys / name)[found]
        reference = np.asarray(np.load(lab / name, mmap_mode="r")[first:last])[index]
        bits = ours.view(np.int32).astype(np.int64) - reference.view(np.int32).astype(np.int64)
        report[name] = dict(
            different=int(np.count_nonzero(ours != reference)),
            different_bits=int(np.count_nonzero(bits)),
            max_abs=float(np.max(np.abs(ours.astype(np.float64) - reference))),
            max_ulp=int(np.max(np.abs(bits))) if bits.size else 0,
            columns_with_differences=[
                COLUMNS[j] for j in np.flatnonzero(np.any(ours != reference, axis=0))
            ],
        )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    making = commands.add_parser("construir", help="Claves de una vista")
    making.add_argument("--manifest", type=Path, required=True, help="Manifiesto de la cohorte")
    making.add_argument("--output", type=Path, required=True)
    making.add_argument("--market", action="append", help="Mercados (todos si se omite)")
    checking = commands.add_parser("comparar", help="Diferencias con la caché del laboratorio")
    checking.add_argument("--keys", type=Path, required=True)
    checking.add_argument("--lab", type=Path, required=True)
    checking.add_argument("--lab-names", type=Path, required=True)
    args = parser.parse_args(argv)

    def log(line):
        print(line, flush=True)

    if args.command == "construir":
        receipt = build(args.manifest, args.output, markets=args.market, log=log)
        print(json.dumps({k: receipt[k] for k in ("view_sha256", "outputs", "seconds")}, indent=1))
    else:
        print(
            json.dumps(compare(args.keys, args.lab, args.lab_names), ensure_ascii=False, indent=1)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
