"""Diagnóstico de cobertura y estabilidad de la etiqueta residual (MT-016).

Lee unos objetivos ya confirmados (`targets-v3` o una edición posterior), recalcula con el
mismo código la regresión de cada sesión y comprueba bit a bit que el objetivo guardado es el
que se diagnostica. Después describe, solo en entrenamiento y validación:

- la cobertura: cada muestra de la edición en una categoría, con la causa de las que no tienen
  objetivo separada en historia corta del activo, factor ausente, sesión siguiente ausente o
  con precio inválido y factor ausente en la sesión siguiente;
- el retorno bruto y el residual sobre exactamente las mismas observaciones;
- alfa y beta por fecha, sus saltos entre sesiones contiguas y las ventanas degeneradas;
- la asociación que queda entre el residual y el factor de mercado;
- las sensibilidades declaradas de antemano, sobre la población común a todas ellas.

Los precios posteriores al corte no llegan al cálculo: se filtran al leer la tabla, antes de
convertirla. Una etiqueta de 2024 con objetivo detiene el diagnóstico. No ajusta ningún modelo.
"""

import hashlib
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, date, datetime
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .batches import read_bounded_table
from .budget_targets import _aligned_returns
from .cohort_files import read_manifest
from .residual_arrays import residual_targets_array
from .storage import sha256
from .temporal import MarketClock

KIND = "mars_titan_residual_diagnostics"
DECLARATION_KIND = "mars_titan_residual_diagnostics_declaration"
SCHEMA_VERSION = 1
PARTITIONS = ("train", "validation")
# Categorías de una muestra sin objetivo. Las cuatro primeras desdoblan los motivos de la
# generación según la serie que falla, y las demás se conservan tal como se generaron.
CAUSES = (
    "insufficient_stock_history",
    "missing_factor_history",
    "next_stock_session_absent",
    "next_stock_price_invalid",
    "next_factor_return_missing",
    "zero_market_variance",
    "target_after_cutoff",
    "target_crosses_partition_boundary",
    "outside_label_calendar",
    "final_test_reserved",
)
CATEGORIES = (*PARTITIONS, *CAUSES)
_STORED = {
    "insufficient_history",
    "missing_next_session",
    "zero_market_variance",
    "target_after_cutoff",
}
_DECLARATION = {
    "schema_version",
    "kind",
    "name",
    "declared_at",
    "notes",
    "cutoff",
    "partitions",
    "main",
    "sensitivities",
    "return_thresholds",
    "beta_bound",
    "beta_jump_thresholds",
    "quantiles",
    "extremes",
    "minimum_assets_per_session",
    "sector",
}
# Ventana del protocolo y de los objetivos confirmados. El diagnóstico no la elige.
PROTOCOL_WINDOW = {"history": 252, "minimum": 126}
FINAL_TEST_START = pd.Timestamp("2024-01-01", tz="UTC")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _increasing(values, low, high):
    return (
        isinstance(values, list)
        and values
        and all(type(v) in (int, float) and low < v < high for v in values)
        and all(a < b for a, b in zip(values, values[1:], strict=False))
    )


def load_declaration(path):
    """Declaración de sensibilidades y umbrales, validada y con su huella."""
    document, digest = read_manifest(Path(path), 1024**2)
    _require(isinstance(document, dict), "La declaración no es un objeto")
    names = [s.get("name") for s in document.get("sensitivities", []) if isinstance(s, dict)]
    _require(
        set(document) == _DECLARATION
        and document["schema_version"] == SCHEMA_VERSION
        and document["kind"] == DECLARATION_KIND
        and document["cutoff"] == "2023-12-31"
        and document["partitions"] == list(PARTITIONS)
        and document["main"] == PROTOCOL_WINDOW
        and isinstance(document["sensitivities"], list)
        and len(names) == len(document["sensitivities"]) == len(set(names))
        and all(_sensitivity(s) for s in document["sensitivities"])
        and _increasing(document["return_thresholds"], 0, float("inf"))
        and _increasing(document["beta_jump_thresholds"], 0, float("inf"))
        and _increasing(document["quantiles"], 0, 1)
        and type(document["beta_bound"]) in (int, float)
        and document["beta_bound"] > 0
        and type(document["extremes"]) is int
        and 1 <= document["extremes"] <= 100
        and type(document["minimum_assets_per_session"]) is int
        and document["minimum_assets_per_session"] >= 2
        and isinstance(document["sector"], dict)
        and set(document["sector"]) == {"admitted", "reason"}
        and document["sector"]["admitted"] is False
        and isinstance(document["sector"]["reason"], str),
        "La declaración del diagnóstico no cumple su contrato",
    )
    return dict(document, sha256=digest)


def _sensitivity(item):
    if not isinstance(item, dict) or not re.fullmatch(r"[a-z0-9_]+", str(item.get("name"))):
        return False
    if item.get("kind") == "market_adjusted":
        return set(item) == {"name", "kind"}
    window = {k: item.get(k) for k in ("history", "minimum")}
    return (
        item.get("kind") == "window"
        and set(item) == {"name", "kind", "history", "minimum"}
        and all(type(v) is int for v in window.values())
        and 2 <= window["minimum"] <= window["history"]
        and window != PROTOCOL_WINDOW
    )


def read_prices(path, digest, cutoff):
    """Precios de un activo o factor hasta el corte. Las filas posteriores no se convierten."""
    path = Path(path)
    _require(not path.is_symlink() and path.is_file(), f"Falta el archivo de precios {path}")
    _require(sha256(path) == digest, f"Los precios de {path} no coinciden con su huella")
    table = read_bounded_table(path, max_rows=200_000)
    session = pc.cast(table["session"], pa.string())
    table = table.set_column(table.schema.get_field_index("session"), "session", session)
    return table.filter(pc.less_equal(session, cutoff)).to_pandas()


def _calendar(prices, factor, clock, declaration):
    """Una fila por sesión hasta el corte con la regresión principal y sus sensibilidades."""
    cutoff = declaration["cutoff"]
    end = date.fromisoformat(cutoff)
    sessions = [day.isoformat() for day in clock.days if day <= end]
    stock, stock_at = _aligned_returns(prices, sessions, cutoff)
    market, market_at = _aligned_returns(factor, sessions, cutoff)
    main = residual_targets_array(prices, factor, clock, cutoff=cutoff, **declaration["main"])
    _require(len(main) == len(sessions), "La regresión no cubre el calendario")
    frame = main.assign(
        slot=np.arange(len(sessions)),
        session=sessions,
        next_session=[*sessions[1:], None],
        raw=np.append(stock[1:], np.nan),
        factor=np.append(market[1:], np.nan),
    )
    present = set(prices.session.astype(str))
    frame["next_present"] = [session in present for session in frame.next_session]
    indexed = prices.assign(session=prices.session.astype(str)).set_index("session")
    quotes = indexed.reindex(frame.next_session)[["open", "close"]].to_numpy(dtype=float)
    frame["next_open"], frame["next_close"] = quotes[:, 0], quotes[:, 1]
    # Fecha de maduración común a todas las variantes: la de los dos retornos siguientes.
    later = pd.concat([stock_at.shift(-1), market_at.shift(-1)], axis=1).max(axis=1, skipna=False)
    frame["maturity"] = later.reset_index(drop=True)
    # Una maduración posterior al corte, en fecha UTC, no admite etiqueta.
    limit = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    for item in declaration["sensitivities"]:
        name = item["name"]
        if item["kind"] == "window":
            window = dict(history=item["history"], minimum=item["minimum"])
            variant = residual_targets_array(prices, factor, clock, cutoff=cutoff, **window)
            frame[f"{name}:target"] = variant.target.to_numpy(dtype=float)
            frame[f"{name}:beta"] = variant.beta.to_numpy(dtype=float)
            frame[f"{name}:accepted"] = (variant.reason == "accepted").to_numpy()
        else:
            finite = np.isfinite(frame.raw) & np.isfinite(frame.factor)
            mature = frame.maturity.notna() & (frame.maturity < limit)
            frame[f"{name}:target"] = (frame.raw - frame.factor).where(finite)
            frame[f"{name}:beta"] = np.where(finite, 1.0, np.nan)
            frame[f"{name}:accepted"] = (finite & mature).to_numpy()
    return frame, stock, stock_at


def _partition(prediction, maturity):
    # Misma regla que la generación de etiquetas: entrenamiento hasta 2022 y validación 2023,
    # con la prueba final aparte y las etiquetas que cruzan la frontera purgadas.
    if prediction.year <= 2022 and maturity.year <= 2022:
        return "train"
    if prediction.year == maturity.year == 2023:
        return "validation"
    return None


def _insufficient(row, stock, stock_at, history, minimum):
    slot = int(row.slot)
    window = slice(max(0, slot - history + 1), slot + 1)
    known = np.isfinite(stock[window]) & (stock_at.iloc[window] <= row.prediction_at).to_numpy()
    return "insufficient_stock_history" if known.sum() < minimum else "missing_factor_history"


def _missing_next(row):
    if not row.next_present:
        return "next_stock_session_absent"
    if not np.isfinite(row.raw):
        return "next_stock_price_invalid"
    return "next_factor_return_missing"


def classify(labels, frame, stock, stock_at, declaration):
    """Categoría de cada muestra, tras comprobar que la etiqueta guardada es la recalculada."""
    _require(labels.prediction_at.is_unique, "Hay dos muestras con la misma decisión")
    late = labels.prediction_at >= FINAL_TEST_START
    _require(
        (labels.reason[late] == "final_test_reserved").all()
        and labels.target[late].isna().all()
        and labels.partition[late].isna().all()
        and (labels.reason[~late] != "final_test_reserved").all(),
        "Una etiqueta de la prueba final tiene objetivo o un motivo indebido",
    )
    joined = labels.loc[~late].merge(
        frame, on="prediction_at", how="left", suffixes=("", "_calendar"), validate="1:1"
    )
    categories = []
    history, minimum = declaration["main"]["history"], declaration["main"]["minimum"]
    for row in joined.itertuples(index=False):
        if row.reason == "outside_label_calendar":
            _require(pd.isna(row.slot), "Una muestra fuera del calendario tiene sesión")
            categories.append(row.reason)
            continue
        _require(pd.notna(row.slot), "Una muestra con etiqueta no está en el calendario")
        if row.partition in PARTITIONS:
            _require(
                row.reason == "accepted"
                and row.reason_calendar == "accepted"
                and row.target == row.target_calendar
                and row.target_available_at == row.target_available_at_calendar
                and row.partition == _partition(row.prediction_at, row.target_available_at),
                "El objetivo guardado no coincide con el recalculado",
            )
            categories.append(row.partition)
        elif row.reason == "target_crosses_partition_boundary":
            _require(
                row.reason_calendar == "accepted"
                and _partition(row.prediction_at, row.target_available_at_calendar) is None,
                "Una etiqueta purgada no cruza ninguna frontera",
            )
            categories.append(row.reason)
        else:
            _require(
                row.reason in _STORED
                and row.reason_calendar == row.reason
                and pd.isna(row.target)
                and pd.isna(row.partition),
                "El motivo guardado no coincide con el recalculado",
            )
            if row.reason == "insufficient_history":
                categories.append(_insufficient(row, stock, stock_at, history, minimum))
            elif row.reason == "missing_next_session":
                categories.append(_missing_next(row))
            else:
                categories.append(row.reason)
    joined["category"] = categories
    reserved = labels.loc[late].assign(category="final_test_reserved")
    return joined, reserved


def presence_patterns(path, digest, rows):
    """Patrón de máscaras de cada muestra (bits en el orden del contrato), o None."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        return None, "missing"
    if sha256(path) != digest:
        return None, "changed"
    table = pq.read_table(path, columns=["prediction_at", "presence"])
    _require(table.num_rows == rows, "Las muestras no tienen las filas de sus etiquetas")
    bits = np.asarray(table["presence"].combine_chunks().flatten(), dtype=np.int16)
    width = len(bits) // max(rows, 1)
    _require(width * rows == len(bits) and width > 0, "La presencia no tiene anchura fija")
    weights = 1 << np.arange(width - 1, -1, -1, dtype=np.int16)
    patterns = bits.reshape(rows, width) @ weights
    return pd.Series(patterns, index=table["prediction_at"].to_pandas()), "verified"


def _sums(y, f, main):
    d = y - main
    return np.array(
        [
            len(y),
            y.sum(),
            (y * y).sum(),
            f.sum(),
            (f * f).sum(),
            (y * f).sum(),
            (y * main).sum(),
            (main * main).sum(),
            main.sum(),
            np.abs(d).sum(),
            (d * d).sum(),
        ],
        dtype=float,
    )


def _jumps(beta, slot, thresholds):
    adjacent = np.diff(slot) == 1
    change = np.abs(np.diff(beta))[adjacent]
    above = [(change > t).sum() for t in thresholds]
    return np.array([len(change), change.sum(), change.max(initial=0.0), *above], dtype=float)


def _extremes(rows, column, count, prices_sha256):
    """Observaciones más extremas con la referencia a su precio fuente."""
    order = np.argsort(-np.abs(rows[column].to_numpy(dtype=float)), kind="stable")[:count]
    picked = rows.iloc[order]
    numbers = ("next_open", "next_close", "raw", "target", "alpha", "beta", "factor")
    return [
        dict(
            partition=row.partition,
            session=row.session,
            next_session=row.next_session,
            **{name: float(getattr(row, name)) for name in numbers},
            prices_sha256=prices_sha256,
        )
        for row in picked.itertuples(index=False)
    ]


def diagnose_asset(task):
    """Resultado de un activo: recuentos, filas aceptadas y acumuladores de sensibilidad."""
    declaration, market, symbol = task["declaration"], task["market"], task["symbol"]
    receipt = task["receipt"]
    labels_path = Path(task["labels"])
    _require(sha256(labels_path) == receipt["labels_sha256"], "Las etiquetas han cambiado")
    labels = pq.read_table(
        labels_path,
        columns=["prediction_at", "target_available_at", "target", "partition", "reason"],
    ).to_pandas()
    _require(len(labels) == receipt["samples"], "Las etiquetas no tienen las filas del recibo")
    clock = MarketClock(market, task["calendar_start"], "2024-01-05")
    prices = read_prices(task["prices"], receipt["prices_sha256"], declaration["cutoff"])
    factor = read_prices(task["factor"], task["factor_sha256"], declaration["cutoff"])
    frame, stock, stock_at = _calendar(prices, factor, clock, declaration)
    joined, reserved = classify(labels, frame, stock, stock_at, declaration)
    patterns, samples_state = None, "not_requested"
    if task.get("samples"):
        patterns, samples_state = presence_patterns(
            task["samples"], receipt["samples_sha256"], len(labels)
        )
    pattern = (
        np.full(len(joined), -1, dtype=np.int16)
        if patterns is None
        else patterns.reindex(joined.prediction_at).to_numpy(dtype=np.int16)
    )
    joined["pattern"] = pattern
    years = joined.prediction_at.dt.year.to_numpy()
    counts = Counter(zip(years.tolist(), joined.category, pattern.tolist(), strict=True))
    counts.update(Counter((d.year, "final_test_reserved", -1) for d in reserved.prediction_at))
    accepted = joined.loc[joined.category.isin(PARTITIONS)].sort_values("slot")
    names = [s["name"] for s in declaration["sensitivities"]]
    variant_counts = Counter()
    for name in names:
        chosen = joined.loc[joined[f"{name}:accepted"].eq(True)]
        for prediction, maturity in zip(chosen.prediction_at, chosen.maturity, strict=True):
            variant_counts[name, _partition(prediction, maturity)] += 1
    common = accepted.loc[
        np.logical_and.reduce([accepted[f"{n}:accepted"].eq(True) for n in names])
        if names
        else np.ones(len(accepted), dtype=bool)
    ]
    sums, jumps = {}, {}
    for partition in PARTITIONS:
        part = common.loc[common.partition == partition]
        main = part.target.to_numpy(dtype=float)
        f = part.factor.to_numpy(dtype=float)
        slot = part.slot.to_numpy(dtype=np.int64)
        sums["main", partition] = _sums(main, f, main)
        jumps["main", partition] = _jumps(
            part.beta.to_numpy(dtype=float), slot, declaration["beta_jump_thresholds"]
        )
        for name in names:
            y = part[f"{name}:target"].to_numpy(dtype=float)
            sums[name, partition] = _sums(y, f, main)
            jumps[name, partition] = _jumps(
                part[f"{name}:beta"].to_numpy(dtype=float),
                slot,
                declaration["beta_jump_thresholds"],
            )
    mask = np.packbits(joined.prediction_at.isin(common.prediction_at).to_numpy())
    columns = dict(
        slot=accepted.slot.to_numpy(dtype=np.int32),
        partition=(accepted.partition == "validation").to_numpy(dtype=np.int8),
        raw=accepted.raw.to_numpy(dtype=float),
        residual=accepted.target.to_numpy(dtype=float),
        factor=accepted.factor.to_numpy(dtype=float),
        alpha=accepted.alpha.to_numpy(dtype=float),
        beta=accepted.beta.to_numpy(dtype=float),
        pairs=accepted.history_pairs.to_numpy(dtype=np.int16),
        pattern=accepted.pattern.to_numpy(dtype=np.int16),
    )
    _require(
        np.isfinite(columns["raw"]).all() and np.isfinite(columns["factor"]).all(),
        "Un objetivo aceptado no tiene retornos siguientes finitos",
    )
    return dict(
        market=market,
        symbol=symbol,
        samples=len(labels),
        counts=sorted(counts.items()),
        variant_counts=sorted(variant_counts.items(), key=str),
        columns=columns,
        common=dict(
            rows=len(common),
            digest=hashlib.sha256(mask.tobytes()).hexdigest(),
            sums={k: v.tolist() for k, v in sums.items()},
            jumps={k: v.tolist() for k, v in jumps.items()},
        ),
        extremes=dict(
            raw=_extremes(accepted, "raw", declaration["extremes"], receipt["prices_sha256"]),
            residual=_extremes(
                accepted, "target", declaration["extremes"], receipt["prices_sha256"]
            ),
        ),
        samples_state=samples_state,
    )


# Agregación sobre todos los activos.


def moments(x, quantiles, thresholds):
    """Momentos, cuantiles y colas fijas de una muestra, sin recortar extremos."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return dict(n=0)
    centered = x - x.mean()
    variance = (centered @ centered) / len(x)
    std = float(np.sqrt(variance))
    return dict(
        n=len(x),
        mean=float(x.mean()),
        std=float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        skewness=float((centered**3).mean() / std**3) if std > 0 else None,
        excess_kurtosis=float((centered**4).mean() / variance**2 - 3) if std > 0 else None,
        mean_absolute=float(np.abs(x).mean()),
        quantiles=dict(zip(map(str, quantiles), np.quantile(x, quantiles).tolist(), strict=True)),
        max_absolute=float(np.abs(x).max()),
        beyond={str(t): int((np.abs(x) > t).sum()) for t in thresholds},
    )


def association(y, f):
    """Correlación y pendiente de y sobre f. Describe asociación, no causalidad."""
    y, f = np.asarray(y, dtype=float), np.asarray(f, dtype=float)
    if len(y) < 3:
        return dict(n=len(y), correlation=None, slope=None)
    fc, yc = f - f.mean(), y - y.mean()
    spread = float(np.sqrt((fc @ fc) * (yc @ yc)))
    return dict(
        n=len(y),
        correlation=float((fc @ yc) / spread) if spread > 0 else None,
        slope=float((fc @ yc) / (fc @ fc)) if fc @ fc > 0 else None,
    )


def _from_sums(s):
    n, sy, syy, sf, sff, syf, sym, smm, sm, sad, ssd = s
    if n < 3:
        return dict(n=int(n))
    vy, vf, vm = syy / n - (sy / n) ** 2, sff / n - (sf / n) ** 2, smm / n - (sm / n) ** 2
    cyf, cym = syf / n - sy * sf / n**2, sym / n - sy * sm / n**2
    return dict(
        n=int(n),
        std=float(np.sqrt(vy * n / (n - 1))),
        correlation_with_factor=float(cyf / np.sqrt(vy * vf)) if vy > 0 and vf > 0 else None,
        correlation_with_main=float(cym / np.sqrt(vy * vm)) if vy > 0 and vm > 0 else None,
        mean_absolute_difference=float(sad / n),
        root_mean_square_difference=float(np.sqrt(ssd / n)),
    )


def _from_jumps(j, thresholds):
    n = int(j[0])
    return dict(
        adjacent_pairs=n,
        mean_absolute_change=float(j[1] / n) if n else None,
        max_absolute_change=float(j[2]),
        share_above={
            str(t): float(c / n) if n else None for t, c in zip(thresholds, j[3:], strict=True)
        },
    )


def _concatenate(results):
    keys = results[0]["columns"].keys() if results else ()
    columns = {k: np.concatenate([r["columns"][k] for r in results]) for k in keys}
    columns["asset"] = np.concatenate(
        [np.full(len(r["columns"]["slot"]), i, dtype=np.int32) for i, r in enumerate(results)]
    )
    return columns


def _coverage(results, manifest, edition):
    counts = defaultdict(Counter)
    by_year = defaultdict(lambda: defaultdict(Counter))
    by_pattern = defaultdict(lambda: defaultdict(Counter))
    per_asset = []
    for r in results:
        asset = Counter()
        for (year, category, pattern), n in r["counts"]:
            counts[r["market"]][category] += n
            by_year[r["market"]][year][category] += n
            if pattern >= 0:
                by_pattern[r["market"]][pattern][category] += n
            asset[category] += n
        _require(sum(asset.values()) == r["samples"], "Las categorías no suman las muestras")
        # La reserva de la prueba final no es una pérdida de cobertura.
        excluded = sum(n for c, n in asset.items() if c not in (*PARTITIONS, "final_test_reserved"))
        per_asset.append((r["market"], r["symbol"], excluded, r["samples"], asset))
    total = Counter()
    for market_counts in counts.values():
        total.update(market_counts)
    _require(
        sum(total.values()) == manifest["samples"]
        and all(total[p] == manifest["counts"][p] for p in PARTITIONS),
        "El diagnóstico no reconcilia las muestras y los recuentos del manifiesto",
    )
    states = Counter(item["state"] for item in edition["coverage"])
    empty = sorted(
        f"{item['market']}:{item['symbol']}"
        for item in edition["coverage"]
        if item["state"] == "encoded" and item["samples"] == 0
    )
    excluded_total = sum(a[2] for a in per_asset)
    ranked = sorted(per_asset, key=lambda a: (-a[2], a[0], a[1]))
    top = max(1, len(ranked) // 100)
    without = [a for a in per_asset if not any(a[4][p] for p in PARTITIONS)]
    return dict(
        assets=dict(
            candidates=edition["candidate_count"],
            states=dict(states),
            encoded_without_samples=empty,
            labelled=len(results),
            labelled_by_market=dict(Counter(r["market"] for r in results)),
            without_accepted_labels=len(without),
            without_accepted_labels_by_dominant_cause=dict(
                Counter(max(a[4], key=lambda c: (a[4][c], c)) for a in without)
            ),
            with_insufficient_history=sum(
                1
                for a in per_asset
                if a[4]["insufficient_stock_history"] + a[4]["missing_factor_history"]
            ),
        ),
        samples=dict(
            total=manifest["samples"],
            by_market={m: dict(sorted(c.items())) for m, c in sorted(counts.items())},
            by_market_and_year={
                m: {str(y): dict(sorted(c.items())) for y, c in sorted(years.items())}
                for m, years in sorted(by_year.items())
            },
        ),
        losses=dict(
            excluded=excluded_total,
            top_assets=[
                dict(
                    market=m,
                    symbol=s,
                    excluded=e,
                    samples=n,
                    causes={c: v for c, v in sorted(a.items()) if c not in PARTITIONS},
                )
                for m, s, e, n, a in ranked[:10]
            ],
            top_percent_assets=top,
            top_percent_share=(
                sum(a[2] for a in ranked[:top]) / excluded_total if excluded_total else None
            ),
        ),
        by_pattern={
            m: {format(p, "05b"): dict(sorted(c.items())) for p, c in sorted(patterns.items())}
            for m, patterns in sorted(by_pattern.items())
        },
    )


def _distributions(columns, markets, declaration):
    quantiles, thresholds = declaration["quantiles"], declaration["return_thresholds"]
    out = {}
    for market, code in markets.items():
        for index, partition in enumerate(PARTITIONS):
            keep = (columns["market"] == code) & (columns["partition"] == index)
            raw, residual = columns["raw"][keep], columns["residual"][keep]
            out[f"{market}:{partition}"] = dict(
                raw=moments(raw, quantiles, thresholds),
                residual=moments(residual, quantiles, thresholds),
                variance_ratio=float(residual.var() / raw.var()) if raw.var() > 0 else None,
                correlation_raw_residual=association(residual, raw)["correlation"],
            )
    return out


def _by_year(columns, markets, years, declaration):
    out = {}
    largest = declaration["return_thresholds"][-1]
    bound = declaration["beta_bound"]
    for market, code in markets.items():
        rows = {}
        for year in sorted(set(years[columns["market"] == code].tolist())):
            keep = (columns["market"] == code) & (years == year)
            raw, residual = columns["raw"][keep], columns["residual"][keep]
            beta, alpha = columns["beta"][keep], columns["alpha"][keep]
            rows[str(year)] = dict(
                n=int(keep.sum()),
                partition=PARTITIONS[int(columns["partition"][keep].max())],
                std_raw=float(raw.std(ddof=1)) if len(raw) > 1 else None,
                std_residual=float(residual.std(ddof=1)) if len(raw) > 1 else None,
                beyond_largest_raw=int((np.abs(raw) > largest).sum()),
                beyond_largest_residual=int((np.abs(residual) > largest).sum()),
                beta=dict(
                    zip(
                        ("p05", "p25", "p50", "p75", "p95"),
                        np.quantile(beta, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
                        strict=True,
                    )
                ),
                alpha_p50=float(np.median(alpha)),
                beta_beyond_bound=int((np.abs(beta) > bound).sum()),
                partial_windows=int(
                    (columns["pairs"][keep] < declaration["main"]["history"]).sum()
                ),
                residual_on_factor=association(residual, columns["factor"][keep]),
                raw_on_factor=association(raw, columns["factor"][keep]),
            )
        out[market] = rows
    return out


def _monthly_beta(columns, markets, sessions):
    out = {}
    for market, code in markets.items():
        keep = columns["market"] == code
        codes = np.array([int(s[:4]) * 12 + int(s[5:7]) - 1 for s in sessions[market]])
        months = codes[columns["slot"][keep]]
        order = np.argsort(months, kind="stable")
        months, beta = months[order], columns["beta"][keep][order]
        bounds = np.flatnonzero(np.r_[True, months[1:] != months[:-1], True])
        out[market] = [
            dict(
                month=f"{months[a] // 12:04d}-{months[a] % 12 + 1:02d}",
                n=int(b - a),
                **dict(
                    zip(
                        ("p10", "p50", "p90"),
                        np.quantile(beta[a:b], [0.1, 0.5, 0.9]).tolist(),
                        strict=True,
                    )
                ),
            )
            for a, b in zip(bounds[:-1], bounds[1:], strict=True)
        ]
    return out


def _stability(columns, markets, declaration):
    out = {}
    thresholds = declaration["beta_jump_thresholds"]
    same = np.r_[False, columns["asset"][1:] == columns["asset"][:-1]]
    adjacent = same & np.r_[False, np.diff(columns["slot"]) == 1]
    beta_change = np.r_[np.nan, np.abs(np.diff(columns["beta"]))]
    alpha_change = np.r_[np.nan, np.abs(np.diff(columns["alpha"]))]
    for market, code in markets.items():
        for index, partition in enumerate(PARTITIONS):
            keep = adjacent & (columns["market"] == code) & (columns["partition"] == index)
            rows = (columns["market"] == code) & (columns["partition"] == index)
            beta = beta_change[keep]
            out[f"{market}:{partition}"] = dict(
                adjacent_pairs=int(keep.sum()),
                beta_change=moments(beta, declaration["quantiles"], thresholds),
                alpha_change_p99=float(np.quantile(alpha_change[keep], 0.99))
                if keep.any()
                else None,
                beta_beyond_bound=int(
                    (np.abs(columns["beta"][rows]) > declaration["beta_bound"]).sum()
                ),
                partial_windows=int(
                    (columns["pairs"][rows] < declaration["main"]["history"]).sum()
                ),
                minimum_pairs=int(columns["pairs"][rows].min()) if rows.any() else None,
            )
    return out


def _exposure(columns, markets, declaration):
    out = {}
    for market, code in markets.items():
        for index, partition in enumerate(PARTITIONS):
            keep = (columns["market"] == code) & (columns["partition"] == index)
            slot = columns["slot"][keep]
            factor, residual, raw = (
                columns["factor"][keep],
                columns["residual"][keep],
                columns["raw"][keep],
            )
            size = int(slot.max()) + 1 if len(slot) else 0
            assets = np.bincount(slot, minlength=size)
            daily_factor = np.bincount(slot, weights=factor, minlength=size)
            spread = np.zeros(size)
            np.maximum.at(
                spread, slot, np.abs(factor - (daily_factor / np.maximum(assets, 1))[slot])
            )
            _require(spread.max(initial=0.0) <= 1e-12, "El factor de una sesión no es único")
            days = assets >= declaration["minimum_assets_per_session"]
            mean_residual = np.bincount(slot, weights=residual, minlength=size)[days] / assets[days]
            mean_raw = np.bincount(slot, weights=raw, minlength=size)[days] / assets[days]
            daily = daily_factor[days] / assets[days]
            out[f"{market}:{partition}"] = dict(
                pooled_residual=association(residual, factor),
                pooled_raw=association(raw, factor),
                daily_mean_residual=association(mean_residual, daily),
                daily_mean_raw=association(mean_raw, daily),
                sessions=int(days.sum()),
                sessions_below_minimum=int(((assets > 0) & ~days).sum()),
            )
    return out


def _sensitivities(results, declaration):
    names = ["main", *(s["name"] for s in declaration["sensitivities"])]
    thresholds = declaration["beta_jump_thresholds"]
    out = {}
    digest = hashlib.sha256()
    for r in results:
        digest.update(f"{r['market']}:{r['symbol']}:{r['common']['digest']}\n".encode())
    for market in sorted({r["market"] for r in results}):
        chosen = [r for r in results if r["market"] == market]
        coverage = Counter()
        for r in chosen:
            for (name, partition), n in r["variant_counts"]:
                coverage[name, partition] += n
            for (_year, category, _pattern), n in r["counts"]:
                if category in PARTITIONS:
                    coverage["main", category] += n
        for partition in PARTITIONS:
            for name in names:
                sums = np.sum([r["common"]["sums"][name, partition] for r in chosen], axis=0)
                jumps = np.sum([r["common"]["jumps"][name, partition] for r in chosen], axis=0)
                jumps[2] = max(r["common"]["jumps"][name, partition][2] for r in chosen)
                out[f"{market}:{partition}:{name}"] = dict(
                    accepted_samples=coverage[name, partition],
                    coverage_change=coverage[name, partition] - coverage["main", partition],
                    common=_from_sums(sums),
                    beta_stability=_from_jumps(jumps, thresholds),
                )
    return dict(
        population=dict(
            rule="muestras con objetivo aceptado en la regresión principal y en cada sensibilidad",
            rows=sum(r["common"]["rows"] for r in results),
            sha256=digest.hexdigest(),
        ),
        variants=out,
    )


def _top_extremes(results, kind, count):
    out = {}
    for market in sorted({r["market"] for r in results}):
        rows = [
            dict(item, symbol=r["symbol"])
            for r in results
            if r["market"] == market
            for item in r["extremes"][kind]
        ]
        column = "raw" if kind == "raw" else "target"
        rows.sort(key=lambda item: (-abs(item[column]), item["symbol"], item["session"]))
        out[market] = rows[:count]
    return out


def _tasks(manifest, edition, declaration, *, samples):
    roots = manifest["roots"]
    factors = manifest["market_factors"]
    for receipt in manifest["assets"]:
        market, symbol = receipt["market"], receipt["symbol"]
        base = Path(roots["labels"]) / market / symbol
        yield dict(
            declaration=declaration,
            market=market,
            symbol=symbol,
            receipt=receipt,
            labels=str(base / "labels.parquet"),
            prices=str(Path(roots["prepared"]) / market / symbol / "prices.parquet"),
            factor=factors[market]["prices_path"],
            factor_sha256=factors[market]["prices_sha256"],
            calendar_start=edition["calendar_start"][market],
            samples=(
                str(Path(roots["samples"]) / market / symbol / "samples.parquet")
                if samples
                else None
            ),
        )


def _samples_available(manifest):
    root = Path(manifest["roots"]["samples"])
    missing = [
        f"{a['market']}:{a['symbol']}"
        for a in manifest["assets"]
        if not (root / a["market"] / a["symbol"] / "samples.parquet").is_file()
    ]
    return missing


def diagnose(targets, edition, declaration, *, workers=1, patterns=True, progress=None):
    """Informe completo. Lee los objetivos, no escribe nada y no abre la prueba final."""
    started = datetime.now(UTC)
    manifest, manifest_sha = read_manifest(Path(targets), 8 * 1024**2)
    meta, edition_sha = read_manifest(Path(edition), 64 * 1024**2)
    _require(
        manifest.get("kind") == "corpus_supervision"
        and manifest.get("final_test_opened") is False
        and manifest["configuration"]["source_manifest_sha256"] == edition_sha
        and sum(a["samples"] for a in manifest["assets"]) == manifest["samples"],
        "Los objetivos no corresponden a la edición o no cumplen su contrato",
    )
    missing = _samples_available(manifest) if patterns else []
    tasks = list(_tasks(manifest, meta, declaration, samples=patterns and not missing))
    results = []
    if workers == 1:
        for task in tasks:
            results.append(diagnose_asset(task))
            if progress:
                progress(len(results), len(tasks))
    else:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
            for result in pool.map(diagnose_asset, tasks, chunksize=4):
                results.append(result)
                if progress:
                    progress(len(results), len(tasks))
    markets = {m: i for i, m in enumerate(sorted({r["market"] for r in results}))}
    columns = _concatenate(results)
    codes = np.array([markets[r["market"]] for r in results], dtype=np.int8)
    columns["market"] = codes[columns["asset"]]
    sessions = {}
    end = date.fromisoformat(declaration["cutoff"])
    for market in markets:
        clock = MarketClock(market, meta["calendar_start"][market], "2024-01-05")
        sessions[market] = [d.isoformat() for d in clock.days if d <= end]
    years = np.zeros(len(columns["slot"]), dtype=np.int16)
    for market, code in markets.items():
        keep = columns["market"] == code
        calendar_years = np.array([int(s[:4]) for s in sessions[market]], dtype=np.int16)
        years[keep] = calendar_years[columns["slot"][keep]]
    states = Counter(r["samples_state"] for r in results)
    report = dict(
        schema_version=SCHEMA_VERSION,
        kind=KIND,
        scope="Diagnóstico de datos sobre entrenamiento y validación. Sin modelos ni aprendizaje.",
        declaration={k: v for k, v in declaration.items() if k != "sha256"},
        declaration_sha256=declaration["sha256"],
        targets=dict(path=str(targets), manifest_sha256=manifest_sha),
        edition=dict(path=str(edition), manifest_sha256=edition_sha),
        market_factors={
            m: dict(
                symbol=f["symbol"],
                prices_sha256=f["prices_sha256"],
                point_in_time_verified=f.get("point_in_time_verified"),
            )
            for m, f in sorted(manifest["market_factors"].items())
        },
        code_sha256=dict(
            diagnostics=sha256(Path(__file__)),
            residual_arrays=sha256(Path(__file__).with_name("residual_arrays.py")),
            budget_targets=sha256(Path(__file__).with_name("budget_targets.py")),
        ),
        versions=dict(numpy=np.__version__, pandas=pd.__version__, pyarrow=pa.__version__),
        stored_targets_recomputed=dict(accepted=int(len(columns["slot"])), bitwise_equal=True),
        coverage=_coverage(results, manifest, meta),
        mask_patterns=dict(
            available=bool(patterns and not missing and states.get("verified") == len(results)),
            missing_samples=len(missing),
            states=dict(states),
        ),
        distributions=_distributions(columns, markets, declaration),
        by_year=_by_year(columns, markets, years, declaration),
        monthly_beta=_monthly_beta(columns, markets, sessions),
        stability=_stability(columns, markets, declaration),
        exposure=_exposure(columns, markets, declaration),
        sector=declaration["sector"],
        sensitivities=_sensitivities(results, declaration),
        extremes=dict(
            raw=_top_extremes(results, "raw", declaration["extremes"]),
            residual=_top_extremes(results, "residual", declaration["extremes"]),
        ),
        started_at_utc=started.isoformat(),
        finished_at_utc=datetime.now(UTC).isoformat(),
        workers=workers,
        training_executed=False,
        final_test_opened=False,
    )
    if not report["mask_patterns"]["available"]:
        report["coverage"]["by_pattern"] = {}
    return report
