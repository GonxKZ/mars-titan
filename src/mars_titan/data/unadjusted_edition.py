"""Edición de precios sin ajustar de la población preparada, recuperable por activo.

Cada activo se reconstruye desde su CSV original, con la huella registrada en la
preparación, y se alinea con las sesiones de ``prices.parquet``. La salida conserva
los precios negociados reconstruidos, los factores, la comprobación de rejilla por
fila y los tramos verificados entre acciones corporativas. No modifica ``dataset/``
ni la población preparada y no lee precios posteriores al corte.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .storage import atomic_json, outside_source, sha256
from .unadjusted_prices import (
    DEFAULT_FIT_WINDOWS,
    DEFAULT_GAMMA_MAX,
    DEFAULT_RELATIVE_TOLERANCE,
    FIT_MIN_SHARE,
    FIT_RELATIVE_TOLERANCE,
    METHOD,
    MIN_FIT_MARGIN,
    MIN_FIT_SHARE,
    ProviderHistory,
    grid_concordance,
    price_grid,
    read_provider_history,
    reconstruct_unadjusted,
    shifted_events,
)

SCHEMA_VERSION = 1
CUTOFF = "2023-12-31"
SEGMENT_MIN_ROWS = 5
SEGMENT_MIN_RATE = 0.9
# La preparación leyó el CSV con pandas. Frente al analizador de pyarrow, la mayor
# diferencia relativa medida en los 5.012 activos es 3,8e-13, en precios muy pequeños.
PARSE_RTOL = 1e-12
POLICY = {
    "method": METHOD,
    "cutoff": CUTOFF,
    "fit_windows": list(DEFAULT_FIT_WINDOWS),
    "gamma_max": DEFAULT_GAMMA_MAX,
    "relative_tolerance": DEFAULT_RELATIVE_TOLERANCE,
    "min_fit_share": MIN_FIT_SHARE,
    "min_fit_margin": MIN_FIT_MARGIN,
    "fit_relative_tolerance": FIT_RELATIVE_TOLERANCE,
    "fit_min_share": FIT_MIN_SHARE,
    "segment_min_rows": SEGMENT_MIN_ROWS,
    "segment_min_rate": SEGMENT_MIN_RATE,
    "post_cutoff_reads": "split_ratios_only",
}
US_STRATA = (("2000-01-31", "start_2000"), ("2009-12-31", "2000s"), ("2017-12-31", "2010s"))
CN_STRATA = (("2006-12-31", "start_2006"), ("2014-12-31", "2007_2014"))
PRICE_SCHEMA = pa.schema(
    [
        ("session", pa.string()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.float64()),
        ("split_multiplier", pa.float64()),
        ("dividend_offset", pa.float64()),
        ("on_grid", pa.bool_()),
        ("verified", pa.bool_()),
    ]
)
EVENT_SCHEMA = pa.schema(
    [
        ("session", pa.string()),
        ("provider_dividend", pa.float64()),
        ("provider_split", pa.float64()),
        ("raw_dividend", pa.float64()),
    ]
)


def code_identity() -> dict:
    here = Path(__file__).parent
    return {name: sha256(here / name) for name in ("unadjusted_prices.py", "unadjusted_edition.py")}


def population_assets(prepared_root: Path) -> list[dict]:
    """Activos con precios de la población preparada y las huellas de sus fuentes."""
    prepared_root = Path(prepared_root)
    manifest = json.loads((prepared_root / "manifest.json").read_text())
    if manifest.get("kind") != "prepared_cohort" or manifest.get("status") != "completed":
        raise ValueError("La población preparada no está completa")
    records = []
    for asset in manifest["assets"]:
        if asset.get("state") != "prepared" or not asset["counts"].get("prices"):
            continue
        folder = prepared_root / "prepared" / asset["market"] / asset["symbol"]
        detail = json.loads((folder / "manifest.json").read_text())
        series = [k for k in detail["sources"] if k.startswith("time_series/")]
        if (
            len(series) != 1
            or detail["price_audit"]["source_sha256"] != detail["sources"][series[0]]
        ):
            raise ValueError(f"Fuente de precios ambigua en {asset['symbol']}")
        records.append(
            {
                "market": asset["market"],
                "symbol": asset["symbol"],
                "source": series[0],
                "source_sha256": detail["sources"][series[0]],
                "prices": str(folder / "prices.parquet"),
                "prices_sha256": detail["artifacts"]["prices.parquet"],
                "rows": asset["counts"]["prices"],
            }
        )
    return sorted(records, key=lambda r: (r["market"], r["symbol"]))


def _stratum(record: dict, first: str) -> str:
    strata = US_STRATA if record["market"] == "US" else CN_STRATA
    label = next((name for end, name in strata if first <= end), "late")
    exchange = record["symbol"].rsplit(".", 1)[-1] if record["market"] == "CN" else ""
    return f"{record['market']}:{exchange}:{label}"


def select_audit_sample(records, first_sessions, *, seed: str, per_stratum: dict, exclude=()):
    """Muestra estratificada por mercado, bolsa y antigüedad, ordenada por una huella fija."""
    excluded = set(exclude)
    groups = {}
    for record in records:
        key = (record["market"], record["symbol"])
        if key in excluded:
            continue
        stratum = _stratum(record, first_sessions[key])
        digest = hashlib.sha256(
            f"{seed}:{record['market']}:{record['symbol']}".encode()
        ).hexdigest()
        groups.setdefault(stratum, []).append((digest, record))
    chosen = []
    for stratum, count in sorted(per_stratum.items()):
        members = sorted(groups.get(stratum, []), key=lambda item: item[0])
        if len(members) < count:
            raise ValueError(f"El estrato {stratum} no tiene {count} activos")
        chosen.extend({**record, "stratum": stratum} for _, record in members[:count])
    return chosen


def _segments(sessions, events, on_grid):
    """Tramos entre acciones corporativas y filas verificadas.

    Se recorren desde el más reciente. Un tramo de al menos ``SEGMENT_MIN_ROWS`` filas
    con menos de ``SEGMENT_MIN_RATE`` en rejilla detiene la verificación. Un tramo
    más corto que falla no la detiene, porque no basta para rechazar los factores
    anteriores, pero sus filas quedan sin verificar.
    """
    boundaries = np.searchsorted(sessions, events, side="left")
    labels = np.searchsorted(boundaries, np.arange(len(sessions)), side="right")
    summary, verified = [], np.zeros(len(sessions), dtype=bool)
    for label in range(labels.max(initial=-1), -1, -1):
        rows = np.flatnonzero(labels == label)
        if not len(rows):
            continue
        rate = float(on_grid[rows].mean())
        summary.append({"start": str(sessions[rows[0]]), "rows": int(len(rows)), "rate": rate})
        if rate < SEGMENT_MIN_RATE:
            if len(rows) >= SEGMENT_MIN_ROWS:
                break
            continue
        verified[rows] = True
    return summary[::-1], verified


def _validation(market, sessions, close, first_fit):
    rows = sessions < np.datetime64(first_fit)
    return grid_concordance(*price_grid(market, sessions[rows], close[rows]))


def _control(history, market, dividends, rows, adjusted_close, affected, sessions):
    """Concordancia de una variante incorrecta en las filas afectadas por dividendos."""
    variant = ProviderHistory(
        history.sessions,
        history.open,
        history.high,
        history.low,
        history.close,
        history.volume,
        dividends,
        history.splits,
    )
    result = reconstruct_unadjusted(variant, market, CUTOFF)
    if result["fit"]["gamma"] is None:
        return {"fit": False, "affected": None}
    close = adjusted_close * result["scale"][rows]
    return {
        "fit": True,
        "affected": grid_concordance(*price_grid(market, sessions[affected], close[affected])),
    }


def reconstruct_asset(record: dict, dataset_root: Path) -> tuple[pa.Table, pa.Table, dict]:
    """Reconstruir un activo y devolver precios, eventos y recibo, sin escribir."""
    source = Path(dataset_root) / record["source"]
    if sha256(source) != record["source_sha256"]:
        raise ValueError(f"El CSV de {record['symbol']} no coincide con la preparación")
    if sha256(Path(record["prices"])) != record["prices_sha256"]:
        raise ValueError(f"Los precios preparados de {record['symbol']} han cambiado")
    market = record["market"]
    history = read_provider_history(source)
    result = reconstruct_unadjusted(history, market, CUTOFF)
    kept = result["history"]
    prepared = pq.read_table(record["prices"], columns=["session", "open", "high", "low", "close"])
    sessions = np.array(prepared.column("session").to_pylist(), dtype="datetime64[D]")
    rows = np.searchsorted(kept.sessions, sessions)
    if (rows >= len(kept.sessions)).any() or (
        kept.sessions[np.minimum(rows, len(kept.sessions) - 1)] != sessions
    ).any():
        raise ValueError(f"Las sesiones preparadas de {record['symbol']} no están en el CSV")
    adjusted = {name: prepared.column(name).to_numpy() for name in ("open", "high", "low", "close")}
    for name, values in adjusted.items():
        if not np.allclose(values, getattr(kept, name)[rows], rtol=PARSE_RTOL, atol=0):
            raise ValueError(f"El precio preparado {name} de {record['symbol']} no es el original")
    scale = result["scale"][rows]
    raw = {name: values * scale for name, values in adjusted.items()}

    fit = result["fit"]
    close = raw["close"]
    on_grid, chance = price_grid(market, sessions, close)
    events_mask = (kept.dividends > 0) | (kept.splits > 0)
    event_sessions = kept.sessions[events_mask]
    receipt = {
        "market": market,
        "symbol": record["symbol"],
        "source": record["source"],
        "source_sha256": record["source_sha256"],
        "prices_sha256": record["prices_sha256"],
        "rows": int(len(sessions)),
        "fit": fit,
        "split_ratio_after_cutoff": result["split_ratio_after_cutoff"],
        "events": int(events_mask.sum()),
        "zero_volume_rows": int((result["volume"][rows] == 0).sum()),
        "all_rows": grid_concordance(on_grid, chance),
    }
    verified = np.zeros(len(sessions), dtype=bool)
    if fit["gamma"] is None:
        receipt["status"] = "fit_failed"
    else:
        segments, verified = _segments(sessions, event_sessions, on_grid)
        receipt.update(
            validation=_validation(market, sessions, close, fit["first_fit_session"]),
            segments=len(segments),
            failed_segments=[s for s in segments if s["rate"] < SEGMENT_MIN_RATE],
            verified_since=str(sessions[verified][0]) if verified.any() else None,
            verified_rows=int(verified.sum()),
            status="verified" if verified.all() else "partial" if verified.any() else "unverified",
        )
        # Filas de validación anteriores al último dividendo previo a la ventana de ajuste.
        fit_start = np.datetime64(fit["first_fit_session"])
        ex_dates = kept.sessions[1:][kept.dividends[1:] > 0]
        ex_dates = ex_dates[ex_dates < fit_start]
        affected = sessions < (ex_dates[-1] if len(ex_dates) else sessions[0])
        receipt["dividends_before_fit"] = int(len(ex_dates))
        receipt["affected"] = grid_concordance(
            *price_grid(market, sessions[affected], close[affected])
        )
        if affected.any():
            variants = {
                "no_dividends": np.zeros_like(history.dividends),
                "ex_date_plus_one": shifted_events(history.dividends, 1),
                "ex_date_minus_one": shifted_events(history.dividends, -1),
            }
            receipt["controls"] = {
                name: _control(history, market, values, rows, adjusted["close"], affected, sessions)
                for name, values in variants.items()
            }
    prices = pa.Table.from_arrays(
        [
            pa.array(prepared.column("session").to_pylist(), pa.string()),
            *(pa.array(raw[name]) for name in ("open", "high", "low", "close")),
            pa.array(result["volume"][rows]),
            pa.array(result["split_multiplier"][rows]),
            pa.array(result["dividend_offset"][rows]),
            pa.array(on_grid),
            pa.array(verified),
        ],
        schema=PRICE_SCHEMA,
    )
    events = pa.Table.from_arrays(
        [
            pa.array([str(s) for s in event_sessions], pa.string()),
            pa.array(kept.dividends[events_mask]),
            pa.array(kept.splits[events_mask]),
            pa.array(result["raw_dividend"][events_mask]),
        ],
        schema=EVENT_SCHEMA,
    )
    return prices, events, receipt


def _write_asset(destination: Path, prices: pa.Table, events: pa.Table, receipt: dict) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        pq.write_table(prices, staging / "prices.parquet", compression="zstd")
        pq.write_table(events, staging / "events.parquet", compression="zstd")
        receipt = {
            **receipt,
            "artifacts": {
                name: sha256(staging / name) for name in ("prices.parquet", "events.parquet")
            },
        }
        atomic_json(staging / "receipt.json", receipt)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return receipt


def _reusable(destination: Path, record: dict, identity: dict) -> dict | None:
    path = destination / "receipt.json"
    if not path.is_file():
        return None
    receipt = json.loads(path.read_text())
    if (
        receipt.get("identity") != identity
        or receipt.get("source_sha256") != record["source_sha256"]
    ):
        return None
    if any(sha256(destination / name) != value for name, value in receipt["artifacts"].items()):
        return None
    return receipt


def build_edition(prepared_root: Path, dataset_root: Path, output: Path, *, symbols=None) -> dict:
    """Escribir o reanudar la edición. ``symbols`` limita los activos a una lista."""
    prepared_root, dataset_root, output = Path(prepared_root), Path(dataset_root), Path(output)
    outside_source(dataset_root, output)
    outside_source(prepared_root, output)
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    records = population_assets(prepared_root)
    if symbols is not None:
        wanted = {(m, s) for m, s in symbols}
        records = [r for r in records if (r["market"], r["symbol"]) in wanted]
        if len(records) != len(wanted):
            raise ValueError("La muestra contiene activos ausentes de la población")
    identity = {"schema_version": SCHEMA_VERSION, "policy": POLICY, "code": code_identity()}
    started = time.perf_counter()
    receipts, reused = [], 0
    for record in records:
        destination = output / "assets" / record["market"] / record["symbol"]
        receipt = _reusable(destination, record, identity)
        if receipt is None:
            prices, events, receipt = reconstruct_asset(record, dataset_root)
            receipt = _write_asset(destination, prices, events, {**receipt, "identity": identity})
        else:
            reused += 1
        receipts.append(receipt)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    for receipt in receipts:
        digest.update(json.dumps(receipt["artifacts"], sort_keys=True).encode())
    manifest = {
        "kind": "unadjusted_price_edition",
        "edition_id": digest.hexdigest(),
        "identity": identity,
        "prepared_manifest_sha256": sha256(prepared_root / "manifest.json"),
        "price_basis": "unadjusted_reconstructed",
        "corporate_actions_complete": False,
        "assets": len(receipts),
        "reused_assets": reused,
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summarize(receipts),
        "receipts": {f"{r['market']}/{r['symbol']}": r["artifacts"] for r in receipts},
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def summarize(receipts) -> dict:
    """Recuentos por mercado y estado, y concordancia de validación agregada."""
    result = {}
    for market in sorted({r["market"] for r in receipts}):
        group = [r for r in receipts if r["market"] == market]
        status = {}
        for r in group:
            status[r["status"]] = status.get(r["status"], 0) + 1
        validated = [
            r["validation"] for r in group if r.get("validation") and r["validation"]["rows"]
        ]
        rows = sum(v["rows"] for v in validated)
        result[market] = {
            "assets": len(group),
            "status": status,
            "rows": sum(r["rows"] for r in group),
            "verified_rows": sum(r.get("verified_rows", 0) for r in group),
            "validation_rows": rows,
            "validation_rate": sum(v["on_grid"] for v in validated) / rows if rows else None,
            "validation_chance": sum(v["chance"] * v["rows"] for v in validated) / rows
            if rows
            else None,
            "median_asset_validation_rate": float(np.median([v["rate"] for v in validated]))
            if validated
            else None,
            "zero_volume_rows": sum(r["zero_volume_rows"] for r in group),
            "controls": _control_summary([r for r in group if "controls" in r]),
        }
    return result


def _control_summary(receipts) -> dict:
    """Concordancia en filas afectadas por dividendos del método y de sus controles."""
    if not receipts:
        return {"assets": 0}

    def pooled(items):
        rows = sum(item["rows"] for item in items)
        return sum(item["on_grid"] for item in items) / rows if rows else None

    result = {
        "assets": len(receipts),
        "method_affected_rate": pooled([r["affected"] for r in receipts]),
    }
    for name in ("no_dividends", "ex_date_plus_one", "ex_date_minus_one"):
        fitted = [r["controls"][name]["affected"] for r in receipts if r["controls"][name]["fit"]]
        result[name] = {"fit_failed": len(receipts) - len(fitted), "affected_rate": pooled(fitted)}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="Elegir la muestra auditada")
    sample.add_argument("--prepared", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--seed", default="mars-titan-unadjusted-20261009")
    sample.add_argument("--exclude", nargs="*", default=[], help="MERCADO:SÍMBOLO")
    build = commands.add_parser("build", help="Construir o reanudar la edición")
    build.add_argument("--prepared", type=Path, required=True)
    build.add_argument("--dataset", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--assets", type=Path, help="JSON de la orden sample")
    args = parser.parse_args(argv)
    if args.command == "sample":
        records = population_assets(args.prepared)
        first = {
            (r["market"], r["symbol"]): pq.read_table(r["prices"], columns=["session"])
            .column("session")[0]
            .as_py()
            for r in records
        }
        per_stratum = {f"US::{name}": 10 for name in ("start_2000", "2000s", "2010s", "late")}
        per_stratum.update(
            {
                f"CN:{ex}:{name}": 3
                for ex in ("SS", "SZ")
                for name in ("start_2006", "2007_2014", "late")
            }
        )
        exclude = [tuple(item.split(":", 1)) for item in args.exclude]
        chosen = select_audit_sample(
            records, first, seed=args.seed, per_stratum=per_stratum, exclude=exclude
        )
        atomic_json(
            args.output,
            {
                "seed": args.seed,
                "per_stratum": per_stratum,
                "exclude": args.exclude,
                "assets": chosen,
            },
        )
        print(f"Muestra: {len(chosen)} activos")
        return
    symbols = None
    if args.assets:
        symbols = [
            (a["market"], a["symbol"]) for a in json.loads(args.assets.read_text())["assets"]
        ]
    manifest = build_edition(args.prepared, args.dataset, args.output, symbols=symbols)
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
