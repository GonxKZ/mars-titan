"""Recorrer el inventario completo y confirmar cada activo de una cohorte."""

import argparse
import fcntl
import json
import os
import sqlite3
import tempfile
import time
from contextlib import closing
from datetime import UTC, date, datetime
from pathlib import Path

from ijson.common import JSONError

from .cohort_news import COHORT_POLICIES
from .cohort_preparation import prepare_cohort_asset
from .corpus_catalog import corpus_candidates
from .news_registry import reviews_for_asset
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock

STARTS = {"US": "1990-01-01", "CN": "1990-12-19"}


def _snapshot(registry, output, source):
    target, receipt = output / "reviews.sqlite", output / "review-snapshot.json"
    if target.is_symlink() or receipt.is_symlink():
        raise ValueError("La instantánea editorial no puede ser un enlace")
    if not target.exists():
        fd, name = tempfile.mkstemp(prefix=".reviews-", suffix=".sqlite", dir=output)
        os.close(fd)
        try:
            with closing(
                sqlite3.connect(registry.resolve().as_uri() + "?mode=ro", uri=True)
            ) as origin:
                metadata = dict(origin.execute("SELECT * FROM metadata"))
                if metadata.get("source_root") != str(source.resolve()):
                    raise ValueError("Las revisiones pertenecen a otro origen")
                with closing(sqlite3.connect(name)) as frozen:
                    origin.backup(frozen, pages=4096)
            with open(name, "rb") as stream:
                os.fsync(stream.fileno())
            os.replace(name, target)
        finally:
            if os.path.exists(name):
                os.unlink(name)
    digest = sha256(target)
    if receipt.exists():
        if json.loads(receipt.read_text())["sha256"] != digest:
            raise ValueError("La instantánea editorial ha cambiado")
    else:
        with closing(sqlite3.connect(target.as_uri() + "?mode=ro", uri=True)) as db:
            if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("La instantánea editorial está dañada")
            if dict(db.execute("SELECT * FROM metadata")).get("source_root") != str(
                source.resolve()
            ):
                raise ValueError("La instantánea corresponde a otro origen")
        atomic_json(receipt, dict(sha256=digest, confirmed_at_utc=datetime.now(UTC).isoformat()))
    return target, digest


def prepare_cohort(
    source: Path,
    inventory: Path,
    registry: Path,
    output: Path,
    *,
    cohort: str,
    markets: tuple[str, ...] = ("US", "CN"),
    cutoff: str = "2023-12-31",
    stop_after_assets: int | None = None,
) -> dict:
    """Reanudar las confirmaciones sin sustituir errores por una selección menor."""
    if (
        cohort not in COHORT_POLICIES
        or not markets
        or len(set(markets)) != len(markets)
        or not set(markets) <= STARTS.keys()
    ):
        raise ValueError("La cohorte o los mercados no son válidos")
    if date.fromisoformat(cutoff) > date(2023, 12, 31):
        raise ValueError("La reserva final permanece cerrada desde 2024")
    if stop_after_assets is not None and (
        type(stop_after_assets) is not int or stop_after_assets < 1
    ):
        raise ValueError("El corte operativo debe ser positivo")
    source, inventory, registry, output = (
        p.resolve() for p in (source, inventory, registry, output)
    )
    outside_source(source, output)
    for protected in (source, inventory, registry):
        outside_source(output, protected)
    if not inventory.is_file() or not registry.is_file():
        raise ValueError("Faltan el inventario o las revisiones editoriales")
    configuration = dict(
        cohort_id=cohort,
        markets=list(markets),
        cutoff=cutoff,
        source_root=str(source),
        inventory_sha256=sha256(inventory),
        registry_source=str(registry),
        code_sha256=sha256(Path(__file__)),
        preparation_sha256=sha256(Path(__file__).with_name("cohort_preparation.py")),
    )
    for name in (
        ".edition.lock",
        "configuration.json",
        "manifest.json",
        "prepared",
        "review-snapshot.json",
        "reviews.sqlite",
    ):
        if (output / name).is_symlink():
            raise ValueError("Un artefacto de la edición es un enlace")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".edition.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / "configuration.json"
        if path.exists():
            if json.loads(path.read_text()) != configuration:
                raise ValueError("La configuración pertenece a otra edición")
        elif any(p.name != ".edition.lock" for p in output.iterdir()):
            raise ValueError("El directorio contiene una edición no identificada")
        else:
            atomic_json(path, configuration)
        snapshot, snapshot_hash = _snapshot(registry, output, source)
        candidates = [
            asset
            for market in markets
            for asset in corpus_candidates(source, inventory, market, validate_files=False)
        ]
        clocks = {market: MarketClock(market, STARTS[market], "2026-01-01") for market in markets}
        report = dict(
            schema_version=1,
            kind="prepared_cohort",
            status="running",
            cohort_id=cohort,
            candidate_count=len(candidates),
            snapshot_sha256=snapshot_hash,
            configuration=configuration,
            assets=[],
            failed_assets=0,
            training_ready=False,
            prepared_root=str(output / "prepared"),
        )
        started = time.perf_counter()
        for asset in candidates:
            if stop_after_assets is not None and len(report["assets"]) >= stop_after_assets:
                report["status"] = "paused"
                break
            row = dict(market=asset["market"], symbol=asset["symbol"])
            if asset["source_errors"]:
                row["source_errors"] = asset["source_errors"]
            if not asset["has_all_sources"]:
                row.update(state="missing_modalities", missing=asset["missing_modalities"])
            else:
                try:
                    reviews = reviews_for_asset(snapshot, asset["symbol"], market=asset["market"])
                    result = prepare_cohort_asset(
                        source,
                        output / "prepared",
                        asset,
                        clocks[asset["market"]],
                        cohort=cohort,
                        reviews=reviews,
                        cutoff=cutoff,
                    )
                    row.update(
                        state="prepared",
                        counts=result["counts"],
                        reused=result["reused"],
                        manifest_sha256=sha256(
                            output
                            / "prepared"
                            / asset["market"]
                            / asset["symbol"]
                            / "manifest.json"
                        ),
                    )
                except (ValueError, OSError, sqlite3.Error, JSONError) as error:
                    row.update(state="failed", error_type=type(error).__name__, detail=str(error))
                    report["failed_assets"] += 1
            report["assets"].append(row)
            report["elapsed_seconds"] = time.perf_counter() - started
            atomic_json(output / "progress.json", report)
        else:
            report["status"] = "completed_with_errors" if report["failed_assets"] else "completed"
        report["elapsed_seconds"] = time.perf_counter() - started
        if sha256(inventory) != configuration["inventory_sha256"]:
            raise ValueError("El inventario cambió durante el recorrido")
        atomic_json(output / "manifest.json", report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("dataset"))
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", choices=tuple(COHORT_POLICIES), required=True)
    parser.add_argument("--market", choices=("US", "CN", "all"), default="all")
    parser.add_argument(
        "--stop-after-assets", type=int, help="Pausa operativa, no tamaño de muestra"
    )
    args = parser.parse_args()
    result = prepare_cohort(
        args.source,
        args.inventory,
        args.reviews,
        args.output,
        cohort=args.cohort,
        markets=("US", "CN") if args.market == "all" else (args.market,),
        stop_after_assets=args.stop_after_assets,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "assets"}, ensure_ascii=False))
    return 2 if result["failed_assets"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
