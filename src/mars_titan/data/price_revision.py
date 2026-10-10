"""Revisar los precios de una preparación completa con una auditoría nueva, sin rehacer el resto.

La edición v3.1 cambia solo la lectura de precios (conversión exacta y tolerancia de redondeo).
Noticias y fundamentales no dependen de los precios, así que se enlazan desde la preparación
padre tras comprobar su huella. Los precios se vuelven a leer de la auditoría nueva con el mismo
lector de la preparación. Si el Parquet resultante es idéntico al del padre, también se enlaza.
La revisión deriva además las sesiones sin filas en toda la fuente de cada mercado, que el
contrato de ventanas admite como huecos.
"""

import argparse
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .audited_prices import audit_catalog, read_audited_prices
from .cohort_files import read_manifest, safe_destination
from .cohort_samples import _prepared
from .corpus_preparation import STARTS
from .input_policy import HISTORICAL_MASKED, policy_identity
from .preparation import atomic_parquet
from .price_windows import calendar_digest, market_absent_sessions
from .source_numbers import NUMBER_PARSING
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock

REVISION = "audited_price_revision_v1"
_CODE = (
    "price_revision.py",
    "prices.py",
    "source_numbers.py",
    "audit.py",
    "audited_prices.py",
    "price_windows.py",
    "cohort_samples.py",
)
_LINKED = (
    "fundamentals.parquet",
    "news/news.parquet",
    "news/excluded.parquet",
    "news/manifest.json",
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def link_verified(source: Path, destination: Path, signature: str) -> None:
    """Enlazar un artefacto confirmado. Un enlace duro no copia datos ni cambia el origen."""
    safe_destination(destination)
    if destination.exists():
        if destination.is_symlink() or sha256(destination) != signature:
            raise ValueError(f"El destino enlazado no conserva su huella: {destination}")
        return
    if source.is_symlink() or sha256(source) != signature:
        raise ValueError(f"El artefacto de origen no conserva su huella: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.link-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    os.link(source, temporary)
    try:
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    if sha256(destination) != signature:
        raise ValueError(f"El enlace no conserva la huella del artefacto: {destination}")


def observed_sessions(audit_state: Path) -> dict[str, set[str]]:
    """Sesiones con alguna fila en algún archivo de la fuente, aceptada o excluida."""
    manifest, _ = read_manifest(Path(audit_state), 64 * 1024**2)
    root = Path(manifest["details_root"])
    result = {}
    for row in manifest["files"].values():
        folder = root / row["market"] / row["symbol"]
        sessions = result.setdefault(row["market"], set())
        artifacts = row.get("artifacts", {})
        for name, column in (("prices.parquet", "session"), ("exclusions.parquet", "source_date")):
            path = folder / name
            if sha256(path) != artifacts[name]["sha256"]:
                raise ValueError(f"Ha cambiado un derivado de la auditoría: {path}")
            values = pq.read_table(path, columns=[column])[column].to_pylist()
            sessions.update(value[:10] for value in values if isinstance(value, str))
    return result


def revise_prepared_prices(preparation, audit_state, output, *, clocks=None):
    """Preparar una edición con los precios de `audit_state` y el resto de `preparation`."""
    preparation, audit_state, output = (
        Path(p).absolute() for p in (preparation, audit_state, output)
    )
    parent, parent_hash = read_manifest(preparation)
    if (
        parent.get("schema_version") != 2
        or parent.get("kind") != "prepared_cohort"
        or parent.get("status") != "completed"
        or parent.get("failed_assets") != 0
        or any(parent.get(k) != v for k, v in policy_identity(HISTORICAL_MASKED).items())
        or not isinstance(parent.get("assets"), list)
    ):
        raise ValueError("La preparación padre debe estar completa, sin errores y con máscaras")
    records, audit_identity = audit_catalog(audit_state)
    source_root = Path(parent["prepared_root"]).resolve()
    for path in (source_root, preparation, audit_state, Path("dataset")):
        outside_source(path, output)
        outside_source(output, path)
    markets = sorted({row["market"] for row in parent["assets"]})
    clocks = clocks or {m: MarketClock(m, STARTS[m], "2026-01-01") for m in markets}
    code = {name: sha256(Path(__file__).with_name(name)) for name in _CODE}
    rtol = {key for record in records.values() for key in [record.get("ordering_rtol")]}
    if len(rtol) != 1:
        raise ValueError("La auditoría debe declarar una única tolerancia de orden")
    identity = dict(
        policy=REVISION,
        input_policy=HISTORICAL_MASKED,
        parent_sha256=parent_hash,
        parent_root=str(source_root),
        audit_state_sha256=audit_identity,
        number_parsing=NUMBER_PARSING,
        ordering_rtol=next(iter(rtol)),
        calendars={m: calendar_digest(clocks[m]) for m in markets},
        linked_artifacts=list(_LINKED),
        code=code,
    )
    safe_destination(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".edition.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = output / "configuration.json"
        if not config.exists():
            if any(path.name != ".edition.lock" for path in output.iterdir()):
                raise ValueError("El destino contiene una edición no identificada")
            atomic_json(config, identity)
        configured, config_hash = read_manifest(config)
        if configured != json.loads(json.dumps(identity)):
            raise ValueError("La configuración corresponde a otra revisión de precios")
        started = time.perf_counter()
        report = dict(
            {k: v for k, v in parent.items() if k not in {"elapsed_seconds", "reused_assets"}},
            status="running",
            assets=[],
            failed_assets=0,
            prepared_root=str(output / "prepared"),
            parent_preparation=dict(path=str(preparation), sha256=parent_hash),
            configuration=dict(parent.get("configuration", {}), price_revision_sha256=config_hash),
            price_revision=dict(
                number_parsing=NUMBER_PARSING,
                ordering_rtol=identity["ordering_rtol"],
                audit_state_sha256=audit_identity,
                prices_linked=0,
                prices_rewritten=0,
                accepted_price_rows_delta=0,
            ),
            training_ready=False,
        )
        totals = report["price_revision"]
        for row in parent["assets"]:
            current = dict(row)
            if row["state"] == "prepared":
                market, symbol = row["market"], row["symbol"]
                source = source_root / market / symbol
                destination = output / "prepared" / market / symbol
                safe_destination(destination)
                receipt = _revise_asset(
                    source,
                    destination,
                    records[market, symbol],
                    clocks[market],
                    parent["cohort_id"],
                    dict(parent_manifest_sha256=row["manifest_sha256"], revision=config_hash),
                )
                totals["prices_linked" if receipt["linked"] else "prices_rewritten"] += 1
                totals["accepted_price_rows_delta"] += receipt["delta"]
                current.update(
                    counts=receipt["counts"], manifest_sha256=sha256(destination / "manifest.json")
                )
            elif records.get((row["market"], row["symbol"]), {}).get("rows"):
                # La nueva auditoría admite filas de un activo que antes no tenía precios.
                raise ValueError(
                    f"El activo {row['market']}/{row['symbol']} necesita una preparación completa"
                )
            report["assets"].append(current)
            atomic_json(output / "progress.json", report)
        if code != {name: sha256(Path(__file__).with_name(name)) for name in code}:
            raise ValueError("El código cambió durante la revisión")
        if sha256(preparation) != parent_hash:
            raise ValueError("La preparación padre cambió durante la revisión")
        report["status"] = "completed"
        final = output / "manifest.json"
        if final.exists():
            prior, _ = read_manifest(final)
            if prior != json.loads(json.dumps(report)):
                raise ValueError("El manifiesto confirmado no coincide con la revisión")
        else:
            atomic_json(final, report)
        return dict(report, elapsed_seconds=time.perf_counter() - started)


def _revise_asset(source, destination, record, clock, cohort, lineage):
    """Reescribir los precios de un activo y enlazar el resto de sus artefactos."""
    origin, _, origin_hash = _prepared(source, clock, cohort, HISTORICAL_MASKED)
    if lineage["parent_manifest_sha256"] != origin_hash:
        raise ValueError("El activo no conserva la huella de su preparación padre")
    if record["source_sha256"] not in origin["sources"].values():
        raise ValueError("La auditoría nueva lee otro archivo de precios")
    marker = destination / "manifest.json"
    revision = dict(**lineage, audited_prices_sha256=record["sha256"])
    if marker.exists():
        current, _, _ = _prepared(destination, clock, cohort, HISTORICAL_MASKED)
        if current.get("price_revision") != revision:
            raise ValueError("El activo revisado pertenece a otra revisión")
        return dict(
            counts=current["counts"],
            linked=current["price_revision_linked"],
            delta=current["counts"]["prices"] - origin["counts"]["prices"],
        )
    for name in _LINKED:
        link_verified(source / name, destination / name, origin["artifacts"][name])
    prices, receipt, reserved = read_audited_prices(
        record, record["source_sha256"], clock, origin["policy"]["cutoff"]
    )
    staging = destination / ".prices.parquet"
    atomic_parquet(staging, pa.Table.from_pandas(prices, preserve_index=False))
    digest = sha256(staging)
    linked = digest == origin["artifacts"]["prices.parquet"]
    if linked:
        staging.unlink()
        link_verified(source / "prices.parquet", destination / "prices.parquet", digest)
    else:
        os.replace(staging, destination / "prices.parquet")
    manifest = dict(
        origin,
        artifacts=dict(origin["artifacts"], **{"prices.parquet": digest}),
        counts=dict(origin["counts"], prices=len(prices)),
        reserved_counts=dict(origin["reserved_counts"], prices=reserved),
        price_audit=receipt,
        price_revision=revision,
        price_revision_linked=linked,
        fingerprint=_digest(dict(parent=origin["fingerprint"], revision=revision)),
    )
    atomic_json(marker, manifest)
    return dict(
        counts=manifest["counts"],
        linked=linked,
        delta=len(prices) - origin["counts"]["prices"],
    )


def market_absences(audit_state, clocks, *, last="2023-12-31") -> dict:
    """Recibo de las sesiones sin filas en toda la fuente de cada mercado."""
    observed = observed_sessions(audit_state)
    return {
        market: dict(
            sessions=market_absent_sessions(clocks[market], observed[market], last=last),
            first_observed_session=min(observed[market]),
            last_observed_session=max(observed[market]),
            last_considered_session=min(max(observed[market]), last),
            calendar_start=clocks[market].days[0].isoformat(),
            calendar_end=clocks[market].days[-1].isoformat(),
            decisions_sha256=calendar_digest(clocks[market]),
        )
        for market in sorted(clocks)
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--audit-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = revise_prepared_prices(args.preparation, args.audit_state, args.output)
    markets = sorted({row["market"] for row in result["assets"]})
    clocks = {m: MarketClock(m, STARTS[m], "2026-01-01") for m in markets}
    absences = market_absences(args.audit_state, clocks)
    atomic_json(args.output / "market-absent-sessions.json", absences)
    print(
        json.dumps(
            dict(
                status=result["status"],
                price_revision=result["price_revision"],
                market_absent_sessions={m: v["sessions"] for m, v in absences.items()},
                elapsed_seconds=result["elapsed_seconds"],
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
