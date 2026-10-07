"""Contextos macro desde H.15 revisado, con metadatos históricos comprobados."""

import argparse
import hashlib
import io
import json
import resource
import tempfile
import time
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa

from .batches import read_bounded_table
from .cohort_files import read_manifest, safe_destination
from .macro import _available, calculate_macro, macro_calculation_contract
from .macro_acquisition import _archive_text, _readme_metadata
from .macro_coverage import _publish_directory, _read_catalog
from .macro_h15_archive import _Sources, prepare_h15_archive
from .macro_h15_documents import _SERIES, _day
from .macro_recalculation import _SCHEMA
from .preparation import atomic_parquet
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock

POLICY = "H15_REVIEWED_DOCUMENT_CONTEXTS_V1"
_RAW = dict(_SERIES.values())
_MAX_BYTES = 64 * 1024**2
_METADATA_BYTES = 8 * 1024**2
_MAX_CELLS = 1_000_000


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _confirm(hashes):
    for name, digest in hashes.items():
        path = Path(name)
        safe_destination(path)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError("Una fuente o un artefacto ha cambiado de huella")


def _metadata(manifest):
    if manifest is None:
        return {}, {}
    sources = _Sources(Path(manifest).absolute())
    meta = sources.meta
    if (
        not isinstance(meta, dict)
        or set(meta) != {"schema_version", "archives"}
        or type(meta["schema_version"]) is not int
        or meta["schema_version"] != 1
        or not isinstance(meta["archives"], dict)
        or set(meta["archives"]) != set(_RAW)
    ):
        raise ValueError("Los metadatos necesitan un archivo identificado por cada serie H.15")
    result = {}
    for identifier, ref in meta["archives"].items():
        series = _RAW[identifier]
        url = f"https://alfred.stlouisfed.org/series?seid={series}"
        if (
            not isinstance(ref, dict)
            or set(ref) != {"path", "sha256", "series_id", "url"}
            or ref["series_id"] != series
            or ref["url"] != url
        ):
            raise ValueError("El archivo de metadatos no identifica la serie oficial")
        _, content = sources.read(ref, _METADATA_BYTES)
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                if sum(info.file_size for info in archive.infolist()) > _METADATA_BYTES:
                    raise ValueError("La expansión de metadatos supera ocho MiB")
            readme, _ = _archive_text(content)
        except zipfile.BadZipFile as error:
            raise ValueError("El archivo de metadatos no es un ZIP válido") from error
        if (
            f"Series ID: {series}" not in readme.splitlines()
            or f"Link: {url}" not in readme.splitlines()
        ):
            raise ValueError("El README no corresponde a la serie y fuente declaradas")
        result[identifier] = dict(intervals=_readme_metadata(readme), source_hash=ref["sha256"])
    sources.confirm()
    return result, {str(path): digest for path, digest in sources.hashes.items()}


def _historical_metadata(row, records):
    publication = row["publication_date"]
    active = [r for r in records.get("intervals", []) if r["start"] <= publication <= r["end"]]
    fields = {
        name: {r["description"] for r in active if r["field"] == name}
        for name in ("native_unit", "seasonal_adjustment")
    }
    reason = (
        "ambiguous_historical_metadata"
        if any(len(values) > 1 for values in fields.values())
        else "missing_historical_metadata"
        if any(not values for values in fields.values())
        else None
    )
    if reason:
        return dict(
            value=None, native_unit=row["unit"], seasonal_adjustment=None, missing_reason=reason
        )
    if fields != {"native_unit": {"Percent"}, "seasonal_adjustment": {"Not Seasonally Adjusted"}}:
        raise ValueError("Los metadatos contradicen la unidad o el ajuste del catálogo H.15")
    return dict(
        value=row["value"],
        native_unit="Percent",
        seasonal_adjustment="Not Seasonally Adjusted",
        missing_reason=row["missing_reason"],
    )


def load_h15_events(manifest_path, document_edition, catalog_path, *, metadata_manifest=None):
    """Verificar el archivo documental y obtener hechos únicos, sin mezclar observaciones ALFRED."""
    manifest_path, document_edition, catalog_path = map(
        Path, (manifest_path, document_edition, catalog_path)
    )
    for path in (manifest_path, document_edition, catalog_path):
        safe_destination(path)
    required = ("configuration.json", "report.json", "observations.parquet", "source-manifest.json")
    if not all((document_edition / name).is_file() for name in required):
        raise ValueError("La edición documental debe estar confirmada antes de adaptarla")
    configuration, _ = read_manifest(document_edition / "configuration.json", 2 * 1024**2)
    verified = prepare_h15_archive(
        manifest_path,
        document_edition,
        markets=configuration["markets"],
        cutoff=configuration["cutoff"],
    )
    entries = _read_catalog(catalog_path)
    if any(len(entry["unit"]) > 256 for entry in entries.values()):
        raise ValueError("La unidad del catálogo supera la longitud permitida")
    for identifier, series in _RAW.items():
        expected = dict(
            kind="raw",
            series_id=series,
            frequency="D7" if series == "DFF" else "D",
            unit="percent_pa_NSA",
            vintage_policy="ALFRED_OR_RELEASE_ARCHIVE",
        )
        if identifier not in entries or any(
            entries[identifier].get(k) != v for k, v in expected.items()
        ):
            raise ValueError("El catálogo cambia la definición de una serie H.15")
    metadata, hashes = _metadata(metadata_manifest)
    for path in (manifest_path, catalog_path, *(document_edition / name for name in required)):
        hashes[str(path.absolute())] = sha256(path)
    manifest, signature = read_manifest(manifest_path, 2 * 1024**2)
    if signature != verified["source_manifest_sha256"]:
        raise ValueError("El manifiesto documental cambió después de comprobarlo")
    for document in manifest["documents"]:
        for ref in (
            document["html"],
            document["pdf"],
            document["review"],
            document["html"]["receipt"],
            document["pdf"]["receipt"],
        ):
            hashes[str((manifest_path.parent / ref["path"]).absolute())] = ref["sha256"]
    clocks = {
        m: MarketClock(m, configuration["calendar_start"], configuration["cutoff"])
        for m in configuration["markets"]
    }
    table = read_bounded_table(
        document_edition / "observations.parquet", max_rows=2048 * 30 * 2, max_bytes=_MAX_BYTES
    )
    unique = {}
    for row in table.to_pylist():
        market, available = row.pop("market"), row.pop("available_at")
        expected = _available(_day(row["publication_date"]), row["source_timezone"], clocks[market])
        if available != expected:
            raise ValueError("La disponibilidad documental no coincide con el calendario")
        key = row["indicator_id"], row["period_start"], row["publication_date"]
        prior, markets = unique.setdefault(key, (row, {}))
        if prior != row or market in markets:
            raise ValueError("La publicación contiene una versión en conflicto o duplicada")
        markets[market] = available.isoformat() if available is not None else None
    if len(unique) != verified["observations"] or any(
        set(markets) != set(clocks) for _, markets in unique.values()
    ):
        raise ValueError("Los mercados no concilian las observaciones documentales")
    events = []
    for row, availability in (unique[key] for key in sorted(unique)):
        records = metadata.get(row["indicator_id"], {})
        events.append(
            dict(
                indicator_id=row["indicator_id"],
                series_id=row["series_id"],
                period_start=row["period_start"],
                realtime_start=row["publication_date"],
                realtime_end="9999-12-31",
                original_realtime_start=row["publication_date"],
                source_timezone=row["source_timezone"],
                source_hash=row["pdf_sha256"],
                value_exact=row["value_exact"],
                document_value=row["value"],
                document_availability=availability,
                review_sha256=row["review_sha256"],
                html_sha256=row["html_sha256"],
                source_url=row["pdf_url"],
                metadata_source_hash=records.get("source_hash"),
                **_historical_metadata(row, records),
            )
        )
    effective = [
        dict(
            entry,
            acquisition_status="complete" if name in _RAW else "unavailable",
            acquisition_reason=None if name in _RAW else "not_in_h15_document_bundle",
        )
        if entry["kind"] == "raw"
        else dict(entry)
        for name, entry in entries.items()
    ]
    identity = dict(
        document_observations=len(events),
        document_markets=list(clocks),
        document_cutoff=configuration["cutoff"],
        calendar_start=configuration["calendar_start"],
        document_configuration_sha256=verified["configuration_sha256"],
        sources=hashes,
        metadata_scope="declared_intervals_at_document_publication",
    )
    _confirm(hashes)
    return events, effective, identity


def _code():
    names = (
        "macro_h15_contexts.py",
        "macro_h15_archive.py",
        "macro_h15_documents.py",
        "macro.py",
        "macro_formulas.py",
        "macro_recalculation.py",
        "macro_acquisition.py",
        "macro_model_vintages.py",
        "macro_release_contracts.py",
        "temporal.py",
        "batches.py",
        "cohort_files.py",
        "macro_coverage.py",
        "preparation.py",
        "storage.py",
    )
    return {name: sha256(Path(__file__).with_name(name)) for name in names}


def _coverage(table, decisions):
    values, reasons = table["value"].to_pylist(), table["missing_reason"].to_pylist()
    return dict(
        rows=table.num_rows,
        decisions=decisions,
        observed=table.num_rows - table["value"].null_count,
        missing_reasons=dict(
            Counter(reason for value, reason in zip(values, reasons, strict=True) if value is None)
        ),
    )


def _recover(output, configuration, events):
    saved, digest = read_manifest(output / "configuration.json", 2 * 1024**2)
    report, _ = read_manifest(output / "report.json", 2 * 1024**2)
    if (
        saved != configuration
        or report.get("configuration_sha256") != digest
        or any(
            report.get(k) != v
            for k, v in {
                "schema_version": 1,
                "kind": "h15_macro_contexts",
                "policy": POLICY,
                "status": "completed",
                "observations": len(events),
                "indicator_ids": configuration["indicator_ids"],
                "scope": configuration["scope"],
                "training_ready": False,
                "admission_required": True,
                "final_test_opened": False,
                "reused": False,
            }.items()
        )
    ):
        raise ValueError("El recibo no corresponde a esta edición de contextos")
    expected = {"events.json", *(f"macro-{market}.parquet" for market in configuration["markets"])}
    if set(report.get("artifacts", {})) != expected:
        raise ValueError("El recibo no conserva sus artefactos")
    if any((output / name).stat().st_size > _MAX_BYTES for name in expected):
        raise ValueError("Un artefacto de la edición supera el presupuesto")
    _confirm({str(output / name): digest for name, digest in report["artifacts"].items()})
    if read_manifest(output / "events.json", _MAX_BYTES)[0] != events:
        raise ValueError("El recibo no conserva los eventos documentales")
    coverage = {}
    for market in configuration["markets"]:
        table = read_bounded_table(
            output / f"macro-{market}.parquet", max_rows=_MAX_CELLS, max_bytes=_MAX_BYTES
        )
        count = configuration["decisions"][market]
        if table.schema != _SCHEMA or table.num_rows != count * len(configuration["indicator_ids"]):
            raise ValueError("El panel no conserva su esquema o recuento")
        coverage[market] = _coverage(table, count)
    if report.get("markets") != coverage:
        raise ValueError("El recibo no conserva los recuentos del panel")
    return dict(report, reused=True)


def prepare_h15_contexts(
    manifest_path,
    document_edition,
    catalog_path,
    output,
    *,
    start,
    end,
    markets=("US", "CN"),
    metadata_manifest=None,
    daily_lag_policy="source_records",
    alfred_source=None,
):
    """Publicar un panel nuevo con ausencias, sin habilitar aprendizaje ni unir fuentes ALFRED."""
    began = time.perf_counter()
    if alfred_source is not None:
        raise ValueError(
            "La unión de observaciones ALFRED requiere una política posterior explícita"
        )
    first, last = _day(start), _day(end)
    if first.year < 2000 or first > last:
        raise ValueError("Las decisiones deben pertenecer al intervalo histórico desde 2000")
    if (
        not isinstance(markets, (tuple, list))
        or not markets
        or len(set(markets)) != len(markets)
        or any(m not in {"US", "CN"} for m in markets)
    ):
        raise ValueError("Los mercados deben ser US y/o CN sin duplicados")
    calculation = macro_calculation_contract(daily_lag_policy=daily_lag_policy)
    output = Path(output).absolute()
    safe_destination(output)
    events, catalog, inputs = load_h15_events(
        manifest_path, document_edition, catalog_path, metadata_manifest=metadata_manifest
    )
    if end > inputs["document_cutoff"] or not set(markets) <= set(inputs["document_markets"]):
        raise ValueError("La salida excede el corte o los mercados documentales verificados")
    for source in (Path(document_edition), Path("dataset"), *(Path(p) for p in inputs["sources"])):
        outside_source(source, output)
        outside_source(output, source)
    calendar_start = min(start, inputs["calendar_start"])
    clocks = {m: MarketClock(m, calendar_start, end) for m in markets}
    decisions = {m: sum(first <= day <= last for day in clock.days) for m, clock in clocks.items()}
    if not all(decisions.values()) or sum(decisions.values()) * len(catalog) > _MAX_CELLS:
        raise ValueError("El cálculo debe tener sesiones y no superar un millón de celdas")
    configuration = dict(
        schema_version=1,
        policy=POLICY,
        inputs=inputs,
        catalog_sha256=sha256(Path(catalog_path)),
        effective_catalog_sha256=_digest(catalog),
        indicator_ids=[entry["id"] for entry in catalog],
        markets=list(markets),
        decisions=decisions,
        start=start,
        end=end,
        calendar_start=calendar_start,
        calendars={m: _digest([v.isoformat() for v in c.decisions]) for m, c in clocks.items()},
        calculation=calculation,
        pyarrow=pa.__version__,
        code_sha256=_code(),
        scope="retrospective_dated_document_archive_not_contemporaneous_capture",
        alfred_observation_policy="not_combined",
        limits=dict(cells=_MAX_CELLS, artifact_bytes=_MAX_BYTES),
    )
    if output.exists():
        report = _recover(output, configuration, events)
        _confirm(inputs["sources"])
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        stage = Path(temporary) / "edition"
        stage.mkdir()
        atomic_json(stage / "configuration.json", configuration)
        payload = json.dumps(events, ensure_ascii=False, allow_nan=False).encode()
        if len(payload) > _MAX_BYTES:
            raise ValueError("Los eventos documentales superan el presupuesto")
        atomic_json(stage / "events.json", events)
        coverage = {}
        for market, clock in clocks.items():
            rows = calculate_macro(
                events,
                catalog,
                clock,
                daily_lag_policy=daily_lag_policy,
                decision_start=datetime.combine(first, datetime.min.time(), UTC),
            )
            if len(rows) != decisions[market] * len(catalog):
                raise ValueError("El panel no conserva todas las posiciones del catálogo")
            table = pa.Table.from_pylist(rows, schema=_SCHEMA)
            if table.nbytes > _MAX_BYTES:
                raise ValueError("El panel decodificado supera el presupuesto")
            path = stage / f"macro-{market}.parquet"
            atomic_parquet(path, table)
            if path.stat().st_size > _MAX_BYTES:
                raise ValueError("El archivo de panel supera el presupuesto")
            coverage[market] = _coverage(table, decisions[market])
        report = dict(
            schema_version=1,
            kind="h15_macro_contexts",
            policy=POLICY,
            status="completed",
            configuration_sha256=sha256(stage / "configuration.json"),
            observations=len(events),
            indicator_ids=[e["id"] for e in catalog],
            markets=coverage,
            scope=configuration["scope"],
            training_ready=False,
            admission_required=True,
            final_test_opened=False,
            artifacts={
                name: sha256(stage / name)
                for name in ("events.json", *(f"macro-{market}.parquet" for market in markets))
            },
            elapsed_seconds=time.perf_counter() - began,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            reused=False,
        )
        atomic_json(stage / "report.json", report)
        _confirm(inputs["sources"])
        if _code() != configuration["code_sha256"]:
            raise ValueError("El código cambió durante el cálculo")
        _publish_directory(stage, output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "documents", "catalog", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--metadata-manifest", type=Path)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--market", action="append", choices=("US", "CN"))
    parser.add_argument(
        "--daily-lag-policy",
        choices=("source_records", "valid_observations"),
        default="source_records",
    )
    args = parser.parse_args(argv)
    result = prepare_h15_contexts(
        args.manifest,
        args.documents,
        args.catalog,
        args.output,
        start=args.start,
        end=args.end,
        markets=args.market or ("US", "CN"),
        metadata_manifest=args.metadata_manifest,
        daily_lag_policy=args.daily_lag_policy,
    )
    print(
        f"Conservadas {result['observations']} observaciones documentales. "
        "Admisión todavía pendiente."
    )


if __name__ == "__main__":
    main()
