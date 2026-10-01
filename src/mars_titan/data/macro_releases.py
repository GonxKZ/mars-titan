"""Preparar paneles macro desde un archivo explícito de comunicados oficiales."""

import argparse
import csv
import hashlib
import json
import os
import re
import resource
import subprocess
import tempfile
import time
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa

from .macro import calculate_macro
from .macro_acquisition import _atomic_bytes
from .macro_coverage import _publish_directory
from .macro_release_contracts import RELEASE_CONCEPTS, release_exclusion, release_source
from .macro_release_documents import extract_nbs_release, extract_pboc_release
from .preparation import atomic_parquet
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock

_MAX_BYTES = 3 * 1024**2
_OBSERVATION_SCHEMA = pa.schema(
    [
        pa.field(name, pa.string())
        for name in (
            "source_url",
            "source_title",
            "source_hash",
            "source_timezone",
            "declared_publication_date",
            "archive_path_date",
            "realtime_start",
            "realtime_end",
            "availability_policy",
            "indicator_id",
            "period_start",
            "reference_period_end",
            "native_unit",
            "seasonal_adjustment",
            "missing_reason",
            "source_value",
            "source_unit",
        )
    ]
    + [pa.field("value", pa.float64()), pa.field("publication_timestamp_verified", pa.bool_())]
)


def _month(value):
    if not isinstance(value, str) or not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", value):
        raise ValueError("El periodo de referencia debe tener formato YYYY-MM")
    return date.fromisoformat(value + "-01")


def _manifest(path):
    if path.stat().st_size > 2 * 1024**2:
        raise ValueError("El manifiesto de comunicados supera dos MiB")
    manifest = json.loads(path.read_text())
    fields = {"schema_version", "first_reference_period", "last_reference_period", "documents"}
    if not isinstance(manifest, dict) or set(manifest) != fields or manifest["schema_version"] != 1:
        raise ValueError("El manifiesto de comunicados no cumple su contrato")
    first, last = (
        _month(manifest[key]) for key in ("first_reference_period", "last_reference_period")
    )
    if last < first or (last - first).days > 366 * 50:
        raise ValueError("El intervalo del archivo está invertido o supera cincuenta años")
    documents = manifest["documents"]
    if not isinstance(documents, list) or not 1 <= len(documents) <= 4096:
        raise ValueError("El archivo requiere entre uno y 4096 comunicados")
    seen = set()
    for item in documents:
        expected = {
            "url",
            "sha256",
            "indicator_id",
            "reference_periods",
            "availability_bound",
            "exclusion",
        }
        if not isinstance(item, dict) or set(item) != expected:
            raise ValueError("El comunicado declarado no cumple el contrato")
        release_source(item["url"])
        if item["url"] in seen:
            raise ValueError("El manifiesto repite un comunicado")
        seen.add(item["url"])
        if not isinstance(item["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("El comunicado necesita una huella SHA-256")
        if item["indicator_id"] not in RELEASE_CONCEPTS:
            raise ValueError("El comunicado declara un concepto desconocido")
        periods = item["reference_periods"]
        if not isinstance(periods, list) or not 1 <= len(periods) <= 2:
            raise ValueError("Se necesita el periodo explícito del comunicado")
        for period in periods:
            _month(period)
        date.fromisoformat(item["availability_bound"])
    expected_periods = {
        f"{i // 12:04d}-{i % 12 + 1:02d}"
        for i in range(first.year * 12 + first.month - 1, last.year * 12 + last.month)
    }
    return manifest, expected_periods


def _download(url):
    release_source(url)
    with tempfile.TemporaryDirectory(prefix="mars-release-download-") as directory:
        path = Path(directory) / "document.html"
        result = subprocess.run(
            [
                "curl",
                "--silent",
                "--show-error",
                "--fail",
                "--proto",
                "=https",
                "--connect-timeout",
                "15",
                "--max-time",
                "45",
                "--max-filesize",
                str(_MAX_BYTES),
                "--output",
                str(path),
                "--write-out",
                "%{http_code}\n%{url_effective}",
                url,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=50,
        )
        if result.returncode or result.stdout.splitlines() != ["200", url]:
            raise OSError("No se pudo descargar el comunicado oficial: " + result.stderr[:500])
        if not path.is_file() or path.stat().st_size > _MAX_BYTES:
            raise ValueError("El comunicado descargado supera el presupuesto")
        return path.read_bytes()


def _content(item, cache, download):
    path = cache / (item["sha256"] + ".html")
    if path.is_symlink():
        raise ValueError("El archivo de comunicados no admite enlaces simbólicos")
    present = path.is_file()
    if present:
        if not 0 < path.stat().st_size <= _MAX_BYTES:
            raise ValueError("El comunicado vacío o demasiado grande no puede admitirse")
        content = path.read_bytes()
    elif download:
        content = _download(item["url"])
    else:
        raise FileNotFoundError("Falta el comunicado con SHA-256 " + item["sha256"])
    if len(content) > _MAX_BYTES or hashlib.sha256(content).hexdigest() != item["sha256"]:
        raise ValueError("La huella del comunicado difiere del manifiesto")
    if not present:
        _atomic_bytes(path, content)
    return content


def _verify_document(item, result):
    if (
        result["realtime_start"] != item["availability_bound"]
        or result["exclusion"] != item["exclusion"]
    ):
        raise ValueError("El documento no acredita la disponibilidad o exclusión declaradas")
    records = result["observations"]
    if records:
        periods = [max(row["period_start"] for row in records)[:7]]
        if {row["indicator_id"] for row in records} != {item["indicator_id"]}:
            raise ValueError("El comunicado no contiene el concepto declarado")
    elif (
        result["exclusion"] == "nonmonthly_reference_period"
        and item["indicator_id"] == "cn_industrial_yoy_published"
    ):
        year = re.findall(r"\b20\d{2}\b", result["source_title"])
        if len(year) != 1:
            raise ValueError("El periodo industrial conjunto no identifica su año")
        periods = [year[0] + "-01", year[0] + "-02"]
    else:
        raise ValueError("El comunicado no acredita la observación mensual requerida")
    if periods != item["reference_periods"]:
        raise ValueError("El comunicado no corresponde al periodo de referencia declarado")
    return records


def prepare_releases(
    manifest_path: Path,
    catalog_path: Path,
    source_cache: Path,
    output: Path,
    *,
    market: str,
    start: str,
    end: str,
    download: bool = False,
) -> dict:
    """Publicar una edición inmutable después de comprobar fuentes, periodos y cobertura."""
    began = time.perf_counter()
    manifest_path, catalog_path, source_cache, output = map(
        Path, (manifest_path, catalog_path, source_cache, output)
    )
    for source in (manifest_path, catalog_path, source_cache, Path("dataset")):
        outside_source(source, output)
    outside_source(Path("dataset"), source_cache)
    if output.exists() or output.is_symlink():
        raise FileExistsError("La edición de comunicados ya existe")
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last or (last - first).days > 366 * 50:
        raise ValueError("El intervalo de decisiones está invertido o supera cincuenta años")
    manifest_hash, catalog_hash = sha256(manifest_path), sha256(catalog_path)
    manifest, periods = _manifest(manifest_path)
    final_reference = _month(manifest["last_reference_period"])
    if last.year * 12 + last.month > final_reference.year * 12 + final_reference.month + 1:
        raise ValueError("El archivo no cubre los meses necesarios para el final solicitado")
    if catalog_path.stat().st_size > 2 * 1024**2:
        raise ValueError("El catálogo supera el presupuesto")
    with catalog_path.open(newline="", encoding="utf-8") as stream:
        catalog = [r for r in csv.DictReader(stream) if r["id"] in RELEASE_CONCEPTS]
    if (
        len(catalog) != 6
        or {r["id"] for r in catalog} != set(RELEASE_CONCEPTS)
        or any(release_exclusion(r) for r in catalog)
    ):
        raise ValueError("El catálogo no acredita los seis conceptos de los comunicados")
    clock = MarketClock(market, start, end)
    if not clock.decisions:
        raise ValueError("El intervalo solicitado no contiene sesiones de mercado")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        stage = Path(temporary) / "edition"
        (stage / "sources").mkdir(parents=True)
        records, exclusions, consumed_bytes = [], [], 0
        coverage = {identifier: set() for identifier in RELEASE_CONCEPTS}
        for item in manifest["documents"]:
            if date.fromisoformat(item["availability_bound"]) >= last:
                raise ValueError(
                    "El archivo incluye un comunicado posterior al intervalo solicitado"
                )
            content = _content(item, source_cache, download)
            family, _ = release_source(item["url"])
            result = (extract_nbs_release if family == "nbs" else extract_pboc_release)(
                content, item["url"]
            )
            records.extend(_verify_document(item, result))
            if len(records) > 200_000:
                raise ValueError("El archivo supera el presupuesto de observaciones")
            coverage[item["indicator_id"]].update(item["reference_periods"])
            if item["exclusion"]:
                exclusions.append(item)
            _atomic_bytes(stage / "sources" / (item["sha256"] + ".html"), content)
            consumed_bytes += len(content)
        missing = {
            key: sorted(periods - covered) for key, covered in coverage.items() if periods - covered
        }
        if missing:
            raise ValueError(
                "Faltan publicaciones del archivo declarado: " + json.dumps(missing, sort_keys=True)
            )
        rows = calculate_macro(records, catalog, clock)
        atomic_parquet(stage / "macro.parquet", pa.Table.from_pylist(rows))
        atomic_parquet(
            stage / "observations.parquet",
            pa.Table.from_pylist(records, schema=_OBSERVATION_SCHEMA),
        )
        atomic_json(stage / "source-manifest.json", manifest)
        if sha256(manifest_path) != manifest_hash or sha256(catalog_path) != catalog_hash:
            raise ValueError("Cambió una entrada de configuración durante la preparación")
        report = dict(
            schema_version=1,
            market=market,
            start=start,
            end=end,
            documents=len(manifest["documents"]),
            rows=len(rows),
            decisions=len(clock.decisions),
            vintage_rows=len(records),
            computed_values=sum(row["value"] is not None for row in rows),
            source_manifest_sha256=manifest_hash,
            catalog_sha256=catalog_hash,
            sha256=sha256(stage / "macro.parquet"),
            observations_sha256=sha256(stage / "observations.parquet"),
            archive_complete=True,
            coverage={key: sorted(values) for key, values in coverage.items()},
            exclusions=exclusions,
            publication_timestamp_verified=False,
            completed_at_utc=datetime.now(UTC).isoformat(),
            elapsed_seconds=time.perf_counter() - began,
            source_bytes_read=consumed_bytes,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        )
        atomic_json(stage / "report.json", report)
        _publish_directory(stage, output)
        descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "catalog", "source-cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--market", choices=("US", "CN"), required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--download", action="store_true", help="Descargar fuentes ausentes verificando sus huellas"
    )
    args = parser.parse_args(argv)
    result = prepare_releases(
        args.manifest,
        args.catalog,
        args.source_cache,
        args.output,
        market=args.market,
        start=args.start,
        end=args.end,
        download=args.download,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
