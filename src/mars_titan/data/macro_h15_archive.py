"""Extraer y conservar ediciones H.15 revisadas, sin admitirlas como panel macro."""

import argparse
import hashlib
import json
import os
import re
import resource
import tempfile
import time
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import bs4
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .cohort_files import read_manifest, safe_destination
from .macro import _available
from .macro_acquisition import _atomic_bytes
from .macro_coverage import _publish_directory
from .macro_h15_documents import (
    _MAX_DOCUMENT_BYTES,
    _day,
    _decimal,
    _issue_url,
    extract_h15_release,
)
from .preparation import atomic_parquet
from .storage import atomic_json, sha256
from .temporal import MarketClock

POLICY = "FED_H15_REVIEWED_WEEKLY_ARCHIVE_V1"
AVAILABILITY_RULE = "SOURCE_DAY_END_TARGET_NEXT_SESSION_V1"
REVIEW_POLICY = "PDF_HTML_REVIEWED_CELLS_V1"
_MAX_JSON_BYTES = 2 * 1024**2
_MAX_TOTAL_BYTES = 512 * 1024**2
_MAX_DOCUMENTS = 2048
_MAX_PARQUET_BYTES = 64 * 1024**2
_SCHEMA = pa.schema(
    [
        pa.field(name, pa.string())
        for name in (
            "market",
            "indicator_id",
            "series_id",
            "period_start",
            "publication_date",
            "value_exact",
            "missing_reason",
            "unit",
            "source_timezone",
            "source_label",
            "html_url",
            "pdf_url",
            "html_sha256",
            "pdf_sha256",
            "review_sha256",
            "review_status",
        )
    ]
    + [pa.field("value", pa.float64()), pa.field("available_at", pa.timestamp("us", tz="UTC"))]
    + [pa.field(name, pa.int32()) for name in ("source_row", "source_column", "source_page")]
)


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


class _Sources:
    """Vincular cada lectura acotada a su huella para comprobarla antes de confirmar."""

    def __init__(self, manifest):
        safe_destination(manifest)
        self.root = manifest.parent.resolve()
        self.meta, signature = read_manifest(manifest, _MAX_JSON_BYTES)
        self.hashes = {manifest: signature}
        self.total = manifest.stat().st_size

    def read(self, ref, maximum):
        if not isinstance(ref, dict) or not {"path", "sha256"} <= set(ref):
            raise ValueError("La referencia de fuente está incompleta")
        name, expected = ref["path"], ref["sha256"]
        if not isinstance(name, str) or not name or len(name) > 4096:
            raise ValueError("La ruta de fuente no es válida")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("La fuente sale del directorio del manifiesto")
        path = self.root / relative
        safe_destination(path)
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError("La fuente requiere SHA-256 explícito")
        if not path.is_file() or not 0 < path.stat().st_size <= maximum:
            raise ValueError("La fuente supera su presupuesto o no es un archivo")
        if path not in self.hashes:
            self.total += path.stat().st_size
        if self.total > _MAX_TOTAL_BYTES:
            raise ValueError("Las fuentes superan el presupuesto total")
        with path.open("rb") as stream:
            content = stream.read(maximum + 1)
        actual = hashlib.sha256(content).hexdigest()
        if len(content) > maximum or actual != expected or self.hashes.get(path, actual) != actual:
            raise ValueError("La huella de la fuente no coincide o ha cambiado")
        self.hashes[path] = actual
        return path, content

    def json(self, ref):
        path, content = self.read(ref, _MAX_JSON_BYTES)
        value, signature = read_manifest(path, _MAX_JSON_BYTES)
        if signature != hashlib.sha256(content).hexdigest():
            raise ValueError("El JSON cambió entre lecturas")
        return value

    def confirm(self):
        for path in self.hashes:
            safe_destination(path)
        if any(sha256(path) != expected for path, expected in self.hashes.items()):
            raise ValueError("Cambió una fuente o un recibo durante la preparación")


def _manifest(meta, cutoff):
    fields = {"schema_version", "policy", "availability_rule", "review_policy", "documents"}
    if (
        not isinstance(meta, dict)
        or set(meta) != fields
        or type(meta["schema_version"]) is not int
        or meta["schema_version"] != 1
    ):
        raise ValueError("El manifiesto H.15 no cumple el esquema")
    if (meta["policy"], meta["availability_rule"], meta["review_policy"]) != (
        POLICY,
        AVAILABILITY_RULE,
        REVIEW_POLICY,
    ):
        raise ValueError("El manifiesto no declara las políticas documentales admitidas")
    documents = meta["documents"]
    if not isinstance(documents, list) or not 1 <= len(documents) <= _MAX_DOCUMENTS:
        raise ValueError("El archivo supera el presupuesto de ediciones")
    seen, pdfs = set(), set()
    for doc in documents:
        if not isinstance(doc, dict) or set(doc) != {"publication_date", "html", "pdf", "review"}:
            raise ValueError("La descripción de edición está incompleta")
        day = _day(doc["publication_date"])
        if day > cutoff:
            raise ValueError("La publicación supera el corte autorizado")
        if day in seen:
            raise ValueError("El manifiesto repite una edición y no puede sobrescribir versiones")
        signature = doc["pdf"].get("sha256") if isinstance(doc["pdf"], dict) else None
        if not isinstance(signature, str) or signature in pdfs:
            raise ValueError("Cada fecha de publicación requiere su propio original PDF")
        seen.add(day)
        pdfs.add(signature)
    return documents


def _document(sources, item):
    candidate = None
    for kind in ("html", "pdf"):
        ref = item[kind]
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256", "url", "receipt"}:
            raise ValueError("La representación requiere URL, huella y recibo")
        if (
            _issue_url(ref["url"], "htm" if kind == "html" else "pdf").isoformat()
            != item["publication_date"]
        ):
            raise ValueError("Las representaciones no corresponden a la misma edición")
        _, content = sources.read(ref, _MAX_DOCUMENT_BYTES)
        receipt = sources.json(ref["receipt"])
        hash_key, size_key = (
            ("body_sha256", "body_bytes") if kind == "html" else ("sha256", "bytes")
        )
        if (
            not isinstance(receipt, dict)
            or type(receipt.get("status")) is not int
            or receipt["status"] != 200
            or receipt.get("complete") is not True
            or receipt.get("url") != ref["url"]
            or receipt.get("final_url") != ref["url"]
            or receipt.get(hash_key) != ref["sha256"]
            or type(receipt.get(size_key)) is not int
            or receipt[size_key] != len(content)
        ):
            raise ValueError("El recibo no confirma la representación completa")
        if kind == "html":
            candidate = extract_h15_release(content, ref["url"])
        elif (
            receipt.get("valid_pdf") is not True
            or not content.startswith(b"%PDF-")
            or b"%%EOF" not in content[-1024:]
        ):
            raise ValueError("El original PDF no está confirmado")
    if candidate["pdf_url"] != item["pdf"]["url"]:
        raise ValueError("El PDF declarado no es el enlazado por el HTML")
    review = sources.json(item["review"])
    fields = {
        "schema_version",
        "policy",
        "status",
        "publication_date",
        "html_sha256",
        "pdf_sha256",
        "publication_date_page",
        "reviewed_pages",
        "unit",
        "correction_notice",
        "observations",
    }
    if not isinstance(review, dict) or set(review) != fields:
        raise ValueError("La revisión no cumple el contrato de celdas")
    fixed = {
        "schema_version": 1,
        "policy": REVIEW_POLICY,
        "status": "document_reviewed_not_admitted",
        "publication_date": item["publication_date"],
        "html_sha256": item["html"]["sha256"],
        "pdf_sha256": item["pdf"]["sha256"],
        "unit": "percent_per_annum",
        "correction_notice": False,
    }
    if not _same({k: review[k] for k in fixed}, fixed):
        raise ValueError("La revisión no acredita fecha, unidad, estado y fuentes")
    pages = review["reviewed_pages"]
    if (
        not isinstance(pages, list)
        or not pages
        or len(pages) > 16
        or len(set(pages)) != len(pages)
        or any(type(p) is not int or not 1 <= p <= 16 for p in pages)
    ):
        raise ValueError("La revisión no identifica páginas válidas")
    if (
        type(review["publication_date_page"]) is not int
        or review["publication_date_page"] not in pages
    ):
        raise ValueError("No consta la página revisada de la fecha impresa")
    cells = review["observations"]
    if not isinstance(cells, list) or len(cells) != 30:
        raise ValueError("La revisión requiere treinta celdas diarias")
    keyed = {}
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != {
            "indicator_id",
            "period_start",
            "value_exact",
            "source_page",
            "source_column",
            "source_label",
        }:
            raise ValueError("La celda revisada no cumple su esquema")
        if (
            type(cell["source_page"]) is not int
            or cell["source_page"] not in pages
            or type(cell["source_column"]) is not int
        ):
            raise ValueError("La celda no pertenece a una página y columna revisadas")
        key = (cell["indicator_id"], cell["period_start"])
        if key in keyed:
            raise ValueError("La revisión repite una observación")
        keyed[key] = cell
    for row in candidate["observations"]:
        key = row["indicator_id"], row["period_start"]
        cell = keyed.get(key)
        if (
            cell is None
            or cell["source_column"] != row["source_column"]
            or cell["source_label"] != row["source_label"]
        ):
            raise ValueError("Las celdas PDF y HTML no tienen la misma localización")
        exact = cell["value_exact"]
        if exact is not None and not isinstance(exact, str):
            raise ValueError("La cifra revisada debe conservar su decimal como texto")
        number = _decimal(exact) if exact is not None else None
        if (
            (number is None) != (row["value_exact"] is None)
            or number is not None
            and Decimal(number) != Decimal(row["value_exact"])
        ):
            raise ValueError("Las cifras PDF revisadas y HTML no coinciden")
        row["source_page"] = cell["source_page"]
    return candidate


def _code_identity():
    return {
        name: sha256(Path(__file__).with_name(name))
        for name in (
            "macro_h15_archive.py",
            "macro_h15_documents.py",
            "macro_release_documents.py",
            "macro.py",
            "temporal.py",
            "cohort_files.py",
            "macro_coverage.py",
            "macro_acquisition.py",
            "preparation.py",
            "storage.py",
        )
    }


def _confirm(sources, code):
    sources.confirm()
    if _code_identity() != code:
        raise ValueError("Cambió el código de la edición durante la preparación")


def _check_parquet(path, table):
    """Comparar por columnas sin expandir diccionarios fuera del presupuesto."""
    with pq.ParquetFile(path) as file:
        metadata = file.metadata
        encoded = sum(
            metadata.row_group(i).column(j).total_uncompressed_size
            for i in range(metadata.num_row_groups)
            for j in range(metadata.num_columns)
        )
        if (
            file.schema_arrow != _SCHEMA
            or metadata.num_rows != table.num_rows
            or encoded > _MAX_PARQUET_BYTES
        ):
            raise ValueError("El Parquet confirmado supera su esquema o presupuesto")
    strings = [field.name for field in _SCHEMA if pa.types.is_string(field.type)]
    expanded = 0
    with pq.ParquetFile(path, metadata=metadata, read_dictionary=strings) as file:
        for field in _SCHEMA:
            offset = 0
            for batch in file.iter_batches(batch_size=512, columns=[field.name], use_threads=False):
                column = batch.column(0)
                if pa.types.is_dictionary(column.type):
                    # Se suman longitudes por índice, sin repetir los textos del diccionario.
                    lengths = pc.take(pc.binary_length(column.dictionary), column.indices)
                    expanded += (
                        (pc.sum(pc.cast(lengths, pa.int64())).as_py() or 0)
                        + 4 * (len(column) + 1)
                        + (len(column) + 7) // 8
                    )
                else:
                    expanded += column.nbytes
                if expanded > _MAX_PARQUET_BYTES:
                    raise ValueError("La expansión del Parquet supera el presupuesto")
                expected = table[field.name].slice(offset, batch.num_rows).combine_chunks()
                if not column.cast(field.type).equals(expected):
                    raise ValueError("El Parquet no coincide con las observaciones revisadas")
                offset += batch.num_rows
            if offset != table.num_rows:
                raise ValueError("El Parquet no contiene todas las observaciones revisadas")


def _recover(output, configuration, table, stable, sources, code):
    saved, config_hash = read_manifest(output / "configuration.json", _MAX_JSON_BYTES)
    report, report_hash = read_manifest(output / "report.json", _MAX_JSON_BYTES)
    fields = set(stable) | {
        "configuration_sha256",
        "artifacts",
        "output_bytes",
        "source_bytes",
        "elapsed_seconds",
        "process_peak_rss_bytes",
        "reused",
    }
    if (
        not _same(saved, configuration)
        or not isinstance(report, dict)
        or set(report) != fields
        or any(not _same(report.get(k), v) for k, v in stable.items())
        or report.get("configuration_sha256") != config_hash
        or report.get("reused") is not False
        or not _same(report.get("source_bytes"), sources.total)
    ):
        raise ValueError("La edición existente no conserva su identidad y recuentos")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "observations.parquet",
        "source-manifest.json",
    }:
        raise ValueError("El recibo no acredita los artefactos completos")
    for name, expected in artifacts.items():
        path = output / name
        safe_destination(path)
        if (
            not isinstance(expected, str)
            or not path.is_file()
            or path.stat().st_size > _MAX_PARQUET_BYTES
            or sha256(path) != expected
        ):
            raise ValueError("Un artefacto confirmado ha cambiado")
        sources.hashes[path] = expected
    if artifacts["source-manifest.json"] != configuration["source_manifest_sha256"]:
        raise ValueError("La copia del manifiesto no corresponde a la fuente")
    if not _same(report["output_bytes"], (output / "observations.parquet").stat().st_size):
        raise ValueError("El recibo no conserva el tamaño del Parquet")
    _check_parquet(output / "observations.parquet", table)
    sources.hashes[output / "configuration.json"] = config_hash
    sources.hashes[output / "report.json"] = report_hash
    _confirm(sources, code)
    return dict(report, reused=True)


def prepare_h15_archive(manifest_path, output, *, markets=("US", "CN"), cutoff="2023-12-31"):
    """Publicar una edición documental recuperable. Requiere admisión macro posterior."""
    began = time.perf_counter()
    manifest_path, output = Path(manifest_path).absolute(), Path(output).absolute()
    safe_destination(output)
    if output == manifest_path or manifest_path.is_relative_to(output):
        raise ValueError("La salida contiene el manifiesto original")
    last = _day(cutoff)
    if (
        not isinstance(markets, (list, tuple))
        or not markets
        or len(set(markets)) != len(markets)
        or any(m not in {"US", "CN"} for m in markets)
    ):
        raise ValueError("Los mercados deben ser US y/o CN sin duplicados")
    code = _code_identity()
    sources = _Sources(manifest_path)
    documents = _manifest(sources.meta, last)
    first = min(_day(d["publication_date"]) for d in documents) - timedelta(days=2)
    clocks = {market: MarketClock(market, first.isoformat(), cutoff) for market in markets}
    configuration = dict(
        schema_version=1,
        policy=POLICY,
        availability_rule=AVAILABILITY_RULE,
        review_policy=REVIEW_POLICY,
        source_manifest=str(manifest_path),
        source_manifest_sha256=sources.hashes[manifest_path],
        markets=list(markets),
        cutoff=cutoff,
        code_sha256=code,
        pyarrow=pa.__version__,
        beautifulsoup=bs4.__version__,
        calendars={
            m: hashlib.sha256(json.dumps([v.isoformat() for v in c.decisions]).encode()).hexdigest()
            for m, c in clocks.items()
        },
    )
    rows = []
    for item in sorted(documents, key=lambda d: d["publication_date"]):
        candidate = _document(sources, item)
        for market, clock in clocks.items():
            available = _available(_day(item["publication_date"]), "America/New_York", clock)
            for row in candidate["observations"]:
                rows.append(
                    dict(
                        row,
                        market=market,
                        value=float(Decimal(row["value_exact"]))
                        if row["value_exact"] is not None
                        else None,
                        publication_date=item["publication_date"],
                        available_at=available,
                        unit=candidate["unit"],
                        source_timezone=candidate["source_timezone"],
                        html_url=item["html"]["url"],
                        pdf_url=item["pdf"]["url"],
                        html_sha256=item["html"]["sha256"],
                        pdf_sha256=item["pdf"]["sha256"],
                        review_sha256=item["review"]["sha256"],
                        review_status="document_reviewed_not_admitted",
                    )
                )
    table = pa.Table.from_pylist(rows, schema=_SCHEMA)
    if table.nbytes > _MAX_PARQUET_BYTES:
        raise ValueError("Las observaciones superan el presupuesto de salida")
    stable = dict(
        schema_version=1,
        kind="h15_reviewed_document_edition",
        status="completed",
        documents=len(documents),
        observations=30 * len(documents),
        rows=len(rows),
        missing_source_rows=sum(r["value_exact"] is None for r in rows),
        unavailable_before_cutoff_rows=sum(r["available_at"] is None for r in rows),
        source_manifest_sha256=sources.hashes[manifest_path],
        admission_required=True,
        training_ready=False,
        final_test_opened=False,
    )
    if output.exists():
        return _recover(output, configuration, table, stable, sources, code)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        stage = Path(temporary) / "edition"
        stage.mkdir()
        atomic_parquet(stage / "observations.parquet", table)
        _, manifest_hash = read_manifest(manifest_path, _MAX_JSON_BYTES)
        if manifest_hash != sources.hashes[manifest_path]:
            raise ValueError("El manifiesto cambió antes de publicar")
        _atomic_bytes(stage / "source-manifest.json", manifest_path.read_bytes())
        atomic_json(stage / "configuration.json", configuration)
        report = dict(
            stable,
            configuration_sha256=sha256(stage / "configuration.json"),
            artifacts={
                name: sha256(stage / name)
                for name in ("observations.parquet", "source-manifest.json")
            },
            output_bytes=(stage / "observations.parquet").stat().st_size,
            source_bytes=sources.total,
            elapsed_seconds=time.perf_counter() - began,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            reused=False,
        )
        if report["artifacts"]["source-manifest.json"] != sources.hashes[manifest_path]:
            raise ValueError("La copia del manifiesto no conserva sus bytes")
        atomic_json(stage / "report.json", report)
        _confirm(sources, code)
        _publish_directory(stage, output)
        descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    _confirm(sources, code)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--market", choices=("US", "CN"), action="append")
    parser.add_argument("--cutoff", default="2023-12-31")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            prepare_h15_archive(
                args.manifest, args.output, markets=args.market or ("US", "CN"), cutoff=args.cutoff
            ),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
