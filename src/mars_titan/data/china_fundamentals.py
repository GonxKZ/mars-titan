"""Materializar revisiones contables chinas sin fechar de nuevo los datos originales."""

import argparse
import hashlib
import math
import tempfile
import time
from collections import Counter, defaultdict
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import ijson
import pyarrow as pa
import pyarrow.parquet as pq

from .china_sources import reconcile_chinese_fact
from .cohort_files import read_manifest, safe_destination
from .cohort_preparation import FACT_SCHEMA
from .corpus_catalog import _source_path
from .macro_coverage import _publish_directory
from .preparation import atomic_parquet
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock

_MAX_RECORDS = 100_000
_SCHEMA = pa.schema(
    list(FACT_SCHEMA)
    + [
        pa.field(name, pa.string())
        for name in (
            "value_exact",
            "accounting_standard",
            "statement_scope",
            "document_sha256",
            "publication_evidence_sha256",
            "source_url",
        )
    ]
    + [
        pa.field("source_page", pa.int32()),
        pa.field("source_records", pa.list_(pa.int64())),
        pa.field("published_at", pa.timestamp("us", tz="UTC")),
    ]
)


class _Record(dict):
    duplicate_keys = False

    def __setitem__(self, key, value):
        if key in self:
            # El error se lanza al consumir el registro, fuera de la llamada C.
            self.duplicate_keys = True
        super().__setitem__(key, value)


def _file_hash(path, limit=64 * 1024**2):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("La fuente no es un archivo regular dentro del presupuesto")
    return sha256(path)


def _verify_identity(sources, code):
    if any(_file_hash(path) != signature for path, signature in sources.items()):
        raise ValueError("Una fuente cambió durante la materialización")
    if any(sha256(Path(__file__).with_name(name)) != signature for name, signature in code.items()):
        raise ValueError("El código cambió durante la materialización")


def _check_announcement(evidence, notices):
    matching = [r for r in notices if r.get("announcementId") == evidence["report_id"]]
    if len(matching) != 1:
        raise ValueError("El anuncio falta o es ambiguo")
    notice = matching[0]
    stamp = notice.get("announcementTime")
    symbol = evidence["symbol"]
    exchange = "SZ" if symbol.endswith(".SZ") else "SH"
    if (
        type(stamp) is not int
        or not 0 <= stamp <= 4_102_444_800_000
        or notice.get("secCode") != symbol.split(".")[0]
        or not str(notice.get("pageColumn", "")).startswith(exchange)
        or notice.get("adjunctType") != "PDF"
        or "https://static.cninfo.com.cn/" + str(notice.get("adjunctUrl")) != evidence["source_url"]
    ):
        raise ValueError("El anuncio no corresponde al emisor y documento revisados")
    day = datetime.fromtimestamp(stamp / 1000, UTC).astimezone(ZoneInfo("Asia/Shanghai")).date()
    if day.isoformat() != evidence["publication_date"] or evidence.get("published_at") is not None:
        raise ValueError("El anuncio acredita una fecha, no la hora exacta de publicación")


def _requests(review, notices, document_hash, publication_hash):
    results = review.get("results")
    if (
        review.get("schema_version") != 1
        or not isinstance(results, list)
        or not 1 <= len(results) <= 10_000
    ):
        raise ValueError("La revisión no contiene una lista acotada de hechos")
    requests, seen = defaultdict(list), set()
    for item in results:
        evidence, matches = item["evidence"], item["original_matches"]
        if (
            evidence.get("symbol") != review.get("symbol")
            or evidence.get("document_sha256") != document_hash
            or evidence.get("publication_evidence_sha256") != publication_hash
            or not isinstance(matches, list)
            or not matches
        ):
            raise ValueError("La revisión no identifica las fuentes y coincidencias suministradas")
        _check_announcement(evidence, notices)
        for match in matches:
            index = match["array_record_1_based"]
            if type(index) is not int or not 1 <= index <= _MAX_RECORDS:
                raise ValueError("El localizador del registro no es válido")
            key = (index, evidence["field"])
            if key in seen or len(seen) >= _MAX_RECORDS:
                raise ValueError("Las coincidencias se repiten o superan el presupuesto")
            seen.add(key)
            requests[index].append(evidence)
    return requests


def _read_facts(original, requests, clock, cutoff, relative):
    unique, found, count = {}, set(), 0
    numeric_matches = Counter()
    with original.open("rb") as stream:
        for index, raw in enumerate(ijson.items(stream, "item", map_type=_Record), 1):
            if index > _MAX_RECORDS:
                raise ValueError("El origen supera el presupuesto de registros")
            if not isinstance(raw, _Record) or raw.duplicate_keys:
                raise ValueError("El original contiene un registro inválido o claves duplicadas")
            for evidence in requests.get(index, ()):
                result = reconcile_chinese_fact(raw, evidence)
                if result["status"] != "reconciled":
                    raise ValueError(f"Registro {index}: {result['status']}")
                numeric_matches[result.get("numeric_match", {}).get("policy", "exact_decimal")] += 1
                fact = result["fact"]
                available = clock.date_available(fact["publication_date"])
                if available.date() > cutoff:
                    raise ValueError("La disponibilidad contable cruza el corte autorizado")
                value = float(fact["value_cny"])
                if not math.isfinite(value):
                    raise ValueError("La conversión a float64 no es finita")
                row = dict(
                    concept=fact["concept"],
                    unit=fact["currency"],
                    period_start=fact["period_start"],
                    period_end=fact["period_end"],
                    filed=fact["publication_date"],
                    accession=fact["report_id"],
                    source_file=relative,
                    availability_rule="reviewed_cninfo_date_next_session_close",
                    value=value,
                    value_exact=fact["value_cny"],
                    available_at=available,
                    published_at=None,
                    source_records=[index],
                    **{
                        name: fact[name]
                        for name in (
                            "accounting_standard",
                            "statement_scope",
                            "document_sha256",
                            "publication_evidence_sha256",
                            "source_url",
                            "source_page",
                        )
                    },
                )
                key = tuple(
                    row[name]
                    for name in (
                        "concept",
                        "period_start",
                        "period_end",
                        "filed",
                        "accession",
                        "accounting_standard",
                        "statement_scope",
                    )
                )
                if key in unique:
                    previous = unique[key]
                    if Decimal(previous["value_exact"]) != Decimal(row["value_exact"]):
                        raise ValueError("La revisión contiene importes contables contradictorios")
                    previous["source_records"].append(index)
                else:
                    unique[key] = row
                count += 1
                found.add(index)
    if found != requests.keys():
        raise ValueError("El origen no contiene todos los registros revisados")
    rows = sorted(
        unique.values(),
        key=lambda r: (r["available_at"], r["period_end"], r["accession"], r["concept"]),
    )
    return rows, count, dict(numeric_matches)


def materialize_chinese_facts(
    source, review, document, publication, output, *, clock, cutoff="2023-12-31"
):
    """Comprobar originales, eliminar duplicados y publicar una edición contable atómica."""
    started = time.perf_counter()
    source, review, document, publication, output = map(
        Path, (source, review, document, publication, output)
    )
    safe_destination(output)
    if clock.market != "CN" or date.fromisoformat(cutoff) >= date(2024, 1, 1):
        raise ValueError("El calendario debe ser chino y el corte no puede abrir la reserva final")
    for protected in (source, review, document, publication):
        outside_source(protected, output)
        outside_source(output, protected)
    content, review_hash = read_manifest(review, maximum=4 * 1024**2)
    notice, notice_hash = read_manifest(publication, maximum=4 * 1024**2)
    notices = notice.get("announcements")
    if (
        not isinstance(notices, list)
        or not 1 <= len(notices) <= 10_000
        or not all(isinstance(n, dict) for n in notices)
    ):
        raise ValueError("Falta una lista acotada de anuncios")
    original = _source_path(source.resolve(), content["source_file"])
    sources = {
        review: review_hash,
        publication: notice_hash,
        document: _file_hash(document),
        original: _file_hash(original),
    }
    if sources[original] != content["source_file_sha256"]:
        raise ValueError("La huella del original no coincide con la revisión")
    requests = _requests(content, notices, sources[document], notice_hash)
    identity = dict(
        source_sha256=sources[original],
        review_sha256=review_hash,
        document_sha256=sources[document],
        publication_sha256=notice_hash,
        cutoff=cutoff,
        market=clock.market,
        symbol=content["symbol"],
        calendar_sha256=hashlib.sha256(
            "|".join(t.isoformat() for t in clock.decisions).encode()
        ).hexdigest(),
        code={
            name: sha256(Path(__file__).with_name(name))
            for name in (
                "china_fundamentals.py",
                "china_sources.py",
                "cohort_preparation.py",
                "cohort_files.py",
                "corpus_catalog.py",
                "macro_coverage.py",
                "preparation.py",
                "storage.py",
                "temporal.py",
            )
        },
        pyarrow=pa.__version__,
        ijson=ijson.__version__,
        numeric_storage="float64_with_exact_decimal_provenance",
    )
    rows, matched, numeric_matches = _read_facts(
        original, requests, clock, date.fromisoformat(cutoff), content["source_file"]
    )
    expected = dict(
        schema_version=1,
        kind="reconciled_chinese_fundamentals",
        identity=identity,
        facts=len(rows),
        matched_records=matched,
        unique_source_records=len(requests),
        duplicate_matches=matched - len(rows),
        numeric_match_counts=numeric_matches,
        final_test_opened=False,
        training_ready=False,
    )
    _verify_identity(sources, identity["code"])
    if output.exists():
        report_path = output / "report.json"
        report, report_hash = read_manifest(report_path, maximum=1024**2)
        artifact = output / "fundamentals.parquet"
        if any(report.get(k) != v for k, v in expected.items()) or _file_hash(
            artifact
        ) != report.get("sha256"):
            raise ValueError("La edición existente no coincide con los hechos revisados")
        with pq.ParquetFile(artifact) as file:
            if (
                file.schema_arrow != _SCHEMA
                or file.metadata.num_rows != len(rows)
                or file.num_row_groups > len(rows)
                or sum(
                    file.metadata.row_group(i).total_byte_size for i in range(file.num_row_groups)
                )
                > 64 * 1024**2
            ):
                raise ValueError(
                    "La edición existente tiene esquema, población o tamaño incompatibles"
                )
            if file.read(use_threads=False).to_pylist() != rows:
                raise ValueError("El contenido de la edición existente no coincide")
        _verify_identity(
            {**sources, report_path: report_hash, artifact: report["sha256"]}, identity["code"]
        )
        return {**report, "reused": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        stage = Path(temporary) / "edition"
        stage.mkdir()
        atomic_parquet(stage / "fundamentals.parquet", pa.Table.from_pylist(rows, schema=_SCHEMA))
        report = dict(
            expected,
            sha256=sha256(stage / "fundamentals.parquet"),
            elapsed_seconds=time.perf_counter() - started,
        )
        atomic_json(stage / "report.json", report)
        _verify_identity(sources, identity["code"])
        _publish_directory(stage, output)
    return {**report, "reused": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "review", "document", "publication", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cutoff", default="2023-12-31")
    args = parser.parse_args(argv)
    report = materialize_chinese_facts(
        args.source,
        args.review,
        args.document,
        args.publication,
        args.output,
        clock=MarketClock("CN", "1990-12-19", "2024-01-31"),
        cutoff=args.cutoff,
    )
    print(
        f"Hechos únicos: {report['facts']}. "
        f"Coincidencias originales: {report['matched_records']}. Admisión multimodal pendiente."
    )


if __name__ == "__main__":
    main()
