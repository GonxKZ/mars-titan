"""Índice lateral de documentos y menciones, anterior a la media de noticias."""

import hashlib
import json
import re
import sqlite3
import tempfile
from collections import deque
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .batches import atomic_parquet_batches
from .cohort_contexts import NewsWindows
from .cohort_files import read_manifest, safe_destination
from .cohort_news import COHORT_POLICIES
from .cohort_news import SCHEMA as NEWS_SCHEMA
from .embeddings import EmbeddingCache
from .macro_coverage import _publish_directory
from .storage import atomic_json, outside_source, sha256
from .temporal import aware

SCHEMA = pa.schema(
    [field for field in NEWS_SCHEMA if field.name != "text"]
    + [
        pa.field(name, pa.string())
        for name in ("asset_id", "document_id", "mention_id", "embedding_key", "embedding_sha256")
    ]
)
FIRST = datetime(1900, 1, 1, tzinfo=UTC)
LAST = datetime(2023, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _embedding(row, encoder):
    return dict(
        encoder=encoder, kind="news", content=row["content_hash"], policy=row["availability_rule"]
    )


def _vector(cache, row, encoder, width):
    identity = _embedding(row, encoder)
    vector = cache.get(identity, max_bytes=width * 4)
    if vector is None or vector.shape != (width,):
        raise ValueError("Falta una representación compatible en la caché")
    checksum = hashlib.sha256(vector.astype("<f4", copy=False).tobytes()).hexdigest()
    if "embedding_sha256" in row and (
        row["embedding_sha256"] != checksum
        or row["embedding_key"] != EmbeddingCache.identity(identity)[0]
    ):
        raise ValueError("La representación de la caché ha cambiado")
    return vector, checksum


def _check_parquet(path, max_bytes):
    if path.is_symlink() or not path.is_file() or path.stat().st_size < 12:
        raise ValueError("El Parquet no es un archivo regular")
    with path.open("rb") as stream:
        stream.seek(-8, 2)
        footer = stream.read(8)
    if footer[4:] != b"PAR1" or int.from_bytes(footer[:4], "little") > 8 * 1024**2:
        raise ValueError("La cabecera Parquet supera su presupuesto o está corrupta")
    with pq.ParquetFile(path) as file:
        if file.num_row_groups > 100_000 or file.metadata.num_rows > 10_000_000:
            raise ValueError("El índice Parquet supera su presupuesto")
        if any(
            file.metadata.row_group(i).total_byte_size > max_bytes
            for i in range(file.num_row_groups)
        ):
            raise ValueError("Un grupo Parquet supera el presupuesto")


class _SourceWindows(NewsWindows):
    columns = tuple(NEWS_SCHEMA.names)


class _DocumentWindows(NewsWindows):
    columns = tuple(SCHEMA.names)
    dictionary_columns = ()


def _sources(sources, output):
    if not isinstance(sources, dict) or not 1 <= len(sources) <= 4096:
        raise ValueError("Las fuentes necesitan entre uno y 4096 activos")
    result, cohort = {}, None
    for asset, path in sorted(sources.items()):
        if not isinstance(asset, str) or not re.fullmatch(r"(US|CN)/[A-Z0-9.^_=\-]{1,64}", asset):
            raise ValueError("La mención necesita una identidad de mercado y activo")
        path = Path(path)
        outside_source(path.parent, output)
        receipt, digest = read_manifest(path, 1024**2)
        config, config_hash = read_manifest(path.parent / "configuration.json", 8 * 1024**2)
        news = path.parent / "news.parquet"
        current = receipt.get("cohort_id")
        if (
            receipt.get("schema_version") != 2
            or current not in COHORT_POLICIES
            or receipt.get("news_content_policy") != COHORT_POLICIES[current]
            or asset != f"{receipt.get('market')}/{receipt.get('symbol')}"
            or config.get("cutoff", "9999") > "2023-12-31"
            or receipt.get("fingerprint") != _digest(config)
            or config.get("cohort_id") != current
            or (cohort is not None and cohort != current)
            or type(receipt.get("counts", {}).get("accepted")) is not int
            or not 0 <= receipt["counts"]["accepted"] <= 10_000_000
            or news.is_symlink()
            or sha256(news) != receipt.get("artifacts", {}).get("news.parquet")
        ):
            raise ValueError("La fuente no acredita cohorte, procedencia y huella compatibles")
        cohort = current
        result[asset] = dict(
            manifest=str(path.resolve()),
            manifest_sha256=digest,
            configuration_sha256=config_hash,
            news_sha256=sha256(news),
            records=receipt["counts"]["accepted"],
        )
    return result, cohort


def _document(row, asset, cohort, cache, encoder, width):
    if (
        row["cohort_id"] != cohort
        or row["symbol"] != asset.split("/")[1]
        or not isinstance(row["text"], str)
        or len(row["text"].encode()) > 1024**2
        or hashlib.sha256(row["text"].encode()).hexdigest() != row["content_hash"]
        or any(not _hash(row[k]) for k in ("event_id", "content_hash", "source_record_hash"))
        or type(row["line"]) is not int
        or row["line"] < 1
        or not row["source_file"]
        or not row["availability_rule"]
        or row["available_at"] is None
        or not FIRST <= aware(row["available_at"]) <= LAST
        or (
            row["published_at"] is not None
            and aware(row["published_at"]) > aware(row["available_at"])
        )
    ):
        raise ValueError("El documento contiene una identidad, disponibilidad o huella inválida")
    _, checksum = _vector(cache, row, encoder, width)
    result = {key: value for key, value in row.items() if key != "text"}
    publication = row["published_at"].isoformat() if row["published_at"] else row["source_date"]
    result.update(
        asset_id=asset,
        document_id=_digest([row["content_hash"], row["url"], publication]),
        mention_id=_digest([asset, row["event_id"]]),
        embedding_key=EmbeddingCache.identity(_embedding(row, encoder))[0],
        embedding_sha256=checksum,
    )
    return result


def _tables(db, batch_rows, max_bytes):
    rows, size = [], 0
    for (payload,) in db.execute(
        "SELECT payload FROM mentions ORDER BY available,content,event,asset"
    ):
        width = len(payload.encode()) + 1024
        if width > max_bytes:
            raise ValueError("Un documento supera el presupuesto del grupo")
        if rows and (len(rows) == batch_rows or size + width > max_bytes):
            yield pa.Table.from_pylist(rows, schema=SCHEMA)
            rows, size = [], 0
        row = json.loads(payload)
        for key in ("event_at", "published_at", "available_at"):
            if row[key] is not None:
                row[key] = datetime.fromisoformat(row[key])
        rows.append(row)
        size += width
    yield pa.Table.from_pylist(rows, schema=SCHEMA)


def prepare_document_index(
    sources,
    output,
    *,
    cache_path,
    encoder_sha256,
    embedding_width=384,
    batch_rows=256,
    max_group_bytes=16 * 1024**2,
    spool_bytes=1024**3,
):
    """Publicar una edición completa. Un corte previo exige repetir solo su preparación."""
    output, cache_path = Path(output), Path(cache_path)
    safe_destination(output)
    if (
        not _hash(encoder_sha256)
        or type(embedding_width) is not int
        or not 1 <= embedding_width <= 4096
        or type(batch_rows) is not int
        or not 1 <= batch_rows <= 4096
        or type(max_group_bytes) is not int
        or not 1024 <= max_group_bytes <= 64 * 1024**2
        or type(spool_bytes) is not int
        or not 1024**2 <= spool_bytes <= 16 * 1024**3
    ):
        raise ValueError("El codificador o los presupuestos del índice no son válidos")
    inputs, cohort = _sources(sources, output)
    configuration = dict(
        sources=inputs,
        cohort_id=cohort,
        encoder_sha256=encoder_sha256,
        embedding_width=embedding_width,
        batch_rows=batch_rows,
        max_group_bytes=max_group_bytes,
        spool_bytes=spool_bytes,
    )
    identity = _digest(configuration)
    if output.exists():
        with DocumentIndex(output / "manifest.json", cache_path=cache_path) as reader:
            if reader.manifest["configuration_sha256"] != identity:
                raise ValueError("El índice pertenece a otra configuración")
            return reader.manifest | {"reused": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".documents-", dir=output.parent) as temporary:
        stage = Path(temporary) / "edition"
        stage.mkdir()
        cache = EmbeddingCache(cache_path, read_only=True)
        try:
            with closing(sqlite3.connect(Path(temporary) / "sort.sqlite")) as db:
                db.execute("PRAGMA trusted_schema=OFF")
                db.execute("PRAGMA cache_size=-8192")
                db.execute("PRAGMA temp_store=FILE")
                page = db.execute("PRAGMA page_size").fetchone()[0]
                db.execute(f"PRAGMA max_page_count={spool_bytes // page}")
                db.executescript("""
                    CREATE TABLE mentions(id TEXT PRIMARY KEY, available TEXT, content TEXT,
                        event TEXT, asset TEXT, document TEXT, payload TEXT);
                    CREATE INDEX ordering ON mentions(available,content,event,asset);
                    CREATE INDEX documents ON mentions(document);
                """)
                duplicates = 0
                for asset, source in inputs.items():
                    path = Path(source["manifest"]).parent / "news.parquet"
                    _check_parquet(path, max_group_bytes)
                    observed = 0
                    with _SourceWindows(path, max_group_bytes=max_group_bytes) as windows:
                        for row in windows.between(FIRST, LAST):
                            row = _document(
                                row, asset, cohort, cache, encoder_sha256, embedding_width
                            )
                            payload = json.dumps(
                                row, sort_keys=True, default=lambda d: d.isoformat()
                            )
                            previous = db.execute(
                                "SELECT payload FROM mentions WHERE id=?", (row["mention_id"],)
                            ).fetchone()
                            if previous is not None:
                                if previous[0] != payload:
                                    raise ValueError("Hay una mención duplicada ambigua")
                                duplicates += 1
                            else:
                                db.execute(
                                    "INSERT INTO mentions VALUES (?,?,?,?,?,?,?)",
                                    (
                                        row["mention_id"],
                                        row["available_at"].isoformat(),
                                        row["content_hash"],
                                        row["event_id"],
                                        asset,
                                        row["document_id"],
                                        payload,
                                    ),
                                )
                            observed += 1
                            if observed % 256 == 0:
                                db.commit()
                    if observed != source["records"]:
                        raise ValueError("El índice no conserva las menciones admitidas")
                db.commit()
                count = atomic_parquet_batches(
                    stage / "documents.parquet", _tables(db, batch_rows, max_group_bytes)
                )
                documents = db.execute("SELECT COUNT(DISTINCT document) FROM mentions").fetchone()[
                    0
                ]
        finally:
            cache.close()
        if _sources(sources, output)[0] != inputs:
            raise ValueError("Una fuente cambió durante la preparación del índice")
        _check_parquet(stage / "documents.parquet", max_group_bytes)
        report = dict(
            schema_version=1,
            kind="document_index",
            configuration=configuration,
            configuration_sha256=identity,
            mentions=count,
            documents=documents,
            exact_duplicates=duplicates,
            reused=False,
            documents_sha256=sha256(stage / "documents.parquet"),
        )
        atomic_json(stage / "manifest.json", report)
        _publish_directory(stage, output)
    return report


class DocumentIndex:
    """Leer intervalos inclusivos con dos grupos en caché y cursores de prefijo verificables."""

    def __init__(self, manifest, *, cache_path, encoder_sha256=None, max_group_bytes=16 * 1024**2):
        self.path = Path(manifest)
        self.manifest, self.identity = read_manifest(self.path, 8 * 1024**2)
        meta = self.manifest
        config = meta.get("configuration", {})
        if (
            meta.get("schema_version") != 1
            or meta.get("kind") != "document_index"
            or _digest(config) != meta.get("configuration_sha256")
            or not _hash(config.get("encoder_sha256"))
            or type(config.get("embedding_width")) is not int
            or not 1 <= config["embedding_width"] <= 4096
            or config.get("cohort_id") not in COHORT_POLICIES
            or not isinstance(config.get("sources"), dict)
            or not 1 <= len(config["sources"]) <= 4096
            or (encoder_sha256 is not None and encoder_sha256 != config["encoder_sha256"])
            or type(max_group_bytes) is not int
            or not 1 <= max_group_bytes <= 64 * 1024**2
        ):
            raise ValueError("El índice, el codificador o su presupuesto no son compatibles")
        self.encoder, self.width = config["encoder_sha256"], config["embedding_width"]
        self.data = self.path.parent / "documents.parquet"
        self._signatures = {}
        self._verify()
        _check_parquet(self.data, max_group_bytes)
        self.windows = _DocumentWindows(self.data, max_group_bytes=max_group_bytes)
        try:
            if (
                self.windows.file.schema_arrow != SCHEMA
                or self.windows.file.metadata.num_rows != meta.get("mentions")
            ):
                raise ValueError("El índice no conserva el esquema o los recuentos")
            self.cache = EmbeddingCache(Path(cache_path), read_only=True)
        except BaseException:
            self.windows.__exit__()
            raise

    def _verify(self):
        for path, expected in (
            (self.path, self.identity),
            (self.data, self.manifest["documents_sha256"]),
        ):
            before = path.stat()
            signature = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            if self._signatures.get(path) != signature:
                digest = sha256(path)
                after = path.stat()
                if (
                    path.is_symlink()
                    or digest != expected
                    or signature
                    != (
                        after.st_dev,
                        after.st_ino,
                        after.st_size,
                        after.st_mtime_ns,
                        after.st_ctime_ns,
                    )
                ):
                    raise ValueError("Ha cambiado la huella del índice documental")
                self._signatures[path] = signature

    @property
    def cached_row_groups(self):
        return self.windows.cached_row_groups

    def batches(
        self,
        asset_id,
        start,
        end,
        *,
        batch_size=64,
        max_mentions=100_000,
        max_scan_rows=1_000_000,
        max_batch_bytes=8 * 1024**2,
        cursor=None,
    ):
        start, end = aware(start), aware(end)
        if start > end or end > LAST:
            raise ValueError("El intervalo está invertido o abre la reserva final")
        if (
            asset_id not in self.manifest["configuration"]["sources"]
            or type(batch_size) is not int
            or not 1 <= batch_size <= 1024
            or type(max_mentions) is not int
            or not 1 <= max_mentions <= 100_000
            or type(max_scan_rows) is not int
            or not 1 <= max_scan_rows <= 1_000_000
            or type(max_batch_bytes) is not int
            or not 1 <= max_batch_bytes <= 64 * 1024**2
        ):
            raise ValueError("El activo o el presupuesto documental no es válido")
        self._verify()
        identity = dict(
            index_sha256=self.identity,
            asset_id=asset_id,
            start=start.isoformat(),
            end=end.isoformat(),
            batch_size=batch_size,
            max_mentions=max_mentions,
            max_scan_rows=max_scan_rows,
            max_batch_bytes=max_batch_bytes,
        )
        if cursor is not None and (
            not isinstance(cursor, dict)
            or set(cursor) != set(identity) | {"consumed", "prefix_sha256"}
            or any(cursor[k] != value for k, value in identity.items())
            or type(cursor["consumed"]) is not int
            or not 0 <= cursor["consumed"] <= max_mentions
            or not _hash(cursor["prefix_sha256"])
        ):
            raise ValueError("El cursor pertenece a otra consulta")
        skipped = 0 if cursor is None else cursor["consumed"]
        digest, rows, consumed, size = hashlib.sha256(), [], 0, 0
        if cursor is not None and not skipped and cursor["prefix_sha256"] != digest.hexdigest():
            raise ValueError("El cursor no acredita el prefijo")
        for scanned, row in enumerate(self.windows.between(start, end), 1):
            if scanned > max_scan_rows:
                raise ValueError("La lectura de la ventana excede el presupuesto")
            if row["asset_id"] != asset_id:
                continue
            payload = json.dumps(row, sort_keys=True, default=lambda d: d.isoformat()).encode()
            row_bytes = len(payload) + 1024
            if row_bytes > max_batch_bytes:
                raise ValueError("Un documento excede el presupuesto del lote")
            if rows and size + row_bytes > max_batch_bytes:
                yield dict(
                    documents=rows,
                    confirmed_cursor=identity
                    | dict(consumed=consumed, prefix_sha256=digest.hexdigest()),
                )
                rows, size = [], 0
            consumed += 1
            if consumed > max_mentions:
                raise ValueError("La consulta excede el presupuesto de menciones")
            digest.update(payload)
            if consumed <= skipped:
                if consumed == skipped and digest.hexdigest() != cursor["prefix_sha256"]:
                    raise ValueError("El cursor no acredita el prefijo")
                continue
            rows.append(row)
            size += row_bytes
            if len(rows) == batch_size:
                yield dict(
                    documents=rows,
                    confirmed_cursor=identity
                    | dict(consumed=consumed, prefix_sha256=digest.hexdigest()),
                )
                rows, size = [], 0
        if consumed < skipped:
            raise ValueError("El cursor excede el intervalo")
        self._verify()
        if rows:
            yield dict(
                documents=rows,
                confirmed_cursor=identity
                | dict(consumed=consumed, prefix_sha256=digest.hexdigest()),
            )

    def compare(self, asset_id, start, end, *, selected=8, max_mentions=100_000):
        """Comparar media y selección reciente sobre idéntica ventana, sin volver a codificar."""
        if type(selected) is not int or not 1 <= selected <= 1024:
            raise ValueError("La selección excede el presupuesto")
        total, count = np.zeros(self.width, dtype=np.float64), 0
        recent, documents, representations = deque(maxlen=selected), set(), set()
        digest = hashlib.sha256()
        for batch in self.batches(asset_id, start, end, max_mentions=max_mentions):
            for row in batch["documents"]:
                vector, _ = _vector(self.cache, row, self.encoder, self.width)
                total += vector
                count += 1
                recent.append((row["mention_id"], vector))
                documents.add(row["document_id"])
                representations.add(row["embedding_key"])
                digest.update((row["mention_id"] + row["embedding_sha256"]).encode())
        chosen = np.zeros(self.width, dtype=np.float64)
        for _, vector in recent:
            chosen += vector
        return dict(
            mean=(total / count).astype(np.float32).tolist() if count else None,
            selected_mean=(chosen / len(recent)).astype(np.float32).tolist() if recent else None,
            selected_mentions=[key for key, _ in recent],
            examined_mentions=count,
            unique_documents=len(documents),
            unique_representations=len(representations),
            cache_reads=count,
            encoded_documents=0,
            window_sha256=digest.hexdigest(),
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.windows.__exit__()
        self.cache.close()
